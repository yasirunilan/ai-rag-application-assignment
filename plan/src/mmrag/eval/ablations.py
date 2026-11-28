"""Failure Mode Reproduction & Before/After Engineering Harness (Person 5 · Story E7.3; brief E8.5).

Reproduces and fixes two Appendix A failure modes, each measured before and after, by type:

1. Failure Mode 1 — "Right text, wrong figure" (caption–image drift)
   Symptom : figure questions rank the wrong relay schematic (or the page's prose) above the gold figure.
   Cause   : the figure's search text is the bare vision caption; two relay schematics look alike
             and nothing ties a crop to its "Figure 9-14" label, section or model variant.
   Fix     : inject the nearby figure label, section heading and model variant into the caption
             (`enrich.include_nearby_context=True`).

2. Failure Mode 2 — "Table answers are mush & exact part codes missed"
   Symptom : table questions about `AN4-12A` / `FAULT-E409` return a flattened string or the wrong table.
   Cause   : three separate defects, each fixed and measured in its own step so the gain is attributable:
             (a) the table artefact is flattened to a string   -> `extract.flatten_tables=False`
             (b) its search text loses header binding          -> `enrich.table_linearization_mode="col_val_and_markdown"`
             (c) dense vectors smear exact identifiers         -> `retrieve.enable_bm25_fusion=True, alpha=0.60`

The untuned baseline (both failure modes active) is evaluated first and saved before any fix. The
runs also give the cumulative T2 path: Baseline -> After fix 1 -> After fix 2 (= Final).
Outputs: `reports/<experiment>.json` per run, `reports/e7_3_failure_mode_ablations.json` (summary)
and `reports/e7_3_failure_mode_ablations.md` (T2 / T3 / Layer B tables).
"""

from __future__ import annotations

import json
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Any, Optional

from mmrag.config import AppConfig
from mmrag.eval.metrics import UNIT_TYPES, compare_reports, failing_question_ids, save_eval_report
from mmrag.pipeline import MultimodalRAGPipeline
from mmrag.schemas import EvalReport

STRUCTURED = "col_val_and_markdown"
FLAT = "flat_raw"


@dataclass(frozen=True)
class Variant:
    """The five knobs the two failure modes touch; everything else comes from the base config."""

    include_nearby_context: bool
    flatten_tables: bool
    table_linearization_mode: str
    enable_bm25_fusion: bool
    alpha: float


UNTUNED = Variant(False, True, FLAT, False, 1.0)                 # both failure modes active
FM1_BEFORE = Variant(False, False, STRUCTURED, True, 0.60)       # only FM1 active
FIXED = Variant(True, False, STRUCTURED, True, 0.60)             # both fixed (= Final)
FM2_BEFORE = Variant(True, True, FLAT, False, 1.0)               # only FM2 active (= Baseline + fix 1)
FM2_STEP_STRUCTURED_ARTEFACT = Variant(True, False, FLAT, False, 1.0)
FM2_STEP_COL_VAL = Variant(True, False, STRUCTURED, False, 1.0)

FM1_INFO = {
    "symptom": "Caption-image drift causes wrong relay diagram (p.2) to outrank starter relay diagram (p.3)",
    "hypothesis": "Figure crops are indistinguishable by caption alone; adding their label, section and model variant will rank the gold figure first.",
    "root_cause": "Figure search_text is the bare vision caption: two relay schematics share most callouts and nothing ties a crop to its 'Figure 9-14' label or model variant.",
    "fix": "Inject nearby figure label, section heading, and model variant into vision caption prompt & search_text",
    "change": "enrich.include_nearby_context: false -> true",
}
FM2_INFO = {
    "symptom": "Flattened table string loses column structure and dense vectors smear exact part/fault codes (AN4-12A, FAULT-E409)",
    "hypothesis": "Keeping the table structured, binding each cell to its header, and adding exact-token BM25 will return the right table and row.",
    "root_cause": "Tables were flattened to one string (artefact and search text), so header/value binding was lost, and dense-only retrieval does not match exact identifiers.",
    "fix": "Keep structured TableArtefact + col=val row linearization + BM25 lexical fusion (alpha=0.60)",
    "change": "extract.flatten_tables: true -> false; enrich.table_linearization_mode: flat_raw -> col_val_and_markdown; retrieve.enable_bm25_fusion: false -> true (alpha 1.0 -> 0.60) — applied and measured one at a time",
}
EVALUATION_NOTE = (
    "Synthetic 2-manual corpus with 12 gold questions (4 per type), offline embedder and captioner. "
    "The fixes were designed on the same gold questions they are reported on (no held-out split), "
    "so absolute numbers are optimistic; the before/after deltas are the evidence."
)


def _clone_cfg(
    base: AppConfig,
    *,
    include_nearby_context: bool,
    flatten_tables: bool,
    table_linearization_mode: str,
    enable_bm25_fusion: bool,
    alpha: float,
) -> AppConfig:
    return base.model_copy(
        update={
            "extract": base.extract.model_copy(update={"flatten_tables": flatten_tables}),
            "enrich": base.enrich.model_copy(
                update={
                    "include_nearby_context": include_nearby_context,
                    "table_linearization_mode": table_linearization_mode,
                }
            ),
            "retrieve": base.retrieve.model_copy(
                update={
                    "enable_bm25_fusion": enable_bm25_fusion,
                    "alpha": alpha,
                }
            ),
        }
    )


def _run_config_eval(cfg: AppConfig, name: str, force_offline: bool = True) -> EvalReport:
    pipe = MultimodalRAGPipeline(cfg, force_offline=force_offline)
    pipe.ingest_and_index(persist_index=False)
    return pipe.evaluate(experiment_name=name, save_report=True)


class _Runner:
    """Evaluates each distinct variant once; a repeated variant is re-saved under its new name."""

    def __init__(self, base_cfg: AppConfig, force_offline: bool) -> None:
        self.base_cfg = base_cfg
        self.force_offline = force_offline
        self._cache: dict[tuple, EvalReport] = {}

    def run(self, variant: Variant, name: str) -> EvalReport:
        key = astuple(variant)
        if key in self._cache:
            report = self._cache[key].model_copy(update={"experiment_name": name})
            save_eval_report(report, self.base_cfg)
            return report
        cfg = _clone_cfg(self.base_cfg, **variant.__dict__)
        report = _run_config_eval(cfg, name, force_offline=self.force_offline)
        self._cache[key] = report
        return report


# ---------------------------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------------------------
def _t2_row(report: EvalReport, u_type: str) -> dict[str, Any]:
    tm = report.by_type[u_type]
    row: dict[str, Any] = {"n": tm.num_queries}
    for c in report.recall_cutoffs:
        row[f"recall@{c}"] = tm.strict_recall_at.get(c, 0.0)
    row["mrr"] = tm.strict_mrr
    row["precision@k"] = tm.strict_precision_at_k
    row["page_mrr"] = tm.page_mrr
    return row


def _by_type(report: EvalReport) -> dict[str, dict[str, Any]]:
    return {t: _t2_row(report, t) for t in UNIT_TYPES if t in report.by_type}


def _layer_b(report: EvalReport) -> dict[str, Any]:
    overall = (report.answer_quality or {}).get("overall", {})
    return {
        "faithfulness_pass_rate": report.faithfulness_pass_rate,
        "answer_correctness": overall.get("answer_correctness"),
        "citation_accuracy": overall.get("citation_accuracy"),
        "wrong_source_rate": overall.get("wrong_source_rate"),
        "artefact_correctness": overall.get("artefact_correctness"),
    }


def _legacy_block(report: EvalReport, u_type: str) -> dict[str, Any]:
    tm = report.by_type[u_type]
    return {
        f"{u_type}_page_recall_at_k": tm.page_recall_at_k,
        f"{u_type}_page_mrr": tm.page_mrr,
        f"{u_type}_strict_recall_at_k": tm.strict_recall_at_k,
        f"{u_type}_strict_mrr": tm.strict_mrr,
        "faithfulness_pass_rate": report.faithfulness_pass_rate,
    }


def _metric_moved(before: EvalReport, after: EvalReport, u_type: str) -> bool:
    """A fix only counts if a measured number moved (brief E8.5)."""
    b, a = before.by_type[u_type], after.by_type[u_type]
    pairs = [(b.strict_mrr, a.strict_mrr), (b.page_mrr, a.page_mrr), (before.faithfulness_pass_rate or 0.0, after.faithfulness_pass_rate or 0.0)]
    pairs += [(b.strict_recall_at.get(c, 0.0), a.strict_recall_at.get(c, 0.0)) for c in after.recall_cutoffs]
    bq, aq = _layer_b(before), _layer_b(after)
    pairs += [(bq[m] or 0.0, aq[m] or 0.0) for m in ("artefact_correctness", "answer_correctness", "citation_accuracy")]
    return any(after_v > before_v for before_v, after_v in pairs)


def _experiment(info: dict[str, str], target: str, before: EvalReport, after: EvalReport) -> dict[str, Any]:
    delta_mrr = round(after.by_type[target].strict_mrr - before.by_type[target].strict_mrr, 4)
    return {
        **info,
        "target_type": target,
        "before_experiment": before.experiment_name,
        "after_experiment": after.experiment_name,
        "failing_question_ids_before": failing_question_ids(before, unit_type=target, k=1),
        "failing_question_ids_after": failing_question_ids(after, unit_type=target, k=1),
        "before": _legacy_block(before, target),
        "after": _legacy_block(after, target),
        f"delta_{target}_strict_mrr": delta_mrr,
        "delta_faithfulness": round((after.faithfulness_pass_rate or 0.0) - (before.faithfulness_pass_rate or 0.0), 4),
        "by_type_before": _by_type(before),
        "by_type_after": _by_type(after),
        "layer_b_before": _layer_b(before),
        "layer_b_after": _layer_b(after),
        "moved": compare_reports(before, after),
        "measured_move": _metric_moved(before, after, target),
    }


def run_e7_3_failure_mode_experiments(
    base_cfg: AppConfig,
    force_offline: bool = True,
) -> dict[str, Any]:
    """Runs Before/After benchmarks for Failure Mode 1 and Failure Mode 2 (Story E7.3)."""
    runner = _Runner(base_cfg, force_offline)

    # 0. Record the untuned baseline before changing anything
    untuned = runner.run(UNTUNED, "baseline_untuned")

    # 1. FM1 in isolation (FM2 already fixed)
    fm1_before = runner.run(FM1_BEFORE, "fm1_before_caption_drift")
    fm1_after = runner.run(FIXED, "fm1_after_context_grounded_caption")

    # 2. FM2 in isolation (FM1 already fixed), one change per step
    fm2_before = runner.run(FM2_BEFORE, "fm2_before_flat_table_vector_only")
    fm2_step_a = runner.run(FM2_STEP_STRUCTURED_ARTEFACT, "fm2_step_a_structured_artefact")
    fm2_step_b = runner.run(FM2_STEP_COL_VAL, "fm2_step_b_col_val_linearization")
    fm2_after = runner.run(FIXED, "fm2_after_structured_colval_bm25")

    fm2_steps = []
    for label, change, b, a in (
        ("a", "extract.flatten_tables: true -> false", fm2_before, fm2_step_a),
        ("b", "enrich.table_linearization_mode: flat_raw -> col_val_and_markdown", fm2_step_a, fm2_step_b),
        ("c", "retrieve.enable_bm25_fusion: false -> true (alpha 1.0 -> 0.60)", fm2_step_b, fm2_after),
    ):
        fm2_steps.append(
            {
                "step": label,
                "change": change,
                "experiment": a.experiment_name,
                "table": _t2_row(a, "table"),
                "delta_table_strict_mrr": round(a.by_type["table"].strict_mrr - b.by_type["table"].strict_mrr, 4),
                "delta_table_artefact_correctness": round(
                    (_layer_b(a)["artefact_correctness"] or 0.0) - (_layer_b(b)["artefact_correctness"] or 0.0), 4
                ),
                "delta_faithfulness": round((a.faithfulness_pass_rate or 0.0) - (b.faithfulness_pass_rate or 0.0), 4),
                "improved_question_ids": compare_reports(b, a)["improved_question_ids"],
                "regressed_question_ids": compare_reports(b, a)["regressed_question_ids"],
            }
        )

    fm1 = _experiment(FM1_INFO, "figure", fm1_before, fm1_after)
    fm2 = _experiment(FM2_INFO, "table", fm2_before, fm2_after)
    fm2["steps"] = fm2_steps

    # T2 cumulative path: Baseline -> After fix 1 -> After fix 2 (= Final)
    stages = [("Baseline", untuned), ("After fix 1", fm2_before), ("After fix 2", fm2_after), ("Final", fm2_after)]

    summary: dict[str, Any] = {
        "evaluation_note": EVALUATION_NOTE,
        "untuned_baseline": {
            "overall_strict_mrr": untuned.overall_strict_mrr,
            "figure_strict_mrr": untuned.by_type["figure"].strict_mrr,
            "table_strict_mrr": untuned.by_type["table"].strict_mrr,
            "faithfulness_pass_rate": untuned.faithfulness_pass_rate,
            "by_type": _by_type(untuned),
            "layer_b": _layer_b(untuned),
        },
        "failure_mode_1_right_text_wrong_figure": fm1,
        "failure_mode_2_table_mush_and_exact_codes": fm2,
        "t2_cumulative": [
            {"stage": stage, "experiment": rep.experiment_name, "by_type": _by_type(rep), "layer_b": _layer_b(rep)}
            for stage, rep in stages
        ],
        "baseline_vs_final": compare_reports(untuned, fm2_after),
    }

    rep_dir = Path(base_cfg.project.reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "e7_3_failure_mode_ablations.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (rep_dir / "e7_3_failure_mode_ablations.md").write_text(render_markdown_report(summary), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------------------------
# Markdown report (templates T2, T3 and the Layer B part of T4)
# ---------------------------------------------------------------------------------------------
def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def _t2_table(rows: list[tuple[str, str, dict[str, Any]]]) -> list[str]:
    cutoffs = sorted({int(k.split("@")[1]) for _, _, r in rows for k in r if k.startswith("recall@")})
    head = "| Type | n | " + " | ".join(f"Recall@{c}" for c in cutoffs) + " | MRR | Precision@k | Stage |"
    out = [head, "|" + "---|" * (len(cutoffs) + 5)]
    for u_type, stage, r in rows:
        out.append(
            f"| {u_type} | {r['n']} | "
            + " | ".join(_fmt(r.get(f"recall@{c}")) for c in cutoffs)
            + f" | {_fmt(r['mrr'])} | {_fmt(r['precision@k'])} | {stage} |"
        )
    return out


def _ids(ids: list[str]) -> str:
    return ", ".join(ids) if ids else "none"


def render_markdown_report(summary: dict[str, Any]) -> str:
    fm1 = summary["failure_mode_1_right_text_wrong_figure"]
    fm2 = summary["failure_mode_2_table_mush_and_exact_codes"]
    lines = [
        "# E7.3 Failure-mode ablations (before / after)",
        "",
        f"> {summary['evaluation_note']}",
        "",
        "Relevance is strict: the hit must be on a gold page **and** of the expected type.",
        "",
        "## T2 · Layer A retrieval by type (cumulative)",
        "",
    ]
    rows = []
    for u_type in UNIT_TYPES:
        for stage in summary["t2_cumulative"]:
            rows.append((u_type, stage["stage"], stage["by_type"][u_type]))
    lines += _t2_table(rows)

    lines += ["", "## T3 · Failure-mode log", "", "| # | Symptom and question IDs | Root cause | Change made | Before | After |", "|---|---|---|---|---|---|"]
    for idx, fm in ((1, fm1), (2, fm2)):
        t = fm["target_type"]
        b, a = fm["by_type_before"][t], fm["by_type_after"][t]
        lb, la = fm["layer_b_before"], fm["layer_b_after"]
        lines.append(
            f"| {idx} | {fm['symptom']}. Failing at rank 1 before: {_ids(fm['failing_question_ids_before'])}; "
            f"moved: {_ids(fm['moved']['improved_question_ids'])} | {fm['root_cause']} | {fm['change']} | "
            f"{t} MRR {_fmt(b['mrr'])}, R@1 {_fmt(b.get('recall@1'))}, artefact correct {_fmt(lb['artefact_correctness'])}, "
            f"faithfulness {_fmt(lb['faithfulness_pass_rate'])} | "
            f"{t} MRR {_fmt(a['mrr'])}, R@1 {_fmt(a.get('recall@1'))}, artefact correct {_fmt(la['artefact_correctness'])}, "
            f"faithfulness {_fmt(la['faithfulness_pass_rate'])} |"
        )

    lines += ["", "### Failure mode 2, one change at a time", "", "| Step | Change | Table MRR | Δ MRR | Δ artefact correctness | Δ faithfulness | Improved IDs | Regressed IDs |", "|---|---|---|---|---|---|---|---|"]
    for s in fm2["steps"]:
        lines.append(
            f"| {s['step']} | {s['change']} | {_fmt(s['table']['mrr'])} | {s['delta_table_strict_mrr']:+.2f} | "
            f"{s['delta_table_artefact_correctness']:+.2f} | {s['delta_faithfulness']:+.2f} | "
            f"{_ids(s['improved_question_ids'])} | {_ids(s['regressed_question_ids'])} |"
        )

    lines += ["", "### Side effects on the other types", ""]
    for idx, fm in ((1, fm1), (2, fm2)):
        moved = fm["moved"]
        lines.append(
            f"- FM{idx}: improved {_ids(moved['improved_question_ids'])}; regressed {_ids(moved['regressed_question_ids'])}; "
            f"measured move on target type: {'yes' if fm['measured_move'] else 'NO (fix does not count)'}."
        )

    lines += ["", "## Layer B · answer and grounding quality", "", "| Stage | Faithfulness (5) | Answer correctness | Citation accuracy | Wrong-source rate | Artefact correctness |", "|---|---|---|---|---|---|"]
    for stage in summary["t2_cumulative"][:3]:
        lb = stage["layer_b"]
        lines.append(
            f"| {stage['stage']} | {_fmt(lb['faithfulness_pass_rate'])} | {_fmt(lb['answer_correctness'])} | "
            f"{_fmt(lb['citation_accuracy'])} | {_fmt(lb['wrong_source_rate'])} | {_fmt(lb['artefact_correctness'])} |"
        )
    lines.append("")
    return "\n".join(lines)
