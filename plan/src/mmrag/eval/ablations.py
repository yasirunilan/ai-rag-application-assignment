"""Failure Mode Reproduction & Before/After Engineering Harness (Person 5 · Stories E7.3 / E8.5 & E8.7).

Reproduces and fixes two Appendix failure modes with per-type before/after metrics
and exact moved question IDs:
1. Failure Mode 1 ("Right text, wrong figure" / caption–image drift)
2. Failure Mode 2 ("Table answers are mush & exact part codes missed")
Also quantifies cost levers (selective figure captioning + SHA-256 disk cache + batch-16 embedding).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from mmrag.config import AppConfig
from mmrag.index.embedder import EMBED_BATCH_SIZE
from mmrag.pipeline import MultimodalRAGPipeline
from mmrag.schemas import EvalReport


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


def _run_config_eval(cfg: AppConfig, name: str, force_offline: bool = True) -> tuple[EvalReport, dict[str, Any]]:
    pipe = MultimodalRAGPipeline(cfg, force_offline=force_offline)
    stats = pipe.ingest_and_index(persist_index=False)
    return pipe.evaluate(experiment_name=name, save_report=True), stats


def _diff_question_ids(before: EvalReport, after: EvalReport) -> tuple[list[str], list[str]]:
    """Lists exact question IDs whose strict rank improved or regressed (Stories E8.4 & E8.5)."""
    b_map = {q.query_id: (q.strict_rank or 9999) for q in before.per_question}
    a_map = {q.query_id: (q.strict_rank or 9999) for q in after.per_question}
    improved = sorted(qid for qid in a_map if a_map[qid] < b_map.get(qid, 9999))
    regressed = sorted(qid for qid in a_map if a_map[qid] > b_map.get(qid, 9999))
    return improved, regressed


def _type_table_snapshot(report: EvalReport) -> dict[str, dict[str, float]]:
    return {
        t: {
            "n": float(tm.num_queries),
            "recall_at_1": tm.strict_recall_at_1,
            "recall_at_3": tm.strict_recall_at_3,
            "recall_at_5": tm.strict_recall_at_k,
            "mrr": tm.strict_mrr,
        }
        for t, tm in report.by_type.items()
    }


def run_e7_3_failure_mode_experiments(
    base_cfg: AppConfig,
    force_offline: bool = True,
) -> dict[str, Any]:
    """Runs Before/After benchmarks for Failure Mode 1 and Failure Mode 2 (Stories E7.3 / E8.5)."""
    # -------------------------------------------------------------------------
    # Experiment 1: Failure Mode 1 — "Right text, wrong figure" (Caption-Image Drift)
    # -------------------------------------------------------------------------
    fm1_before_cfg = _clone_cfg(
        base_cfg,
        include_nearby_context=False,
        flatten_tables=False,
        table_linearization_mode="col_val_and_markdown",
        enable_bm25_fusion=True,
        alpha=0.60,
    )
    fm1_after_cfg = _clone_cfg(
        base_cfg,
        include_nearby_context=True,
        flatten_tables=False,
        table_linearization_mode="col_val_and_markdown",
        enable_bm25_fusion=True,
        alpha=0.60,
    )
    fm1_before, _ = _run_config_eval(fm1_before_cfg, "fm1_before_caption_drift", force_offline=force_offline)
    fm1_after, final_stats = _run_config_eval(
        fm1_after_cfg, "fm1_after_context_grounded_caption", force_offline=force_offline
    )
    fm1_improved, fm1_regressed = _diff_question_ids(fm1_before, fm1_after)

    # -------------------------------------------------------------------------
    # Experiment 2: Failure Mode 2 — "Table answers are mush & exact part codes missed"
    # -------------------------------------------------------------------------
    fm2_before_cfg = _clone_cfg(
        base_cfg,
        include_nearby_context=True,
        flatten_tables=True,
        table_linearization_mode="flat_raw",
        enable_bm25_fusion=False,
        alpha=1.0,
    )
    fm2_after_cfg = _clone_cfg(
        base_cfg,
        include_nearby_context=True,
        flatten_tables=False,
        table_linearization_mode="col_val_and_markdown",
        enable_bm25_fusion=True,
        alpha=0.60,
    )
    fm2_before, _ = _run_config_eval(fm2_before_cfg, "fm2_before_flat_table_vector_only", force_offline=force_offline)
    fm2_after, _ = _run_config_eval(fm2_after_cfg, "fm2_after_structured_colval_bm25", force_offline=force_offline)
    fm2_improved, fm2_regressed = _diff_question_ids(fm2_before, fm2_after)

    # -------------------------------------------------------------------------
    # Untuned Baseline (Both Failure Modes Active)
    # -------------------------------------------------------------------------
    untuned_cfg = _clone_cfg(
        base_cfg,
        include_nearby_context=False,
        flatten_tables=True,
        table_linearization_mode="flat_raw",
        enable_bm25_fusion=False,
        alpha=1.0,
    )
    untuned_baseline, _ = _run_config_eval(untuned_cfg, "baseline_untuned", force_offline=force_offline)

    # Cost levers (Story E8.5 Step 5 & Story E8.7)
    num_pages = int(final_stats.get("num_pages", 0))
    num_figures = int(final_stats.get("num_figure_units", 0))
    total_units = int(final_stats.get("total_indexed_units", 0))
    batched_embed_calls = math.ceil(total_units / EMBED_BATCH_SIZE) if total_units else 0

    summary = {
        "untuned_baseline": {
            "overall_strict_mrr": untuned_baseline.overall_strict_mrr,
            "figure_strict_mrr": untuned_baseline.by_type["figure"].strict_mrr,
            "table_strict_mrr": untuned_baseline.by_type["table"].strict_mrr,
            "faithfulness_pass_rate": untuned_baseline.faithfulness_pass_rate,
            "by_type": _type_table_snapshot(untuned_baseline),
        },
        "failure_mode_1_right_text_wrong_figure": {
            "symptom": "Caption-image drift causes wrong relay diagram (p.2) to outrank starter relay diagram (p.3)",
            "root_cause": "Vision captioning saw only internal schematic shapes without the PDF's nearby figure label, heading, or aircraft model variant",
            "fix": "Inject nearby figure label, section heading, and model variant into vision caption prompt & search_text",
            "improved_question_ids": fm1_improved,
            "regressed_question_ids": fm1_regressed,
            "before": {
                "figure_page_recall_at_k": fm1_before.by_type["figure"].page_recall_at_k,
                "figure_page_mrr": fm1_before.by_type["figure"].page_mrr,
                "figure_strict_recall_at_k": fm1_before.by_type["figure"].strict_recall_at_k,
                "figure_strict_mrr": fm1_before.by_type["figure"].strict_mrr,
                "faithfulness_pass_rate": fm1_before.faithfulness_pass_rate,
                "by_type": _type_table_snapshot(fm1_before),
            },
            "after": {
                "figure_page_recall_at_k": fm1_after.by_type["figure"].page_recall_at_k,
                "figure_page_mrr": fm1_after.by_type["figure"].page_mrr,
                "figure_strict_recall_at_k": fm1_after.by_type["figure"].strict_recall_at_k,
                "figure_strict_mrr": fm1_after.by_type["figure"].strict_mrr,
                "faithfulness_pass_rate": fm1_after.faithfulness_pass_rate,
                "by_type": _type_table_snapshot(fm1_after),
            },
            "delta_figure_strict_mrr": round(
                fm1_after.by_type["figure"].strict_mrr - fm1_before.by_type["figure"].strict_mrr, 4
            ),
            "delta_faithfulness": round(
                (fm1_after.faithfulness_pass_rate or 0.0) - (fm1_before.faithfulness_pass_rate or 0.0), 4
            ),
        },
        "failure_mode_2_table_mush_and_exact_codes": {
            "symptom": "Flattened table string loses column structure and dense vectors smear exact part/fault codes (AN4-12A, NAS1304-15, FAULT-E409)",
            "root_cause": "Flattening destroys 2D cell-header bindings and dense embeddings map distinct alphanumeric part codes into a shared generic cluster",
            "fix": "Keep structured TableArtefact + col=val row linearization + BM25 lexical fusion (alpha=0.60)",
            "improved_question_ids": fm2_improved,
            "regressed_question_ids": fm2_regressed,
            "before": {
                "table_page_recall_at_k": fm2_before.by_type["table"].page_recall_at_k,
                "table_page_mrr": fm2_before.by_type["table"].page_mrr,
                "table_strict_recall_at_k": fm2_before.by_type["table"].strict_recall_at_k,
                "table_strict_mrr": fm2_before.by_type["table"].strict_mrr,
                "faithfulness_pass_rate": fm2_before.faithfulness_pass_rate,
                "by_type": _type_table_snapshot(fm2_before),
            },
            "after": {
                "table_page_recall_at_k": fm2_after.by_type["table"].page_recall_at_k,
                "table_page_mrr": fm2_after.by_type["table"].page_mrr,
                "table_strict_recall_at_k": fm2_after.by_type["table"].strict_recall_at_k,
                "table_strict_mrr": fm2_after.by_type["table"].strict_mrr,
                "faithfulness_pass_rate": fm2_after.faithfulness_pass_rate,
                "by_type": _type_table_snapshot(fm2_after),
            },
            "delta_table_strict_mrr": round(
                fm2_after.by_type["table"].strict_mrr - fm2_before.by_type["table"].strict_mrr, 4
            ),
            "delta_faithfulness": round(
                (fm2_after.faithfulness_pass_rate or 0.0) - (fm2_before.faithfulness_pass_rate or 0.0), 4
            ),
        },
        "cost_levers": {
            "naive_every_page_vision_calls": num_pages,
            "selective_first_run_vision_calls": num_figures,
            "cached_rerun_vision_calls": 0,
            "unbatched_unit_embed_calls": total_units,
            "batch_16_unit_embed_calls": batched_embed_calls,
            "calls_saved_on_rerun": num_pages + max(0, total_units - batched_embed_calls),
        },
    }

    rep_dir = Path(base_cfg.project.reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    out_file = rep_dir / "e7_3_failure_mode_ablations.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
