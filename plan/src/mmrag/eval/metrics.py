"""Per-Type Retrieval Evaluator (Layer A): Recall@k, MRR and Precision@k by Text / Figure / Table
(Person 5 · Story E7.1; brief E8.1 / E8.4).

Generator out of the loop: only the retriever is scored. Two relevance definitions:
1. Page-level: `hit.doc_id == gold_doc_id` and `hit.page in gold_pages`.
2. Strict modality+page: the hit must also have `hit.unit.type == gold.expected_type`, so a prose
   chunk on page 212 cannot hide a missed wiring-diagram figure on page 212.

Reports recall at several cut-offs (default 1, 3, 5), MRR, strict precision@k, retrieval latency
p50/p95, and per-question results so the question IDs that move after a fix can be listed:

    python -m mmrag.eval.metrics diff reports/fm1_before_caption_drift.json reports/fm1_after_context_grounded_caption.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from mmrag.config import AppConfig
from mmrag.retrieve.retriever import HybridRetriever
from mmrag.schemas import EvalReport, GoldQuestion, SearchHit, TypeMetrics

UNIT_TYPES = ("text", "figure", "table")
DEFAULT_RECALL_CUTOFFS = (1, 3, 5)


def _is_page_match(hit: SearchHit, q: GoldQuestion) -> bool:
    prov = hit.unit.provenance
    return prov.doc_id == q.gold_doc_id and prov.page is not None and int(prov.page) in q.gold_pages


def _first_relevant_ranks(q: GoldQuestion, hits: list[SearchHit]) -> tuple[Optional[int], Optional[int]]:
    """Returns `(first page-level relevant rank, first strict relevant rank)`; None when absent."""
    page_rank: Optional[int] = None
    strict_rank: Optional[int] = None
    for rank_idx, h in enumerate(hits, start=1):
        if _is_page_match(h, q):
            if page_rank is None:
                page_rank = rank_idx
            if strict_rank is None and h.unit.type == q.expected_type:
                strict_rank = rank_idx
    return page_rank, strict_rank


def _score_single_query(q: GoldQuestion, hits: list[SearchHit]) -> tuple[float, float, float, float]:
    """Returns `(page_recall, page_rr, strict_recall, strict_rr)` for one gold question over `hits`."""
    page_rank, strict_rank = _first_relevant_ranks(q, hits)
    page_rr = 1.0 / page_rank if page_rank else 0.0
    strict_rr = 1.0 / strict_rank if strict_rank else 0.0
    return float(page_rank is not None), page_rr, float(strict_rank is not None), strict_rr


def _recall_at(rank: Optional[int], k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def _mean(values: Sequence[float]) -> float:
    return round(float(sum(values)) / len(values), 4) if values else 0.0


def _percentiles_ms(samples: Sequence[float], prefix: str) -> dict[str, float]:
    if not samples:
        return {}
    arr = np.asarray(samples, dtype=np.float64)
    return {
        f"{prefix}_p50": round(float(np.percentile(arr, 50)), 3),
        f"{prefix}_p95": round(float(np.percentile(arr, 95)), 3),
        f"{prefix}_mean": round(float(arr.mean()), 3),
    }


def _type_metrics(u_type: str, rows: list[dict[str, Any]], k: int, cutoffs: Sequence[int]) -> TypeMetrics:
    page_ranks = [r["page_rank"] for r in rows]
    strict_ranks = [r["strict_rank"] for r in rows]
    return TypeMetrics(
        unit_type=u_type,
        num_queries=len(rows),
        page_recall_at_k=_mean([_recall_at(x, k) for x in page_ranks]),
        page_mrr=_mean([1.0 / x if x and x <= k else 0.0 for x in page_ranks]),
        strict_recall_at_k=_mean([_recall_at(x, k) for x in strict_ranks]),
        strict_mrr=_mean([1.0 / x if x and x <= k else 0.0 for x in strict_ranks]),
        page_recall_at={c: _mean([_recall_at(x, c) for x in page_ranks]) for c in cutoffs},
        strict_recall_at={c: _mean([_recall_at(x, c) for x in strict_ranks]) for c in cutoffs},
        strict_precision_at_k=_mean([r["strict_relevant_in_top_k"] / k for r in rows]),
    )


def evaluate_retrieval_by_type(
    retriever: HybridRetriever,
    questions: list[GoldQuestion],
    cfg: AppConfig,
    experiment_name: str = "baseline",
    k: Optional[int] = None,
    save_report: bool = True,
    recall_cutoffs: Sequence[int] = DEFAULT_RECALL_CUTOFFS,
) -> EvalReport:
    """Evaluates retrieval by `text`, `figure` and `table` with the generator out of the loop (E7.1).

    `k` (default `retrieve.top_k`) is the depth for `*_recall_at_k`, MRR and precision@k; the
    retriever is asked for `max(k, *recall_cutoffs)` hits so every cut-off is measurable.
    """
    top_k = k or cfg.retrieve.top_k
    cutoffs = sorted({int(c) for c in recall_cutoffs if int(c) >= 1})
    depth = max([top_k, *cutoffs])

    per_question: list[dict[str, Any]] = []
    latencies: list[float] = []
    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.retrieve(q.query, k=depth)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        page_rank, strict_rank = _first_relevant_ranks(q, hits)
        top = hits[:top_k]
        per_question.append(
            {
                "query_id": q.query_id,
                "expected_type": q.expected_type,
                "query": q.query,
                "gold": f"{q.gold_doc_id} p.{','.join(str(p) for p in q.gold_pages)}",
                "page_rank": page_rank,
                "strict_rank": strict_rank,
                "strict_relevant_in_top_k": sum(
                    1 for h in top if _is_page_match(h, q) and h.unit.type == q.expected_type
                ),
                "retrieval_ms": round(elapsed_ms, 3),
                "top_hits": [
                    {
                        "rank": h.rank,
                        "unit_id": h.unit.unit_id,
                        "type": h.unit.type,
                        "citation": h.citation,
                        "score": h.score,
                        "relevant": _is_page_match(h, q) and h.unit.type == q.expected_type,
                    }
                    for h in top
                ],
            }
        )

    by_type: dict[str, TypeMetrics] = {}
    types = list(UNIT_TYPES) + sorted({r["expected_type"] for r in per_question} - set(UNIT_TYPES))
    for u_type in types:
        rows = [r for r in per_question if r["expected_type"] == u_type]
        by_type[u_type] = _type_metrics(u_type, rows, top_k, cutoffs)
    overall = _type_metrics("overall", per_question, top_k, cutoffs)

    report = EvalReport(
        experiment_name=experiment_name,
        k=top_k,
        overall_page_recall_at_k=overall.page_recall_at_k,
        overall_page_mrr=overall.page_mrr,
        overall_strict_recall_at_k=overall.strict_recall_at_k,
        overall_strict_mrr=overall.strict_mrr,
        by_type=by_type,
        recall_cutoffs=cutoffs,
        per_question=per_question,
        latency_ms=_percentiles_ms(latencies, "retrieval"),
        config_snapshot={
            "include_nearby_context": cfg.enrich.include_nearby_context,
            "flatten_tables": cfg.extract.flatten_tables,
            "table_linearization_mode": cfg.enrich.table_linearization_mode,
            "enable_bm25_fusion": cfg.retrieve.enable_bm25_fusion,
            "alpha": cfg.retrieve.alpha,
            "ocr_gate_enabled": cfg.ingest.ocr_quality_gate.enabled,
            "embedding_provider": cfg.index.embedding_provider,
            "embedding_model_name": cfg.index.embedding_model_name,
        },
    )

    if save_report:
        save_eval_report(report, cfg)
    return report


def save_eval_report(report: EvalReport, cfg: AppConfig) -> Path:
    rep_dir = Path(cfg.project.reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    out_path = rep_dir / f"{report.experiment_name}.json"
    out_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------------------------
# Before / after comparison (E7.3 needs "the question IDs that moved")
# ---------------------------------------------------------------------------------------------
def _rank_key(rank: Optional[int]) -> float:
    return float(rank) if rank else float("inf")


def compare_reports(before: EvalReport, after: EvalReport) -> dict[str, Any]:
    """Per-type metric deltas plus the question IDs whose strict rank improved or regressed."""
    by_type: dict[str, dict[str, Any]] = {}
    for u_type in sorted(set(before.by_type) | set(after.by_type)):
        b, a = before.by_type.get(u_type), after.by_type.get(u_type)
        if b is None or a is None:
            continue
        row: dict[str, Any] = {
            "n": a.num_queries,
            "strict_mrr": {"before": b.strict_mrr, "after": a.strict_mrr, "delta": round(a.strict_mrr - b.strict_mrr, 4)},
            "page_mrr": {"before": b.page_mrr, "after": a.page_mrr, "delta": round(a.page_mrr - b.page_mrr, 4)},
        }
        for c in sorted(set(b.strict_recall_at) & set(a.strict_recall_at)):
            row[f"strict_recall@{c}"] = {
                "before": b.strict_recall_at[c],
                "after": a.strict_recall_at[c],
                "delta": round(a.strict_recall_at[c] - b.strict_recall_at[c], 4),
            }
        by_type[u_type] = row

    before_q = {r["query_id"]: r for r in before.per_question}
    improved, regressed = [], []
    for r in after.per_question:
        prev = before_q.get(r["query_id"])
        if prev is None:
            continue
        b_rank, a_rank = _rank_key(prev["strict_rank"]), _rank_key(r["strict_rank"])
        move = {"query_id": r["query_id"], "type": r["expected_type"], "strict_rank_before": prev["strict_rank"], "strict_rank_after": r["strict_rank"]}
        if a_rank < b_rank:
            improved.append(move)
        elif a_rank > b_rank:
            regressed.append(move)

    return {
        "before": before.experiment_name,
        "after": after.experiment_name,
        "by_type": by_type,
        "improved_question_ids": [m["query_id"] for m in improved],
        "regressed_question_ids": [m["query_id"] for m in regressed],
        "moves": improved + regressed,
    }


def failing_question_ids(report: EvalReport, unit_type: Optional[str] = None, k: int = 1) -> list[str]:
    """Question IDs whose strict-relevant unit is not within the top `k` (the Stage 4 worklist)."""
    return [
        r["query_id"]
        for r in report.per_question
        if (unit_type is None or r["expected_type"] == unit_type) and _recall_at(r["strict_rank"], k) == 0.0
    ]


def load_eval_report(path: str | Path) -> EvalReport:
    return EvalReport.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mmrag.eval.metrics", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_diff = sub.add_parser("diff", help="Compare two saved eval reports and list moved question IDs")
    p_diff.add_argument("before")
    p_diff.add_argument("after")
    args = parser.parse_args(argv)
    if args.command == "diff":
        print(json.dumps(compare_reports(load_eval_report(args.before), load_eval_report(args.after)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
