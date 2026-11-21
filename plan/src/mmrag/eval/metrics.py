"""Per-Type Retrieval Evaluator: Recall@k and MRR by Text / Figure / Table (Person 5 · Story E7.1).

Computes both:
1. Page-Level Recall@k & MRR: A hit is relevant if `hit.unit.provenance.doc_id == gold_doc_id`
   and `hit.unit.provenance.page in gold_pages`.
2. Strict Modality+Page Recall@k & MRR: The hit must also match `hit.unit.type == gold.expected_type`
   (ensuring a prose chunk on page 212 cannot disguise a missed wiring diagram figure on page 212).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from mmrag.config import AppConfig
from mmrag.retrieve.retriever import HybridRetriever
from mmrag.schemas import EvalReport, GoldQuestion, SearchHit, TypeMetrics


def _score_single_query(
    q: GoldQuestion,
    hits: list[SearchHit],
) -> tuple[float, float, float, float]:
    """Returns `(page_recall, page_rr, strict_recall, strict_rr)` for one gold question."""
    page_rr = 0.0
    strict_rr = 0.0

    for rank_idx, h in enumerate(hits, start=1):
        u = h.unit
        doc_match = u.provenance.doc_id == q.gold_doc_id
        page_match = (u.provenance.page is not None) and (int(u.provenance.page) in q.gold_pages)

        if doc_match and page_match:
            if page_rr == 0.0:
                page_rr = 1.0 / rank_idx
            if u.type == q.expected_type and strict_rr == 0.0:
                strict_rr = 1.0 / rank_idx

    page_recall = 1.0 if page_rr > 0.0 else 0.0
    strict_recall = 1.0 if strict_rr > 0.0 else 0.0
    return page_recall, page_rr, strict_recall, strict_rr


def evaluate_retrieval_by_type(
    retriever: HybridRetriever,
    questions: list[GoldQuestion],
    cfg: AppConfig,
    experiment_name: str = "baseline",
    k: Optional[int] = None,
    save_report: bool = True,
) -> EvalReport:
    """Evaluates Recall@k and MRR broken out by `text`, `figure`, and `table` (Story E7.1)."""
    top_k = k or cfg.retrieve.top_k
    buckets: dict[str, list[tuple[float, float, float, float]]] = {
        "text": [],
        "figure": [],
        "table": [],
    }
    all_scores: list[tuple[float, float, float, float]] = []

    for q in questions:
        hits = retriever.retrieve(q.query, k=top_k)
        s = _score_single_query(q, hits)
        buckets.setdefault(q.expected_type, []).append(s)
        all_scores.append(s)

    by_type: dict[str, TypeMetrics] = {}
    for u_type, scores in buckets.items():
        n = len(scores)
        if n == 0:
            by_type[u_type] = TypeMetrics(
                unit_type=u_type,
                num_queries=0,
                page_recall_at_k=0.0,
                page_mrr=0.0,
                strict_recall_at_k=0.0,
                strict_mrr=0.0,
            )
        else:
            by_type[u_type] = TypeMetrics(
                unit_type=u_type,
                num_queries=n,
                page_recall_at_k=round(sum(x[0] for x in scores) / n, 4),
                page_mrr=round(sum(x[1] for x in scores) / n, 4),
                strict_recall_at_k=round(sum(x[2] for x in scores) / n, 4),
                strict_mrr=round(sum(x[3] for x in scores) / n, 4),
            )

    total_n = max(1, len(all_scores))
    report = EvalReport(
        experiment_name=experiment_name,
        k=top_k,
        overall_page_recall_at_k=round(sum(x[0] for x in all_scores) / total_n, 4),
        overall_page_mrr=round(sum(x[1] for x in all_scores) / total_n, 4),
        overall_strict_recall_at_k=round(sum(x[2] for x in all_scores) / total_n, 4),
        overall_strict_mrr=round(sum(x[3] for x in all_scores) / total_n, 4),
        by_type=by_type,
        config_snapshot={
            "include_nearby_context": cfg.enrich.include_nearby_context,
            "flatten_tables": cfg.extract.flatten_tables,
            "table_linearization_mode": cfg.enrich.table_linearization_mode,
            "enable_bm25_fusion": cfg.retrieve.enable_bm25_fusion,
            "alpha": cfg.retrieve.alpha,
            "ocr_gate_enabled": cfg.ingest.ocr_quality_gate.enabled,
        },
    )

    if save_report:
        rep_dir = Path(cfg.project.reports_dir)
        rep_dir.mkdir(parents=True, exist_ok=True)
        out_path = rep_dir / f"{experiment_name}.json"
        out_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")

    return report
