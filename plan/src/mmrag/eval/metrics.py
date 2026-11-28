"""Three-Layer Evaluation Harness: Layers A, B, and C (Person 5 · Stories E7.1 / E8.1–E8.4).

- Layer A (Retrieval, generator out of loop, by type): Recall@1, Recall@3, Recall@5/@k,
  Precision@k, and MRR for both Page-Level and Strict Modality+Page matching, plus per-question trace.
- Layer B (Answer & Grounding): Answer correctness (key-fact rubric), faithfulness,
  citation accuracy, explicit wrong-source rate, and artefact correctness.
- Layer C (Operational): Retrieval vs generation latency (p50 / p95), calls per query,
  ingestion time, index size on disk, and queries/min headroom under the 20 req/min free-tier cap.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from mmrag.answer.responder import build_answer_from_hits
from mmrag.config import AppConfig, get_openrouter_api_key
from mmrag.eval.faithfulness import _is_valid_png_file, run_faithfulness_spotcheck
from mmrag.retrieve.retriever import HybridRetriever
from mmrag.schemas import (
    EvalReport,
    GoldQuestion,
    LayerBMetrics,
    LayerCMetrics,
    QuestionEvalResult,
    SearchHit,
    TableArtefact,
    TypeMetrics,
)

FREE_TIER_RPM_CAP = 20.0


@dataclass
class _QueryScore:
    page_r1: float
    page_r3: float
    page_rk: float
    page_prec: float
    page_rr: float
    strict_r1: float
    strict_r3: float
    strict_rk: float
    strict_prec: float
    strict_rr: float
    trace: QuestionEvalResult


def _score_single_query_detailed(q: GoldQuestion, hits: list[SearchHit], top_k: int) -> _QueryScore:
    page_rank: Optional[int] = None
    strict_rank: Optional[int] = None
    page_rel_count = 0
    strict_rel_count = 0

    for rank_idx, h in enumerate(hits[:top_k], start=1):
        u = h.unit
        doc_match = u.provenance.doc_id == q.gold_doc_id
        page_match = (u.provenance.page is not None) and (int(u.provenance.page) in q.gold_pages)
        if doc_match and page_match:
            page_rel_count += 1
            if page_rank is None:
                page_rank = rank_idx
            if u.type == q.expected_type:
                strict_rel_count += 1
                if strict_rank is None:
                    strict_rank = rank_idx

    denom = max(1, top_k)
    top_hit = hits[0] if hits else None
    trace = QuestionEvalResult(
        query_id=q.query_id,
        expected_type=q.expected_type,
        gold_doc_id=q.gold_doc_id,
        gold_pages=list(q.gold_pages),
        top_hit_unit_id=top_hit.unit.unit_id if top_hit else None,
        top_hit_citation=top_hit.citation if top_hit else None,
        top_hit_type=top_hit.unit.type if top_hit else None,
        page_rank=page_rank,
        strict_rank=strict_rank,
        strict_hit_at_1=(strict_rank == 1),
        strict_hit_at_3=(strict_rank is not None and strict_rank <= 3),
        strict_hit_at_5=(strict_rank is not None and strict_rank <= 5),
    )
    return _QueryScore(
        page_r1=1.0 if page_rank == 1 else 0.0,
        page_r3=1.0 if (page_rank is not None and page_rank <= 3) else 0.0,
        page_rk=1.0 if page_rank is not None else 0.0,
        page_prec=page_rel_count / denom,
        page_rr=(1.0 / page_rank) if page_rank is not None else 0.0,
        strict_r1=1.0 if strict_rank == 1 else 0.0,
        strict_r3=1.0 if (strict_rank is not None and strict_rank <= 3) else 0.0,
        strict_rk=1.0 if strict_rank is not None else 0.0,
        strict_prec=strict_rel_count / denom,
        strict_rr=(1.0 / strict_rank) if strict_rank is not None else 0.0,
        trace=trace,
    )


def evaluate_retrieval_by_type(
    retriever: HybridRetriever,
    questions: list[GoldQuestion],
    cfg: AppConfig,
    experiment_name: str = "baseline",
    k: Optional[int] = None,
    save_report: bool = True,
) -> EvalReport:
    """Layer A: Evaluates Recall@1/3/k, Precision@k, and MRR by `text`, `figure`, and `table`."""
    top_k = k or cfg.retrieve.top_k
    buckets: dict[str, list[_QueryScore]] = {"text": [], "figure": [], "table": []}
    all_scores: list[_QueryScore] = []

    for q in questions:
        hits = retriever.retrieve(q.query, k=top_k)
        s = _score_single_query_detailed(q, hits, top_k=top_k)
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
                page_recall_at_1=round(sum(x.page_r1 for x in scores) / n, 4),
                page_recall_at_3=round(sum(x.page_r3 for x in scores) / n, 4),
                page_recall_at_k=round(sum(x.page_rk for x in scores) / n, 4),
                page_precision_at_k=round(sum(x.page_prec for x in scores) / n, 4),
                page_mrr=round(sum(x.page_rr for x in scores) / n, 4),
                strict_recall_at_1=round(sum(x.strict_r1 for x in scores) / n, 4),
                strict_recall_at_3=round(sum(x.strict_r3 for x in scores) / n, 4),
                strict_recall_at_k=round(sum(x.strict_rk for x in scores) / n, 4),
                strict_precision_at_k=round(sum(x.strict_prec for x in scores) / n, 4),
                strict_mrr=round(sum(x.strict_rr for x in scores) / n, 4),
            )

    total_n = max(1, len(all_scores))
    report = EvalReport(
        experiment_name=experiment_name,
        k=top_k,
        overall_page_recall_at_k=round(sum(x.page_rk for x in all_scores) / total_n, 4),
        overall_page_mrr=round(sum(x.page_rr for x in all_scores) / total_n, 4),
        overall_strict_recall_at_k=round(sum(x.strict_rk for x in all_scores) / total_n, 4),
        overall_strict_mrr=round(sum(x.strict_rr for x in all_scores) / total_n, 4),
        by_type=by_type,
        per_question=[x.trace for x in all_scores],
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


def evaluate_layers_b_and_c(
    retriever: HybridRetriever,
    questions: list[GoldQuestion],
    cfg: AppConfig,
    offline: bool = True,
    ingestion_time_ms: Optional[float] = None,
) -> tuple[LayerBMetrics, LayerCMetrics, dict]:
    """Evaluates Layer B (answer/grounding) and Layer C (operational latency/cost) together."""
    faith = run_faithfulness_spotcheck(retriever, questions, cfg, sample_size=5)

    ret_times: list[float] = []
    gen_times: list[float] = []
    tot_times: list[float] = []
    call_counts: list[int] = []

    correctness_scores: list[float] = []
    cited_right = 0
    confident_answers = 0
    wrong_source_answers = 0
    abstained_count = 0
    artefact_questions = 0
    artefact_correct = 0

    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.retrieve(q.query, k=cfg.retrieve.top_k)
        t1 = time.perf_counter()
        ans = build_answer_from_hits(q.query, hits, cfg, offline=offline)
        t2 = time.perf_counter()

        r_ms = (t1 - t0) * 1000.0
        g_ms = (t2 - t1) * 1000.0
        ret_times.append(r_ms)
        gen_times.append(g_ms)
        tot_times.append(r_ms + g_ms)
        call_counts.append(ans.model_calls)

        if q.expected_type in ("figure", "table"):
            artefact_questions += 1

        if ans.abstained or ans.source_unit is None:
            abstained_count += 1
            correctness_scores.append(0.0)
            continue

        confident_answers += 1
        u = ans.source_unit
        doc_ok = u.provenance.doc_id == q.gold_doc_id
        page_ok = (u.provenance.page is not None) and (int(u.provenance.page) in q.gold_pages)
        hint_ok = True
        if q.gold_artefact_hint:
            meta_blob = " ".join(
                [u.provenance.nearby_caption or "", u.provenance.heading_context or "", u.search_text or ""]
            )
            hint_ok = q.gold_artefact_hint.lower() in meta_blob.lower()

        if doc_ok and page_ok:
            cited_right += 1
        if not (doc_ok and page_ok and hint_ok):
            wrong_source_answers += 1

        # Artefact correctness for figure & table questions
        if q.expected_type == "figure":
            if doc_ok and page_ok and hint_ok and ans.modality == "figure" and isinstance(ans.artefact, str) and _is_valid_png_file(ans.artefact):
                artefact_correct += 1
        elif q.expected_type == "table":
            if doc_ok and page_ok and hint_ok and ans.modality == "table" and isinstance(ans.artefact, TableArtefact) and bool(ans.artefact.rows):
                artefact_correct += 1

        # Key-fact rubric score over expected_keywords (0 if wrong source page)
        if not (doc_ok and page_ok):
            correctness_scores.append(0.0)
        elif not q.expected_keywords:
            correctness_scores.append(1.0)
        else:
            if isinstance(ans.artefact, TableArtefact):
                evidence_text = ans.artefact.to_markdown() + " " + (ans.grounded_caption or "")
            elif ans.modality == "figure":
                evidence_text = (u.search_text or "") + " " + (ans.grounded_caption or "")
            else:
                evidence_text = str(ans.artefact) + " " + (ans.grounded_answer or "")
            ev_lower = evidence_text.lower()
            matched = sum(1 for kw in q.expected_keywords if kw.lower() in ev_lower)
            correctness_scores.append(matched / len(q.expected_keywords))

    n_q = max(1, len(questions))
    layer_b = LayerBMetrics(
        answer_correctness=round(sum(correctness_scores) / n_q, 4),
        faithfulness_pass_rate=faith["faithfulness_pass_rate"],
        citation_accuracy=round(cited_right / n_q, 4),
        wrong_source_rate=round(wrong_source_answers / max(1, confident_answers), 4),
        artefact_correctness=round(artefact_correct / max(1, artefact_questions), 4),
        abstention_count=abstained_count,
    )

    # Read persisted index size and ingestion_time_ms if available
    idx_dir = Path(cfg.project.index_dir)
    idx_bytes = sum(f.stat().st_size for f in idx_dir.glob("*") if f.is_file()) if idx_dir.exists() else 0
    ing_ms = ingestion_time_ms or 0.0
    rep_file = idx_dir / "index_report.json"
    if not ing_ms and rep_file.exists():
        try:
            ing_ms = float(json.loads(rep_file.read_text(encoding="utf-8")).get("ingestion_time_ms", 0.0))
        except Exception:  # noqa: BLE001
            ing_ms = 0.0

    def _pct(vals: list[float], q_pct: float) -> float:
        return round(float(np.percentile(vals, q_pct)), 2) if vals else 0.0

    mean_calls = round(sum(call_counts) / n_q, 2)
    # Under OpenRouter's 20 req/min cap: if remote calls are made, QPM = 20 / calls_per_query;
    # if 0 remote calls per query (local embeddings + artefact return), QPM is latency-bound,
    # while remote generation mode (1–2 calls/query) allows 10–20 QPM under the free cap.
    effective_remote_calls = mean_calls if mean_calls > 0 else (
        1.0 if (not offline and cfg.answer.enable_remote_generation and get_openrouter_api_key()) else 0.0
    )
    qpm_under_cap = (
        round(FREE_TIER_RPM_CAP / effective_remote_calls, 2)
        if effective_remote_calls > 0
        else FREE_TIER_RPM_CAP  # worst-case 1 remote call/query ceiling under 20 RPM cap
    )

    layer_c = LayerCMetrics(
        retrieval_p50_ms=_pct(ret_times, 50),
        retrieval_p95_ms=_pct(ret_times, 95),
        generation_p50_ms=_pct(gen_times, 50),
        generation_p95_ms=_pct(gen_times, 95),
        total_p50_ms=_pct(tot_times, 50),
        total_p95_ms=_pct(tot_times, 95),
        calls_per_query=mean_calls,
        ingestion_time_ms=round(ing_ms, 2),
        index_size_bytes=idx_bytes,
        queries_per_min_under_cap=qpm_under_cap,
        rate_limit_headroom_qpm=round(max(0.0, FREE_TIER_RPM_CAP - effective_remote_calls), 2),
    )
    return layer_b, layer_c, faith
