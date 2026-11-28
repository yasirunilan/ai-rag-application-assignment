"""Answer & Grounding Quality (Layer B) and the Five-Answer Faithfulness Check
(Person 5 · Stories E7.2 / E8.2 & E8.6).
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Optional

import numpy as np

from mmrag.answer.responder import build_answer_from_hits
from mmrag.config import AppConfig
from mmrag.retrieve.retriever import HybridRetriever
from mmrag.schemas import AnswerResponse, GoldQuestion, TableArtefact

FLAG_OK = "OK_EXACT_ARTEFACT"
FLAG_ABSTAINED = "ABSTAINED_NO_ANSWER"
FLAG_WRONG_FIGURE = "DANGEROUS_WRONG_FIGURE_DRIFT"
FLAG_TABLE_MUSH = "TABLE_FLATTENED_MUSH"
FLAG_WRONG_SOURCE = "WRONG_SOURCE_PAGE_OR_ARTEFACT"


def _gold_citation(q: GoldQuestion) -> str:
    return f"{q.gold_doc_id} p.{q.gold_pages[0]}"


def _is_valid_png_file(path_str: str) -> bool:
    p = Path(path_str)
    if not p.exists() or not p.is_file():
        return False
    try:
        return p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    except OSError:
        return False


def _answer_visible_text(ans: AnswerResponse) -> str:
    """Everything the user actually sees: grounded answer, caption and the artefact content."""
    parts = [ans.grounded_answer or "", ans.grounded_caption or ""]
    if isinstance(ans.artefact, TableArtefact):
        parts.append(ans.artefact.caption or "")
        parts.append(" ".join(ans.artefact.headers))
        parts.extend(" ".join(r) for r in ans.artefact.rows)
    elif ans.modality == "text" and isinstance(ans.artefact, str):
        parts.append(ans.artefact)
    return " ".join(parts)


def judge_answer(q: GoldQuestion, ans: AnswerResponse) -> dict[str, Any]:
    """Grades one answer against its gold question. Pure function: no retrieval, no I/O."""
    if ans.abstained or ans.source_unit is None:
        return {
            "query_id": q.query_id,
            "query": q.query,
            "expected_type": q.expected_type,
            "gold_citation": _gold_citation(q),
            "gold_hint": q.gold_artefact_hint,
            "returned_citation": None,
            "returned_modality": None,
            "abstained": True,
            "abstain_code": ans.abstain_code,
            "citation_correct": False,
            "artefact_correct": False,
            "keyword_coverage": 0.0,
            "answer_correct": False,
            "faithful": False,
            "flag": FLAG_ABSTAINED,
        }

    u = ans.source_unit
    doc_ok = u.provenance.doc_id == q.gold_doc_id
    page_ok = u.provenance.page is not None and int(u.provenance.page) in q.gold_pages
    mod_ok = ans.modality == q.expected_type

    hint_ok = True
    if q.gold_artefact_hint:
        artefact_meta = " ".join(
            [u.provenance.nearby_caption or "", u.provenance.heading_context or "", u.search_text or "", _answer_visible_text(ans)]
        )
        hint_ok = q.gold_artefact_hint.lower() in artefact_meta.lower()

    structure_ok = not (q.expected_type == "table" and not isinstance(ans.artefact, TableArtefact))
    if q.expected_type == "figure" and isinstance(ans.artefact, str) and Path(ans.artefact).exists():
        structure_ok = _is_valid_png_file(ans.artefact)

    visible = _answer_visible_text(ans).lower()
    keywords = q.expected_keywords
    kw_hits = sum(1 for kw in keywords if kw.lower() in visible)
    kw_cov = round(kw_hits / len(keywords), 4) if keywords else (1.0 if page_ok else 0.0)

    citation_correct = doc_ok and page_ok
    artefact_correct = citation_correct and mod_ok and hint_ok and structure_ok
    faithful = artefact_correct

    if faithful:
        flag = FLAG_OK
    elif ans.modality == "figure" and (not citation_correct or (q.expected_type == "figure" and not hint_ok)):
        flag = FLAG_WRONG_FIGURE
    elif q.expected_type == "table" and not structure_ok:
        flag = FLAG_TABLE_MUSH
    elif not mod_ok:
        flag = f"WRONG_MODALITY (expected {q.expected_type}, got {ans.modality})"
    else:
        flag = FLAG_WRONG_SOURCE

    return {
        "query_id": q.query_id,
        "query": q.query,
        "expected_type": q.expected_type,
        "gold_citation": _gold_citation(q),
        "gold_hint": q.gold_artefact_hint,
        "returned_citation": ans.citation,
        "returned_citation_path": ans.citation_path,
        "returned_modality": ans.modality,
        "grounded_caption": ans.grounded_caption,
        "grounded_answer": ans.grounded_answer,
        "abstained": False,
        "abstain_code": None,
        "citation_correct": citation_correct,
        "artefact_correct": artefact_correct,
        "keyword_coverage": kw_cov,
        "answer_correct": kw_cov >= 1.0,
        "faithful": faithful,
        "flag": flag,
    }


def select_spotcheck_sample(questions: list[GoldQuestion], sample_size: int = 5) -> list[GoldQuestion]:
    """Stratified, deterministic sample prioritizing figures and tables (2 figure, 2 table, 1 text)."""
    by_type = {t: [q for q in questions if q.expected_type == t] for t in ("figure", "table", "text")}
    ordered = by_type["figure"][:2] + by_type["table"][:2] + by_type["text"][:1]
    if len(ordered) < sample_size:
        ordered.extend([q for q in questions if q not in ordered][: sample_size - len(ordered)])
    return ordered[:sample_size]


def run_faithfulness_spotcheck(
    retriever: HybridRetriever,
    questions: list[GoldQuestion],
    cfg: AppConfig,
    sample_size: int = 5,
) -> dict[str, Any]:
    """Reviews `sample_size` answers and flags any wrong-artefact or wrong-figure response (E7.2)."""
    sampled = select_spotcheck_sample(questions, sample_size)
    reviews: list[dict[str, Any]] = []
    for q in sampled:
        hits = retriever.retrieve(q.query, k=cfg.retrieve.top_k)
        reviews.append(judge_answer(q, build_answer_from_hits(q.query, hits, cfg, offline=True)))

    passed = sum(1 for r in reviews if r["faithful"])
    abstained = sum(1 for r in reviews if r["abstained"])
    wrong_figs = sum(1 for r in reviews if r["flag"] == FLAG_WRONG_FIGURE)
    return {
        "reviewed_count": len(reviews),
        "passed_count": passed,
        "abstained_count": abstained,
        "flagged_wrong_artefact_count": len(reviews) - passed - abstained,
        "dangerous_wrong_figure_count": wrong_figs,
        "wrong_figure_count": wrong_figs,
        "faithfulness_pass_rate": round(passed / max(1, len(reviews)), 4),
        "reviews": reviews,
    }


def _rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [r for r in rows if not r["abstained"]]
    artefact_rows = [r for r in rows if r["expected_type"] in ("figure", "table")]
    return {
        "n": len(rows),
        "answered": len(answered),
        "abstention_rate": _rate(len(rows) - len(answered), len(rows)),
        "answer_correctness": _rate(sum(r["answer_correct"] for r in rows), len(rows)),
        "mean_keyword_coverage": round(float(np.mean([r["keyword_coverage"] for r in rows])), 4) if rows else None,
        "citation_accuracy": _rate(sum(r["citation_correct"] for r in answered), len(answered)),
        "wrong_source_rate": _rate(sum(not r["citation_correct"] for r in answered), len(answered)),
        "artefact_correctness": _rate(sum(r["artefact_correct"] for r in artefact_rows), len(artefact_rows)),
    }


def evaluate_answer_quality(
    retriever: HybridRetriever,
    questions: list[GoldQuestion],
    cfg: AppConfig,
) -> dict[str, Any]:
    """Layer B over every gold question, overall and by type, with per-question results."""
    rows: list[dict[str, Any]] = []
    answer_ms: list[float] = []
    for q in questions:
        hits = retriever.retrieve(q.query, k=cfg.retrieve.top_k)
        t0 = time.perf_counter()
        ans = build_answer_from_hits(q.query, hits, cfg, offline=True)
        answer_ms.append((time.perf_counter() - t0) * 1000.0)
        rows.append(judge_answer(q, ans))

    by_type = {t: _summarize([r for r in rows if r["expected_type"] == t]) for t in ("text", "figure", "table")}
    latency: dict[str, float] = {}
    if answer_ms:
        arr = np.asarray(answer_ms)
        latency = {
            "answer_p50": round(float(np.percentile(arr, 50)), 3),
            "answer_p95": round(float(np.percentile(arr, 95)), 3),
        }
    return {
        "method": (
            "extractive answers; correctness = all expected_keywords visible in the answer "
            "(grounded answer + caption + artefact); citation = cited doc+page in gold_pages"
        ),
        "overall": _summarize(rows),
        "by_type": by_type,
        "latency_ms": latency,
        "per_question": rows,
    }
