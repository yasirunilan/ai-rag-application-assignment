"""Faithfulness Spot-Checker: Catches 'Right Text, Wrong Figure' & Flattened Tables (Person 5 · Story E7.2).

Reviews five answers end-to-end against gold evidence pages and flags the most
dangerous failure mode: an answer that sounds plausible / returns an image,
but shows the WRONG figure or a flattened mush table.
"""

from __future__ import annotations

from typing import Any

from mmrag.answer.responder import build_answer_from_hits
from mmrag.config import AppConfig
from mmrag.retrieve.retriever import HybridRetriever
from mmrag.schemas import GoldQuestion, TableArtefact


def run_faithfulness_spotcheck(
    retriever: HybridRetriever,
    questions: list[GoldQuestion],
    cfg: AppConfig,
    sample_size: int = 5,
) -> dict[str, Any]:
    """Reviews `sample_size` answers and flags any wrong-artefact or wrong-figure response (Story E7.2)."""
    # Pick a stratified 5-question sample prioritizing figures and tables
    figures = [q for q in questions if q.expected_type == "figure"]
    tables = [q for q in questions if q.expected_type == "table"]
    texts = [q for q in questions if q.expected_type == "text"]
    ordered = figures[:2] + tables[:2] + texts[:1]
    if len(ordered) < sample_size:
        remaining = [q for q in questions if q not in ordered]
        ordered.extend(remaining[: sample_size - len(ordered)])
    sampled = ordered[:sample_size]

    reviews: list[dict[str, Any]] = []
    passed_count = 0
    flagged_wrong_artefact_count = 0

    for q in sampled:
        hits = retriever.retrieve(q.query, k=cfg.retrieve.top_k)
        ans = build_answer_from_hits(q.query, hits, cfg)

        if ans.abstained or ans.source_unit is None:
            reviews.append(
                {
                    "query_id": q.query_id,
                    "query": q.query,
                    "expected_type": q.expected_type,
                    "gold_citation": f"{q.gold_doc_id} p.{q.gold_pages[0]}",
                    "returned_citation": None,
                    "returned_modality": None,
                    "faithful": False,
                    "flag": "ABSTAINED_NO_ANSWER",
                }
            )
            flagged_wrong_artefact_count += 1
            continue

        u = ans.source_unit
        doc_ok = u.provenance.doc_id == q.gold_doc_id
        page_ok = (u.provenance.page is not None) and (int(u.provenance.page) in q.gold_pages)
        mod_ok = ans.modality == q.expected_type

        # Check hint & structure
        hint_ok = True
        if q.gold_artefact_hint:
            combined_meta = " ".join(
                [
                    u.provenance.nearby_caption or "",
                    u.provenance.heading_context or "",
                    u.search_text or "",
                ]
            )
            hint_ok = q.gold_artefact_hint.lower() in combined_meta.lower()

        structure_ok = True
        if q.expected_type == "table" and not isinstance(ans.artefact, TableArtefact):
            structure_ok = False

        faithful = doc_ok and page_ok and mod_ok and hint_ok and structure_ok
        flag = "OK_EXACT_ARTEFACT"
        if not faithful:
            flagged_wrong_artefact_count += 1
            if q.expected_type == "figure" and ans.modality == "figure" and not (doc_ok and page_ok and hint_ok):
                flag = "DANGEROUS_WRONG_FIGURE_DRIFT"
            elif q.expected_type == "table" and not structure_ok:
                flag = "TABLE_FLATTENED_MUSH"
            elif not mod_ok:
                flag = f"WRONG_MODALITY (expected {q.expected_type}, got {ans.modality})"
            else:
                flag = "WRONG_SOURCE_PAGE_OR_ARTEFACT"
        else:
            passed_count += 1

        reviews.append(
            {
                "query_id": q.query_id,
                "query": q.query,
                "expected_type": q.expected_type,
                "gold_citation": f"{q.gold_doc_id} p.{q.gold_pages[0]}",
                "gold_hint": q.gold_artefact_hint,
                "returned_citation": ans.citation,
                "returned_modality": ans.modality,
                "grounded_caption": ans.grounded_caption,
                "faithful": faithful,
                "flag": flag,
            }
        )

    pass_rate = round(passed_count / max(1, len(sampled)), 4)
    return {
        "reviewed_count": len(sampled),
        "passed_count": passed_count,
        "flagged_wrong_artefact_count": flagged_wrong_artefact_count,
        "faithfulness_pass_rate": pass_rate,
        "reviews": reviews,
    }
