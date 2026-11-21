"""Fail-Closed Artefact Answerer (Person 5 · Stories E6.1 & E6.2, Fig 3).

Never returns a prose paraphrase of a figure or table.
Never presents an unsourced hit (missing `doc_id` or `page < 1` or missing artefact file).
"""

from __future__ import annotations

import re
from typing import Optional

from mmrag.config import AppConfig
from mmrag.schemas import AnswerResponse, SearchHit, UnitType


FIGURE_INTENT_PATTERN = re.compile(
    r"\b(diagram|schematic|figure|fig\.|drawing|show\s+the|wiring\s+diagram)\b",
    re.IGNORECASE,
)
TABLE_INTENT_PATTERN = re.compile(
    r"\b(table|torque\s+spec|specifications\s+table|troubleshooting\s+table|rows)\b",
    re.IGNORECASE,
)


def detect_explicit_query_modality(query: str) -> Optional[UnitType]:
    if FIGURE_INTENT_PATTERN.search(query):
        return "figure"
    if TABLE_INTENT_PATTERN.search(query):
        return "table"
    return None


def build_answer_from_hits(
    query: str,
    hits: list[SearchHit],
    cfg: AppConfig,
    verify_files: bool = True,
) -> AnswerResponse:
    """Implements Fig 3 decision tree: filter unsourced hits -> decide modality -> return real artefact."""
    if not hits:
        return AnswerResponse(
            query=query,
            abstained=True,
            abstain_reason="no_retrieved_hits",
            top_hits=[],
        )

    # Story E6.2: Filter out any retrieved unit whose provenance can't name a page or valid artefact
    if cfg.answer.fail_closed_on_missing_provenance:
        sourced_hits = [h for h in hits if h.unit.has_valid_provenance(verify_files=verify_files)]
    else:
        sourced_hits = list(hits)

    if not sourced_hits:
        return AnswerResponse(
            query=query,
            abstained=True,
            abstain_reason="no_sourced_hits_with_valid_page_provenance (fail-closed)",
            top_hits=hits,
        )

    chosen = sourced_hits[0]
    if chosen.score < cfg.answer.min_score_threshold:
        return AnswerResponse(
            query=query,
            abstained=True,
            abstain_reason=f"top_sourced_score_below_threshold ({chosen.score:.3f} < {cfg.answer.min_score_threshold})",
            top_hits=sourced_hits,
        )

    # Optional modality promotion when query explicitly requests a diagram/table
    if cfg.answer.promote_explicit_modality_intent:
        intent_mod = detect_explicit_query_modality(query)
        if intent_mod and chosen.unit.type != intent_mod:
            for candidate in sourced_hits[:3]:
                if candidate.unit.type == intent_mod and (chosen.score - candidate.score) <= 0.15:
                    chosen = candidate
                    break

    u = chosen.unit
    grounded_caption: Optional[str] = None
    if u.type == "figure":
        grounded_caption = (
            u.provenance.nearby_caption
            or u.provenance.heading_context
            or f"Figure on {u.provenance.citation}"
        )
    elif u.type == "table":
        grounded_caption = (
            u.provenance.nearby_caption
            or u.provenance.heading_context
            or f"Table on {u.provenance.citation}"
        )

    return AnswerResponse(
        query=query,
        abstained=False,
        abstain_reason=None,
        modality=u.type,
        artefact=u.artefact,
        citation=u.provenance.citation,
        grounded_caption=grounded_caption,
        source_unit=u,
        top_hits=sourced_hits,
    )
