"""Fail-Closed Artefact Answerer (Person 5 · Stories E6.1 & E6.2, Fig 3).

Decision tree (Fig 3): retrieve -> drop unsourced -> check something relevant was retrieved ->
decide modality from the top sourced hit -> return the real artefact plus its page citation.

- Never returns a prose paraphrase of a figure or table: figures return the image path, tables
  the structured `TableArtefact` (plus the indices of the rows that answer the question).
- The `grounded_answer` is extractive: it quotes the cited unit verbatim (sentences, table rows
  or the figure caption) with the `doc > page > unit` citation inline, so every claim traces to
  the cited source.
- Never presents an unsourced hit (missing `doc_id`, `page < 1`, empty artefact or a figure image
  missing on disk). When nothing sourced and relevant remains it abstains and says why.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from mmrag.config import AppConfig
from mmrag.retrieve.retriever import BM25_STOPWORDS, CODE_TOKEN_PATTERN, tokenize_technical_text
from mmrag.schemas import AnswerResponse, SearchHit, TableArtefact, Unit, UnitType


FIGURE_INTENT_PATTERN = re.compile(
    r"\b(diagram|schematic|figure|fig\.|drawing|show\s+the|wiring\s+diagram)\b",
    re.IGNORECASE,
)
TABLE_INTENT_PATTERN = re.compile(
    r"\b(table|torque\s+spec|specifications\s+table|troubleshooting\s+table|rows)\b",
    re.IGNORECASE,
)

# Stable abstention codes (machine-readable; `abstain_reason` carries the human sentence).
ABSTAIN_NO_HITS = "no_retrieved_hits"
ABSTAIN_UNSOURCED = "no_sourced_hits"
ABSTAIN_LOW_SCORE = "top_score_below_threshold"
ABSTAIN_NOT_RELEVANT = "no_relevant_hits"

# Words that say *how* to answer ("show the table") rather than *what* about. They are ignored by
# the relevance gate, otherwise "show me the table about cake" would match any table's "Table" label.
_INTENT_AND_FILLER_WORDS = frozenset(
    {
        "about", "all", "any", "been", "being", "could", "describe", "diagram", "diagrams",
        "drawing", "entry", "explain", "fig", "figure", "figures", "find", "give", "has", "have",
        "if", "into", "its", "know", "list", "lookup", "look", "many", "much", "my", "need", "no",
        "not", "our", "schematic", "should", "showing", "table", "tables", "than", "then", "there",
        "these", "those", "up", "us", "want", "was", "we", "were", "who", "why", "will", "would",
        "you", "your",
    }
)
_CODE_SEPARATORS = re.compile(r"[-./]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_PROMOTION_MAX_SCORE_GAP = 0.15


def detect_explicit_query_modality(query: str) -> Optional[UnitType]:
    if FIGURE_INTENT_PATTERN.search(query):
        return "figure"
    if TABLE_INTENT_PATTERN.search(query):
        return "table"
    return None


# ---------------------------------------------------------------------------------------------
# Provenance (E6.2)
# ---------------------------------------------------------------------------------------------
def provenance_problem(unit: Unit, verify_files: bool = True) -> Optional[str]:
    """Returns why `unit` cannot be cited, or None if its provenance is valid.

    `Unit.has_valid_provenance` stays the single source of truth; this only explains a failure.
    """
    if unit.has_valid_provenance(verify_files=verify_files):
        return None
    prov = unit.provenance
    if not prov.doc_id:
        return "missing doc_id"
    if not prov.unit_id:
        return "missing unit_id"
    if prov.page is None:
        return "missing page"
    if int(prov.page) < 1:
        return f"invalid page {prov.page}"
    if unit.type == "figure":
        if not isinstance(unit.artefact, str) or not unit.artefact.strip():
            return "figure has no image path"
        return f"figure image missing on disk ({Path(unit.artefact).name})"
    return f"empty {unit.type} artefact"


def citation_path(unit: Unit) -> str:
    """`doc_id > p.N > unit_id` — the full provenance chain shown on every answer (E7.1 UI)."""
    prov = unit.provenance
    return f"{prov.doc_id} > p.{prov.page} > {prov.unit_id}"


# ---------------------------------------------------------------------------------------------
# Query-term relevance (E6.2: "nothing relevant was retrieved")
# ---------------------------------------------------------------------------------------------
def _query_term_groups(query: str) -> list[frozenset[str]]:
    """One group per distinct query term; a code like `AN4-12A` also matches its `an412a` form."""
    groups: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()
    for tok in CODE_TOKEN_PATTERN.findall((query or "").lower()):
        if _CODE_SEPARATORS.search(tok) and any(c.isdigit() for c in tok):
            group = frozenset({tok, _CODE_SEPARATORS.sub("", tok)})
        elif tok in BM25_STOPWORDS or tok in _INTENT_AND_FILLER_WORDS:
            continue
        else:
            group = frozenset({tok})
        if group not in seen:
            seen.add(group)
            groups.append(group)
    return groups


def _is_code_group(group: frozenset[str]) -> bool:
    return any(_CODE_SEPARATORS.search(t) for t in group)


def _unit_content_text(unit: Unit) -> str:
    prov = unit.provenance
    parts = [unit.search_text, prov.nearby_caption or "", prov.heading_context or "", prov.model_variant or ""]
    if isinstance(unit.artefact, TableArtefact):
        parts.append(unit.artefact.caption or "")
        parts.append(" ".join(unit.artefact.headers))
        parts.extend(" ".join(r) for r in unit.artefact.rows)
    elif unit.type == "text":
        parts.append(str(unit.artefact))
    return " ".join(parts)


def query_term_coverage(query: str, unit: Unit) -> Optional[float]:
    """Share of the query's content terms that occur in the unit. None when the query has none."""
    groups = _query_term_groups(query)
    if not groups:
        return None
    unit_tokens = set(tokenize_technical_text(_unit_content_text(unit), expand_code_parts=True))
    matched = sum(1 for g in groups if g & unit_tokens)
    return round(matched / len(groups), 4)


# ---------------------------------------------------------------------------------------------
# Extractive grounded answers (E6.1)
# ---------------------------------------------------------------------------------------------
def match_table_rows(query: str, table: TableArtefact) -> list[int]:
    """Indices of the rows that answer the query.

    An exact identifier in the query (`AN4-12A`, `FAULT-E409`) selects the rows containing it.
    Otherwise the rows sharing the most query terms win, unless every row ties (then the whole
    table is the answer and no row is singled out).
    """
    groups = _query_term_groups(query)
    if not groups or not table.rows:
        return []
    row_tokens = [set(tokenize_technical_text(" ".join(r), expand_code_parts=False)) for r in table.rows]

    code_groups = [g for g in groups if _is_code_group(g)]
    if code_groups:
        hits = [i for i, toks in enumerate(row_tokens) if any(g & toks for g in code_groups)]
        if hits:
            return hits

    scores = [sum(1 for g in groups if g & toks) for toks in row_tokens]
    best = max(scores)
    if best == 0 or all(s == best for s in scores):
        return []
    return [i for i, s in enumerate(scores) if s == best]


def _best_sentences(query: str, text: str, max_sentences: int = 2) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s and s.strip()]
    if not sentences:
        return []
    groups = _query_term_groups(query)
    scored = []
    for idx, sent in enumerate(sentences):
        toks = set(tokenize_technical_text(sent, expand_code_parts=True))
        scored.append((sum(1 for g in groups if g & toks), idx))
    matching = [s for s in scored if s[0] > 0]
    # Prefer real sentences over heading lines (which carry many title words but no facts)
    proper = [s for s in matching if sentences[s[1]].endswith((".", "!", "?"))]
    picked = sorted(proper or matching, key=lambda x: (-x[0], x[1]))[:max_sentences]
    if not picked:
        return [sentences[0]]
    return [sentences[idx] for _, idx in sorted(picked, key=lambda x: x[1])]


def build_grounded_answer(query: str, unit: Unit, grounded_caption: Optional[str]) -> tuple[str, list[int]]:
    """Returns `(grounded_answer, matched_rows)`; the answer only quotes the cited unit."""
    cite = f"[{citation_path(unit)}]"
    if unit.type == "table" and isinstance(unit.artefact, TableArtefact):
        table = unit.artefact
        label = table.caption or grounded_caption or "Table"
        rows = match_table_rows(query, table)
        if not rows:
            return f"{label}: {len(table.rows)} rows x {len(table.headers)} columns, returned as structured rows {cite}", []
        rendered = []
        for i in rows:
            cells = list(table.rows[i]) + [""] * max(0, len(table.headers) - len(table.rows[i]))
            rendered.append("; ".join(f"{h} = {c}" for h, c in zip(table.headers, cells)))
        return f"{label} — " + " | ".join(rendered) + f" {cite}", rows
    if unit.type == "figure":
        return f"{grounded_caption or 'Figure'} (figure image returned) {cite}", []
    sentences = _best_sentences(query, str(unit.artefact))
    return " ".join(sentences) + f" {cite}", []


# ---------------------------------------------------------------------------------------------
# Fig 3 decision tree
# ---------------------------------------------------------------------------------------------
def _abstain(query: str, code: str, reason: str, hits: list[SearchHit], **extra) -> AnswerResponse:
    return AnswerResponse(query=query, abstained=True, abstain_code=code, abstain_reason=reason, top_hits=hits, **extra)


def build_answer_from_hits(
    query: str,
    hits: list[SearchHit],
    cfg: AppConfig,
    verify_files: bool = True,
) -> AnswerResponse:
    """Implements Fig 3: filter unsourced hits -> relevance check -> decide modality -> real artefact."""
    if not hits:
        return _abstain(
            query, ABSTAIN_NO_HITS,
            "Nothing was retrieved for this question, so there is nothing to cite.", [],
        )

    # Story E6.2: drop any retrieved unit whose provenance can't name a page or a valid artefact
    dropped: list[dict] = []
    if cfg.answer.fail_closed_on_missing_provenance:
        sourced_hits = []
        for h in hits:
            problem = provenance_problem(h.unit, verify_files=verify_files)
            if problem is None:
                sourced_hits.append(h)
            else:
                dropped.append({"unit_id": h.unit.unit_id, "rank": h.rank, "reason": problem})
    else:
        sourced_hits = list(hits)

    if not sourced_hits:
        reasons = sorted({d["reason"] for d in dropped})
        return _abstain(
            query, ABSTAIN_UNSOURCED,
            f"fail-closed: none of the {len(hits)} retrieved units has resolvable provenance "
            f"({'; '.join(reasons)}), so the system refuses to cite anything.",
            hits, dropped_hits=dropped,
        )

    top = sourced_hits[0]
    if top.score < cfg.answer.min_score_threshold:
        return _abstain(
            query, ABSTAIN_LOW_SCORE,
            f"The best sourced match scores {top.score:.3f}, below the {cfg.answer.min_score_threshold} "
            "threshold, so it is not trusted as an answer.",
            sourced_hits, dropped_hits=dropped,
        )

    # Story E6.2: fused scores are min-max normalized, so the top hit always looks confident.
    # Require that the cited unit actually talks about the question's terms.
    coverage = query_term_coverage(query, top.unit)
    min_cov = cfg.answer.min_query_term_coverage
    if min_cov > 0:
        if coverage is None:
            return _abstain(
                query, ABSTAIN_NOT_RELEVANT,
                "The question has no searchable terms (only filler words), so no retrieved unit can be "
                "shown to answer it.",
                sourced_hits, dropped_hits=dropped,
            )
        if coverage < min_cov:
            return _abstain(
                query, ABSTAIN_NOT_RELEVANT,
                f"Nothing relevant was retrieved: the best sourced match ({top.unit.provenance.citation}) "
                f"contains only {coverage:.0%} of the question's terms (minimum {min_cov:.0%}). "
                "The manuals indexed here do not appear to cover this question.",
                sourced_hits, dropped_hits=dropped, query_term_coverage=coverage,
            )

    # The top sourced hit decides the modality. When the question explicitly asks for a diagram or
    # table, promote a figure/table from the *same page* only: a different page is a different
    # artefact, and swapping one in is how "right text, wrong figure" happens silently.
    chosen = top
    if cfg.answer.promote_explicit_modality_intent:
        intent_mod = detect_explicit_query_modality(query)
        if intent_mod and chosen.unit.type != intent_mod:
            for candidate in sourced_hits[:3]:
                same_page = (
                    candidate.unit.provenance.doc_id == top.unit.provenance.doc_id
                    and candidate.unit.provenance.page == top.unit.provenance.page
                )
                if (
                    candidate.unit.type == intent_mod
                    and same_page
                    and (top.score - candidate.score) <= _PROMOTION_MAX_SCORE_GAP
                ):
                    chosen = candidate
                    break

    u = chosen.unit
    grounded_caption: Optional[str] = None
    if u.type in ("figure", "table"):
        label = "Figure" if u.type == "figure" else "Table"
        grounded_caption = (
            u.provenance.nearby_caption
            or u.provenance.heading_context
            or f"{label} on {u.provenance.citation}"
        )
    grounded_answer, matched_rows = build_grounded_answer(query, u, grounded_caption)

    return AnswerResponse(
        query=query,
        abstained=False,
        abstain_reason=None,
        modality=u.type,
        artefact=u.artefact,
        citation=u.provenance.citation,
        citation_path=citation_path(u),
        grounded_caption=grounded_caption,
        grounded_answer=grounded_answer,
        matched_rows=matched_rows,
        query_term_coverage=coverage,
        source_unit=u,
        top_hits=sourced_hits,
        dropped_hits=dropped,
    )
