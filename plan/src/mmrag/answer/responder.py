"""Fail-Closed Artefact Answerer (Person 5 · Stories E6.1 & E6.2, Fig 3).

Decision tree (Fig 3): retrieve -> drop unsourced -> check something relevant was retrieved ->
decide modality from the top sourced hit -> return the real artefact plus its page citation.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import time
from typing import Callable, Optional
import urllib.request

from mmrag.config import AppConfig, BillingGuardError, _is_free_slug, get_openrouter_api_key
from mmrag.enrich.captioner import _downscale_png
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

ABSTAIN_NO_HITS = "no_retrieved_hits"
ABSTAIN_UNSOURCED = "no_sourced_hits"
ABSTAIN_LOW_SCORE = "top_score_below_threshold"
ABSTAIN_NOT_RELEVANT = "no_relevant_hits"

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


def provenance_problem(unit: Unit, verify_files: bool = True) -> Optional[str]:
    """Returns why `unit` cannot be cited, or None if its provenance is valid."""
    if unit.has_valid_provenance(verify_files=verify_files):
        return None
    prov = unit.provenance
    if not prov.doc_id:
        return "missing doc_id"
    if not prov.unit_id or not unit.unit_id:
        return "missing unit_id"
    if prov.unit_id != unit.unit_id:
        return "mismatched unit_id"
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


def _query_term_groups(query: str) -> list[frozenset[str]]:
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


def match_table_rows(query: str, table: TableArtefact) -> list[int]:
    """Indices of the rows that answer the query."""
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
    proper = [s for s in matching if sentences[s[1]].endswith((".", "!", "?"))]
    picked = sorted(proper or matching, key=lambda x: (-x[0], x[1]))[:max_sentences]
    if not picked:
        return [sentences[0]]
    return [sentences[idx] for _, idx in sorted(picked, key=lambda x: x[1])]


def build_grounded_answer(query: str, unit: Unit, grounded_caption: Optional[str]) -> tuple[str, list[int]]:
    """Returns `(grounded_answer, matched_rows)`; quotes the cited unit with `[doc > p.N > unit_id]`."""
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


def _remote_grounded_answer(
    query: str,
    unit: Unit,
    cfg: AppConfig,
    urlopen: Callable[..., object] = urllib.request.urlopen,
) -> Optional[str]:
    """Optional `:free` OpenRouter grounded answer citing `[doc p.page]` inline (Story E6.1)."""
    api_key = get_openrouter_api_key()
    if not api_key or not cfg.answer.enable_remote_generation:
        return None

    slug = cfg.answer.figure_model if unit.type == "figure" else cfg.answer.text_model
    if cfg.answer.enforce_free_only and not _is_free_slug(slug):
        raise BillingGuardError(f"Paid generator slug '{slug}' refused by responder billing guard.")

    cite = unit.provenance.citation
    if unit.type == "figure":
        img_path = Path(str(unit.artefact))
        if not img_path.exists():
            return None
        b64 = base64.b64encode(_downscale_png(img_path.read_bytes())).decode("ascii")
        messages: list[dict] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Answer concisely in 1-2 sentences using ONLY this figure from [{cite}]. "
                            f"Cite [{cite}] inline. Question: {query}"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ]
    else:
        context_body = (
            unit.artefact.to_markdown()
            if isinstance(unit.artefact, TableArtefact)
            else str(unit.artefact)
        )
        messages = [
            {
                "role": "user",
                "content": (
                    f"Answer concisely in 1-2 sentences using ONLY the source below from [{cite}]. "
                    f"Cite [{cite}] inline.\n\nSource:\n{context_body}\n\nQuestion: {query}"
                ),
            }
        ]

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(
            {
                "model": slug,
                "messages": messages,
                "max_tokens": 180,
                "reasoning": {"enabled": False},
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/mmrag-lab-a",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=20) as resp:  # type: ignore[attr-defined]
            body = json.loads(resp.read().decode("utf-8"))
        text = ((body.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()
        if text:
            return text if cite in text else f"{text} [{cite}]"
    except Exception:  # noqa: BLE001
        return None
    return None


def _abstain(query: str, code: str, reason: str, hits: list[SearchHit], t0: float, **extra) -> AnswerResponse:
    gen_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return AnswerResponse(
        query=query,
        abstained=True,
        abstain_code=code,
        abstain_reason=reason,
        top_hits=hits,
        timings_ms={"generation_ms": gen_ms},
        **extra,
    )


def build_answer_from_hits(
    query: str,
    hits: list[SearchHit],
    cfg: AppConfig,
    verify_files: bool = True,
    offline: bool = False,
    urlopen: Callable[..., object] = urllib.request.urlopen,
) -> AnswerResponse:
    """Implements Fig 3: filter unsourced hits -> relevance check -> decide modality -> real artefact."""
    t0 = time.perf_counter()
    if not (query or "").strip():
        return _abstain(
            query,
            ABSTAIN_NOT_RELEVANT,
            "The question has no searchable terms (empty query), so no retrieved unit can be shown to answer it.",
            [],
            t0,
        )

    if not hits:
        return _abstain(
            query,
            ABSTAIN_NO_HITS,
            "Nothing was retrieved for this question, so there is nothing to cite.",
            [],
            t0,
        )

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
            query,
            ABSTAIN_UNSOURCED,
            f"fail-closed: none of the {len(hits)} retrieved units has resolvable provenance "
            f"({'; '.join(reasons)}), so the system refuses to cite anything.",
            hits,
            t0,
            dropped_hits=dropped,
        )

    top = sourced_hits[0]
    if top.score < cfg.answer.min_score_threshold:
        return _abstain(
            query,
            ABSTAIN_LOW_SCORE,
            f"The best sourced match scores {top.score:.3f}, below the {cfg.answer.min_score_threshold} "
            "threshold, so it is not trusted as an answer.",
            sourced_hits,
            t0,
            dropped_hits=dropped,
        )

    coverage = query_term_coverage(query, top.unit)
    min_cov = cfg.answer.min_query_term_coverage
    if min_cov > 0:
        if coverage is None:
            return _abstain(
                query,
                ABSTAIN_NOT_RELEVANT,
                "The question has no searchable terms (only filler words), so no retrieved unit can be "
                "shown to answer it.",
                sourced_hits,
                t0,
                dropped_hits=dropped,
            )
        if coverage < min_cov:
            return _abstain(
                query,
                ABSTAIN_NOT_RELEVANT,
                f"Nothing relevant was retrieved: the best sourced match ({top.unit.provenance.citation}) "
                f"contains only {coverage:.0%} of the question's terms (minimum {min_cov:.0%}). "
                "The manuals indexed here do not appear to cover this question.",
                sourced_hits,
                t0,
                dropped_hits=dropped,
                query_term_coverage=coverage,
            )

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

    extractive_answer, matched_rows = build_grounded_answer(query, u, grounded_caption)
    model_calls = 0
    grounded_answer: Optional[str] = None
    if not offline and cfg.answer.enable_remote_generation and get_openrouter_api_key():
        model_calls = 1
        grounded_answer = _remote_grounded_answer(query, u, cfg, urlopen=urlopen)
    if not grounded_answer:
        grounded_answer = extractive_answer

    gen_ms = round((time.perf_counter() - t0) * 1000.0, 2)
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
        timings_ms={"generation_ms": gen_ms},
        model_calls=model_calls,
    )
