"""Fail-Closed Artefact Answerer (Person 5 · Stories E6.1 & E6.2, Fig 3).

Never returns a prose paraphrase in place of a figure or table artefact.
Never presents an unsourced hit (missing `doc_id`, `page < 1`, mismatched `unit_id`, or missing file).
Supports optional `:free` OpenRouter grounded synthesis with deterministic offline fallback.
"""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from mmrag.config import AppConfig, BillingGuardError, _is_free_slug, get_openrouter_api_key
from mmrag.enrich.captioner import _downscale_png
from mmrag.schemas import AnswerResponse, SearchHit, TableArtefact, Unit, UnitType


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


def _deterministic_grounded_answer(query: str, unit: Unit, caption: Optional[str]) -> str:
    """Produces a concise, source-cited grounded statement without replacing the artefact."""
    cite = f"[{unit.provenance.citation}]"
    if unit.type == "figure":
        label = caption or f"Figure {unit.unit_id}"
        return f"See {label} {cite}."
    if unit.type == "table" and isinstance(unit.artefact, TableArtefact):
        q_tokens = {
            tok.lower()
            for tok in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+", query)
        }
        matched_rows = [
            row for row in unit.artefact.rows
            if any(tok in " ".join(row).lower() for tok in q_tokens)
        ]
        target_rows = matched_rows or unit.artefact.rows[:1]
        if target_rows and unit.artefact.headers:
            pairs = ", ".join(
                f"{h}={v}" for h, v in zip(unit.artefact.headers, target_rows[0]) if v
            )
            return f"{pairs} {cite}."
        return f"Structured table ({len(unit.artefact.rows)} rows) {cite}."
    snippet = " ".join(str(unit.artefact).split())
    if len(snippet) > 220:
        snippet = snippet[:217].rstrip() + "..."
    return f"{snippet} {cite}"


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
        messages = [
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


def build_answer_from_hits(
    query: str,
    hits: list[SearchHit],
    cfg: AppConfig,
    verify_files: bool = True,
    offline: bool = False,
    urlopen: Callable[..., object] = urllib.request.urlopen,
) -> AnswerResponse:
    """Implements Fig 3 decision tree: filter unsourced hits -> decide modality -> return real artefact."""
    t0 = time.perf_counter()
    if not (query or "").strip():
        return AnswerResponse(
            query=query,
            abstained=True,
            abstain_reason="empty_query",
            top_hits=[],
            timings_ms={"generation_ms": round((time.perf_counter() - t0) * 1000.0, 2)},
        )

    if not hits:
        return AnswerResponse(
            query=query,
            abstained=True,
            abstain_reason="no_retrieved_hits",
            top_hits=[],
            timings_ms={"generation_ms": round((time.perf_counter() - t0) * 1000.0, 2)},
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
            timings_ms={"generation_ms": round((time.perf_counter() - t0) * 1000.0, 2)},
        )

    chosen = sourced_hits[0]
    if chosen.score < cfg.answer.min_score_threshold:
        return AnswerResponse(
            query=query,
            abstained=True,
            abstain_reason=f"top_sourced_score_below_threshold ({chosen.score:.3f} < {cfg.answer.min_score_threshold})",
            top_hits=sourced_hits,
            timings_ms={"generation_ms": round((time.perf_counter() - t0) * 1000.0, 2)},
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

    model_calls = 0
    grounded_answer: Optional[str] = None
    if not offline and cfg.answer.enable_remote_generation and get_openrouter_api_key():
        model_calls = 1
        grounded_answer = _remote_grounded_answer(query, u, cfg, urlopen=urlopen)
    if not grounded_answer:
        grounded_answer = _deterministic_grounded_answer(query, u, grounded_caption)

    gen_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return AnswerResponse(
        query=query,
        abstained=False,
        abstain_reason=None,
        modality=u.type,
        artefact=u.artefact,
        citation=u.provenance.citation,
        grounded_caption=grounded_caption,
        grounded_answer=grounded_answer,
        source_unit=u,
        top_hits=sourced_hits,
        timings_ms={"generation_ms": gen_ms},
        model_calls=model_calls,
    )
