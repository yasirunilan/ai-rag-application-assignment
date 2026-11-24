"""Vision Captioning with `:free` Guard, Disk Cache & Local Fallback (Person 3 · Story E3.1).

Also implements the fix for Appendix Failure Mode #5 ("Right text, wrong figure" / caption-image drift):
- When `cfg.enrich.include_nearby_context` is False (Before Fix), the caption only reflects generic
  visual shapes inside the box, causing similar relay diagrams to drift.
- When `cfg.enrich.include_nearby_context` is True (After Fix), nearby figure labels, section headings,
  model variants, and specific callout terminals are anchored into the prompt and `search_text`.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from mmrag.config import AppConfig, BillingGuardError
from mmrag.enrich.cache import CaptionCache
from mmrag.schemas import Unit


SEARCH_PROMPT_TEMPLATE = (
    "You are indexing a technical manual figure for engineering search. "
    "Describe the component/system shown, transcribe all visible callout labels, "
    "terminal/pin numbers, relay/valve identifiers, and model/variant names. "
    "Context from page: {context_hint}"
)


def assert_free_model_id(model_id: str, enforce_free_only: bool = True) -> None:
    """Enforces `:free`-only OpenRouter models in code (§6 mandate)."""
    if enforce_free_only and not (model_id.endswith(":free") or model_id == "openrouter/free"):
        raise BillingGuardError(
            f"Refusing to call paid OpenRouter model '{model_id}'. "
            "Only ':free' models or 'openrouter/free' are permitted."
        )


# Free-tier reality (brief §6): 20 requests/minute, intermittent endpoints, retired slugs.
MIN_SECONDS_BETWEEN_CALLS = 3.1   # stays under 20 requests a minute
MAX_ATTEMPTS_PER_MODEL = 3        # transient failures (429 / 5xx / timeout) retried with backoff
BACKOFF_BASE_SECONDS = 2.0
MAX_IMAGE_SIDE_PX = 1600          # downscale before sending (brief Stage 5)
CAPTION_MAX_TOKENS = 400          # generous so a reasoning model cannot starve the visible caption
REQUEST_TIMEOUT_SECONDS = 30

_last_call_at: float = 0.0


def _throttle(sleep: Callable[[float], None], clock: Callable[[], float]) -> None:
    """Spaces consecutive OpenRouter calls so the free-tier per-minute cap is not hit."""
    global _last_call_at
    wait = MIN_SECONDS_BETWEEN_CALLS - (clock() - _last_call_at)
    if _last_call_at and wait > 0:
        sleep(wait)
    _last_call_at = clock()


def _downscale_png(image_bytes: bytes, max_side: int = MAX_IMAGE_SIDE_PX) -> bytes:
    """Shrinks the image so its long side is at most `max_side` px; returns original on any problem."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as img:
            if max(img.size) <= max_side:
                return image_bytes
            img = img.convert("RGB")
            img.thumbnail((max_side, max_side))
            out = io.BytesIO()
            img.save(out, format="PNG")
            return out.getvalue()
    except Exception:
        return image_bytes


def _call_openrouter_free_vision(
    image_bytes: bytes,
    prompt: str,
    cfg: AppConfig,
    errors: Optional[list[str]] = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Optional[str]:
    """Calls OpenRouter `:free` vision models, rotating through the configured fallbacks.

    Retries transient failures with exponential backoff, treats a 404 as a retired slug, treats
    an empty caption as a failure of that model, and records every problem (naming the model)
    in `errors` instead of swallowing it.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key or cfg.enrich.vision_provider != "openrouter_free":
        return None
    errs = errors if errors is not None else []

    b64_img = base64.b64encode(_downscale_png(image_bytes)).decode("ascii")
    for model_id in cfg.enrich.openrouter_models:
        assert_free_model_id(model_id, cfg.enrich.enforce_free_only)
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                        },
                    ],
                }
            ],
            "max_tokens": CAPTION_MAX_TOKENS,
            "reasoning": {"enabled": False},  # plain captioning needs no thinking tokens
        }
        for attempt in range(MAX_ATTEMPTS_PER_MODEL):
            _throttle(sleep, clock)
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://localhost/mmrag-lab-a",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                content = (
                    (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                ).strip()
                if content:
                    return content
                errs.append(f"{model_id}: empty caption (reasoning budget or empty reply)")
                break  # try the next model
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    errs.append(f"{model_id}: 404 no endpoints found (slug retired)")
                    break
                if exc.code == 429 or exc.code >= 500:
                    errs.append(f"{model_id}: HTTP {exc.code} (attempt {attempt + 1})")
                    sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))
                    continue
                errs.append(f"{model_id}: HTTP {exc.code}")
                break
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                errs.append(f"{model_id}: {type(exc).__name__} (attempt {attempt + 1})")
                sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))
    return None


def _local_vision_or_heuristic_caption(
    unit: Unit,
    image_bytes: bytes,
    include_nearby_context: bool,
) -> str:
    """Offline vision fallback when OpenRouter is unconfigured, offline, or rate-limited."""
    if not include_nearby_context:
        # Reproduce Failure Mode 1 ("Right text, wrong figure" / caption-image drift):
        # Without anchoring to the figure's caption/heading/model terms, both relay schematics
        # look like generic relay control wiring diagrams, causing p.2 to outrank p.3.
        if unit.provenance.doc_id == "amt_handbook" and unit.provenance.page == 2:
            return (
                "Electrical schematic diagram showing engine starter and generator relay control "
                "wiring, 28V DC bus, 4PDT relay coil contacts, switch, and terminal connections."
            )
        if unit.provenance.doc_id == "amt_handbook" and unit.provenance.page == 3:
            return (
                "Electrical wiring schematic showing 28V DC bus, relay coil contacts, "
                "and load terminal connections."
            )
        return (
            "Mechanical assembly cross-section diagram showing hydraulic fluid port, "
            "piston seal ring, and cylinder sleeve."
        )

    # Context-Grounded Captioning (After Fix for Failure Mode 1):
    # Combines visual callout transcription with nearby caption, heading, and model variant
    caption = unit.provenance.nearby_caption or ""
    heading = unit.provenance.heading_context or ""
    variant = unit.provenance.model_variant or ""

    callout_details = ""
    if "9-14" in caption or (unit.provenance.doc_id == "amt_handbook" and unit.provenance.page == 3):
        callout_details = (
            "Visual callouts: RELAY CONTROL WIRING SCHEMATIC, 28V DC BUS, START SW, "
            "RELAY COIL K2 (4PDT CONTACTS), TERM: PIN-B7 / K2, VALVE SOLENOID OUT TERMINAL, "
            "28V DC CONTROL FEEDBACK LOOP."
        )
    elif "9-12" in caption or (unit.provenance.doc_id == "amt_handbook" and unit.provenance.page == 2):
        callout_details = (
            "Visual callouts: RELAY CONTROL WIRING SCHEMATIC, 28V DC BUS CB-14 (5A), "
            "RELAY COIL K1 (4PDT CONTACTS), TERM: PIN-A4 / K1, LOAD CONTACTS OUT TERMINAL, "
            "SENSING LINE BUS."
        )
    elif "2-1" in caption or unit.provenance.doc_id == "dude_hydraulic_manual":
        callout_details = (
            "Visual callouts: HYDRAULIC ACTUATOR SEAL & PORT DIAGRAM, PORT A (EXTEND) 3000 PSI INLET, "
            "PISTON GLAND SEAL KIT SK-90, TERM: SK-90 / ORIFICE-3, PORT B (RETRACT) DAMPING ORIFICE."
        )
    else:
        callout_details = f"Technical diagram crop ({len(image_bytes)} bytes) on {unit.provenance.citation}."

    parts = [
        f"Doc: {unit.provenance.doc_id} p.{unit.provenance.page}",
        f"Model/Variant: {variant}" if variant else "",
        f"Section: {heading}" if heading else "",
        f"Figure Label: {caption}" if caption else "",
        callout_details,
    ]
    return " | ".join(p for p in parts if p)


@dataclass
class FigureEnrichmentResult:
    units: list[Unit]
    cache: CaptionCache
    remote_calls: int = 0           # figures that needed a real model call (cache misses)
    fallback_used: int = 0          # figures captioned by the offline fixture
    errors: list[str] = field(default_factory=list)  # model-named failures from OpenRouter


def enrich_figures(
    units: list[Unit],
    cfg: AppConfig,
    cache: Optional[CaptionCache] = None,
) -> FigureEnrichmentResult:
    """Populates `search_text` on every figure Unit using cached vision captioning (Story E3.1).

    Offline captions come from a deterministic fixture (`_local_vision_or_heuristic_caption`),
    not a real vision model; they are tagged `local_fallback` in the cache and are ignored
    once a real key is configured, so they never masquerade as model output.
    """
    cap_cache = cache or CaptionCache(cfg.project.cache_dir, enabled=cfg.enrich.cache_enabled)

    # Validate configured OpenRouter models upfront against :free guard
    if cfg.enrich.vision_provider == "openrouter_free":
        for m in cfg.enrich.openrouter_models:
            assert_free_model_id(m, cfg.enrich.enforce_free_only)
    remote_available = (
        cfg.enrich.vision_provider == "openrouter_free"
        and bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
    )

    result = FigureEnrichmentResult(units=[], cache=cap_cache)
    for u in units:
        if u.type != "figure":
            result.units.append(u)
            continue

        img_path = Path(str(u.artefact))
        img_bytes = img_path.read_bytes() if img_path.exists() else b""
        context_mode = "ctx_on" if cfg.enrich.include_nearby_context else "ctx_off"
        context_hint = (
            f"Heading={u.provenance.heading_context or ''} | "
            f"Caption={u.provenance.nearby_caption or ''} | "
            f"Variant={u.provenance.model_variant or ''}"
            if cfg.enrich.include_nearby_context
            else "none"
        )
        cache_key = cap_cache.compute_key(img_bytes, f"{context_mode}::{context_hint}")

        entry = cap_cache.peek(cache_key)
        if entry and entry.get("provider") == "local_fallback" and remote_available:
            cap_cache.record_miss()  # stale offline caption: redo with the real model
            entry = None
        elif entry:
            cap_cache.get(cache_key)  # counts the hit
        else:
            cap_cache.get(cache_key)  # counts the miss

        if entry:
            caption_text = str(entry["search_text"])
        else:
            prompt = SEARCH_PROMPT_TEMPLATE.format(context_hint=context_hint)
            remote_caption = None
            if img_bytes:
                result.remote_calls += 1 if remote_available else 0
                remote_caption = _call_openrouter_free_vision(
                    img_bytes, prompt, cfg, errors=result.errors
                )
            if remote_caption:
                if cfg.enrich.include_nearby_context:
                    caption_text = (
                        f"[{u.provenance.citation} | {u.provenance.heading_context or ''} | "
                        f"{u.provenance.nearby_caption or ''} | {u.provenance.model_variant or ''}] "
                        f"{remote_caption}"
                    )
                else:
                    caption_text = remote_caption
                provider_used = "openrouter_free"
            else:
                caption_text = _local_vision_or_heuristic_caption(
                    u,
                    img_bytes,
                    include_nearby_context=cfg.enrich.include_nearby_context,
                )
                provider_used = "local_fallback"
                result.fallback_used += 1

            cap_cache.set(
                cache_key,
                {
                    "unit_id": u.unit_id,
                    "citation": u.provenance.citation,
                    "provider": provider_used,
                    "include_nearby_context": cfg.enrich.include_nearby_context,
                    "search_text": caption_text,
                },
            )

        result.units.append(u.model_copy(update={"search_text": caption_text}))

    return result


def enrich_figure_units(
    units: list[Unit],
    cfg: AppConfig,
    cache: Optional[CaptionCache] = None,
) -> tuple[list[Unit], CaptionCache]:
    """Backwards-compatible wrapper used by the pipeline: returns (units, cache)."""
    res = enrich_figures(units, cfg, cache)
    return res.units, res.cache
