"""Model & Vector Backend Readiness Checks (Person 1 · Stories E0.1 & E0.2).

- `check_models`: Stops before any network call when `OPENROUTER_API_KEY` is missing; otherwise
  runs one tiny embeddings probe, one chat probe, and one 1x1 PNG vision probe against OpenRouter,
  returning `OK` or `FAIL` with model slug, latency_ms, and reason (e.g. 404 retired slug).
- `check_vector_backend`: Verifies the configured vector backend creates, queries, and resets
  a collection cleanly.
"""

from __future__ import annotations

import base64
import io
import json
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from PIL import Image

from mmrag.config import AppConfig, BillingGuardError, _is_free_slug, get_openrouter_api_key
from mmrag.index.embedder import DeterministicSemanticEmbedder
from mmrag.index.store import MixedVectorStore, _try_import_faiss
from mmrag.schemas import Provenance, Unit


def _tiny_png_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(240, 240, 240)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _post_openrouter_json(
    url: str,
    payload: dict[str, Any],
    api_key: str,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
    timeout: int = 20,
) -> tuple[bool, float, str, dict[str, Any]]:
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/mmrag-lab-a",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return True, ms, "OK", body
    except urllib.error.HTTPError as exc:
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if exc.code == 404:
            return False, ms, "HTTP 404 No endpoints found (slug retired)", {}
        if exc.code == 401:
            return False, ms, "HTTP 401 Unauthorized (invalid OPENROUTER_API_KEY)", {}
        if exc.code == 429:
            return False, ms, "HTTP 429 Rate limited (20 req/min or daily cap)", {}
        return False, ms, f"HTTP {exc.code}", {}
    except Exception as exc:  # noqa: BLE001
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return False, ms, f"{type(exc).__name__}: {exc}", {}


def check_vector_backend(cfg: AppConfig) -> dict[str, Any]:
    """Story E0.2: Confirms the configured vector store creates, queries, and resets cleanly."""
    t0 = time.perf_counter()
    emb = DeterministicSemanticEmbedder(dim=cfg.index.embedding_dim)
    probe_unit = Unit(
        unit_id="probe__p0001__text_001",
        type="text",
        search_text="hydraulic pressure check probe",
        artefact="hydraulic pressure check probe",
        provenance=Provenance(doc_id="probe", page=1, unit_id="probe__p0001__text_001"),
    )
    with tempfile.TemporaryDirectory() as tmp:
        store = MixedVectorStore(cfg, embedder=emb)
        store.build([probe_unit], apply_chunking=False)
        store.save(tmp)
        reloaded = MixedVectorStore.load(cfg, index_dir=tmp, embedder=emb)
        hits = reloaded.search_dense("hydraulic pressure", top_k=1)
        # Reset check
        store.build([], apply_chunking=False)
        store.save(tmp)
        empty = MixedVectorStore.load(cfg, index_dir=tmp, embedder=emb)

    ok = len(hits) == 1 and hits[0][0] == 0 and len(empty.units) == 0
    return {
        "ok": ok,
        "vector_backend": cfg.index.vector_backend,
        "faiss_installed": _try_import_faiss() is not None,
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }


def check_models(
    cfg: AppConfig,
    offline: bool = False,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    """Story E0.1 (Template T1): Verifies configured model slugs or stops clearly on missing key."""
    vec_status = check_vector_backend(cfg)
    if offline:
        return {
            "ok": vec_status["ok"],
            "mode": "offline",
            "vector_store": vec_status,
            "checks": [
                {
                    "role": "embeddings",
                    "slug": f"offline:{cfg.index.embedding_provider}",
                    "ok": True,
                    "latency_ms": vec_status["latency_ms"],
                    "reason": "OK (offline deterministic embedder)",
                },
                {
                    "role": "vision_captioning",
                    "slug": "offline:heuristic_fixture",
                    "ok": True,
                    "latency_ms": 0.0,
                    "reason": "OK (offline captioner)",
                },
            ],
        }

    api_key = get_openrouter_api_key()
    if not api_key:
        return {
            "ok": False,
            "mode": "online",
            "error": (
                "Missing OPENROUTER_API_KEY. Set it in .env or environment before live calls, "
                "or pass --offline for the zero-network path."
            ),
            "vector_store": vec_status,
            "checks": [],
        }

    vision_slug = cfg.enrich.openrouter_models[0]
    embed_slug = cfg.index.openrouter_embed_model
    text_slug = cfg.answer.text_model
    figure_slug = cfg.answer.figure_model

    if cfg.enrich.enforce_free_only:
        for slug in (embed_slug, vision_slug, text_slug, figure_slug):
            if not _is_free_slug(slug):
                raise BillingGuardError(f"Paid slug '{slug}' refused by check_models billing guard.")

    checks: list[dict[str, Any]] = []

    # 1. Tiny embeddings probe
    ok_e, ms_e, reason_e, body_e = _post_openrouter_json(
        "https://openrouter.ai/api/v1/embeddings",
        {"model": embed_slug, "input": ["ok"]},
        api_key,
        urlopen=urlopen,
    )
    if ok_e and not (body_e.get("data") or []):
        ok_e, reason_e = False, "empty embedding vector returned"
    checks.append({"role": "embeddings", "slug": embed_slug, "ok": ok_e, "latency_ms": ms_e, "reason": reason_e})

    # 2. Tiny chat probe
    ok_t, ms_t, reason_t, body_t = _post_openrouter_json(
        "https://openrouter.ai/api/v1/chat/completions",
        {
            "model": text_slug,
            "messages": [{"role": "user", "content": "Reply with OK"}],
            "max_tokens": 16,
            "reasoning": {"enabled": False},
        },
        api_key,
        urlopen=urlopen,
    )
    reply_t = ((body_t.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()
    if ok_t and not reply_t:
        ok_t, reason_t = False, "empty chat reply"
    checks.append({"role": "text_generation", "slug": text_slug, "ok": ok_t, "latency_ms": ms_t, "reason": reason_t})

    # 3. Tiny vision probe
    b64_png = _tiny_png_b64()
    for role_name, v_slug in (("vision_captioning", vision_slug), ("figure_answers", figure_slug)):
        if role_name == "figure_answers" and v_slug == vision_slug:
            checks.append({**checks[-1], "role": "figure_answers"})
            continue
        ok_v, ms_v, reason_v, body_v = _post_openrouter_json(
            "https://openrouter.ai/api/v1/chat/completions",
            {
                "model": v_slug,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Say OK"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_png}"}},
                        ],
                    }
                ],
                "max_tokens": 32,
                "reasoning": {"enabled": False},
            },
            api_key,
            urlopen=urlopen,
        )
        reply_v = ((body_v.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()
        if ok_v and not reply_v:
            ok_v, reason_v = False, "empty vision reply (check reasoning budget)"
        checks.append({"role": role_name, "slug": v_slug, "ok": ok_v, "latency_ms": ms_v, "reason": reason_v})

    all_ok = vec_status["ok"] and all(c["ok"] for c in checks)
    return {
        "ok": all_ok,
        "mode": "online",
        "vector_store": vec_status,
        "checks": checks,
    }
