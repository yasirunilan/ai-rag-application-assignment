"""Environment & Model Check (Person 1 · Story E0.1).

"Provide a model check that makes one tiny embeddings call, one chat call and one vision call, printing
OK or FAIL with the model name and reason." A fresh clone plus a key passes; a missing key stops with a
clear message; swapping a slug is a configuration change only.

Network calls made (only when a key is present and `offline` is False):
- `GET /api/v1/models` (unauthenticated, free): every configured slug must still be listed, cost $0 and
  accept images. This is what catches a retired `:free` slug before the lab hits it mid-run.
- `GET /api/v1/key`: free-tier status (under 10 credits bought = 50 free calls/day).
- One tiny `POST /api/v1/chat/completions` with a 32x32 image to the first live slug: the vision call,
  which also proves chat completion works.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional

from mmrag.config import AppConfig

OPENROUTER_API = "https://openrouter.ai/api/v1"
KEY_ENV = "OPENROUTER_API_KEY"
TIMEOUT_S = 30.0

Fetch = Callable[[str, Optional[dict[str, Any]], Optional[str]], dict[str, Any]]


@dataclass
class CheckResult:
    name: str
    status: str  # "OK" | "WARN" | "FAIL" | "SKIP"
    detail: str

    def line(self) -> str:
        return f"{self.status:<4}  {self.name:<38} {self.detail}"


def _http_json(url: str, payload: Optional[dict[str, Any]] = None, api_key: Optional[str] = None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "HTTP-Referer": "https://localhost/mmrag-lab-a"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_error_reason(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", "replace")[:160]
        except Exception:  # noqa: BLE001
            body = ""
        return f"HTTP {exc.code} {body}".strip()
    return f"{type(exc).__name__}: {exc}"


def _tiny_png() -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (32, 32), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([6, 6, 25, 25], outline=(0, 0, 0), width=3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def check_embeddings(cfg: AppConfig, offline: bool) -> CheckResult:
    from mmrag.index.embedder import create_embedder

    name = f"embeddings [{cfg.index.embedding_provider}]"
    try:
        emb = create_embedder(cfg, force_offline=offline)
        vec = emb.embed_texts(["torque spec AN4-12A"])
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, "FAIL", f"{cfg.index.embedding_model_name}: {type(exc).__name__}: {exc}")
    dim = int(vec.shape[-1])
    if dim != cfg.index.embedding_dim:
        return CheckResult(name, "FAIL", f"{emb.model_id}: dim {dim} != configured {cfg.index.embedding_dim}")
    wanted_local = cfg.index.embedding_provider == "local_bge" and not offline
    if wanted_local and not emb.model_id.startswith("fastembed:"):
        return CheckResult(
            name, "WARN",
            f"{cfg.index.embedding_model_name} unavailable, fell back to {emb.model_id} "
            "(pip install -e '.[fastembed]')",
        )
    return CheckResult(name, "OK", f"{emb.model_id} dim={dim}")


def check_vector_backend(cfg: AppConfig) -> CheckResult:
    name = f"vector backend [{cfg.index.vector_backend}]"
    if cfg.index.vector_backend == "numpy":
        return CheckResult(name, "OK", "exact NumPy inner product")
    try:
        import faiss  # noqa: F401

        return CheckResult(name, "OK", "faiss-cpu IndexFlatIP")
    except Exception:  # noqa: BLE001
        return CheckResult(name, "WARN", "faiss-cpu not installed; using exact NumPy fallback")


def check_ocr(cfg: AppConfig) -> CheckResult:
    from mmrag.ingest.ocr_gate import tesseract_status

    name = f"ocr engine [{cfg.ingest.ocr_engine}]"
    if cfg.ingest.ocr_engine != "tesseract":
        return CheckResult(name, "WARN", "heuristic fallback: scanned pages are queued for review, never indexed")
    ok, detail = tesseract_status()
    return CheckResult(name, "OK" if ok else "WARN", detail if ok else f"{detail}; scanned pages will be queued")


def check_openrouter(
    cfg: AppConfig,
    api_key: str,
    live_call: bool = True,
    fetch: Fetch = _http_json,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    models = cfg.enrich.openrouter_models

    try:
        listing = fetch(f"{OPENROUTER_API}/models", None, None)
        catalogue = {m.get("id"): m for m in listing.get("data", [])}
    except Exception as exc:  # noqa: BLE001
        results.append(CheckResult("openrouter model catalogue", "FAIL", _http_error_reason(exc)))
        return results

    live: list[str] = []
    for slug in models:
        name = f"vision slug {slug}"
        m = catalogue.get(slug)
        if m is None:
            results.append(CheckResult(name, "FAIL", "not listed by OpenRouter (slug retired?) - replace it in config.yaml"))
            continue
        pricing = m.get("pricing") or {}
        paid = [k for k in ("prompt", "completion", "image", "request") if str(pricing.get(k, "0")) not in ("0", "0.0", "")]
        if paid:
            results.append(CheckResult(name, "FAIL", f"not free: pricing {pricing}"))
            continue
        modalities = (m.get("architecture") or {}).get("input_modalities") or []
        if "image" not in modalities:
            results.append(CheckResult(name, "FAIL", f"does not accept images (input_modalities={modalities})"))
            continue
        live.append(slug)
        results.append(CheckResult(name, "OK", "listed, $0, accepts images"))

    try:
        key_info = fetch(f"{OPENROUTER_API}/key", None, api_key).get("data", {})
        free_tier = key_info.get("is_free_tier")
        note = "under 10 credits bought: 50 free calls/day" if free_tier else "credits bought: 1000 free calls/day"
        results.append(CheckResult("openrouter key", "OK", f"valid; usage={key_info.get('usage')} ({note})"))
    except Exception as exc:  # noqa: BLE001
        results.append(CheckResult("openrouter key", "FAIL", f"key rejected: {_http_error_reason(exc)}"))
        return results

    if not live_call:
        results.append(CheckResult("vision + chat call", "SKIP", "--no-live-call"))
        return results
    if not live:
        results.append(CheckResult("vision + chat call", "FAIL", "no live :free vision slug configured"))
        return results

    b64 = base64.b64encode(_tiny_png()).decode("ascii")
    errors: list[str] = []
    for slug in live:
        payload = {
            "model": slug,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply with one word: what shape is drawn?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            "max_tokens": 16,
            "reasoning": {"enabled": False},
        }
        try:
            body = fetch(f"{OPENROUTER_API}/chat/completions", payload, api_key)
            reply = ((body.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()
            if reply:
                served_by = body.get("model", slug)
                results.append(CheckResult("vision + chat call", "OK", f"{slug} -> {served_by}: {reply[:40]!r}"))
                return results
            errors.append(f"{slug}: empty reply")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{slug}: {_http_error_reason(exc)}")
    results.append(CheckResult("vision + chat call", "FAIL", " | ".join(errors)[:300]))
    return results


def run_environment_check(
    cfg: AppConfig,
    offline: bool = False,
    live_call: bool = True,
    fetch: Fetch = _http_json,
) -> tuple[int, list[CheckResult]]:
    """Returns (exit_code, results). 0 = all OK/WARN, 1 = a FAIL, 2 = API key missing (online mode)."""
    results = [
        CheckResult(
            "config + billing guard", "OK",
            f"{len(cfg.enrich.openrouter_models)} vision slugs, all ':free' (enforce_free_only={cfg.enrich.enforce_free_only})",
        ),
        check_embeddings(cfg, offline=offline),
        check_vector_backend(cfg),
        check_ocr(cfg),
        CheckResult("chat model", "OK", "none required: answers are extractive (real artefact + citation)"),
    ]

    if offline or cfg.enrich.vision_provider != "openrouter_free":
        results.append(CheckResult("openrouter", "SKIP", "offline mode / vision_provider is not openrouter_free"))
        return (1 if any(r.status == "FAIL" for r in results) else 0), results

    api_key = os.environ.get(KEY_ENV, "").strip()
    if not api_key:
        results.append(
            CheckResult(
                KEY_ENV, "FAIL",
                f"not set. Create a free key at https://openrouter.ai/keys, then `export {KEY_ENV}=...` "
                "or put it in a git-ignored .env file. Offline path still works: `python -m mmrag.cli smoke`.",
            )
        )
        return 2, results

    results.extend(check_openrouter(cfg, api_key, live_call=live_call, fetch=fetch))
    return (1 if any(r.status == "FAIL" for r in results) else 0), results
