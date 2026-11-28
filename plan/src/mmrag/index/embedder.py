"""Swappable Text Embedder with OpenRouter `:free` + Disk Cache + Offline Fallback (Person 4 · Stories E0.1, E4.1, E4.2 & E8.7).

Supports:
1. `OpenRouterEmbedder` (`nvidia/nemotron-3.5-embed-1b:free`, `:free` billing guard, batch-16, SHA-256 disk cache).
2. `FastEmbedWrapper` local ONNX model (`BAAI/bge-small-en-v1.5`) when installed and cached.
3. `DeterministicSemanticEmbedder` (384-d L2-normalized) for zero-network offline execution (`smoke`).

Every embedder exposes a stable `model_id`. The index records it at build time and refuses
to be queried by a different embedder (Story E4.2: "vectors are never mixed").
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import re
import time
from typing import Callable, Iterator, Optional, Protocol, Sequence
import urllib.error
import urllib.request

import numpy as np

from mmrag.config import AppConfig, BillingGuardError, _is_free_slug, get_openrouter_api_key


STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "with", "by", "at", "from", "that", "this", "what", "how", "when", "does",
    "show", "me", "lookup",
}

# Semantic concept synonym buckets to mirror dense embedding clustering offline
SEMANTIC_CLUSTERS: list[set[str]] = [
    {"wiring", "schematic", "diagram", "circuit", "circuitry", "electrical", "pinout", "pinouts"},
    {"relay", "coil", "contacts", "solenoid", "switch", "interlock", "energize"},
    {"starter", "starting", "start", "ignition", "gnd", "n2", "pneumatic", "shutoff"},
    {"generator", "idg", "gcu", "exciter", "overvoltage", "ac", "bus", "tie"},
    {"torque", "tightened", "wrench", "dry", "lubed", "in-lbs", "fastener", "fasteners", "bolt", "nuts"},
    {"hydraulic", "pressure", "psi", "manifold", "reservoir", "fluid", "skydrol", "pump", "ptu"},
    {"accumulator", "nitrogen", "pre-charge", "depressurize", "celsius", "ambient"},
    {"actuator", "retraction", "landing", "gear", "seal", "gland", "piston", "orifice", "damping"},
    {"fault", "error", "troubleshooting", "corrective", "action", "trigger", "condition", "code", "codes"},
    {"table", "specifications", "spec", "specs", "rows", "columns", "limits"},
]

PART_CODE_REGEX = re.compile(r"\b(?:AN|MS|NAS|FAULT|PIN|CB|SK)[A-Z0-9-]{2,}\b", re.IGNORECASE)

# Story E4.1: embed searchable texts in batches of 16
EMBED_BATCH_SIZE = 16

logger = logging.getLogger(__name__)


class EmbedderProtocol(Protocol):
    dim: int
    model_id: str

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        ...


def iter_batches(texts: Sequence[str], batch_size: int = EMBED_BATCH_SIZE) -> Iterator[list[str]]:
    """Yields consecutive slices of `texts` of at most `batch_size` items."""
    size = max(1, int(batch_size))
    for start in range(0, len(texts), size):
        yield list(texts[start : start + size])


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    return (vecs / norms).astype(np.float32)


class DeterministicSemanticEmbedder:
    """Zero-network 384-d dense embedder combining semantic concept clusters + feature hashing."""

    MODEL_FAMILY = "deterministic-semantic-hash-v1"

    def __init__(self, dim: int = 384, batch_size: int = EMBED_BATCH_SIZE) -> None:
        if dim <= 48:
            raise ValueError(f"DeterministicSemanticEmbedder needs dim > 48 (got {dim})")
        self.dim = dim
        self.batch_size = batch_size
        self.model_id = f"{self.MODEL_FAMILY}-{dim}d"

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        lower = text.lower()

        has_part_code = bool(PART_CODE_REGEX.search(text))
        cleaned_lower = PART_CODE_REGEX.sub(" generic_part_code ", lower)

        raw_tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", cleaned_lower)
        tokens = [t for t in raw_tokens if t not in STOPWORDS and len(t) > 1]

        token_set = set(tokens)
        for c_idx, cluster in enumerate(SEMANTIC_CLUSTERS):
            overlap = len(token_set & cluster)
            if overlap > 0:
                base_dim = (c_idx * 4) % 40
                vec[base_dim : base_dim + 4] += float(1.5 + 0.6 * min(overlap, 4))

        if has_part_code:
            vec[40:44] += 1.25

        usable_span = self.dim - 48
        for tok in tokens:
            for feat, weight in ((tok, 1.0), (tok[:4], 0.55)):
                digest = hashlib.md5(feat.encode("utf-8")).digest()
                idx = 48 + (int.from_bytes(digest[:4], "little") % usable_span)
                sign = 1.0 if (digest[4] & 1) == 0 else -1.0
                vec[idx] += sign * weight

        for i in range(len(tokens) - 1):
            bg = f"{tokens[i]}_{tokens[i+1]}"
            digest = hashlib.md5(bg.encode("utf-8")).digest()
            idx = 48 + (int.from_bytes(digest[:4], "little") % usable_span)
            sign = 1.0 if (digest[5] & 1) == 0 else -1.0
            vec[idx] += sign * 0.85

        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec /= norm
        else:
            vec[0] = 1.0
        return vec

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[np.ndarray] = []
        for batch in iter_batches(texts, self.batch_size):
            out.extend(self._encode_one(t or "") for t in batch)
        return np.vstack(out).astype(np.float32)


class FastEmbedWrapper:
    """Wrapper around `fastembed.TextEmbedding` (e.g. `BAAI/bge-small-en-v1.5`)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dim: int = 384,
        batch_size: int = EMBED_BATCH_SIZE,
    ) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self.model_name = model_name
        self.batch_size = batch_size
        self.model_id = f"fastembed:{model_name}"
        probe = np.asarray(next(iter(self._model.embed(["dimension probe"]))), dtype=np.float32)
        if probe.shape[-1] != dim:
            logger.warning(
                "Configured embedding_dim=%d but %s produces %d-d vectors; using %d.",
                dim, model_name, probe.shape[-1], probe.shape[-1],
            )
        self.dim = int(probe.shape[-1])

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        safe = [t if t and t.strip() else " " for t in texts]
        vecs = np.vstack(list(self._model.embed(safe, batch_size=self.batch_size))).astype(np.float32)
        return _l2_normalize(vecs)


class OpenRouterEmbedder:
    """OpenRouter `:free` embedder with batch-16 requests and SHA-256 disk cache (Stories E4.1 & E8.7)."""

    def __init__(
        self,
        model_name: str = "nvidia/nemotron-3.5-embed-1b:free",
        dim: int = 384,
        cache_dir: Optional[str | Path] = None,
        api_key: Optional[str] = None,
        batch_size: int = EMBED_BATCH_SIZE,
        urlopen: Callable[..., object] = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not _is_free_slug(model_name):
            raise BillingGuardError(f"Paid embedding model '{model_name}' refused by §6 billing guard.")
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self.model_id = f"openrouter:{model_name}:{dim}d"
        self.api_key = (api_key if api_key is not None else get_openrouter_api_key()).strip()
        if not self.api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY for OpenRouterEmbedder.")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._urlopen = urlopen
        self._sleep = sleep
        self.remote_calls = 0
        self.cache_hits = 0

    def _cache_path(self, text: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(f"{self.model_id}::{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _fit_dim(self, raw: Sequence[float]) -> np.ndarray:
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
        if arr.size >= self.dim:
            return arr[: self.dim]
        out = np.zeros(self.dim, dtype=np.float32)
        out[: arr.size] = arr
        return out

    def _fetch_batch(self, batch: list[str]) -> list[np.ndarray]:
        payload = json.dumps({"model": self.model_name, "input": batch}).encode("utf-8")
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/embeddings",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://localhost/mmrag-lab-a",
                },
                method="POST",
            )
            try:
                self.remote_calls += 1
                with self._urlopen(req, timeout=30) as resp:  # type: ignore[attr-defined]
                    body = json.loads(resp.read().decode("utf-8"))
                rows = sorted(body.get("data") or [], key=lambda r: int(r.get("index", 0)))
                if len(rows) != len(batch):
                    raise RuntimeError(f"{self.model_name}: expected {len(batch)} embeddings, got {len(rows)}")
                return [self._fit_dim(r["embedding"]) for r in rows]
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code == 404:
                    raise RuntimeError(f"{self.model_name}: 404 slug retired on OpenRouter") from exc
                if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                    self._sleep(1.5 * (2**attempt))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_exc = exc
                if attempt < 2:
                    self._sleep(1.5 * (2**attempt))
                    continue
                raise
        raise RuntimeError(f"{self.model_name}: embedding request failed ({last_exc})")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        safe = [t if t and t.strip() else " " for t in texts]
        vectors: list[Optional[np.ndarray]] = [None] * len(safe)
        miss_indices: list[int] = []

        for idx, txt in enumerate(safe):
            cp = self._cache_path(txt)
            if cp and cp.exists():
                try:
                    vectors[idx] = self._fit_dim(json.loads(cp.read_text(encoding="utf-8"))["vec"])
                    self.cache_hits += 1
                    continue
                except Exception:  # noqa: BLE001
                    pass
            miss_indices.append(idx)

        for batch_idxs in iter_batches(miss_indices, self.batch_size):  # type: ignore[arg-type]
            batch_texts = [safe[i] for i in batch_idxs]
            fetched = self._fetch_batch(batch_texts)
            for idx, vec in zip(batch_idxs, fetched):
                vectors[idx] = vec
                cp = self._cache_path(safe[idx])
                if cp:
                    cp.write_text(json.dumps({"model_id": self.model_id, "vec": vec.tolist()}), encoding="utf-8")

        stacked = np.vstack([v for v in vectors if v is not None]).astype(np.float32)
        return _l2_normalize(stacked)


def create_embedder(cfg: AppConfig, force_offline: bool = False) -> EmbedderProtocol:
    """Instantiates the configured embedder with automatic offline fallback."""
    if force_offline or cfg.index.embedding_provider == "deterministic_hash":
        return DeterministicSemanticEmbedder(dim=cfg.index.embedding_dim)

    if cfg.index.embedding_provider == "openrouter_free":
        try:
            return OpenRouterEmbedder(
                model_name=cfg.index.openrouter_embed_model,
                dim=cfg.index.embedding_dim,
                cache_dir=cfg.project.embedding_cache_dir,
            )
        except BillingGuardError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "OpenRouter embedder %r unavailable (%s: %s); falling back to DeterministicSemanticEmbedder.",
                cfg.index.openrouter_embed_model, type(exc).__name__, exc,
            )
            return DeterministicSemanticEmbedder(dim=cfg.index.embedding_dim)

    try:
        return FastEmbedWrapper(
            model_name=cfg.index.embedding_model_name,
            dim=cfg.index.embedding_dim,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "fastembed model %r unavailable (%s: %s); falling back to the offline "
            "DeterministicSemanticEmbedder. Install with: pip install -e '.[fastembed]'",
            cfg.index.embedding_model_name, type(exc).__name__, exc,
        )
        return DeterministicSemanticEmbedder(dim=cfg.index.embedding_dim)
