"""Swappable Local Text Embedder (Person 4 · Story E0.1 & E4.1).

Supports:
1. `fastembed` local ONNX model (`BAAI/bge-small-en-v1.5`) when installed and cached.
2. `DeterministicSemanticEmbedder` (384-d L2-normalized) for zero-network offline execution (`smoke`).
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from mmrag.config import AppConfig


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


class EmbedderProtocol(Protocol):
    dim: int

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        ...


class DeterministicSemanticEmbedder:
    """Zero-network 384-d dense embedder combining semantic concept clusters + feature hashing.

    Realistic dense-vector behavior: mirrors how dense models group semantic concepts well
    while smearing exact alphanumeric part/fault codes (`AN4-12A` vs `NAS1304-15` vs `FAULT-E409`)
    into a shared identifier subspace (which Story E5.2's BM25 fusion then disambiguates).
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        lower = text.lower()

        # Normalize exact part/fault codes into a generic semantic code token in dense space
        # (Dense embeddings smear exact alphanumeric codes; BM25 matches them literally)
        has_part_code = bool(PART_CODE_REGEX.search(text))
        cleaned_lower = PART_CODE_REGEX.sub(" generic_part_code ", lower)

        raw_tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", cleaned_lower)
        tokens = [t for t in raw_tokens if t not in STOPWORDS and len(t) > 1]

        # 1. Semantic cluster activations (dims 0..39)
        token_set = set(tokens)
        for c_idx, cluster in enumerate(SEMANTIC_CLUSTERS):
            overlap = len(token_set & cluster)
            if overlap > 0:
                base_dim = (c_idx * 4) % 40
                vec[base_dim : base_dim + 4] += float(1.5 + 0.6 * min(overlap, 4))

        if has_part_code:
            vec[40:44] += 1.25

        # 2. Signed token & stem hashing (dims 48..dim)
        usable_span = self.dim - 48
        for tok in tokens:
            # Word and 4-char prefix stem
            for feat, weight in ((tok, 1.0), (tok[:4], 0.55)):
                digest = hashlib.md5(feat.encode("utf-8")).digest()
                idx = 48 + (int.from_bytes(digest[:4], "little") % usable_span)
                sign = 1.0 if (digest[4] & 1) == 0 else -1.0
                vec[idx] += sign * weight

        # 3. Bigram hashing for phrase specificity
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
        return np.vstack([self._encode_one(t) for t in texts]).astype(np.float32)


class FastEmbedWrapper:
    """Wrapper around `fastembed.TextEmbedding` (e.g. `BAAI/bge-small-en-v1.5`)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dim: int = 384) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = np.vstack(list(self._model.embed(texts))).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        return vecs / norms


def create_embedder(cfg: AppConfig, force_offline: bool = False) -> EmbedderProtocol:
    """Instantiates the configured local embedder with automatic offline fallback."""
    if force_offline or cfg.index.embedding_provider == "deterministic_hash":
        return DeterministicSemanticEmbedder(dim=cfg.index.embedding_dim)

    try:
        return FastEmbedWrapper(
            model_name=cfg.index.embedding_model_name,
            dim=cfg.index.embedding_dim,
        )
    except Exception:
        return DeterministicSemanticEmbedder(dim=cfg.index.embedding_dim)
