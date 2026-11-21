"""Hybrid Vector + BM25 Lexical Retriever (Person 4 · Stories E5.1 & E5.2).

Implements:
- `retrieve(query, k)` returning top-k `SearchHit` objects with provenance and human-readable
  citations (`doc p.212`).
- Technical-identifier-aware BM25 lexical scoring over `search_text` fused via min-max
  normalization and tunable `alpha` (`score = alpha * vec_norm + (1 - alpha) * bm25_norm`).
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from mmrag.config import AppConfig
from mmrag.index.store import MixedVectorStore
from mmrag.schemas import SearchHit


CODE_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*", re.IGNORECASE)


def tokenize_technical_text(text: str) -> list[str]:
    """Tokenizes text while preserving hyphenated part/fault numbers (e.g. `an4-12a`, `fault-e409`)."""
    lower = text.lower()
    raw = CODE_TOKEN_PATTERN.findall(lower)
    tokens: list[str] = []
    for tok in raw:
        tokens.append(tok)
        # If token is a compound code like 'an4-12a', repeat exact code token to boost exact match weight
        if "-" in tok and any(c.isdigit() for c in tok):
            tokens.append(tok)
    return tokens or [""]


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    s_min = float(np.min(scores))
    s_max = float(np.max(scores))
    if s_max - s_min < 1e-8:
        return np.zeros_like(scores, dtype=np.float32) if s_max <= 0.0 else np.ones_like(scores, dtype=np.float32)
    return ((scores - s_min) / (s_max - s_min)).astype(np.float32)


class HybridRetriever:
    """Combines dense vector search with BM25 lexical fusion and provenance citations."""

    def __init__(self, store: MixedVectorStore, cfg: AppConfig) -> None:
        self.store = store
        self.cfg = cfg
        corpus_tokens = [
            tokenize_technical_text(u.search_text or str(u.artefact))
            for u in self.store.units
        ]
        self._bm25: Optional[BM25Okapi] = BM25Okapi(corpus_tokens) if corpus_tokens else None

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        alpha: Optional[float] = None,
        enable_bm25_fusion: Optional[bool] = None,
    ) -> list[SearchHit]:
        """Returns top-k `SearchHit` results with citations (`doc p.212`) and score breakdowns."""
        n_units = len(self.store.units)
        if n_units == 0:
            return []

        top_k = min(k or self.cfg.retrieve.top_k, n_units)
        use_bm25 = (
            self.cfg.retrieve.enable_bm25_fusion
            if enable_bm25_fusion is None
            else enable_bm25_fusion
        )
        weight_alpha = self.cfg.retrieve.alpha if alpha is None else alpha
        if not use_bm25:
            weight_alpha = 1.0

        # 1. Dense scores across candidate pool
        dense_pairs = self.store.search_dense(query, top_k=n_units)
        dense_raw = np.zeros(n_units, dtype=np.float32)
        for idx, s in dense_pairs:
            dense_raw[idx] = float(s)
        dense_norm = _min_max_normalize(dense_raw)

        # 2. BM25 lexical scores across all units
        if self._bm25 is not None and use_bm25 and weight_alpha < 1.0:
            q_tokens = tokenize_technical_text(query)
            bm25_raw = np.asarray(self._bm25.get_scores(q_tokens), dtype=np.float32)
            bm25_norm = _min_max_normalize(bm25_raw)
        else:
            bm25_norm = np.zeros(n_units, dtype=np.float32)

        # 3. Linear fusion: alpha * dense + (1 - alpha) * bm25
        fused = (weight_alpha * dense_norm) + ((1.0 - weight_alpha) * bm25_norm)
        order = np.argsort(-fused)[:top_k]

        hits: list[SearchHit] = []
        for rank_idx, unit_idx in enumerate(order, start=1):
            u = self.store.units[int(unit_idx)]
            hits.append(
                SearchHit(
                    rank=rank_idx,
                    unit=u,
                    score=round(float(fused[unit_idx]), 4),
                    vector_score=round(float(dense_norm[unit_idx]), 4),
                    bm25_score=round(float(bm25_norm[unit_idx]), 4),
                    citation=u.provenance.citation,
                )
            )
        return hits
