"""Hybrid Vector + BM25 Lexical Retriever (Person 4 · Stories E5.1 & E5.2).

Implements:
- `retrieve(query, k, type_filter=...)` returning top-k `SearchHit` objects with provenance and
  human-readable citations (`doc p.212`). The type filter is applied *inside* both the vector
  search and the BM25 scorer, so "figure-only" / "table-only" searches return k units of that type.
- Technical-identifier-aware BM25 lexical scoring over the same `search_text` the vectors embed,
  fused via min-max normalization and tunable `alpha`
  (`score = alpha * vec_norm + (1 - alpha) * bm25_norm`).
- Candidate pool: the top `candidate_pool_size` dense hits ∪ the top `candidate_pool_size` BM25
  hits are normalized and fused, so an exact-code match is never lost just because the dense
  model ranked it low.
- `compare_fused_vs_vector(...)` (E5.2: "compare fused and vector-only ranking on a part-number
  query"); also runnable as `python -m mmrag.retrieve.retriever "NAS1304-15"`.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional, Union

import numpy as np
from rank_bm25 import BM25Okapi

from mmrag.config import AppConfig
from mmrag.index.store import MixedVectorStore
from mmrag.schemas import SearchHit


CODE_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*", re.IGNORECASE)
_CODE_SEPARATORS = re.compile(r"[-./]")

VALID_UNIT_TYPES = ("text", "figure", "table")
TypeFilter = Optional[Union[str, Iterable[str]]]

BM25_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
        "how", "i", "in", "is", "it", "me", "of", "on", "or", "please", "show", "tell", "that",
        "the", "this", "to", "what", "when", "where", "which", "with",
    }
)


def _is_code_token(tok: str) -> bool:
    return bool(_CODE_SEPARATORS.search(tok)) and any(c.isdigit() for c in tok)


def tokenize_technical_text(text: str, expand_code_parts: bool = True) -> list[str]:
    """Tokenizes text while preserving hyphenated part/fault numbers (e.g. `an4-12a`, `fault-e409`).

    For every compound code token we emit:
    - the exact token twice (boosts exact-identifier weight),
    - its separator-free form (`an412a`) so `AN412A` matches `AN4-12A`,
    - and, when `expand_code_parts` (document side), its alphanumeric parts (`an4`, `12a`) so a
      query typed as `AN4 12A` still matches. Queries keep parts off so a near-miss code
      (`AN4-16A`) does not collect partial credit for sharing the `an4` prefix.
    Common English stopwords are dropped; identifiers are never dropped.
    """
    tokens: list[str] = []
    for tok in CODE_TOKEN_PATTERN.findall((text or "").lower()):
        if _is_code_token(tok):
            tokens.extend((tok, tok, _CODE_SEPARATORS.sub("", tok)))
            if expand_code_parts:
                tokens.extend(
                    p for p in _CODE_SEPARATORS.split(tok)
                    if len(p) >= 2 and any(c.isalpha() for c in p)
                )
        elif tok not in BM25_STOPWORDS:
            tokens.append(tok)
    return tokens or [""]


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores.astype(np.float32)
    s_min = float(np.min(scores))
    s_max = float(np.max(scores))
    if s_max - s_min < 1e-8:
        return np.zeros_like(scores, dtype=np.float32) if s_max <= 0.0 else np.ones_like(scores, dtype=np.float32)
    return ((scores - s_min) / (s_max - s_min)).astype(np.float32)


def normalize_type_filter(type_filter: TypeFilter) -> Optional[tuple[str, ...]]:
    """`None` / `"figure"` / `["figure", "table"]` -> validated tuple (or None for no filter)."""
    if type_filter is None:
        return None
    raw = [type_filter] if isinstance(type_filter, str) else list(type_filter)
    wanted = tuple(sorted({str(t).strip().lower() for t in raw if str(t).strip()}))
    if not wanted:
        return None
    unknown = [t for t in wanted if t not in VALID_UNIT_TYPES]
    if unknown:
        raise ValueError(f"Unknown unit type filter {unknown}; expected any of {list(VALID_UNIT_TYPES)}")
    return wanted


class HybridRetriever:
    """Combines dense vector search with BM25 lexical fusion and provenance citations."""

    def __init__(self, store: MixedVectorStore, cfg: AppConfig) -> None:
        self.store = store
        self.cfg = cfg
        corpus_tokens = [
            tokenize_technical_text(u.search_text or str(u.artefact), expand_code_parts=True)
            for u in self.store.units
        ]
        self._bm25: Optional[BM25Okapi] = BM25Okapi(corpus_tokens) if corpus_tokens else None

    def _bm25_scores(self, query: str) -> np.ndarray:
        n_units = len(self.store.units)
        if self._bm25 is None:
            return np.zeros(n_units, dtype=np.float32)
        q_tokens = [t for t in tokenize_technical_text(query, expand_code_parts=False) if t]
        if not q_tokens:
            return np.zeros(n_units, dtype=np.float32)
        return np.asarray(self._bm25.get_scores(q_tokens), dtype=np.float32)

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        alpha: Optional[float] = None,
        enable_bm25_fusion: Optional[bool] = None,
        type_filter: TypeFilter = None,
    ) -> list[SearchHit]:
        """Returns top-k `SearchHit` results with citations (`doc p.212`) and score breakdowns.

        `type_filter` ("text" | "figure" | "table" or an iterable of them) restricts the search
        space before ranking; scores are normalized among the allowed candidates only.
        """
        n_units = len(self.store.units)
        if n_units == 0 or not (query or "").strip():
            return []

        types = normalize_type_filter(type_filter)
        allowed = self.store.rows_for_types(types)
        n_allowed = n_units if allowed is None else int(allowed.size)
        if n_allowed == 0:
            return []

        top_k = max(1, min(k or self.cfg.retrieve.top_k, n_allowed))
        use_bm25 = (
            self.cfg.retrieve.enable_bm25_fusion
            if enable_bm25_fusion is None
            else enable_bm25_fusion
        )
        weight_alpha = self.cfg.retrieve.alpha if alpha is None else float(alpha)
        if not 0.0 <= weight_alpha <= 1.0:
            raise ValueError(f"alpha must be within [0, 1] (got {weight_alpha})")
        if not use_bm25:
            weight_alpha = 1.0
        pool_size = max(top_k, int(self.cfg.retrieve.candidate_pool_size or 0))

        # 1. Dense candidate pool (verify model match BEFORE embedding the query)
        self.store.assert_queryable()
        q_vec = self.store.embedder.embed_texts([query])
        dense_pairs = self.store.search_dense(query, top_k=pool_size, unit_types=types, query_vector=q_vec)
        dense_by_row: dict[int, float] = {idx: s for idx, s in dense_pairs}

        # 2. BM25 candidate pool over the same allowed rows
        bm25_by_row: dict[int, float] = {}
        if use_bm25 and weight_alpha < 1.0:
            bm25_all = self._bm25_scores(query)
            rows = np.arange(n_units) if allowed is None else allowed
            row_scores = bm25_all[rows]
            for pos in np.argsort(-row_scores, kind="stable")[:pool_size]:
                if row_scores[pos] > 0.0:
                    bm25_by_row[int(rows[pos])] = float(row_scores[pos])
            # Lexical scores for dense candidates outside the BM25 pool (usually 0)
            for idx in dense_by_row:
                bm25_by_row.setdefault(idx, float(bm25_all[idx]))

        # 3. Exact dense similarity for BM25-only candidates, then normalize within the pool
        candidates = sorted(set(dense_by_row) | set(bm25_by_row))
        missing = [i for i in candidates if i not in dense_by_row]
        if missing:
            sims = self.store.vectors[np.asarray(missing)] @ q_vec[0]
            dense_by_row.update({i: float(s) for i, s in zip(missing, sims)})

        cand = np.asarray(candidates, dtype=np.int64)
        dense_norm = _min_max_normalize(np.asarray([dense_by_row[i] for i in candidates], dtype=np.float32))
        if bm25_by_row:
            bm25_norm = _min_max_normalize(np.asarray([bm25_by_row.get(i, 0.0) for i in candidates], dtype=np.float32))
        else:
            bm25_norm = np.zeros(len(candidates), dtype=np.float32)

        # 4. Linear fusion: alpha * dense + (1 - alpha) * bm25 ; stable on ties
        fused = (weight_alpha * dense_norm) + ((1.0 - weight_alpha) * bm25_norm)
        order = np.lexsort((cand, -dense_norm, -fused))[:top_k]

        hits: list[SearchHit] = []
        for rank_idx, pos in enumerate(order, start=1):
            u = self.store.units[int(cand[pos])]
            hits.append(
                SearchHit(
                    rank=rank_idx,
                    unit=u,
                    score=round(float(fused[pos]), 4),
                    vector_score=round(float(dense_norm[pos]), 4),
                    bm25_score=round(float(bm25_norm[pos]), 4),
                    citation=u.provenance.citation,
                )
            )
        return hits

    def compare_fused_vs_vector(
        self,
        query: str,
        k: Optional[int] = None,
        alpha: Optional[float] = None,
        type_filter: TypeFilter = None,
    ) -> dict[str, Any]:
        """Story E5.2: side-by-side vector-only vs fused ranking for one (part-number) query."""
        top_k = k or self.cfg.retrieve.top_k
        weight_alpha = self.cfg.retrieve.alpha if alpha is None else alpha
        vector_only = self.retrieve(query, k=top_k, enable_bm25_fusion=False, type_filter=type_filter)
        fused = self.retrieve(query, k=top_k, alpha=weight_alpha, enable_bm25_fusion=True, type_filter=type_filter)

        def rows(hits: list[SearchHit]) -> list[dict[str, Any]]:
            return [
                {
                    "rank": h.rank,
                    "unit_id": h.unit.unit_id,
                    "type": h.unit.type,
                    "citation": h.citation,
                    "score": h.score,
                    "vector_score": h.vector_score,
                    "bm25_score": h.bm25_score,
                }
                for h in hits
            ]

        vec_rank = {h.unit.unit_id: h.rank for h in vector_only}
        return {
            "query": query,
            "k": top_k,
            "alpha": weight_alpha,
            "vector_only": rows(vector_only),
            "fused": rows(fused),
            "fused_top1_rank_in_vector_only": (
                vec_rank.get(fused[0].unit.unit_id) if fused else None
            ),
        }


def _main(argv: Optional[list[str]] = None) -> int:
    """`python -m mmrag.retrieve.retriever QUERY [--k N] [--type T] [--alpha A] [--offline]`."""
    import argparse
    import json

    from mmrag.config import load_config
    from mmrag.index.embedder import create_embedder

    parser = argparse.ArgumentParser(description="Compare vector-only vs BM25-fused retrieval (Story E5.2).")
    parser.add_argument("query")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--type", dest="type_filter", choices=VALID_UNIT_TYPES, default=None)
    parser.add_argument("--offline", action="store_true", help="use the zero-network embedder")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    store = MixedVectorStore.load(cfg, embedder=create_embedder(cfg, force_offline=args.offline))
    report = HybridRetriever(store, cfg).compare_fused_vs_vector(
        args.query, k=args.k, alpha=args.alpha, type_filter=args.type_filter
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
