"""Unified Mixed Vector Index with Persisted Provenance Metadata (Person 4 · Stories E4.1 & E4.2).

Indexes `text`, `figure` (caption `search_text`), and `table` (linearized `search_text`)
units together in a single vector space, persisting full provenance (`doc -> page -> id`,
`bbox`, `source_path`, `artefact`) alongside vectors so reloading reconstructs every citation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from mmrag.config import AppConfig
from mmrag.index.chunker import chunk_units_for_indexing
from mmrag.index.embedder import EmbedderProtocol, create_embedder
from mmrag.schemas import Unit


def _try_import_faiss() -> Any:
    try:
        import faiss

        return faiss
    except ImportError:
        return None


class MixedVectorStore:
    """Single mixed index over text + figure-caption + table-text with persisted provenance."""

    def __init__(self, cfg: AppConfig, embedder: Optional[EmbedderProtocol] = None) -> None:
        self.cfg = cfg
        self.embedder = embedder or create_embedder(cfg)
        self.units: list[Unit] = []
        self.vectors: np.ndarray = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._faiss_index: Any = None

    def build(self, raw_units: list[Unit], apply_chunking: bool = True) -> None:
        """Chunks prose (keeping figures/tables atomic), embeds `search_text`, and builds the index."""
        self.units = chunk_units_for_indexing(raw_units, self.cfg) if apply_chunking else list(raw_units)
        search_texts = [u.search_text or str(u.artefact) for u in self.units]
        self.vectors = self.embedder.embed_texts(search_texts)

        faiss = _try_import_faiss()
        if faiss is not None and self.cfg.index.vector_backend == "faiss_or_numpy" and len(self.units) > 0:
            idx = faiss.IndexFlatIP(self.vectors.shape[1])
            idx.add(np.ascontiguousarray(self.vectors, dtype=np.float32))
            self._faiss_index = idx
        else:
            self._faiss_index = None

    def search_dense(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """Returns `[(unit_index, cosine_score), ...]` sorted descending by dense vector similarity."""
        if not self.units or self.vectors.shape[0] == 0:
            return []

        q_vec = self.embedder.embed_texts([query])
        k = min(top_k, len(self.units))

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(np.ascontiguousarray(q_vec, dtype=np.float32), k)
            results: list[tuple[int, float]] = []
            for idx_val, s_val in zip(indices[0], scores[0]):
                if int(idx_val) >= 0:
                    results.append((int(idx_val), float(s_val)))
            return results

        # Exact L2-normalized Inner Product fallback via NumPy
        sims = (self.vectors @ q_vec[0]).astype(np.float32)
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]

    def save(self, index_dir: str | Path | None = None) -> Path:
        """Persists vectors + units with full provenance metadata (Story E4.2)."""
        target_dir = Path(index_dir or self.cfg.project.index_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Persist every unit + full provenance (`doc -> page -> id`, bbox, artefact)
        units_path = target_dir / "units.jsonl"
        with units_path.open("w", encoding="utf-8") as f:
            for row_idx, u in enumerate(self.units):
                record = {
                    "row_id": row_idx,
                    "citation": u.provenance.citation,
                    "unit": u.model_dump(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 2. Persist dense vectors
        np.save(str(target_dir / "vectors.npy"), self.vectors)

        faiss = _try_import_faiss()
        if faiss is not None and self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(target_dir / "index.faiss"))

        # 3. Persist summary metadata
        type_counts = {
            "text": sum(1 for u in self.units if u.type == "text"),
            "figure": sum(1 for u in self.units if u.type == "figure"),
            "table": sum(1 for u in self.units if u.type == "table"),
        }
        meta = {
            "total_units": len(self.units),
            "embedding_dim": int(self.vectors.shape[1]) if self.vectors.ndim == 2 else self.embedder.dim,
            "type_counts": type_counts,
            "has_faiss_file": (target_dir / "index.faiss").exists(),
        }
        (target_dir / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return target_dir

    @classmethod
    def load(
        cls,
        cfg: AppConfig,
        index_dir: str | Path | None = None,
        embedder: Optional[EmbedderProtocol] = None,
    ) -> "MixedVectorStore":
        """Reloads persisted vectors + units and reconstructs all provenance citations (Story E4.2)."""
        target_dir = Path(index_dir or cfg.project.index_dir)
        store = cls(cfg=cfg, embedder=embedder)

        units_path = target_dir / "units.jsonl"
        loaded_units: list[Unit] = []
        with units_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                unit_data = payload["unit"] if "unit" in payload else payload
                loaded_units.append(Unit.model_validate(unit_data))
        store.units = loaded_units

        vec_path = target_dir / "vectors.npy"
        if vec_path.exists():
            store.vectors = np.load(str(vec_path)).astype(np.float32)

        faiss = _try_import_faiss()
        faiss_path = target_dir / "index.faiss"
        if faiss is not None and faiss_path.exists():
            store._faiss_index = faiss.read_index(str(faiss_path))
        elif faiss is not None and store.vectors.shape[0] > 0 and cfg.index.vector_backend == "faiss_or_numpy":
            idx = faiss.IndexFlatIP(store.vectors.shape[1])
            idx.add(np.ascontiguousarray(store.vectors, dtype=np.float32))
            store._faiss_index = idx

        return store
