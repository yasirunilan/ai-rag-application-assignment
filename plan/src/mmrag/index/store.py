"""Unified Mixed Vector Index with Persisted Provenance Metadata (Person 4 · Stories E4.1–E4.3).

Indexes `text`, `figure` (caption `search_text`), and `table` (linearized `search_text`)
units together in a single vector space, persisting full provenance (`doc -> page -> id`,
`bbox`, `source_path`, `artefact`) alongside vectors so reloading reconstructs every citation.

- E4.1: one mixed index; each row is a vector plus a payload (`doc_id`, `page`, `unit_id`, `type`).
- E4.2: units missing doc / page / unit id are rejected at write time and logged
  (`rejected_units.jsonl`); the embedder's `model_id` is saved with the index and a store
  loaded with a different embedder refuses to be queried (`EmbeddingModelMismatchError`).
- E4.3: `save()` resets the index directory (stale FAISS / vectors / rejections are removed)
  and writes every file atomically, so rebuilding twice yields the same unit count.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from mmrag.config import AppConfig
from mmrag.index.chunker import chunk_units_for_indexing
from mmrag.index.embedder import EmbedderProtocol, create_embedder
from mmrag.schemas import Unit

logger = logging.getLogger(__name__)

INDEX_FORMAT_VERSION = 2
UNITS_FILE = "units.jsonl"
VECTORS_FILE = "vectors.npy"
FAISS_FILE = "index.faiss"
META_FILE = "index_meta.json"
REJECTED_FILE = "rejected_units.jsonl"
INDEX_FILES = (UNITS_FILE, VECTORS_FILE, FAISS_FILE, META_FILE, REJECTED_FILE)


class IndexRebuildRequired(RuntimeError):
    """Base class for index problems that are fixed by rebuilding (`python -m mmrag.cli ingest`)."""


class EmbeddingModelMismatchError(IndexRebuildRequired):
    """The index was built by a different embedding model than the one configured to query it."""


class IndexIntegrityError(IndexRebuildRequired):
    """Persisted index files are missing or inconsistent with each other."""


REBUILD_HINT = "Rebuild the index with: python -m mmrag.cli ingest (add --offline for the zero-network embedder)."


def _try_import_faiss() -> Any:
    try:
        import faiss

        return faiss
    except ImportError:
        return None


def provenance_rejection_reasons(unit: Unit) -> list[str]:
    """Write-time provenance gate (Story E4.2). Empty list means the unit may be indexed."""
    reasons: list[str] = []
    prov = unit.provenance
    if not (prov.doc_id or "").strip():
        reasons.append("missing_doc_id")
    if prov.page is None:
        reasons.append("missing_page")
    elif int(prov.page) < 1:
        reasons.append(f"invalid_page:{prov.page}")
    if not (unit.unit_id or "").strip() or not (prov.unit_id or "").strip():
        reasons.append("missing_unit_id")
    elif prov.unit_id != unit.unit_id:
        reasons.append(f"unit_id_mismatch:{unit.unit_id}!={prov.unit_id}")
    if not (unit.search_text or "").strip() and not str(unit.artefact or "").strip():
        reasons.append("empty_searchable_text")
    return reasons


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class MixedVectorStore:
    """Single mixed index over text + figure-caption + table-text with persisted provenance."""

    def __init__(self, cfg: AppConfig, embedder: Optional[EmbedderProtocol] = None) -> None:
        self.cfg = cfg
        self.embedder = embedder or create_embedder(cfg)
        self.units: list[Unit] = []
        self.vectors: np.ndarray = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self.rejected: list[dict[str, Any]] = []
        self.index_model_id: str = getattr(self.embedder, "model_id", "unknown")
        self._faiss_index: Any = None
        self._type_rows: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ build / upsert
    @property
    def embedder_model_id(self) -> str:
        return getattr(self.embedder, "model_id", "unknown")

    def build(self, raw_units: list[Unit], apply_chunking: bool = True) -> None:
        """Resets the index, then chunks prose (keeping figures/tables atomic), gates provenance,
        embeds `search_text` in batches and builds the vector index."""
        candidates = chunk_units_for_indexing(raw_units, self.cfg) if apply_chunking else list(raw_units)

        self.units = []
        self.rejected = []
        seen: set[str] = set()
        for u in candidates:
            reasons = provenance_rejection_reasons(u)
            if not reasons and u.unit_id in seen:
                reasons = ["duplicate_unit_id"]
            if reasons:
                self._reject(u, reasons)
                continue
            seen.add(u.unit_id)
            self.units.append(u)

        search_texts = [self._searchable_text(u) for u in self.units]
        if self.units:
            self.vectors = self.embedder.embed_texts(search_texts).astype(np.float32)
        else:
            self.vectors = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self.index_model_id = self.embedder_model_id
        self._rebuild_search_structures()

    @staticmethod
    def _searchable_text(u: Unit) -> str:
        return u.search_text if (u.search_text or "").strip() else str(u.artefact)

    def _reject(self, u: Unit, reasons: list[str]) -> None:
        record = {
            "unit_id": u.unit_id,
            "type": u.type,
            "doc_id": u.provenance.doc_id,
            "page": u.provenance.page,
            "reasons": reasons,
            "search_text_preview": (u.search_text or str(u.artefact))[:160],
        }
        self.rejected.append(record)
        logger.warning(
            "Index write rejected unit %r (%s): %s", u.unit_id or "<no id>", u.type, ", ".join(reasons)
        )

    def _rebuild_search_structures(self) -> None:
        self._type_rows = {}
        for unit_type in ("text", "figure", "table"):
            rows = [i for i, u in enumerate(self.units) if u.type == unit_type]
            self._type_rows[unit_type] = np.asarray(rows, dtype=np.int64)

        self._faiss_index = None
        faiss = _try_import_faiss()
        if faiss is not None and self.cfg.index.vector_backend == "faiss_or_numpy" and len(self.units) > 0:
            idx = faiss.IndexFlatIP(self.vectors.shape[1])
            idx.add(np.ascontiguousarray(self.vectors, dtype=np.float32))
            self._faiss_index = idx

    # ------------------------------------------------------------------ query
    def assert_queryable(self) -> None:
        """Refuses to query when the configured embedder differs from the one that built the index."""
        current = self.embedder_model_id
        if self.index_model_id != current:
            raise EmbeddingModelMismatchError(
                f"Index was built with embedding model '{self.index_model_id}' but the configured "
                f"embedder is '{current}'. Vectors from different models are not comparable. {REBUILD_HINT}"
            )
        if self.vectors.ndim == 2 and self.vectors.shape[0] and self.vectors.shape[1] != self.embedder.dim:
            raise EmbeddingModelMismatchError(
                f"Index vectors are {self.vectors.shape[1]}-d but the embedder produces "
                f"{self.embedder.dim}-d vectors. {REBUILD_HINT}"
            )

    def rows_for_types(self, unit_types: Optional[Iterable[str]]) -> Optional[np.ndarray]:
        """Row indices whose unit type is in `unit_types` (None => all rows)."""
        if unit_types is None:
            return None
        wanted = sorted({str(t) for t in unit_types})
        parts = [self._type_rows.get(t, np.zeros(0, dtype=np.int64)) for t in wanted]
        return np.sort(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)

    def search_dense(
        self,
        query: str,
        top_k: int = 50,
        unit_types: Optional[Iterable[str]] = None,
        query_vector: Optional[np.ndarray] = None,
    ) -> list[tuple[int, float]]:
        """Returns `[(unit_index, cosine_score), ...]` sorted descending by dense vector similarity.

        `unit_types` restricts the search *inside* the vector search (Story E5.1 type filter),
        so the top-k is taken among allowed rows only rather than post-filtered.
        """
        if not self.units or self.vectors.shape[0] == 0 or top_k <= 0:
            return []
        self.assert_queryable()

        q_vec = query_vector if query_vector is not None else self.embedder.embed_texts([query])
        q_vec = np.ascontiguousarray(np.asarray(q_vec, dtype=np.float32).reshape(1, -1))
        allowed = self.rows_for_types(unit_types)

        if allowed is not None:
            if allowed.size == 0:
                return []
            sims = (self.vectors[allowed] @ q_vec[0]).astype(np.float32)
            k = min(top_k, allowed.size)
            order = np.argsort(-sims, kind="stable")[:k]
            return [(int(allowed[i]), float(sims[i])) for i in order]

        k = min(top_k, len(self.units))
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q_vec, k)
            return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if int(i) >= 0]

        # Exact L2-normalized Inner Product fallback via NumPy
        sims = (self.vectors @ q_vec[0]).astype(np.float32)
        order = np.argsort(-sims, kind="stable")[:k]
        return [(int(i), float(sims[i])) for i in order]

    # ------------------------------------------------------------------ persistence
    def save(self, index_dir: str | Path | None = None) -> Path:
        """Clean rebuild (E4.3): resets the index dir, then persists vectors + units + payload
        + rejection log + embedding model metadata atomically (E4.1 & E4.2)."""
        target_dir = Path(index_dir or self.cfg.project.index_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        # 0. Reset: no stale FAISS / vectors / rejections can survive a rebuild
        for name in INDEX_FILES:
            for stale in (target_dir / name, target_dir / f"{name}.tmp"):
                if stale.exists():
                    stale.unlink()

        # 1. Every unit as payload (doc, page, unit id, type) + full provenance
        lines = []
        for row_idx, u in enumerate(self.units):
            record = {
                "row_id": row_idx,
                "unit_id": u.unit_id,
                "doc_id": u.provenance.doc_id,
                "page": u.provenance.page,
                "type": u.type,
                "citation": u.provenance.citation,
                "unit": u.model_dump(),
            }
            lines.append(json.dumps(record, ensure_ascii=False))
        _atomic_write_text(target_dir / UNITS_FILE, "".join(f"{line}\n" for line in lines))

        # 2. Dense vectors (+ FAISS mirror when available)
        tmp_vec = target_dir / f"{VECTORS_FILE}.tmp"
        with tmp_vec.open("wb") as fh:
            np.save(fh, self.vectors)
        os.replace(tmp_vec, target_dir / VECTORS_FILE)

        faiss = _try_import_faiss()
        if faiss is not None and self._faiss_index is not None:
            tmp_faiss = target_dir / f"{FAISS_FILE}.tmp"
            faiss.write_index(self._faiss_index, str(tmp_faiss))
            os.replace(tmp_faiss, target_dir / FAISS_FILE)

        # 3. Write-time rejections (E4.2: "show it happening")
        if self.rejected:
            _atomic_write_text(
                target_dir / REJECTED_FILE,
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.rejected),
            )

        # 4. Summary metadata incl. the embedding model that produced the vectors
        type_counts = Counter(u.type for u in self.units)
        meta = {
            "index_format_version": INDEX_FORMAT_VERSION,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "embedding_model_id": self.index_model_id,
            "embedding_provider": self.cfg.index.embedding_provider,
            "embedding_model_name": self.cfg.index.embedding_model_name,
            "total_units": len(self.units),
            "embedding_dim": int(self.vectors.shape[1]) if self.vectors.ndim == 2 else self.embedder.dim,
            "type_counts": {t: int(type_counts.get(t, 0)) for t in ("text", "figure", "table")},
            "rejected_units": len(self.rejected),
            "has_faiss_file": (target_dir / FAISS_FILE).exists(),
        }
        _atomic_write_text(target_dir / META_FILE, json.dumps(meta, indent=2))
        return target_dir

    @staticmethod
    def read_meta(index_dir: str | Path) -> dict[str, Any]:
        meta_path = Path(index_dir) / META_FILE
        if not meta_path.exists():
            return {}
        return json.loads(meta_path.read_text(encoding="utf-8"))

    @classmethod
    def load(
        cls,
        cfg: AppConfig,
        index_dir: str | Path | None = None,
        embedder: Optional[EmbedderProtocol] = None,
        verify_embedder: bool = True,
    ) -> "MixedVectorStore":
        """Reloads persisted vectors + units and reconstructs all provenance citations (Story E4.2).

        Raises `EmbeddingModelMismatchError` when `verify_embedder` is set and the index was built
        by a different embedding model, and `IndexIntegrityError` when files are inconsistent.
        Pass `verify_embedder=False` for read-only inspection (e.g. provenance spot-checks);
        querying such a store still raises on mismatch.
        """
        target_dir = Path(index_dir or cfg.project.index_dir)
        store = cls(cfg=cfg, embedder=embedder)

        units_path = target_dir / UNITS_FILE
        if not units_path.exists():
            raise IndexIntegrityError(f"No index found at {units_path}. {REBUILD_HINT}")

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

        vec_path = target_dir / VECTORS_FILE
        if vec_path.exists():
            store.vectors = np.load(str(vec_path)).astype(np.float32)
        elif loaded_units:
            raise IndexIntegrityError(f"{vec_path} is missing for {len(loaded_units)} units. {REBUILD_HINT}")
        if store.vectors.ndim != 2 or store.vectors.shape[0] != len(store.units):
            raise IndexIntegrityError(
                f"Index is inconsistent: {len(store.units)} units but vectors shape "
                f"{tuple(store.vectors.shape)}. {REBUILD_HINT}"
            )

        meta = cls.read_meta(target_dir)
        recorded = meta.get("embedding_model_id")
        if recorded:
            store.index_model_id = recorded
        else:
            # Legacy (pre-E4.2) index: model was never recorded. Trust the embedder but say so.
            logger.warning(
                "Index at %s does not record its embedding model (legacy format); assuming %r. %s",
                target_dir, store.embedder_model_id, REBUILD_HINT,
            )
            store.index_model_id = store.embedder_model_id

        if verify_embedder:
            store.assert_queryable()

        store._rebuild_search_structures()
        faiss_path = target_dir / FAISS_FILE
        if store._faiss_index is not None and faiss_path.exists():
            # Prefer the persisted FAISS file when it agrees with the vectors
            faiss = _try_import_faiss()
            persisted = faiss.read_index(str(faiss_path))
            if persisted.ntotal == store.vectors.shape[0] and persisted.d == store.vectors.shape[1]:
                store._faiss_index = persisted
            else:
                logger.warning("Ignoring stale %s (ntotal=%d, d=%d); rebuilt from vectors.npy",
                               faiss_path, persisted.ntotal, persisted.d)
        return store
