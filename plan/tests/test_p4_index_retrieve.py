"""Person 4 Unit & Acceptance Tests: Mixed Index, Provenance Persistence & Hybrid Retrieval (Stories E4.1, E4.2, E5.1, E5.2)."""

from __future__ import annotations

from mmrag.config import AppConfig
from mmrag.index.embedder import create_embedder
from mmrag.index.store import MixedVectorStore
from mmrag.pipeline import MultimodalRAGPipeline
from mmrag.retrieve.retriever import HybridRetriever


def test_e4_1_and_e4_2_mixed_index_persistence_and_reload(lab_cfg: AppConfig) -> None:
    """Stories E4.1 & E4.2 Acceptance:
    - Index persists; text, figure-caption and table-text units are all present and searchable.
    - Each indexed row carries doc->page->id; reloading the index preserves citations.
    """
    pipe = MultimodalRAGPipeline(lab_cfg, force_offline=True)
    pipe.ingest_and_index(persist_index=True)

    assert pipe.store is not None
    unit_types = {u.type for u in pipe.store.units}
    assert unit_types == {"text", "figure", "table"}

    # Reload from disk and verify every single unit preserves doc -> page -> id
    reloaded = MixedVectorStore.load(
        lab_cfg,
        embedder=create_embedder(lab_cfg, force_offline=True),
    )
    assert len(reloaded.units) == len(pipe.store.units)
    for orig, rel in zip(pipe.store.units, reloaded.units):
        assert rel.unit_id == orig.unit_id
        assert rel.provenance.doc_id == orig.provenance.doc_id
        assert rel.provenance.page == orig.provenance.page
        assert rel.provenance.citation == orig.provenance.citation
        assert rel.has_valid_provenance(verify_files=True)


def test_e5_1_retrieval_returns_traceable_citations(lab_cfg: AppConfig) -> None:
    """Story E5.1 Acceptance: Retrieval returns units, each with a resolvable citation (e.g. 'amt_handbook p.3')."""
    pipe = MultimodalRAGPipeline(lab_cfg, force_offline=True)
    pipe.ingest_and_index(persist_index=False)
    assert pipe.retriever is not None

    hits = pipe.retriever.retrieve("Show the engine starter relay control wiring diagram", k=3)
    assert len(hits) == 3
    for h in hits:
        assert " p." in h.citation
        assert h.unit.has_valid_provenance(verify_files=True)


def test_e5_2_bm25_fusion_matches_exact_part_and_fault_codes(lab_cfg: AppConfig) -> None:
    """Story E5.2 Acceptance: An exact-identifier query surfaces the right unit at or near the top."""
    pipe = MultimodalRAGPipeline(lab_cfg, force_offline=True)
    pipe.ingest_and_index(persist_index=False)
    assert pipe.retriever is not None

    # Exact fault code query: FAULT-E409 (dude_hydraulic_manual p.1 Table 2-3)
    fused_hits = pipe.retriever.retrieve("FAULT-E409", k=3, enable_bm25_fusion=True, alpha=0.50)
    assert fused_hits[0].unit.provenance.doc_id == "dude_hydraulic_manual"
    assert fused_hits[0].unit.provenance.page == 1
    assert fused_hits[0].unit.type == "table"
    assert fused_hits[0].bm25_score > 0.5

    # Exact bolt part number query: NAS1304-15 (amt_handbook p.1 Table 4-1)
    part_hits = pipe.retriever.retrieve("NAS1304-15", k=3, enable_bm25_fusion=True, alpha=0.50)
    assert part_hits[0].unit.provenance.doc_id == "amt_handbook"
    assert part_hits[0].unit.provenance.page == 1
    assert part_hits[0].unit.type == "table"


# ---------------------------------------------------------------------------
# Hardening tests (E4.1 chunking & batching, E4.2 write-time gate & model lock,
# E4.3 clean rebuild, E5.1 type filter, E5.2 fused-vs-vector comparison)
# ---------------------------------------------------------------------------

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from mmrag.index.chunker import chunk_units_for_indexing, split_prose
from mmrag.index.embedder import DeterministicSemanticEmbedder, iter_batches
from mmrag.index.store import (
    EmbeddingModelMismatchError,
    IndexIntegrityError,
    provenance_rejection_reasons,
)
from mmrag.retrieve.retriever import tokenize_technical_text
from mmrag.schemas import Provenance, TableArtefact, Unit


def _text_unit(unit_id: str, text: str, doc_id: str = "amt_handbook", page=4, heading=None) -> Unit:
    return Unit(
        unit_id=unit_id,
        type="text",
        search_text=text,
        artefact=text,
        provenance=Provenance(doc_id=doc_id, page=page, unit_id=unit_id, heading_context=heading),
    )


def _indexed_pipeline(cfg: AppConfig, persist: bool = False) -> MultimodalRAGPipeline:
    pipe = MultimodalRAGPipeline(cfg, force_offline=True)
    pipe.ingest_and_index(persist_index=persist)
    return pipe


class _OtherModelEmbedder(DeterministicSemanticEmbedder):
    """Same vector width, different model identity: must never be mixed with the index."""

    def __init__(self, dim: int = 384) -> None:
        super().__init__(dim=dim)
        self.model_id = "fastembed:some/other-model"


# ---------------------------------------------------------------- E4.1


def test_e4_1_long_prose_splits_on_structure_without_cutting_words(lab_cfg: AppConfig) -> None:
    body_a = " ".join(f"Step {i} calibrates the click-type torque wrench before use." for i in range(20))
    body_b = " ".join(f"Rule {i} routes the lockwire in the tightening direction." for i in range(20))
    text = f"4.1 Torque Wrench Practice\n{body_a}\n\n4.2 Safety Wiring\n{body_b}"
    lab_cfg.index.prose_chunk_max_chars = 300
    lab_cfg.index.prose_chunk_overlap_chars = 80

    unit = _text_unit("amt_handbook__p0004__text_001", text, heading="Chapter 4: Fasteners")
    chunks = chunk_units_for_indexing([unit], lab_cfg)

    assert len(chunks) > 2
    words = set(text.split())
    for i, c in enumerate(chunks, start=1):
        assert len(c.artefact) <= 300
        assert set(c.artefact.split()) <= words, "a chunk cut a word in half"
        assert c.unit_id == f"amt_handbook__p0004__text_001_c{i:02d}"
        assert c.provenance.unit_id == c.unit_id
        assert (c.provenance.doc_id, c.provenance.page) == ("amt_handbook", 4)
        assert c.provenance.extraction_meta["chunk_parent_unit_id"] == unit.unit_id
        assert c.provenance.extraction_meta["chunk_count"] == len(chunks)
        assert c.search_text.startswith("[Chapter 4: Fasteners")

    # The second heading opens a fresh chunk and no chunk mixes the two sections
    section_b = [c for c in chunks if "Rule" in c.artefact]
    assert section_b[0].artefact.startswith("4.2 Safety Wiring")
    assert all("Step" not in c.artefact for c in section_b)
    # Chunks deep in section 4.2 are still findable by the section title
    assert all("4.2 Safety Wiring" in c.search_text for c in section_b)
    # Overlap is whole sentences carried from the previous chunk
    first_a, second_a = chunks[0].artefact, chunks[1].artefact
    assert second_a.split(". ")[0].rstrip(".") in first_a


def test_e4_1_figures_and_tables_stay_atomic_even_when_huge(lab_cfg: AppConfig) -> None:
    lab_cfg.index.prose_chunk_max_chars = 100
    huge = "Row data " * 500
    table = Unit(
        unit_id="amt_handbook__p0001__table_009",
        type="table",
        search_text=huge,
        artefact=TableArtefact(headers=["A"], rows=[["x"]] * 50),
        provenance=Provenance(doc_id="amt_handbook", page=1, unit_id="amt_handbook__p0001__table_009"),
    )
    figure = Unit(
        unit_id="amt_handbook__p0002__figure_009",
        type="figure",
        search_text=huge,
        artefact="data/artifacts/figures/x.png",
        provenance=Provenance(doc_id="amt_handbook", page=2, unit_id="amt_handbook__p0002__figure_009"),
    )
    out = chunk_units_for_indexing([table, figure], lab_cfg)
    assert [u.unit_id for u in out] == [table.unit_id, figure.unit_id]
    assert out[0].search_text == huge and out[1].search_text == huge


def test_e4_1_split_prose_handles_single_overlong_token() -> None:
    blob = "A" * 250
    chunks = split_prose(f"Intro sentence here. {blob} trailing words.", max_chars=100, overlap=20)
    assert all(len(c) <= 100 for c, _ in chunks)
    assert "".join(c for c, _ in chunks).count("A") >= 250


def test_e4_1_embeds_in_batches_of_16_with_identical_vectors() -> None:
    texts = [f"hydraulic accumulator pre-charge check {i}" for i in range(37)]
    assert [len(b) for b in iter_batches(texts)] == [16, 16, 5]

    emb = DeterministicSemanticEmbedder(dim=384)
    batched = emb.embed_texts(texts)
    one_by_one = np.vstack([emb.embed_texts([t]) for t in texts])
    assert batched.shape == (37, 384)
    np.testing.assert_allclose(batched, one_by_one)


def test_e4_1_units_jsonl_rows_carry_doc_page_id_type_payload(lab_cfg: AppConfig) -> None:
    _indexed_pipeline(lab_cfg, persist=True)
    rows = [json.loads(l) for l in (Path(lab_cfg.project.index_dir) / "units.jsonl").read_text().splitlines()]
    assert rows
    for row in rows:
        assert row["doc_id"] and row["page"] >= 1 and row["unit_id"] and row["type"] in {"text", "figure", "table"}
        assert row["unit_id"] == row["unit"]["unit_id"]
        assert row["citation"] == f"{row['doc_id']} p.{row['page']}"


# ---------------------------------------------------------------- E4.2


def test_e4_2_broken_units_are_rejected_at_write_time_and_logged(
    lab_cfg: AppConfig, caplog: pytest.LogCaptureFixture
) -> None:
    good = _text_unit("amt_handbook__p0004__text_001", "Torque AN4-12A bolts to 50-70 in-lbs.")
    no_page = _text_unit("amt_handbook__pXXXX__text_002", "Orphan paragraph about AN4-12A.", page=None)
    zero_page = _text_unit("amt_handbook__p0000__text_003", "Front matter.", page=0)
    no_doc = _text_unit("nodoc__text_004", "Unattributed AN4-12A note.", doc_id="")
    dup = _text_unit("amt_handbook__p0004__text_001", "Duplicate id.")

    assert provenance_rejection_reasons(good) == []
    assert "missing_page" in provenance_rejection_reasons(no_page)

    store = MixedVectorStore(lab_cfg, embedder=create_embedder(lab_cfg, force_offline=True))
    with caplog.at_level(logging.WARNING, logger="mmrag.index.store"):
        store.build([good, no_page, zero_page, no_doc, dup])

    assert [u.unit_id for u in store.units] == [good.unit_id]
    reasons = {r["unit_id"]: r["reasons"] for r in store.rejected}
    assert reasons[no_page.unit_id] == ["missing_page"]
    assert reasons[zero_page.unit_id] == ["invalid_page:0"]
    assert reasons[no_doc.unit_id] == ["missing_doc_id"]
    assert store.rejected[-1]["reasons"] == ["duplicate_unit_id"]
    assert "Index write rejected unit 'amt_handbook__pXXXX__text_002'" in caplog.text

    # Nothing unattributable is retrievable
    hits = HybridRetriever(store, lab_cfg).retrieve("AN4-12A", k=5)
    assert [h.unit.unit_id for h in hits] == [good.unit_id]

    # Rejections are persisted next to the index
    target = store.save()
    rejected_rows = (target / "rejected_units.jsonl").read_text().splitlines()
    assert len(rejected_rows) == 4
    assert json.loads((target / "index_meta.json").read_text())["rejected_units"] == 4


def test_e4_2_embedding_model_is_recorded_and_mismatch_refuses_to_query(lab_cfg: AppConfig) -> None:
    _indexed_pipeline(lab_cfg, persist=True)
    meta = MixedVectorStore.read_meta(lab_cfg.project.index_dir)
    assert meta["embedding_model_id"] == "deterministic-semantic-hash-v1-384d"

    with pytest.raises(EmbeddingModelMismatchError, match="Rebuild the index"):
        MixedVectorStore.load(lab_cfg, embedder=_OtherModelEmbedder())

    # Inspection-only load is allowed, but querying it is still refused
    inspect = MixedVectorStore.load(lab_cfg, embedder=_OtherModelEmbedder(), verify_embedder=False)
    assert inspect.units
    with pytest.raises(EmbeddingModelMismatchError, match="some/other-model"):
        inspect.search_dense("starter relay wiring")

    # Same model reloads fine and retrieves
    same = MixedVectorStore.load(lab_cfg, embedder=create_embedder(lab_cfg, force_offline=True))
    assert HybridRetriever(same, lab_cfg).retrieve("starter relay wiring", k=1)


def test_e4_2_inconsistent_or_missing_files_raise_integrity_error(lab_cfg: AppConfig) -> None:
    pipe = _indexed_pipeline(lab_cfg, persist=True)
    index_dir = Path(lab_cfg.project.index_dir)
    emb = create_embedder(lab_cfg, force_offline=True)

    np.save(str(index_dir / "vectors.npy"), pipe.store.vectors[:-1])
    with pytest.raises(IndexIntegrityError, match="inconsistent"):
        MixedVectorStore.load(lab_cfg, embedder=emb)

    (index_dir / "vectors.npy").unlink()
    with pytest.raises(IndexIntegrityError, match="missing"):
        MixedVectorStore.load(lab_cfg, embedder=emb)

    with pytest.raises(IndexIntegrityError, match="No index found"):
        MixedVectorStore.load(lab_cfg, index_dir=index_dir / "nope", embedder=emb)


def test_e4_2_legacy_index_without_model_id_still_loads(
    lab_cfg: AppConfig, caplog: pytest.LogCaptureFixture
) -> None:
    _indexed_pipeline(lab_cfg, persist=True)
    meta_path = Path(lab_cfg.project.index_dir) / "index_meta.json"
    meta = json.loads(meta_path.read_text())
    meta.pop("embedding_model_id")
    meta_path.write_text(json.dumps(meta))

    with caplog.at_level(logging.WARNING, logger="mmrag.index.store"):
        store = MixedVectorStore.load(lab_cfg, embedder=create_embedder(lab_cfg, force_offline=True))
    assert "legacy format" in caplog.text
    assert HybridRetriever(store, lab_cfg).retrieve("FAULT-E409", k=1)


# ---------------------------------------------------------------- E4.3


def test_e4_3_clean_rebuild_twice_gives_same_count_and_no_stale_files(lab_cfg: AppConfig) -> None:
    index_dir = Path(lab_cfg.project.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "rejected_units.jsonl").write_text('{"stale": true}\n')
    (index_dir / "units.jsonl.tmp").write_text("partial write")

    first = _indexed_pipeline(lab_cfg, persist=True).store
    counts_1 = json.loads((index_dir / "index_meta.json").read_text())["type_counts"]
    second = _indexed_pipeline(lab_cfg, persist=True).store
    counts_2 = json.loads((index_dir / "index_meta.json").read_text())["type_counts"]

    assert len(first.units) == len(second.units)
    assert counts_1 == counts_2
    assert not (index_dir / "rejected_units.jsonl").exists()
    assert not list(index_dir.glob("*.tmp"))
    reloaded = MixedVectorStore.load(lab_cfg, embedder=create_embedder(lab_cfg, force_offline=True))
    assert len(reloaded.units) == len(second.units) == reloaded.vectors.shape[0]


# ---------------------------------------------------------------- E5.1


def test_e5_1_type_filtered_searches_return_only_that_type(lab_cfg: AppConfig) -> None:
    pipe = _indexed_pipeline(lab_cfg)
    retriever = pipe.retriever
    assert retriever is not None

    figs = retriever.retrieve("wiring diagram", k=5, type_filter="figure")
    tables = retriever.retrieve("wiring diagram", k=5, type_filter="table")
    n_fig = sum(u.type == "figure" for u in pipe.store.units)
    n_tab = sum(u.type == "table" for u in pipe.store.units)

    # Filter is applied inside the search: we get every unit of the type, not a post-filtered top-k
    assert len(figs) == min(5, n_fig) and {h.unit.type for h in figs} == {"figure"}
    assert len(tables) == min(5, n_tab) and {h.unit.type for h in tables} == {"table"}
    assert [h.rank for h in figs] == list(range(1, len(figs) + 1))
    assert figs[0].unit.provenance.page == 3  # starter relay wiring diagram
    for h in figs + tables:
        assert h.citation == h.unit.provenance.citation and " p." in h.citation

    multi = retriever.retrieve("FAULT-E409", k=10, type_filter=["table", "figure"])
    assert {h.unit.type for h in multi} <= {"table", "figure"}

    with pytest.raises(ValueError, match="Unknown unit type"):
        retriever.retrieve("anything", type_filter="video")


def test_e5_1_empty_query_and_determinism(lab_cfg: AppConfig) -> None:
    retriever = _indexed_pipeline(lab_cfg).retriever
    assert retriever is not None
    assert retriever.retrieve("   ") == []
    a = retriever.retrieve("hydraulic accumulator nitrogen pre-charge", k=5)
    b = retriever.retrieve("hydraulic accumulator nitrogen pre-charge", k=5)
    assert [(h.unit.unit_id, h.score) for h in a] == [(h.unit.unit_id, h.score) for h in b]
    with pytest.raises(ValueError, match="alpha"):
        retriever.retrieve("torque", alpha=1.5)


# ---------------------------------------------------------------- E5.2


def test_e5_2_fused_beats_vector_only_on_part_number_query(lab_cfg: AppConfig) -> None:
    retriever = _indexed_pipeline(lab_cfg).retriever
    assert retriever is not None

    report = retriever.compare_fused_vs_vector("NAS1304-15", k=3)
    assert report["alpha"] == lab_cfg.retrieve.alpha
    vector_top, fused_top = report["vector_only"][0], report["fused"][0]

    # Dense vectors smear the code onto the wrong table; BM25 fusion recovers the exact one
    assert vector_top["unit_id"] != "amt_handbook__p0001__table_001"
    assert fused_top["unit_id"] == "amt_handbook__p0001__table_001"
    assert fused_top["citation"] == "amt_handbook p.1"
    assert report["fused_top1_rank_in_vector_only"] > 1
    assert all(r["bm25_score"] == 0.0 for r in report["vector_only"])


def test_e5_2_code_tokens_match_across_separator_variants(lab_cfg: AppConfig) -> None:
    doc_tokens = tokenize_technical_text("Part Number = AN4-12A ; Dry Torque = 50-70")
    assert doc_tokens.count("an4-12a") == 2
    assert {"an412a", "an4", "12a"} <= set(doc_tokens)
    # Query side keeps exact + compact form only, so near-miss codes get no partial credit
    assert set(tokenize_technical_text("AN4-16A", expand_code_parts=False)) == {"an4-16a", "an416a"}
    assert "the" not in tokenize_technical_text("What is the torque for the bolt")

    retriever = _indexed_pipeline(lab_cfg).retriever
    assert retriever is not None
    for variant in ("AN412A", "AN4 12A"):
        bm25 = retriever._bm25_scores(variant)
        best = retriever.store.units[int(np.argmax(bm25))]
        assert best.unit_id == "amt_handbook__p0001__table_001", variant


def test_e5_2_bm25_candidate_pool_rescues_exact_code_outside_dense_pool(lab_cfg: AppConfig) -> None:
    lab_cfg.retrieve.candidate_pool_size = 1
    retriever = _indexed_pipeline(lab_cfg).retriever
    assert retriever is not None
    dense_top = retriever.store.search_dense("NAS1304-15", top_k=1)
    assert retriever.store.units[dense_top[0][0]].unit_id != "amt_handbook__p0001__table_001"

    # Dense-only pool of 1 can never contain the right table ...
    vec_only = retriever.retrieve("NAS1304-15", k=1, enable_bm25_fusion=False)
    assert vec_only[0].unit.unit_id != "amt_handbook__p0001__table_001"
    # ... but the BM25 pool adds it as a candidate, so a lexical-leaning fusion can rank it first
    hits = retriever.retrieve("NAS1304-15", k=1, alpha=0.4)
    assert hits[0].unit.unit_id == "amt_handbook__p0001__table_001"
    assert hits[0].bm25_score == 1.0
