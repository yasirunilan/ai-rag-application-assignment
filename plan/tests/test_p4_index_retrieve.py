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
