"""Person 3 Unit & Acceptance Tests: Vision Captioning, Disk Cache & Table Linearization (Stories E3.1, E3.2)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from mmrag.config import AppConfig
from mmrag.enrich.captioner import enrich_figure_units
from mmrag.enrich.table_linearizer import enrich_table_units
from mmrag.extract.figures import extract_figure_units
from mmrag.extract.tables import extract_table_units
from mmrag.ingest.router import parse_and_route_pdf
from mmrag.schemas import TableArtefact


def test_e3_1_figure_captioning_and_disk_cache(lab_cfg: AppConfig) -> None:
    """Story E3.1 Acceptance: Every figure unit has non-empty search_text; re-runs served from cache; image untouched."""
    amt_pdf = Path(lab_cfg.corpus.manuals_dir) / "amt_handbook.pdf"
    _, regions = parse_and_route_pdf(amt_pdf, lab_cfg)
    fig_units, _ = extract_figure_units(regions, lab_cfg)

    # Record SHA-256 of image files before enrichment
    before_hashes = {
        u.unit_id: hashlib.sha256(Path(str(u.artefact)).read_bytes()).hexdigest()
        for u in fig_units
    }

    # First run: populates cache
    enriched_1, cache_1 = enrich_figure_units(fig_units, lab_cfg)
    assert cache_1.misses == len(fig_units)
    for u in enriched_1:
        assert u.search_text.strip() != ""
        assert "amt_handbook" in u.search_text
        # Image bytes on disk must be untouched
        after_hash = hashlib.sha256(Path(str(u.artefact)).read_bytes()).hexdigest()
        assert after_hash == before_hashes[u.unit_id]

    # Second run: 100% served from disk cache
    enriched_2, cache_2 = enrich_figure_units(fig_units, lab_cfg)
    assert cache_2.hits == len(fig_units)
    assert cache_2.misses == 0
    assert [u.search_text for u in enriched_2] == [u.search_text for u in enriched_1]


def test_e3_2_table_linearization_preserves_structured_artefact(lab_cfg: AppConfig) -> None:
    """Story E3.2 Acceptance: Every table unit has search_text; the structured cells are unchanged."""
    amt_pdf = Path(lab_cfg.corpus.manuals_dir) / "amt_handbook.pdf"
    _, regions = parse_and_route_pdf(amt_pdf, lab_cfg)
    tbl_units = extract_table_units(regions, lab_cfg)

    original_artefact = tbl_units[0].artefact
    assert isinstance(original_artefact, TableArtefact)

    enriched_tables = enrich_table_units(tbl_units, lab_cfg)
    t_unit = enriched_tables[0]

    assert t_unit.search_text.strip() != ""
    assert "Part Number = AN4-12A" in t_unit.search_text
    assert "Dry Torque (in-lbs) = 50-70" in t_unit.search_text
    assert "Table 4-1" in t_unit.search_text

    # Structured artefact must remain completely unchanged
    assert isinstance(t_unit.artefact, TableArtefact)
    assert t_unit.artefact.headers == original_artefact.headers
    assert t_unit.artefact.rows == original_artefact.rows
