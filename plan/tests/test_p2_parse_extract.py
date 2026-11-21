"""Person 2 Unit & Acceptance Tests: Structural Routing & Artefact Extraction (Stories E1.1, E2.1, E2.2, E2.3)."""

from __future__ import annotations

from pathlib import Path

from mmrag.config import AppConfig
from mmrag.extract.figures import extract_figure_units
from mmrag.extract.spotcheck import run_provenance_spotcheck
from mmrag.extract.tables import extract_table_units, extract_text_units
from mmrag.ingest.router import explain_naive_loader_drops, parse_and_route_pdf
from mmrag.schemas import TableArtefact


def test_e1_1_structural_routing_and_naive_loader_analysis(lab_cfg: AppConfig) -> None:
    """Story E1.1 Acceptance: Every page is classified; each figure/table region carries a page number."""
    amt_pdf = Path(lab_cfg.corpus.manuals_dir) / "amt_handbook.pdf"

    drops = explain_naive_loader_drops(amt_pdf)
    assert drops["figures_dropped_count"] >= 2
    assert drops["tables_flattened_count"] >= 1
    assert drops["scanned_pages_silently_lost"] == 1

    classifications, regions = parse_and_route_pdf(amt_pdf, lab_cfg)
    assert len(classifications) == 4
    assert [c.route_decision for c in classifications] == [
        "digital_parse",
        "digital_parse",
        "digital_parse",
        "ocr_review_queue",
    ]

    # Verify page numbers and region types
    assert all(r.page >= 1 for r in regions)
    region_types = {r.region_type for r in regions}
    assert {"text", "figure", "table"}.issubset(region_types)
    # Page 4 failed OCR quality gate and must NOT appear in indexed regions
    assert all(r.page != 4 for r in regions)


def test_e2_1_figure_crop_export_and_icon_filtering(lab_cfg: AppConfig) -> None:
    """Story E2.1 Acceptance: Figure images are written to disk; each resolves to its exact page; tiny rules filtered."""
    amt_pdf = Path(lab_cfg.corpus.manuals_dir) / "amt_handbook.pdf"
    _, regions = parse_and_route_pdf(amt_pdf, lab_cfg)

    fig_units, filtered = extract_figure_units(regions, lab_cfg)
    # Tiny decorative rule on page 1 (28x10 pt) must be filtered out
    assert len(filtered) >= 1
    assert any(f["page"] == 1 for f in filtered)

    # Pages 2 and 3 have real schematic diagrams (Figures 9-12 and 9-14)
    assert len(fig_units) == 2
    pages = sorted(u.provenance.page for u in fig_units if u.provenance.page)
    assert pages == [2, 3]
    for u in fig_units:
        img_path = Path(str(u.artefact))
        assert img_path.exists()
        assert img_path.stat().st_size > 1000
        assert u.search_text == ""  # Filled in Stage 3
        assert u.provenance.nearby_caption is not None


def test_e2_2_structured_table_extraction(lab_cfg: AppConfig) -> None:
    """Story E2.2 Acceptance: Tables stored structured + page; no table is reduced to a flat string."""
    amt_pdf = Path(lab_cfg.corpus.manuals_dir) / "amt_handbook.pdf"
    _, regions = parse_and_route_pdf(amt_pdf, lab_cfg)

    tbl_units = extract_table_units(regions, lab_cfg)
    assert len(tbl_units) == 1
    t_unit = tbl_units[0]
    assert t_unit.provenance.page == 1
    assert t_unit.search_text == ""  # Story E2.2: leave search_text empty here
    assert isinstance(t_unit.artefact, TableArtefact)
    assert "Part Number" in t_unit.artefact.headers
    assert any("AN4-12A" in row for row in t_unit.artefact.rows)


def test_e2_3_provenance_spotcheck_passes(lab_cfg: AppConfig) -> None:
    """Story E2.3 Acceptance: Spot-check passes; every sampled unit resolves to a page."""
    amt_pdf = Path(lab_cfg.corpus.manuals_dir) / "amt_handbook.pdf"
    _, regions = parse_and_route_pdf(amt_pdf, lab_cfg)
    fig_units, _ = extract_figure_units(regions, lab_cfg)
    tbl_units = extract_table_units(regions, lab_cfg)
    txt_units = extract_text_units(regions)

    all_units = txt_units + fig_units + tbl_units
    report = run_provenance_spotcheck(all_units, sample_size=5)
    assert report["passed"] is True
    assert report["sampled_count"] == 5
    assert len(report["failures"]) == 0
