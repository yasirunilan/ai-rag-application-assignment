"""Person 2 Unit & Acceptance Tests: Structural Routing & Artefact Extraction (Stories E1.1, E2.1, E2.2, E2.3)."""

from __future__ import annotations

from pathlib import Path

import fitz

from mmrag.config import AppConfig
from mmrag.extract.figures import extract_figure_units
from mmrag.extract.spotcheck import run_provenance_spotcheck
from mmrag.extract.tables import extract_table_units, extract_text_units
from mmrag.ingest.router import explain_naive_loader_drops, parse_and_route_pdf
from mmrag.schemas import Provenance, RawRegion, TableArtefact, Unit


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


def _build_edge_case_pdf(path: Path) -> Path:
    """Two-page manual: vector-drawn schematic, two-column captions, and a heading-less continuation page."""
    doc = fitz.open()
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text((50, 50), "Chapter 7: Fuel System Plumbing (Model 787-900)", fontsize=12)
    p1.insert_text((50, 72), "7.3 Boost Pump Wiring", fontsize=11)
    p1.insert_text((50, 95), "Before working on this model is a good time to depressurize the lines.", fontsize=9)
    # Vector-drawn wiring schematic (no embedded raster image)
    shape = p1.new_shape()
    shape.draw_rect(fitz.Rect(60, 120, 280, 300))
    shape.draw_line((80, 150), (260, 150))
    shape.draw_line((80, 200), (260, 270))
    shape.draw_circle((170, 230), 25)
    shape.finish(color=(0, 0, 0), width=1.2)
    shape.commit()
    p1.insert_text((90, 140), "K1 BOOST RELAY", fontsize=8)
    p1.insert_text((60, 318), "Figure 7-5. Boost Pump Relay Schematic", fontsize=9)
    # Right-column caption at the same height must not attach to the left-column figure
    p1.insert_text((330, 318), "Figure 7-6. Crossfeed Valve Detail", fontsize=9)

    p2 = doc.new_page(width=612, height=792)
    p2.insert_text((50, 60), "Continued boost pump troubleshooting steps for the left wing tank.", fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


def test_e1_1_vector_figures_headings_and_variants(lab_cfg: AppConfig, tmp_path: Path) -> None:
    """Vector schematics are routed as figures with column-matched captions; headings carry across pages."""
    pdf = _build_edge_case_pdf(tmp_path / "edge_manual.pdf")
    _, regions = parse_and_route_pdf(pdf, lab_cfg)

    figs = [r for r in regions if r.region_type == "figure"]
    assert len(figs) == 1
    fig = figs[0]
    assert fig.page == 1
    assert fig.meta["source"] == "vector"
    assert fig.nearby_caption == "Figure 7-5. Boost Pump Relay Schematic"
    assert fig.heading_context == "Chapter 7: Fuel System Plumbing (Model 787-900) — 7.3 Boost Pump Wiring"
    assert fig.model_variant == "Model 787-900"
    assert "K1 BOOST RELAY" in fig.meta["figure_labels"]

    p2_text = [r for r in regions if r.page == 2 and r.region_type == "text"]
    assert p2_text and p2_text[0].heading_context == fig.heading_context
    assert p2_text[0].model_variant != "model is"

    fig_units, _ = extract_figure_units(regions, lab_cfg)
    assert len(fig_units) == 1
    assert fig_units[0].provenance.extraction_meta["figure_source"] == "vector"
    assert fig_units[0].provenance.extraction_meta["dpi"] == 200
    assert Path(str(fig_units[0].artefact)).stat().st_size > 1000


def test_e2_2_pymupdf_table_backend_is_honored(lab_cfg: AppConfig) -> None:
    """`extract.table_backend = pymupdf` is honored and still yields a structured TableArtefact."""
    amt_pdf = Path(lab_cfg.corpus.manuals_dir) / "amt_handbook.pdf"
    lab_cfg.extract.table_backend = "pymupdf"
    _, regions = parse_and_route_pdf(amt_pdf, lab_cfg)

    tbl_units = extract_table_units(regions, lab_cfg)
    assert len(tbl_units) == 1
    assert tbl_units[0].provenance.page == 1
    assert isinstance(tbl_units[0].artefact, TableArtefact)
    assert len(tbl_units[0].artefact.headers) == 5
    assert len(tbl_units[0].artefact.rows) == 4
    assert tbl_units[0].search_text == ""


def test_e2_2_table_grid_is_squared_with_named_headers(lab_cfg: AppConfig) -> None:
    """Blank header cells get placeholder names and ragged rows are padded to the header width."""
    region = RawRegion(
        doc_id="manual",
        page=3,
        region_type="table",
        bbox=(50.0, 100.0, 400.0, 200.0),
        reading_order=1,
        meta={"rows": [["Part Number", "", "Torque"], ["AN4-12A", "Dry"], ["AN5-7", "Lubed", "95"]]},
    )
    unit = extract_table_units([region], lab_cfg)[0]
    assert isinstance(unit.artefact, TableArtefact)
    assert unit.artefact.headers == ["Part Number", "Col_2", "Torque"]
    assert unit.artefact.rows == [["AN4-12A", "Dry", ""], ["AN5-7", "Lubed", "95"]]


def test_e2_3_spotcheck_rejects_broken_provenance(lab_cfg: AppConfig) -> None:
    """Spot-check fails units whose source PDF is missing, whose caption is on another page, or whose id lies."""
    amt_pdf = Path(lab_cfg.corpus.manuals_dir) / "amt_handbook.pdf"
    _, regions = parse_and_route_pdf(amt_pdf, lab_cfg)
    tbl = extract_table_units(regions, lab_cfg)[0]

    wrong_page = tbl.model_copy(
        update={
            "unit_id": "amt_handbook__p0002__table_001",
            "provenance": tbl.provenance.model_copy(update={"page": 2, "unit_id": "amt_handbook__p0002__table_001"}),
        }
    )
    missing_pdf = tbl.model_copy(
        update={"provenance": tbl.provenance.model_copy(update={"source_path": str(amt_pdf.with_name("gone.pdf"))})}
    )
    lying_id = Unit(
        unit_id="amt_handbook__p0003__text_001",
        type="text",
        search_text="x",
        artefact="x",
        provenance=Provenance(doc_id="amt_handbook", page=1, unit_id="amt_handbook__p0003__text_001"),
    )

    for bad, expected in [
        (wrong_page, "caption_not_found_on_cited_page"),
        (missing_pdf, "source_pdf_missing"),
        (lying_id, "unit_id_mismatch"),
    ]:
        report = run_provenance_spotcheck([bad], sample_size=1)
        assert report["passed"] is False
        assert report["samples"][0]["status"].startswith(expected)
