"""Figure Extraction & Rule/Icon Filtering (Person 2 · Story E2.1).

Exports the cropped figure image to disk, stamps `doc -> page -> figure_id`, bbox,
and nearby caption text on the Unit, and filters out decorative rules/icons below
the minimum geometry gate.
"""

from __future__ import annotations

from pathlib import Path

import fitz

from mmrag.config import AppConfig
from mmrag.schemas import Provenance, RawRegion, Unit


def is_valid_figure_bbox(
    bbox: tuple[float, float, float, float],
    cfg: AppConfig,
) -> tuple[bool, str]:
    """Geometry filter to reject decorative rules, borders, and tiny icons (Story E2.1)."""
    x0, y0, x1, y1 = bbox
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    area = w * h

    if w < cfg.extract.min_figure_width_pt:
        return False, f"width_too_small ({w:.1f} < {cfg.extract.min_figure_width_pt})"
    if h < cfg.extract.min_figure_height_pt:
        return False, f"height_too_small ({h:.1f} < {cfg.extract.min_figure_height_pt})"
    if area < cfg.extract.min_figure_area_pt2:
        return False, f"area_too_small ({area:.1f} < {cfg.extract.min_figure_area_pt2})"

    aspect = w / max(1.0, h)
    if aspect < cfg.extract.min_aspect_ratio or aspect > cfg.extract.max_aspect_ratio:
        return False, f"aspect_ratio_out_of_bounds ({aspect:.2f})"

    return True, "ok"


def extract_figure_units(
    regions: list[RawRegion],
    cfg: AppConfig,
) -> tuple[list[Unit], list[dict]]:
    """Extracts validated figure crops from figure regions and writes `.png` files to disk."""
    out_dir = Path(cfg.project.artifacts_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_units: list[Unit] = []
    filtered_out: list[dict] = []
    per_page_seq: dict[tuple[str, int], int] = {}

    # Group regions by source PDF so we open each PDF once
    by_pdf: dict[str, list[RawRegion]] = {}
    for reg in regions:
        if reg.region_type == "figure":
            by_pdf.setdefault(reg.source_path, []).append(reg)

    for pdf_path, fig_regions in by_pdf.items():
        with fitz.open(pdf_path) as doc:
            for reg in fig_regions:
                valid, reason = is_valid_figure_bbox(reg.bbox, cfg)
                if not valid:
                    filtered_out.append(
                        {
                            "doc_id": reg.doc_id,
                            "page": reg.page,
                            "bbox": reg.bbox,
                            "reason": reason,
                        }
                    )
                    continue

                page = doc[reg.page - 1]
                clip_rect = fitz.Rect(*reg.bbox) & page.rect
                if clip_rect.is_empty:
                    filtered_out.append(
                        {
                            "doc_id": reg.doc_id,
                            "page": reg.page,
                            "bbox": reg.bbox,
                            "reason": "bbox_outside_page",
                        }
                    )
                    continue

                key = (reg.doc_id, reg.page)
                seq = per_page_seq.get(key, 0) + 1
                per_page_seq[key] = seq
                unit_id = f"{reg.doc_id}__p{reg.page:04d}__figure_{seq:03d}"

                pix = page.get_pixmap(clip=clip_rect, dpi=cfg.extract.figure_render_dpi)
                img_path = out_dir / f"{unit_id}.png"
                pix.save(str(img_path))

                prov = Provenance(
                    doc_id=reg.doc_id,
                    page=reg.page,
                    unit_id=unit_id,
                    bbox=reg.bbox,
                    source_path=reg.source_path,
                    heading_context=reg.heading_context,
                    nearby_caption=reg.nearby_caption,
                    model_variant=reg.model_variant,
                    extraction_meta={
                        "figure_source": reg.meta.get("source", "raster"),
                        "figure_labels": reg.meta.get("figure_labels", []),
                        "xref": reg.meta.get("xref", 0),
                        "dpi": cfg.extract.figure_render_dpi,
                        "width_px": pix.width,
                        "height_px": pix.height,
                    },
                )
                figure_units.append(
                    Unit(
                        unit_id=unit_id,
                        type="figure",
                        search_text="",  # Story E2.1/E3.1: Enrichment fills search_text
                        artefact=str(img_path),
                        provenance=prov,
                    )
                )

    return figure_units, filtered_out
