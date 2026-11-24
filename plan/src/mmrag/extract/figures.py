"""Figure Extraction & Rule/Icon Filtering (Person 2 · Story E2.1).

Exports the original embedded raster where possible (else renders the bbox at
`figure_render_dpi`), stamps `doc -> page -> figure_id`, bbox, and nearby caption text
on the Unit, and filters out decorative rules/icons below the minimum geometry gate.
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


TRANSFORM_TOL = 1e-3
# Fraction of the figure bbox a placement may differ by and still count as the whole image.
PLACEMENT_TOL = 0.02
# Drawings this close to the figure edge are treated as frames/borders, not overlays.
OVERLAY_EDGE_MARGIN_PT = 2.0


def _find_placement(page: fitz.Page, xref: int, bbox: fitz.Rect) -> dict | None:
    """The `get_image_info` entry for this xref drawn at this bbox (an xref can be placed several times)."""
    tol = PLACEMENT_TOL * max(bbox.width, bbox.height)
    for info in page.get_image_info(xrefs=True):
        if info.get("xref") != xref:
            continue
        r = fitz.Rect(info["bbox"])
        if all(abs(a - b) <= tol for a, b in zip(r, bbox)):
            return info
    return None


def _has_overlay(page: fitz.Page, img_rect: fitz.Rect) -> str | None:
    """Text or vector strokes drawn on top of the image would be lost by exporting the raw raster."""
    inner = fitz.Rect(img_rect) + (OVERLAY_EDGE_MARGIN_PT, OVERLAY_EDGE_MARGIN_PT, -OVERLAY_EDGE_MARGIN_PT, -OVERLAY_EDGE_MARGIN_PT)
    if inner.is_empty:
        return None
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        if text.strip() and fitz.Rect(x0, y0, x1, y1).intersects(inner):
            return "text_overlay"
    for d in page.get_drawings():
        d_rect = fitz.Rect(d["rect"])
        # A frame drawn around the image only touches it at the edges
        if d_rect.intersects(inner) and not d_rect.contains(img_rect):
            return "vector_overlay"
    return None


def _try_embedded_raster(
    doc: fitz.Document,
    page: fitz.Page,
    reg: RawRegion,
    clip_rect: fitz.Rect,
) -> tuple[fitz.Pixmap | None, str | None]:
    """Returns the original embedded image as an RGB(A) pixmap, or `(None, reason)` to render the bbox instead.

    Falls back whenever the raw raster would not match what the page shows: vector figures,
    images placed rotated/flipped or cropped by the page edge, and labels or leader lines drawn on top.
    """
    xref = int(reg.meta.get("xref", 0) or 0)
    if reg.meta.get("source", "raster") != "raster" or xref <= 0:
        return None, "not_embedded_raster"
    if clip_rect != fitz.Rect(*reg.bbox):
        return None, "cropped_by_page"

    placement = _find_placement(page, xref, clip_rect)
    if placement is None:
        return None, "placement_not_found"
    a, b, c, d, _, _ = placement.get("transform", (1, 0, 0, 1, 0, 0))
    if abs(b) > TRANSFORM_TOL or abs(c) > TRANSFORM_TOL or a <= 0 or d <= 0:
        return None, "rotated_or_flipped"

    overlay = _has_overlay(page, clip_rect)
    if overlay:
        return None, overlay

    try:
        pix = fitz.Pixmap(doc, xref)
        smask = doc.extract_image(xref).get("smask", 0)
        if smask:
            pix = fitz.Pixmap(pix, fitz.Pixmap(doc, smask))
        if pix.colorspace is None:  # stencil/image mask: no colour data to export
            return None, "no_colorspace"
        if pix.colorspace.n not in (1, 3):  # CMYK, Lab, indexed... -> PNG-safe RGB
            pix = fitz.Pixmap(fitz.csRGB, pix)
    except Exception as exc:  # corrupt or unsupported image stream
        return None, f"extract_failed ({type(exc).__name__})"
    return pix, None


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

                # Story E2.1: export the embedded raster where possible, else render the bbox
                pix, fallback_reason = _try_embedded_raster(doc, page, reg, clip_rect)
                if pix is not None:
                    export_method = "embedded_raster"
                    effective_dpi = round(pix.width / max(1e-6, clip_rect.width / 72.0))
                else:
                    export_method = "rendered_bbox"
                    pix = page.get_pixmap(clip=clip_rect, dpi=cfg.extract.figure_render_dpi)
                    effective_dpi = cfg.extract.figure_render_dpi
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
                        "export_method": export_method,
                        "export_fallback_reason": fallback_reason,
                        "dpi": effective_dpi,
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
