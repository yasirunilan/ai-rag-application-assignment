"""Structural Parser & Content-Type Router (Person 2 · Story E1.1).

Routes each PDF page into:
- `text` blocks (with heading context and model variant)
- `figure` regions (with bbox, nearby caption, and heading context)
- `table` regions (with bbox, nearby caption, and heading context)
- `scanned_page` (routed through Person 1's OCR Quality Gate)

Also documents and measures what a naive flat-text loader (`page.get_text("text")`) drops.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import fitz
import pdfplumber

from mmrag.config import AppConfig
from mmrag.ingest.ocr_gate import evaluate_ocr_quality_gate, is_scanned_page, run_ocr_on_page
from mmrag.schemas import PageClassification, RawRegion


HEADING_PATTERN = re.compile(
    r"^(Chapter\s+\d+|Section\s+\d+(?:\.\d+)?|\d+\.\d+\s+[A-Z])",
    re.IGNORECASE,
)
FIGURE_CAPTION_PATTERN = re.compile(
    r"^(Figure|Fig\.|Diagram|Schematic)\s+\d+(?:[-.]\d+)?",
    re.IGNORECASE,
)
TABLE_CAPTION_PATTERN = re.compile(
    r"^(Table|Tbl\.)\s+\d+(?:[-.]\d+)?",
    re.IGNORECASE,
)
# Model token must contain a digit so prose like "this model is" is not mistaken for a variant.
MODEL_VARIANT_PATTERN = re.compile(
    r"\b(Model\s+(?=[A-Z0-9-]*\d)[A-Z0-9-]+|7[0-8]7-\d{3}|A3[0-8]\d(?:-\d{3})?)\b",
    re.IGNORECASE,
)
CHAPTER_HEADING_PATTERN = re.compile(r"^Chapter\s+\d+", re.IGNORECASE)

# Vector drawing clusters covering more of the page than this are page frames/backgrounds, not figures.
MAX_VECTOR_FIGURE_PAGE_COVERAGE = 0.80
# Raster/vector figure candidates overlapping an existing candidate beyond this IoU are duplicates.
DUPLICATE_FIGURE_IOU = 0.90


def explain_naive_loader_drops(pdf_path: str | Path) -> dict[str, Any]:
    """Story E1.1 Requirement: Document and quantify what a naive text loader drops.

    A standard first-reach loader (`pypdf` / `fitz.Page.get_text("text")`):
    1. Completely drops embedded raster figures and vector wiring schematics (0 bytes of image data).
    2. Collapses 2D table row/column cell boundaries into unaligned whitespace strings.
    3. Silently emits empty strings on scanned pages without flagging them for OCR review.
    """
    path = Path(pdf_path)
    dropped_figures = 0
    collapsed_tables = 0
    silent_scanned_pages = 0

    with fitz.open(str(path)) as doc, pdfplumber.open(str(path)) as pdoc:
        for idx, page in enumerate(doc):
            scanned, _, _ = is_scanned_page(page)
            if scanned:
                silent_scanned_pages += 1
                continue
            imgs = [
                info
                for info in page.get_image_info(xrefs=True)
                if (info["bbox"][2] - info["bbox"][0]) >= 60
                and (info["bbox"][3] - info["bbox"][1]) >= 50
            ]
            dropped_figures += len(imgs)
            p_tables = pdoc.pages[idx].find_tables()
            collapsed_tables += len(p_tables)

    return {
        "naive_loader": "PyMuPDF page.get_text('text') / pypdf extract_text()",
        "what_it_drops": [
            "All embedded figures and vector wiring diagrams (lost completely; not retrievable)",
            "All 2D table row/column cell boundaries (flattened into unaligned strings)",
            "All scanned image-only pages (silently skipped with 0 text chars)",
        ],
        "measured_on_file": str(path.name),
        "figures_dropped_count": dropped_figures,
        "tables_flattened_count": collapsed_tables,
        "scanned_pages_silently_lost": silent_scanned_pages,
    }


def _rect_overlap_ratio(inner: fitz.Rect, outer: fitz.Rect) -> float:
    inter = inner & outer
    if inter.is_empty:
        return 0.0
    inner_area = max(1.0, inner.width * inner.height)
    return (inter.width * inter.height) / inner_area


def _rect_iou(a: fitz.Rect, b: fitz.Rect) -> float:
    inter = a & b
    if inter.is_empty:
        return 0.0
    inter_area = inter.width * inter.height
    union_area = a.width * a.height + b.width * b.height - inter_area
    return inter_area / max(1.0, union_area)


def _horizontal_overlap(a: fitz.Rect, b: fitz.Rect) -> bool:
    return min(a.x1, b.x1) > max(a.x0, b.x0)


def _extract_model_variant(text: str) -> Optional[str]:
    m = MODEL_VARIANT_PATTERN.search(text)
    return m.group(1) if m else None


def _heading_level(line: str) -> int:
    """1 = chapter-level heading, 2 = section-level heading."""
    return 1 if CHAPTER_HEADING_PATTERN.match(line) else 2


def _join_heading_state(state: dict[int, str]) -> Optional[str]:
    parts = [state[lvl] for lvl in sorted(state)]
    return " — ".join(parts) if parts else None


def _heading_at(
    y: float,
    carried_state: dict[int, str],
    page_headings: list[dict[str, Any]],
) -> Optional[str]:
    """Hierarchical heading context in force at vertical position `y` on the page."""
    state = dict(carried_state)
    for h in page_headings:
        if h["y0"] > y:
            break
        if h["level"] == 1:
            state = {}
        state[h["level"]] = h["text"]
    return _join_heading_state(state)


def _find_tables(pdf_page: Any, fitz_page: fitz.Page, backend: str) -> list[tuple[tuple[float, ...], list]]:
    """Returns `(bbox, rows)` per detected table using the configured backend (Story E1.1 / E2.2)."""
    if backend == "pymupdf":
        found = fitz_page.find_tables().tables
    else:
        found = pdf_page.find_tables()
    return [(tuple(t.bbox), t.extract() or []) for t in found]


def _match_caption(
    target: fitz.Rect,
    captions: list[dict[str, Any]],
    margin: float,
) -> Optional[str]:
    """Nearest caption above or below `target` that shares its column, within `margin` points."""
    best: Optional[str] = None
    best_dist = 9999.0
    for cap in captions:
        c_rect: fitz.Rect = cap["rect"]
        if not _horizontal_overlap(target, c_rect):
            continue
        dist = min(abs(target.y1 - c_rect.y0), abs(c_rect.y1 - target.y0))
        if dist <= margin and dist < best_dist:
            best_dist = dist
            best = cap["text"]
    return best


def parse_and_route_pdf(
    pdf_path: str | Path,
    cfg: AppConfig,
    doc_id: Optional[str] = None,
) -> tuple[list[PageClassification], list[RawRegion]]:
    """Stage 1 Structural Parser: classifies every page and routes regions by content type."""
    path = Path(pdf_path)
    resolved_doc_id = doc_id or path.stem

    classifications: list[PageClassification] = []
    regions: list[RawRegion] = []
    # Heading hierarchy carried from previous pages so continuation pages keep their section context
    carried_headings: dict[int, str] = {}

    with fitz.open(str(path)) as doc, pdfplumber.open(str(path)) as pdoc:
        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1  # 1-indexed provenance page number
            scanned, char_count, img_cov = is_scanned_page(
                page,
                max_text_chars=cfg.ingest.scanned_max_text_chars,
                min_image_coverage=cfg.ingest.scanned_min_image_coverage,
            )

            if scanned:
                ocr_text, mean_conf = run_ocr_on_page(page, engine=cfg.ingest.ocr_engine)
                passed, garbled_ratio, alnum_ratio, reasons = evaluate_ocr_quality_gate(
                    doc_id=resolved_doc_id,
                    page_num=page_num,
                    source_path=str(path),
                    ocr_text=ocr_text,
                    mean_confidence=mean_conf,
                    cfg=cfg,
                    append_to_queue=True,
                )
                classifications.append(
                    PageClassification(
                        doc_id=resolved_doc_id,
                        page=page_num,
                        is_scanned=True,
                        text_char_count=char_count,
                        image_coverage_ratio=round(img_cov, 4),
                        ocr_attempted=True,
                        ocr_passed=passed,
                        ocr_mean_confidence=round(mean_conf, 2),
                        ocr_garbled_ratio=round(garbled_ratio, 4),
                        ocr_alnum_token_ratio=round(alnum_ratio, 4),
                        route_decision="ocr_indexed" if passed else "ocr_review_queue",
                    )
                )
                if passed and ocr_text.strip():
                    regions.append(
                        RawRegion(
                            doc_id=resolved_doc_id,
                            page=page_num,
                            region_type="text",
                            bbox=(0.0, 0.0, float(page.rect.width), float(page.rect.height)),
                            reading_order=1,
                            raw_text=ocr_text.strip(),
                            heading_context=_join_heading_state(carried_headings) or f"Scanned Page {page_num}",
                            source_path=str(path),
                            meta={"from_ocr": True, "ocr_confidence": mean_conf},
                        )
                    )
                continue

            # Digital Page Route
            classifications.append(
                PageClassification(
                    doc_id=resolved_doc_id,
                    page=page_num,
                    is_scanned=False,
                    text_char_count=char_count,
                    image_coverage_ratio=round(img_cov, 4),
                    ocr_attempted=False,
                    route_decision="digital_parse",
                )
            )

            # 1. Collect raw text blocks (with per-line boxes) sorted in top-to-bottom reading order
            text_blocks: list[dict[str, Any]] = []
            for b in page.get_text("dict")["blocks"]:
                if b.get("type") != 0:
                    continue
                lines = [
                    {"rect": fitz.Rect(ln["bbox"]), "text": "".join(sp["text"] for sp in ln["spans"]).strip()}
                    for ln in b.get("lines", [])
                ]
                lines = [ln for ln in lines if ln["text"]]
                if not lines:
                    continue
                x0, y0, x1, y1 = b["bbox"]
                text_blocks.append(
                    {
                        "rect": fitz.Rect(x0, y0, x1, y1),
                        "text": "\n".join(ln["text"] for ln in lines),
                        "lines": lines,
                        "y0": float(y0),
                        "x0": float(x0),
                    }
                )
            text_blocks.sort(key=lambda item: (round(item["y0"], 1), round(item["x0"], 1)))

            # Identify page-level headings, captions, and model variants
            page_headings: list[dict[str, Any]] = []
            figure_captions: list[dict[str, Any]] = []
            table_captions: list[dict[str, Any]] = []
            page_model_variant: Optional[str] = None

            for tb in text_blocks:
                single_line = " ".join(tb["text"].split())
                first_line = " ".join(tb["lines"][0]["text"].split())
                if not page_model_variant:
                    page_model_variant = _extract_model_variant(single_line)
                if HEADING_PATTERN.match(first_line):
                    page_headings.append(
                        {"y0": tb["y0"], "text": first_line, "level": _heading_level(first_line)}
                    )
                elif FIGURE_CAPTION_PATTERN.match(first_line) or TABLE_CAPTION_PATTERN.match(first_line):
                    # One block can hold several captions (e.g. side-by-side columns on one baseline):
                    # start a new caption at every caption-pattern line, folding continuation lines in.
                    tb["is_caption"] = True
                    segments: list[dict[str, Any]] = []
                    for ln in tb["lines"]:
                        ln_text = " ".join(ln["text"].split())
                        if FIGURE_CAPTION_PATTERN.match(ln_text) or TABLE_CAPTION_PATTERN.match(ln_text):
                            segments.append({"rect": fitz.Rect(ln["rect"]), "text": ln_text})
                        else:
                            segments[-1]["rect"] |= ln["rect"]
                            segments[-1]["text"] += f" {ln_text}"
                    for seg in segments:
                        if FIGURE_CAPTION_PATTERN.match(seg["text"]):
                            figure_captions.append(seg)
                        else:
                            table_captions.append(seg)

            page_heading_texts = [h["text"] for h in page_headings]
            page_heading_context = (
                " — ".join(page_heading_texts) if page_heading_texts else _join_heading_state(carried_headings)
            )

            def region_context(rect: fitz.Rect, caption: Optional[str]) -> tuple[Optional[str], Optional[str]]:
                heading = _heading_at(rect.y0, carried_headings, page_headings)
                variant = (
                    _extract_model_variant(caption or "")
                    or _extract_model_variant(heading or "")
                    or page_model_variant
                )
                return heading, variant

            # 2. Detect Table Regions via configured backend (pdfplumber default, PyMuPDF optional)
            p_page = pdoc.pages[page_idx]
            found_tables = _find_tables(p_page, page, cfg.extract.table_backend)
            table_rects: list[fitz.Rect] = []
            page_candidates: list[dict[str, Any]] = []

            for t_idx, (t_bbox, extracted_rows) in enumerate(found_tables):
                tx0, ty0, tx1, ty1 = t_bbox
                t_rect = fitz.Rect(tx0, ty0, tx1, ty1)
                table_rects.append(t_rect)

                best_cap = _match_caption(t_rect, table_captions, cfg.extract.caption_search_margin_pt)
                heading, variant = region_context(t_rect, best_cap)

                clean_rows: list[list[str]] = []
                for row in extracted_rows:
                    if not row:
                        continue
                    cleaned = [" ".join(str(cell or "").split()) for cell in row]
                    if any(cleaned):
                        clean_rows.append(cleaned)

                page_candidates.append(
                    {
                        "y0": float(ty0),
                        "x0": float(tx0),
                        "region_type": "table",
                        "bbox": (float(tx0), float(ty0), float(tx1), float(ty1)),
                        "raw_text": best_cap or "",
                        "heading_context": heading,
                        "nearby_caption": best_cap,
                        "model_variant": variant,
                        "meta": {"rows": clean_rows, "table_index": t_idx},
                    }
                )

            # 3. Detect Figure Regions: embedded raster images + clustered vector drawings (schematics).
            # All candidates go to Stage 2 for size/aspect filtering of rules and icons.
            figure_candidates: list[tuple[fitz.Rect, dict[str, Any]]] = []
            for img_idx, img_info in enumerate(page.get_image_info(xrefs=True)):
                ibbox = img_info.get("bbox")
                if not ibbox:
                    continue
                i_rect = fitz.Rect(ibbox)
                if any(_rect_iou(i_rect, prev) >= DUPLICATE_FIGURE_IOU for prev, _ in figure_candidates):
                    continue
                figure_candidates.append(
                    (i_rect, {"source": "raster", "xref": img_info.get("xref", 0), "figure_index": img_idx})
                )

            page_area = max(1.0, page.rect.width * page.rect.height)
            for v_rect in page.cluster_drawings():
                v_rect = fitz.Rect(v_rect)
                if (v_rect.width * v_rect.height) / page_area > MAX_VECTOR_FIGURE_PAGE_COVERAGE:
                    continue
                # Table grid lines are drawings too; those are already routed as tables
                if any(_rect_overlap_ratio(v_rect, tr) > 0.50 for tr in table_rects):
                    continue
                if any(_rect_iou(v_rect, prev) >= DUPLICATE_FIGURE_IOU for prev, _ in figure_candidates):
                    continue
                # Leader lines / callouts drawn over a raster belong to that image, not a second figure
                if any(
                    m["source"] == "raster" and _rect_overlap_ratio(v_rect, prev) > 0.50
                    for prev, m in figure_candidates
                ):
                    continue
                figure_candidates.append(
                    (v_rect, {"source": "vector", "xref": 0, "figure_index": len(figure_candidates)})
                )

            for f_rect, f_meta in figure_candidates:
                best_fcap = _match_caption(f_rect, figure_captions, cfg.extract.caption_search_margin_pt)
                heading, variant = region_context(f_rect, best_fcap)
                labels = [
                    " ".join(tb["text"].split())
                    for tb in text_blocks
                    if _rect_overlap_ratio(tb["rect"], f_rect) > 0.50
                ]
                if labels:
                    f_meta["figure_labels"] = labels

                page_candidates.append(
                    {
                        "y0": float(f_rect.y0),
                        "x0": float(f_rect.x0),
                        "region_type": "figure",
                        "bbox": (float(f_rect.x0), float(f_rect.y0), float(f_rect.x1), float(f_rect.y1)),
                        "raw_text": best_fcap or "",
                        "heading_context": heading,
                        "nearby_caption": best_fcap,
                        "model_variant": variant,
                        "meta": f_meta,
                    }
                )

            # 4. Collect Prose Text Regions (excluding blocks inside table bboxes or standalone figure/table captions)
            prose_lines: list[str] = []
            prose_rects: list[fitz.Rect] = []

            for tb in text_blocks:
                tb_rect: fitz.Rect = tb["rect"]
                single_line = " ".join(tb["text"].split())
                if any(_rect_overlap_ratio(tb_rect, tr) > 0.50 for tr in table_rects):
                    continue
                if tb.get("is_caption"):
                    continue
                prose_lines.append(single_line)
                prose_rects.append(tb_rect)

            if prose_lines:
                union_rect = prose_rects[0]
                for r in prose_rects[1:]:
                    union_rect |= r
                page_candidates.append(
                    {
                        "y0": float(union_rect.y0),
                        "x0": float(union_rect.x0),
                        "region_type": "text",
                        "bbox": (
                            float(union_rect.x0),
                            float(union_rect.y0),
                            float(union_rect.x1),
                            float(union_rect.y1),
                        ),
                        "raw_text": "\n".join(prose_lines),
                        "heading_context": page_heading_context,
                        "nearby_caption": None,
                        "model_variant": page_model_variant,
                        "meta": {},
                    }
                )

            for h in page_headings:
                if h["level"] == 1:
                    carried_headings = {}
                carried_headings[h["level"]] = h["text"]

            # Sort all regions on the page in deterministic reading order
            page_candidates.sort(key=lambda c: (round(c["y0"], 1), round(c["x0"], 1)))
            for order_idx, cand in enumerate(page_candidates, start=1):
                regions.append(
                    RawRegion(
                        doc_id=resolved_doc_id,
                        page=page_num,
                        region_type=cand["region_type"],
                        bbox=cand["bbox"],
                        reading_order=order_idx,
                        raw_text=cand["raw_text"],
                        heading_context=cand["heading_context"],
                        nearby_caption=cand["nearby_caption"],
                        model_variant=cand["model_variant"],
                        source_path=str(path),
                        meta=cand["meta"],
                    )
                )

    return classifications, regions
