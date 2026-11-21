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
MODEL_VARIANT_PATTERN = re.compile(
    r"\b(Model\s+[A-Z0-9-]+|737-\d{3}|A320(?:-\d{3})?)\b",
    re.IGNORECASE,
)


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


def _extract_model_variant(text: str) -> Optional[str]:
    m = MODEL_VARIANT_PATTERN.search(text)
    return m.group(1) if m else None


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
                            heading_context=f"Scanned Page {page_num}",
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

            # 1. Collect raw text blocks sorted in top-to-bottom reading order
            raw_blocks = page.get_text("blocks")
            text_blocks: list[dict[str, Any]] = []
            for b in raw_blocks:
                x0, y0, x1, y1, txt, block_no, block_type = b
                clean = str(txt).strip()
                if block_type == 0 and clean:
                    text_blocks.append(
                        {
                            "rect": fitz.Rect(x0, y0, x1, y1),
                            "text": clean,
                            "y0": float(y0),
                            "x0": float(x0),
                        }
                    )
            text_blocks.sort(key=lambda item: (round(item["y0"], 1), round(item["x0"], 1)))

            # Identify page-level headings, captions, and model variants
            page_headings: list[str] = []
            figure_captions: list[dict[str, Any]] = []
            table_captions: list[dict[str, Any]] = []
            page_model_variant: Optional[str] = None

            for tb in text_blocks:
                single_line = " ".join(tb["text"].split())
                if not page_model_variant:
                    page_model_variant = _extract_model_variant(single_line)
                if HEADING_PATTERN.match(single_line):
                    page_headings.append(single_line)
                elif FIGURE_CAPTION_PATTERN.match(single_line):
                    figure_captions.append({"rect": tb["rect"], "text": single_line})
                elif TABLE_CAPTION_PATTERN.match(single_line):
                    table_captions.append({"rect": tb["rect"], "text": single_line})

            active_heading = " — ".join(page_headings) if page_headings else None

            # 2. Detect Table Regions via pdfplumber (or PyMuPDF fallback)
            p_page = pdoc.pages[page_idx]
            found_tables = p_page.find_tables()
            table_rects: list[fitz.Rect] = []
            page_candidates: list[dict[str, Any]] = []

            for t_idx, tbl in enumerate(found_tables):
                tx0, ty0, tx1, ty1 = tbl.bbox
                t_rect = fitz.Rect(tx0, ty0, tx1, ty1)
                table_rects.append(t_rect)

                # Find nearest table caption within margin above or below
                best_cap: Optional[str] = None
                best_dist = 9999.0
                for cap in table_captions:
                    c_rect: fitz.Rect = cap["rect"]
                    dist = min(abs(t_rect.y0 - c_rect.y1), abs(c_rect.y0 - t_rect.y1))
                    if dist <= cfg.extract.caption_search_margin_pt and dist < best_dist:
                        best_dist = dist
                        best_cap = cap["text"]

                extracted_rows = tbl.extract() or []
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
                        "heading_context": active_heading,
                        "nearby_caption": best_cap,
                        "model_variant": _extract_model_variant(best_cap or "") or page_model_variant,
                        "meta": {"rows": clean_rows, "table_index": t_idx},
                    }
                )

            # 3. Detect Figure Regions (pass all candidate image regions to Stage 2 for size/aspect filtering)
            for img_idx, img_info in enumerate(page.get_image_info(xrefs=True)):
                ibbox = img_info.get("bbox")
                if not ibbox:
                    continue
                ix0, iy0, ix1, iy1 = ibbox
                i_rect = fitz.Rect(ix0, iy0, ix1, iy1)

                # Match closest figure caption within search margin
                best_fcap: Optional[str] = None
                best_fdist = 9999.0
                for cap in figure_captions:
                    c_rect = cap["rect"]
                    dist = min(abs(i_rect.y1 - c_rect.y0), abs(c_rect.y1 - i_rect.y0))
                    if dist <= cfg.extract.caption_search_margin_pt and dist < best_fdist:
                        best_fdist = dist
                        best_fcap = cap["text"]

                page_candidates.append(
                    {
                        "y0": float(iy0),
                        "x0": float(ix0),
                        "region_type": "figure",
                        "bbox": (float(ix0), float(iy0), float(ix1), float(iy1)),
                        "raw_text": best_fcap or "",
                        "heading_context": active_heading,
                        "nearby_caption": best_fcap,
                        "model_variant": _extract_model_variant(best_fcap or "") or page_model_variant,
                        "meta": {"xref": img_info.get("xref", 0), "figure_index": img_idx},
                    }
                )

            # 4. Collect Prose Text Regions (excluding blocks inside table bboxes or standalone figure/table captions)
            caption_texts = {c["text"] for c in figure_captions} | {c["text"] for c in table_captions}
            prose_lines: list[str] = []
            prose_rects: list[fitz.Rect] = []

            for tb in text_blocks:
                tb_rect: fitz.Rect = tb["rect"]
                single_line = " ".join(tb["text"].split())
                if any(_rect_overlap_ratio(tb_rect, tr) > 0.50 for tr in table_rects):
                    continue
                if single_line in caption_texts:
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
                        "heading_context": active_heading,
                        "nearby_caption": None,
                        "model_variant": page_model_variant,
                        "meta": {},
                    }
                )

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
