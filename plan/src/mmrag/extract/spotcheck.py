"""Provenance Spot-Check Validator (Person 2 · Story E2.3).

Samples five (or N) extracted units across modalities and confirms each one:
1. Carries a non-empty `doc_id`, valid 1-indexed `page`, and deterministic `unit_id`.
2. Resolves to an actual page within the source PDF on disk.
3. Has a non-empty artefact (and existing `.png` file on disk for figures).
4. Has a `unit_id` that encodes the same `doc_id` and page it cites.
5. Has a bbox that lies inside that page, and a nearby caption that actually appears on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz

from mmrag.schemas import TableArtefact, Unit

BBOX_TOLERANCE_PT = 1.0


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _check_against_page(u: Unit, page: fitz.Page) -> str:
    """Returns "ok" or the first reason the unit does not match the physical page it cites."""
    if u.provenance.bbox is not None:
        x0, y0, x1, y1 = u.provenance.bbox
        r = page.rect
        if (
            x0 < r.x0 - BBOX_TOLERANCE_PT
            or y0 < r.y0 - BBOX_TOLERANCE_PT
            or x1 > r.x1 + BBOX_TOLERANCE_PT
            or y1 > r.y1 + BBOX_TOLERANCE_PT
        ):
            return f"bbox_outside_page ({u.provenance.bbox} not within {tuple(r)})"
    if u.provenance.nearby_caption:
        if _normalize_ws(u.provenance.nearby_caption) not in _normalize_ws(page.get_text("text")):
            return "caption_not_found_on_cited_page"
    return "ok"


def run_provenance_spotcheck(units: list[Unit], sample_size: int = 5) -> dict[str, Any]:
    """Samples up to `sample_size` units (stratified by type) and verifies provenance integrity."""
    if not units:
        return {"passed": False, "sampled_count": 0, "failures": ["No units available to spot-check."], "samples": []}

    # Stratified deterministic selection across figure, table, and text units
    figures = [u for u in units if u.type == "figure"]
    tables = [u for u in units if u.type == "table"]
    texts = [u for u in units if u.type == "text"]

    ordered_pool: list[Unit] = []
    max_len = max(len(figures), len(tables), len(texts))
    for i in range(max_len):
        if i < len(figures):
            ordered_pool.append(figures[i])
        if i < len(tables):
            ordered_pool.append(tables[i])
        if i < len(texts):
            ordered_pool.append(texts[i])

    sampled = ordered_pool[: min(sample_size, len(ordered_pool))]
    failures: list[str] = []
    sample_reports: list[dict[str, Any]] = []

    for u in sampled:
        unit_ok = u.has_valid_provenance(verify_files=True)
        pdf_page_ok = False
        reason = "ok"

        src_path = Path(u.provenance.source_path) if u.provenance.source_path else None
        expected_prefix = f"{u.provenance.doc_id}__p{(u.provenance.page or 0):04d}__"
        if not unit_ok:
            reason = "invalid_provenance_or_missing_artefact"
        elif not u.unit_id.startswith(expected_prefix) or u.provenance.unit_id != u.unit_id:
            unit_ok = False
            reason = f"unit_id_mismatch ({u.unit_id} does not encode {expected_prefix})"
        elif src_path and not src_path.exists():
            reason = f"source_pdf_missing ({src_path})"
        elif src_path:
            with fitz.open(str(src_path)) as doc:
                if u.provenance.page is not None and 1 <= u.provenance.page <= len(doc):
                    reason = _check_against_page(u, doc[u.provenance.page - 1])
                    pdf_page_ok = reason == "ok"
                else:
                    reason = f"page_out_of_range ({u.provenance.page} not in 1..{len(doc)})"
        else:
            # If unit was constructed in-memory without a physical PDF path, unit_ok is sufficient
            pdf_page_ok = unit_ok

        passed = unit_ok and pdf_page_ok
        if not passed:
            failures.append(f"{u.unit_id}: {reason}")

        artefact_preview = (
            f"Table({len(u.artefact.headers)} cols x {len(u.artefact.rows)} rows)"
            if isinstance(u.artefact, TableArtefact)
            else str(u.artefact)[:90]
        )
        sample_reports.append(
            {
                "unit_id": u.unit_id,
                "type": u.type,
                "doc_id": u.provenance.doc_id,
                "page": u.provenance.page,
                "citation": u.provenance.citation,
                "bbox": u.provenance.bbox,
                "heading_context": u.provenance.heading_context,
                "nearby_caption": u.provenance.nearby_caption,
                "model_variant": u.provenance.model_variant,
                "artefact_preview": artefact_preview,
                "passed": passed,
                "status": reason,
            }
        )

    return {
        "passed": len(failures) == 0 and len(sampled) > 0,
        "sampled_count": len(sampled),
        "failures": failures,
        "samples": sample_reports,
    }
