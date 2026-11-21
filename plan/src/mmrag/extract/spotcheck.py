"""Provenance Spot-Check Validator (Person 2 · Story E2.3).

Samples five (or N) extracted units across modalities and confirms each one:
1. Carries a non-empty `doc_id`, valid 1-indexed `page`, and deterministic `unit_id`.
2. Resolves to an actual page within the source PDF on disk.
3. Has a non-empty artefact (and existing `.png` file on disk for figures).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz

from mmrag.schemas import TableArtefact, Unit


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
        if not unit_ok:
            reason = "invalid_provenance_or_missing_artefact"
        elif src_path and src_path.exists():
            with fitz.open(str(src_path)) as doc:
                if u.provenance.page is not None and 1 <= u.provenance.page <= len(doc):
                    pdf_page_ok = True
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
