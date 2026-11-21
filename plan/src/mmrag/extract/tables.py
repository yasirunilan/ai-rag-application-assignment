"""Structured Table & Text Unit Extraction (Person 2 · Story E2.2).

Extracts each table preserving rows/cols into `TableArtefact` with full provenance
(`doc_id -> page -> unit_id`), leaving `search_text = ""` for Stage 3 Enrichment.
"""

from __future__ import annotations

from mmrag.config import AppConfig
from mmrag.schemas import Provenance, RawRegion, TableArtefact, Unit


def extract_table_units(regions: list[RawRegion], cfg: AppConfig) -> list[Unit]:
    """Converts routed table regions into structured `Unit(type='table')` objects (Story E2.2)."""
    table_units: list[Unit] = []
    per_page_seq: dict[tuple[str, int], int] = {}

    for reg in regions:
        if reg.region_type != "table":
            continue
        rows: list[list[str]] = reg.meta.get("rows", [])
        if not rows:
            continue

        key = (reg.doc_id, reg.page)
        seq = per_page_seq.get(key, 0) + 1
        per_page_seq[key] = seq
        unit_id = f"{reg.doc_id}__p{reg.page:04d}__table_{seq:03d}"

        headers = rows[0] if len(rows) >= 1 else []
        data_rows = rows[1:] if len(rows) > 1 else rows

        prov = Provenance(
            doc_id=reg.doc_id,
            page=reg.page,
            unit_id=unit_id,
            bbox=reg.bbox,
            source_path=reg.source_path,
            heading_context=reg.heading_context,
            nearby_caption=reg.nearby_caption,
            model_variant=reg.model_variant,
            extraction_meta={"num_rows": len(data_rows), "num_cols": len(headers)},
        )

        if cfg.extract.flatten_tables:
            # E7.3 Failure Mode 2 reproduction ("Table answers are mush" when flattened to a string)
            flat_lines = [" ".join(r) for r in rows]
            artefact_val = " ".join(flat_lines)
        else:
            artefact_val = TableArtefact(
                headers=headers,
                rows=data_rows,
                caption=reg.nearby_caption,
            )

        table_units.append(
            Unit(
                unit_id=unit_id,
                type="table",
                search_text="",  # Story E2.2: Leave search_text empty here; E3.2 fills it
                artefact=artefact_val,
                provenance=prov,
            )
        )

    return table_units


def extract_text_units(regions: list[RawRegion]) -> list[Unit]:
    """Converts routed digital/OCR text regions into `Unit(type='text')` objects with provenance."""
    text_units: list[Unit] = []
    per_page_seq: dict[tuple[str, int], int] = {}

    for reg in regions:
        if reg.region_type != "text" or not reg.raw_text.strip():
            continue

        key = (reg.doc_id, reg.page)
        seq = per_page_seq.get(key, 0) + 1
        per_page_seq[key] = seq
        unit_id = f"{reg.doc_id}__p{reg.page:04d}__text_{seq:03d}"

        clean_span = reg.raw_text.strip()
        heading_prefix = f"[{reg.heading_context}] " if reg.heading_context else ""
        search_text = f"{heading_prefix}{clean_span}"

        prov = Provenance(
            doc_id=reg.doc_id,
            page=reg.page,
            unit_id=unit_id,
            bbox=reg.bbox,
            source_path=reg.source_path,
            heading_context=reg.heading_context,
            nearby_caption=reg.nearby_caption,
            model_variant=reg.model_variant,
            extraction_meta=dict(reg.meta),
        )
        text_units.append(
            Unit(
                unit_id=unit_id,
                type="text",
                search_text=search_text,
                artefact=clean_span,
                provenance=prov,
            )
        )

    return text_units
