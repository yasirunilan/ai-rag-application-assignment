"""Table Linearization for Search (Person 3 · Story E3.2 & Failure Mode 2 Fix).

Turns each structured `TableArtefact` into normalized `col=value` rows + Markdown
in `unit.search_text`, including the table caption and nearby section headings,
while keeping `unit.artefact` (`TableArtefact`) completely untouched for the answer step.
"""

from __future__ import annotations

import re

from mmrag.config import AppConfig
from mmrag.schemas import TableArtefact, Unit


def _normalize_cell(val: str) -> str:
    return re.sub(r"\s+", " ", str(val or "")).strip()


def linearize_table_unit(unit: Unit, cfg: AppConfig) -> Unit:
    """Populates `search_text` on a table Unit without mutating `unit.artefact`."""
    if unit.type != "table":
        return unit

    if isinstance(unit.artefact, str):
        # Reproduce Failure Mode 2 ("Table answers are mush" when flattened to a raw string)
        raw_norm = _normalize_cell(unit.artefact)
        return unit.model_copy(update={"search_text": raw_norm})

    tbl: TableArtefact = unit.artefact
    headers = [_normalize_cell(h) for h in tbl.headers]

    if cfg.enrich.table_linearization_mode == "flat_raw":
        # Before Fix for Failure Mode 2: raw unlabelled cell numbers without header binding
        flat_rows = [" ".join(_normalize_cell(c) for c in row) for row in tbl.rows]
        return unit.model_copy(update={"search_text": " ; ".join(flat_rows)})

    # After Fix for Failure Mode 2: explicit header=value row binding + caption + heading context
    context_parts = [
        f"Doc: {unit.provenance.doc_id} p.{unit.provenance.page}",
        f"Heading: {_normalize_cell(unit.provenance.heading_context)}"
        if unit.provenance.heading_context
        else "",
        f"Table Caption: {_normalize_cell(tbl.caption or unit.provenance.nearby_caption or '')}"
        if (tbl.caption or unit.provenance.nearby_caption)
        else "",
        f"Columns: {', '.join(headers)}" if headers else "",
    ]
    prefix = " | ".join(p for p in context_parts if p)

    row_sentences: list[str] = []
    for r_idx, row in enumerate(tbl.rows, start=1):
        pairs: list[str] = []
        for c_idx, cell in enumerate(row):
            col_name = headers[c_idx] if c_idx < len(headers) and headers[c_idx] else f"Col_{c_idx+1}"
            val = _normalize_cell(cell)
            if val:
                pairs.append(f"{col_name} = {val}")
        if pairs:
            row_sentences.append(f"Row {r_idx}: " + " ; ".join(pairs))

    md_block = tbl.to_markdown()
    full_search_text = f"{prefix}\n" + "\n".join(row_sentences) + f"\n{md_block}"
    return unit.model_copy(update={"search_text": full_search_text.strip()})


def enrich_table_units(units: list[Unit], cfg: AppConfig) -> list[Unit]:
    """Linearizes all table units in a list while leaving other unit types and artefacts unchanged."""
    return [linearize_table_unit(u, cfg) if u.type == "table" else u for u in units]
