"""Structure-Aware Prose Chunker (Person 4 · Story E4.1).

Splits long prose `text` units along heading/paragraph boundaries while strictly keeping
every `figure` unit and every `table` unit as ONE atomic unit (never split).
"""

from __future__ import annotations

from mmrag.config import AppConfig
from mmrag.schemas import Unit


def chunk_units_for_indexing(units: list[Unit], cfg: AppConfig) -> list[Unit]:
    """Chunks prose units exceeding `prose_chunk_max_chars`; keeps figures and tables atomic."""
    max_chars = cfg.index.prose_chunk_max_chars
    overlap = cfg.index.prose_chunk_overlap_chars
    output: list[Unit] = []

    for u in units:
        if u.type in ("figure", "table"):
            # Non-negotiable (Story E4.1): keep a table or figure-caption as ONE unit
            output.append(u)
            continue

        text = str(u.artefact).strip()
        if len(text) <= max_chars:
            output.append(u)
            continue

        # Split long prose into overlapping structural chunks preserving provenance
        start = 0
        part_idx = 1
        while start < len(text):
            end = min(len(text), start + max_chars)
            span = text[start:end].strip()
            if span:
                sub_id = f"{u.unit_id}_c{part_idx:02d}"
                heading_prefix = (
                    f"[{u.provenance.heading_context}] "
                    if u.provenance.heading_context
                    else ""
                )
                sub_prov = u.provenance.model_copy(update={"unit_id": sub_id})
                output.append(
                    u.model_copy(
                        update={
                            "unit_id": sub_id,
                            "artefact": span,
                            "search_text": f"{heading_prefix}{span}",
                            "provenance": sub_prov,
                        }
                    )
                )
                part_idx += 1
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)

    return output
