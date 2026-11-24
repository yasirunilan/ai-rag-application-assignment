"""Visible rejection of units that cannot be made retrievable (Person 3 · Story E3.3).

Rejected units are returned with a reason and written to an ingestion-report file;
they must never reach the index.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from mmrag.schemas import Unit

# Captions the brief warns about: too generic for retrieval to ever find the right figure.
GENERIC_CAPTIONS = {
    "a technical diagram",
    "technical diagram",
    "a diagram",
    "diagram",
    "an image",
    "image",
    "a figure",
    "figure",
}
MIN_CAPTION_CHARS = 20


@dataclass
class RejectedUnit:
    unit_id: str
    type: str
    citation: str
    reason: str


def rejection_reason(unit: Unit) -> str | None:
    """Returns why a figure/table unit is not retrievable, or None if it is fine."""
    if unit.type == "text":
        return None
    if not unit.has_valid_provenance(verify_files=True):
        return "missing or unresolvable provenance (doc_id/page/unit_id/artefact)"
    text = unit.search_text.strip()
    if not text:
        return "empty search_text (no usable caption)"
    if unit.type == "figure":
        normalised = re.sub(r"[^a-z ]", "", text.lower()).strip()
        if normalised in GENERIC_CAPTIONS or len(text) < MIN_CAPTION_CHARS:
            return "generic or too-short caption"
    return None


def reject_unenrichable(units: list[Unit]) -> tuple[list[Unit], list[RejectedUnit]]:
    """Splits units into (accepted, rejected). Text units always pass through."""
    accepted: list[Unit] = []
    rejected: list[RejectedUnit] = []
    for u in units:
        reason = rejection_reason(u)
        if reason is None:
            accepted.append(u)
        else:
            rejected.append(RejectedUnit(u.unit_id, u.type, u.provenance.citation, reason))
    return accepted, rejected


def write_rejection_report(rejected: list[RejectedUnit], path: str | Path) -> None:
    """Writes the ingestion-report rejection list as JSON Lines (one rejected unit per line)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
