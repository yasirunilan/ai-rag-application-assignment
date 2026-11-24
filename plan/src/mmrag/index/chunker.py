"""Structure-Aware Prose Chunker (Person 4 · Story E4.1).

Splits long prose `text` units along heading/paragraph/sentence boundaries while strictly
keeping every `figure` unit and every `table` unit as ONE atomic unit (never split).

Splitting order (coarsest first, finer only when a piece is still over `max_chars`):
    heading lines / blank-line paragraphs  ->  sentences  ->  words  ->  hard character cut
Chunks never cut a word unless a single token is longer than `max_chars`. Overlap between
consecutive chunks is made of whole trailing sentences, never a mid-word slice. A detected
in-text heading always starts a new chunk and is carried into the `search_text` of every
chunk under it, so a chunk deep inside a section is still findable by its section title.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from mmrag.config import AppConfig
from mmrag.schemas import Unit


_BLANK_LINE = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[\"'(\[]?[A-Z0-9])")
_NUMBERED_HEADING = re.compile(
    r"^(?:(?:chapter|section|part|appendix)\s+[\dA-Z]+|\d+(?:\.\d+)*\.?\s+[A-Z])",
    re.IGNORECASE,
)


def is_heading_line(line: str) -> bool:
    """Conservative heading detector for lines inside a prose region."""
    s = line.strip()
    if not s or len(s) > 100 or s.endswith((".", ",", ";")):
        return False
    if _NUMBERED_HEADING.match(s):
        return True
    words = s.split()
    letters = [c for c in s if c.isalpha()]
    return len(words) >= 2 and len(s) <= 60 and bool(letters) and all(c.isupper() for c in letters)


@dataclass
class _Block:
    text: str
    is_heading: bool


def _structural_blocks(text: str) -> list[_Block]:
    """Paragraph blocks, with heading lines broken out as their own blocks."""
    blocks: list[_Block] = []
    for para in _BLANK_LINE.split(text):
        buf: list[str] = []
        for line in para.splitlines():
            if is_heading_line(line):
                if buf:
                    blocks.append(_Block(" ".join(buf), False))
                    buf = []
                blocks.append(_Block(line.strip(), True))
            elif line.strip():
                buf.append(line.strip())
        if buf:
            blocks.append(_Block(" ".join(buf), False))
    return blocks


def _split_oversized(piece: str, max_chars: int) -> list[str]:
    """Splits one piece into <= max_chars parts: sentences, then words, then hard cuts."""
    if len(piece) <= max_chars:
        return [piece]
    sentences = [s.strip() for s in _SENTENCE_END.split(piece) if s.strip()]
    if len(sentences) > 1:
        out: list[str] = []
        for sent in sentences:
            out.extend(_split_oversized(sent, max_chars))
        return out

    parts: list[str] = []
    current = ""
    for word in piece.split():
        while len(word) > max_chars:  # pathological single token (e.g. a long hash/URL)
            if current:
                parts.append(current)
                current = ""
            parts.append(word[:max_chars])
            word = word[max_chars:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            parts.append(current)
            current = word
    if current:
        parts.append(current)
    return parts


def _sentences_for_overlap(chunk_pieces: list[str], overlap: int) -> list[str]:
    """Whole trailing sentences from the previous chunk whose combined length <= overlap."""
    if overlap <= 0:
        return []
    tail: list[str] = []
    total = 0
    for piece in reversed(chunk_pieces):
        for sent in reversed([s for s in _SENTENCE_END.split(piece) if s.strip()]):
            sent = sent.strip()
            add = len(sent) + (1 if tail else 0)
            if total + add > overlap:
                return tail
            tail.insert(0, sent)
            total += add
    return tail


def split_prose(text: str, max_chars: int, overlap: int) -> list[tuple[str, Optional[str]]]:
    """Returns `[(chunk_text, section_heading_or_None), ...]` for one prose span."""
    text = text.strip()
    if not text:
        return []
    if max_chars <= 0 or len(text) <= max_chars:
        return [(text, None)]
    overlap = max(0, min(overlap, max_chars // 2))

    chunks: list[tuple[str, Optional[str]]] = []
    pieces: list[str] = []
    length = 0
    section: Optional[str] = None
    fresh = True  # current chunk holds only carried-over overlap (nothing new yet)
    headings_only = False  # current chunk holds only heading lines so far

    def flush() -> None:
        nonlocal pieces, length, fresh, headings_only
        if pieces and not fresh:
            chunks.append((" ".join(pieces), section))
        pieces, length, fresh, headings_only = [], 0, True, False

    for block in _structural_blocks(text):
        if block.is_heading:
            add = len(block.text) + 1
            if headings_only and length + add <= max_chars:
                # Stacked headings ("Chapter 4" then "4.1 ...") stay together with their body
                pieces.append(block.text)
                length += add
            else:
                flush()  # a heading always opens a new chunk; no overlap across sections
                pieces, length, fresh = [block.text], len(block.text), False
            section = block.text
            headings_only = True
            continue
        for piece in _split_oversized(block.text, max_chars):
            add = len(piece) + (1 if pieces else 0)
            if pieces and length + add > max_chars:
                carried = _sentences_for_overlap(pieces, overlap)
                flush()
                carried_len = sum(len(c) for c in carried) + max(0, len(carried) - 1)
                if carried and carried_len + 1 + len(piece) <= max_chars:
                    pieces, length = list(carried), carried_len
                add = len(piece) + (1 if pieces else 0)
            pieces.append(piece)
            length += add
            fresh = False
            headings_only = False
    flush()
    return chunks


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
        if max_chars <= 0 or len(text) <= max_chars:
            output.append(u)
            continue

        pieces = split_prose(text, max_chars, overlap)
        if len(pieces) <= 1:
            output.append(u)
            continue

        context = u.provenance.heading_context
        for part_idx, (span, section) in enumerate(pieces, start=1):
            sub_id = f"{u.unit_id}_c{part_idx:02d}"
            labels = [context] if context else []
            if section and (not context or section not in context) and not span.startswith(section):
                labels.append(section)
            prefix = f"[{' — '.join(labels)}] " if labels else ""
            meta = {
                **u.provenance.extraction_meta,
                "chunk_parent_unit_id": u.unit_id,
                "chunk_index": part_idx,
                "chunk_count": len(pieces),
            }
            if section:
                meta["chunk_section_heading"] = section
            sub_prov = u.provenance.model_copy(update={"unit_id": sub_id, "extraction_meta": meta})
            output.append(
                u.model_copy(
                    update={
                        "unit_id": sub_id,
                        "artefact": span,
                        "search_text": f"{prefix}{span}",
                        "provenance": sub_prov,
                    }
                )
            )

    return output
