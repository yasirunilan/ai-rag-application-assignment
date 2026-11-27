"""Corpus & Gold Evidence Loader (Person 1 · Story E0.2).

The corpus has two layers:

1. **Core manuals** (`corpus.manuals_dir`): the gold-labelled `amt_handbook` + `dude_hydraulic_manual`
   slice. They are always loaded first so the gold questions stay answerable under the page cap.
2. **Real sources** (`corpus.real_sources`): real FAA AMT Handbook PDFs, DUDE sample PDFs, or any other
   manual. Each one is materialised as `<slices_dir>/<doc_id>.pdf` so the router's `doc_id` (the file
   stem) is the configured id.

A global `corpus.max_pages` budget is spent in that order. A document that does not fit is truncated
**from page 1**, never cut from the middle, so page `p` of a slice is page `p` of the original PDF and
every citation stays valid.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import fitz

from mmrag.config import AppConfig
from mmrag.corpus.synthetic import ensure_default_corpus
from mmrag.schemas import GoldQuestion

logger = logging.getLogger(__name__)

MANIFEST_NAME = "corpus_manifest.json"
_SAFE_ID = re.compile(r"[^A-Za-z0-9_\-]+")
_WARNED_MISSING: set[str] = set()


@dataclass
class CorpusDocument:
    """One PDF the pipeline will ingest, with how much of its source was kept."""

    doc_id: str
    path: str               # file handed to the router (core manual or slice)
    source_path: str        # original PDF
    origin: str             # "core" | "real"
    source_pages: int
    loaded_pages: int

    @property
    def truncated(self) -> bool:
        return self.loaded_pages < self.source_pages


@dataclass
class CorpusResolution:
    documents: list[CorpusDocument] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    max_pages: int = 0

    @property
    def total_pages(self) -> int:
        return sum(d.loaded_pages for d in self.documents)

    def pages_by_doc(self) -> dict[str, int]:
        return {d.doc_id: d.loaded_pages for d in self.documents}

    def to_dict(self) -> dict:
        return {
            "max_pages": self.max_pages,
            "total_pages": self.total_pages,
            "num_documents": len(self.documents),
            "documents": [{**asdict(d), "truncated": d.truncated} for d in self.documents],
            "skipped": self.skipped,
        }


def safe_doc_id(raw: str) -> str:
    """Normalises a file stem into a router-safe doc_id."""
    cleaned = _SAFE_ID.sub("_", raw).strip("_")
    return cleaned or "doc"


def _page_count(pdf_path: Path) -> int:
    with fitz.open(str(pdf_path)) as doc:
        return len(doc)


def _write_slice(source: Path, dest: Path, n_pages: int) -> None:
    """Writes the first `n_pages` of `source` to `dest`, reusing an up-to-date slice."""
    if dest.exists() and dest.stat().st_mtime >= source.stat().st_mtime:
        try:
            if _page_count(dest) == n_pages:
                return
        except Exception:  # noqa: BLE001 - a corrupt slice is simply rebuilt
            pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".pdf.part")
    with fitz.open(str(source)) as src, fitz.open() as out:
        out.insert_pdf(src, from_page=0, to_page=n_pages - 1)
        out.save(str(tmp), garbage=3, deflate=True)
    tmp.replace(dest)


def _expand_real_sources(cfg: AppConfig) -> list[tuple[str, Path, Optional[int]]]:
    """Turns `corpus.real_sources` into (doc_id, pdf_path, per_doc_cap) rows; dirs expand per PDF."""
    rows: list[tuple[str, Path, Optional[int]]] = []
    for src in cfg.corpus.real_sources:
        if not src.enabled:
            continue
        p = Path(src.path)
        if p.is_dir():
            for pdf in sorted(p.glob("*.pdf")):
                rows.append((safe_doc_id(f"{src.doc_id}_{pdf.stem}"), pdf, src.max_pages))
        else:
            rows.append((src.doc_id, p, src.max_pages))
    return rows


def resolve_corpus(cfg: AppConfig, auto_seed: bool = True) -> CorpusResolution:
    """Selects the PDFs to ingest under the `max_pages` budget and materialises real-source slices."""
    manuals_dir = Path(cfg.corpus.manuals_dir)
    slices_dir = Path(cfg.corpus.slices_dir)
    gold_path = Path(cfg.corpus.gold_eval_path)
    if cfg.corpus.real_sources and slices_dir.resolve() == manuals_dir.resolve():
        raise ValueError("corpus.slices_dir must differ from corpus.manuals_dir.")

    res = CorpusResolution(max_pages=cfg.corpus.max_pages)
    budget = cfg.corpus.max_pages
    seen_ids: set[str] = set()

    if cfg.corpus.include_core_manuals:
        if auto_seed and (not manuals_dir.exists() or not list(manuals_dir.glob("*.pdf"))):
            ensure_default_corpus(manuals_dir, gold_path)
        for pdf in sorted(manuals_dir.glob("*.pdf")):
            n_src = _page_count(pdf)
            if budget <= 0:
                res.skipped.append({"doc_id": pdf.stem, "path": str(pdf), "reason": "page_budget_exhausted"})
                continue
            n = min(n_src, budget)
            path = pdf
            if n < n_src:
                path = slices_dir / f"{pdf.stem}.pdf"
                _write_slice(pdf, path, n)
            res.documents.append(CorpusDocument(pdf.stem, str(path), str(pdf), "core", n_src, n))
            seen_ids.add(pdf.stem)
            budget -= n

    for doc_id, pdf, per_doc_cap in _expand_real_sources(cfg):
        if doc_id in seen_ids:
            raise ValueError(f"Corpus doc_id '{doc_id}' is used twice (core manual or real source).")
        if not pdf.is_file():
            res.skipped.append({"doc_id": doc_id, "path": str(pdf), "reason": "missing_file"})
            if str(pdf) not in _WARNED_MISSING:
                _WARNED_MISSING.add(str(pdf))
                logger.warning("Corpus source %s not found at %s; skipped (see config.yaml corpus.real_sources).", doc_id, pdf)
            continue
        if budget <= 0:
            res.skipped.append({"doc_id": doc_id, "path": str(pdf), "reason": "page_budget_exhausted"})
            continue
        try:
            n_src = _page_count(pdf)
        except Exception as exc:  # noqa: BLE001 - unreadable/corrupt PDF must not abort the run
            res.skipped.append({"doc_id": doc_id, "path": str(pdf), "reason": f"unreadable_pdf: {exc}"})
            continue
        n = min(n_src, budget, per_doc_cap or n_src)
        if n < 1:
            res.skipped.append({"doc_id": doc_id, "path": str(pdf), "reason": "empty_pdf"})
            continue
        dest = slices_dir / f"{doc_id}.pdf"
        _write_slice(pdf, dest, n)
        res.documents.append(CorpusDocument(doc_id, str(dest), str(pdf), "real", n_src, n))
        seen_ids.add(doc_id)
        budget -= n

    if cfg.corpus.real_sources:
        slices_dir.mkdir(parents=True, exist_ok=True)
        (slices_dir / MANIFEST_NAME).write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")

    if not res.documents:
        raise FileNotFoundError(
            f"No corpus PDFs found (manuals_dir={manuals_dir}, real_sources={len(cfg.corpus.real_sources)})."
        )
    return res


def load_corpus_manuals(cfg: AppConfig, auto_seed: bool = True) -> list[Path]:
    """Returns the PDFs to ingest, enforcing the global `max_pages` cap (Story E0.2)."""
    return [Path(d.path) for d in resolve_corpus(cfg, auto_seed=auto_seed).documents]


def _read_gold_file(path: Path) -> list[GoldQuestion]:
    questions: list[GoldQuestion] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            q = GoldQuestion.model_validate(json.loads(line))
            if not q.gold_doc_id or not q.gold_pages or any(p < 1 for p in q.gold_pages):
                raise ValueError(
                    f"Gold question {q.query_id} ({path}:{line_no}) lacks valid doc_id or evidence page(s)."
                )
            questions.append(q)
    return questions


def load_gold_questions(
    cfg: AppConfig,
    auto_seed: bool = True,
    restrict_to_corpus: bool = True,
) -> list[GoldQuestion]:
    """Loads gold questions (core file + `corpus.extra_gold_paths`); every one names a doc + page(s).

    With `restrict_to_corpus`, questions whose document was not loaded, or whose evidence page was cut
    by the page cap, are dropped (and logged) so they cannot count as retrieval misses.
    """
    gold_path = Path(cfg.corpus.gold_eval_path)
    if auto_seed and not gold_path.exists():
        ensure_default_corpus(cfg.corpus.manuals_dir, gold_path)

    questions: list[GoldQuestion] = []
    if gold_path.exists():
        questions.extend(_read_gold_file(gold_path))
    for extra in cfg.corpus.extra_gold_paths:
        extra_path = Path(extra)
        if not extra_path.exists():
            logger.warning("Extra gold file %s not found; skipping.", extra_path)
            continue
        questions.extend(_read_gold_file(extra_path))

    seen: set[str] = set()
    for q in questions:
        if q.query_id in seen:
            raise ValueError(f"Duplicate gold query_id '{q.query_id}'.")
        seen.add(q.query_id)

    if restrict_to_corpus and questions:
        loaded = resolve_corpus(cfg, auto_seed=auto_seed).pages_by_doc()
        kept = [q for q in questions if q.gold_doc_id in loaded and max(q.gold_pages) <= loaded[q.gold_doc_id]]
        dropped = len(questions) - len(kept)
        if dropped:
            logger.warning(
                "Dropped %d gold question(s) whose document or evidence page is outside the loaded corpus.",
                dropped,
            )
        questions = kept
    return questions
