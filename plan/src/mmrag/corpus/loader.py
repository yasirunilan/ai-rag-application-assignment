"""Corpus & Gold Evidence Loader (Person 1 · Stories E0.2, E0.3 & E1.3).

Supports:
1. Core manuals (`corpus.manuals_dir`) + Real sources (`corpus.real_sources`), truncated from page 1
   under the global `corpus.max_pages` budget so cited page numbers always match the original PDF.
2. Unified `CorpusLoaderProtocol` (`PdfCorpusLoader` and `EnginuityTask2Loader`).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import re
from typing import Optional, Protocol

import fitz

from mmrag.config import AppConfig
from mmrag.corpus.synthetic import ensure_default_corpus
from mmrag.schemas import GoldQuestion, Provenance, TableArtefact, Unit

logger = logging.getLogger(__name__)

MANIFEST_NAME = "corpus_manifest.json"
_SAFE_ID = re.compile(r"[^A-Za-z0-9_\-]+")
_STRAY_FIG_CAPTION_RE = re.compile(r"\s*(?:Figure|Fig\.)\s+\d+[-.]?\d*[^\n]*$", re.IGNORECASE)
_WARNED_MISSING: set[str] = set()


class CorpusLoaderProtocol(Protocol):
    """Story E1.3: Unified loader contract returning `(units, gold_questions)`."""

    def load(self, cfg: AppConfig) -> tuple[list[Unit], list[GoldQuestion]]:
        ...


@dataclass
class CorpusDocument:
    """One PDF the pipeline will ingest, with how much of its source was kept."""

    doc_id: str
    path: str
    source_path: str
    origin: str  # "core" | "real"
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
        except Exception:  # noqa: BLE001
            pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".pdf.part")
    with fitz.open(str(source)) as src, fitz.open() as out:
        out.insert_pdf(src, from_page=0, to_page=n_pages - 1)
        out.save(str(tmp), garbage=3, deflate=True)
    tmp.replace(dest)


def _expand_real_sources(cfg: AppConfig) -> list[tuple[str, Path, Optional[int]]]:
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
        except Exception as exc:  # noqa: BLE001
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
    """Loads gold questions (core file + `corpus.extra_gold_paths` or Enginuity Task 2)."""
    if cfg.corpus.eval_dataset == "enginuity_task2" and cfg.corpus.enginuity_dir:
        _, questions = EnginuityTask2Loader(cfg.corpus.enginuity_dir).load(cfg)
        return questions

    gold_path = Path(cfg.corpus.gold_eval_path)
    if auto_seed and not gold_path.exists():
        ensure_default_corpus(cfg.corpus.manuals_dir, gold_path)

    questions = []
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


def clean_enginuity_context(raw_context: str) -> str:
    """Brief §7.1: Strips stray trailing caption fragments from other figures in Enginuity context."""
    cleaned = " ".join((raw_context or "").split())
    return _STRAY_FIG_CAPTION_RE.sub("", cleaned).strip()


def _parse_tsv_table(tsv_path: Path, caption: str | None = None) -> TableArtefact:
    with tsv_path.open("r", encoding="utf-8") as f:
        rows = [[cell.strip() for cell in row] for row in csv.reader(f, delimiter="\t") if any(row)]
    if not rows:
        return TableArtefact(headers=["Col_1"], rows=[[""]], caption=caption)
    width = max(len(r) for r in rows)
    grid = [r + [""] * (width - len(r)) for r in rows]
    headers = [h if h else f"Col_{i+1}" for i, h in enumerate(grid[0])]
    data_rows = grid[1:] if len(grid) > 1 else grid
    return TableArtefact(headers=headers, rows=data_rows, caption=caption)


class PdfCorpusLoader:
    """Loader implementation #1: parses and extracts Units from PDF technical manuals."""

    def __init__(self, pdf_paths: list[Path] | None = None) -> None:
        self.pdf_paths = pdf_paths

    def load(self, cfg: AppConfig) -> tuple[list[Unit], list[GoldQuestion]]:
        from mmrag.extract.figures import extract_figure_units
        from mmrag.extract.tables import extract_table_units, extract_text_units
        from mmrag.ingest.router import parse_and_route_pdf

        manuals = self.pdf_paths or load_corpus_manuals(cfg, auto_seed=True)
        regions = []
        for pdf in manuals:
            _, regs = parse_and_route_pdf(pdf, cfg)
            regions.extend(regs)
        figs, _ = extract_figure_units(regions, cfg)
        tbls = extract_table_units(regions, cfg)
        txts = extract_text_units(regions)
        return txts + figs + tbls, load_gold_questions(cfg, auto_seed=True)


class EnginuityTask2Loader:
    """Loader implementation #2 (Story E1.3): loads Enginuity Bench Task 2 (`qa_dataset.json`, PNGs, TSVs)."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    def load(self, cfg: AppConfig) -> tuple[list[Unit], list[GoldQuestion]]:
        qa_path = self.root_dir / "qa_dataset.json"
        if not qa_path.exists():
            raise FileNotFoundError(f"Enginuity qa_dataset.json not found at {qa_path}")

        records = json.loads(qa_path.read_text(encoding="utf-8"))
        units_by_id: dict[str, Unit] = {}
        questions: list[GoldQuestion] = []
        doc_id = "enginuity_tm"

        for idx, rec in enumerate(records, start=1):
            fig_rel = str(rec.get("figure", "")).strip()
            fig_stem = Path(fig_rel).stem or f"fig_{idx}"
            m = re.search(r"(\d+)", fig_stem)
            page_num = int(m.group(1)) if m else idx

            if fig_rel:
                fig_path = self.root_dir / fig_rel
                fig_uid = f"{doc_id}__p{page_num:04d}__figure_001"
                if fig_uid not in units_by_id and fig_path.exists():
                    ctx = clean_enginuity_context(rec.get("context", ""))
                    units_by_id[fig_uid] = Unit(
                        unit_id=fig_uid,
                        type="figure",
                        search_text="",
                        artefact=str(fig_path),
                        provenance=Provenance(
                            doc_id=doc_id,
                            page=page_num,
                            unit_id=fig_uid,
                            source_path=str(fig_path),
                            nearby_caption=f"Figure {page_num} ({fig_stem})",
                            heading_context=ctx[:140] if ctx else None,
                        ),
                    )

            tsv_list = rec.get("tsv") or []
            for t_i, tsv_rel in enumerate(tsv_list, start=1):
                tsv_path = self.root_dir / str(tsv_rel)
                tbl_uid = f"{doc_id}__p{page_num:04d}__table_{t_i:03d}"
                if tbl_uid not in units_by_id and tsv_path.exists():
                    tbl_art = _parse_tsv_table(tsv_path, caption=f"Parts List Table for Figure {page_num}")
                    units_by_id[tbl_uid] = Unit(
                        unit_id=tbl_uid,
                        type="table",
                        search_text="",
                        artefact=tbl_art,
                        provenance=Provenance(
                            doc_id=doc_id,
                            page=page_num,
                            unit_id=tbl_uid,
                            source_path=str(tsv_path),
                            nearby_caption=tbl_art.caption,
                        ),
                    )

            q_text = str(rec.get("question", "")).strip()
            ans_text = str(rec.get("answer", "")).strip()
            if q_text:
                q_lower = q_text.lower()
                exp_type = "table" if any(w in q_lower for w in ("part number", "table", "quantity", "included")) and tsv_list else "figure"
                keywords = [w.strip(".,;:()") for w in ans_text.split() if len(w.strip(".,;:()")) >= 4][:4]
                questions.append(
                    GoldQuestion(
                        query_id=f"ENG_{idx:03d}",
                        query=q_text,
                        gold_doc_id=doc_id,
                        gold_pages=[page_num],
                        expected_type=exp_type,
                        gold_artefact_hint=f"Figure {page_num}" if exp_type == "figure" else f"Parts List Table for Figure {page_num}",
                        expected_keywords=keywords,
                    )
                )

        return list(units_by_id.values()), questions
