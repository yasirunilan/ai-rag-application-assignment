"""Corpus & Gold Evidence Loader Contract (Person 1 · Stories E0.2, E0.3 & E1.3).

Defines one loader contract returning `(units, gold_questions)` across multiple corpora:
1. `PdfCorpusLoader`: PDF manual sets (`amt_handbook` + `dude_slice` or dropped-in FAA / ViDoRe PDFs).
2. `EnginuityTask2Loader`: Enginuity Bench Task 2 (`qa_dataset.json` + `figures/` + `table_data/*.tsv`),
   cleaning stray trailing figure captions from `context` as warned in Brief §7.1.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Protocol

import fitz

from mmrag.config import AppConfig
from mmrag.corpus.synthetic import ensure_default_corpus
from mmrag.schemas import GoldQuestion, Provenance, TableArtefact, Unit


_STRAY_FIG_CAPTION_RE = re.compile(r"\s*(?:Figure|Fig\.)\s+\d+[-.]?\d*[^\n]*$", re.IGNORECASE)


class CorpusLoaderProtocol(Protocol):
    """Story E1.3: Unified loader contract returning `(units, gold_questions)`."""

    def load(self, cfg: AppConfig) -> tuple[list[Unit], list[GoldQuestion]]:
        ...


def load_corpus_manuals(cfg: AppConfig, auto_seed: bool = True) -> list[Path]:
    """Discovers PDFs in `cfg.corpus.manuals_dir`, enforcing `max_pages` cap (Story E0.2)."""
    manuals_dir = Path(cfg.corpus.manuals_dir)
    gold_path = Path(cfg.corpus.gold_eval_path)

    if auto_seed and (not manuals_dir.exists() or not list(manuals_dir.glob("*.pdf"))):
        ensure_default_corpus(manuals_dir, gold_path)

    pdf_paths = sorted(manuals_dir.glob("*.pdf"))
    selected: list[Path] = []
    total_pages = 0

    for pdf_path in pdf_paths:
        with fitz.open(str(pdf_path)) as doc:
            n_pages = len(doc)
        if total_pages + n_pages <= cfg.corpus.max_pages or not selected:
            selected.append(pdf_path)
            total_pages += n_pages

    return selected


def load_gold_questions(cfg: AppConfig, auto_seed: bool = True) -> list[GoldQuestion]:
    """Loads gold evaluation questions and verifies every question names a doc + evidence page(s)."""
    if cfg.corpus.eval_dataset == "enginuity_task2" and cfg.corpus.enginuity_dir:
        _, questions = EnginuityTask2Loader(cfg.corpus.enginuity_dir).load(cfg)
        return questions

    gold_path = Path(cfg.corpus.gold_eval_path)
    if auto_seed and not gold_path.exists():
        ensure_default_corpus(cfg.corpus.manuals_dir, gold_path)

    questions: list[GoldQuestion] = []
    with gold_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = GoldQuestion.model_validate(json.loads(line))
            if not q.gold_doc_id or not q.gold_pages or any(p < 1 for p in q.gold_pages):
                raise ValueError(f"Gold question {q.query_id} lacks valid doc_id or evidence page(s).")
            questions.append(q)
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

            # 1. Figure unit
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

            # 2. Paired TSV table unit(s)
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
