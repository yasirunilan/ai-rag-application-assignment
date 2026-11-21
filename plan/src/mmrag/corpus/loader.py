"""Corpus & Gold Evidence Loader (Person 1 · Story E0.2)."""

from __future__ import annotations

import json
from pathlib import Path

import fitz

from mmrag.config import AppConfig
from mmrag.corpus.synthetic import ensure_default_corpus
from mmrag.schemas import GoldQuestion


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
