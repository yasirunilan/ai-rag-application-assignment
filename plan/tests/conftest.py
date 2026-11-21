"""Shared Pytest Fixtures for the 5-Engineer Multimodal RAG Test Suite."""

from __future__ import annotations

from pathlib import Path
import pytest

from mmrag.config import AppConfig
from mmrag.corpus.synthetic import ensure_default_corpus


@pytest.fixture()
def lab_cfg(tmp_path: Path) -> AppConfig:
    """Provides an isolated AppConfig and pre-built technical manuals + gold eval set in tmp_path."""
    data_dir = tmp_path / "data"
    manuals_dir = data_dir / "corpus" / "manuals"
    gold_path = data_dir / "corpus" / "gold_eval.jsonl"
    ensure_default_corpus(manuals_dir, gold_path)

    cfg = AppConfig()
    cfg.project.data_dir = str(data_dir)
    cfg.project.artifacts_dir = str(data_dir / "artifacts")
    cfg.project.cache_dir = str(data_dir / "cache" / "captions")
    cfg.project.index_dir = str(data_dir / "index")
    cfg.project.review_queue_path = str(data_dir / "review_queue.jsonl")
    cfg.project.reports_dir = str(tmp_path / "reports")
    cfg.corpus.manuals_dir = str(manuals_dir)
    cfg.corpus.gold_eval_path = str(gold_path)
    cfg.enrich.vision_provider = "offline_heuristic"
    cfg.index.embedding_provider = "deterministic_hash"
    return cfg
