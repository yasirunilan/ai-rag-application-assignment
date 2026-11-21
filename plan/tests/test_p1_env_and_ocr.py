"""Person 1 Unit & Acceptance Tests: Environment, Corpus & Scanned OCR Quality Gate (Stories E0.1, E0.2, E1.2)."""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from mmrag.config import AppConfig, BillingGuardError, EnrichConfig
from mmrag.corpus.loader import load_corpus_manuals, load_gold_questions
from mmrag.ingest.ocr_gate import (
    compute_garbled_char_ratio,
    evaluate_ocr_quality_gate,
    is_scanned_page,
    run_ocr_on_page,
)


def test_e0_1_billing_guard_rejects_paid_openrouter_models() -> None:
    """Story E0.1 & §6: Paid models must be rejected by the config billing guard."""
    with pytest.raises(BillingGuardError):
        EnrichConfig(
            vision_provider="openrouter_free",
            enforce_free_only=True,
            openrouter_models=["openai/gpt-4o"],  # Missing ':free'
        )


def test_e0_2_corpus_and_gold_evidence_labels(lab_cfg: AppConfig) -> None:
    """Story E0.2 Acceptance: Corpus loaded; every gold question names a doc + evidence page(s)."""
    manuals = load_corpus_manuals(lab_cfg)
    assert len(manuals) >= 2
    assert any(p.name == "amt_handbook.pdf" for p in manuals)

    questions = load_gold_questions(lab_cfg)
    assert len(questions) >= 10
    for q in questions:
        assert q.gold_doc_id
        assert len(q.gold_pages) >= 1
        assert all(p >= 1 for p in q.gold_pages)
        assert q.expected_type in ("text", "figure", "table")


def test_e1_2_scanned_page_detection_and_quality_gate(lab_cfg: AppConfig) -> None:
    """Story E1.2 Acceptance: Scanned pages are OCR'd or flagged; review queue exists; sub-threshold pages never indexed."""
    amt_pdf = Path(lab_cfg.corpus.manuals_dir) / "amt_handbook.pdf"
    with fitz.open(str(amt_pdf)) as doc:
        # Page 1 is a digital page
        is_scan_p1, chars_p1, cov_p1 = is_scanned_page(doc[0])
        assert not is_scan_p1
        assert chars_p1 > 100

        # Page 4 is an image-only degraded scanned page
        is_scan_p4, chars_p4, cov_p4 = is_scanned_page(doc[3])
        assert is_scan_p4
        assert chars_p4 == 0
        assert cov_p4 >= 0.60

        ocr_text, mean_conf = run_ocr_on_page(doc[3], engine="heuristic_fallback")
        passed, garbled_ratio, alnum_ratio, reasons = evaluate_ocr_quality_gate(
            doc_id="amt_handbook",
            page_num=4,
            source_path=str(amt_pdf),
            ocr_text=ocr_text,
            mean_confidence=mean_conf,
            cfg=lab_cfg,
            append_to_queue=True,
        )
        assert not passed
        assert garbled_ratio > lab_cfg.ingest.ocr_quality_gate.max_garbled_char_ratio
        assert len(reasons) >= 1

    rq_path = Path(lab_cfg.project.review_queue_path)
    assert rq_path.exists()
    queued = [json.loads(line) for line in rq_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(queued) == 1
    assert queued[0]["doc_id"] == "amt_handbook"
    assert queued[0]["page"] == 4


def test_e1_2_garbled_ratio_allows_clean_technical_text() -> None:
    clean = "Nominal hydraulic pressure is 3,000 PSI ± 50 PSI at 20°C with 15Ω relay coil."
    assert compute_garbled_char_ratio(clean) < 0.05
    garbled = "~|1lI| %%% ^&* @#$ ~|1lI| ??? === /// |||"
    assert compute_garbled_char_ratio(garbled) > 0.40
