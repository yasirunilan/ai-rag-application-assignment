"""Person 1 Unit & Acceptance Tests: Environment, Model Checks, Multi-Corpus Loader & Scanned OCR Gate (Stories E0.1–E0.3, E1.2, E1.3)."""

from __future__ import annotations

import io
import json
from pathlib import Path
import urllib.error

import fitz
from PIL import Image
import pytest

from mmrag.config import (
    AnswerConfig,
    AppConfig,
    BillingGuardError,
    EnrichConfig,
    IndexConfig,
    get_openrouter_api_key,
    load_config,
)
from mmrag.corpus.loader import (
    EnginuityTask2Loader,
    PdfCorpusLoader,
    clean_enginuity_context,
    load_corpus_manuals,
    load_gold_questions,
)
from mmrag.diagnostics import check_models, check_vector_backend
from mmrag.ingest.ocr_gate import (
    compute_garbled_char_ratio,
    evaluate_ocr_quality_gate,
    is_scanned_page,
    run_ocr_on_page,
    summarize_review_queue,
)
from mmrag.pipeline import MultimodalRAGPipeline


def test_e0_1_billing_guard_rejects_paid_openrouter_models() -> None:
    """Story E0.1 & §6: Paid models must be rejected across vision, embed, and generator configs."""
    with pytest.raises(BillingGuardError):
        EnrichConfig(
            vision_provider="openrouter_free",
            enforce_free_only=True,
            openrouter_models=["openai/gpt-4o"],
        )
    with pytest.raises(BillingGuardError):
        IndexConfig(openrouter_embed_model="openai/text-embedding-3-large")
    with pytest.raises(BillingGuardError):
        AnswerConfig(text_model="anthropic/claude-3.5-sonnet")


def test_e0_1_dotenv_strips_quotes_and_applies_slug_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Story E0.1: `.env` strips stray spaces/quotes around keys and swaps slugs via config only."""
    for var in ("OPENROUTER_API_KEY", "VISION_MODEL", "EMBED_MODEL", "TEXT_MODEL"):
        monkeypatch.delenv(var, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        'OPENROUTER_API_KEY = "  sk-or-test-123  "\n'
        "VISION_MODEL=google/gemma-4-31b-it:free\n"
        "EMBED_MODEL=nvidia/nemotron-3-embed-1b:free\n"
        "TEXT_MODEL=nvidia/nemotron-3.5-lightning:free\n",
        encoding="utf-8",
    )
    cfg = load_config(path=tmp_path / "nonexistent.yaml", dotenv_path=env_file)
    assert get_openrouter_api_key() == "sk-or-test-123"
    assert cfg.enrich.openrouter_models[0] == "google/gemma-4-31b-it:free"
    assert cfg.index.openrouter_embed_model == "nvidia/nemotron-3-embed-1b:free"
    assert cfg.answer.text_model == "nvidia/nemotron-3.5-lightning:free"


def test_e0_1_and_e0_2_model_and_vector_readiness_checks(
    lab_cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stories E0.1 & E0.2: Missing key stops before any call; live check reports OK or retired 404."""
    vec = check_vector_backend(lab_cfg)
    assert vec["ok"] is True

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    calls: list[str] = []

    def never_called(req, timeout=0):
        calls.append(req.full_url)
        raise AssertionError("Should not make network calls when key is missing")

    no_key = check_models(lab_cfg, offline=False, urlopen=never_called)
    assert no_key["ok"] is False
    assert "Missing OPENROUTER_API_KEY" in no_key["error"]
    assert calls == []

    # Offline mode succeeds with zero network
    assert check_models(lab_cfg, offline=True, urlopen=never_called)["ok"] is True

    # Online check with mocked OpenRouter responses (all OK, then 404 retired slug)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    class _Resp:
        def __init__(self, payload: dict) -> None:
            self._raw = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self._raw

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_ok(req, timeout=0):
        if req.full_url.endswith("/embeddings"):
            return _Resp({"data": [{"embedding": [0.1, 0.2]}]})
        return _Resp({"choices": [{"message": {"content": "OK"}}]})

    live_ok = check_models(lab_cfg, offline=False, urlopen=fake_ok)
    assert live_ok["ok"] is True
    assert {c["role"] for c in live_ok["checks"]} == {
        "embeddings",
        "text_generation",
        "vision_captioning",
        "figure_answers",
    }

    def fake_404(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b""))

    retired = check_models(lab_cfg, offline=False, urlopen=fake_404)
    assert retired["ok"] is False
    assert all("retired" in c["reason"] for c in retired["checks"])


def test_e0_2_and_e0_3_corpus_gold_labels_and_stage0_prediction(lab_cfg: AppConfig) -> None:
    """Stories E0.2 & E0.3: Corpus loaded; gold questions name doc + page(s); prediction recorded."""
    manuals = load_corpus_manuals(lab_cfg)
    assert len(manuals) >= 2
    assert any(p.name == "amt_handbook.pdf" for p in manuals)
    assert lab_cfg.corpus.stage0_prediction_worst_type == "figure"
    assert len(lab_cfg.corpus.stage0_prediction_reason) > 20

    questions = load_gold_questions(lab_cfg)
    assert len(questions) >= 10
    for q in questions:
        assert q.gold_doc_id
        assert len(q.gold_pages) >= 1
        assert all(p >= 1 for p in q.gold_pages)
        assert q.expected_type in ("text", "figure", "table")


def test_e1_2_scanned_page_detection_dirty_ocr_layer_and_quality_gate(
    lab_cfg: AppConfig, tmp_path: Path
) -> None:
    """Story E1.2: Detects image-only & dirty-hidden-OCR scans, deduplicates queue, reports failed count."""
    amt_pdf = Path(lab_cfg.corpus.manuals_dir) / "amt_handbook.pdf"
    with fitz.open(str(amt_pdf)) as doc:
        is_scan_p1, chars_p1, _ = is_scanned_page(doc[0])
        assert not is_scan_p1 and chars_p1 > 100

        is_scan_p4, chars_p4, cov_p4 = is_scanned_page(doc[3])
        assert is_scan_p4 and chars_p4 == 0 and cov_p4 >= 0.60

        ocr_text, mean_conf = run_ocr_on_page(doc[3], engine="heuristic_fallback")
        for _ in range(2):  # Duplicate run should not duplicate the review_queue entry
            passed, garbled_ratio, _, reasons = evaluate_ocr_quality_gate(
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

    summary = summarize_review_queue(lab_cfg.project.review_queue_path)
    assert summary["failed_pages_count"] == 1
    assert summary["pages"] == ["amt_handbook p.4"]

    # Dirty invisible OCR layer over full-page scan must also be caught as scanned
    dirty_doc = fitz.open()
    dp = dirty_doc.new_page(width=600, height=800)
    buf = io.BytesIO()
    Image.new("RGB", (600, 800), (220, 220, 220)).save(buf, format="PNG")
    dp.insert_image(fitz.Rect(0, 0, 600, 800), stream=buf.getvalue())
    dp.insert_text((20, 40), "~|1lI| %%% ^&* @#$ ~|1lI| ??? === /// ||| " * 3, fontsize=8)
    is_dirty_scan, _, _ = is_scanned_page(dp)
    assert is_dirty_scan is True
    dirty_doc.close()


def test_e1_2_garbled_ratio_allows_clean_technical_text() -> None:
    clean = "Nominal hydraulic pressure is 3,000 PSI ± 50 PSI at 20°C with 15Ω relay coil."
    assert compute_garbled_char_ratio(clean) < 0.05
    garbled = "~|1lI| %%% ^&* @#$ ~|1lI| ??? === /// |||"
    assert compute_garbled_char_ratio(garbled) > 0.40


def test_e1_3_two_corpora_run_through_unified_loader_and_pipeline(
    lab_cfg: AppConfig, tmp_path: Path
) -> None:
    """Story E1.3 + E3.3: Both PDF manuals and Enginuity Task 2 run through the exact same downstream pipeline."""
    # Corpus 1: PDF manual loader
    pdf_units, pdf_qs = PdfCorpusLoader().load(lab_cfg)
    assert pdf_units and pdf_qs

    # Corpus 2: Synthetic Enginuity Bench Task 2 structure
    eng_dir = tmp_path / "enginuity_task2"
    (eng_dir / "figures").mkdir(parents=True)
    (eng_dir / "table_data").mkdir(parents=True)
    fig_buf = io.BytesIO()
    Image.new("RGB", (300, 200), (245, 245, 250)).save(fig_buf, format="PNG")
    (eng_dir / "figures" / "figure_01.png").write_bytes(fig_buf.getvalue())
    (eng_dir / "table_data" / "table_01.tsv").write_text(
        "Item\tPart Number\tDescription\tQty\n1\tTM-9981-A\tFuel Injector Seal\t2\n",
        encoding="utf-8",
    )
    qa_records = [
        {
            "figure": "figures/figure_01.png",
            "tables": ["tables/table_01.png"],
            "tsv": ["table_data/table_01.tsv"],
            "question": "What is the part number for the Fuel Injector Seal in the table?",
            "answer": "The Fuel Injector Seal uses part number TM-9981-A with quantity 2.",
            "context": "Assembly breakdown of the diesel fuel injector manifold. Figure 09. Unrelated brake drum.",
        }
    ]
    (eng_dir / "qa_dataset.json").write_text(json.dumps(qa_records), encoding="utf-8")

    assert "Figure 09" not in clean_enginuity_context(qa_records[0]["context"])

    eng_cfg = lab_cfg.model_copy(
        update={
            "corpus": lab_cfg.corpus.model_copy(
                update={"eval_dataset": "enginuity_task2", "enginuity_dir": str(eng_dir)}
            )
        }
    )
    pipe = MultimodalRAGPipeline(eng_cfg, force_offline=True)
    stats = pipe.ingest_and_index(persist_index=True)
    assert stats["num_figure_units"] == 1
    assert stats["num_table_units"] == 1
    assert (Path(eng_cfg.project.index_dir) / "index_report.json").exists()

    ans = pipe.answer("What is the part number for TM-9981-A Fuel Injector Seal table?")
    assert not ans.abstained and ans.citation == "enginuity_tm p.1"
