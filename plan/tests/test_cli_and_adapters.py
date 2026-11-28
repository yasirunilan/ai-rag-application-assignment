"""Comprehensive tests for CLI commands, optional adapters (OpenRouter embedder, FastEmbed, Docling, local vision),
OCR calibration / Tesseract wrapper, corpus fetchers, and HTTP helpers so every module exceeds 80%+ coverage."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import types
import urllib.error

import fitz
import numpy as np
from PIL import Image
import pytest
import yaml

from mmrag import cli, env_check
from mmrag.config import AppConfig, BillingGuardError, CorpusSource
from mmrag.corpus import sources
from mmrag.enrich import captioner
from mmrag.index.embedder import FastEmbedWrapper, OpenRouterEmbedder, create_embedder
from mmrag.ingest import ocr_calibration, ocr_gate, router


def _write_cfg(cfg: AppConfig, path: Path) -> Path:
    path.write_text(yaml.safe_dump(cfg.model_dump()), encoding="utf-8")
    return path


def test_cli_end_to_end_subcommands(
    lab_cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercises all CLI subcommands offline with zero network."""
    cfg_path = str(_write_cfg(lab_cfg, tmp_path / "config.yaml"))

    assert cli.main(["--config", cfg_path, "check", "--offline"]) == 0
    assert cli.main(["--config", cfg_path, "check-models", "--offline"]) == 0
    assert cli.main(["--config", cfg_path, "corpus"]) == 0
    assert cli.main(["--config", cfg_path, "ingest", "--offline"]) == 0
    assert cli.main(["--config", cfg_path, "spotcheck"]) == 0
    assert (
        cli.main(
            [
                "--config",
                cfg_path,
                "answer",
                "--offline",
                "--type",
                "figure",
                "-q",
                "Show the engine starter relay control wiring diagram for Model 737-800",
            ]
        )
        == 0
    )
    assert cli.main(["--config", cfg_path, "eval", "--offline"]) == 0
    assert cli.main(["--config", cfg_path, "fetch-corpus"]) == 0

    # Stub heavy ablation loop inside smoke/ablate CLI dispatch so CLI paths run fast
    monkeypatch.setattr(cli, "run_e7_3_failure_mode_experiments", lambda cfg, force_offline=True: {"ok": True})
    assert cli.main(["--config", cfg_path, "ablate", "--offline"]) == 0
    assert cli.main(["--config", cfg_path, "smoke"]) == 0

    # ocr-calibrate: graceful exit when Tesseract missing, and full --apply path when mocked
    monkeypatch.setattr(ocr_gate, "tesseract_status", lambda: (False, "missing"))
    assert cli.main(["--config", cfg_path, "ocr-calibrate"]) == 2

    fake_rep = {
        "num_samples": 12,
        "num_reference_pages": 1,
        "num_acceptable": 6,
        "current_thresholds": {
            "min_mean_confidence": 60.0,
            "max_garbled_char_ratio": 0.15,
            "min_alnum_token_ratio": 0.5,
            "min_word_count": 10,
            "precision": 1.0,
            "recall": 0.8,
            "bad_pages_indexed": 0,
            "good_pages_flagged": 1,
        },
        "recommended_thresholds": {
            "min_mean_confidence": 65.0,
            "max_garbled_char_ratio": 0.12,
            "min_alnum_token_ratio": 0.55,
            "min_word_count": 10,
            "precision": 1.0,
            "recall": 1.0,
            "bad_pages_indexed": 0,
            "good_pages_flagged": 0,
        },
        "selection_strategy": "max recall s.t. precision >= 0.95",
        "real_scanned_pages": [{"passes": {"current": False, "recommended": False}}],
        "report_path": str(tmp_path / "cal.json"),
        "review_by_eye": {"flagged": [{}], "passed": [{}]},
    }
    monkeypatch.setattr(ocr_gate, "tesseract_status", lambda: (True, "tesseract 5.4"))
    monkeypatch.setattr(ocr_calibration, "run_ocr_gate_calibration", lambda *a, **k: fake_rep)
    assert cli.main(["--config", cfg_path, "ocr-calibrate", "--pages", "1", "--apply"]) == 0


def test_ocr_calibration_and_tesseract_wrappers_with_mock_ocr(
    lab_cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercises all 12 image degradations, reference selection, real-scan review, and Tesseract OCR wrapper."""
    amt_pdf = Path(lab_cfg.corpus.manuals_dir) / "amt_handbook.pdf"
    with fitz.open(str(amt_pdf)) as doc:
        ref_text = doc[0].get_text("text")

    call_idx = {"n": 0}

    def fake_ocr(img: Image.Image, lang: str = "eng", timeout: float = 20.0) -> tuple[str, float]:
        call_idx["n"] += 1
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        if float(np.std(arr)) > 25.0 and call_idx["n"] % 3 != 0:
            return ref_text, 91.0
        return "~~ || @#$ %^& 1lI", 32.0

    monkeypatch.setattr(ocr_calibration, "ocr_image_tesseract", fake_ocr)
    rep = ocr_calibration.run_ocr_gate_calibration(
        lab_cfg, [amt_pdf], max_reference_pages=1, save_images=True
    )
    assert rep["num_samples"] == len(ocr_calibration.DEGRADATIONS)
    assert len(rep["by_degradation"]) == len(ocr_calibration.DEGRADATIONS)
    assert Path(rep["report_path"]).exists()

    # Exercise ocr_gate.ocr_image_tesseract, tesseract_status, _page_identity, and memoized run_ocr_on_page
    ocr_gate._OCR_MEMO.clear()
    monkeypatch.setattr(ocr_gate, "_find_tesseract_cmd", lambda: "tesseract")
    fake_pytess = types.SimpleNamespace(
        pytesseract=types.SimpleNamespace(tesseract_cmd="tesseract"),
        Output=types.SimpleNamespace(DICT="dict"),
        get_tesseract_version=lambda: "5.4.0",
        image_to_data=lambda img, lang="eng", output_type=None, timeout=45.0: {
            "text": ["Hydraulic", "pressure", "3000", "PSI", ""],
            "conf": ["92", "89", "95", "90", "-1"],
        },
    )
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytess)
    ok, status = ocr_gate.tesseract_status()
    assert ok and "5.4.0" in status

    with fitz.open(str(amt_pdf)) as doc:
        txt1, conf1 = ocr_gate.run_ocr_on_page(doc[3], engine="tesseract")
        txt2, conf2 = ocr_gate.run_ocr_on_page(doc[3], engine="tesseract")
    assert txt1 == txt2 == "Hydraulic pressure 3000 PSI"
    assert conf1 == conf2 == pytest.approx(91.5)

    mem_doc = fitz.open()
    mp = mem_doc.new_page()
    assert len(ocr_gate._page_identity(mp)) == 64
    mem_doc.close()


def test_openrouter_and_fastembed_embedders_with_disk_cache(
    lab_cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies OpenRouterEmbedder billing guard, batching, SHA-256 disk cache, and FastEmbedWrapper."""
    with pytest.raises(BillingGuardError):
        OpenRouterEmbedder(model_name="openai/text-embedding-3-small", api_key="sk-test")

    class _EmbedResp:
        def __init__(self, n: int, vec_len: int = 3) -> None:
            data = [{"index": i, "embedding": [float(i + 1)] * vec_len} for i in range(n)]
            self._raw = json.dumps({"data": data}).encode("utf-8")

        def read(self) -> bytes:
            return self._raw

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=30):
        body = json.loads(req.data.decode("utf-8"))
        return _EmbedResp(len(body["input"]))

    emb = OpenRouterEmbedder(
        model_name="nvidia/nemotron-3.5-embed-1b:free",
        dim=8,
        cache_dir=tmp_path / "emb_cache",
        api_key="sk-test",
        urlopen=fake_urlopen,
    )
    v1 = emb.embed_texts(["AN4-12A bolt torque", "FAULT-E409 action"])
    assert v1.shape == (2, 8) and emb.remote_calls == 1 and emb.cache_hits == 0
    v2 = emb.embed_texts(["AN4-12A bolt torque", "FAULT-E409 action"])
    assert np.allclose(v1, v2) and emb.remote_calls == 1 and emb.cache_hits == 2
    assert emb.embed_texts([]).shape == (0, 8)

    # Retry on HTTP 429 + dimension truncation + HTTP 404 retired slug
    attempts = {"n": 0}

    def retry_then_ok(req, timeout=30):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many", {}, io.BytesIO(b""))
        return _EmbedResp(1, vec_len=16)

    emb_retry = OpenRouterEmbedder(
        model_name="nvidia/nemotron-3.5-embed-1b:free",
        dim=4,
        api_key="sk-test",
        urlopen=retry_then_ok,
        sleep=lambda _s: None,
    )
    assert emb_retry.embed_texts(["retry test"]).shape == (1, 4) and attempts["n"] == 2

    emb_404 = OpenRouterEmbedder(
        model_name="nvidia/nemotron-3.5-embed-1b:free",
        dim=4,
        api_key="sk-test",
        urlopen=lambda req, timeout=30: (_ for _ in ()).throw(urllib.error.HTTPError(req.full_url, 404, "Gone", {}, io.BytesIO(b""))),
    )
    with pytest.raises(RuntimeError, match="retired"):
        emb_404.embed_texts(["retired slug"])

    # create_embedder with openrouter_free falls back cleanly when key missing
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    or_cfg = lab_cfg.model_copy(
        update={"index": lab_cfg.index.model_copy(update={"embedding_provider": "openrouter_free"})}
    )
    fallback_emb = create_embedder(or_cfg, force_offline=False)
    assert fallback_emb.model_id.startswith("deterministic-semantic-hash")

    # FastEmbedWrapper via lightweight fake fastembed module
    class _FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def embed(self, texts, batch_size: int = 16):
            for idx, _ in enumerate(texts):
                vec = np.zeros(384, dtype=np.float32)
                vec[idx % 384] = 1.0
                yield vec

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=_FakeTextEmbedding))
    fe = FastEmbedWrapper(model_name="BAAI/bge-small-en-v1.5", dim=384)
    fe_vecs = fe.embed_texts(["alpha", "beta"])
    assert fe_vecs.shape == (2, 384) and fe.model_id == "fastembed:BAAI/bge-small-en-v1.5"


def test_docling_and_local_vision_optional_adapters(
    lab_cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies optional Docling parser backend and local Hugging Face vision captioner adapters."""
    amt_pdf = Path(lab_cfg.corpus.manuals_dir) / "amt_handbook.pdf"

    # 1. Missing docling falls back cleanly; fake docling extracts table rows
    assert router._try_docling_tables(amt_pdf) == {}

    class _FakeDf:
        columns = np.array(["Code", "Meaning"])
        values = np.array([["FAULT-E409", "Low reservoir"]])

    fake_tbl = types.SimpleNamespace(
        prov=[types.SimpleNamespace(page_no=2)],
        export_to_dataframe=lambda: _FakeDf(),
    )
    fake_conv = types.SimpleNamespace(
        DocumentConverter=lambda: types.SimpleNamespace(
            convert=lambda p: types.SimpleNamespace(document=types.SimpleNamespace(tables=[fake_tbl]))
        )
    )
    monkeypatch.setitem(sys.modules, "docling", types.SimpleNamespace(document_converter=fake_conv))
    monkeypatch.setitem(sys.modules, "docling.document_converter", fake_conv)
    dl_cfg = lab_cfg.model_copy(
        update={"ingest": lab_cfg.ingest.model_copy(update={"parser_backend": "docling"})}
    )
    cls_list, regs = router.parse_and_route_pdf(amt_pdf, dl_cfg)
    assert len(cls_list) == 4 and any(r.region_type == "table" and r.page == 2 for r in regs)

    # 2. Local vision model adapter via fake transformers.pipeline
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (240, 240, 240)).save(buf, format="PNG")
    lv_cfg = lab_cfg.model_copy(
        update={"enrich": lab_cfg.enrich.model_copy(update={"vision_provider": "local_vision"})}
    )
    assert captioner._call_local_vision_model(buf.getvalue(), "describe", lv_cfg) is None

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            pipeline=lambda task, model=None: (lambda img, prompt=None, max_new_tokens=200: [{"generated_text": "Starter relay schematic K2"}])
        ),
    )
    assert captioner._call_local_vision_model(buf.getvalue(), "describe", lv_cfg) == "Starter relay schematic K2"


def test_env_check_http_helpers_and_corpus_source_fetchers(
    lab_cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Covers env_check._http_json / _http_error_reason / CheckResult.line and sources.fetch_*."""
    res_line = env_check.CheckResult("vector backend", "OK", "faiss").line()
    assert "OK" in res_line and "vector backend" in res_line

    class _JsonResp:
        def read(self) -> bytes:
            return b'{"ok": true}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=30: _JsonResp())
    assert env_check._http_json("https://openrouter.ai/api/v1/models") == {"ok": True}
    assert env_check._http_json("https://openrouter.ai/api/v1/chat/completions", {"a": 1}, "sk-1") == {"ok": True}

    http_err = urllib.error.HTTPError("https://x", 429, "Too Many", {}, io.BytesIO(b"rate limited"))
    assert "HTTP 429" in env_check._http_error_reason(http_err)
    assert "URLError" in env_check._http_error_reason(urllib.error.URLError("offline"))

    # Restore real urlopen for file:// URI downloads in corpus.sources
    monkeypatch.undo()
    monkeypatch.setattr(urllib.request, "urlopen", urllib.request.urlopen)

    good_pdf = tmp_path / "src.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Manual page 1")
    doc.save(str(good_pdf))
    doc.close()
    bad_html = tmp_path / "bad.html"
    bad_html.write_text("<html>not pdf</html>", encoding="utf-8")

    lab_cfg.corpus.real_sources = [
        CorpusSource(doc_id="good_src", path=str(tmp_path / "dl" / "good.pdf"), url=good_pdf.as_uri()),
        CorpusSource(doc_id="bad_ext", path=str(tmp_path / "dl" / "folder"), url=good_pdf.as_uri()),
        CorpusSource(doc_id="bad_body", path=str(tmp_path / "dl" / "bad.pdf"), url=bad_html.as_uri()),
    ]
    fetched = {r["doc_id"]: r["status"] for r in sources.fetch_configured_sources(lab_cfg)}
    assert fetched == {"good_src": "OK", "bad_ext": "FAIL", "bad_body": "FAIL"}

    ds_file = tmp_path / "dude.json"
    ds_file.write_text("[]", encoding="utf-8")
    tar_file = tmp_path / "dude.tar.gz"
    tar_file.write_bytes(b"tarbytes")
    monkeypatch.setattr(sources, "DUDE_DATASET_URL", ds_file.as_uri())
    monkeypatch.setattr(sources, "DUDE_PDFS_URL", tar_file.as_uri())
    lab_cfg.corpus.downloads_dir = str(tmp_path / "downloads")
    ds_out, tar_out = sources.fetch_dude_sample(lab_cfg)
    assert ds_out.exists() and tar_out.exists()
