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


# ---------------------------------------------------------------------------------------------------
# E0.1 — configuration source, .env key loading, model check
# ---------------------------------------------------------------------------------------------------

import io
import os
import tarfile

from PIL import Image, ImageDraw, ImageFont

from mmrag.config import CorpusConfig, CorpusSource, load_config, load_dotenv
from mmrag.corpus.loader import resolve_corpus
from mmrag.corpus.sources import dude_doc_id, download_file, import_dude_sample
from mmrag.env_check import run_environment_check
from mmrag.ingest.ocr_calibration import (
    CalibrationSample,
    apply_thresholds_to_config_yaml,
    score_thresholds,
    sweep_thresholds,
    word_accuracy,
)
from mmrag.ingest.ocr_gate import compute_alnum_token_ratio, tesseract_status
from mmrag.ingest.router import parse_and_route_pdf

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_e0_1_repo_config_loads_and_every_slug_is_free() -> None:
    cfg = load_config(REPO_ROOT / "config.yaml")
    assert cfg.enrich.openrouter_models
    assert all(m.endswith(":free") or m == "openrouter/free" for m in cfg.enrich.openrouter_models)
    assert cfg.corpus.max_pages == 250


def test_e0_1_dotenv_loads_key_without_overriding_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text('# comment\nexport OPENROUTER_API_KEY="sk-or-test"\nMMRAG_KEEP=from_file\n\nBROKEN_LINE\n')
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("MMRAG_KEEP", "from_env")
    loaded = load_dotenv(env)
    assert loaded == ["OPENROUTER_API_KEY"]
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-test"
    assert os.environ["MMRAG_KEEP"] == "from_env"
    assert load_dotenv(tmp_path / "missing.env") == []


def test_e0_1_missing_key_stops_with_clear_message(lab_cfg: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    lab_cfg.enrich.vision_provider = "openrouter_free"

    def no_network(*_a, **_k):  # the missing-key path must not touch the network
        raise AssertionError("network called without a key")

    code, results = run_environment_check(lab_cfg, fetch=no_network)
    assert code == 2
    key_line = next(r for r in results if r.name == "OPENROUTER_API_KEY")
    assert key_line.status == "FAIL" and "openrouter.ai/keys" in key_line.detail

    code_off, results_off = run_environment_check(lab_cfg, offline=True, fetch=no_network)
    assert code_off == 0
    assert all(r.status != "FAIL" for r in results_off)


def _fake_openrouter(catalogue: list[dict], reply: str = "square"):
    calls: list[str] = []

    def fetch(url: str, payload, api_key):
        calls.append(url)
        if url.endswith("/models"):
            return {"data": catalogue}
        if url.endswith("/key"):
            assert api_key == "sk-or-test"
            return {"data": {"usage": 0, "is_free_tier": True}}
        if url.endswith("/chat/completions"):
            assert payload["model"].endswith(":free") or payload["model"] == "openrouter/free"
            return {"model": payload["model"], "choices": [{"message": {"content": reply}}]}
        raise AssertionError(url)

    return fetch, calls


def test_e0_1_model_check_flags_retired_and_paid_slugs(lab_cfg: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    lab_cfg.enrich.vision_provider = "openrouter_free"
    lab_cfg.enrich.openrouter_models = ["retired/model:free", "live/vision:free", "sneaky/paid:free", "text/only:free"]
    free_img = {"pricing": {"prompt": "0", "completion": "0"}, "architecture": {"input_modalities": ["text", "image"]}}
    fetch, calls = _fake_openrouter(
        [
            {"id": "live/vision:free", **free_img},
            {"id": "sneaky/paid:free", "pricing": {"prompt": "0.000001", "completion": "0"},
             "architecture": {"input_modalities": ["image"]}},
            {"id": "text/only:free", "pricing": {"prompt": "0"}, "architecture": {"input_modalities": ["text"]}},
        ]
    )
    code, results = run_environment_check(lab_cfg, fetch=fetch)
    by_name = {r.name: r for r in results}
    assert by_name["vision slug retired/model:free"].status == "FAIL"
    assert "retired" in by_name["vision slug retired/model:free"].detail
    assert by_name["vision slug sneaky/paid:free"].status == "FAIL"
    assert by_name["vision slug text/only:free"].status == "FAIL"
    assert by_name["vision slug live/vision:free"].status == "OK"
    assert by_name["vision + chat call"].status == "OK" and "live/vision:free" in by_name["vision + chat call"].detail
    assert sum(u.endswith("/chat/completions") for u in calls) == 1  # one tiny call, not one per slug
    assert code == 1  # any FAIL fails the check

    # Swapping the slug is a configuration change only
    lab_cfg.enrich.openrouter_models = ["live/vision:free"]
    code_ok, _ = run_environment_check(lab_cfg, fetch=fetch)
    assert code_ok == 0


# ---------------------------------------------------------------------------------------------------
# E0.2 — real-source corpus expansion under the max_pages budget
# ---------------------------------------------------------------------------------------------------

def _make_pdf(path: Path, n_pages: int, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        page.insert_text(fitz.Point(72, 72), f"{label} original page {i + 1} hydraulic servicing text", fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


def _with_real_sources(cfg: AppConfig, tmp_path: Path, sources: list[dict], max_pages: int) -> AppConfig:
    cfg.corpus.real_sources = [CorpusSource(**s) for s in sources]
    cfg.corpus.slices_dir = str(tmp_path / "slices")
    cfg.corpus.max_pages = max_pages
    return cfg


def test_e0_2_real_sources_truncate_from_page_one_under_budget(lab_cfg: AppConfig, tmp_path: Path) -> None:
    faa = _make_pdf(tmp_path / "dl" / "FAA-H-8083-30B.pdf", 10, "FAA")
    extra = _make_pdf(tmp_path / "dl" / "extra.pdf", 4, "EXTRA")
    cfg = _with_real_sources(
        lab_cfg, tmp_path,
        [
            {"doc_id": "faa_amt_general", "path": str(faa)},
            {"doc_id": "missing_manual", "path": str(tmp_path / "dl" / "nope.pdf")},
            {"doc_id": "extra_manual", "path": str(extra)},
        ],
        max_pages=6 + 3,  # 6 core pages + 3 real pages
    )
    res = resolve_corpus(cfg)
    by_id = {d.doc_id: d for d in res.documents}

    assert res.total_pages == 9
    assert by_id["amt_handbook"].origin == "core" and by_id["dude_hydraulic_manual"].origin == "core"
    faa_doc = by_id["faa_amt_general"]
    assert faa_doc.loaded_pages == 3 and faa_doc.source_pages == 10 and faa_doc.truncated
    assert Path(faa_doc.path).stem == "faa_amt_general"  # router doc_id == configured doc_id
    with fitz.open(faa_doc.path) as sliced:
        assert len(sliced) == 3
        assert "original page 2 " in sliced[1].get_text()  # slice page 2 IS source page 2

    skipped = {s["doc_id"]: s["reason"] for s in res.skipped}
    assert skipped == {"missing_manual": "missing_file", "extra_manual": "page_budget_exhausted"}
    manifest = json.loads((Path(cfg.corpus.slices_dir) / "corpus_manifest.json").read_text())
    assert manifest["total_pages"] == 9

    assert [p.stem for p in load_corpus_manuals(cfg)] == ["amt_handbook", "dude_hydraulic_manual", "faa_amt_general"]


def test_e0_2_directory_source_and_per_doc_cap(lab_cfg: AppConfig, tmp_path: Path) -> None:
    _make_pdf(tmp_path / "dude" / "abc123.pdf", 5, "A")
    _make_pdf(tmp_path / "dude" / "def456.pdf", 2, "B")
    cfg = _with_real_sources(
        lab_cfg, tmp_path, [{"doc_id": "dude", "path": str(tmp_path / "dude"), "max_pages": 3}], max_pages=250
    )
    pages = resolve_corpus(cfg).pages_by_doc()
    assert pages["dude_abc123"] == 3 and pages["dude_def456"] == 2


def test_e0_2_doc_id_rules(lab_cfg: AppConfig, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CorpusSource(doc_id="bad id/../x", path="x.pdf")
    with pytest.raises(ValueError):
        CorpusConfig(real_sources=[{"doc_id": "a", "path": "x.pdf"}, {"doc_id": "a", "path": "y.pdf"}])
    clash = _make_pdf(tmp_path / "clash.pdf", 1, "C")
    cfg = _with_real_sources(lab_cfg, tmp_path, [{"doc_id": "amt_handbook", "path": str(clash)}], max_pages=250)
    with pytest.raises(ValueError, match="used twice"):
        resolve_corpus(cfg)


def test_e0_2_gold_questions_restricted_to_loaded_pages(lab_cfg: AppConfig, tmp_path: Path) -> None:
    faa = _make_pdf(tmp_path / "faa.pdf", 10, "FAA")
    cfg = _with_real_sources(lab_cfg, tmp_path, [{"doc_id": "faa", "path": str(faa)}], max_pages=6 + 3)
    extra_gold = tmp_path / "faa_gold.jsonl"
    rows = [
        {"query_id": "FAA_01", "query": "q1", "gold_doc_id": "faa", "gold_pages": [2], "expected_type": "text"},
        {"query_id": "FAA_02", "query": "q2", "gold_doc_id": "faa", "gold_pages": [7], "expected_type": "text"},
        {"query_id": "FAA_03", "query": "q3", "gold_doc_id": "not_loaded", "gold_pages": [1], "expected_type": "text"},
    ]
    extra_gold.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    cfg.corpus.extra_gold_paths = [str(extra_gold), str(tmp_path / "absent.jsonl")]

    all_q = load_gold_questions(cfg, restrict_to_corpus=False)
    in_corpus = load_gold_questions(cfg)
    assert len(all_q) == 12 + 3
    ids = {q.query_id for q in in_corpus}
    assert "FAA_01" in ids and "FAA_02" not in ids and "FAA_03" not in ids
    assert len(in_corpus) == 12 + 1

    extra_gold.write_text(json.dumps({**rows[0], "query_id": "Q_FIG_01"}) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_gold_questions(cfg)


def test_e0_2_dude_import_converts_zero_indexed_pages_and_matches_loader_ids(lab_cfg: AppConfig, tmp_path: Path) -> None:
    raw_a, raw_b, raw_c = "a" * 32, "b" * 32, "c" * 32
    pdfs = {raw_a: _make_pdf(tmp_path / "src" / f"{raw_a}.pdf", 3, "A"),
            raw_b: _make_pdf(tmp_path / "src" / f"{raw_b}.pdf", 2, "B"),
            raw_c: _make_pdf(tmp_path / "src" / f"{raw_c}.pdf", 9, "C")}
    tar_path = tmp_path / "DUDE_sample_pdfs.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for raw, p in pdfs.items():
            tar.add(p, arcname=f"DUDE_sample_pdfs/{raw}.pdf")
        info = tarfile.TarInfo("DUDE_sample_pdfs/link.pdf")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)

    def q(qid, doc, pages, atype="extractive"):
        return {"questionId": f"{doc}_{qid:012d}", "question": f"question {qid}", "answers": ["ans"],
                "answers_page_bounding_boxes": [[{"left": 1, "top": 1, "width": 1, "height": 1, "page": p} for p in pages]],
                "answer_type": atype, "docId": doc}

    dataset = [q(1, raw_a, [0]), q(2, raw_a, [2]), q(3, raw_b, [1]), q(4, raw_b, [0], "not-answerable"),
               q(5, raw_c, [8]), {**q(6, raw_a, []), "answers_page_bounding_boxes": [[]]}]
    ds_path = tmp_path / "DUDE_sample_dataset.json"
    ds_path.write_text(json.dumps(dataset))

    out_dir, gold_out = tmp_path / "dl" / "dude", tmp_path / "dl" / "dude_gold.jsonl"
    summary = import_dude_sample(ds_path, tar_path, out_dir, gold_out, max_pages=6)
    assert summary["num_pages"] == 5 and summary["num_documents"] == 2  # 9-page doc does not fit the budget
    assert not (out_dir / "link.pdf").exists()

    gold = [json.loads(line) for line in gold_out.read_text().splitlines()]
    assert sorted((g["gold_doc_id"], g["gold_pages"][0]) for g in gold) == sorted(
        [(dude_doc_id(raw_a), 1), (dude_doc_id(raw_a), 3), (dude_doc_id(raw_b), 2)]
    )

    cfg = _with_real_sources(lab_cfg, tmp_path, [{"doc_id": "dude", "path": str(out_dir)}], max_pages=250)
    cfg.corpus.extra_gold_paths = [str(gold_out)]
    assert {dude_doc_id(raw_a), dude_doc_id(raw_b)} <= set(resolve_corpus(cfg).pages_by_doc())
    assert len(load_gold_questions(cfg)) == 12 + 3


def test_e0_2_download_rejects_non_pdf_body(tmp_path: Path) -> None:
    html = tmp_path / "blocked.html"
    html.write_text("<html>403 Forbidden</html>")
    with pytest.raises(ValueError, match="did not return a PDF"):
        download_file(html.as_uri(), tmp_path / "out" / "manual.pdf")
    assert not (tmp_path / "out" / "manual.pdf").exists()
    good = _make_pdf(tmp_path / "good.pdf", 1, "G")
    assert download_file(good.as_uri(), tmp_path / "out" / "manual.pdf").read_bytes().startswith(b"%PDF")


# ---------------------------------------------------------------------------------------------------
# E1.2 — scanned detection, fail-closed OCR, gate calibration
# ---------------------------------------------------------------------------------------------------

def _text_image(lines: list[str], size=(1700, 2200)) -> Image.Image:
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=42)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for i, line in enumerate(lines):
        draw.text((120, 150 + i * 70), line, fill=(0, 0, 0), font=font)
    return img


def _scan_pdf(path: Path, img: Image.Image, hidden_text: str | None = None) -> Path:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=buf.getvalue())
    if hidden_text:
        page.insert_text(fitz.Point(40, 60), hidden_text, fontsize=8, render_mode=3)  # invisible OCR layer
    doc.save(str(path))
    doc.close()
    return path


CLEAN_LINES = [
    "Hydraulic system servicing procedure",
    "Check the reservoir fluid level before flight.",
    "Nominal system pressure is 3000 PSI at the pump.",
    "Torque the fitting to 135 inch pounds and safety wire it.",
    "Record the inspection in the aircraft maintenance log.",
]


def test_e1_2_invisible_text_layer_is_still_treated_as_scanned(tmp_path: Path) -> None:
    hidden = "garbage ocr layer " * 20
    pdf = _scan_pdf(tmp_path / "scan_with_layer.pdf", _text_image(CLEAN_LINES), hidden_text=hidden)
    with fitz.open(str(pdf)) as doc:
        assert len(doc[0].get_text().strip()) > 50  # a naive char count would call this digital
        scanned, visible_chars, coverage = is_scanned_page(doc[0])
    assert scanned and visible_chars == 0 and coverage >= 0.60


def test_e1_2_heuristic_fallback_never_invents_text(lab_cfg: AppConfig, tmp_path: Path) -> None:
    pdf = _scan_pdf(tmp_path / "clean_scan.pdf", _text_image(CLEAN_LINES))
    with fitz.open(str(pdf)) as doc:
        text, conf = run_ocr_on_page(doc[0], engine="heuristic_fallback")
    # Without a real OCR engine the page is either empty or simulated noise -> never indexed.
    assert text == "" or compute_garbled_char_ratio(text) > lab_cfg.ingest.ocr_quality_gate.max_garbled_char_ratio
    passed, _, _, reasons = evaluate_ocr_quality_gate("clean_scan", 1, str(pdf), text, conf, lab_cfg)
    assert not passed and reasons


def test_e1_2_router_keeps_failed_scans_out_of_the_index(lab_cfg: AppConfig) -> None:
    lab_cfg.ingest.ocr_engine = "heuristic_fallback"
    amt_pdf = Path(lab_cfg.corpus.manuals_dir) / "amt_handbook.pdf"
    classes, regions = parse_and_route_pdf(amt_pdf, lab_cfg)
    p4 = next(c for c in classes if c.page == 4)
    assert p4.is_scanned and p4.route_decision == "ocr_review_queue"
    assert not [r for r in regions if r.page == 4]
    assert sum(c.route_decision == "ocr_review_queue" for c in classes) == 1  # the number failed is reported


def test_e1_2_garbled_ratio_accepts_typographic_unicode() -> None:
    text = "The mechanic’s “final check” — torque ≤ 35 in-lb; see §43.13 and µF ratings… Ölstand prüfen."
    assert compute_garbled_char_ratio(text) == 0.0
    assert compute_garbled_char_ratio("�� ~~ || __ <<>>") > 0.9
    assert compute_alnum_token_ratio("") == 0.0


HAS_TESSERACT = tesseract_status()[0]


@pytest.mark.skipif(not HAS_TESSERACT, reason="Tesseract binary not installed")
def test_e1_2_tesseract_passes_clean_scan_and_rejects_degraded_scan(lab_cfg: AppConfig, tmp_path: Path) -> None:
    pdf = _scan_pdf(tmp_path / "clean_scan.pdf", _text_image(CLEAN_LINES))
    with fitz.open(str(pdf)) as doc:
        text, conf = run_ocr_on_page(doc[0], engine="tesseract")
    passed, _, _, reasons = evaluate_ocr_quality_gate("clean_scan", 1, str(pdf), text, conf, lab_cfg)
    assert passed, (reasons, text[:120])
    assert word_accuracy(" ".join(CLEAN_LINES), text) > 0.9

    amt_pdf = Path(lab_cfg.corpus.manuals_dir) / "amt_handbook.pdf"
    with fitz.open(str(amt_pdf)) as doc:
        bad_text, bad_conf = run_ocr_on_page(doc[3], engine="tesseract")
    bad_passed, _, _, bad_reasons = evaluate_ocr_quality_gate("amt_handbook", 4, str(amt_pdf), bad_text, bad_conf, lab_cfg)
    assert not bad_passed and bad_reasons


def _sample(conf: float, garbled: float, alnum: float, acc: float, i: int) -> CalibrationSample:
    return CalibrationSample(f"s{i}", "d", 1, "x", conf, garbled, alnum, acc, acc >= 0.85, 50, "text")


def test_e1_2_calibration_sweep_prefers_zero_bad_pages_indexed() -> None:
    assert word_accuracy("torque the AN4-12A bolt", "torque the AN4-12A bolt") == 1.0
    assert word_accuracy("torque the AN4-12A bolt", "tOrque") == 0.25
    samples = [_sample(c, 0.02, 0.9, 0.97, i) for i, c in enumerate([88, 91, 84, 79])]
    samples += [_sample(c, 0.02, 0.9, 0.40, 10 + i) for i, c in enumerate([62, 55, 41])]
    samples += [_sample(70, 0.30, 0.45, 0.20, 20)]  # confident but garbled
    loose = score_thresholds(samples, 40, 0.5, 0.0)
    assert loose["bad_pages_indexed"] == 4
    best = sweep_thresholds(samples, min_precision=1.0)
    rec = best["recommended"]
    assert best["constraint_met"] and rec["bad_pages_indexed"] == 0 and rec["recall"] == 1.0
    assert 62 < rec["min_mean_confidence"] <= 79


def test_e1_2_apply_thresholds_rewrites_only_gate_keys(tmp_path: Path) -> None:
    cfg_copy = tmp_path / "config.yaml"
    original = (REPO_ROOT / "config.yaml").read_text(encoding="utf-8")
    cfg_copy.write_text(original, encoding="utf-8")
    changed = apply_thresholds_to_config_yaml(
        cfg_copy,
        {"min_mean_confidence": 72.5, "max_garbled_char_ratio": 0.08, "min_alnum_token_ratio": 0.6, "min_word_count": 7},
    )
    assert sorted(changed) == ["max_garbled_char_ratio", "min_alnum_token_ratio", "min_mean_confidence", "min_word_count"]
    cfg = load_config(cfg_copy)
    gate = cfg.ingest.ocr_quality_gate
    assert (gate.min_mean_confidence, gate.max_garbled_char_ratio, gate.min_alnum_token_ratio) == (72.5, 0.08, 0.6)
    assert gate.min_word_count == 7
    new_text = cfg_copy.read_text(encoding="utf-8")
    assert len(new_text.splitlines()) == len(original.splitlines())
    assert "# 0-100 scale from Tesseract word-level confidences" in new_text
