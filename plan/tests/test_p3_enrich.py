"""Person 3 Unit & Acceptance Tests: Vision Captioning, Disk Cache & Table Linearization (Stories E3.1, E3.2)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from mmrag.config import AppConfig
from mmrag.enrich.captioner import enrich_figure_units
from mmrag.enrich.table_linearizer import enrich_table_units
from mmrag.extract.figures import extract_figure_units
from mmrag.extract.tables import extract_table_units
from mmrag.ingest.router import parse_and_route_pdf
from mmrag.schemas import TableArtefact


def test_e3_1_figure_captioning_and_disk_cache(lab_cfg: AppConfig) -> None:
    """Story E3.1 Acceptance: Every figure unit has non-empty search_text; re-runs served from cache; image untouched."""
    amt_pdf = Path(lab_cfg.corpus.manuals_dir) / "amt_handbook.pdf"
    _, regions = parse_and_route_pdf(amt_pdf, lab_cfg)
    fig_units, _ = extract_figure_units(regions, lab_cfg)

    # Record SHA-256 of image files before enrichment
    before_hashes = {
        u.unit_id: hashlib.sha256(Path(str(u.artefact)).read_bytes()).hexdigest()
        for u in fig_units
    }

    # First run: populates cache
    enriched_1, cache_1 = enrich_figure_units(fig_units, lab_cfg)
    assert cache_1.misses == len(fig_units)
    for u in enriched_1:
        assert u.search_text.strip() != ""
        assert "amt_handbook" in u.search_text
        # Image bytes on disk must be untouched
        after_hash = hashlib.sha256(Path(str(u.artefact)).read_bytes()).hexdigest()
        assert after_hash == before_hashes[u.unit_id]

    # Second run: 100% served from disk cache
    enriched_2, cache_2 = enrich_figure_units(fig_units, lab_cfg)
    assert cache_2.hits == len(fig_units)
    assert cache_2.misses == 0
    assert [u.search_text for u in enriched_2] == [u.search_text for u in enriched_1]


def test_e3_2_table_linearization_preserves_structured_artefact(lab_cfg: AppConfig) -> None:
    """Story E3.2 Acceptance: Every table unit has search_text; the structured cells are unchanged."""
    amt_pdf = Path(lab_cfg.corpus.manuals_dir) / "amt_handbook.pdf"
    _, regions = parse_and_route_pdf(amt_pdf, lab_cfg)
    tbl_units = extract_table_units(regions, lab_cfg)

    original_artefact = tbl_units[0].artefact
    assert isinstance(original_artefact, TableArtefact)

    enriched_tables = enrich_table_units(tbl_units, lab_cfg)
    t_unit = enriched_tables[0]

    assert t_unit.search_text.strip() != ""
    assert "Part Number = AN4-12A" in t_unit.search_text
    assert "Dry Torque (in-lbs) = 50-70" in t_unit.search_text
    assert "Table 4-1" in t_unit.search_text

    # Structured artefact must remain completely unchanged
    assert isinstance(t_unit.artefact, TableArtefact)
    assert t_unit.artefact.headers == original_artefact.headers
    assert t_unit.artefact.rows == original_artefact.rows


# ---------------------------------------------------------------------------
# Extra Person 3 tests: free-tier resilience, cache integrity, E3.3 rejection
# ---------------------------------------------------------------------------
import io
import json
import urllib.error

import pytest

from mmrag.config import BillingGuardError
from mmrag.enrich import captioner
from mmrag.enrich.cache import CaptionCache
from mmrag.enrich.captioner import assert_free_model_id, enrich_figures
from mmrag.enrich.rejects import reject_unenrichable, write_rejection_report
from mmrag.schemas import Provenance, Unit


class _FakeResp:
    def __init__(self, content: str) -> None:
        self._body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(b""))


def _png(size=(40, 30)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def remote_cfg(lab_cfg: AppConfig, monkeypatch):
    lab_cfg.enrich.vision_provider = "openrouter_free"
    lab_cfg.enrich.openrouter_models = ["a/model:free", "b/model:free"]
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(captioner, "_last_call_at", 0.0)
    return lab_cfg


def _run_remote(cfg, urlopen, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(captioner.urllib.request, "urlopen", urlopen)
    errors: list[str] = []
    out = captioner._call_openrouter_free_vision(
        _png(), "p", cfg, errors=errors, sleep=sleeps.append, clock=lambda: 0.0
    )
    return out, errors, sleeps


def test_paid_model_is_refused() -> None:
    with pytest.raises(BillingGuardError):
        assert_free_model_id("openai/gpt-4o")
    assert_free_model_id("nvidia/nemotron-3.5-super-vl:free")


def test_request_disables_reasoning_and_has_generous_token_budget(remote_cfg, monkeypatch) -> None:
    seen = {}

    def urlopen(req, timeout=0):
        seen.update(json.loads(req.data))
        return _FakeResp("a relay wiring schematic with labelled terminals")

    out, _, _ = _run_remote(remote_cfg, urlopen, monkeypatch)
    assert out
    assert seen["reasoning"] == {"enabled": False}
    assert seen["max_tokens"] >= 300


def test_empty_caption_falls_through_to_next_model_and_is_reported(remote_cfg, monkeypatch) -> None:
    replies = iter([_FakeResp("   "), _FakeResp("good caption of the figure shown")])
    out, errors, _ = _run_remote(remote_cfg, lambda req, timeout=0: next(replies), monkeypatch)
    assert out == "good caption of the figure shown"
    assert any("a/model:free" in e and "empty caption" in e for e in errors)


def test_retired_slug_404_moves_on_without_retrying(remote_cfg, monkeypatch) -> None:
    calls = []

    def urlopen(req, timeout=0):
        model = json.loads(req.data)["model"]
        calls.append(model)
        if model == "a/model:free":
            raise _http_error(404)
        return _FakeResp("caption from the second model works")

    out, errors, _ = _run_remote(remote_cfg, urlopen, monkeypatch)
    assert calls == ["a/model:free", "b/model:free"]
    assert out and any("retired" in e for e in errors)


def test_rate_limit_429_is_retried_with_exponential_backoff(remote_cfg, monkeypatch) -> None:
    n = {"i": 0}

    def urlopen(req, timeout=0):
        n["i"] += 1
        if n["i"] < 3:
            raise _http_error(429)
        return _FakeResp("caption after two rate limits passed")

    out, errors, sleeps = _run_remote(remote_cfg, urlopen, monkeypatch)
    assert out and n["i"] == 3
    assert [s for s in sleeps if s >= 2.0][:2] == [2.0, 4.0]
    assert sum("429" in e for e in errors) == 2


def test_all_models_failing_returns_none_and_names_models(remote_cfg, monkeypatch) -> None:
    def urlopen(req, timeout=0):
        raise _http_error(404)

    out, errors, _ = _run_remote(remote_cfg, urlopen, monkeypatch)
    assert out is None
    assert any("a/model:free" in e for e in errors) and any("b/model:free" in e for e in errors)


def test_calls_are_throttled_under_free_tier_cap(remote_cfg, monkeypatch) -> None:
    now = {"t": 100.0}
    sleeps: list[float] = []

    def sleep(s: float) -> None:
        sleeps.append(s)
        now["t"] += s

    monkeypatch.setattr(captioner.urllib.request, "urlopen", lambda r, timeout=0: _FakeResp("x" * 30))
    for _ in range(2):
        captioner._call_openrouter_free_vision(_png(), "p", remote_cfg, sleep=sleep, clock=lambda: now["t"])
    assert sleeps and sleeps[0] >= captioner.MIN_SECONDS_BETWEEN_CALLS - 0.01


def test_large_images_are_downscaled() -> None:
    from PIL import Image

    big = _png((3200, 1000))
    small = captioner._downscale_png(big)
    with Image.open(io.BytesIO(small)) as img:
        assert max(img.size) <= captioner.MAX_IMAGE_SIDE_PX


def test_stale_offline_caption_is_not_reused_once_a_key_is_available(lab_cfg, monkeypatch) -> None:
    _, regions = parse_and_route_pdf(Path(lab_cfg.corpus.manuals_dir) / "amt_handbook.pdf", lab_cfg)
    fig_units, _ = extract_figure_units(regions, lab_cfg)

    offline = enrich_figures(fig_units, lab_cfg)  # populates cache with local_fallback entries
    assert offline.fallback_used == len(fig_units)

    lab_cfg.enrich.vision_provider = "openrouter_free"
    lab_cfg.enrich.openrouter_models = ["a/model:free"]
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(captioner, "_last_call_at", 0.0)
    monkeypatch.setattr(captioner.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        captioner.urllib.request, "urlopen", lambda r, timeout=0: _FakeResp("real model caption text here")
    )
    online = enrich_figures(fig_units, lab_cfg)
    assert online.remote_calls == len(fig_units)
    assert all("real model caption" in u.search_text for u in online.units if u.type == "figure")


def test_cache_ignores_corrupt_and_empty_entries_and_writes_atomically(tmp_path) -> None:
    cache = CaptionCache(tmp_path)
    (tmp_path / "bad.json").write_text("{not json")
    (tmp_path / "empty.json").write_text(json.dumps({"search_text": "  "}))
    assert cache.get("bad") is None and cache.get("empty") is None
    assert cache.corrupt == 2 and cache.misses == 2
    cache.set("ok", {"search_text": "hello"})
    assert cache.get("ok")["search_text"] == "hello"
    assert not list(tmp_path.glob("*.tmp"))


def _fig_unit(tmp_path, text: str, page: int | None = 2, unit_id: str = "u1") -> Unit:
    img = tmp_path / f"{unit_id}.png"
    img.write_bytes(_png())
    return Unit(
        unit_id=unit_id, type="figure", search_text=text, artefact=str(img),
        provenance=Provenance(doc_id="doc", page=page, unit_id=unit_id),
    )


def test_e3_3_rejects_unenrichable_units_with_reasons_and_report(tmp_path) -> None:
    good = _fig_unit(tmp_path, "Relay control wiring schematic with terminals A4 and B7", unit_id="good")
    no_page = _fig_unit(tmp_path, "Relay control wiring schematic with terminals", page=None, unit_id="nopage")
    empty = _fig_unit(tmp_path, "  ", unit_id="empty")
    generic = _fig_unit(tmp_path, "A technical diagram", unit_id="generic")

    accepted, rejected = reject_unenrichable([good, no_page, empty, generic])
    assert [u.unit_id for u in accepted] == ["good"]
    reasons = {r.unit_id: r.reason for r in rejected}
    assert set(reasons) == {"nopage", "empty", "generic"}
    assert "provenance" in reasons["nopage"]
    assert "empty" in reasons["empty"] and "generic" in reasons["generic"]

    report = tmp_path / "rejected_units.jsonl"
    write_rejection_report(rejected, report)
    assert len(report.read_text().strip().splitlines()) == 3


def test_e3_3_default_corpus_has_no_rejections(lab_cfg: AppConfig) -> None:
    _, regions = parse_and_route_pdf(Path(lab_cfg.corpus.manuals_dir) / "amt_handbook.pdf", lab_cfg)
    figs, _ = extract_figure_units(regions, lab_cfg)
    tbls = enrich_table_units(extract_table_units(regions, lab_cfg), lab_cfg)
    figs, _c = captioner.enrich_figure_units(figs, lab_cfg)
    accepted, rejected = reject_unenrichable(figs + tbls)
    assert rejected == [] and len(accepted) == len(figs) + len(tbls)
