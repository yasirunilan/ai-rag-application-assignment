"""Person 5 Unit & Acceptance Tests: Fail-Closed Answering, Layers A/B/C Eval & Failure Mode Fixes (Stories E6.1–E6.2, E7.1–E7.3 / E8.1–E8.7)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from mmrag.answer.responder import build_answer_from_hits
from mmrag.api.server import create_app
from mmrag.config import AppConfig, BillingGuardError
from mmrag.eval.ablations import run_e7_3_failure_mode_experiments
from mmrag.eval.faithfulness import run_faithfulness_spotcheck
from mmrag.pipeline import MultimodalRAGPipeline
from mmrag.retrieve.retriever import HybridRetriever
from mmrag.index.embedder import create_embedder
from mmrag.index.store import MixedVectorStore
from mmrag.schemas import GoldQuestion, Provenance, SearchHit, TableArtefact, Unit


def test_e6_1_returns_real_figure_and_structured_table_artefacts(lab_cfg: AppConfig) -> None:
    """Story E6.1 Acceptance: Figure and table answers return the real artefact + page citation + inline-cited grounded answer."""
    pipe = MultimodalRAGPipeline(lab_cfg, force_offline=True)
    pipe.ingest_and_index(persist_index=False)

    # 1. Figure Answer -> actual image file path on disk + page citation
    fig_ans = pipe.answer("Show the engine starter relay control wiring diagram for Model 737-800")
    assert fig_ans.abstained is False
    assert fig_ans.modality == "figure"
    assert fig_ans.citation == "amt_handbook p.3"
    assert isinstance(fig_ans.artefact, str)
    assert Path(fig_ans.artefact).exists() and Path(fig_ans.artefact).suffix == ".png"
    assert fig_ans.grounded_answer and "[amt_handbook p.3]" in fig_ans.grounded_answer
    assert set(fig_ans.timings_ms) >= {"retrieval_ms", "generation_ms", "total_ms"}

    # 2. Table Answer -> structured TableArtefact (headers + rows) + page citation
    tbl_ans = pipe.answer("What is the dry and lubed torque spec table for bolt part number AN4-12A?")
    assert tbl_ans.abstained is False
    assert tbl_ans.modality == "table"
    assert tbl_ans.citation == "amt_handbook p.1"
    assert isinstance(tbl_ans.artefact, TableArtefact)
    assert any("AN4-12A" in row for row in tbl_ans.artefact.rows)
    assert tbl_ans.grounded_answer and "[amt_handbook p.1]" in tbl_ans.grounded_answer


def test_e6_1_optional_remote_generation_cites_inline_and_falls_back(
    lab_cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Story E6.1: Optional `:free` generator cites `[doc p.N]` inline and falls back cleanly on error."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cfg = lab_cfg.model_copy(
        update={"answer": lab_cfg.answer.model_copy(update={"enable_remote_generation": True})}
    )
    unit = Unit(
        unit_id="amt_handbook__p0001__text_001",
        type="text",
        search_text="Nominal pressure is 3,000 PSI.",
        artefact="Nominal pressure is 3,000 PSI.",
        provenance=Provenance(doc_id="amt_handbook", page=1, unit_id="amt_handbook__p0001__text_001"),
    )
    hit = SearchHit(rank=1, unit=unit, score=0.9, citation="amt_handbook p.1")

    class _FakeResp:
        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "System operates at 3,000 PSI."}}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    ans = build_answer_from_hits("operating pressure?", [hit], cfg, offline=False, urlopen=lambda r, timeout=0: _FakeResp())
    assert ans.model_calls == 1
    assert ans.grounded_answer == "System operates at 3,000 PSI. [amt_handbook p.1]"

    # Paid slug bypassing config construction is still blocked at call time
    bad_cfg = cfg.model_copy()
    bad_cfg.answer.__dict__["text_model"] = "openai/gpt-4o"
    with pytest.raises(BillingGuardError):
        build_answer_from_hits("operating pressure?", [hit], bad_cfg, offline=False)


def test_e6_2_fail_closed_abstention_when_provenance_missing(lab_cfg: AppConfig) -> None:
    """Story E6.2 Acceptance: No-provenance, mismatched-id, low-score, or empty queries abstain cleanly."""
    no_page = Unit(
        unit_id="amt_handbook__p0000__text_001",
        type="text",
        search_text="Corrupted unit with no resolvable page number.",
        artefact="Corrupted unit with no resolvable page number.",
        provenance=Provenance(doc_id="amt_handbook", page=None, unit_id="amt_handbook__p0000__text_001"),
    )
    lying_id = Unit(
        unit_id="amt_handbook__p0002__text_001",
        type="text",
        search_text="Mismatched provenance unit_id.",
        artefact="Mismatched provenance unit_id.",
        provenance=Provenance(doc_id="amt_handbook", page=2, unit_id="different_id"),
    )
    missing_png = Unit(
        unit_id="amt_handbook__p0003__figure_099",
        type="figure",
        search_text="Missing image file on disk.",
        artefact="nonexistent_dir/missing.png",
        provenance=Provenance(doc_id="amt_handbook", page=3, unit_id="amt_handbook__p0003__figure_099"),
    )
    hits = [
        SearchHit(rank=i, unit=u, score=0.95, citation="UNSOURCED")
        for i, u in enumerate((no_page, lying_id, missing_png), start=1)
    ]

    ans = build_answer_from_hits("relay wiring", hits, lab_cfg)
    assert ans.abstained is True
    assert ans.artefact is None and ans.citation is None
    assert "fail-closed" in (ans.abstain_reason or "")

    assert build_answer_from_hits("   ", hits, lab_cfg).abstained is True

    good_unit = Unit(
        unit_id="amt_handbook__p0001__text_001",
        type="text",
        search_text="Valid text",
        artefact="Valid text",
        provenance=Provenance(doc_id="amt_handbook", page=1, unit_id="amt_handbook__p0001__text_001"),
    )
    low_hit = SearchHit(rank=1, unit=good_unit, score=0.01, citation="amt_handbook p.1")
    low_ans = build_answer_from_hits("unrelated query", [low_hit], lab_cfg)
    assert low_ans.abstained is True and "below_threshold" in (low_ans.abstain_reason or "")


def test_e7_1_and_e7_2_layers_a_b_c_and_faithfulness_check(lab_cfg: AppConfig) -> None:
    """Stories E7.1–E7.2 / E8.1–E8.6: Reports Layer A (R@1/3/5, P@k, MRR + per_question), Layer B, Layer C, and faithfulness."""
    pipe = MultimodalRAGPipeline(lab_cfg, force_offline=True)
    pipe.ingest_and_index(persist_index=True)

    report = pipe.evaluate(experiment_name="test_eval", save_report=True)
    assert set(report.by_type.keys()) == {"text", "figure", "table"}
    for t_name in ("text", "figure", "table"):
        tm = report.by_type[t_name]
        assert tm.num_queries >= 3
        assert tm.strict_recall_at_1 >= 0.75
        assert tm.strict_recall_at_3 >= 0.75
        assert tm.strict_recall_at_k >= 0.75
        assert tm.strict_precision_at_k > 0.0
        assert tm.strict_mrr >= 0.75

    # Per-question trace recorded (E8.4)
    assert len(report.per_question) == 12
    assert all(q.query_id and q.strict_rank is not None for q in report.per_question)

    # Layer B & Layer C populated (E8.2 & E8.3)
    assert report.layer_b is not None
    assert report.layer_b.citation_accuracy == 1.0
    assert report.layer_b.wrong_source_rate == 0.0
    assert report.layer_b.artefact_correctness == 1.0
    assert report.layer_b.answer_correctness >= 0.90

    assert report.layer_c is not None
    assert report.layer_c.index_size_bytes > 0
    assert report.layer_c.ingestion_time_ms > 0.0
    assert report.layer_c.queries_per_min_under_cap == 20.0

    # Faithfulness check (5 answers) + explicit wrong-figure detection
    assert len(report.faithfulness_details) == 5
    assert (report.faithfulness_pass_rate or 0.0) >= 0.80

    # Inject a wrong-page figure expectation to prove DANGEROUS_WRONG_FIGURE_DRIFT is caught
    assert pipe.retriever is not None
    bad_gold = [
        GoldQuestion(
            query_id="DRIFT_TEST",
            query="Show the engine starter relay control wiring diagram for Model 737-800",
            gold_doc_id="amt_handbook",
            gold_pages=[2],  # Starter relay is on p.3, so p.2 expectation triggers wrong-figure flag
            expected_type="figure",
            gold_artefact_hint="Figure 9-12",
        )
    ]
    drift_check = run_faithfulness_spotcheck(pipe.retriever, bad_gold, lab_cfg, sample_size=1)
    assert drift_check["wrong_figure_count"] == 1
    assert drift_check["reviews"][0]["flag"] == "DANGEROUS_WRONG_FIGURE_DRIFT"


def test_e7_3_two_failure_modes_reproduced_and_fixed_with_before_after_metrics(
    lab_cfg: AppConfig,
) -> None:
    """Stories E7.3 / E8.5 & E8.7: Two failure modes fixed with before/after metrics, moved QIDs, and cost levers."""
    summary = run_e7_3_failure_mode_experiments(lab_cfg, force_offline=True)

    fm1 = summary["failure_mode_1_right_text_wrong_figure"]
    assert fm1["after"]["figure_strict_mrr"] > fm1["before"]["figure_strict_mrr"]
    assert fm1["after"]["faithfulness_pass_rate"] > fm1["before"]["faithfulness_pass_rate"]
    assert len(fm1["improved_question_ids"]) >= 1
    assert fm1["regressed_question_ids"] == []

    fm2 = summary["failure_mode_2_table_mush_and_exact_codes"]
    assert fm2["after"]["table_strict_mrr"] > fm2["before"]["table_strict_mrr"]
    assert fm2["after"]["faithfulness_pass_rate"] > fm2["before"]["faithfulness_pass_rate"]
    assert len(fm2["improved_question_ids"]) >= 1
    assert fm2["regressed_question_ids"] == []

    levers = summary["cost_levers"]
    assert levers["selective_first_run_vision_calls"] < levers["naive_every_page_vision_calls"]
    assert levers["cached_rerun_vision_calls"] == 0
    assert levers["batch_16_unit_embed_calls"] < levers["unbatched_unit_embed_calls"]


def test_e5_3_api_type_filter_runs_inside_retriever(lab_cfg: AppConfig) -> None:
    """Regression test: API `/query` passes `type_filter` into `retriever.retrieve`, not post-filtering."""
    MultimodalRAGPipeline(lab_cfg, force_offline=True).ingest_and_index(persist_index=True)
    client = TestClient(create_app(lab_cfg, offline=True))
    # Even on a prose-leaning query, requesting `type_filter="figure"` searches inside figure units
    res = client.post(
        "/query",
        json={"question": "Model 737-800 starter switch N2 cutout", "type_filter": "figure"},
    )
    body = res.json()
    assert res.status_code == 200 and not body["abstained"]
    assert body["modality"] == "figure"
    assert all(s["type"] == "figure" for s in body["sources"])
