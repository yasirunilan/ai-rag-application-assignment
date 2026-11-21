"""Person 5 Unit & Acceptance Tests: Fail-Closed Answering, Per-Type Eval & Failure Mode Fixes (Stories E6.1–E7.3)."""

from __future__ import annotations

from pathlib import Path

from mmrag.answer.responder import build_answer_from_hits
from mmrag.config import AppConfig
from mmrag.eval.ablations import run_e7_3_failure_mode_experiments
from mmrag.pipeline import MultimodalRAGPipeline
from mmrag.schemas import Provenance, SearchHit, TableArtefact, Unit


def test_e6_1_returns_real_figure_and_structured_table_artefacts(lab_cfg: AppConfig) -> None:
    """Story E6.1 Acceptance: Figure and table answers return the real artefact plus its page citation."""
    pipe = MultimodalRAGPipeline(lab_cfg, force_offline=True)
    pipe.ingest_and_index(persist_index=False)

    # 1. Figure Answer -> actual image file path on disk + page citation
    fig_ans = pipe.answer("Show the engine starter relay control wiring diagram for Model 737-800")
    assert fig_ans.abstained is False
    assert fig_ans.modality == "figure"
    assert fig_ans.citation == "amt_handbook p.3"
    assert isinstance(fig_ans.artefact, str)
    assert Path(fig_ans.artefact).exists()
    assert Path(fig_ans.artefact).suffix == ".png"

    # 2. Table Answer -> structured TableArtefact (headers + rows) + page citation
    tbl_ans = pipe.answer("What is the dry and lubed torque spec table for bolt part number AN4-12A?")
    assert tbl_ans.abstained is False
    assert tbl_ans.modality == "table"
    assert tbl_ans.citation == "amt_handbook p.1"
    assert isinstance(tbl_ans.artefact, TableArtefact)
    assert any("AN4-12A" in row for row in tbl_ans.artefact.rows)


def test_e6_2_fail_closed_abstention_when_provenance_missing(lab_cfg: AppConfig) -> None:
    """Story E6.2 Acceptance: No-provenance hits are never presented as sourced; system abstains cleanly."""
    corrupted_unit = Unit(
        unit_id="amt_handbook__p0000__text_001",
        type="text",
        search_text="Corrupted unit with no resolvable page number.",
        artefact="Corrupted unit with no resolvable page number.",
        provenance=Provenance(
            doc_id="amt_handbook",
            page=None,  # Missing page provenance
            unit_id="amt_handbook__p0000__text_001",
        ),
    )
    bad_hit = SearchHit(
        rank=1,
        unit=corrupted_unit,
        score=0.99,
        vector_score=0.99,
        bm25_score=0.99,
        citation="UNSOURCED",
    )

    ans = build_answer_from_hits("relay wiring", [bad_hit], lab_cfg)
    assert ans.abstained is True
    assert ans.artefact is None
    assert ans.citation is None
    assert "fail-closed" in (ans.abstain_reason or "")


def test_e7_1_and_e7_2_per_type_metrics_and_faithfulness_check(lab_cfg: AppConfig) -> None:
    """Stories E7.1 & E7.2 Acceptance:
    - Per-type metrics (text / figure / table) are reported and saved.
    - Five answers reviewed for faithfulness; any wrong-artefact answer is flagged.
    """
    pipe = MultimodalRAGPipeline(lab_cfg, force_offline=True)
    pipe.ingest_and_index(persist_index=False)

    report = pipe.evaluate(experiment_name="test_eval", save_report=True)
    assert set(report.by_type.keys()) == {"text", "figure", "table"}
    for t_name in ("text", "figure", "table"):
        tm = report.by_type[t_name]
        assert tm.num_queries >= 3
        assert tm.strict_recall_at_k >= 0.75
        assert tm.strict_mrr >= 0.75

    assert len(report.faithfulness_details) == 5
    assert (report.faithfulness_pass_rate or 0.0) >= 0.80


def test_e7_3_two_failure_modes_reproduced_and_fixed_with_before_after_metrics(
    lab_cfg: AppConfig,
) -> None:
    """Story E7.3 Acceptance: Two failure modes fixed, each with before/after metrics."""
    summary = run_e7_3_failure_mode_experiments(lab_cfg, force_offline=True)

    fm1 = summary["failure_mode_1_right_text_wrong_figure"]
    assert fm1["after"]["figure_strict_mrr"] > fm1["before"]["figure_strict_mrr"]
    assert fm1["after"]["faithfulness_pass_rate"] > fm1["before"]["faithfulness_pass_rate"]

    fm2 = summary["failure_mode_2_table_mush_and_exact_codes"]
    assert fm2["after"]["table_strict_mrr"] >= fm2["before"]["table_strict_mrr"]
    assert fm2["after"]["faithfulness_pass_rate"] > fm2["before"]["faithfulness_pass_rate"]
