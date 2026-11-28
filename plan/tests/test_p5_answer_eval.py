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


# =============================================================================================
# Hardening tests (E6.1, E6.2, E7.1, E7.2, E7.3)
# =============================================================================================
import json

import pytest

from mmrag.answer.responder import (
    ABSTAIN_NOT_RELEVANT,
    ABSTAIN_UNSOURCED,
    match_table_rows,
    query_term_coverage,
)
from mmrag.corpus.loader import load_gold_questions
from mmrag.eval.faithfulness import (
    FLAG_ABSTAINED,
    FLAG_WRONG_FIGURE,
    judge_answer,
    run_faithfulness_spotcheck,
)
from mmrag.eval.metrics import (
    _main as metrics_main,
    compare_reports,
    evaluate_retrieval_by_type,
    failing_question_ids,
    load_eval_report,
)
from mmrag.schemas import AnswerResponse, GoldQuestion


@pytest.fixture()
def indexed_pipe(lab_cfg: AppConfig) -> MultimodalRAGPipeline:
    pipe = MultimodalRAGPipeline(lab_cfg, force_offline=True)
    pipe.ingest_and_index(persist_index=False)
    return pipe


def _unit(uid: str, u_type: str, page, text: str, artefact=None, doc: str = "amt_handbook", **prov) -> Unit:
    return Unit(
        unit_id=uid,
        type=u_type,
        search_text=text,
        artefact=artefact if artefact is not None else text,
        provenance=Provenance(doc_id=doc, page=page, unit_id=uid, **prov),
    )


def _hit(rank: int, unit: Unit, score: float) -> SearchHit:
    return SearchHit(rank=rank, unit=unit, score=score, citation=unit.provenance.citation)


# ---- E6.1: real artefact + extractive grounded answer -----------------------------------------
def test_e6_1_table_answer_returns_exact_matching_row_with_inline_citation(indexed_pipe) -> None:
    ans = indexed_pipe.answer("What is the dry and lubed torque spec table for bolt part number AN4-12A?")
    assert isinstance(ans.artefact, TableArtefact)
    assert len(ans.matched_rows) == 1
    row = ans.artefact.rows[ans.matched_rows[0]]
    assert row[0] == "AN4-12A" and "50-70" in row and "35-45" in row
    # Grounded answer quotes the row, header-bound, and cites doc > page > unit inline
    assert ans.citation_path == "amt_handbook > p.1 > amt_handbook__p0001__table_001"
    assert "Part Number = AN4-12A" in ans.grounded_answer
    assert "Dry Torque (in-lbs) = 50-70" in ans.grounded_answer
    assert f"[{ans.citation_path}]" in ans.grounded_answer
    assert "NAS1304-15" not in ans.grounded_answer  # other rows are not mixed in


def test_e6_1_text_answer_quotes_cited_span_verbatim(indexed_pipe) -> None:
    ans = indexed_pipe.answer(
        "What nitrogen pre-charge pressure is required on the Model A320 hydraulic accumulator at 20 degrees Celsius?"
    )
    assert ans.modality == "text" and ans.citation == "dude_hydraulic_manual p.1"
    quoted = ans.grounded_answer.replace(f"[{ans.citation_path}]", "").strip()
    assert "1,450 PSI" in quoted
    assert quoted in " ".join(str(ans.artefact).split())  # no paraphrase: a verbatim span of the unit


def test_e6_1_figure_answer_carries_caption_and_full_citation(indexed_pipe) -> None:
    ans = indexed_pipe.answer("Show the engine starter relay control wiring diagram for Model 737-800")
    assert ans.modality == "figure"
    assert ans.citation_path == "amt_handbook > p.3 > amt_handbook__p0003__figure_001"
    assert "Figure 9-14" in ans.grounded_answer and ans.citation_path in ans.grounded_answer


def test_e6_1_modality_promotion_never_swaps_in_a_figure_from_another_page(lab_cfg: AppConfig, tmp_path: Path) -> None:
    img = tmp_path / "fig.png"
    img.write_bytes(b"\x89PNG")
    text_p3 = _unit("amt_handbook__p0003__text_001", "text", 3, "Starter relay wiring for the engine start circuit.")
    fig_p2 = _unit("amt_handbook__p0002__figure_001", "figure", 2, "Generator relay wiring schematic", str(img))
    fig_p3 = _unit("amt_handbook__p0003__figure_001", "figure", 3, "Starter relay wiring schematic", str(img))
    q = "Show the starter relay wiring diagram"

    other_page = build_answer_from_hits(q, [_hit(1, text_p3, 1.0), _hit(2, fig_p2, 0.95)], lab_cfg)
    assert other_page.modality == "text" and other_page.citation == "amt_handbook p.3"

    same_page = build_answer_from_hits(q, [_hit(1, text_p3, 1.0), _hit(2, fig_p2, 0.95), _hit(3, fig_p3, 0.9)], lab_cfg)
    assert same_page.modality == "figure" and same_page.source_unit.unit_id == fig_p3.unit_id


def test_e6_1_match_table_rows_by_identifier_and_by_terms() -> None:
    tbl = TableArtefact(
        headers=["Fault Code", "Subsystem", "Action"],
        rows=[["FAULT-E409", "Green Reservoir", "Refill"], ["FAULT-H204", "PTU Manifold", "Replace valve"]],
    )
    assert match_table_rows("action for FAULTE409", tbl) == [0]  # separator-free form still matches
    assert match_table_rows("what to do for the PTU manifold", tbl) == [1]
    assert match_table_rows("FAULT-Z999", tbl) == []  # unknown code: no row singled out


# ---- E6.2: abstain when nothing sourced or nothing relevant ------------------------------------
@pytest.mark.parametrize(
    "query",
    ["How do I bake a chocolate cake?", "What is the capital of France?", "Show me the table about cake"],
)
def test_e6_2_off_topic_question_abstains_and_says_why(indexed_pipe, query: str) -> None:
    ans = indexed_pipe.answer(query)
    assert ans.abstained is True
    assert ans.abstain_code == ABSTAIN_NOT_RELEVANT
    assert ans.artefact is None and ans.citation is None and ans.source_unit is None
    assert "Nothing relevant was retrieved" in ans.abstain_reason
    assert ans.top_hits  # the sources panel still shows what was retrieved


def test_e6_2_filler_only_question_abstains(indexed_pipe) -> None:
    ans = indexed_pipe.answer("what is it?")
    assert ans.abstained and ans.abstain_code == ABSTAIN_NOT_RELEVANT
    assert "no searchable terms" in ans.abstain_reason


def test_e6_2_relevance_gate_can_be_disabled(indexed_pipe) -> None:
    indexed_pipe.cfg.answer.min_query_term_coverage = 0.0
    assert indexed_pipe.answer("How do I bake a chocolate cake?").abstained is False


def test_e6_2_unsourced_top_hit_is_dropped_with_reason_and_next_sourced_hit_used(lab_cfg: AppConfig) -> None:
    no_page = _unit("amt_handbook__p0000__text_001", "text", None, "Starter relay coil K2 wiring")
    missing_img = _unit("amt_handbook__p0003__figure_001", "figure", 3, "Starter relay coil K2 wiring", "/nope/missing.png")
    good = _unit("amt_handbook__p0003__text_001", "text", 3, "The starter relay coil K2 is wired to PIN-B7.")
    ans = build_answer_from_hits(
        "starter relay coil K2 wiring", [_hit(1, no_page, 1.0), _hit(2, missing_img, 0.9), _hit(3, good, 0.8)], lab_cfg
    )
    assert ans.abstained is False and ans.citation == "amt_handbook p.3"
    assert ans.source_unit.unit_id == good.unit_id
    assert [h.unit.unit_id for h in ans.top_hits] == [good.unit_id]  # unsourced never shown as sourced
    reasons = {d["unit_id"]: d["reason"] for d in ans.dropped_hits}
    assert reasons[no_page.unit_id] == "missing page"
    assert reasons[missing_img.unit_id].startswith("figure image missing on disk")


def test_e6_2_all_unsourced_abstains_listing_each_problem(lab_cfg: AppConfig) -> None:
    hits = [
        _hit(1, _unit("a__p0000__text_001", "text", None, "relay wiring"), 1.0),
        _hit(2, _unit("a__p0000__text_002", "text", 0, "relay wiring"), 0.9),
    ]
    ans = build_answer_from_hits("relay wiring", hits, lab_cfg)
    assert ans.abstained and ans.abstain_code == ABSTAIN_UNSOURCED
    assert "fail-closed" in ans.abstain_reason
    assert "missing page" in ans.abstain_reason and "invalid page 0" in ans.abstain_reason
    assert ans.artefact is None and ans.citation is None


def test_e6_2_query_term_coverage_counts_code_variants_once() -> None:
    u = _unit("u1", "text", 1, "Bolt AN4-12A torque is 50-70 in-lbs")
    assert query_term_coverage("AN4-12A torque", u) == 1.0
    assert query_term_coverage("AN4-16A torque", u) == 0.5  # near-miss code earns no credit
    assert query_term_coverage("the what is", u) is None


# ---- E7.1: Layer A metrics ----------------------------------------------------------------------
class _CannedRetriever:
    def __init__(self, results: dict[str, list[SearchHit]]) -> None:
        self.results = results

    def retrieve(self, query: str, k: int = 5, **_: object) -> list[SearchHit]:
        return self.results[query][:k]


def test_e7_1_metrics_math_recall_at_cutoffs_mrr_precision_and_strictness(lab_cfg: AppConfig) -> None:
    q_fig = GoldQuestion(query_id="F1", query="fig q", gold_doc_id="d", gold_pages=[3], expected_type="figure")
    q_tab = GoldQuestion(query_id="T1", query="tab q", gold_doc_id="d", gold_pages=[1], expected_type="table")
    text_p3 = _unit("d__p3__text", "text", 3, "x", doc="d")
    fig_p2 = _unit("d__p2__fig", "figure", 2, "x", "a.png", doc="d")
    fig_p3 = _unit("d__p3__fig", "figure", 3, "x", "b.png", doc="d")
    tab_p9 = _unit("d__p9__tab", "table", 9, "x", doc="d")
    retriever = _CannedRetriever(
        {
            # page-level hit at rank 1 (prose), strict figure hit only at rank 3
            "fig q": [_hit(1, text_p3, 1.0), _hit(2, fig_p2, 0.9), _hit(3, fig_p3, 0.8)],
            # gold never retrieved
            "tab q": [_hit(1, tab_p9, 1.0)],
        }
    )
    rep = evaluate_retrieval_by_type(retriever, [q_fig, q_tab], lab_cfg, experiment_name="unit_math", save_report=False)
    fig = rep.by_type["figure"]
    assert fig.page_recall_at[1] == 1.0 and fig.strict_recall_at[1] == 0.0
    assert fig.strict_recall_at[3] == 1.0 and fig.strict_recall_at[5] == 1.0
    assert fig.page_mrr == 1.0 and fig.strict_mrr == pytest.approx(0.3333, abs=1e-4)
    assert fig.strict_precision_at_k == pytest.approx(1 / 5)
    tab = rep.by_type["table"]
    assert tab.strict_recall_at[5] == 0.0 and tab.strict_mrr == 0.0
    assert rep.by_type["text"].num_queries == 0
    assert failing_question_ids(rep, k=1) == ["F1", "T1"]
    assert failing_question_ids(rep, unit_type="figure", k=3) == []
    by_id = {r["query_id"]: r for r in rep.per_question}
    assert by_id["F1"]["page_rank"] == 1 and by_id["F1"]["strict_rank"] == 3
    assert by_id["T1"]["strict_rank"] is None


def test_e7_1_report_saved_with_cutoffs_per_question_and_latency(indexed_pipe, lab_cfg: AppConfig) -> None:
    rep = indexed_pipe.evaluate(experiment_name="e7_1_check", save_report=True)
    assert rep.recall_cutoffs == [1, 3, 5]
    for tm in rep.by_type.values():
        assert set(tm.strict_recall_at) == {1, 3, 5}
        assert tm.strict_recall_at[1] <= tm.strict_recall_at[3] <= tm.strict_recall_at[5]
        assert tm.strict_recall_at[5] == tm.strict_recall_at_k
    assert len(rep.per_question) == len(load_gold_questions(lab_cfg))
    assert {"retrieval_p50", "retrieval_p95"} <= set(rep.latency_ms)
    reloaded = load_eval_report(Path(lab_cfg.project.reports_dir) / "e7_1_check.json")
    assert reloaded.by_type["figure"].strict_recall_at == rep.by_type["figure"].strict_recall_at


def test_e7_1_compare_reports_lists_moved_question_ids(lab_cfg: AppConfig, capsys) -> None:
    summary = run_e7_3_failure_mode_experiments(lab_cfg, force_offline=True)
    rep_dir = Path(lab_cfg.project.reports_dir)
    before = load_eval_report(rep_dir / "fm1_before_caption_drift.json")
    after = load_eval_report(rep_dir / "fm1_after_context_grounded_caption.json")
    diff = compare_reports(before, after)
    assert diff["improved_question_ids"] == summary["failure_mode_1_right_text_wrong_figure"]["moved"]["improved_question_ids"]
    assert diff["improved_question_ids"] and all(qid.startswith("Q_FIG") for qid in diff["improved_question_ids"])
    assert diff["by_type"]["figure"]["strict_mrr"]["delta"] > 0

    assert metrics_main(["diff", str(rep_dir / "fm1_before_caption_drift.json"), str(rep_dir / "fm1_after_context_grounded_caption.json")]) == 0
    assert json.loads(capsys.readouterr().out)["improved_question_ids"] == diff["improved_question_ids"]


# ---- E7.2: faithfulness + Layer B ---------------------------------------------------------------
def test_e7_2_wrong_figure_is_flagged_dangerous_and_abstention_counted_apart(tmp_path: Path) -> None:
    q = GoldQuestion(
        query_id="Q", query="starter relay diagram", gold_doc_id="amt_handbook", gold_pages=[3],
        expected_type="figure", gold_artefact_hint="Figure 9-14", expected_keywords=["Figure 9-14"],
    )
    wrong_fig = _unit("amt_handbook__p0002__figure_001", "figure", 2, "relay schematic", "x.png",
                      nearby_caption="Figure 9-12. Generator Control Relay")
    ans = AnswerResponse(query=q.query, abstained=False, modality="figure", artefact="x.png",
                         citation="amt_handbook p.2", grounded_caption="Figure 9-12. Generator Control Relay",
                         source_unit=wrong_fig)
    verdict = judge_answer(q, ans)
    assert verdict["flag"] == FLAG_WRONG_FIGURE
    assert verdict["faithful"] is False and verdict["citation_correct"] is False

    abstained = judge_answer(q, AnswerResponse(query=q.query, abstained=True, abstain_code=ABSTAIN_NOT_RELEVANT))
    assert abstained["flag"] == FLAG_ABSTAINED and abstained["abstained"] is True


def test_e7_2_spotcheck_reports_wrong_artefacts_separately_from_abstentions(indexed_pipe, lab_cfg: AppConfig) -> None:
    questions = load_gold_questions(lab_cfg)
    res = run_faithfulness_spotcheck(indexed_pipe.retriever, questions, lab_cfg, sample_size=5)
    assert res["reviewed_count"] == 5
    assert res["passed_count"] + res["abstained_count"] + res["flagged_wrong_artefact_count"] == 5
    assert {r["expected_type"] for r in res["reviews"]} == {"figure", "table", "text"}


def test_e7_2_layer_b_answer_quality_reported_by_type(indexed_pipe) -> None:
    rep = indexed_pipe.evaluate(experiment_name="layer_b_check", save_report=False)
    aq = rep.answer_quality
    overall = aq["overall"]
    for key in ("answer_correctness", "citation_accuracy", "wrong_source_rate", "artefact_correctness", "abstention_rate"):
        assert overall[key] is not None
    assert overall["citation_accuracy"] + overall["wrong_source_rate"] == pytest.approx(1.0)
    assert overall["wrong_source_rate"] == 0.0 and overall["artefact_correctness"] == 1.0
    assert overall["answer_correctness"] >= 0.9
    assert set(aq["by_type"]) == {"text", "figure", "table"}
    assert len(aq["per_question"]) == overall["n"] and {"answer_p50", "answer_p95"} <= set(aq["latency_ms"])


# ---- E7.3: attributable before/after ------------------------------------------------------------
def test_e7_3_each_change_measured_alone_with_baseline_first_and_markdown_report(lab_cfg: AppConfig) -> None:
    summary = run_e7_3_failure_mode_experiments(lab_cfg, force_offline=True)
    rep_dir = Path(lab_cfg.project.reports_dir)

    # Baseline recorded (and saved) before any fix
    assert (rep_dir / "baseline_untuned.json").exists()
    assert summary["t2_cumulative"][0]["stage"] == "Baseline"
    assert [s["stage"] for s in summary["t2_cumulative"]] == ["Baseline", "After fix 1", "After fix 2", "Final"]

    for key, target in (("failure_mode_1_right_text_wrong_figure", "figure"), ("failure_mode_2_table_mush_and_exact_codes", "table")):
        fm = summary[key]
        assert fm["measured_move"] is True
        assert fm["root_cause"] and fm["hypothesis"] and fm["change"]
        assert set(fm["by_type_before"]) == set(fm["by_type_after"]) == {"text", "figure", "table"}
        assert fm["failing_question_ids_before"]
        assert fm["moved"]["improved_question_ids"]
        assert not fm["moved"]["regressed_question_ids"]
        assert fm["layer_b_after"]["artefact_correctness"] > fm["layer_b_before"]["artefact_correctness"]

    steps = summary["failure_mode_2_table_mush_and_exact_codes"]["steps"]
    assert [s["step"] for s in steps] == ["a", "b", "c"]
    assert sum(s["delta_table_strict_mrr"] for s in steps) == pytest.approx(
        summary["failure_mode_2_table_mush_and_exact_codes"]["delta_table_strict_mrr"], abs=1e-3
    )

    md = (rep_dir / "e7_3_failure_mode_ablations.md").read_text(encoding="utf-8")
    for heading in ("## T2", "## T3", "one change at a time", "## Layer B"):
        assert heading in md
