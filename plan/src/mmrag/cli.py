"""CLI Entrypoint for Multimodal RAG over Technical Manuals (Person 1 · Story E0.1 & Person 5 · E8.4).

Commands:
- `python -m mmrag.cli smoke`     : Zero-network end-to-end ingest -> spotcheck -> answer -> evaluate -> ablate
- `python -m mmrag.cli ingest`    : Parse, route, extract, enrich, reject unenrichable, and index
- `python -m mmrag.cli spotcheck` : Run Story E2.3 5-unit provenance spot-check
- `python -m mmrag.cli answer -q "..."` : Answer query with real artefact + page citation
- `python -m mmrag.cli eval`      : One-command Layers A, B, C evaluation + faithfulness check
- `python -m mmrag.cli ablate`    : Run Story E7.3 / E8.5 before/after failure-mode benchmarks
- `python -m mmrag.cli check`     : Story E0.1 environment check (embeddings, vector backend, OCR, :free vision slugs)
- `python -m mmrag.cli check-models` : Diagnostics check over OpenRouter model slugs + vector store
- `python -m mmrag.cli corpus`    : Story E0.2 corpus manifest under the `max_pages` budget
- `python -m mmrag.cli fetch-corpus [--dude]` : Download configured real manuals / import the DUDE sample
- `python -m mmrag.cli ocr-calibrate [--apply]` : Story E1.2 sweep of the OCR quality-gate thresholds
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from mmrag.config import load_config, load_dotenv
from mmrag.diagnostics import check_models
from mmrag.eval.ablations import run_e7_3_failure_mode_experiments
from mmrag.index.embedder import create_embedder
from mmrag.index.store import MixedVectorStore
from mmrag.ingest.router import explain_naive_loader_drops
from mmrag.pipeline import MultimodalRAGPipeline
from mmrag.retrieve.retriever import HybridRetriever
from mmrag.schemas import TableArtefact


def _print_banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def cmd_check_models(config_path: str, offline: bool) -> int:
    cfg = load_config(config_path)
    res = check_models(cfg, offline=offline)
    print(json.dumps(res, indent=2))
    return 0 if res["ok"] else 1


def cmd_smoke(config_path: str) -> int:
    """Runs ingest -> answer -> evaluate end-to-end offline with no API key and no GPU (Story E0.1)."""
    cfg = load_config(config_path)
    from mmrag.corpus.synthetic import ensure_default_corpus

    ensure_default_corpus(cfg.corpus.manuals_dir, cfg.corpus.gold_eval_path)
    pipe = MultimodalRAGPipeline(cfg, force_offline=True)

    _print_banner("STAGE 0: VECTOR BACKEND & OFFLINE MODEL CHECK")
    print(json.dumps(check_models(cfg, offline=True), indent=2))

    _print_banner("STAGE 1-4: INGEST, ROUTE, EXTRACT, ENRICH & INDEX (OFFLINE SMOKE)")
    stats = pipe.ingest_and_index(persist_index=True)
    print(json.dumps(stats, indent=2))
    print(
        f"\n[E1.2 OCR Gate]: {stats['num_scanned_pages']} scanned page(s), "
        f"{stats['num_review_queued_pages']} failed the gate -> {cfg.project.review_queue_path} (not indexed)"
    )

    first_pdf = pipe.raw_regions[0].source_path if pipe.raw_regions else ""
    if first_pdf:
        naive_drops = explain_naive_loader_drops(first_pdf)
        print("\n[E1.1 Naive Loader Drop Analysis]:")
        print(json.dumps(naive_drops, indent=2))

    _print_banner("STAGE 2.3: PROVENANCE SPOT-CHECK (5 UNITS)")
    spot = pipe.spotcheck_provenance(sample_size=5)
    print(json.dumps(spot, indent=2))
    if not spot["passed"]:
        print("ERROR: Provenance spot-check failed!", file=sys.stderr)
        return 1

    _print_banner("STAGE 5-6: RETRIEVE BY TEXT, ANSWER WITH REAL ARTEFACT")
    demo_queries = [
        "Show the engine starter relay control wiring diagram for Model 737-800",
        "What is the dry and lubed torque spec table for bolt part number AN4-12A?",
        "What is the nominal operating pressure and thermal relief valve cracking pressure of the 737-800 hydraulic flight control system?",
    ]
    for dq in demo_queries:
        ans = pipe.answer(dq)
        print(f"\nQuery    : {ans.query}")
        print(f"Modality : {ans.modality} | Citation: {ans.citation} | Abstained: {ans.abstained}")
        print(f"Answer   : {ans.grounded_answer}")
        if isinstance(ans.artefact, TableArtefact):
            print("Artefact : Structured TableArtefact ->")
            print(ans.artefact.to_markdown())
        else:
            print(f"Artefact : {ans.artefact}")
            if ans.grounded_caption:
                print(f"Caption  : {ans.grounded_caption}")

    _print_banner("STAGE 7 (LAYERS A, B, C): PER-TYPE RECALL@K, MRR, GROUNDING & OPS")
    report = pipe.evaluate(experiment_name="smoke_after_fixes", save_report=True)
    for t_name, tm in report.by_type.items():
        print(
            f"  Type={t_name:<7} | n={tm.num_queries} | "
            f"R@1={tm.strict_recall_at_1:.2f} R@3={tm.strict_recall_at_3:.2f} "
            f"R@{report.k}={tm.strict_recall_at_k:.2f} | Strict MRR={tm.strict_mrr:.2f}"
        )
    if report.layer_b:
        print(f"  Layer B  : {report.layer_b.model_dump()}")
    if report.layer_c:
        print(f"  Layer C  : {report.layer_c.model_dump()}")

    _print_banner("STAGE 7.3 / E8.5: BEFORE / AFTER FAILURE MODE ABLATIONS")
    ablations = run_e7_3_failure_mode_experiments(cfg, force_offline=True)
    print(json.dumps(ablations, indent=2))

    _print_banner("SMOKE RUN PASSED ALL ACCEPTANCE GATES (ZERO API KEYS, ZERO GPU)")
    return 0


def cmd_ingest(config_path: str, offline: bool) -> int:
    cfg = load_config(config_path)
    pipe = MultimodalRAGPipeline(cfg, force_offline=offline)
    stats = pipe.ingest_and_index(persist_index=True)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_spotcheck(config_path: str) -> int:
    cfg = load_config(config_path)
    store = MixedVectorStore.load(
        cfg,
        embedder=create_embedder(cfg, force_offline=True),
        verify_embedder=False,
    )
    from mmrag.extract.spotcheck import run_provenance_spotcheck

    res = run_provenance_spotcheck(store.units, sample_size=5)
    print(json.dumps(res, indent=2))
    return 0 if res["passed"] else 1


def cmd_answer(config_path: str, query: str, offline: bool, type_filter: str | None = None) -> int:
    cfg = load_config(config_path)
    embedder = create_embedder(cfg, force_offline=offline)
    store = MixedVectorStore.load(cfg, embedder=embedder)
    retriever = HybridRetriever(store, cfg)
    from mmrag.answer.responder import build_answer_from_hits

    hits = retriever.retrieve(query, k=cfg.retrieve.top_k, type_filter=type_filter)
    ans = build_answer_from_hits(query, hits, cfg, offline=offline)
    print(json.dumps(ans.model_dump(), indent=2))
    return 0


def cmd_eval(config_path: str, offline: bool) -> int:
    cfg = load_config(config_path)
    pipe = MultimodalRAGPipeline(cfg, force_offline=offline)
    pipe.ingest_and_index(persist_index=True)
    rep = pipe.evaluate(experiment_name="eval_run", save_report=True)
    print(json.dumps(rep.model_dump(), indent=2))
    return 0


def cmd_ablate(config_path: str, offline: bool) -> int:
    cfg = load_config(config_path)
    summary = run_e7_3_failure_mode_experiments(cfg, force_offline=offline)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_check(config_path: str, offline: bool, live_call: bool) -> int:
    from mmrag.env_check import run_environment_check

    cfg = load_config(config_path)
    _print_banner("E0.1 ENVIRONMENT & MODEL CHECK" + (" (OFFLINE)" if offline else ""))
    code, results = run_environment_check(cfg, offline=offline, live_call=live_call)
    for r in results:
        print(r.line())
    if code == 2:
        print("\nSTOP: OPENROUTER_API_KEY is missing (see the FAIL line above).", file=sys.stderr)
    elif code:
        print("\nFAIL: fix the lines marked FAIL (a slug swap is a config.yaml change only).", file=sys.stderr)
    else:
        print("\nAll checks passed.")
    return code


def cmd_corpus(config_path: str) -> int:
    from mmrag.corpus.loader import load_gold_questions, resolve_corpus

    cfg = load_config(config_path)
    res = resolve_corpus(cfg, auto_seed=True)
    summary = res.to_dict()
    summary["num_gold_questions_in_corpus"] = len(load_gold_questions(cfg, restrict_to_corpus=True))
    summary["num_gold_questions_total"] = len(load_gold_questions(cfg, restrict_to_corpus=False))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_fetch_corpus(config_path: str, dude: bool, dude_max_pages: int, overwrite: bool) -> int:
    from pathlib import Path

    from mmrag.corpus.sources import fetch_configured_sources, fetch_dude_sample, import_dude_sample

    cfg = load_config(config_path)
    failed = False
    results = fetch_configured_sources(cfg, overwrite=overwrite)
    for r in results:
        print(json.dumps(r))
        failed |= r["status"] != "OK"
    if not results:
        print("No real_sources with a `url` in config.yaml (place PDFs at their `path` manually).")

    if dude:
        dude_dir = Path(cfg.corpus.downloads_dir) / "dude"
        gold_out = Path(cfg.corpus.downloads_dir) / "dude_gold_eval.jsonl"
        try:
            ds, tar = fetch_dude_sample(cfg, overwrite=overwrite)
            summary = import_dude_sample(ds, tar, dude_dir, gold_out, max_pages=dude_max_pages)
            print(json.dumps({k: v for k, v in summary.items() if k != "documents"}, indent=2))
            src_paths = {Path(s.path) for s in cfg.corpus.real_sources if s.enabled}
            if dude_dir not in src_paths or str(gold_out) not in cfg.corpus.extra_gold_paths:
                print(
                    "To ingest it, enable in config.yaml:\n"
                    f"  corpus.real_sources: - {{doc_id: dude, path: {dude_dir}}}\n"
                    f"  corpus.extra_gold_paths: [{gold_out}]"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL dude sample: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


def cmd_ocr_calibrate(config_path: str, pages: int, apply: bool) -> int:
    from mmrag.ingest.ocr_calibration import apply_thresholds_to_config_yaml, run_ocr_gate_calibration
    from mmrag.ingest.ocr_gate import tesseract_status

    ok, detail = tesseract_status()
    if not ok:
        print(f"FAIL: OCR calibration needs Tesseract: {detail}", file=sys.stderr)
        return 2
    cfg = load_config(config_path)
    from mmrag.corpus.loader import load_corpus_manuals

    rep = run_ocr_gate_calibration(cfg, load_corpus_manuals(cfg), max_reference_pages=pages)
    _print_banner("E1.2 OCR QUALITY-GATE CALIBRATION")
    print(
        f"Samples: {rep['num_samples']} ({rep['num_reference_pages']} pages x degradations), "
        f"acceptable: {rep['num_acceptable']}"
    )
    for label, key in (("current", "current_thresholds"), ("recommended", "recommended_thresholds")):
        r = rep[key]
        print(
            f"  {label:<11} conf>={r['min_mean_confidence']:<5} garbled<={r['max_garbled_char_ratio']:<5} "
            f"alnum>={r['min_alnum_token_ratio']:<5} words>={r['min_word_count']:<3} | precision={r['precision']:.3f} recall={r['recall']:.3f} "
            f"bad_indexed={r['bad_pages_indexed']} good_flagged={r['good_pages_flagged']}"
        )
    print(f"Selection: {rep['selection_strategy']}")
    real = rep["real_scanned_pages"]
    if real:
        for label in ("current", "recommended"):
            n_pass = sum(r["passes"][label] for r in real)
            print(f"  real scanned pages passing {label:<11}: {n_pass}/{len(real)}")
        print("  Check these by eye before --apply: synthetic degradations understate real mid-confidence scans.")
    print(f"Report: {rep['report_path']}")
    print(
        f"Review by eye: {len(rep['review_by_eye']['flagged'])} flagged + {len(rep['review_by_eye']['passed'])} "
        f"passed images under {cfg.project.data_dir}/ocr_calibration/"
    )
    if apply:
        rec = rep["recommended_thresholds"]
        changed = apply_thresholds_to_config_yaml(
            config_path,
            {k: rec[k] for k in ("min_mean_confidence", "max_garbled_char_ratio", "min_alnum_token_ratio", "min_word_count")},
        )
        print(f"Applied to {config_path}: {', '.join(changed) or 'nothing (keys not found)'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmrag",
        description="Multimodal Retrieval over Technical Manuals (Course 3 · Day 2 · Lab A)",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_chk_models = sub.add_parser(
        "check-models", help="Verify OpenRouter model slugs and vector store (Stories E0.1 & E0.2)"
    )
    p_chk_models.add_argument("--offline", action="store_true", help="Verify offline local path only")

    sub.add_parser("smoke", help="Run zero-network end-to-end smoke test (Story E0.1)")

    p_ing = sub.add_parser("ingest", help="Ingest, extract, enrich, and index manuals")
    p_ing.add_argument("--offline", action="store_true", help="Force offline models")

    sub.add_parser("spotcheck", help="Run 5-unit provenance spot-check (Story E2.3)")

    p_ans = sub.add_parser("answer", help="Answer a query with the real artefact + citation")
    p_ans.add_argument("-q", "--query", required=True, help="Engineering query string")
    p_ans.add_argument("--type", dest="type_filter", choices=("text", "figure", "table"), default=None)
    p_ans.add_argument("--offline", action="store_true", help="Force offline models")

    p_ev = sub.add_parser("eval", help="Run Layers A, B, C evaluation and faithfulness check")
    p_ev.add_argument("--offline", action="store_true", help="Force offline models")

    p_ab = sub.add_parser("ablate", help="Run Before/After failure mode benchmarks (Stories E7.3 / E8.5)")
    p_ab.add_argument("--offline", action="store_true", help="Force offline models")

    p_chk = sub.add_parser("check", help="Verify embeddings, vector backend, OCR and :free vision slugs (Story E0.1)")
    p_chk.add_argument("--offline", action="store_true", help="Skip network/key checks")
    p_chk.add_argument("--no-live-call", action="store_true", help="Skip the one tiny vision completion call")

    sub.add_parser("corpus", help="Show the corpus manifest under the max_pages budget (Story E0.2)")

    p_fc = sub.add_parser("fetch-corpus", help="Download configured real manuals; --dude imports the DUDE sample")
    p_fc.add_argument("--dude", action="store_true", help="Download + import the DUDE sample (~121 MB)")
    p_fc.add_argument("--dude-max-pages", type=int, default=120, help="Page budget for imported DUDE docs")
    p_fc.add_argument("--overwrite", action="store_true", help="Re-download files that already exist")

    p_oc = sub.add_parser("ocr-calibrate", help="Sweep OCR gate thresholds against degraded digital pages (Story E1.2)")
    p_oc.add_argument("--pages", type=int, default=12, help="Digital reference pages to degrade and OCR")
    p_oc.add_argument("--apply", action="store_true", help="Write the recommended thresholds into config.yaml")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_dotenv()  # OPENROUTER_API_KEY from a git-ignored .env, never from config.yaml
    if args.command == "check-models":
        return cmd_check_models(args.config, args.offline)
    if args.command == "smoke":
        return cmd_smoke(args.config)
    if args.command == "ingest":
        return cmd_ingest(args.config, args.offline)
    if args.command == "spotcheck":
        return cmd_spotcheck(args.config)
    if args.command == "answer":
        return cmd_answer(args.config, args.query, args.offline, args.type_filter)
    if args.command == "eval":
        return cmd_eval(args.config, args.offline)
    if args.command == "ablate":
        return cmd_ablate(args.config, args.offline)
    if args.command == "check":
        return cmd_check(args.config, args.offline, not args.no_live_call)
    if args.command == "corpus":
        return cmd_corpus(args.config)
    if args.command == "fetch-corpus":
        return cmd_fetch_corpus(args.config, args.dude, args.dude_max_pages, args.overwrite)
    if args.command == "ocr-calibrate":
        return cmd_ocr_calibrate(args.config, args.pages, args.apply)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
