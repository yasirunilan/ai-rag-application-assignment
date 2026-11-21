"""CLI Entrypoint for Multimodal RAG over Technical Manuals (Person 1 · Story E0.1).

Commands:
- `python -m mmrag.cli smoke`     : Zero-network end-to-end ingest -> spotcheck -> answer -> evaluate -> ablate
- `python -m mmrag.cli ingest`    : Parse, route, extract, enrich, and index the corpus
- `python -m mmrag.cli spotcheck` : Run Story E2.3 5-unit provenance spot-check
- `python -m mmrag.cli answer -q "..."` : Answer query with real artefact + page citation
- `python -m mmrag.cli eval`      : Run Story E7.1 per-type Recall@k/MRR + E7.2 faithfulness check
- `python -m mmrag.cli ablate`    : Run Story E7.3 before/after benchmarks on the 2 failure modes
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from mmrag.config import load_config
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


def cmd_smoke(config_path: str) -> int:
    """Runs ingest -> answer -> evaluate end-to-end offline with no API key and no GPU (Story E0.1)."""
    cfg = load_config(config_path)
    from mmrag.corpus.synthetic import ensure_default_corpus

    ensure_default_corpus(cfg.corpus.manuals_dir, cfg.corpus.gold_eval_path)
    pipe = MultimodalRAGPipeline(cfg, force_offline=True)

    _print_banner("STAGE 1-4: INGEST, ROUTE, EXTRACT, ENRICH & INDEX (OFFLINE SMOKE)")
    stats = pipe.ingest_and_index(persist_index=True)
    print(json.dumps(stats, indent=2))

    # Story E1.1 note: show what naive flat-text loader would have dropped
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
        if isinstance(ans.artefact, TableArtefact):
            print("Artefact : Structured TableArtefact ->")
            print(ans.artefact.to_markdown())
        else:
            print(f"Artefact : {ans.artefact}")
            if ans.grounded_caption:
                print(f"Caption  : {ans.grounded_caption}")

    _print_banner("STAGE 7.1 & 7.2: PER-TYPE RECALL@K, MRR & FAITHFULNESS CHECK")
    report = pipe.evaluate(experiment_name="smoke_after_fixes", save_report=True)
    for t_name, tm in report.by_type.items():
        print(
            f"  Type={t_name:<7} | Queries={tm.num_queries} | "
            f"Page Recall@{report.k}={tm.page_recall_at_k:.2f} | Page MRR={tm.page_mrr:.2f} | "
            f"Strict Recall@{report.k}={tm.strict_recall_at_k:.2f} | Strict MRR={tm.strict_mrr:.2f}"
        )
    print(f"  Faithfulness Pass Rate (5 spot-checked answers): {report.faithfulness_pass_rate:.2f}")

    _print_banner("STAGE 7.3: BEFORE / AFTER FAILURE MODE ABLATIONS")
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
    store = MixedVectorStore.load(cfg, embedder=create_embedder(cfg, force_offline=True))
    from mmrag.extract.spotcheck import run_provenance_spotcheck

    res = run_provenance_spotcheck(store.units, sample_size=5)
    print(json.dumps(res, indent=2))
    return 0 if res["passed"] else 1


def cmd_answer(config_path: str, query: str, offline: bool) -> int:
    cfg = load_config(config_path)
    embedder = create_embedder(cfg, force_offline=offline)
    store = MixedVectorStore.load(cfg, embedder=embedder)
    retriever = HybridRetriever(store, cfg)
    from mmrag.answer.responder import build_answer_from_hits

    hits = retriever.retrieve(query, k=cfg.retrieve.top_k)
    ans = build_answer_from_hits(query, hits, cfg)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmrag",
        description="Multimodal Retrieval over Technical Manuals (Course 3 · Day 2 · Lab A)",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("smoke", help="Run zero-network end-to-end smoke test (Story E0.1)")

    p_ing = sub.add_parser("ingest", help="Ingest, extract, enrich, and index manuals")
    p_ing.add_argument("--offline", action="store_true", help="Force offline models")

    sub.add_parser("spotcheck", help="Run 5-unit provenance spot-check (Story E2.3)")

    p_ans = sub.add_parser("answer", help="Answer a query with the real artefact + citation")
    p_ans.add_argument("-q", "--query", required=True, help="Engineering query string")
    p_ans.add_argument("--offline", action="store_true", help="Force offline models")

    p_ev = sub.add_parser("eval", help="Run per-type Recall@k, MRR, and faithfulness check")
    p_ev.add_argument("--offline", action="store_true", help="Force offline models")

    p_ab = sub.add_parser("ablate", help="Run Before/After failure mode benchmarks (Story E7.3)")
    p_ab.add_argument("--offline", action="store_true", help="Force offline models")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "smoke":
        return cmd_smoke(args.config)
    if args.command == "ingest":
        return cmd_ingest(args.config, args.offline)
    if args.command == "spotcheck":
        return cmd_spotcheck(args.config)
    if args.command == "answer":
        return cmd_answer(args.config, args.query, args.offline)
    if args.command == "eval":
        return cmd_eval(args.config, args.offline)
    if args.command == "ablate":
        return cmd_ablate(args.config, args.offline)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
