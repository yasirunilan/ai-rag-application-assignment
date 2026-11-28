"""End-to-End Multimodal Retrieval Pipeline Orchestrator (Stages 1 to 7)."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time
from typing import Any, Optional

from mmrag.answer.responder import build_answer_from_hits
from mmrag.config import AppConfig
from mmrag.corpus.loader import EnginuityTask2Loader, load_corpus_manuals, load_gold_questions
from mmrag.enrich.cache import CaptionCache
from mmrag.enrich.captioner import FigureEnrichmentResult, enrich_figures
from mmrag.enrich.rejects import RejectedUnit, reject_unenrichable, write_rejection_report
from mmrag.enrich.table_linearizer import enrich_table_units
from mmrag.eval.metrics import evaluate_layers_b_and_c, evaluate_retrieval_by_type
from mmrag.extract.figures import extract_figure_units
from mmrag.extract.spotcheck import run_provenance_spotcheck
from mmrag.extract.tables import extract_table_units, extract_text_units
from mmrag.index.embedder import create_embedder
from mmrag.index.store import MixedVectorStore
from mmrag.ingest.router import parse_and_route_pdf
from mmrag.retrieve.retriever import HybridRetriever
from mmrag.schemas import AnswerResponse, EvalReport, PageClassification, RawRegion, Unit


class MultimodalRAGPipeline:
    """Connects Stage 1 (Parse & Route) -> Stage 7 (Evaluate) with end-to-end provenance."""

    def __init__(self, cfg: AppConfig, force_offline: bool = False) -> None:
        self.cfg = cfg
        self.force_offline = force_offline
        self.page_classifications: list[PageClassification] = []
        self.raw_regions: list[RawRegion] = []
        self.filtered_figures: list[dict] = []
        self.enrichment_rejected: list[RejectedUnit] = []
        self.enrichment_result: Optional[FigureEnrichmentResult] = None
        self.units: list[Unit] = []
        self.caption_cache: Optional[CaptionCache] = None
        self.store: Optional[MixedVectorStore] = None
        self.retriever: Optional[HybridRetriever] = None
        self.last_ingestion_time_ms: float = 0.0

    def ingest_and_index(
        self,
        pdf_paths: Optional[list[Path]] = None,
        persist_index: bool = True,
    ) -> dict[str, Any]:
        """Runs Stages 1, 2, 3 (incl. E3.3 rejections), and 4 over the configured corpus."""
        t0 = time.perf_counter()

        # Clear previous review queue & enrichment rejection log before fresh ingest
        for stale_path in (
            Path(self.cfg.project.review_queue_path),
            Path(self.cfg.project.enrichment_rejects_path),
        ):
            if stale_path.exists():
                stale_path.unlink()

        if (
            pdf_paths is None
            and self.cfg.corpus.eval_dataset == "enginuity_task2"
            and self.cfg.corpus.enginuity_dir
        ):
            loaded_units, _ = EnginuityTask2Loader(self.cfg.corpus.enginuity_dir).load(self.cfg)
            manuals: list[Path] = []
            self.page_classifications = []
            self.raw_regions = []
            self.filtered_figures = []
            txt_units = [u for u in loaded_units if u.type == "text"]
            fig_units = [u for u in loaded_units if u.type == "figure"]
            tbl_units = [u for u in loaded_units if u.type == "table"]
        else:
            manuals = pdf_paths or load_corpus_manuals(self.cfg, auto_seed=True)
            self.page_classifications = []
            self.raw_regions = []
            for pdf in manuals:
                p_cls, p_regs = parse_and_route_pdf(pdf, self.cfg)
                self.page_classifications.extend(p_cls)
                self.raw_regions.extend(p_regs)

            # Stage 2: Extract figures, tables, and prose text with provenance
            fig_units, self.filtered_figures = extract_figure_units(self.raw_regions, self.cfg)
            tbl_units = extract_table_units(self.raw_regions, self.cfg)
            txt_units = extract_text_units(self.raw_regions)

        # Stage 3: Enrich figures (vision captioning + cache) and tables (col=val linearization)
        enrich_cfg = self.cfg
        if self.force_offline and self.cfg.enrich.vision_provider == "openrouter_free":
            enrich_cfg = self.cfg.model_copy(
                update={
                    "enrich": self.cfg.enrich.model_copy(
                        update={"vision_provider": "offline_heuristic"}
                    )
                }
            )
        self.enrichment_result = enrich_figures(fig_units, enrich_cfg)
        fig_units = self.enrichment_result.units
        self.caption_cache = self.enrichment_result.cache
        tbl_units = enrich_table_units(tbl_units, enrich_cfg)

        # Story E3.3: Reject unenrichable figure/table units visibly before indexing
        accepted_non_text, self.enrichment_rejected = reject_unenrichable(fig_units + tbl_units)
        if self.enrichment_rejected:
            write_rejection_report(self.enrichment_rejected, self.cfg.project.enrichment_rejects_path)

        self.units = txt_units + accepted_non_text

        # Stage 4: Embed & build unified mixed index
        embedder = create_embedder(self.cfg, force_offline=self.force_offline)
        self.store = MixedVectorStore(self.cfg, embedder=embedder)
        self.store.build(self.units, apply_chunking=True)
        self.last_ingestion_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        route_counts = Counter(c.route_decision for c in self.page_classifications)
        stats: dict[str, Any] = {
            "num_manuals": len(manuals),
            "num_pages": len(self.page_classifications),
            "page_route_counts": dict(route_counts),
            "num_scanned_pages": sum(1 for c in self.page_classifications if c.is_scanned),
            "num_review_queued_pages": sum(
                1 for c in self.page_classifications if c.route_decision == "ocr_review_queue"
            ),
            "num_text_units": len(txt_units),
            "num_figure_units": sum(1 for u in accepted_non_text if u.type == "figure"),
            "num_filtered_decorative_icons": len(self.filtered_figures),
            "num_table_units": sum(1 for u in accepted_non_text if u.type == "table"),
            "num_enrichment_rejected_units": len(self.enrichment_rejected),
            "num_store_rejected_units": len(self.store.rejected),
            "total_indexed_units": len(self.store.units),
            "embedding_model_id": self.store.index_model_id,
            "remote_caption_calls": self.enrichment_result.remote_calls,
            "caption_cache_hits": self.caption_cache.hits if self.caption_cache else 0,
            "caption_cache_misses": self.caption_cache.misses if self.caption_cache else 0,
            "ingestion_time_ms": self.last_ingestion_time_ms,
        }

        if persist_index:
            target_dir = self.store.save()
            idx_bytes = sum(f.stat().st_size for f in target_dir.glob("*") if f.is_file())
            index_report = {
                **stats,
                "index_size_bytes": idx_bytes,
                "stage0_prediction": {
                    "worst_type": self.cfg.corpus.stage0_prediction_worst_type,
                    "reason": self.cfg.corpus.stage0_prediction_reason,
                },
                "enrichment_rejected": [
                    {"unit_id": r.unit_id, "type": r.type, "citation": r.citation, "reason": r.reason}
                    for r in self.enrichment_rejected
                ],
                "store_rejected": self.store.rejected,
            }
            (target_dir / "index_report.json").write_text(
                json.dumps(index_report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # Stage 5: Initialize hybrid retriever
        self.retriever = HybridRetriever(self.store, self.cfg)
        return stats

    def answer(self, query: str, k: Optional[int] = None) -> AnswerResponse:
        """Runs Stage 5 (Retrieve) + Stage 6 (Fail-Closed Answer with Artefact)."""
        if self.retriever is None:
            raise RuntimeError("Pipeline has not been indexed yet. Call ingest_and_index() first.")
        t0 = time.perf_counter()
        hits = self.retriever.retrieve(query, k=k or self.cfg.retrieve.top_k)
        ret_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        ans = build_answer_from_hits(query, hits, self.cfg, offline=self.force_offline)
        gen_ms = ans.timings_ms.get("generation_ms", 0.0)
        ans.timings_ms = {
            "retrieval_ms": ret_ms,
            "generation_ms": gen_ms,
            "total_ms": round(ret_ms + gen_ms, 2),
        }
        return ans

    def spotcheck_provenance(self, sample_size: int = 5) -> dict[str, Any]:
        """Runs Story E2.3 provenance spot-check on extracted units."""
        return run_provenance_spotcheck(self.units, sample_size=sample_size)

    def evaluate(
        self,
        experiment_name: str = "baseline",
        k: Optional[int] = None,
        save_report: bool = True,
    ) -> EvalReport:
        """Runs Stage 7 Layers A, B, and C evaluation + 5-sample faithfulness check."""
        if self.retriever is None:
            raise RuntimeError("Pipeline has not been indexed yet.")
        questions = load_gold_questions(self.cfg, auto_seed=True)
        report = evaluate_retrieval_by_type(
            self.retriever,
            questions,
            self.cfg,
            experiment_name=experiment_name,
            k=k,
            save_report=False,
        )
        layer_b, layer_c, faith = evaluate_layers_b_and_c(
            self.retriever,
            questions,
            self.cfg,
            offline=self.force_offline,
            ingestion_time_ms=self.last_ingestion_time_ms,
        )
        report.faithfulness_pass_rate = faith["faithfulness_pass_rate"]
        report.faithfulness_details = faith["reviews"]
        report.layer_b = layer_b
        report.layer_c = layer_c

        if save_report:
            rep_dir = Path(self.cfg.project.reports_dir)
            rep_dir.mkdir(parents=True, exist_ok=True)
            out_path = rep_dir / f"{experiment_name}.json"
            out_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")

        return report
