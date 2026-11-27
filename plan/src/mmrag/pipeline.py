"""End-to-End Multimodal Retrieval Pipeline Orchestrator (Stages 1 to 7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from mmrag.answer.responder import build_answer_from_hits
from mmrag.config import AppConfig
from mmrag.corpus.loader import load_corpus_manuals, load_gold_questions
from mmrag.enrich.cache import CaptionCache
from mmrag.enrich.captioner import enrich_figure_units
from mmrag.enrich.table_linearizer import enrich_table_units
from mmrag.eval.faithfulness import evaluate_answer_quality, run_faithfulness_spotcheck
from mmrag.eval.metrics import evaluate_retrieval_by_type, save_eval_report
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
        self.units: list[Unit] = []
        self.caption_cache: Optional[CaptionCache] = None
        self.store: Optional[MixedVectorStore] = None
        self.retriever: Optional[HybridRetriever] = None

    def ingest_and_index(
        self,
        pdf_paths: Optional[list[Path]] = None,
        persist_index: bool = True,
    ) -> dict[str, Any]:
        """Runs Stages 1, 2, 3, and 4 over the corpus PDFs."""
        manuals = pdf_paths or load_corpus_manuals(self.cfg, auto_seed=True)

        # Clear previous review queue before fresh ingest
        rq_path = Path(self.cfg.project.review_queue_path)
        if rq_path.exists():
            rq_path.unlink()

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
        fig_units, self.caption_cache = enrich_figure_units(fig_units, enrich_cfg)
        tbl_units = enrich_table_units(tbl_units, enrich_cfg)

        self.units = txt_units + fig_units + tbl_units

        # Stage 4: Embed & build unified mixed index
        embedder = create_embedder(self.cfg, force_offline=self.force_offline)
        self.store = MixedVectorStore(self.cfg, embedder=embedder)
        self.store.build(self.units, apply_chunking=True)
        if persist_index:
            self.store.save()

        # Stage 5: Initialize hybrid retriever
        self.retriever = HybridRetriever(self.store, self.cfg)

        return {
            "num_manuals": len(manuals),
            "num_pages": len(self.page_classifications),
            "num_scanned_pages": sum(1 for c in self.page_classifications if c.is_scanned),
            "num_review_queued_pages": sum(
                1 for c in self.page_classifications if c.route_decision == "ocr_review_queue"
            ),
            "num_text_units": len(txt_units),
            "num_figure_units": len(fig_units),
            "num_filtered_decorative_icons": len(self.filtered_figures),
            "num_table_units": len(tbl_units),
            "total_indexed_units": len(self.store.units),
        }

    def answer(self, query: str, k: Optional[int] = None) -> AnswerResponse:
        """Runs Stage 5 (Retrieve) + Stage 6 (Fail-Closed Answer with Artefact)."""
        if self.retriever is None:
            raise RuntimeError("Pipeline has not been indexed yet. Call ingest_and_index() first.")
        hits = self.retriever.retrieve(query, k=k or self.cfg.retrieve.top_k)
        return build_answer_from_hits(query, hits, self.cfg)

    def spotcheck_provenance(self, sample_size: int = 5) -> dict[str, Any]:
        """Runs Story E2.3 provenance spot-check on extracted units."""
        return run_provenance_spotcheck(self.units, sample_size=sample_size)

    def evaluate(
        self,
        experiment_name: str = "baseline",
        k: Optional[int] = None,
        save_report: bool = True,
    ) -> EvalReport:
        """Runs Stage 7: Layer A per-type retrieval metrics, Layer B answer quality and the
        5-answer faithfulness check, saved as one report (`reports/<experiment_name>.json`)."""
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
        faith = run_faithfulness_spotcheck(self.retriever, questions, self.cfg, sample_size=5)
        report.faithfulness_pass_rate = faith["faithfulness_pass_rate"]
        report.faithfulness_details = faith["reviews"]
        report.answer_quality = evaluate_answer_quality(self.retriever, questions, self.cfg)

        if save_report:
            save_eval_report(report, self.cfg)

        return report
