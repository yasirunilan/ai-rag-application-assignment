"""Configuration Loader & Billing Guard (Person 1 · Story E0.1 & E3.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
import yaml
from pydantic import BaseModel, Field, model_validator


class BillingGuardError(RuntimeError):
    """Raised when a paid model is configured on the default free/local path."""


class ProjectConfig(BaseModel):
    name: str = "mmrag-technical-manuals"
    data_dir: str = "data"
    artifacts_dir: str = "data/artifacts"
    cache_dir: str = "data/cache/captions"
    index_dir: str = "data/index"
    review_queue_path: str = "data/review_queue.jsonl"
    reports_dir: str = "reports"


class CorpusConfig(BaseModel):
    eval_dataset: str = "dude_slice"
    real_manual: str = "amt_handbook"
    max_pages: int = 250
    gold_eval_path: str = "data/corpus/gold_eval.jsonl"
    manuals_dir: str = "data/corpus/manuals"


class OcrQualityGateConfig(BaseModel):
    enabled: bool = True
    min_mean_confidence: float = 60.0
    max_garbled_char_ratio: float = 0.15
    min_alnum_token_ratio: float = 0.50


class IngestConfig(BaseModel):
    parser_backend: Literal["pymupdf", "docling"] = "pymupdf"
    scanned_max_text_chars: int = 50
    scanned_min_image_coverage: float = 0.60
    ocr_engine: Literal["tesseract", "heuristic_fallback"] = "tesseract"
    ocr_quality_gate: OcrQualityGateConfig = Field(default_factory=OcrQualityGateConfig)


class ExtractConfig(BaseModel):
    figure_render_dpi: int = 200
    min_figure_width_pt: float = 70.0
    min_figure_height_pt: float = 55.0
    min_figure_area_pt2: float = 5000.0
    min_aspect_ratio: float = 0.15
    max_aspect_ratio: float = 6.50
    caption_search_margin_pt: float = 45.0
    table_backend: Literal["pdfplumber", "pymupdf"] = "pdfplumber"
    flatten_tables: bool = False


class EnrichConfig(BaseModel):
    vision_provider: Literal["openrouter_free", "local_vision", "offline_heuristic"] = "openrouter_free"
    enforce_free_only: bool = True
    openrouter_models: list[str] = Field(
        default_factory=lambda: [
            "qwen/qwen2.5-vl-72b-instruct:free",
            "qwen/qwen2.5-vl-7b-instruct:free",
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "openrouter/free",
        ]
    )
    local_vision_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    cache_enabled: bool = True
    selective_captioning: bool = True
    include_nearby_context: bool = True
    table_linearization_mode: Literal["col_val_and_markdown", "flat_raw"] = "col_val_and_markdown"

    @model_validator(mode="after")
    def verify_free_models_only(self) -> "EnrichConfig":
        if self.enforce_free_only:
            for model_id in self.openrouter_models:
                if not (model_id.endswith(":free") or model_id == "openrouter/free"):
                    raise BillingGuardError(
                        f"Paid model '{model_id}' is forbidden by §6 billing guard. "
                        "All OpenRouter models must end with ':free' or be 'openrouter/free'."
                    )
        return self


class IndexConfig(BaseModel):
    embedding_provider: Literal["local_bge", "deterministic_hash"] = "local_bge"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    vector_backend: Literal["faiss_or_numpy", "numpy"] = "faiss_or_numpy"
    prose_chunk_max_chars: int = 750
    prose_chunk_overlap_chars: int = 100


class RetrieveConfig(BaseModel):
    top_k: int = 5
    candidate_pool_size: int = 50
    enable_bm25_fusion: bool = True
    alpha: float = 0.60


class AnswerConfig(BaseModel):
    fail_closed_on_missing_provenance: bool = True
    min_score_threshold: float = 0.05
    promote_explicit_modality_intent: bool = True
    # Story E6.2: fused scores are min-max normalized, so the top hit always scores high even for an
    # off-topic question. Abstain unless this share of the query's terms appears in the cited unit.
    min_query_term_coverage: float = 0.25


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    enrich: EnrichConfig = Field(default_factory=EnrichConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieve: RetrieveConfig = Field(default_factory=RetrieveConfig)
    answer: AnswerConfig = Field(default_factory=AnswerConfig)


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return AppConfig()
    with cfg_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)
