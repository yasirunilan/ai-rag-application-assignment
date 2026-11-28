"""Configuration Loader & Billing Guard (Person 1 · Stories E0.1, E0.2, E0.3 & E3.1).

Single source of truth for paths, model slugs, vector backend, and `.env` / environment
overrides. Strips accidental whitespace/quotes around keys and enforces `:free`-only slugs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional
import yaml
from pydantic import BaseModel, Field, model_validator


class BillingGuardError(RuntimeError):
    """Raised when a paid model is configured on the default free/local path."""


def _is_free_slug(model_id: str) -> bool:
    return model_id.endswith(":free") or model_id == "openrouter/free"


def load_dotenv(path: str | Path = ".env", override: bool = False) -> list[str]:
    """Loads KEY=VALUE lines from a local, git-ignored `.env` file into `os.environ` (Story E0.1).

    Keeps the API key out of `config.yaml` without adding a dependency. Existing environment
    variables win unless `override` is set. Returns the names of the variables that were set.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return []
    loaded: list[str] = []
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1].strip()
        if not key or (key in os.environ and not override):
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded


def get_openrouter_api_key() -> str:
    """Returns sanitized `OPENROUTER_API_KEY` (strips stray spaces/quotes warned about in Stage 0)."""
    return os.environ.get("OPENROUTER_API_KEY", "").strip().strip("'\"")


class ProjectConfig(BaseModel):
    name: str = "mmrag-technical-manuals"
    data_dir: str = "data"
    artifacts_dir: str = "data/artifacts"
    cache_dir: str = "data/cache/captions"
    index_dir: str = "data/index"
    review_queue_path: str = "data/review_queue.jsonl"
    enrichment_rejects_path: str = "data/rejected_enrichment_units.jsonl"
    reports_dir: str = "reports"


class CorpusSource(BaseModel):
    """One real document (or a directory of documents) added on top of the gold-labelled core corpus."""

    doc_id: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    path: str
    url: Optional[str] = None
    max_pages: Optional[int] = Field(default=None, ge=1)
    enabled: bool = True


class CorpusConfig(BaseModel):
    eval_dataset: str = "dude_slice"
    real_manual: str = "amt_handbook"
    max_pages: int = Field(default=250, ge=1)
    gold_eval_path: str = "data/corpus/gold_eval.jsonl"
    manuals_dir: str = "data/corpus/manuals"
    include_core_manuals: bool = True
    downloads_dir: str = "data/corpus/downloads"
    slices_dir: str = "data/corpus/slices"
    real_sources: list[CorpusSource] = Field(default_factory=list)
    extra_gold_paths: list[str] = Field(default_factory=list)
    enginuity_dir: Optional[str] = None
    stage0_prediction_worst_type: Literal["text", "figure", "table"] = "figure"
    stage0_prediction_reason: str = (
        "Figures contain zero native lexical text until captioned, and visually similar "
        "wiring schematics drift without nearby figure labels and model variants."
    )

    @model_validator(mode="after")
    def unique_doc_ids(self) -> "CorpusConfig":
        seen: set[str] = set()
        for src in self.real_sources:
            if src.doc_id in seen:
                raise ValueError(f"Duplicate corpus.real_sources doc_id '{src.doc_id}'.")
            seen.add(src.doc_id)
        return self


class OcrQualityGateConfig(BaseModel):
    enabled: bool = True
    min_mean_confidence: float = 60.0
    max_garbled_char_ratio: float = 0.15
    min_alnum_token_ratio: float = 0.50
    min_word_count: int = 10


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
            "google/gemma-4-31b-it:free",
            "nvidia/nemotron-3.5-super-vl:free",
            "google/gemma-4-26b-a4b-it:free",
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
                if not _is_free_slug(model_id):
                    raise BillingGuardError(
                        f"Paid model '{model_id}' is forbidden by §6 billing guard. "
                        "All OpenRouter models must end with ':free' or be 'openrouter/free'."
                    )
        return self


class IndexConfig(BaseModel):
    embedding_provider: Literal["local_bge", "deterministic_hash"] = "local_bge"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    openrouter_embed_model: str = "nvidia/nemotron-3.5-embed-1b:free"
    embedding_dim: int = 384
    vector_backend: Literal["faiss_or_numpy", "numpy"] = "faiss_or_numpy"
    prose_chunk_max_chars: int = 750
    prose_chunk_overlap_chars: int = 100

    @model_validator(mode="after")
    def verify_free_embed_slug(self) -> "IndexConfig":
        if self.openrouter_embed_model and not _is_free_slug(self.openrouter_embed_model):
            raise BillingGuardError(
                f"Paid embedding model '{self.openrouter_embed_model}' is forbidden; use a ':free' slug."
            )
        return self


class RetrieveConfig(BaseModel):
    top_k: int = 5
    candidate_pool_size: int = 50
    enable_bm25_fusion: bool = True
    alpha: float = 0.60


class AnswerConfig(BaseModel):
    fail_closed_on_missing_provenance: bool = True
    min_score_threshold: float = 0.05
    promote_explicit_modality_intent: bool = True
    min_query_term_coverage: float = 0.25
    enable_remote_generation: bool = False
    enforce_free_only: bool = True
    text_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    figure_model: str = "nvidia/nemotron-3.5-super-vl:free"

    @model_validator(mode="after")
    def verify_free_generation_slugs(self) -> "AnswerConfig":
        if self.enforce_free_only:
            for slug in (self.text_model, self.figure_model):
                if not _is_free_slug(slug):
                    raise BillingGuardError(
                        f"Paid generator model '{slug}' is forbidden by §6 billing guard."
                    )
        return self


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    enrich: EnrichConfig = Field(default_factory=EnrichConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieve: RetrieveConfig = Field(default_factory=RetrieveConfig)
    answer: AnswerConfig = Field(default_factory=AnswerConfig)


def _apply_env_overrides(cfg: AppConfig) -> AppConfig:
    """Applies standard `.env` / environment slug overrides from a single place (Story E0.1)."""
    vision_slug = os.environ.get("VISION_MODEL", "").strip()
    if vision_slug:
        rest = [m for m in cfg.enrich.openrouter_models if m != vision_slug]
        cfg.enrich = cfg.enrich.model_copy(update={"openrouter_models": [vision_slug, *rest]})

    embed_slug = os.environ.get("EMBED_MODEL", "").strip()
    if embed_slug:
        cfg.index = cfg.index.model_copy(update={"openrouter_embed_model": embed_slug})

    vec_backend = os.environ.get("VECTOR_BACKEND", "").strip()
    if vec_backend in ("faiss_or_numpy", "numpy"):
        cfg.index = cfg.index.model_copy(update={"vector_backend": vec_backend})

    text_slug = os.environ.get("TEXT_MODEL", "").strip()
    figure_slug = os.environ.get("FIGURE_MODEL", "").strip()
    if text_slug or figure_slug:
        updates: dict[str, Any] = {}
        if text_slug:
            updates["text_model"] = text_slug
        if figure_slug:
            updates["figure_model"] = figure_slug
        cfg.answer = cfg.answer.model_copy(update=updates)
    return cfg


def load_config(path: str | Path = "config.yaml", dotenv_path: str | Path | None = None) -> AppConfig:
    if dotenv_path is not None:
        load_dotenv(dotenv_path)
    cfg_path = Path(path)
    if not cfg_path.exists():
        return _apply_env_overrides(AppConfig())
    with cfg_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return _apply_env_overrides(AppConfig.model_validate(raw))
