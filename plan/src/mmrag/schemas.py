"""Phase-0 Data Backbone & Shared Schemas (Fig 2 & Fig 3 of Brief).

Every stored thing is a `Unit` that separates what we search (`search_text`)
from what we return (`artefact`), tied together by `provenance` (`doc_id -> page -> unit_id`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, Field


UnitType = Literal["text", "figure", "table"]


class Provenance(BaseModel):
    """End-to-end provenance carried on every Unit (Non-Negotiable #3)."""

    doc_id: str
    page: Optional[int] = None  # 1-indexed page number; None or <1 => not a sourced answer (fail closed)
    unit_id: str
    bbox: Optional[tuple[float, float, float, float]] = None  # (x0, y0, x1, y1) in PDF points
    source_path: str = ""
    heading_context: Optional[str] = None
    nearby_caption: Optional[str] = None
    model_variant: Optional[str] = None
    extraction_meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def citation(self) -> str:
        if not self.doc_id or self.page is None or self.page < 1:
            return "UNSOURCED"
        return f"{self.doc_id} p.{self.page}"


class TableArtefact(BaseModel):
    """Structured table artefact returned to the user (Story E2.2 & E6.1).

    Never flatten this into a plain string in the final answer.
    """

    headers: list[str]
    rows: list[list[str]]
    caption: Optional[str] = None

    def to_markdown(self) -> str:
        if not self.headers and not self.rows:
            return ""
        cols = self.headers if self.headers else [f"Col_{i+1}" for i in range(len(self.rows[0]))]
        header_line = "| " + " | ".join(str(c).strip() for c in cols) + " |"
        sep_line = "| " + " | ".join("---" for _ in cols) + " |"
        row_lines = []
        for row in self.rows:
            padded = list(row) + [""] * max(0, len(cols) - len(row))
            row_lines.append("| " + " | ".join(str(c).strip() for c in padded[: len(cols)]) + " |")
        return "\n".join([header_line, sep_line, *row_lines])


class Unit(BaseModel):
    """The atom of retrieval (Fig 2: type: text | figure | table)."""

    unit_id: str
    type: UnitType
    search_text: str = ""
    artefact: Union[str, TableArtefact]
    provenance: Provenance

    def has_valid_provenance(self, verify_files: bool = True) -> bool:
        """Fail-closed provenance verification (Story E2.3 & E6.2).

        A unit is only sourced if:
        1. doc_id is non-empty
        2. page is an integer >= 1
        3. unit_id is non-empty
        4. Its artefact is present (and for figures, the image file exists on disk).
        """
        if not self.provenance.doc_id or not self.provenance.unit_id:
            return False
        if self.provenance.page is None or int(self.provenance.page) < 1:
            return False
        if self.type == "figure":
            if not isinstance(self.artefact, str) or not self.artefact.strip():
                return False
            if verify_files and not Path(self.artefact).exists():
                return False
        elif self.type == "table":
            if isinstance(self.artefact, TableArtefact):
                if not self.artefact.rows and not self.artefact.headers:
                    return False
            elif isinstance(self.artefact, str):
                if not self.artefact.strip():
                    return False
            else:
                return False
        elif self.type == "text":
            if not isinstance(self.artefact, str) or not self.artefact.strip():
                return False
        return True


class RawRegion(BaseModel):
    """Intermediate layout region produced by Stage 1 Structural Router (Story E1.1)."""

    doc_id: str
    page: int
    region_type: Literal["text", "figure", "table", "scanned_page"]
    bbox: tuple[float, float, float, float]
    reading_order: int
    raw_text: str = ""
    heading_context: Optional[str] = None
    nearby_caption: Optional[str] = None
    model_variant: Optional[str] = None
    source_path: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class PageClassification(BaseModel):
    """Per-page classification & OCR gate diagnostics (Story E1.1 & E1.2)."""

    doc_id: str
    page: int
    is_scanned: bool
    text_char_count: int
    image_coverage_ratio: float
    ocr_attempted: bool = False
    ocr_passed: Optional[bool] = None
    ocr_mean_confidence: Optional[float] = None
    ocr_garbled_ratio: Optional[float] = None
    ocr_alnum_token_ratio: Optional[float] = None
    route_decision: Literal["digital_parse", "ocr_indexed", "ocr_review_queue"]


class ReviewQueueItem(BaseModel):
    """Sub-threshold scanned page entry written to review_queue.jsonl (Story E1.2)."""

    doc_id: str
    page: int
    source_path: str
    mean_confidence: float
    garbled_char_ratio: float
    alnum_token_ratio: float
    failure_reasons: list[str]
    raw_ocr_preview: str


class SearchHit(BaseModel):
    """A single ranked retrieval result with provenance and score breakdown (Story E5.1 & E5.2)."""

    rank: int
    unit: Unit
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    citation: str


class AnswerResponse(BaseModel):
    """Grounded multimodal response returned by Stage 6 (Fig 3, Story E6.1 & E6.2)."""

    query: str
    abstained: bool
    abstain_reason: Optional[str] = None
    modality: Optional[UnitType] = None
    artefact: Optional[Union[str, TableArtefact]] = None
    citation: Optional[str] = None
    grounded_caption: Optional[str] = None
    source_unit: Optional[Unit] = None
    top_hits: list[SearchHit] = Field(default_factory=list)


class GoldQuestion(BaseModel):
    """Evaluation item with page-level gold evidence & modality label (Story E0.2 & E7.1)."""

    query_id: str
    query: str
    gold_doc_id: str
    gold_pages: list[int]
    expected_type: UnitType
    gold_artefact_hint: Optional[str] = None
    expected_keywords: list[str] = Field(default_factory=list)


class TypeMetrics(BaseModel):
    """Recall@k and MRR metrics for a specific content type (Story E7.1)."""

    unit_type: str
    num_queries: int
    page_recall_at_k: float
    page_mrr: float
    strict_recall_at_k: float
    strict_mrr: float


class EvalReport(BaseModel):
    """Complete evaluation report across all types + faithfulness checks (Story E7.1–E7.3)."""

    experiment_name: str
    k: int
    overall_page_recall_at_k: float
    overall_page_mrr: float
    overall_strict_recall_at_k: float
    overall_strict_mrr: float
    by_type: dict[str, TypeMetrics]
    faithfulness_pass_rate: Optional[float] = None
    faithfulness_details: list[dict[str, Any]] = Field(default_factory=list)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
