"""Unit, Provenance and API types. A unit with no resolvable page is not a sourced answer."""
from typing import Any, Literal, Optional

from pydantic import BaseModel, field_validator

UnitType = Literal["text", "figure", "table"]


class Provenance(BaseModel):
    document: str
    page: int
    unit_id: str
    bbox: Optional[list[float]] = None
    source_path: Optional[str] = None

    @field_validator("document", "unit_id")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("provenance fields must be non-empty")
        return v


class Unit(BaseModel):
    type: UnitType
    search_text: str            # what we EMBED
    artefact: Any               # what we RETURN: text span | image path | structured table rows
    provenance: Provenance


class Hit(BaseModel):
    unit: Unit
    score: float
    citation: str               # e.g. "manual.pdf > p.12 > fig_03"


class QueryRequest(BaseModel):
    question: str
    k: int = 5
    type_filter: Optional[UnitType] = None


class QueryResponse(BaseModel):
    answer: str
    answer_type: Optional[UnitType] = None
    artefact: Any = None
    citation: Optional[str] = None
    sources: list[Hit] = []
    abstained: bool = False
    reason: Optional[str] = None
    timings: dict[str, float] = {}
