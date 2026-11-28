"""FastAPI service: /query /health /examples /artefact/{unit_id} (Story E5.3).

One entry point shared by the UI and the evaluator. Every answer carries the real artefact
reference, its citation, the sources panel data and per-stage timings. Configuration problems
(no index, wrong embedder) return 503 with `kind: "configuration"`; upstream model problems
return 502 with `kind: "upstream"`, so the UI can show different messages.

Run:  MMRAG_OFFLINE=1 uvicorn --factory mmrag.api.server:create_app --port 8000
      (build the index first: python -m mmrag.cli ingest --offline)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from mmrag.answer.responder import build_answer_from_hits
from mmrag.config import AppConfig, BillingGuardError, load_config
from mmrag.index.embedder import create_embedder
from mmrag.index.store import MixedVectorStore
from mmrag.retrieve.retriever import HybridRetriever
from mmrag.schemas import SearchHit, TableArtefact, Unit

EXAMPLE_QUESTIONS = [
    {"type": "figure", "question": "Show the engine starter relay control wiring diagram for Model 737-800"},
    {"type": "table", "question": "What is the dry and lubed torque spec table for bolt part number AN4-12A?"},
    {"type": "table", "question": "Lookup table entry and mandatory action for code FAULT-E409"},
    {"type": "text", "question": "What nitrogen pre-charge pressure is required on the Model A320 hydraulic accumulator at 20 degrees Celsius?"},
]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: Optional[int] = Field(default=None, ge=1, le=20)
    type_filter: Optional[str] = Field(default=None, pattern="^(text|figure|table)$")


class SourceItem(BaseModel):
    rank: int
    unit_id: str
    type: str
    citation: str
    score: float
    cited: bool
    preview: str


class QueryResponse(BaseModel):
    question: str
    abstained: bool
    abstain_reason: Optional[str] = None
    modality: Optional[str] = None
    citation: Optional[str] = None
    unit_id: Optional[str] = None
    caption: Optional[str] = None
    image_url: Optional[str] = None           # figures: GET this to get the real image
    table: Optional[dict[str, Any]] = None    # tables: exact structured rows
    text: Optional[str] = None                # text answers: the supporting span
    sources: list[SourceItem] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


class ServiceState:
    """Holds the loaded index; load failures are kept so /health can explain them."""

    def __init__(self, cfg: AppConfig, offline: bool) -> None:
        self.cfg = cfg
        self.offline = offline
        self.retriever: Optional[HybridRetriever] = None
        self.store: Optional[MixedVectorStore] = None
        self.problems: list[dict[str, str]] = []
        self.load()

    def load(self) -> None:
        self.problems = []
        self.retriever = self.store = None
        index_dir = Path(self.cfg.project.index_dir)
        if not (index_dir / "units.jsonl").exists():
            self.problems.append({"problem": "stale_or_missing_index", "fix": "run: python -m mmrag.cli ingest"})
            return
        try:
            embedder = create_embedder(self.cfg, force_offline=self.offline)
            store = MixedVectorStore.load(self.cfg, embedder=embedder)
        except BillingGuardError as exc:
            self.problems.append({"problem": f"billing_guard: {exc}", "fix": "use :free models only"})
            return
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim to /health
            self.problems.append({"problem": f"index_load_failed: {exc}", "fix": "rebuild the index"})
            return
        meta_path = index_dir / "index_meta.json"
        if meta_path.exists():
            dim = json.loads(meta_path.read_text(encoding="utf-8")).get("embedding_dim")
            if dim and dim != self.cfg.index.embedding_dim and not self.offline:
                self.problems.append(
                    {
                        "problem": f"embedding model mismatch: index dim {dim} vs configured {self.cfg.index.embedding_dim}",
                        "fix": "rebuild the index with the configured embedding model",
                    }
                )
                return
        if not store.units:
            self.problems.append({"problem": "index_is_empty", "fix": "run: python -m mmrag.cli ingest"})
            return
        self.store = store
        self.retriever = HybridRetriever(store, self.cfg)

    def find_unit(self, unit_id: str) -> Optional[Unit]:
        if not self.store:
            return None
        return next((u for u in self.store.units if u.unit_id == unit_id), None)


def _preview(u: Unit) -> str:
    if isinstance(u.artefact, TableArtefact):
        return f"{len(u.artefact.rows)} rows x {len(u.artefact.headers)} cols"
    return (u.search_text or str(u.artefact))[:160].replace("\n", " ")


def _sources(hits: list[SearchHit], cited_id: Optional[str]) -> list[SourceItem]:
    return [
        SourceItem(
            rank=h.rank, unit_id=h.unit.unit_id, type=h.unit.type, citation=h.citation,
            score=round(h.score, 4), cited=h.unit.unit_id == cited_id, preview=_preview(h.unit),
        )
        for h in hits
    ]


def create_app(cfg: Optional[AppConfig] = None, offline: Optional[bool] = None) -> FastAPI:
    cfg = cfg or load_config(os.environ.get("MMRAG_CONFIG", "config.yaml"))
    if offline is None:
        offline = os.environ.get("MMRAG_OFFLINE", "") == "1"
    state = ServiceState(cfg, offline)
    app = FastAPI(title="mmrag API", version="0.1.0")
    app.state.svc = state

    origins = [o for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",") if o]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])

    def require_ready() -> HybridRetriever:
        if state.retriever is None:
            raise HTTPException(
                status_code=503,
                detail={"kind": "configuration", "problems": state.problems},
            )
        return state.retriever

    @app.get("/health")
    def health() -> dict[str, Any]:
        ok = state.retriever is not None
        counts: dict[str, int] = {}
        for u in (state.store.units if state.store else []):
            counts[u.type] = counts.get(u.type, 0) + 1
        return {
            "status": "healthy" if ok else "degraded",
            "problems": state.problems,
            "units": counts,
            "embedding_provider": cfg.index.embedding_provider,
            "embedding_model": cfg.index.embedding_model_name,
            "offline": state.offline,
        }

    @app.get("/examples")
    def examples() -> list[dict[str, str]]:
        return EXAMPLE_QUESTIONS

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        retriever = require_ready()
        t0 = time.perf_counter()
        try:
            hits = retriever.retrieve(
                req.question,
                k=req.k or cfg.retrieve.top_k,
                type_filter=req.type_filter,
            )
        except Exception as exc:  # noqa: BLE001 - embedder / upstream failure
            raise HTTPException(
                status_code=502, detail={"kind": "upstream", "message": f"{type(exc).__name__}: {exc}"}
            ) from exc
        t1 = time.perf_counter()
        ans = build_answer_from_hits(req.question, hits, cfg)
        t2 = time.perf_counter()
        timings = {"retrieval": round((t1 - t0) * 1000, 2), "answer": round((t2 - t1) * 1000, 2),
                   "total": round((t2 - t0) * 1000, 2)}

        if ans.abstained or ans.source_unit is None:
            return QueryResponse(
                question=req.question, abstained=True, abstain_reason=ans.abstain_reason,
                sources=_sources(ans.top_hits, None), timings_ms=timings,
            )
        u = ans.source_unit
        resp = QueryResponse(
            question=req.question, abstained=False, modality=u.type, citation=ans.citation,
            unit_id=u.unit_id, caption=ans.grounded_caption,
            sources=_sources(ans.top_hits, u.unit_id), timings_ms=timings,
        )
        if u.type == "figure":
            resp.image_url = f"/artefact/{u.unit_id}"
        elif u.type == "table" and isinstance(u.artefact, TableArtefact):
            resp.table = u.artefact.model_dump()
        else:
            resp.text = str(u.artefact)
        return resp

    @app.get("/artefact/{unit_id:path}")
    def artefact(unit_id: str):
        require_ready()
        u = state.find_unit(unit_id)
        if u is None or not u.has_valid_provenance(verify_files=True):
            raise HTTPException(status_code=404, detail={"kind": "not_found", "message": "unknown or unsourced unit"})
        if u.type == "figure":
            path = Path(str(u.artefact)).resolve()
            data_root = Path(cfg.project.data_dir).resolve()
            if data_root not in path.parents:  # never read outside the data folder
                raise HTTPException(status_code=403, detail={"kind": "forbidden", "message": "outside data folder"})
            return FileResponse(path, media_type="image/png")
        if isinstance(u.artefact, TableArtefact):
            return JSONResponse(u.artefact.model_dump())
        return JSONResponse({"text": str(u.artefact), "citation": u.provenance.citation})

    return app

