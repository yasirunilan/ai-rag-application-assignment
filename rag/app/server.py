"""HTTP API: POST /query, GET /health, GET /examples, GET /artefact/{id}."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import load_settings

settings = load_settings(require_key=False)
app = FastAPI(title="Multimodal RAG")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "degraded" if not settings.api_key else "ok",
            "reason": None if settings.api_key else "OPENROUTER_API_KEY missing"}


@app.get("/examples")
def examples():
    return {"examples": []}  # TODO: three clickable example questions (text, figure, table)


# TODO: POST /query (E5/E6), GET /artefact/{id} with path-traversal guard (E5.3)
