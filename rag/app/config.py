"""Single source of truth for key, model slugs, vector backend, dimensions.
Nothing else in the project reads the environment."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    embed_model: str
    embed_dimensions: int
    embed_images: bool
    vision_model: str
    text_model: str
    figure_model: str
    vector_backend: str
    qdrant_url: str
    cors_origins: list[str]
    data_dir: Path = DATA_DIR


def load_settings(require_key: bool = True) -> Settings:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if require_key and not key:
        raise ConfigError("OPENROUTER_API_KEY is missing. Set it in rag/.env (never commit it).")
    return Settings(
        api_key=key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        embed_model=os.getenv("EMBED_MODEL", "nvidia/nemotron-3.5-embed-1b:free"),
        embed_dimensions=int(os.getenv("EMBED_DIMENSIONS", "1024")),
        embed_images=os.getenv("EMBED_IMAGES", "false").lower() == "true",
        vision_model=os.getenv("VISION_MODEL", "nvidia/nemotron-3.5-super-vl:free"),
        text_model=os.getenv("TEXT_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
        figure_model=os.getenv("FIGURE_MODEL", "nvidia/nemotron-3.5-super-vl:free"),
        vector_backend=os.getenv("VECTOR_BACKEND", "qdrant"),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        cors_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")],
    )
