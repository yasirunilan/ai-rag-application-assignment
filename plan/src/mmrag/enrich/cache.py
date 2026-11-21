"""Content-Addressed Disk Cache for Vision Captions (Person 3 · Story E3.1)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


class CaptionCache:
    """Stores vision captions keyed by SHA-256(image_bytes + prompt_context)."""

    def __init__(self, cache_dir: str | Path, enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def compute_key(image_bytes: bytes, context_key: str) -> str:
        h = hashlib.sha256()
        h.update(image_bytes)
        h.update(b"||")
        h.update(context_key.encode("utf-8"))
        return h.hexdigest()

    def get(self, key: str) -> Optional[dict[str, Any]]:
        if not self.enabled:
            self.misses += 1
            return None
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.hits += 1
                return data
            except Exception:
                pass
        self.misses += 1
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
