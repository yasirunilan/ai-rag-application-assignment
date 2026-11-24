"""Content-Addressed Disk Cache for Vision Captions (Person 3 · Story E3.1)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional


class CaptionCache:
    """Stores vision captions keyed by SHA-256(image_bytes + prompt_context).

    Entries record which provider produced them so a caption written by the offline
    fallback is never mistaken for a real model caption once a key is available.
    """

    def __init__(self, cache_dir: str | Path, enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.corrupt = 0

    @staticmethod
    def compute_key(image_bytes: bytes, context_key: str) -> str:
        h = hashlib.sha256()
        h.update(image_bytes)
        h.update(b"||")
        h.update(context_key.encode("utf-8"))
        return h.hexdigest()

    def peek(self, key: str) -> Optional[dict[str, Any]]:
        """Reads an entry without touching hit/miss counters. Unreadable or empty entries are None."""
        if not self.enabled:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.corrupt += 1
            return None
        if not isinstance(data, dict) or not str(data.get("search_text", "")).strip():
            self.corrupt += 1
            return None
        return data

    def get(self, key: str) -> Optional[dict[str, Any]]:
        data = self.peek(key)
        if data is None:
            self.misses += 1
            return None
        self.hits += 1
        return data

    def record_miss(self) -> None:
        """Counts a miss for an entry that existed but was refused (e.g. stale offline caption)."""
        self.misses += 1

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Atomic write so a crash or a teammate's concurrent run never leaves half a JSON file."""
        if not self.enabled:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{key}.json"
        fd, tmp_name = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2, ensure_ascii=False)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
