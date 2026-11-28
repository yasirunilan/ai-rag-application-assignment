"""Scanned Page Detector & OCR Quality Gate (Person 1 · Story E1.2).

Detects image-only pages (no visible text layer + page-filling image) AND full-page scans
carrying corrupted/garbled OCR text, renders and OCRs them, scores mean confidence +
garbled-char ratio + alphanumeric-token ratio + word count, and sends sub-threshold pages
to `review_queue.jsonl` instead of the index.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import io
import json
import logging
from pathlib import Path
import re
import shutil
import string
from typing import Any, Optional

import fitz
import numpy as np
from PIL import Image

from mmrag.config import AppConfig
from mmrag.schemas import ReviewQueueItem

logger = logging.getLogger(__name__)

OCR_RENDER_DPI = 300          # Tesseract is tuned for ~300 DPI input
HEURISTIC_RENDER_DPI = 150    # the offline image heuristic was calibrated at 150 DPI
OCR_TIMEOUT_SECONDS = 45.0    # heavy speckle noise can make Tesseract run for minutes; treat as failed OCR

GARBLED_NOISE_CHARS = set("~|^&*@#$_\\<>`{}")
ALLOWED_SYMBOLS = set(".,;:-/()[]%+='\"!?") | set("‘’“”–—•·×÷≤≥≈≠µμΩ°±ØøΦφ…§©®™½¼¾²³")
_REPEATED_NOISE = re.compile(r"[~|%^&*@#?=!_]{2,}|1lI")
ALLOWED_TECH_CHARS = set(string.ascii_letters + string.digits) | ALLOWED_SYMBOLS | set(" \t\n\r")


def _is_clean_char(c: str) -> bool:
    if c in GARBLED_NOISE_CHARS or c == "\ufffd":
        return False
    return c.isalnum() or c in ALLOWED_SYMBOLS


def compute_garbled_char_ratio(text: str) -> float:
    """Ratio of noise characters among non-whitespace characters (Unicode letters/digits are clean)."""
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 1.0
    garbled = sum(1 for c in non_ws if not _is_clean_char(c))
    repeated_noise = len(_REPEATED_NOISE.findall(text)) * 3
    return min(1.0, (garbled + repeated_noise) / max(1, len(non_ws)))


def compute_alnum_token_ratio(text: str) -> float:
    """Fraction of tokens that contain at least 2 alphanumeric characters."""
    tokens = text.split()
    if not tokens:
        return 0.0
    valid = sum(1 for tok in tokens if sum(c.isalnum() for c in tok) >= 2 and "1lI" not in tok)
    return valid / len(tokens)


def _visible_text_char_count(page: fitz.Page) -> Optional[int]:
    """Non-whitespace characters drawn visibly; None if PyMuPDF cannot trace text on this page."""
    try:
        traces = page.get_texttrace()
    except Exception:  # noqa: BLE001
        return None
    visible = 0
    for span in traces:
        if span.get("type") == 3 or float(span.get("opacity", 1.0)) <= 0.0:
            continue  # render mode 3 = invisible (typical OCR layer under a scan)
        for ch in span.get("chars", ()):
            code = ch[0] if ch else 32
            if not chr(code).isspace():
                visible += 1
    return visible


def is_scanned_page(
    page: fitz.Page,
    max_text_chars: int = 50,
    min_image_coverage: float = 0.60,
) -> tuple[bool, int, float]:
    """Detects whether a PDF page is a scanned image page (Story E1.2)."""
    clean_text = page.get_text("text").strip()
    text_char_count = len(clean_text)

    page_area = max(1.0, abs(page.rect.width * page.rect.height))
    image_area = 0.0
    for img_info in page.get_image_info(xrefs=True):
        bbox = img_info.get("bbox")
        if bbox:
            rect = fitz.Rect(bbox) & page.rect
            if not rect.is_empty:
                image_area += rect.width * rect.height
    coverage = min(1.0, image_area / page_area)

    if text_char_count > max_text_chars and coverage >= min_image_coverage:
        visible = _visible_text_char_count(page)
        if visible is not None and visible <= max_text_chars:
            return True, visible, coverage
        if coverage >= 0.80 and compute_garbled_char_ratio(clean_text) > 0.25:
            return True, text_char_count, coverage

    is_scanned = (text_char_count <= max_text_chars) and (coverage >= min_image_coverage)
    return is_scanned, text_char_count, coverage


@lru_cache(maxsize=1)
def _find_tesseract_cmd() -> Optional[str]:
    cmd = shutil.which("tesseract")
    if cmd:
        return cmd
    for candidate in (
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path("/opt/homebrew/bin/tesseract"),
        Path("/usr/local/bin/tesseract"),
    ):
        if candidate.exists():
            return str(candidate)
    return None


def tesseract_status() -> tuple[bool, str]:
    """(available, version-or-reason) for the model/environment check."""
    cmd = _find_tesseract_cmd()
    if not cmd:
        return False, "tesseract binary not found (brew install tesseract | apt install tesseract-ocr)"
    try:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = cmd
        return True, f"tesseract {pytesseract.get_tesseract_version()} at {cmd}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


@lru_cache(maxsize=1)
def _warn_tesseract_missing_once() -> None:
    logger.warning(
        "ocr_engine=tesseract but no working Tesseract was found; scanned pages fall back to the offline "
        "heuristic and are queued for review instead of indexed. Install Tesseract to OCR them."
    )


def render_page_image(page: fitz.Page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def ocr_image_tesseract(
    img: Image.Image,
    lang: str = "eng",
    timeout: float = OCR_TIMEOUT_SECONDS,
) -> tuple[str, float]:
    """OCRs an image with Tesseract; returns (text, mean word confidence 0..100)."""
    cmd = _find_tesseract_cmd()
    if not cmd:
        raise RuntimeError("tesseract binary not found")
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = cmd
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT, timeout=timeout)
    words: list[str] = []
    confs: list[float] = []
    for word, conf in zip(data.get("text", []), data.get("conf", [])):
        w = str(word).strip()
        if not w:
            continue
        words.append(w)
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if c >= 0:
            confs.append(c)
    ocr_text = " ".join(words)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return ocr_text, mean_conf


def heuristic_ocr_image(img: Image.Image) -> tuple[str, float]:
    """Deterministic offline stand-in for OCR (no engine installed / `heuristic_fallback`)."""
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    contrast_std = float(np.std(arr))
    row_means = np.mean(arr, axis=1)
    row_oscillation = float(np.mean(np.abs(np.diff(row_means))))
    if row_oscillation > 3.0 or contrast_std < 38.0:
        simulated_garbled_text = (
            "~|1lI| %%% ^&* R3l4y c0ntr0l w1r1ng @#$ ~|1lI| ??? === /// ||| "
            "t0rqu3 sp3c AN4-12A ~|1lI| %%% ^&* FAULT-E409"
        )
        return simulated_garbled_text, 28.5
    return "", 0.0


_OCR_MEMO: dict[tuple[str, int, str], tuple[str, float]] = {}


def _page_identity(page: fitz.Page) -> str:
    name = getattr(page.parent, "name", "") or ""
    if name and Path(name).is_file():
        st = Path(name).stat()
        return f"{Path(name).resolve()}|{st.st_size}|{st.st_mtime_ns}|{page.number}"
    return hashlib.sha256(page.get_pixmap(dpi=72).samples).hexdigest()


def run_ocr_on_page(
    page: fitz.Page,
    engine: str = "tesseract",
    dpi: int = OCR_RENDER_DPI,
    lang: str = "eng",
) -> tuple[str, float]:
    """Renders a scanned PDF page and returns (ocr_text, mean_confidence 0..100)."""
    if engine == "tesseract":
        if _find_tesseract_cmd():
            try:
                key = (_page_identity(page), dpi, lang)
                if key not in _OCR_MEMO:
                    _OCR_MEMO[key] = ocr_image_tesseract(render_page_image(page, dpi), lang=lang)
                return _OCR_MEMO[key]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tesseract failed on page %s (%s: %s); queuing for review.",
                               page.number + 1, type(exc).__name__, exc)
                return "", 0.0
        _warn_tesseract_missing_once()
    return heuristic_ocr_image(render_page_image(page, HEURISTIC_RENDER_DPI))


def gate_failure_reasons(
    ocr_text: str,
    mean_confidence: float,
    garbled_ratio: float,
    alnum_ratio: float,
    min_mean_confidence: float,
    max_garbled_char_ratio: float,
    min_alnum_token_ratio: float,
    min_word_count: int = 0,
) -> list[str]:
    """Pure threshold check shared by the gate and the calibration sweep."""
    reasons: list[str] = []
    n_words = len(ocr_text.split())
    if not ocr_text.strip():
        reasons.append("empty_ocr_output")
    elif n_words < min_word_count:
        reasons.append(f"too_few_words ({n_words} < {min_word_count})")
    if mean_confidence < min_mean_confidence:
        reasons.append(f"low_mean_confidence ({mean_confidence:.1f} < {min_mean_confidence:.1f})")
    if garbled_ratio > max_garbled_char_ratio:
        reasons.append(f"high_garbled_char_ratio ({garbled_ratio:.2f} > {max_garbled_char_ratio:.2f})")
    if alnum_ratio < min_alnum_token_ratio:
        reasons.append(f"low_alnum_token_ratio ({alnum_ratio:.2f} < {min_alnum_token_ratio:.2f})")
    return reasons


def evaluate_ocr_quality_gate(
    doc_id: str,
    page_num: int,
    source_path: str,
    ocr_text: str,
    mean_confidence: float,
    cfg: AppConfig,
    append_to_queue: bool = True,
) -> tuple[bool, float, float, list[str]]:
    """Scores OCR output and sends sub-threshold pages to review_queue.jsonl (Story E1.2)."""
    garbled_ratio = compute_garbled_char_ratio(ocr_text)
    alnum_ratio = compute_alnum_token_ratio(ocr_text)
    gate_cfg = cfg.ingest.ocr_quality_gate

    reasons = gate_failure_reasons(
        ocr_text,
        mean_confidence,
        garbled_ratio,
        alnum_ratio,
        gate_cfg.min_mean_confidence,
        gate_cfg.max_garbled_char_ratio,
        gate_cfg.min_alnum_token_ratio,
        gate_cfg.min_word_count,
    )

    passed = (len(reasons) == 0) if gate_cfg.enabled else True

    if len(reasons) > 0 and append_to_queue and gate_cfg.enabled:
        q_path = Path(cfg.project.review_queue_path)
        q_path.parent.mkdir(parents=True, exist_ok=True)
        existing_keys: set[tuple[str, int]] = set()
        if q_path.exists():
            for line in q_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    existing_keys.add((row.get("doc_id", ""), int(row.get("page", 0))))
        if (doc_id, page_num) not in existing_keys:
            item = ReviewQueueItem(
                doc_id=doc_id,
                page=page_num,
                source_path=source_path,
                mean_confidence=round(mean_confidence, 2),
                garbled_char_ratio=round(garbled_ratio, 4),
                alnum_token_ratio=round(alnum_ratio, 4),
                failure_reasons=reasons,
                raw_ocr_preview=ocr_text[:200],
            )
            with q_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")

    return passed, garbled_ratio, alnum_ratio, reasons


def summarize_review_queue(queue_path: str | Path) -> dict[str, Any]:
    """Story E1.2: Reads `review_queue.jsonl` and reports failed scanned page count & reasons."""
    p = Path(queue_path)
    if not p.exists():
        return {"failed_pages_count": 0, "items": []}
    items = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "failed_pages_count": len(items),
        "pages": [f"{it['doc_id']} p.{it['page']}" for it in items],
        "items": items,
    }
