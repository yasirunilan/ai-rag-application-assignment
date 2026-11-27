"""OCR Quality-Gate Threshold Calibration (Person 1 · Story E1.2 "choose a threshold").

Brief: "Choose a threshold. Pages below it go to a review queue and are not indexed. Look at ten flagged
pages and ten passed pages by eye, then adjust the threshold."

Ground truth is free: we take *digital* corpus pages (their text layer is the reference transcript),
render them as images, apply scan degradations of increasing severity, OCR every variant with
Tesseract, and score OCR word accuracy against the text layer. A variant is `acceptable` when its word
accuracy is at least `min_word_accuracy`. We then sweep the three gate thresholds and pick the setting
that keeps the most acceptable pages while letting (almost) no unacceptable page into the index.

Outputs: a JSON report (`reports/ocr_gate_calibration.json`) with the sweep, the recommended
thresholds, the current thresholds' precision/recall, and ten flagged + ten passed samples whose
rendered images are saved under `data/ocr_calibration/` for review by eye.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from mmrag.config import AppConfig
from mmrag.ingest.ocr_gate import (
    OCR_RENDER_DPI,
    compute_alnum_token_ratio,
    compute_garbled_char_ratio,
    gate_failure_reasons,
    is_scanned_page,
    ocr_image_tesseract,
    render_page_image,
)

_WORD = re.compile(r"[A-Za-z0-9]+(?:[.,\-/][A-Za-z0-9]+)*")


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def word_accuracy(reference: str, hypothesis: str) -> float:
    """Bag-of-words recall: share of reference word occurrences that the OCR recovered.

    Order-insensitive on purpose: multi-column and table pages are read in a different order by
    Tesseract than they are stored in the text layer, and retrieval (BM25 + embeddings) does not
    depend on word order either. Hallucinated extra words are caught by the garbled/alnum metrics.
    """
    ref, hyp = Counter(_words(reference)), Counter(_words(hypothesis))
    total = sum(ref.values())
    if not total:
        return 0.0
    return sum((ref & hyp).values()) / total


# ---------------------------------------------------------------------------------------------------
# Scan degradations (deterministic per seed), ordered roughly by severity
# ---------------------------------------------------------------------------------------------------

def _gray(img: Image.Image) -> Image.Image:
    return img.convert("L")


def _noise(img: Image.Image, sigma: float, seed: int) -> Image.Image:
    arr = np.asarray(_gray(img), dtype=np.float32)
    rng = np.random.default_rng(seed)
    arr = arr + rng.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _contrast(img: Image.Image, factor: float, base: float = 200.0) -> Image.Image:
    arr = np.asarray(_gray(img), dtype=np.float32)
    arr = base + (arr - base) * factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _resample(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def _streaks(img: Image.Image, seed: int, every_px: int = 18, width: int = 3) -> Image.Image:
    out = _gray(img).copy()
    draw = ImageDraw.Draw(out)
    rng = random.Random(seed)
    for y in range(0, out.size[1], every_px):
        jitter = rng.randint(-4, 4)
        draw.line([(0, y), (out.size[0], y + jitter)], fill=90, width=width)
    return out


def _salt_pepper(img: Image.Image, amount: float, seed: int) -> Image.Image:
    arr = np.asarray(_gray(img), dtype=np.uint8).copy()
    rng = np.random.default_rng(seed)
    mask = rng.random(arr.shape)
    arr[mask < amount / 2] = 0
    arr[mask > 1 - amount / 2] = 255
    return Image.fromarray(arr)


DEGRADATIONS: dict[str, Callable[[Image.Image, int], Image.Image]] = {
    "clean": lambda im, s: _gray(im),
    "blur_r1": lambda im, s: _gray(im).filter(ImageFilter.GaussianBlur(1.0)),
    "noise_s25": lambda im, s: _noise(im, 25, s),
    "low_dpi_100": lambda im, s: _resample(im, 1 / 3),
    "blur_r2": lambda im, s: _gray(im).filter(ImageFilter.GaussianBlur(2.0)),
    "faded_noise": lambda im, s: _noise(_contrast(im, 0.30), 18, s),
    "noise_s70": lambda im, s: _noise(im, 70, s),
    "low_dpi_60": lambda im, s: _resample(im, 0.2),
    "streaks": lambda im, s: _streaks(_noise(im, 20, s), s),
    "salt_pepper_25": lambda im, s: _salt_pepper(im, 0.25, s),
    "blur_r3_noise": lambda im, s: _noise(_gray(im).filter(ImageFilter.GaussianBlur(3.0)), 35, s),
    "microfiche": lambda im, s: _streaks(_noise(_resample(_contrast(im, 0.25), 0.25), 30, s), s, 12, 4),
}


@dataclass
class CalibrationSample:
    sample_id: str
    doc_id: str
    page: int
    degradation: str
    mean_confidence: float
    garbled_char_ratio: float
    alnum_token_ratio: float
    word_accuracy: float
    acceptable: bool
    word_count: int
    ocr_preview: str
    image_path: Optional[str] = None


def _select_reference_pages(pdf_paths: list[Path], max_pages: int, min_chars: int, seed: int) -> list[tuple[Path, int]]:
    candidates: list[tuple[Path, int]] = []
    for pdf in pdf_paths:
        with fitz.open(str(pdf)) as doc:
            for i, page in enumerate(doc):
                scanned, chars, _ = is_scanned_page(page)
                if not scanned and chars >= min_chars:
                    candidates.append((pdf, i))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return sorted(candidates[:max_pages], key=lambda t: (t[0].name, t[1]))


def _degraded_image(pdf: Path, page_idx: int, name: str, seed: int) -> Image.Image:
    fn = DEGRADATIONS[name]
    d_idx = list(DEGRADATIONS).index(name)
    with fitz.open(str(pdf)) as doc:
        base = render_page_image(doc[page_idx], OCR_RENDER_DPI)
    return fn(base, seed * 1000 + page_idx * 31 + d_idx)


def collect_calibration_samples(
    pdf_paths: list[Path],
    max_reference_pages: int = 12,
    min_word_accuracy: float = 0.85,
    min_chars: int = 300,
    seed: int = 7,
    lang: str = "eng",
    ocr_timeout: float = 20.0,
    workers: Optional[int] = None,
) -> list[CalibrationSample]:
    """OCRs every (reference page x degradation) variant in parallel (Tesseract runs as a subprocess)."""
    jobs = [
        (pdf, page_idx, name)
        for pdf, page_idx in _select_reference_pages(pdf_paths, max_reference_pages, min_chars, seed)
        for name in DEGRADATIONS
    ]
    references: dict[tuple[str, int], str] = {}
    for pdf, page_idx, _ in jobs:
        key = (str(pdf), page_idx)
        if key not in references:
            with fitz.open(str(pdf)) as doc:
                references[key] = doc[page_idx].get_text("text")

    def run(job: tuple[Path, int, str]) -> CalibrationSample:
        pdf, page_idx, name = job
        img = _degraded_image(pdf, page_idx, name, seed)
        try:
            text, conf = ocr_image_tesseract(img, lang=lang, timeout=ocr_timeout)
        except RuntimeError:  # timeout: production would queue it, so it counts as an empty OCR
            text, conf = "", 0.0
        acc = word_accuracy(references[(str(pdf), page_idx)], text)
        return CalibrationSample(
            sample_id=f"{pdf.stem}__p{page_idx + 1:04d}__{name}",
            doc_id=pdf.stem,
            page=page_idx + 1,
            degradation=name,
            mean_confidence=round(conf, 2),
            garbled_char_ratio=round(compute_garbled_char_ratio(text), 4),
            alnum_token_ratio=round(compute_alnum_token_ratio(text), 4),
            word_accuracy=round(acc, 4),
            acceptable=acc >= min_word_accuracy,
            word_count=len(text.split()),
            ocr_preview=text[:240],
            image_path=str(pdf),  # replaced by the saved PNG path for review-by-eye samples
        )

    n_workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(run, jobs))


def _gate_passes(s: CalibrationSample, conf: float, garbled: float, alnum: float, words: int = 0) -> bool:
    if s.word_count == 0:
        return False  # empty OCR output never passes
    if s.word_count < words:
        return False
    return not gate_failure_reasons(
        "x", s.mean_confidence, s.garbled_char_ratio, s.alnum_token_ratio, conf, garbled, alnum
    )


def score_thresholds(
    samples: list[CalibrationSample], conf: float, garbled: float, alnum: float, words: int = 0
) -> dict[str, Any]:
    passed = [s for s in samples if _gate_passes(s, conf, garbled, alnum, words)]
    good = [s for s in samples if s.acceptable]
    tp = sum(1 for s in passed if s.acceptable)
    precision = tp / len(passed) if passed else 1.0
    recall = tp / len(good) if good else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "min_mean_confidence": conf,
        "max_garbled_char_ratio": garbled,
        "min_alnum_token_ratio": alnum,
        "min_word_count": words,
        "num_passed": len(passed),
        "num_flagged": len(samples) - len(passed),
        "bad_pages_indexed": len(passed) - tp,
        "good_pages_flagged": len(good) - tp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


CONF_GRID = [float(c) for c in range(30, 91, 5)]
GARBLED_GRID = [round(g, 3) for g in np.arange(0.02, 0.301, 0.02)]
ALNUM_GRID = [round(a, 2) for a in np.arange(0.30, 0.901, 0.05)]
WORDS_GRID = [0, 5, 10, 15, 20, 30]


def sweep_thresholds(
    samples: list[CalibrationSample],
    min_precision: float = 0.95,
    fallback_min_recall: float = 0.80,
) -> dict[str, Any]:
    """Pick gate thresholds from a grid sweep.

    Primary: maximise recall subject to precision >= `min_precision`. If no setting reaches that
    precision, keep garbage out first: maximise precision subject to recall >= `fallback_min_recall`.
    Ties go to the more lenient setting (lower thresholds), which flags fewer pages for review.
    """
    rows = [
        score_thresholds(samples, c, g, a, w)
        for c, g, a, w in itertools.product(CONF_GRID, GARBLED_GRID, ALNUM_GRID, WORDS_GRID)
    ]
    lenient = lambda r: (-r["min_mean_confidence"], r["max_garbled_char_ratio"],  # noqa: E731
                         -r["min_alnum_token_ratio"], -r["min_word_count"])
    feasible = [r for r in rows if r["precision"] >= min_precision and r["num_passed"] > 0]
    if feasible:
        best = max(feasible, key=lambda r: (r["recall"], r["f1"], *lenient(r)))
        strategy = f"max recall s.t. precision >= {min_precision}"
    else:
        pool = [r for r in rows if r["recall"] >= fallback_min_recall] or rows
        best = max(pool, key=lambda r: (r["precision"], r["recall"], *lenient(r)))
        strategy = f"precision >= {min_precision} unreachable: max precision s.t. recall >= {fallback_min_recall}"
    return {"recommended": best, "constraint_met": bool(feasible), "strategy": strategy, "grid_size": len(rows)}


def review_real_scanned_pages(
    pdf_paths: list[Path],
    thresholds: dict[str, dict[str, float]],
    lang: str = "eng",
    ocr_timeout: float = 45.0,
    workers: Optional[int] = None,
) -> list[dict[str, Any]]:
    """OCRs the corpus's genuinely scanned pages and shows each threshold set's gate decision.

    Synthetic degradations make confidence unrealistically bimodal; real scans (old forms, receipts)
    sit in the middle, so a threshold must be checked against them before it is adopted.
    """
    jobs: list[tuple[Path, int]] = []
    for pdf in pdf_paths:
        with fitz.open(str(pdf)) as doc:
            jobs.extend((pdf, i) for i, page in enumerate(doc) if is_scanned_page(page)[0])

    def run(job: tuple[Path, int]) -> dict[str, Any]:
        pdf, i = job
        with fitz.open(str(pdf)) as doc:
            img = render_page_image(doc[i], OCR_RENDER_DPI)
        try:
            text, conf = ocr_image_tesseract(img, lang=lang, timeout=ocr_timeout)
        except RuntimeError:
            text, conf = "", 0.0
        garbled, alnum = compute_garbled_char_ratio(text), compute_alnum_token_ratio(text)
        decisions = {
            name: not gate_failure_reasons(
                text, conf, garbled, alnum, th["min_mean_confidence"], th["max_garbled_char_ratio"],
                th["min_alnum_token_ratio"], int(th.get("min_word_count", 0)),
            )
            for name, th in thresholds.items()
        }
        return {
            "doc_id": pdf.stem, "page": i + 1, "mean_confidence": round(conf, 2), "word_count": len(text.split()),
            "garbled_char_ratio": round(garbled, 4), "alnum_token_ratio": round(alnum, 4),
            "passes": decisions, "ocr_preview": text[:240],
        }

    with ThreadPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 1)) as pool:
        rows = list(pool.map(run, jobs))
    return sorted(rows, key=lambda r: r["mean_confidence"])


def run_ocr_gate_calibration(
    cfg: AppConfig,
    pdf_paths: list[Path],
    max_reference_pages: int = 12,
    min_word_accuracy: float = 0.85,
    min_precision: float = 0.95,
    seed: int = 7,
    save_images: bool = True,
    report_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    samples = collect_calibration_samples(
        pdf_paths, max_reference_pages=max_reference_pages, min_word_accuracy=min_word_accuracy, seed=seed
    )
    if not samples:
        raise RuntimeError("No digital reference pages with enough text were found for calibration.")

    gate = cfg.ingest.ocr_quality_gate
    current = score_thresholds(
        samples, gate.min_mean_confidence, gate.max_garbled_char_ratio, gate.min_alnum_token_ratio, gate.min_word_count
    )
    sweep = sweep_thresholds(samples, min_precision=min_precision)
    rec = sweep["recommended"]

    def passes_rec(s: CalibrationSample) -> bool:
        return _gate_passes(
            s, rec["min_mean_confidence"], rec["max_garbled_char_ratio"], rec["min_alnum_token_ratio"], rec["min_word_count"]
        )

    # Ten flagged + ten passed for review by eye: those closest to the confidence boundary first.
    boundary = rec["min_mean_confidence"]
    flagged = sorted([s for s in samples if not passes_rec(s)], key=lambda s: abs(s.mean_confidence - boundary))[:10]
    passed = sorted([s for s in samples if passes_rec(s)], key=lambda s: abs(s.mean_confidence - boundary))[:10]

    if save_images:
        img_dir = Path(cfg.project.data_dir) / "ocr_calibration"
        img_dir.mkdir(parents=True, exist_ok=True)
        for s in flagged + passed:
            p = img_dir / f"{s.sample_id}.png"
            _degraded_image(Path(s.image_path), s.page - 1, s.degradation, seed).save(p)
            s.image_path = str(p)
    for s in samples:
        if s.image_path and not s.image_path.endswith(".png"):
            s.image_path = None

    by_degradation: dict[str, dict[str, float]] = {}
    for name in dict.fromkeys(s.degradation for s in samples):
        group = [s for s in samples if s.degradation == name]
        by_degradation[name] = {
            "mean_word_accuracy": round(float(np.mean([s.word_accuracy for s in group])), 3),
            "mean_confidence": round(float(np.mean([s.mean_confidence for s in group])), 1),
            "share_acceptable": round(sum(s.acceptable for s in group) / len(group), 3),
            "share_passed_recommended": round(sum(passes_rec(s) for s in group) / len(group), 3),
        }

    report = {
        "method": (
            "Digital pages rendered at 300 DPI, degraded, OCR'd with Tesseract; acceptable = word accuracy "
            f">= {min_word_accuracy} (bag-of-words recall vs the text layer); thresholds chosen by grid sweep."
        ),
        "num_reference_pages": len({(s.doc_id, s.page) for s in samples}),
        "num_samples": len(samples),
        "num_acceptable": sum(s.acceptable for s in samples),
        "current_thresholds": current,
        "recommended_thresholds": rec,
        "precision_constraint_met": sweep["constraint_met"],
        "selection_strategy": sweep["strategy"],
        "by_degradation": by_degradation,
        "real_scanned_pages": review_real_scanned_pages(
            pdf_paths,
            {
                "current": {k: current[k] for k in _GATE_KEYS},
                "recommended": {k: rec[k] for k in _GATE_KEYS},
            },
        ),
        "review_by_eye": {
            "flagged": [asdict(s) for s in flagged],
            "passed": [asdict(s) for s in passed],
        },
        "samples": [asdict(s) for s in samples],
    }
    out = Path(report_path) if report_path else Path(cfg.project.reports_dir) / "ocr_gate_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(out)
    return report


_GATE_KEYS = ("min_mean_confidence", "max_garbled_char_ratio", "min_alnum_token_ratio", "min_word_count")


def apply_thresholds_to_config_yaml(config_path: str | Path, thresholds: dict[str, float]) -> list[str]:
    """Rewrites only the three `ocr_quality_gate` values in config.yaml, keeping comments intact."""
    path = Path(config_path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    in_gate = False
    gate_indent = -1
    changed: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped.startswith("ocr_quality_gate:"):
            in_gate, gate_indent = True, indent
            continue
        if in_gate and stripped and not stripped.startswith("#") and indent <= gate_indent:
            in_gate = False
        if not in_gate:
            continue
        for key in _GATE_KEYS:
            m = re.match(rf"^(\s*{key}:\s*)([^\s#]+)(.*)$", line.rstrip("\n"))
            if m and key in thresholds:
                newline = "\n" if line.endswith("\n") else ""
                lines[i] = f"{m.group(1)}{thresholds[key]}{m.group(3)}{newline}"
                changed.append(key)
    path.write_text("".join(lines), encoding="utf-8")
    return changed
