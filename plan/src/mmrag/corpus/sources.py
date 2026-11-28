"""Real-corpus acquisition: direct PDF downloads + DUDE sample import (Person 1 · Story E0.2).

Network use is confined to `mmrag fetch-corpus`; ingest, smoke and tests never download anything.

- `download_file` streams a URL to disk atomically and checks PDF magic bytes.
- `import_dude_sample` turns the DUDE sample release (`DUDE_sample_pdfs.tar.gz` +
  `DUDE_sample_dataset.json`, CC BY 4.0) into page-labelled PDFs plus a gold JSONL whose `gold_doc_id`s
  match the slices the loader builds for a `real_sources` directory entry.
"""

from __future__ import annotations

import json
import logging
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import fitz

from mmrag.config import AppConfig
from mmrag.corpus.loader import safe_doc_id
from mmrag.schemas import GoldQuestion

logger = logging.getLogger(__name__)

USER_AGENT = "mmrag-corpus-fetch/0.1 (course lab; public-domain and CC-BY documents)"
DUDE_BASE_URL = "https://huggingface.co/datasets/jordyvl/DUDE_loader/resolve/main/data"
DUDE_DATASET_URL = f"{DUDE_BASE_URL}/DUDE_sample_dataset.json"
DUDE_PDFS_URL = f"{DUDE_BASE_URL}/DUDE_sample_pdfs.tar.gz"
DUDE_DOC_ID_CHARS = 12


def download_file(url: str, dest: str | Path, timeout: float = 120.0, overwrite: bool = False) -> Path:
    """Streams `url` to `dest` via a `.part` file; refuses non-PDF bodies for `.pdf` targets."""
    out = Path(dest)
    if out.exists() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as f:
            first = True
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                if first and out.suffix.lower() == ".pdf" and not chunk.startswith(b"%PDF"):
                    raise ValueError(f"{url} did not return a PDF (got {chunk[:16]!r}...).")
                first = False
                f.write(chunk)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out)
    return out


def fetch_configured_sources(cfg: AppConfig, overwrite: bool = False) -> list[dict[str, Any]]:
    """Downloads every enabled `real_sources` entry that has a `url` and a file `path`."""
    results: list[dict[str, Any]] = []
    for src in cfg.corpus.real_sources:
        if not src.enabled or not src.url:
            continue
        dest = Path(src.path)
        if dest.suffix.lower() != ".pdf":
            results.append({"doc_id": src.doc_id, "status": "FAIL", "reason": "url sources need a .pdf path"})
            continue
        try:
            existed = dest.exists()
            download_file(src.url, dest, overwrite=overwrite)
            results.append(
                {"doc_id": src.doc_id, "status": "OK", "path": str(dest), "cached": existed and not overwrite}
            )
        except Exception as exc:  # noqa: BLE001 - report every source, keep going
            results.append({"doc_id": src.doc_id, "status": "FAIL", "reason": f"{type(exc).__name__}: {exc}"})
    return results


def dude_doc_id(raw_doc_id: str, prefix: str = "dude") -> str:
    """Doc id the loader assigns to `<prefix dir>/<raw[:12]>.pdf` (`<prefix>_<raw[:12]>`)."""
    return safe_doc_id(f"{prefix}_{raw_doc_id[:DUDE_DOC_ID_CHARS]}")


def _dude_gold_pages(entry: dict[str, Any]) -> list[int]:
    """DUDE bounding boxes carry 0-indexed `page`; provenance pages are 1-indexed."""
    pages: set[int] = set()
    for answer_boxes in entry.get("answers_page_bounding_boxes") or []:
        for box in answer_boxes or []:
            if isinstance(box, dict) and isinstance(box.get("page"), int) and box["page"] >= 0:
                pages.add(box["page"] + 1)
    return sorted(pages)


def _iter_tar_pdfs(tar_path: Path):
    """Yields (doc_stem, file_obj) for regular `.pdf` members only (no links, no path traversal)."""
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".pdf"):
                continue
            fobj = tar.extractfile(member)
            if fobj is not None:
                yield Path(member.name).stem, fobj


def import_dude_sample(
    dataset_json: str | Path,
    pdf_source: str | Path,
    out_dir: str | Path,
    gold_out: str | Path,
    max_pages: Optional[int] = 120,
    max_docs: Optional[int] = None,
    doc_prefix: str = "dude",
) -> dict[str, Any]:
    """Imports page-labelled DUDE documents, shortest first, until `max_pages` is spent.

    `pdf_source` is either `DUDE_sample_pdfs.tar.gz` or a directory of extracted `<docId>.pdf` files.
    Only documents with at least one question carrying an evidence page are kept, so every imported
    page budget has gold coverage. PDFs are written to `out_dir/<docId[:12]>.pdf`; point a
    `real_sources` entry with `doc_id: <doc_prefix>` at `out_dir`.
    """
    entries = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
    by_doc: dict[str, list[tuple[dict[str, Any], list[int]]]] = defaultdict(list)
    for e in entries:
        if e.get("answer_type") == "not-answerable":
            continue
        pages = _dude_gold_pages(e)
        if pages and e.get("docId"):
            by_doc[e["docId"]].append((e, pages))

    # Read candidate PDFs (only labelled docs) into memory-light (stem -> bytes) map.
    src = Path(pdf_source)
    pdf_bytes: dict[str, bytes] = {}
    if src.is_dir():
        for pdf in src.rglob("*.pdf"):
            if pdf.stem in by_doc:
                pdf_bytes[pdf.stem] = pdf.read_bytes()
    else:
        for stem, fobj in _iter_tar_pdfs(src):
            if stem in by_doc:
                pdf_bytes[stem] = fobj.read()

    candidates: list[tuple[int, str]] = []
    for raw_id, data in pdf_bytes.items():
        try:
            with fitz.open(stream=data, filetype="pdf") as doc:
                n_pages = len(doc)
        except Exception:  # noqa: BLE001 - skip corrupt samples
            continue
        max_gold = max(p for _, pages in by_doc[raw_id] for p in pages)
        if n_pages >= 1 and max_gold <= n_pages:
            candidates.append((n_pages, raw_id))
    candidates.sort()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []
    gold_rows: list[GoldQuestion] = []
    used_pages = 0
    for n_pages, raw_id in candidates:
        if max_docs is not None and len(selected) >= max_docs:
            break
        if max_pages is not None and used_pages + n_pages > max_pages:
            continue
        (out / f"{raw_id[:DUDE_DOC_ID_CHARS]}.pdf").write_bytes(pdf_bytes[raw_id])
        doc_id = dude_doc_id(raw_id, doc_prefix)
        used_pages += n_pages
        for e, pages in by_doc[raw_id]:
            answers = [a for a in (e.get("answers") or []) if isinstance(a, str) and a.strip()]
            gold_rows.append(
                GoldQuestion(
                    query_id=f"DUDE_{e['questionId'][-12:]}",
                    query=e["question"],
                    gold_doc_id=doc_id,
                    gold_pages=pages,
                    expected_type="text",  # DUDE has no modality label; evidence is page-level
                    gold_artefact_hint=None,
                    expected_keywords=answers[:3],
                )
            )
        selected.append({"doc_id": doc_id, "pages": n_pages, "questions": len(by_doc[raw_id])})

    gold_path = Path(gold_out)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    with gold_path.open("w", encoding="utf-8") as f:
        for q in gold_rows:
            f.write(json.dumps(q.model_dump(), ensure_ascii=False) + "\n")

    return {
        "num_documents": len(selected),
        "num_pages": used_pages,
        "num_gold_questions": len(gold_rows),
        "out_dir": str(out),
        "gold_out": str(gold_path),
        "documents": selected,
    }


def fetch_dude_sample(cfg: AppConfig, overwrite: bool = False) -> tuple[Path, Path]:
    """Downloads the DUDE sample dataset JSON + PDF archive (~121 MB) into `downloads_dir`."""
    dl = Path(cfg.corpus.downloads_dir)
    ds = download_file(DUDE_DATASET_URL, dl / "DUDE_sample_dataset.json", overwrite=overwrite)
    tar = download_file(DUDE_PDFS_URL, dl / "DUDE_sample_pdfs.tar.gz", timeout=600, overwrite=overwrite)
    return ds, tar
