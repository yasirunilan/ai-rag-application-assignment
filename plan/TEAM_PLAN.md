# 5-Person Engineering Implementation Plan & Agent Playbook
**Course 3 — Agentic Systems & Retrieval · Day 2 · Lab A ("Rich Answers, Not Just Text")**

---

## 1. Locked Architecture & Team Decisions

| Decision Area | Locked Choice | Why (3-Pass Validated) |
| :--- | :--- | :--- |
| **Core PDF Parser (`E0.1`, `E1.1`)** | `PyMuPDF (fitz)` + `pdfplumber` light core, optional `Docling` adapter | Installs in `<10s` with zero PyTorch/CUDA bloat (Story E0.1 compliance) while preserving figure bboxes and 2D table cells. |
| **Corpus & Gold Eval (`E0.2`)** | `DUDE` eval slice (`dude_hydraulic_manual.pdf`) + `FAA AMT Handbook` slice (`amt_handbook.pdf`) | Covers wiring schematics, exploded actuator diagrams, torque tables, fault-code tables, prose, and degraded scans with page-level gold labels. |
| **Scanned OCR Gate (`E1.2`)** | `Tesseract` + 2-metric Quality Gate (`mean_confidence >= 60%`, `garbled_char_ratio <= 15%`) | Blocks degraded scans from silently polluting dense vector hubs; routes failures to `data/review_queue.jsonl`. |
| **Vision Captioning (`E3.1`)** | `OpenRouter :free` (`Qwen2.5-VL` fallback chain, `:free` enforced in code) + SHA-256 git-shared disk cache + offline fallback | Zero billing wall (`BillingGuardError` blocks paid models), shared cache avoids 5x team rate-limiting, offline fallback makes `mmrag smoke` work with zero network. |
| **Embeddings & Mixed Index (`E4`)** | Local `fastembed` (`BAAI/bge-small-en-v1.5`) + zero-network `DeterministicSemanticEmbedder` + `FAISS` / `NumPy` + `units.jsonl` | One unified index over `text`, `figure`, and `table` units; `units.jsonl` persists full provenance (`doc -> page -> id`) alongside every vector row. |
| **Hybrid Retrieval (`E5`)** | Dense Vector + Technical-Code `BM25` (`rank_bm25`) min-max fusion ($\alpha = 0.60$) | Prevents dense vectors from smearing exact part numbers (`AN4-12A`, `NAS1304-15`) and fault codes (`FAULT-E409`). |
| **Answering (`E6`)** | Fail-Closed Provenance Gate + Modality Router | Drops any unit without `doc_id` and `page >= 1`; returns actual `.png` path for figures and structured `TableArtefact` for tables. |
| **E7.3 Failure Modes Fixed** | **FM1**: *Right text, wrong figure* (caption–image drift)<br>**FM2**: *Table answers are mush & exact part codes missed* | Automated before/after ablation runner (`mmrag ablate`) proves measurable gains on `Strict Recall@k`, `Strict MRR`, and `Faithfulness Pass Rate`. |

---

## 2. 5-Engineer Work Split, Branches & File Ownership

Every engineer works on an isolated feature branch and owns dedicated modules and test files so **zero merge conflicts** occur during parallel development.

### Person 1 — Platform, Corpus & Scanned OCR Gate Lead (`E0.1`, `E0.2`, `E1.2`)
- **Feature Branch**: `feat/p1-env-corpus-ocr`
- **Owned Files**:
  - `config.yaml`, `setup.ps1`, `setup.sh`, `src/mmrag/config.py`, `src/mmrag/cli.py`
  - `src/mmrag/corpus/synthetic.py`, `src/mmrag/corpus/loader.py`
  - `src/mmrag/ingest/ocr_gate.py`
  - `tests/test_p1_env_and_ocr.py`
- **Acceptance Command**: `pytest tests/test_p1_env_and_ocr.py`
- **Brief to Hand Person 1's Coding Agent**:
  > "You own Epics E0.1, E0.2, and E1.2 in `src/mmrag/corpus/` and `src/mmrag/ingest/ocr_gate.py`. Expand the corpus loader to ingest additional real FAA AMT Handbook pages or DUDE PDFs up to `max_pages=250`, tune the Tesseract OCR quality gate thresholds (`min_mean_confidence`, `max_garbled_char_ratio`), and verify `pytest tests/test_p1_env_and_ocr.py` and `python -m mmrag.cli smoke` pass with no API key."

---

### Person 2 — Structural Parser & Artefact Extraction Lead (`E1.1`, `E2.1`, `E2.2`, `E2.3`)
- **Feature Branch**: `feat/p2-parse-extract-provenance`
- **Owned Files**:
  - `src/mmrag/ingest/router.py`
  - `src/mmrag/extract/figures.py`
  - `src/mmrag/extract/tables.py`
  - `src/mmrag/extract/spotcheck.py`
  - `tests/test_p2_parse_extract.py`
- **Acceptance Command**: `pytest tests/test_p2_parse_extract.py`
- **Brief to Hand Person 2's Coding Agent**:
  > "You own Epics E1.1, E2.1, E2.2, and E2.3 in `src/mmrag/ingest/router.py` and `src/mmrag/extract/`. Ensure every figure crop exports at 200 DPI with nearby figure captions (`Figure X-Y`), section headings, and model variants attached to `Provenance`, filter out decorative rules/icons below `min_figure_area_pt2`, preserve structured rows/cols in `TableArtefact` with `search_text=''`, and verify `python -m mmrag.cli spotcheck` and `pytest tests/test_p2_parse_extract.py` pass."

---

### Person 3 — Multimodal Enrichment & Caching Lead (`E3.1`, `E3.2`)
- **Feature Branch**: `feat/p3-enrich-caption-linearize`
- **Owned Files**:
  - `src/mmrag/enrich/cache.py`
  - `src/mmrag/enrich/captioner.py`
  - `src/mmrag/enrich/table_linearizer.py`
  - `tests/test_p3_enrich.py`
- **Acceptance Command**: `pytest tests/test_p3_enrich.py`
- **Brief to Hand Person 3's Coding Agent**:
  > "You own Epic E3 (`E3.1` Figure Vision Captioning & `E3.2` Table Linearization) in `src/mmrag/enrich/`. Enforce `:free`-only OpenRouter models in code, cache all vision captions on disk under `data/cache/captions/` keyed by SHA-256, inject nearby figure captions/headings/model variants to prevent Failure Mode 1 ('Right text, wrong figure'), and linearize `TableArtefact` rows into `Header = Value` strings without mutating `unit.artefact`. Verify `pytest tests/test_p3_enrich.py` passes."

---

### Person 4 — Embeddings, Mixed Index & Hybrid Retrieval Lead (`E4.1`, `E4.2`, `E5.1`, `E5.2`)
- **Feature Branch**: `feat/p4-index-hybrid-retrieve`
- **Owned Files**:
  - `src/mmrag/index/chunker.py`
  - `src/mmrag/index/embedder.py`
  - `src/mmrag/index/store.py`
  - `src/mmrag/retrieve/retriever.py`
  - `tests/test_p4_index_retrieve.py`
- **Acceptance Command**: `pytest tests/test_p4_index_retrieve.py`
- **Brief to Hand Person 4's Coding Agent**:
  > "You own Epics E4 (`E4.1`, `E4.2`) and E5 (`E5.1`, `E5.2`) in `src/mmrag/index/` and `src/mmrag/retrieve/`. Keep figure and table units atomic (never split), persist vectors + `units.jsonl` provenance metadata so reload preserves citations, and fuse dense vector scores with technical-code BM25 scores (`alpha=0.60`) so exact part numbers (`AN4-12A`, `NAS1304-15`) and fault codes (`FAULT-E409`) rank at the top. Verify `pytest tests/test_p4_index_retrieve.py` passes."

---

### Person 5 — Fail-Closed Answering, Per-Type Eval & Failure Mode Engineering Lead (`E6.1`, `E6.2`, `E7.1`, `E7.2`, `E7.3`)
- **Feature Branch**: `feat/p5-answer-eval-ablations`
- **Owned Files**:
  - `src/mmrag/answer/responder.py`
  - `src/mmrag/eval/metrics.py`
  - `src/mmrag/eval/faithfulness.py`
  - `src/mmrag/eval/ablations.py`
  - `tests/test_p5_answer_eval.py`
- **Acceptance Command**: `pytest tests/test_p5_answer_eval.py`
- **Brief to Hand Person 5's Coding Agent**:
  > "You own Epics E6 (`E6.1`, `E6.2`) and E7 (`E7.1`, `E7.2`, `E7.3`) in `src/mmrag/answer/` and `src/mmrag/eval/`. Enforce fail-closed abstention whenever provenance lacks a valid page or artefact, return the real `.png` image path or structured `TableArtefact` rows with page citations, compute per-type `Recall@k` and `MRR` (both Page-Level and Strict Modality+Page), run the 5-answer faithfulness check, and run before/after benchmarks on Failure Mode 1 and Failure Mode 2 (`python -m mmrag.cli ablate`). Verify `pytest tests/test_p5_answer_eval.py` passes."
