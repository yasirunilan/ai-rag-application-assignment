# Rich Answers, Not Just Text — Multimodal Retrieval over Technical Manuals
**Course 3 — Agentic Systems and Retrieval · Day 2 · Lab A (Sub-Track 1A: *The Grounded Agent*)**

> **Guiding Principle — Retrieve-by-text, answer-with-artefact**: Index something searchable (a vision caption, a linearized table, or prose/OCR text), but hand back the original figure image crop or structured table rows, linked end-to-end by `doc_id → page → unit_id` provenance.

---

## Quickstart (One-Command Free Setup · Story E0.1)

### Windows (PowerShell)
```powershell
.\setup.ps1
```

### Linux / macOS / Git Bash
```bash
bash setup.sh
```

### Direct CLI Commands
```powershell
# 1. Run end-to-end zero-network smoke test (ingest -> spotcheck -> answer -> eval -> ablate)
python -m mmrag.cli smoke

# 2. Run all 5 engineers' acceptance test suites
pytest

# 3. Answer a specific engineering query with real artefact + citation
python -m mmrag.cli answer --offline -q "Show the engine starter relay control wiring diagram for Model 737-800"
python -m mmrag.cli answer --offline -q "What is the dry and lubed torque spec table for bolt part number AN4-12A?"

# 4. Run Story E2.3 5-unit provenance spot-check
python -m mmrag.cli spotcheck

# 5. Run Story E7.3 Before/After Failure Mode benchmarks
python -m mmrag.cli ablate --offline
```

---

## Definition of Done Checklist (§8)

- [x] **Corpus loaded** (DUDE eval slice + FAA AMT Handbook slice with page-level gold evidence labels).
- [x] **Content routed; scanned pages OCR'd behind a quality gate** (`mean_confidence >= 60%`, `garbled_char_ratio <= 15%`; failures routed to `data/review_queue.jsonl`).
- [x] **Figures & tables extracted with `doc → page → id` provenance** (cropped `.png` figures with icon/rule geometry filter; structured `TableArtefact` rows/cols).
- [x] **Figures captioned / tables linearized** (`OpenRouter :free` with `:free` billing guard, SHA-256 disk cache, and offline fallback; `col=val` table linearization).
- [x] **Mixed index built with provenance in metadata** (local embeddings + FAISS / NumPy + `units.jsonl` persistence).
- [x] **Retrieval returns units with citations** (`doc p.212` + hybrid BM25 lexical fusion for exact part/fault identifiers).
- [x] **Answer step returns the real figure/table + page, not a paraphrase** (with fail-closed abstention on missing provenance).
- [x] **Retrieval measured by type; baseline recorded; 2 failure modes fixed with before/after metrics** (FM1: *Right text, wrong figure*; FM2: *Table answers are mush & exact part codes missed*).

See [`TEAM_PLAN.md`](TEAM_PLAN.md) for the 5-person division of labor, branch ownership, and per-person coding agent briefs.
