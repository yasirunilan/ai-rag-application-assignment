# Provenance, Data-Handling Note & Team Reflection (Stories E9.2 & E9.3)

## 1. Provenance & Data-Handling Note (Story E9.2)

- **End-to-End Provenance Chain (`doc_id → page → unit_id`)**:
  1. **Ingest & Extract (`E1`, `E2`)**: Every routed text block, exported figure crop (`.png`), and structured table (`TableArtefact`) is stamped with `doc_id`, 1-indexed `page`, deterministic `unit_id` (`{doc_id}__p{page:04d}__{type}_{seq:03d}`), `bbox`, and `source_path`.
  2. **Enrich & Index (`E3`, `E4`)**: Unenrichable units (empty/generic captions or missing provenance) are rejected to `data/rejected_enrichment_units.jsonl` (`E3.3`). `MixedVectorStore` enforces write-time provenance (`E4.2` → `data/index/rejected_units.jsonl`), records `embedding_model_id` in `index_meta.json`, and persists payloads in `units.jsonl`.
  3. **Retrieve, Answer & UI (`E5`, `E6`, `E7`)**: `build_answer_from_hits` drops any hit failing `has_valid_provenance` (fail closed), returns the real `.png` or exact `TableArtefact` rows with inline `[doc_id p.N]` citation, and highlights the cited unit in the always-visible UI `SourcesPanel`. `/artefact/{unit_id}` blocks path traversal outside `data/`.
- **OCR-Failed Pages (`E1.2`)**: Scanned pages (`text_chars <= 50` + `image_coverage >= 0.60`, or full-page scans with garbled hidden OCR layers) are scored on `mean_confidence >= 60%`, `garbled_char_ratio <= 15%`, and `alnum_token_ratio >= 50%`. Failures are routed to `data/review_queue.jsonl` for human review and never indexed.
- **Data Residency & Redaction for Confidential Manuals**: Free OpenRouter endpoints are restricted to public datasets (`amt_handbook`, `DUDE`, `Enginuity Bench`). For ITAR/proprietary fleet manuals in production, tail numbers, serials, and customer identifiers must be redacted pre-ingestion, and captioning + embeddings must run on self-hosted/VPC-isolated models (`local_vision` + `local_bge` ONNX) so zero bytes leave tenant boundaries.

---

## 2. Team Reflection (Story E9.3)

- **Stage 0 Prediction vs. Measured Result**: Before running metrics, we predicted `figure` queries would retrieve worst (`stage0_prediction_worst_type = "figure"`) because raw diagrams have zero native lexical tokens and visually similar schematics collide. Untuned baseline confirmed this (`figure_strict_mrr = 0.583` vs `table_strict_mrr = 0.875` and `text_strict_mrr = 1.000`; `faithfulness = 0.20`).
- **Changes That Moved Metrics Most**:
  - **FM1 (Caption–Image Drift)**: Anchoring vision captions with nearby PDF figure labels, section headings, and aircraft model variants lifted `figure_strict_mrr` from `0.417 → 0.875` (`+0.458`) and faithfulness from `0.60 → 1.00` (`Q_FIG_01`, `Q_FIG_04` moved to rank 1).
  - **FM2 (Table Mush & Exact Codes)**: Preserving `TableArtefact` + `Header = Value` row linearization + code-aware BM25 fusion (`alpha = 0.60`) lifted `table_strict_mrr` from `0.875 → 1.000` (`Q_TAB_02` `NAS1304-15` rescued to rank 1) and faithfulness from `0.60 → 1.00`.
- **Latency & Cost Reality**: Selective figure-only captioning + SHA-256 disk caching cuts repeat ingestion vision calls to `0`, and batch-16 embedding cuts embedding requests by `~90%`, keeping queries well inside OpenRouter's `20 req/min` cap.
- **Grounding Ladder Position**: Pure **Retrieve + Cited Artefact Return** (rungs 1–2: prompt + retrieval, no fine-tuning/distillation) — ideal because maintenance specs change per service bulletin and engineers need verifiable primary diagrams/tables, not memorized weights.
- **Honest Limitation a Polished Demo Hides**: Cross-page multi-hop procedures (e.g., a wiring schematic on p.3 whose pinout torque or fault tree lives in a separate table 40 pages away without shared part codes) still return only the single top sourced artefact (`Sub-Track 1B` multi-hop scope).
