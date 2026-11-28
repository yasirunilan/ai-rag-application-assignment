# E7.3 Failure-mode ablations (before / after)

> Synthetic 2-manual corpus with 12 gold questions (4 per type), offline embedder and captioner. The fixes were designed on the same gold questions they are reported on (no held-out split), so absolute numbers are optimistic; the before/after deltas are the evidence.

Relevance is strict: the hit must be on a gold page **and** of the expected type.

## T2 · Layer A retrieval by type (cumulative)

| Type | n | Recall@1 | Recall@3 | Recall@5 | MRR | Precision@k | Stage |
|---|---|---|---|---|---|---|---|
| text | 4 | 0.50 | 1.00 | 1.00 | 0.75 | 0.20 | Baseline |
| text | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.20 | After fix 1 |
| text | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.20 | After fix 2 |
| text | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.20 | Final |
| figure | 4 | 0.25 | 1.00 | 1.00 | 0.58 | 0.20 | Baseline |
| figure | 4 | 0.75 | 1.00 | 1.00 | 0.88 | 0.20 | After fix 1 |
| figure | 4 | 0.75 | 1.00 | 1.00 | 0.88 | 0.20 | After fix 2 |
| figure | 4 | 0.75 | 1.00 | 1.00 | 0.88 | 0.20 | Final |
| table | 4 | 0.75 | 1.00 | 1.00 | 0.88 | 0.20 | Baseline |
| table | 4 | 0.75 | 1.00 | 1.00 | 0.88 | 0.20 | After fix 1 |
| table | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.20 | After fix 2 |
| table | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.20 | Final |

## T3 · Failure-mode log

| # | Symptom and question IDs | Root cause | Change made | Before | After |
|---|---|---|---|---|---|
| 1 | Caption-image drift causes wrong relay diagram (p.2) to outrank starter relay diagram (p.3). Failing at rank 1 before: Q_FIG_01, Q_FIG_02, Q_FIG_03, Q_FIG_04; moved: Q_FIG_01, Q_FIG_03, Q_FIG_04 | Figure search_text is the bare vision caption: two relay schematics share most callouts and nothing ties a crop to its 'Figure 9-14' label or model variant. | enrich.include_nearby_context: false -> true | figure MRR 0.42, R@1 0.00, artefact correct 0.50, faithfulness 0.60 | figure MRR 0.88, R@1 0.75, artefact correct 1.00, faithfulness 1.00 |
| 2 | Flattened table string loses column structure and dense vectors smear exact part/fault codes (AN4-12A, FAULT-E409). Failing at rank 1 before: Q_TAB_02; moved: Q_TAB_02 | Tables were flattened to one string (artefact and search text), so header/value binding was lost, and dense-only retrieval does not match exact identifiers. | extract.flatten_tables: true -> false; enrich.table_linearization_mode: flat_raw -> col_val_and_markdown; retrieve.enable_bm25_fusion: false -> true (alpha 1.0 -> 0.60) — applied and measured one at a time | table MRR 0.88, R@1 0.75, artefact correct 0.50, faithfulness 0.60 | table MRR 1.00, R@1 1.00, artefact correct 1.00, faithfulness 1.00 |

### Failure mode 2, one change at a time

| Step | Change | Table MRR | Δ MRR | Δ artefact correctness | Δ faithfulness | Improved IDs | Regressed IDs |
|---|---|---|---|---|---|---|---|
| a | extract.flatten_tables: true -> false | 0.75 | -0.12 | +0.25 | +0.20 | Q_TAB_02 | Q_TAB_01, Q_TAB_03 |
| b | enrich.table_linearization_mode: flat_raw -> col_val_and_markdown | 0.88 | +0.12 | +0.12 | +0.00 | Q_TAB_01, Q_TAB_03 | Q_TAB_02 |
| c | retrieve.enable_bm25_fusion: false -> true (alpha 1.0 -> 0.60) | 1.00 | +0.12 | +0.12 | +0.20 | Q_TAB_02 | none |

### Side effects on the other types

- FM1: improved Q_FIG_01, Q_FIG_03, Q_FIG_04; regressed none; measured move on target type: yes.
- FM2: improved Q_TAB_02; regressed none; measured move on target type: yes.

## Layer B · answer and grounding quality

| Stage | Faithfulness (5) | Answer correctness | Citation accuracy | Wrong-source rate | Artefact correctness |
|---|---|---|---|---|---|
| Baseline | 0.20 | 0.58 | 0.80 | 0.20 | 0.25 |
| After fix 1 | 0.60 | 0.92 | 1.00 | 0.00 | 0.50 |
| After fix 2 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 |
