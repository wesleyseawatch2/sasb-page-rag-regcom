# Experiment log

Copy this template for every reported run.

## Identity

- Run ID:
- Date/time and timezone:
- Git commit:
- Research question:
- Operator:
- Status: planned / running / complete / failed

## Data

- Dataset version:
- Train/dev/test scope:
- Languages:
- Retained/excluded rows and reasons:
- Query manifest SHA-256:
- PDF manifest SHA-256:
- Gold file SHA-256:
- Alignment method and unique identifier:

## Pipeline

- Candidate generators:
- Retrieval index version:
- Chunk/page representation:
- ToC configuration:
- Fusion formula and weights:
- Candidate count before VLM:
- Neighbor expansion:
- VLM reranker configuration:
- Evidence aggregation mode:
- Task 2 verifier configuration:
- Deterministic rules:
- Abstention threshold:

## Model and prompt

- Provider:
- Exact model identifier:
- API/endpoint version:
- Temperature:
- Top-p:
- Maximum output tokens:
- Image resolution/detail:
- Quantization/device for local models:
- System prompt SHA-256:
- User prompt/template SHA-256:
- Few-shot selection policy:
- Random seed:

## Execution

- Concurrency:
- Retry policy:
- Start/end time:
- Completed/failed rows:
- Cache reuse policy:
- Mean/median/p95 latency:
- Input/output tokens:
- Measured API cost:
- Hardware/software environment:

## Results

- Hit@1/5/10:
- MRR:
- Candidate recall:
- Near@1:
- Retrieval precision/recall/Macro F1:
- End-to-end Micro F1:
- End-to-end Macro F1:
- Per-class metrics:
- Per-language metrics:
- Confidence intervals:
- Paired significance tests:

## Validity and anomalies

- Empty-gold handling:
- Multiple-gold-page handling:
- Duplicate/conflicting labels:
- PDF/page-number anomalies:
- Gold-guided operations, if any:
- Provider/model drift concerns:
- Deviations from the preregistered configuration:

## Error analysis

- Sampling procedure:
- Number annotated:
- Annotators and agreement:
- Error-category counts:
- Representative cases:
- Main interpretation:

## Artifacts

- Config:
- Input manifest:
- Raw retrieval rankings:
- VLM traces:
- Predictions:
- Metrics:
- Timing/cost:
- Error annotations:

---

## Completed diagnostic run: Task 1 local retrieval (2026-08-26)

- Status: complete
- Scope: reconstructed from `test_truth.csv`; not the official Task 1 contract
- Queries/reports/pages: 621 / 63 / 7,179
- Missing PDFs: 0
- Non-empty/empty reconstructed gold: 465 / 156
- Mixed positive/negative page labels: 230 queries (normal for page retrieval)
- Same query-page label conflicts: 3 groups / 6 rows
- Retrieval: page TF-IDF + positive metric/unit rules, weighted RRF (`k=60`)
- Dense retrieval: disabled for this first reproducible local baseline
- VLM: dry-run verified; no API calls or cost incurred
- Hit@1/5/10: 0.151 / 0.383 / 0.514
- Near@1: 0.194
- MRR: 0.254
- Runtime artifacts: `runs/task1-full/` (ignored because inputs are restricted)
- Validity note: diagnostic only; empty-gold handling and official page-set
  scoring must be confirmed before making an official Task 1 claim

## Completed pilot: cached VLM reranking (2026-08-31)

- Status: complete
- Scope: 4 non-empty-gold reconstructed queries per language, 24 total
- Model/configuration: `gpt-5.4-nano`, Responses API, 10 candidates, 96-DPI
  images, low image detail, maximum 900 output tokens
- Cache: 6 earlier successful calls reused; 24 complete traces in the pilot
- Token usage: 344,193 input + 7,199 output tokens
- Estimated API cost: approximately USD 0.078 using the listed nano text-token
  rates; image input is included in the reported input-token usage
- Candidate recall@10: Chinese 1.00, English 0.75, French 0.75, Japanese
  0.25, Korean 0.00, Thai 0.00
- RRF baseline (same 24): Hit@1 0.083, Hit@5 0.292, Hit@10 0.458, MRR 0.200
- VLM reranking (same 24): Hit@1 0.208, Hit@5 0.375, Hit@10 0.458, MRR 0.298
- Interpretation: VLM improved Top-1 and MRR on this small sample, but cannot
  recover pages absent from the local candidate set. Results are preliminary
  and must not be reported as official Task 1 performance.

## Retrieval optimization diagnostic (2026-08-31)

- Same 621-query reconstructed set as the original local baseline
- Query grouping: `report_metric` (answer-value/unit variants merged)
- Indexing: reusable per-report word TF-IDF; original query text retained
- Additional lane: character TF-IDF (`char` 2--5 grams), evaluated as a
  low-weight RRF lane and as a lane-union candidate pool
- Cached-word + rules baseline: Hit@1 0.151, Hit@5 0.389, Hit@10 0.520,
  MRR 0.263 (original baseline: 0.151 / 0.383 / 0.514 / 0.254)
- Top-30 fused candidate recall: 0.662
- Top-30 union of word/character/rule lane candidates: 0.718
- Dense MiniLM: implemented with one model load, 1,500-character page inputs,
  and `.npy` cache; not used for the reported numbers because CPU execution
  was too slow for an efficient local run
- Interpretation: indexing and query preservation give a small, reproducible
  gain; candidate-pool union is more promising for VLM recall than equal-weight
  character fusion. These remain diagnostic, not official Task 1 results.

## Candidate-pool VLM pilot (2026-08-31)

- Scope: one non-empty-gold query per language, 6 total
- Candidate source: interleaved word/character/rule lane union, 30 images/query
- Candidate recall: 5/6 (Thai gold page remained outside the pool)
- VLM: `gpt-5.4-nano`, low detail, 96 DPI, 1,500 output-token retry for one
  truncated Thai response
- Hit@1/5/10: 0.167 / 0.333 / 0.500; MRR 0.261
- Token usage: 253,264 input + 4,251 output; estimated cost USD 0.056
- Comparison: the earlier 10-image pilot reached 0.333 / 0.333 / 0.500 and
  MRR 0.357 on the same six queries
- Interpretation: increasing candidate recall to 30 pages did not improve the
  small VLM's ranking. Keep 30-page pools opt-in; use a stronger second-stage
  model, smaller batches, or local prefiltering before any larger API run.
