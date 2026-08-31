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

## Participant JSON source audit and pilot (2026-09-01)

- Status: complete
- Source: the six downloaded `*_test.json` files in the participant Test Set;
  their SHA-256 hashes match the local copies under `data/test/`
- Page-level records: 982 total (Chinese 189, English 201, French 192,
  Japanese 130, Korean 149, Thai 121)
- Labels present: `yes`, `yes but not complete`, and `no`; both positive labels
  are treated as relevant pages by the pipeline
- Report-metric reconstruction: 490 queries / 63 reports / 7,179 pages;
  369 non-empty and 121 empty gold query groups; missing PDFs 0
- Non-Chinese report-metric subset: 363 query groups, 284 non-empty and 79
  empty groups
- Interpretation: an empty group is not automatically a missing annotation.
  The participant README defines `no` as a sampled negative page instance.
  Direct JSON input lacks some metadata-enriched SASB fields, so its retrieval
  scores are kept separate from the CSV diagnostic.
- VLM pilot: one query per non-Chinese language, 5 calls, 10 candidates,
  `gpt-5.4-nano`, 96-DPI low-detail images; 3 non-empty cases evaluated with
  Hit@1 0.333, Hit@5 1.000, Hit@10 1.000, MRR 0.583. This is a feasibility
  check, not an official Task 1 score.

## Non-Chinese VLM reranking run (2026-09-01)

- Status: complete
- Scope: all 363 non-Chinese `report_metric` query groups (English 77,
  French 64, Japanese 69, Korean 59, Thai 94); 284 groups had at least one
  positive page and 79 groups contained only negative page instances
- Retrieval input: metadata-enriched CSV diagnostic rankings, Top-10 fused
  candidates per query
- VLM: `gpt-5.4-nano-2026-03-17`, Responses API, 96-DPI images, low detail,
  maximum 900 output tokens, one request per query, 90-second timeout
- API completion: 363/363 complete traces; no error traces
- Token usage: 4,748,861 input + 109,510 output tokens
- Estimated API cost: approximately USD 1.09 using the configured nano rates
- Latency: mean 10.36 seconds, p95 36.78 seconds, maximum 97.03 seconds
- Fused retrieval baseline on the same 284 non-empty groups: Hit@1 0.211,
  Hit@5 0.412, Hit@10 0.518, MRR 0.306
- VLM reranking on the same groups: Hit@1 0.289, Hit@5 0.468, Hit@10 0.518,
  Near@1 0.352, MRR 0.362
- Absolute change versus fused baseline: +7.7 percentage points Hit@1,
  +5.6 points Hit@5, unchanged Hit@10, and +0.056 MRR
- Per-language VLM Hit@1/5/10 and MRR: English 0.562/0.797/0.906/0.663;
  French 0.320/0.720/0.800/0.485; Japanese 0.127/0.182/0.200/0.143;
  Korean 0.392/0.569/0.608/0.464; Thai 0.047/0.109/0.109/0.073
- Empty-gold diagnostic: VLM returned `no_relevant_page=true` for 30/79
  empty groups (0.380) and for 111/284 non-empty groups (0.391). Because the
  official Task 1 meaning of these reconstructed groups is not confirmed, this
  is reported as a diagnostic rather than a formal classification score.
- Interpretation: VLM improves ordering within the retrieved Top-10 set,
  especially Top-1 and MRR, but cannot recover pages absent from that set;
  Japanese and Thai remain the main retrieval bottleneck.
