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

## Full VLM comparison/error analysis (2026-09-01)

- Inputs: the metadata-enriched fused rankings and the completed Chinese and
  non-Chinese VLM traces, aligned by `sample_id`
- Overall reconstructed set: 490 query groups; 369 non-empty gold groups and
  121 empty groups
- Same non-empty groups, fused baseline vs VLM: Hit@1 0.192 -> 0.268,
  Hit@5 0.396 -> 0.458, Hit@10 0.507 -> 0.507, MRR 0.291 -> 0.347
- Top-1 transition categories: both correct 57, VLM improved 42, VLM
  degraded 14, neither correct 74; 182 groups had no gold page in the fused
  Top-10 candidate set
- Candidate recall@10 by language: Chinese 0.471, English 0.906, French
  0.800, Japanese 0.200, Korean 0.608, Thai 0.109
- `no_relevant_page=true` diagnostic rate: empty groups 45/121 (0.372);
  non-empty groups 142/369 (0.385). This is not treated as an official score
  because the Task 1 query-level no-answer contract is not confirmed.
- Reproducible analyzer: `src/task1/analyze_vlm.py`; it emits per-language
  metrics, transition categories, candidate-recall flags, and per-query error
  records without making API calls.

## End-to-end VLM label diagnostic (2026-09-01)

- Scope: all 490 reconstructed `report_metric` query groups, using the
  Top-1 page from the completed Chinese and non-Chinese VLM reranking runs
- Second-stage model: `gpt-5.4-nano-2026-03-17` through the Responses API;
  one selected-page image plus extracted page text per query, 96 DPI, low
  image detail, maximum 300 output tokens
- Completion: 490/490 traces completed with no API errors; 1,284,477 input
  and 72,735 output tokens; estimated cost USD 0.348 using $0.20/M input and
  $1.25/M output rates; mean latency 3.67 seconds and p95 11.99 seconds
- Predicted labels: `no` 301, `yes but not complete` 143, `yes` 46
- The selected-page retrieval Hit@1 remains 0.268 overall (Chinese 0.200,
  English 0.563, French 0.320, Japanese 0.127, Korean 0.392, Thai 0.047),
  because the second-stage verifier cannot recover a page missing from the
  candidate set
- Only 74 selected pages have an unambiguous exact page-level label after
  joining the reconstructed groups to the available truth rows; their label
  accuracy is 0.459. This is a limited diagnostic, not an official Task 1
  classification score, because the query-level target label and evaluator
  are not confirmed and many grouped rows have conflicting page labels.
- Reproducible entry points: `src/task1/end_to_end_vlm.py` performs the
  cached second-stage calls; `src/task1/aggregate_e2e.py` merges language runs
  and recomputes token, cost, latency, and diagnostic metrics.

## Dense retrieval and full VLM reranking (2026-09-01)

- Scope: all 490 reconstructed `report_metric` query groups; 369 groups have
  positive gold pages and 121 groups have no positive page annotations.
- Retrieval: multilingual `paraphrase-multilingual-MiniLM-L12-v2` page
  embeddings (1,500-character inputs), word/character TF-IDF, metric/unit
  rules, and reciprocal-rank fusion. Embeddings are cached with short stable
  filenames so the pipeline works in deeply nested Windows workspaces.
- Dense-fused retrieval: Hit@1 0.220, Hit@5 0.493, Hit@10/candidate recall
  0.631, Near@1 0.293, and MRR 0.350. Relative to the previous fused
  diagnostic (0.192 / 0.396 / 0.507 / 0.291), the largest gain is candidate
  recall, which is the prerequisite for any VLM reranker.
- VLM: `gpt-5.4-nano-2026-03-17` through the Responses API; ten 96-DPI,
  low-detail page images per query; 900 output-token cap. The run used four
  concurrent requests initially and single-worker retries for rate-limited
  requests. All 490/490 traces completed successfully after retries.
- Dense + VLM ranking: Hit@1 0.328, Hit@5 0.583, Hit@10 0.631, Near@1
  0.423, and MRR 0.431. Compared with dense-fused retrieval, this is +10.8
  percentage points at Hit@1, +8.9 points at Hit@5, and +0.081 MRR; Hit@10
  is unchanged because reranking cannot recover pages absent from the top-10
  candidate set.
- Per-language VLM Hit@1 / MRR: Chinese 0.259 / 0.375, English 0.547 /
  0.647, French 0.400 / 0.554, Japanese 0.309 / 0.387, Korean 0.392 /
  0.504, and Thai 0.109 / 0.173. Thai remains the main candidate-recall
  bottleneck (0.313), followed by Chinese (0.576) and Japanese (0.545).
- API accounting: 6,758,052 input tokens and 143,304 output tokens;
  estimated cost USD 1.531 using the configured nano rates. Mean latency was
  8.13 seconds, p95 22.55 seconds, and maximum 99.09 seconds.
- Reproducible artifacts: `runs/task1-dense-optimized/retrieval.jsonl`,
  `runs/task1-dense-optimized/vlm-full.jsonl`,
  `runs/task1-dense-optimized/vlm-full-analysis.json`, and
  `runs/task1-dense-optimized/metrics-vlm.json` (generated files are ignored
  by Git). These remain reconstructed diagnostics, not official Task 1 scores,
  because the released query-level gold/evaluator contract is unavailable.

## Neighbor-page candidate ablation (2026-09-01)

- Candidate construction: retain the fused top-10 pages, then append pages
  within a +/-2 PDF-page window around those hits before the lane-union pool.
  This raises candidate-pool recall from 0.631 at top-10 to 0.710 at top-20
  (0.789 at top-30), without changing the fused retrieval ranking.
- VLM run: all 490 groups, 20 candidate images per query, `gpt-5.4-nano`,
  96-DPI low-detail images, 900 output tokens. Completion was 490/490 after
  one rate-limit retry; 13,178,046 input and 275,396 output tokens cost an
  estimated USD 2.980.
- Dense + neighbor-pool VLM ranking: Hit@1 0.314, Hit@5 0.561, Hit@10
  0.648, Near@1 0.398, and MRR 0.423. Relative to the frozen 10-candidate
  run (0.328 / 0.583 / 0.631 / 0.431), the larger pool improves coverage at
  Hit@10 but slightly harms Top-1 and MRR, suggesting that nano becomes less
  decisive when shown 20 visually similar pages.
- Decision: keep the 10-candidate dense + VLM configuration as the primary
  diagnostic; retain this 20-candidate neighbor expansion as an ablation and
  as a possible input to a stronger second-stage model. The results are still
  diagnostic rather than official Subtask 1 scores.

## Neighbor-expanded top-20 VLM reranking, task1-neighbor2 (2026-09-01, complete)

- Status: **complete** -- `runs/task1-neighbor2/vlm-full20-cache.jsonl` reached
  490/490 at 12:42 (Asia/Taipei) with 0 errors. This entry started as an
  in-progress snapshot (183/490) and was updated in place once the run
  finished; it is still not an *official* Task 1/Subtask 1 result (see the
  claim-boundary note repeated throughout this log).
- Configuration: same 490 reconstructed `report_metric` query groups (369
  non-empty gold) as the dense-optimized run above. Retrieval fuses TF-IDF,
  character TF-IDF, metric/unit rules, and dense
  `paraphrase-multilingual-MiniLM-L12-v2` embeddings via RRF (`k=60`, weights
  tfidf=1.0, char\_tfidf=0.02, rules=0.5, dense=1.0), then expands the fused
  top-10 anchors with a `neighbor_window=2` page window (`neighbor_anchor_k=10`)
  before truncating to a **top-20** VLM candidate pool -- twice the candidate
  depth of the dense-optimized run. VLM: `gpt-5.4-nano-2026-03-17`, one request
  per query.
- Final numbers, computed independently by the new API-free
  `src/task1/aggregate_neighbor2.py` script, match the "Neighbor-page candidate
  ablation" entry immediately above **exactly** (same Hit@1/5/10, Near@1, MRR,
  token counts, and cost) -- a useful cross-check between the two analyses.
  This entry adds the per-language breakdown that entry does not include, plus
  a candidate-recall figure computed against the VLM's actual pool rather than
  only the fused top-10.
- Overall (369 non-empty): fused baseline Hit@1 0.220, Hit@5 0.493, Hit@10
  0.631, Near@1 0.293, MRR 0.350 -> VLM (top-20 pool) Hit@1 0.314, Hit@5 0.561,
  Hit@10 0.648, Near@1 0.398, MRR 0.423. Candidate recall at the fused top-10
  is 0.631; recall of the VLM's actual (neighbor-expanded, top-20) pool is
  0.710 -- this is why VLM Hit@10 (0.648) can exceed the fused-top-10 recall
  figure: the VLM sees more candidates than that slice measures.
- Top-1 transition categories (490 rows): both correct 63, VLM improved 48,
  VLM degraded 18, neither correct 104, candidate-recall miss 136, empty gold
  121.
- Per-language, fused Hit@1 -> VLM Hit@1 (MRR fused -> VLM; candidate recall
  at fused-10 -> at VLM pool): Chinese 0.153 -> 0.271 (0.292 -> 0.369; 0.576 ->
  0.635); English 0.453 -> 0.547 (0.591 -> 0.648; 0.891 -> 0.922); French
  0.320 -> 0.300, i.e. **VLM reranking degraded French Top-1** (0.484 -> 0.467;
  0.840 -> 0.900); Japanese 0.182 -> 0.309 (0.293 -> 0.413; 0.545 -> 0.727);
  Korean 0.216 -> 0.392 (0.389 -> 0.508; 0.686 -> 0.784); Thai 0.031 -> 0.094,
  still the weakest language by a wide margin (0.101 -> 0.177; 0.313 -> 0.375).
  The French regression is a genuine finding, not noise from a small sample
  (50 non-empty-gold queries) -- it echoes Cierpa's finding that added context
  (there, few-shot examples; here, a larger candidate pool) does not uniformly
  help every language, and is worth a closer qualitative look via the sample
  below.
- API accounting: 13,178,046 input tokens (315,648 cached) and 275,396 output
  tokens across 490 complete calls, 0 errors; estimated cost USD 2.980 using
  the same $0.20/M input / $1.25/M output nano-rate convention as the entries
  above (cached-token discount not modeled); mean latency 10.393 seconds,
  median 7.506 seconds, p95 33.181 seconds.
- Reproducible, API-free entry point: `py src/task1/aggregate_neighbor2.py`.
  Reads only `runs/task1-neighbor2/retrieval.jsonl` and, preferentially,
  `runs/task1-neighbor2/vlm-full20.jsonl` (falling back to
  `vlm-full20-cache.jsonl`), makes no API calls, and never writes into `runs/`;
  output goes to `artifacts/metrics/task1_neighbor2_summary.json` and
  `artifacts/metrics/task1_neighbor2_per_query.csv`.
- A stratified, human-review-only error sample was regenerated from the
  complete run: `py src/task1/build_error_review_sample.py` writes
  `docs/TASK1_ERROR_REVIEW_SAMPLE.csv` (65 rows, stratified by language and by
  a mechanically derived Hit@1 transition category; now covers all six
  languages). The script never fills in the qualitative error-taxonomy columns
  from Section 13 below -- those are left blank for a human annotator.
- Read alongside the "Decision" line in the "Neighbor-page candidate ablation"
  entry above: the larger candidate pool measurably helps recall and several
  languages' Top-1/MRR, but is not a strict improvement (French regresses, and
  the pooled MRR/Near@1/Hit@5 are each slightly below the 10-candidate run's).
  Both configurations remain diagnostic, not official Task 1 scores.

## Pre-specified local BGE-M3 ablation (2026-09-01, pending)

- Scope: a no-API retrieval ablation on the same 490 reconstructed query groups
  (369 non-empty gold groups). The 490-group diagnostic set is frozen before
  inspecting BGE outcomes; it is not an official Task 1 evaluation.
- Retrieval: recompute word/character TF-IDF and metric/unit rules, add
  multilingual `BAAI/bge-m3`, and fuse the lanes with RRF. Rerank the fused
  Top-50 text candidates with `BAAI/bge-reranker-v2-m3`.
- Prespecified settings: 1 GPU worker on a single RTX-class GPU, page text
  limit 1,500 characters, BGE max length 512 for the first reproducible run,
  reranker max length 512, and candidate-k 50. Any long-context or sparse/
  multi-vector variants must be reported as separate ablations rather than
  selected after viewing held-out results.
- Entry point: `py src/task1/local_bge_ablation.py`; GPU setup and smoke test:
  `docs/TASK1_BGE_GPU_RUNBOOK.md`. Outputs will be written to
  `artifacts/metrics/task1_bge_ablation.json` and the ignored `runs/` directory.
- Status: implementation complete; the previous CPU pilot was stopped before
  completion and its incomplete metric file was removed. Final BGE numbers,
  elapsed time, and per-language analysis remain to be filled after the GPU run.
