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
- Conflicting grouped labels: 230 queries
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
