# Task 1 implementation

`task1_pipeline.py` implements the first runnable Task 1 baseline:

- reconstruct report-level queries and positive page sets from page-level truth;
- index complete PDF reports with PyMuPDF;
- run TF-IDF, metric/unit rules, optional dense retrieval, and RRF;
- evaluate Hit@1/5/10, Near@1, and MRR;
- prepare or execute cached Top-10 OpenAI VLM reranking.

The reconstructed query set is an analysis dataset, not a claim that the
official Task 1 data contract has been recovered exactly.

## Smoke run

```powershell
py src/task1/task1_pipeline.py prepare `
  --truth ../data/datasets/all/test_truth.csv `
  --pdf-root "../Training Set/PDF" `
  --queries-out runs/task1-smoke/queries.jsonl `
  --pages-out runs/task1-smoke/pages.jsonl `
  --audit-out runs/task1-smoke/audit.json `
  --max-queries 5

py src/task1/task1_pipeline.py retrieve `
  --queries runs/task1-smoke/queries.jsonl `
  --pages runs/task1-smoke/pages.jsonl `
  --output runs/task1-smoke/retrieval.jsonl

py src/task1/task1_pipeline.py evaluate `
  --predictions runs/task1-smoke/retrieval.jsonl `
  --metrics-out runs/task1-smoke/metrics.json `
  --per-query-out runs/task1-smoke/per_query.csv

py src/task1/task1_pipeline.py export-task2 `
  --predictions runs/task1-smoke/retrieval.jsonl `
  --evidence-pages 1 `
  --output runs/task1-smoke/task2-input.csv
```

## VLM dry-run

```powershell
py src/task1/task1_pipeline.py rerank `
  --retrieval runs/task1-smoke/retrieval.jsonl `
  --output runs/task1-smoke/reranked-dryrun.jsonl `
  --cache runs/task1-smoke/vlm-cache.jsonl
```

The command above makes no API calls. Inspect prompts and candidate pages first.
To execute, set `OPENAI_API_KEY` in `.env` and explicitly add `--execute-api`.
The default API configuration uses `gpt-5.4-nano`, 96-DPI images, low image
detail, and at most 900 output tokens to keep the pilot inexpensive. Use
`--image-detail high`, a larger `--dpi`, or `gpt-5.4-mini` only for a measured
quality comparison. The cache makes reruns free of duplicate API calls.
Official OpenAI documentation confirms that the Responses API accepts image
inputs and can generate text or JSON output:
https://developers.openai.com/api/reference/cli/resources/responses/methods/create

## Reconstructed full-set diagnostic

The 2026-08-26 local run produced 621 reconstructed queries across 63 reports
and 7,179 pages. Evaluation excludes 156 empty-gold queries and uses 465
non-empty-gold queries: Hit@1 0.151, Hit@5 0.383, Hit@10 0.514, Near@1 0.194,
and MRR 0.254. There were 230 queries whose grouped page rows contained more
than one label. Consequently, these values are a pipeline diagnostic and must
not be presented as official Task 1 results.
