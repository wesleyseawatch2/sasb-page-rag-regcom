# Task 1 implementation

`task1_pipeline.py` implements the first runnable Task 1 baseline:

- reconstruct report-level queries and positive page sets from page-level truth;
- index complete PDF reports with PyMuPDF;
- run reusable word/character TF-IDF, metric/unit rules, optional dense
  retrieval, and RRF;
- evaluate Hit@1/5/10, Near@1, and MRR;
- prepare or execute cached OpenAI VLM reranking;
- optionally export a lane-union `candidate_pool` for a larger VLM budget.

The reconstructed query set is an analysis dataset, not a claim that the
official Task 1 data contract has been recovered exactly.

`prepare --truth` accepts the normalized `test_truth.csv`, one language
`*_test.json`, or a directory containing the six participant Test Set JSONs.
The JSON loader preserves page-level `label` values (including `yes but not
complete`) for auditing. The participant JSONs do not contain all of the
enriched `sasb_key_terms` and `sasb_what_counts` fields used by the optimized
CSV diagnostic, so direct-JSON retrieval numbers should be reported separately
from the metadata-enriched CSV run.

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

## Recommended local retrieval configuration

For the full diagnostic, use the default `report_metric` grouping and keep the
original query text while enabling the low-weight character lane:

```powershell
py src/task1/task1_pipeline.py retrieve `
  --queries runs/task1-full/queries.jsonl `
  --pages runs/task1-full/pages.jsonl `
  --output runs/task1-full/retrieval-optimized.jsonl `
  --query-text-mode raw `
  --char-weight 0.02 `
  --top-k 30 `
  --candidate-pool-k 30
```

`candidate_pool` interleaves the strongest pages from each retrieval lane. Set
`--ranking-field candidate_pool --candidates 30` in `rerank` only after checking
the extra image-token cost. Dense retrieval is optional and CPU-intensive; its
page embeddings are truncated to 1,500 characters and cached under
`cache/task1-dense/`.

The 6-query `candidate_pool` pilot sent 30 images per query. Candidate recall
was 5/6, but VLM reranking reached only Hit@1 0.167, Hit@5 0.333, Hit@10
0.500, and MRR 0.261 (estimated cost USD 0.056). The original 10-image pilot
was stronger, so treat the larger pool as an optional candidate-recall study,
not the default VLM configuration.

For inexpensive VLM pilots, `rerank` also supports `--languages` and
`--max-per-language`; these options only limit a run and do not change the
ranking algorithm.
Official OpenAI documentation confirms that the Responses API accepts image
inputs and can generate text or JSON output:
https://developers.openai.com/api/reference/cli/resources/responses/methods/create

## Reconstructed full-set diagnostic

The 2026-08-26 local run produced 621 reconstructed queries across 63 reports
and 7,179 pages. Evaluation excludes 156 empty-gold queries and uses 465
non-empty-gold queries: Hit@1 0.151, Hit@5 0.383, Hit@10 0.514, Near@1 0.194,
and MRR 0.254. There were 230 queries whose grouped page rows contained both
positive and negative page labels; this is expected for page retrieval and is
not by itself an annotation conflict. Only 3 query-page groups had two
different labels. Consequently, these values are a pipeline diagnostic and
must not be presented as official Task 1 results.
