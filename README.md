# SASBPageRAG at NTCIR-19 RegCom

This repository contains the paper, source snapshots, evaluation utilities, and
selected analysis artifacts for the IMNTPU participation in the NTCIR-19
RegCom task. It includes the submitted Subtask 2 verifier and a runnable,
traceable Subtask 1 baseline for full-report evidence-page retrieval, optional
VLM reranking, and export back into the Subtask 2 input schema.

## Current results

The preserved non-Chinese Subtask 2 evaluation contains 793 test rows.

| System | Micro F1 | Macro F1 |
|---|---:|---:|
| Gemini 2.5 Pro RAG | 0.6507 | 0.6140 |
| GLM-5.2 RAG | 0.6280 | 0.5963 |
| MiniLM + logistic regression | 0.4603 | 0.4489 |

These results exclude Chinese and should not be compared directly with other
submissions that use different retained rows, cleaning rules, or modalities.

## Repository layout

```text
paper/                  Revised paper source, bibliography, figure, and PDF
src/task2/              Text-based RAG and embedding baselines
src/task1/              Full-PDF retrieval, evaluation, and VLM reranking
                        (including cached-trace error analysis)
src/orchestrator/       Preserved orchestrator source snapshots
evaluation/             Evaluation utilities
artifacts/metrics/      Selected metrics and stage-wise analysis tables
docs/TASK1_VLM_PLAN.md  Full Task 1 VLM research and implementation plan
docs/EXPERIMENT_LOG.md   Reproducible run-record template
```

## Task 1 baseline

The implemented first baseline combines PDF parsing, reusable word/character
TF-IDF retrieval, metric/unit rules, optional multilingual dense retrieval,
reciprocal-rank fusion, a lane-union candidate pool, cached VLM reranking,
evaluation, and Task 2 export. It can read the normalized CSV or the original
participant `*_test.json` files directly. See
[src/task1/README.md](src/task1/README.md) for commands and
[docs/TASK1_VLM_PLAN.md](docs/TASK1_VLM_PLAN.md) for the remaining ablations.

On the reconstructed analysis set, the local TF-IDF + rule RRF baseline indexed
63 reports and 7,179 pages. Among 465 queries with non-empty reconstructed gold
sets, it obtained Hit@1 0.151, Hit@5 0.383, Hit@10 0.514, and MRR 0.254. These
are diagnostic results reconstructed from Subtask 2 truth, not official Task 1
scores. No paid VLM calls were used for these numbers.

The optimized cached-word configuration with raw queries and a low-weight
character lane reaches Hit@1 0.151, Hit@5 0.389, Hit@10 0.520, and MRR 0.263
on the same reconstructed set. A lane-union Top-30 candidate pool reaches
0.718 candidate recall, but should be passed to the VLM only with a measured
image budget; the 30-page nano pilot did not improve reranking.

The completed non-Chinese VLM run used 363 report-metric groups and 10 fused
candidate pages per query. On the 284 groups with positive pages, VLM
reranking reached Hit@1 0.289, Hit@5 0.468, Hit@10 0.518, and MRR 0.362,
versus 0.211/0.412/0.518/0.306 for the same fused baseline. These are still
reconstructed diagnostics, not official Task 1 scores.

`src/task1/analyze_vlm.py` aligns the full Chinese and non-Chinese traces with
the fused baseline and reports per-language candidate recall, Top-1 transitions,
and no-answer diagnostics without making additional API calls.

## Setup

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
Copy-Item .env.example .env
```

API keys belong only in `.env`, which is ignored by Git.

## Data availability and public-release boundary

The official RegCom dataset and source ESG PDFs are not redistributed here.
Obtain them through the task organizers and place private inputs under
`data/private/` and source reports under `Training Set/` or `pages/`. Those
paths are ignored. The repository intentionally excludes API keys, credentials,
model caches, restricted PDFs, and raw provider logs.

The available reference files do not provide independently validated gold
labels for `category_match` and `unit_match`. Do not report auxiliary-field
scores unless a validated gold mapping is supplied. Row-level evaluation must
also use a unique sample identifier because some identity tuples repeat and may
contain conflicting labels.

## Paper

The revised draft is available at
[paper/regcomagent_rag_ntcir.pdf](paper/regcomagent_rag_ntcir.pdf).

## Citation

Citation metadata will be updated after the final NTCIR-19 proceedings are
released. The provisional RegCom overview citation is included in the paper
bibliography.

## Status

- Subtask 2 preserved experiments: complete
- Cross-submission comparative analysis: complete
- Task 1 local retrieval/evaluation and VLM interface: implemented
- Task 1 API VLM pilot and full non-Chinese run: complete; full official run remains pending
- Task 1 official reproduction: pending confirmation of official inputs, gold
  page sets, PDF mapping, and evaluation rules
