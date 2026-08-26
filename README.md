# SASBPageRAG at NTCIR-19 RegCom

This repository contains the paper, source snapshots, evaluation utilities, and
selected analysis artifacts for the IMNTPU participation in the NTCIR-19
RegCom task. The current implemented system focuses on Subtask 2: single-page
SASB metric verification. Planned work extends it into a traceable, multimodal
Subtask 1 pipeline for full-report evidence-page retrieval and end-to-end
compliance checking.

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
src/orchestrator/       Preserved orchestrator source snapshots
evaluation/             Evaluation utilities
artifacts/metrics/      Selected metrics and stage-wise analysis tables
docs/TASK1_VLM_PLAN.md  Full Task 1 VLM research and implementation plan
docs/EXPERIMENT_LOG.md   Reproducible run-record template
```

## Task 1 roadmap

The planned system combines PDF parsing, OCR fallback, ToC grounding, BM25,
multilingual dense retrieval, unit/table matching, reciprocal-rank fusion, VLM
reranking, and the existing Task 2 verifier. See
[docs/TASK1_VLM_PLAN.md](docs/TASK1_VLM_PLAN.md).

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
- Task 1 VLM implementation: planned, not yet reported as an experimental result
- Task 1 official reproduction: pending confirmation of official inputs, gold
  page sets, PDF mapping, and evaluation rules

