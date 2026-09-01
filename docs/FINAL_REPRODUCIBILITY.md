# Final Reproducibility Guide

How to regenerate every number that appears in `paper/regcomagent_rag_ntcir_claude.tex` and its
companion docs, as of 2026-09-01. Commands assume the repository root as the working directory and
the Python launcher `py` (also works with `python3` if available on your PATH; none of the scripts
below have third-party dependencies — only the standard library).

## 1. Subtask 2 headline numbers (Tables `tab:overall`, `tab:class`, `tab:lang`)

These come from `artifacts/metrics/model_comparison_793_no_zh.json` and the per-model metrics files
it points to. That file is already checked into the repo and was re-verified byte-for-byte against
the paper's claims in this pass — no regeneration is needed to reproduce the numbers *as reported*.

To regenerate from scratch you would need the original prediction files under
`data/predictions/...` (not present in this checkout) and would need to call the Gemini/GLM APIs
again for `src/task2/gemini_rag_predict_no_zh.py` / `src/task2/glm_rag_predict_english.py` — this
was explicitly out of scope for this pass (no API calls were made) and is not required to verify
the numbers already reported, only to produce new ones.

```
py src/task2/score_and_merge_predictions.py --help   # inspect scoring options; no API calls
```

## 2. Cierpa / CSCU numbers cited in Related Work, Discussion, and the comparison tables

Not computed — read directly from `RegCom_paper_1.pdf` (Cierpa) and `RegCom_paper_2.pdf` (CSCU).
See `docs/CLAUDE_PAPER_REVIEW.md` §4 for the exact page/table each number was checked against.
Re-verify by opening those PDFs; no script involved.

## 3. Task 1 diagnostics (complete): `runs/task1-dense-optimized`

Fully completed (490/490) prior to this pass. Its numbers are quoted narratively from
`docs/EXPERIMENT_LOG.md`'s "Dense retrieval and full VLM reranking (2026-09-01)" entry. The raw
run directory is git-ignored (`runs/` is excluded in `.gitignore`), so it only exists locally;
if present, its own analysis entry points are `src/task1/analyze_vlm.py`,
`src/task1/end_to_end_vlm.py`, and `src/task1/aggregate_e2e.py` (all pre-existing, all API-free
once the cache files exist).

## 4. Task 1 diagnostics (in progress): `runs/task1-neighbor2`

This is the run this pass added tooling for. Two new, dependency-free scripts:

```
# Metrics: Hit@1/5/10, Near@1, MRR, per-language breakdown, candidate recall, API usage/cost/errors
py src/task1/aggregate_neighbor2.py
#   writes artifacts/metrics/task1_neighbor2_summary.json
#   writes artifacts/metrics/task1_neighbor2_per_query.csv
#   prints the summary (everything except the per-query rows) to stdout

# Stratified (language x outcome-category), human-review-only error sample
py src/task1/build_error_review_sample.py
#   reads artifacts/metrics/task1_neighbor2_per_query.csv (must run the aggregator first)
#   writes docs/TASK1_ERROR_REVIEW_SAMPLE.csv (65 rows by default; --target-size to change)
```

Both scripts are **safe to rerun at any time**, including while another process is still appending
to `runs/task1-neighbor2/vlm-full20-cache.jsonl`:

- They only ever *read* from `runs/task1-neighbor2/` (`retrieval.jsonl` and
  `vlm-full20.jsonl`/`vlm-full20-cache.jsonl`, preferring the former if present) and never write
  into that directory.
- A truncated final JSONL line (from reading mid-write) is skipped with a printed warning rather
  than crashing the script.
- `aggregate_neighbor2.py`'s `run_status.is_complete` field tells you whether the snapshot you just
  produced is final (`complete_rows + error_rows >= expected_total` and no pending rows) — check
  this before citing a number as final anywhere.

To regenerate the numbers quoted in the paper's §4.7 second paragraph (the 183/490 snapshot) after
the run has progressed further or finished, just rerun `py src/task1/aggregate_neighbor2.py` and
substitute the new `overall` / `per_language` figures; the paragraph's structure and hedging
language do not need to change until `run_status.is_complete` is `true`, at which point the
"still in progress" framing should be replaced with a normal complete-run report (matching how the
`task1-dense-optimized` paragraph is written) and logged as a new dated `EXPERIMENT_LOG.md` entry.

**Ranking reconstruction note:** `aggregate_neighbor2.py` reimplements (does not import or execute)
the exact `normalize_vlm_ranking` / `hit_at` / `reciprocal_rank` / near@1 logic found in
`src/task1/task1_pipeline.py` (read for reference only, never modified or run by this pass). If
that pipeline's ranking or metric logic changes in the future, `aggregate_neighbor2.py` should be
re-diffed against it to stay faithful.

**Cost estimate caveat:** `aggregate_neighbor2.py` uses the same $0.20/M input, $1.25/M output
`gpt-5.4-nano` rate convention already used in prior `EXPERIMENT_LOG.md` entries. This is not
independently re-verified against a live pricing page (that would require a network call) and does
not model any cached-token discount, so treat the reported cost as an approximation, consistent
with how every other cost figure in this project is already hedged.

## 5. Stratified error-review sample

`docs/TASK1_ERROR_REVIEW_SAMPLE.csv` is generated, not hand-picked. Regenerate with:

```
py src/task1/build_error_review_sample.py --target-size 65 --seed 42
```

The `category` column is a mechanical Hit@1-transition label (candidate_recall_miss /
vlm_improved_top1 / vlm_degraded_top1 / both_top1 / neither_top1 / empty_gold), derived
deterministically from the aggregator's output — it is a fact about the data, not a judgment call.
The qualitative error-taxonomy columns (from `docs/TASK1_VLM_PLAN.md` §13) are intentionally left
blank; no script in this repository fills them in. A human annotator should open the CSV, work
through the `error_taxonomy_category` column using the value list printed in the CSV's first
(comment) row, and record whether each prediction is actually correct.

## 6. LaTeX build

No LaTeX distribution was present on the machine that produced this pass; MiKTeX (basic package
set, per-user install, `AutoInstall=1` so `acmart` and its dependencies fetch automatically on
first use) was installed with explicit user approval. To reproduce the build on a machine with any
working TeX distribution:

```
cd paper
pdflatex -interaction=nonstopmode regcomagent_rag_ntcir_claude.tex
bibtex regcomagent_rag_ntcir_claude
pdflatex -interaction=nonstopmode regcomagent_rag_ntcir_claude.tex
pdflatex -interaction=nonstopmode regcomagent_rag_ntcir_claude.tex
```

Expected result: exit code 0 at every step, a 9-page PDF, zero undefined citations/references,
zero `Overfull \hbox`/`Overfull \vbox` warnings. The only warnings that should appear are the ones
already present in the unmodified `regcomagent_rag_ntcir.tex` (missing affiliation city, ACM
reference-format/CCS-concepts notices, one CJK font-shape substitution, duplicate PDF anchor
names) — verified in this pass by compiling both files with the same toolchain and diffing the
warning sets. If a *new* warning type appears that isn't in that list, treat it as a regression
introduced by whatever changed since this guide was written.

## 7. What is explicitly not reproducible without external access

- Any Gemini/GLM/OpenAI API call (regenerating Subtask 2 predictions from scratch, or extending any
  Task 1 VLM run beyond its current cache) — requires API keys not available to, and not used by,
  this pass.
- An officially-scored Task 1/Subtask 1 result — requires the task organizers' evaluator and gold
  release, which do not exist in this repository.
