# Final Submission Checklist

Snapshot date: 2026-09-01. Scope: `paper/regcomagent_rag_ntcir_claude.tex` and its companion docs.
This checklist is a status report, not a claim that the paper is ready to submit as-is — several
items below are explicitly **blocked** and are listed as such rather than marked done.

## Paper content

- [x] Cross-method comparison (methods, models, data, evaluation, results, limitations) against
  Cierpa and CSCU, verified line-by-line against the source PDFs — see
  `docs/CLAUDE_PAPER_REVIEW.md` §4 "Verification log".
- [x] Cross-method error analysis integrated into §4.6 and expanded in
  `docs/CLAUDE_ERROR_ANALYSIS.md`.
- [x] Pros/cons table added (`tab:pros-cons`, Discussion §5.1) — separate from the existing design
  table (`tab:system-comparison`), with explicit Strengths/Weaknesses columns per instruction.
- [x] Diagnostic vs. official Task 1 results kept clearly separated. Verdict from the review:
  the paper never claims an official Task 1/Subtask 1 result; every retrieval/VLM number in
  §4.7 is labeled diagnostic, dated, and sourced. One `[PLACEHOLDER]` remains for the number that
  does not exist yet (organizer-scored official Task 1 result) — this is intentional, not an
  oversight.
- [x] Unfinished `task1-neighbor2` VLM results reported as an explicit, hedged, in-progress
  snapshot (183/490, 2 of 6 languages) rather than extrapolated or silently completed.
- [x] Every number added or checked in this pass traces to one of: (a) a source PDF page (Cierpa /
  CSCU), (b) `artifacts/metrics/*.json` already on disk, (c) a dated `docs/EXPERIMENT_LOG.md`
  entry, or (d) live output of `src/task1/aggregate_neighbor2.py` (API-free, rerun during this
  pass). No number in the diagnostic section was estimated or interpolated.

## Data / experiment integrity

- [x] Subtask 2 headline numbers (Gemini 2.5 Pro / GLM-5.2 / MiniLM+LR) re-verified against
  `artifacts/metrics/model_comparison_793_no_zh.json` — exact match, no drift since the first
  review pass.
- [x] `runs/`, `cache/`, `.env`, and `src/task1/task1_pipeline.py` were not modified by this pass
  (git status confirms only pre-existing, non-Claude changes to `task1_pipeline.py`, present before
  this session started).
- [x] `runs/task1-neighbor2/vlm-full20*` was not read directly at any point; all neighbor2 numbers
  came from `src/task1/aggregate_neighbor2.py`, which itself only reads `retrieval.jsonl` and the
  VLM cache and writes outside `runs/`.
- [ ] **Blocked:** `task1-neighbor2` run completion. 183/490 as of the snapshot cited in the paper;
  another process was still writing it when this pass finished. Rerun
  `py src/task1/aggregate_neighbor2.py` after it stops growing (see
  `docs/FINAL_REPRODUCIBILITY.md`).
- [ ] **Blocked:** official Subtask 1 evaluator and gold-page-set contract confirmation from the
  task organizers (`docs/TASK1_VLM_PLAN.md` §1's stated claim boundary). This is the only item
  that blocks calling any Task 1 number "official."
- [ ] **Blocked:** human annotation of `docs/TASK1_ERROR_REVIEW_SAMPLE.csv`. The CSV exists (65
  rows, stratified) but every judgment column is intentionally blank; no inter-annotator agreement
  can be computed until a person annotates it.
- [ ] **Open:** whether the non-Chinese Subtask 2 CSV split shares document identities across
  train/test the way CSCU found for the official Subtask 1 split — flagged as an open question in
  the paper (§5.2), not yet audited.

## LaTeX build

- [x] Compiles cleanly end-to-end: `pdflatex -> bibtex -> pdflatex -> pdflatex`, exit code 0 at
  every step, 9-page PDF produced (`paper/regcomagent_rag_ntcir_claude.pdf`).
- [x] Citations: `bibtex` reports 0 warnings/errors; all 18 `\cite` keys resolve against
  `regcomagent_rag_refs.bib`; the post-bibtex log has zero "Citation ... undefined" or
  "Reference ... undefined" warnings.
- [x] Table overflow: zero `Overfull \hbox` and zero `Overfull \vbox` warnings anywhere in the
  final-pass log, including around the two new/modified tables (`tab:pros-cons`,
  `tab:system-comparison`).
- [x] Diagnostic-vs-official separation double-checked post-edit: §4.7 and the Limitations
  paragraph both restate that no retrieval number is official; a full-text search for `Hit@`/`MRR`
  outside §4.7 (and its cross-references) returns nothing.
- [x] Remaining warnings (missing affiliation city, "image without description," ACM
  reference-format / CCS-concepts notices, one CJK font-shape substitution, duplicate PDF anchor
  names) were verified to be **pre-existing** by compiling the unmodified original
  `regcomagent_rag_ntcir.tex` the same way and diffing the warning sets — identical, so nothing in
  this pass introduced a new build issue.

## New artifacts produced this pass

| File | Purpose |
|---|---|
| `src/task1/aggregate_neighbor2.py` | API-free metrics aggregator for the in-progress `task1-neighbor2` run |
| `src/task1/build_error_review_sample.py` | Stratified, human-review-only error-sample CSV generator |
| `artifacts/metrics/task1_neighbor2_summary.json` | Aggregator output (overall + per-language + API accounting) |
| `artifacts/metrics/task1_neighbor2_per_query.csv` | Per-query aggregator output |
| `docs/TASK1_ERROR_REVIEW_SAMPLE.csv` | 65-row stratified sample, blank judgment columns |
| `docs/EXPERIMENT_LOG.md` (updated) | New dated entry for the `task1-neighbor2` in-progress snapshot |
| `docs/CLAUDE_PAPER_REVIEW.md` (updated) | §6/§7 updated: fused-baseline discrepancy marked resolved; neighbor2 follow-up documented |
| `docs/CLAUDE_ERROR_ANALYSIS.md` (updated) | New §3.5 with neighbor2 snapshot; §4 recommendations updated |
| `paper/regcomagent_rag_ntcir_claude.tex` (updated) | §4.7 extended with the neighbor2 paragraph; `gpt-5.4-nano` backtick fixed to `\texttt{}` |
| `paper/regcomagent_rag_ntcir_claude.pdf` (recompiled) | Latest successful build |
| `docs/FINAL_SUBMISSION_CHECKLIST.md` | This file |
| `docs/FINAL_REPRODUCIBILITY.md` | Companion reproducibility guide |

## Not done, and why

- The paper was **not** rewritten to treat `task1-neighbor2` as the primary diagnostic, because it
  is incomplete. The complete `task1-dense-optimized` run (490/490) remains the primary citable
  diagnostic per `docs/CLAUDE_PAPER_REVIEW.md` §6's recommendation.
- No API calls were made anywhere in this pass (aggregation and sampling are pure local
  computation over already-cached files).
- `src/task1/task1_pipeline.py`, `runs/`, `cache/`, and `.env` were read where necessary for
  understanding (e.g., to reverse-engineer the exact VLM-ranking normalization formula so the new
  aggregator matches it) but never modified.
