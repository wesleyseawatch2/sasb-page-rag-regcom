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
  §4.7 is labeled diagnostic, dated, and sourced. The unresolved official-evaluator issue is
  stated in prose rather than left as a submission-facing placeholder.
- [x] Completed `task1-neighbor2` VLM results reported as a final, explicitly diagnostic run
  (490/490 queries, all six languages, 0 errors), with the 183/490 intermediate snapshot removed.
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
- [x] `task1-neighbor2` run completion: 490/490 queries and 0 errors. The final aggregator and
  six-language error-review sample were regenerated and checked into the companion artifacts.
- [ ] **Blocked:** official Subtask 1 evaluator and gold-page-set contract confirmation from the
  task organizers (`docs/TASK1_VLM_PLAN.md` §1's stated claim boundary). This is the only item
  that blocks calling any Task 1 number "official."
- [ ] **Pending:** complete the pre-specified local BGE-M3 + BGE-Reranker-v2-M3 ablation. The
  implementation and GPU runbook are committed, but no BGE result is included in the paper yet.
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
- [x] Table overflow: zero `Overfull \hbox`/`Overfull \vbox` warnings anywhere near any table,
  including the two new/modified ones (`tab:pros-cons`, `tab:system-comparison`). One negligible
  (1.8pt, imperceptible in print) `Overfull \hbox` appeared in body prose (the Section 3.2 prompt
  template quote block, lines 149-155) in the very last recompile of this pass, downstream of the
  same pre-existing CJK-italic font-shape substitution noted below -- not a table, not introduced
  by content in this section, and not fixed further given the effort/value tradeoff at 1.8pt.
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
| `src/task1/aggregate_neighbor2.py` | API-free metrics aggregator for the completed `task1-neighbor2` diagnostic run |
| `src/task1/build_error_review_sample.py` | Stratified, human-review-only error-sample CSV generator |
| `artifacts/metrics/task1_neighbor2_summary.json` | Aggregator output (overall + per-language + API accounting) |
| `artifacts/metrics/task1_neighbor2_per_query.csv` | Per-query aggregator output |
| `docs/TASK1_ERROR_REVIEW_SAMPLE.csv` | 65-row stratified sample, blank judgment columns |
| `docs/EXPERIMENT_LOG.md` (updated) | Dated entries for the completed `task1-neighbor2` run and ablation decision |
| `docs/CLAUDE_PAPER_REVIEW.md` (updated) | §6/§7 updated: fused-baseline discrepancy marked resolved; neighbor2 follow-up documented |
| `docs/CLAUDE_ERROR_ANALYSIS.md` (updated) | New §3.5 with neighbor2 snapshot; §4 recommendations updated |
| `paper/regcomagent_rag_ntcir_claude.tex` (updated) | §4.7 extended with the neighbor2 paragraph; `gpt-5.4-nano` backtick fixed to `\texttt{}` |
| `paper/regcomagent_rag_ntcir_claude.pdf` (recompiled) | Latest successful build |
| `docs/FINAL_SUBMISSION_CHECKLIST.md` | This file |
| `docs/FINAL_REPRODUCIBILITY.md` | Companion reproducibility guide |

## Not done, and why

- The paper was **not** rewritten to treat the now-complete `task1-neighbor2` run as the primary
  diagnostic, even though it also finished during this pass. The reason is no longer "incomplete" --
  it is that the 20-candidate configuration is not a strict improvement over the 10-candidate
  `task1-dense-optimized` run (better candidate recall and Chinese/Japanese/Korean/Thai Hit@1, but
  a French Top-1 regression and slightly lower pooled Hit@5/Near@1/MRR). §4.7 reports both
  configurations' final numbers side by side with this trade-off stated explicitly, rather than
  picking a "winner"; freezing one as *the* primary diagnostic is left as an explicit open decision
  for the authors (`docs/CLAUDE_PAPER_REVIEW.md` §6/§7).
- No API calls were made anywhere in this pass (aggregation and sampling are pure local
  computation over already-cached files).
- `src/task1/task1_pipeline.py`, `runs/`, `cache/`, and `.env` were read where necessary for
  understanding (e.g., to reverse-engineer the exact VLM-ranking normalization formula so the new
  aggregator matches it) but never modified.
