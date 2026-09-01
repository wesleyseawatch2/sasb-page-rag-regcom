# Claude Error Analysis: Cross-Method Synthesis for RegCom Subtask 1 &amp; 2

Review date: 2026-09-01. Companion to `docs/CLAUDE_PAPER_REVIEW.md` (read that first for the
"Task 1" terminology warning — it applies throughout this document too).

**Sources used:** self-reported error analyses in Cierpa (`RegCom_paper_1.pdf`) and CSCU
(`RegCom_paper_2.pdf`); the current paper's own §4.6 Error Analysis; and this repository's cached,
already-computed diagnostics (`docs/EXPERIMENT_LOG.md`,
`runs/task1-dense-optimized/vlm-pilot-analysis.json`). No new model calls were made to produce this
document, and `runs/task1-dense-optimized/vlm-full*` was not opened. Where a number could not be
grounded in an existing artifact, it is marked `[not available in this checkout]` rather than
estimated.

---

## 1. Unified error taxonomy

The three papers describe overlapping but not identical failure categories. This table maps them
onto one shared taxonomy so the papers can be compared error-for-error.

| Unified category | SASBPageRAG (this paper) | Cierpa | CSCU |
|---|---|---|---|
| **Topical match without compliant disclosure** | "topical matches are sometimes confused with compliant disclosures" (§4.6) | Implicit in ybn-boundary discussion; VLM-sort prompt explicitly scores "relevance" separate from disclosure completeness | Verifier prompt explicitly "penalizes pages that merely frame the topic ... when a neighboring page contains the direct quantitative table" (§4.4) |
| **Partial vs. complete disclosure boundary** | Central finding: `yes but not complete` is weakest class for all 3 systems (F1 0.39–0.48) | ybn is the label most sensitive to few-shot configuration (Korean +34pp, Japanese −19pp accuracy swings, Fig. 3) | "yes but not complete remains the hardest class ... partial disclosures are often split between no, partial, and yes predictions" (§6.1) |
| **Table / layout extraction loss** | "extracted text from PDF tables may lose row or column structure" (§4.6) | Motivates the VTE and IE retrieval signals (image-level features outperform text signals, Table 3) | Motivates the image-only/text+image ablation; image-only alone beats text-only (60.5 vs 55.9 macro F1) |
| **Evidence on an adjacent / neighboring page** | Acknowledged as a limitation of strict single-page evaluation (§4.6), but **not measured** because the system receives the gold page by construction | Not directly measured in Cierpa's retrieval tables | **Directly measured**: Near@1 (46.6%) far exceeds exact F1 (31.0%) for the VLM-Guided variant — most remaining Subtask 1 misses land one page off gold |
| **Candidate-generation miss (page never reaches the verifier)** | N/A (task-provided page, no retrieval stage) | English/French are the weakest BM25 languages; VTE+IE recovers much of this | "In the lexical English reference run, candidate misses account for more than four-fifths of metric instances" (§6.1) |
| **Non-unique / conflicting gold labels** | Flagged qualitatively: "some repeated tuples have conflicting labels" (§5.2) | **Quantified**: up to 57.2% of test instances have non-unique PDF-level gold labels (Table 1); training conflicts are removed (942→903 rows) but **test conflicts are kept** because Subtask 1/2 require a prediction for every instance | Not separately quantified, but the excluded Thai "TISCO" document is a related, concrete data-integrity anomaly |
| **Language-specific bottleneck** | Thai weakest for MiniLM baseline; Japanese/Korean strong for both LLMs (Table `tab:lang`) | Japanese and Thai favor zero-shot over few-shot; Korean gains the most from few-shot (+0.161 Macro F1) | Retrieval and verification bottlenecks are **language-specific but not correlated with each other** — Korean is worst at retrieval (8.1 F1) but best at verification (69.6 F1); Chinese is the reverse (28.3 vs 59.8) |
| **Model-specific decision bias** | GLM-5.2 over-predicts `ybnc` (higher recall, lower precision); Gemini is more conservative, predicts `no` more (§4.5) | Not applicable (single verifier model, GPT-5.5) | Not applicable (single verifier model, GPT-5.4) |
| **Retrieval-to-verification error propagation** | Out of scope (no retrieval stage in this paper) | **Quantified**: End-to-End Inference Macro F1 (0.287–0.412 across languages, Table 10) is far below standalone Label Inference (0.59–0.83, Table 8) once retrieval errors and PDF-level label non-uniqueness both apply | Deliberately **not** measured end-to-end — CSCU scores Subtask 2 on task-supplied pages, not on its own Subtask 1 predictions (§7.1) |

**Reading the taxonomy:** every system independently converges on the same two headline failure
modes — the `yes`/`yes but not complete` boundary, and loss of table/layout structure in plain
text — which is strong triangulated evidence that these are real properties of the SASB
verification task rather than artifacts of any one system's design. The categories where the
systems *disagree* are informative too: only CSCU isolates modality as a single-factor ablation,
only Cierpa quantifies label non-uniqueness, and only the current paper's own diagnostic work (§3
below) speaks to retrieval-to-verification propagation on this specific non-Chinese CSV pipeline.

---

## 2. Language-level error patterns, side by side

| Language | SASBPageRAG (Subtask 2, Gemini Macro F1) | Cierpa (Label Inference, Best FI Macro F1) | CSCU (Subtask 1 macro F1, Lex→VLM) | CSCU (Subtask 2 text-only macro F1) |
|---|---|---|---|---|
| English | 0.4833 (weakest for Gemini) | 0.554 (+0.099 vs. ZI) | 5.4 → 37.9 | 47.2 |
| French | 0.6382 | 0.587 (+0.051) | 6.0 → 13.6 (weakest VLM result) | 30.7 (weakest) |
| Japanese | 0.6801 | 0.663 (**−0.085**, only language hurt by few-shot) | 14.4 → 33.5 | 63.1 |
| Korean | 0.7467 (strongest) | 0.832 (**+0.161**, largest gain) | 8.1 → 29.2 (weakest raw retrieval) | 69.6 (strongest) |
| Thai | 0.5129 | 0.590 (+0.037) | 13.3 → 36.9 | 60.0 |
| Chinese | *(excluded from this paper's scope)* | 0.599 (+0.012, smallest gain) | 28.3 → 32.2 (strongest raw retrieval) | 59.8 |

Two cross-system observations that are easy to miss looking at any single paper:

1. **Korean is a consistent outlier in opposite directions.** It is Cierpa's largest few-shot
   winner and this paper's strongest Gemini language, yet CSCU's *weakest* raw retrieval language
   (8.1 macro F1) before VLM guidance — and CSCU independently notes Korean is simultaneously its
   *best* verification language (69.6). Read together, this suggests Korean report pages are
   comparatively easy to classify **once located**, but comparatively hard to locate with lexical
   signals alone (plausibly a script/tokenization effect on keyword-based retrieval that does not
   equally handicap embedding- or LLM-based judgment).
2. **Japanese is the one language where more context reliably hurts.** Cierpa's few-shot inference
   *decreases* Japanese Macro F1 (0.748→0.663), and this repository's own retrieval diagnostics
   (below) separately show Japanese has the lowest non-Chinese candidate recall@10 in the fused
   retrieval baseline. Two independent systems finding Japanese-specific degradation from added
   context (few-shot examples in one case, retrieval candidates in the other) is a stronger signal
   than either finding alone, and is worth a dedicated Japanese-specific ablation before assuming
   the fix is "add more retrieval/examples" generally.

---

## 3. Task 1 (Subtask 1 / retrieval) diagnostic error findings — this repository, preliminary

**Status: diagnostic pilot data only, not an official Task 1 error sample.** The stratified
50–80-case annotated error sample called for in `docs/TASK1_VLM_PLAN.md` §13 has not been run yet.
What follows is drawn from the 12-query cached pilot in
`runs/task1-dense-optimized/vlm-pilot-analysis.json` (already computed, no API calls made to
produce this document) and the dated summaries in `docs/EXPERIMENT_LOG.md`.

### 3.1 Pilot transition categories (12 queries, cached)

| Category | Count | Meaning |
|---|---|---|
| `neither_top1` | 6 | Neither the fused baseline nor the VLM rerank put the gold page at rank 1 |
| `both_top1` | 2 | Both baseline and VLM already correct at rank 1 |
| `empty_gold` | 2 | Reconstructed query has no gold page in this pilot's truth join |
| `vlm_improved_top1` | 1 | VLM reranking fixed a baseline miss |
| `candidate_recall_miss` | 1 | Gold page was outside the top-10 candidate pool entirely — no reranker can fix this |

This is a 12-sample pilot, not a powered error study — the counts above should be read as
illustrative, not as rates. They are reproduced here only because they come with real,
inspectable examples (below), which is more useful for a qualitative error section than a
percentage with n=12.

### 3.2 Concrete examples (real cached records, not illustrative fabrications)

- **Candidate-generation miss** (`sample_id 098ff73d3c56ba51dffa`, Korean, topic "Materials
  Sourcing," metric on critical-material risk management): gold page 76 never appears in either
  the baseline or VLM top-5 (`[87, 29, 78, 44, 96]` → `[44, 87, 77, 78, 96]`). This is the failure
  mode CSCU calls the dominant Subtask 1 bottleneck ("candidate misses account for more than
  four-fifths of metric instances" in their English lexical run) — no amount of reranking helps
  once the page is outside the candidate pool.
- **VLM reranking fixing a baseline miss** (`sample_id 720e30af1baf916000fe`, French, topic
  "Employee Diversity & Inclusion"): gold page 9 was baseline rank 5+ (`[6, 11, 5, 12, 4]`, MRR
  0.10) and VLM reranking moved it to rank 1 (`[9, 6, 5, 12, 11]`, MRR 1.00). This is a genuine
  case of the mechanism Cierpa's VLM-sort and CSCU's VLM-Guided variant both report at scale.
- **Both signals already correct** (`sample_id 188b357ae7e18d78b458`, French, "Financed
  Emissions"): gold page 14 is baseline and VLM rank 1 — a reminder that not every query is hard;
  aggregate metrics should always be read alongside the share of "easy" queries like this one.

### 3.3 Larger, still-diagnostic runs (from `EXPERIMENT_LOG.md`, dated 2026-09-01)

On the full 490 reconstructed `report_metric` query groups (369 non-empty-gold, 121 empty-gold),
using cached VLM reranking traces:

- Fused baseline → VLM, non-empty groups: Hit@1 **0.192 → 0.268**, Hit@5 0.396 → 0.458, Hit@10
  0.507 → 0.507 (unchanged — reranking cannot add pages absent from the top-10), MRR 0.291 → 0.347.
- Top-1 transition breakdown: both correct 57, VLM improved 42, VLM degraded 14, neither correct
  74, **182 groups had no gold page in the fused top-10 candidate set at all** — i.e., for roughly
  half of the scored groups, candidate recall (not reranking quality) is the binding constraint,
  echoing CSCU's finding almost exactly.
- Candidate recall@10 by language: Chinese 0.471, **English 0.906**, French 0.800,
  **Japanese 0.200**, Korean 0.608, **Thai 0.109**. Japanese and Thai are the clear bottleneck
  languages for this pipeline's current retrieval configuration — consistent with §2's
  cross-system Japanese observation above, and worth comparing against Cierpa's per-language BM25
  numbers in a future revision once the retrieval configuration is frozen (see
  `CLAUDE_PAPER_REVIEW.md` §6 — this was a genuine dense-retrieval-on/off configuration difference,
  not an inconsistency, and has since been resolved by the dense-optimized run's completion; see
  §3.5 below for what the retrieval configuration looks like now).
- End-to-end label diagnostic (Top-1 selected page → 3-way label, `gpt-5.4-nano`, 490/490 traces,
  no API errors): predicted-label distribution `no` 301, `yes but not complete` 143, `yes` 46. Of
  these, only 74 selected pages join unambiguously to a truth row (the rest hit the
  non-unique-label problem quantified in §1's taxonomy row above); label accuracy on that 74-row
  subset is **0.459**. The log itself is explicit that this is "a limited diagnostic, not an
  official Task 1 classification score."

**Interpretation, consistent with all three papers' independent findings:** in this pipeline as in
Cierpa's and CSCU's, retrieval/candidate recall is the dominant bottleneck, not the reranker or the
final verifier — improving Japanese and Thai candidate generation would very likely move the
end-to-end numbers more than further tuning the verifier prompt would.

### 3.4 What is *not* available for this document

- A full confusion matrix for the paper's actually-reported `gemini_25_pro` / `glm_5_2_rag`
  793-row runs. `artifacts/metrics/model_comparison_793_no_zh.json` only stores aggregate
  accuracy/Micro F1/Macro F1; the underlying prediction CSVs referenced by its `predictions_path`
  fields (under `data/predictions/...`) are not present in this checkout (this project's `data/`
  directory contains only `SASB_REFERENCE_README.md`). `[not available in this checkout]` — if
  needed, it is reproducible locally via `src/task2/score_and_merge_predictions.py` without any
  API calls, since the predictions already exist on disk elsewhere.
  `artifacts/metrics/metrics_final.json` is a **different** experiment (an orchestrator/meta-review
  pipeline, 201 English rows, Macro F1 0.4228) and was not used here to avoid mixing it into the
  Subtask 2 headline numbers by mistake.
- An official, organizer-scored Task 1/Subtask 1 error sample for this repository's own pipeline.
- Inter-annotator agreement for any of the three papers' qualitative error taxonomies — none of the
  three reports it. This is exactly what the human-review CSV in §3.5 is for; agreement cannot be
  computed until it is actually annotated by a person.

### 3.5 Follow-up (2026-09-01): neighbor-expanded top-20 run and a stratified review sample

Since §3.3 was written, `runs/task1-dense-optimized` finished its full VLM rerank (490/490,
Hit@1 0.328, MRR 0.431 — see `EXPERIMENT_LOG.md`'s "Dense retrieval and full VLM reranking" entry)
and a new, deeper-candidate run (`runs/task1-neighbor2`: dense retrieval + a `neighbor_window=2`
page expansion + a doubled top-20 VLM candidate pool) started and is **still in progress**. Per the
task instructions for this pass, `runs/task1-neighbor2/vlm-full20*` was never read directly; every
number below comes from the new API-free script `src/task1/aggregate_neighbor2.py`.

Snapshot at 2026-09-01 12:06, 183/490 queries complete (Chinese and English only so far; French,
Japanese, Korean, Thai entirely pending — do **not** read the "overall" row below as a 6-language
figure):

| Scope | Fused Hit@1 | VLM Hit@1 | Fused MRR | VLM MRR | Candidate recall@10 |
|---|---|---|---|---|---|
| Overall (133 non-empty, 2 languages only) | 0.271 | 0.391 | 0.411 | 0.487 | 0.707 |
| Chinese (84 non-empty) | 0.155 | 0.262 | 0.289 | 0.362 | 0.571 |
| English (49 non-empty) | 0.469 | 0.612 | 0.620 | 0.702 | 0.939 |

Mechanical Hit@1-transition category counts over the same 183 rows: `neither_top1` 37,
`vlm_improved_top1` 21, `empty_gold` 50, `both_top1` 30, `candidate_recall_miss` 39,
`vlm_degraded_top1` 6. Chinese's English-relative weakness on both fused and VLM Hit@1 is
consistent with the CJK-retrieval difficulty already discussed for Japanese/Thai above; whether it
holds once the harder Japanese/Thai/Korean queries are included remains to be seen and should not
be assumed from this 2-language snapshot.

A stratified (language x mechanical-category), human-review-only sample was built from this same
snapshot: `docs/TASK1_ERROR_REVIEW_SAMPLE.csv` (65 rows, via
`src/task1/build_error_review_sample.py`). It intentionally leaves the qualitative
error-taxonomy columns from `TASK1_VLM_PLAN.md` §13 (candidate-generation miss, wrong ToC section,
exact-vs-adjacent page, SASB index/reference page, evidence-spanning pages, topical-mention-only,
table/OCR extraction failure, missing value/unit/denominator/scope/period/disaggregation,
full-vs-partial boundary, ambiguous/conflicting gold, model JSON/normalization failure) **blank**
for a human annotator — nothing in this pass pre-fills a judgment. Because the source run has only
reached Chinese and English, the sample is currently 2-language only; rerunning the same command
after `runs/task1-neighbor2` completes will regenerate it with full 6-language stratified coverage.

---

## 4. Recommended next error-analysis steps

1. ~~Run the stratified 50–80 case sample~~ — **started**: `docs/TASK1_ERROR_REVIEW_SAMPLE.csv`
   (65 rows) exists with real, mechanically-categorized context, but (a) covers only Chinese/English
   pending `runs/task1-neighbor2` completion, and (b) still needs a human to actually fill in the
   qualitative error-taxonomy columns — nothing in this repository should claim inter-annotator
   agreement or a quantified taxonomy until that happens.
2. ~~Reconcile the fused-baseline discrepancy~~ — **done**, see `CLAUDE_PAPER_REVIEW.md` §6; it was
   a dense-retrieval-on/off configuration difference, not an unresolved inconsistency.
3. Once `runs/task1-neighbor2/vlm-full20-cache.jsonl` stops growing, rerun
   `py src/task1/aggregate_neighbor2.py` and `py src/task1/build_error_review_sample.py`, then
   distill the final numbers into a dated `EXPERIMENT_LOG.md` entry, replacing the "in progress"
   framing used in §3.5 above and in `_claude.tex` §4.7.
4. If a confusion matrix for the actual reported Gemini/GLM 793-row runs is wanted for the paper,
   regenerate it from the existing prediction CSVs with `src/task2/score_and_merge_predictions.py`
   — no new model calls required.
