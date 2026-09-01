# Claude Paper Review: SASBPageRAG vs. Cierpa vs. CSCU

Review date: 2026-09-01
Scope: `paper/regcomagent_rag_ntcir.tex` (IMNTPU / SASBPageRAG submission) compared against
`RegCom_paper_1.pdf` (Cierpa at the NTCIR-19 RegCom Task) and `RegCom_paper_2.pdf` (CSCU at the
NTCIR-19 RegCOM Task), plus this repository's `docs/EXPERIMENT_LOG.md` and
`docs/TASK1_VLM_PLAN.md`.

**Method note:** This review was produced by reading the three papers and the repository's own
logged/cached diagnostics. No OpenAI API calls were made, `src/task1/task1_pipeline.py` was not
modified, and `runs/task1-dense-optimized/vlm-full*` and `cache/` were not read or altered. Every
number quoted below from Cierpa or CSCU was checked against the source PDF page it appears on;
every number quoted from the current `.tex` was checked against `artifacts/metrics/*.json` in this
repo. See "Verification log" below for the line-by-line result.

---

## 1. Terminology warning: "Task 1" means three different things

This is the single most important source of confusion across the three papers and must be handled
carefully in any integrated text:

| Source | What "Task 1" / "Subtask 1" refers to |
|---|---|
| **NTCIR-19 RegCom official task** | "RegCom Task 1" is the overall shared task. It has **Subtask 1** (find the evidence page(s) for a metric inside a full report) and **Subtask 2** (given one page + one metric, predict `pred_label`/`category_match`/`unit_match`). |
| **Current paper (`regcomagent_rag_ntcir.tex`)** | Uses the official meaning: "RegCom Task 1 - Subtask 2" (line 60). The paper reports **only** Subtask 2. It never reports Subtask 1/retrieval numbers of its own. |
| **Cierpa paper** | Uses "Subtask 1 = Full-Report Compliance Matching" as a label for **End-to-End Inference** (retrieval + label, evaluated jointly), and treats "Relevant-Page Retrieval" as *"a subproblem within Subtask 1"* (Cierpa §3, p.2). So Cierpa's "Subtask 1" is broader than the official retrieval-only Subtask 1. |
| **CSCU paper** | Uses "Task 1" as the name for the *entire* shared task (retrieval + verification together — see abstract: *"a traceable system for NTCIR-19 RegCOM Task 1, covering multilingual evidence-page retrieval and page-level compliance verification"*), and then splits it into Subtask 1 (retrieval, official sense) and Subtask 2 (verification, official sense). CSCU's usage matches the official split cleanly. |
| **This repository's own docs/dirs** (`docs/TASK1_VLM_PLAN.md`, `EXPERIMENT_LOG.md`, `runs/task1-*`, `src/task1/`) | "Task 1" is used as shorthand for **retrieval only** — i.e., the official **Subtask 1**. The repo's "Task 2" (`src/task2/`) is what the paper calls Subtask 2 and is what `regcomagent_rag_ntcir.tex` actually reports. |

**Practical consequence:** when the new paper text says "our Task 1 diagnostic," it must be
understood as *official Subtask 1 (retrieval)*, matching CSCU's usage and the retrieval half of
Cierpa's pipeline — not Cierpa's own "Subtask 1" label (which is End-to-End). The `_claude.tex`
draft spells this out explicitly the first time the new section uses the term, to avoid exactly
this ambiguity.

---

## 2. System-by-system summary

| | **SASBPageRAG (this paper)** | **Cierpa** | **CSCU** |
|---|---|---|---|
| Team / venue role | IMNTPU, NTCIR-19 RegCom | Cierpa & Co. + Musashino Univ. + UCSD, NTCIR-19 RegCom | Chulalongkorn Univ., NTCIR-19 RegCOM |
| Subtasks reported | Subtask 2 only, **non-Chinese** (EN/FR/JA/KO/TH) | Subtask 1 (retrieval) + Subtask 2, **all six languages** | Subtask 1 + Subtask 2, **all six languages** |
| Core design | Single-stage: TF-IDF few-shot retrieval → one LLM verifier call, on the **task-provided** page | Two-stage: multi-signal page retrieval (BM25/TE/VTE/IE) + RRF → VLM-sort rerank → few-shot label inference | Two-stage: ToC-first section localization (+ optional VLM guidance) → page verifier (text/image/text+image) |
| Fine-tuning | None (retrieval-augmented prompting) | None (few-shot prompting) | None (prompting) |
| Verifier model(s) | Gemini 2.5 Pro, GLM-5.2, MiniLM+LogReg baseline | GPT-5.5 (native PDF-page input) for Label Inference; GPT-5.4-mini / GPT-5.5 for VLM-sort | Azure OpenAI GPT-5.4 (2026-03-05) for both ToC/section VLM checks and the page verifier |
| Retrieval embedding / matching | TF-IDF (uni+bigram), metric-first + label-balanced selection | Gemini Embedding 2 (dense), BM25, VLM-generated element descriptions, image embeddings, RRF | Keyword ToC/section matching + text-overlap reranking; VLM-Guided variant adds VLM section selection and joint page filtering |
| Test scope actually scored | 793 rows, 5 languages (Chinese excluded) | 615 test instances (retrieval, `no`-label rows excluded) / cross-validated + test Label Inference across 6 languages | 61 anomaly-filtered documents, 438 Subtask 1 metric instances, 967 Subtask 2 page records, 6 languages |
| Data cleaning applied | Deterministic post-processing of auxiliary fields only | Removes train-only page-number offset (`hokkaido_gas.pdf`) and train-only (cid,page,topic,metric) label conflicts (942→903 rows, 4.1%); **test data left unmodified** | Excludes one verified Thai "TISCO" source-alignment anomaly (62→61 test docs); otherwise unmodified |
| Headline result(s) | Gemini RAG: **0.6507 Micro F1 / 0.6140 Macro F1** (793 rows) | Best FI Label Inference: **0.637 Macro F1** (test, all 6 languages); retrieval BM25+VTE+IE **MRR 0.527**, VLM-sort (GPT-5.5) **MRR 0.636** | Subtask 2 text+image: **62.4 Macro F1 / 67.9% Acc** (967 records); Subtask 1 VLM-Guided: **31.0 macro F1** (paired, 14.8 for Lexical Reference) |
| Statistical testing | None reported | None reported | Bootstrap 95% CIs (Subtask 1 gain, Subtask 2 modes), Wilcoxon signed-rank + sign test for the paired Subtask 1 gain |
| Self-reported key limitation | Non-Chinese only; no zero-shot/random-example control; text-only page evidence | OptRRF (trained weights) underperforms untuned BM25+VTE+IE on test — weight tuning did not generalize; VLM-sort cost is high ($11–$62 for 615 instances) | Official split reuses document identities across train/test → "seen-document, unseen-query," not unseen-document generalization; Subtask 1 and Subtask 2 evaluated independently (no true end-to-end score) |

---

## 3. Detailed comparison

### 3.1 Data and preprocessing

- **SASBPageRAG** works from a pre-extracted CSV (SASB metadata + extracted PDF page text), 753
  training rows / 793 test rows, non-Chinese only. It performs no train-set cleaning beyond
  deterministic normalization of `category_match`/`unit_match` when `pred_label = no`.
- **Cierpa** is the only one of the three to publish a quantitative audit of label non-uniqueness
  in the *official* RegCom release (their Table 1): across all six languages, **21.1%/16.5%**
  (train/test) of yes/ybn instances have a non-unique gold *relevant page* for the same
  `(cid, topic, metric, value, unit)` tuple; **55.6%/57.2%** of *all* instances have a non-unique
  gold *label* at the PDF level (because `no` means "not on this page," not "not in this report" —
  so a single report can legitimately have both a `yes` page and a `no` page for the same metric).
  Cierpa cleans only the **training** split (removes a `hokkaido_gas.pdf` page-offset bug and
  page-level label conflicts, 942→903 rows) and explicitly leaves test data untouched.
- **CSCU** audits the release from the retrieval-generalization angle: the official Subtask 1
  train/test split reuses the *same 62 document identities* in both splits, so CSCU explicitly
  reframes its own results as "seen-document, unseen-query" rather than held-out-document
  generalization. CSCU also finds and excludes one Thai document (`TISCO`) whose canonical PDF did
  not match the source used for annotation.
- **Cross-cutting implication:** the current paper's own Discussion (§5.2) already flags "some
  identity tuples recur, and some repeated tuples have conflicting labels" in general terms.
  Cierpa's Table 1 now gives that claim concrete numbers, and CSCU's split audit adds a *second*,
  previously unstated caveat (seen-document evaluation) that plausibly also applies to
  SASBPageRAG's and Cierpa's splits, since all three draw on the same official release — but this
  has not been independently verified for the non-Chinese CSV split used here, so it is stated as
  an open question, not a confirmed fact, in the revised paper.

### 3.2 Retrieval / evidence localization (Subtask 1 side)

| Signal | Cierpa | CSCU |
|---|---|---|
| Lexical | BM25 (H@1 0.151, MRR 0.231) | ToC/section keyword matching + text-overlap score (English macro F1 5.4, Chinese 28.3) |
| Semantic | Text embeddings "TE" (H@1 0.232, MRR 0.368) | — (not used as a separate lane) |
| Visual-assisted text | VLM-generated element descriptions "VTE" (H@1 0.305, MRR 0.436) | — |
| Pure image | Image embeddings "IE" (H@1 0.357, MRR 0.511) — best single signal | — |
| Fusion | RRF; best untuned combo BM25+VTE+IE MRR 0.527; **trained-weight OptRRF regressed to 0.497** | Not a multi-lane fusion; ToC narrowing + local-window text-overlap ranking |
| VLM reranking / guidance | VLM-sort re-ranks OptRRF top-10 candidate **pages** with GPT-5.4-mini/GPT-5.5 (MRR 0.497→0.584→0.636; cost $11.0/$62.1 for 615 instances) | VLM-Guided variant uses a VLM to validate/replace the ToC guess **and** select the disclosure-bearing **section**, then jointly verifies the top-12 local pages (macro F1 14.8→31.0, Near@1 10.4→46.6, 95% CI [11.0, 21.5] on the paired gain) |
| Key finding | No single signal suffices; image-level features are the strongest single lane; naive weight-tuned fusion can *underperform* an untuned union | Retrieval headroom is dominated by candidate generation/section grounding, not by the final page-level judgment — evidenced by Near@1 roughly 1.5x the exact-F1 |

This repository's own `docs/TASK1_VLM_PLAN.md` architecture (BM25/word+char lexical, dense
retrieval, metric/unit rules, optional visual embeddings → weighted RRF → lane-union pool →
VLM reranking of top-10) is structurally closer to **Cierpa's** design (multi-lane fusion + VLM
reranking of candidate pages) than to CSCU's ToC-first narrowing. It does not yet implement a
ToC/structure-grounding lane, which is exactly the component CSCU identifies as its main lever
(macro F1 14.8→31.0 from adding VLM-checked ToC/section selection alone).

### 3.3 Label inference / verification (Subtask 2 side)

- **SASBPageRAG**: single verifier call per page; three TF-IDF-retrieved, label-balanced,
  metric-prioritized demonstrations; text-only page evidence (extracted PDF text, ≤5,000 chars);
  same prompt/schema for both LLMs compared.
- **Cierpa**: compares zero-shot (ZI), few-shot (FI), and *extended* few-shot (xFI, which adds the
  model's own zero-shot prediction + correctness + rationale into each demonstration) with random
  (RS) vs. metric-prioritized (MS) sampling, evaluated via 3-fold company-grouped cross-validation
  to pick a per-language "Best FI" before touching test data. Native PDF-page input to GPT-5.5
  (so **both text and image are implicitly available** to Cierpa's verifier, unlike SASBPageRAG's
  text-only page evidence).
- **CSCU**: fixes the few-shot design (not the focus) and instead ablates **evidence modality**
  directly — text-only vs. image-only vs. text+image — on an otherwise identical GPT-5.4 verifier
  and prompt, which is the cleanest single-factor ablation among the three systems for the
  "does the model need the page image" question. Text-only alone (55.9 macro F1) is *below*
  image-only (60.5), and text+image (62.4) is best but statistically indistinguishable from
  image-only (overlapping bootstrap CIs).
- **Consistent finding across all three**: `yes but not complete` is the hardest class everywhere
  (SASBPageRAG F1 0.39–0.48; Cierpa's ybn accuracy swings are the primary driver of its per-language
  volatility; CSCU calls it "the hardest class" with errors split three ways among `no`/`partial`/`yes`).

### 3.4 Evaluation protocol differences that block a direct leaderboard reading

1. **Language coverage**: SASBPageRAG = 5 languages (no Chinese); Cierpa and CSCU = 6.
2. **Retained-row count and cleaning**: 793 (SASBPageRAG, uncleaned CSV) vs. 615 test instances for
   Cierpa's retrieval eval (excludes `no`-label rows) vs. 967 for CSCU's Subtask 2 (anomaly-filtered,
   one document excluded).
3. **Evidence given to the verifier**: SASBPageRAG and CSCU's text-only condition score the model
   on extracted text; Cierpa's GPT-5.5 and CSCU's image/text+image conditions give the model the
   rendered page. This is a modality confound, not just a model/prompt confound.
4. **Significance testing**: only CSCU reports bootstrap CIs and paired tests; SASBPageRAG and
   Cierpa report point estimates only. None of the three papers report a **cross-system** paired
   test (they couldn't — the row sets differ), which is exactly why the current paper is correct to
   frame Table `tab:reported-comparison` as contextual rather than a ranking (see §5 below).

---

## 4. Verification log

Every quantitative claim already present in `regcomagent_rag_ntcir.tex` that references Cierpa or
CSCU was checked against the source PDF. Every SASBPageRAG-internal number was checked against
`artifacts/metrics/model_comparison_793_no_zh.json`. All checks passed; **no fabricated or
misattributed numbers were found in the existing text.**

| Claim in current `.tex` | Source checked | Result |
|---|---|---|
| Gemini 2.5 Pro RAG: 0.6507 Micro F1 / 0.6140 Macro F1 | `artifacts/metrics/model_comparison_793_no_zh.json` → `gemini_25_pro` | Matches exactly |
| GLM-5.2 RAG: 0.6280 Micro F1 / 0.5963 Macro F1 | same file → `glm_5_2_rag` | Matches exactly |
| MiniLM+LR: 0.4603 Micro F1 / 0.4489 Macro F1 | same file → `minilm_logreg` | Matches exactly |
| "Cierpa reports an overall Macro F1 of 0.637 ... across six languages" | Cierpa Table 8, ALL/Best FI | Matches (0.637) |
| "CSCU reports 0.624 Macro F1 and 67.9% accuracy ... over 967 retained page records" | CSCU Table 4, Text+image row | Matches (62.4 / 67.9%) |
| "Cierpa's ... Macro F1 from 0.621 for zero-shot ... to 0.634 in cross-validation ... test configuration reaches 0.637" | Cierpa Table 5 (ZI ALL 0.621, MS+FI/MS+xFI ALL 0.634) and Table 8 (Best FI ALL 0.637) | Matches |
| "reported Japanese Macro F1 decreases from 0.748 to 0.663 while Korean increases from 0.671 to 0.832" | Cierpa Table 8 | Matches |
| "CSCU's ... text-only, image-only, and text-plus-image ... 0.559, 0.605, and 0.624" | CSCU Table 4 | Matches |
| "confidence intervals for image-only and text-plus-image overlap" | CSCU §6.1, text+image paragraph | Matches (CSCU states the margin "falls within these intervals") |
| "Cierpa reports reranking costs of US$11.0 and US$62.1 ... for 615 test instances" | Cierpa §5.1.3 | Matches |
| "retrieval MRR from 0.497 without reranking to 0.584 with GPT-5.4-mini and 0.636 with GPT-5.5" | Cierpa Table 4 | Matches |
| "CSCU's Subtask 1 analysis ... many retrieval errors fall on a page adjacent to the annotated page" | CSCU Table 3, Near@1 columns (10.4→46.6 paired) | Supported, accurately paraphrased |

No corrections to existing verified numbers were needed. The additions made in `_claude.tex` are
new material (deeper method detail, the two dataset audits in §3.1 above, the pros/cons table, and
the diagnostic Task 1/Subtask 1 extension), not fixes to prior numbers.

---

## 5. Finding: does the paper mislabel diagnostic results as official Task 1 results?

**Verdict: No — as of the version reviewed, the paper does not report any Subtask 1 (retrieval)
numbers of its own at all**, diagnostic or otherwise. A full-text search of
`regcomagent_rag_ntcir.tex` for `Hit@`, `MRR`, and `Task 1 result` returns zero matches. The word
"official" appears three times, each time correctly used to *disclaim* a comparison (e.g., "rather
than an official ranking or a paired significance test," "should not silently replace the official
protocol") — never to claim an official status for an in-repo number.

This means the paper is currently **safe** on this specific question, but only because it says
nothing yet about the extensive Task 1/Subtask 1 retrieval-and-VLM-reranking work recorded in
`docs/EXPERIMENT_LOG.md` (which runs through the date of this review, 2026-09-01). That log is
itself careful — nearly every entry ends with an explicit disclaimer such as *"diagnostic only,"*
*"must not be reported as official Task 1 performance,"* or *"not treated as an official score
because the Task 1 query-level ... contract is not confirmed."* `docs/TASK1_VLM_PLAN.md` §1 states
the claim boundary explicitly: a formal Task 1 result requires the official input records,
report-to-PDF mapping, gold page sets, empty/no-disclosure handling rules, and the official
evaluator; absent any of those, "experiments must be described as a retrieval case study."

**Risk going forward:** the moment this Task 1/Subtask 1 work is added to the paper (which this
revision does, per instruction, as an explicit placeholder-bearing preliminary section — see
`_claude.tex` §4.7), the same care must be taken there. The new section was written to:

1. Never use the words "official," "state of the art," or an unqualified "result" for these numbers.
2. Repeat, in the section's own text (not only a footnote), that the official evaluator and gold
   page-set contract are unconfirmed.
3. Cite `docs/EXPERIMENT_LOG.md` entries by date so the numbers are traceable to a specific,
   reproducible run rather than presented as a single settled figure.
4. Leave an explicit `[PLACEHOLDER]` for the one number that does not exist yet anywhere in the
   repo: an organizer-scored, officially-evaluated Task 1/Subtask 1 metric for this system.

---

## 6. Numbers flagged in the previous review pass — now resolved or superseded

**Update (2026-09-01, follow-up pass):** the fused-baseline discrepancy flagged in the first
review pass has been resolved by newly-completed work, not by this review picking a number.
`docs/EXPERIMENT_LOG.md`'s "Dense retrieval and full VLM reranking (2026-09-01)" entry makes clear
that **0.192** (Hit@1, 369 non-empty groups) and **0.220** (`runs/task1-dense-optimized/metrics-fused.json`,
rounded from 0.2195) were never the same measurement: 0.192 is the fused baseline **without** dense
retrieval, and 0.220 is the same fused baseline **with** the `paraphrase-multilingual-MiniLM-L12-v2`
dense lane enabled — a genuine, documented configuration change (candidate recall was the intended
target of enabling it), not an unreconciled inconsistency. That run has since gone on to complete
its full VLM rerank (490/490 queries, Hit@1 0.328, MRR 0.431) and is now the most recent **complete**
diagnostic in `EXPERIMENT_LOG.md`.

A newer run, `runs/task1-neighbor2` (top-20 candidates, `neighbor_window=2` page expansion, same
dense-enabled fusion), was already **in progress** when this follow-up pass started and remains
in progress (183/490 queries complete as of the 12:06 snapshot cited in `EXPERIMENT_LOG.md`'s
newest entry). This review still did not read `runs/task1-neighbor2/vlm-full20*` directly, in
line with the task instructions each time — all neighbor2 numbers anywhere in this project's docs
or in `_claude.tex` were produced exclusively by the new API-free script
`src/task1/aggregate_neighbor2.py`, which reads only `retrieval.jsonl` and the VLM cache file and
writes its output under `artifacts/metrics/`, never into `runs/`.

**Recommendation, updated:** once `runs/task1-neighbor2` finishes, rerun
`py src/task1/aggregate_neighbor2.py` one more time and treat *that* run as the frozen
configuration for the paper (it methodologically supersedes both the no-dense and the
dense-optimized top-10 runs: dense retrieval, RRF-fused rules, and now a doubled top-20 candidate
pool with neighbor-page expansion). Until then, cite the dense-optimized run's complete numbers as
the primary diagnostic and the neighbor2 snapshot only as a labeled, partial, in-progress data
point — exactly how `_claude.tex` §4.7 now presents them.

---

## 7. Recommendations for the authors

1. ~~Resolve the fused-baseline discrepancy~~ — **done**, see §6 above; it was a genuine
   dense-retrieval-on/off configuration difference, not an error.
2. Once `runs/task1-neighbor2` completes, rerun `src/task1/aggregate_neighbor2.py` and record the
   result as the frozen configuration in `EXPERIMENT_LOG.md`, per `TASK1_VLM_PLAN.md` §16
   ("Definition of done").
3. Confirm the official Subtask 1 evaluator and gold-page-set handling rules (empty-gold credit,
   multi-page gold sets) with the task organizers before reporting any Task 1 number as anything
   other than diagnostic — this is the single blocking item per `TASK1_VLM_PLAN.md` §1, and remains
   unresolved.
4. Consider explicitly auditing whether the non-Chinese CSV split used for Subtask 2 shares
   document identities across train/test, the way CSCU did for the official Subtask 1 split
   (§3.1 above). If it does, the current "793 non-Chinese rows" framing should say so, the same way
   CSCU now frames its own numbers as "seen-document, unseen-query."
5. A stratified error-annotation sample now exists (`docs/TASK1_ERROR_REVIEW_SAMPLE.csv`, 65 rows,
   built by `src/task1/build_error_review_sample.py`), but it currently only has Chinese and
   English coverage because `runs/task1-neighbor2` hadn't reached the other four languages yet at
   sampling time. Rerun the same script after the run completes for full 6-language stratified
   coverage, then have at least one human annotator fill in the (currently blank) error-taxonomy
   columns before treating the taxonomy as quantified rather than qualitative.
6. See `docs/CLAUDE_ERROR_ANALYSIS.md` for the consolidated, cross-method error taxonomy and
   concrete diagnostic examples, and `docs/FINAL_SUBMISSION_CHECKLIST.md` /
   `docs/FINAL_REPRODUCIBILITY.md` for the current submission-readiness state.
