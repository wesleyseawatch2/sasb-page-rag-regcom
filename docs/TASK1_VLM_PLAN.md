# Task 1 VLM: complete implementation and research plan

## 1. Objective and claim boundary

The goal is an end-to-end RegCom system that receives a complete ESG report and
a SASB metric query, retrieves the supporting page or page set, and predicts
`yes`, `yes but not complete`, or `no` using traceable evidence.

The current repository does **not** yet claim a Task 1 result. A formal result
requires the official Task 1 input records, report-to-PDF mapping, gold relevant
page sets, handling rules for empty/no-disclosure instances, and the official
evaluation script. If any item is unavailable, experiments must be described as
a retrieval case study rather than an official Task 1 submission.

## 2. Research questions

- **RQ1:** How much do lexical, semantic, structural, and visual signals each
  contribute to evidence-page recall?
- **RQ2:** Does VLM reranking improve exact-page retrieval after controlling for
  the candidate set?
- **RQ3:** How strongly do page-retrieval errors propagate into the existing
  Subtask 2 compliance verifier?
- **RQ4:** Do Top-3 or neighboring-page evidence windows improve end-to-end
  verification without producing excessive false positives?
- **RQ5:** Which failures arise from language, PDF extraction quality, tables,
  document structure, or ambiguous gold pages?

## 3. System architecture

```text
Full PDF + SASB query
  -> page rendering and text extraction
  -> OCR/table fallback and quality metadata
  -> ToC detection and section grounding
  -> parallel candidate generators
       BM25/word + character lexical retrieval
       multilingual dense retrieval
       metric-code and unit/table matching
       optional visual page embeddings
  -> reciprocal-rank fusion
  -> lane-union candidate pool and adjacent-page expansion
  -> VLM reranking of Top-10 candidates
  -> Top-1 and Top-3 evidence sets
  -> structured evidence extraction
  -> existing Task 2 completeness verifier
  -> deterministic validation, confidence, and trace
```

The VLM must not scan every report page during the main run. Cheap local
retrievers reduce each report to Top-10 candidates; the VLM reranks only those
candidates. This is compatible with a 4 GB RTX 3050 for small local models and
with API-based VLMs for the final experiment.

## 4. Data contract

### Query record

```json
{
  "sample_id": "stable-unique-id",
  "report_id": "report-id",
  "lang": "english",
  "metric_code": "CG-AA-430b.1",
  "topic": "Product Sourcing, Packaging & Marketing",
  "metric": "metric description",
  "expected_value": "expected disclosure",
  "expected_unit": "unit",
  "pdf_path": "data/private/reports/report.pdf"
}
```

### Page record

```json
{
  "report_id": "report-id",
  "pdf_page_index": 37,
  "printed_page": "35",
  "text": "extracted text",
  "table_text": "optional markdown",
  "image_path": "cache/pages/report/0037.webp",
  "char_count": 3281,
  "extraction_method": "pymupdf",
  "quality_flags": []
}
```

The PDF page index, printed page number, and official page identifier must be
stored separately. Cover sheets frequently create offsets.

### Prediction record

```json
{
  "sample_id": "stable-unique-id",
  "pred_pages": [37, 38],
  "pred_label": "yes but not complete",
  "confidence": 0.78,
  "retrieval_trace": {},
  "evidence": [],
  "model_config_id": "vlm-rerank-v1"
}
```

## 5. Stage A: document ingestion

1. Validate each report identifier and PDF hash.
2. Extract page text with PyMuPDF.
3. Render pages to WebP or JPEG at a controlled resolution.
4. Use OCR only when normalized extracted text is empty or demonstrably poor.
5. Preserve tables as Markdown or cell-level JSON when available.
6. Record character count, numeric-token count, image coverage, OCR status, and
   table-likelihood flags.
7. Cache by PDF SHA-256 plus page index so repaired PDFs invalidate old output.

No source report or rendered page is committed to the public repository.

## 6. Stage B: ToC and structure grounding

Search the first 20 pages for multilingual ToC indicators and title/page-number
patterns. Build section ranges and ask a VLM or text model to rank section titles
against the SASB query. ToC grounding contributes a prior only; a full-document
fallback remains mandatory because some reports lack a recoverable ToC.

Store the detected ToC page, parsed entries, selected sections, confidence, and
fallback reason. This permits separate evaluation of ToC quality.

## 7. Stage C: candidate generation

Run independent candidate generators:

1. **BM25:** topic, metric, metric code, expected value, expected unit, synonyms,
   and translated query terms against page text.
2. **Dense retrieval:** multilingual embeddings over paragraphs, tables, and page
   summaries; aggregate the best chunk scores into a page score.
3. **Metric/unit rules:** exact or normalized metric codes, units, quantities,
   denominator terms, and zero-event expressions.
4. **Visual embeddings (optional ablation):** page-image embeddings for layouts,
   figures, and tables not represented faithfully in text.

Each lane writes a complete ranked list before fusion. Never overwrite lane
outputs with the fused ranking.

## 8. Stage D: reciprocal-rank fusion

Use weighted reciprocal-rank fusion rather than mixing incompatible raw scores:

```text
RRF(page) = sum_r weight_r / (k + rank_r(page))
```

Start with `k=60`, equal BM25/dense weights, and smaller ToC/rule priors. Tune
weights only on training/development queries. Add the immediate neighbors of
high-ranked pages and every page containing an exact metric code. Retain Top-10
unique candidates for VLM reranking.

## 9. Stage E: VLM reranking

For each candidate provide the full-page image, extracted text, query metadata,
and neighboring-page titles. Require JSON output with:

- relevance score;
- evidence type: `direct`, `partial`, `index_reference`, `topical_only`, or
  `unrelated`;
- value/unit/scope presence;
- a short grounded reason;
- ranked page identifiers;
- `no_relevant_page` decision.

### Execution choices

- **Primary experiment:** an API VLM for reliable multimodal reasoning.
- **Open baseline:** a small quantized local VLM, batch size 1, low-resolution
  page input, used only for Top-10 reranking.
- **Do not use:** a local 7B+ VLM on the current 4 GB GPU for full-report scans.

All responses are cached. Record provider, exact model string, date, endpoint,
temperature, top-p, output-token limit, image resolution, retry count, latency,
token usage, and measured cost when available.

## 10. Stage F: verification and aggregation

Evaluate both Top-1 and Top-3 evidence modes. First extract structured evidence:

- direct metric relevance;
- values and units;
- scope and reporting period;
- denominator and disaggregation;
- index-page-only flag;
- evidence text and page location.

Then call the existing Task 2 verifier. `yes` requires direct and sufficiently
complete evidence; `yes but not complete` requires relevant but incomplete or
proxy evidence; `no` requires absence of metric-specific evidence. A low-score
candidate set must be allowed to abstain with an empty page list and `no` label.

Save raw model output, normalized output, rule flags, and final output separately.

## 11. Required experiments

### Retrieval ablation

| ID | System |
|---|---|
| R0 | BM25 only |
| R1 | Dense only |
| R2 | BM25 + dense RRF |
| R3 | R2 + metric/unit rules |
| R4 | R3 + ToC prior |
| R5 | R4 + visual embeddings |
| R6 | R4/R5 + VLM reranking |

### End-to-end ablation

| ID | Evidence supplied to Task 2 |
|---|---|
| E0 | Gold page: verification upper reference |
| E1 | BM25 Top-1 |
| E2 | Hybrid Top-1 |
| E3 | VLM-reranked Top-1 |
| E4 | VLM-reranked Top-3 |
| E5 | Top-1 plus adjacent pages |

Do not use test truth to choose thresholds, candidates, prompts, or rows for
review. Diagnostic gold-guided post-review must be labeled separately and never
reported as a blind result.

## 12. Evaluation

### Retrieval

- Hit@1, Hit@5, and Hit@10;
- mean reciprocal rank;
- candidate recall before VLM reranking;
- Near@1 for adjacent-page predictions;
- exact-set precision, recall, and Macro F1 under official rules;
- non-empty-gold performance and empty-gold accuracy reported separately.

If multiple pages are valid, gold must be represented as a set and any valid
member should count as a hit where the official protocol permits.

### Verification and end-to-end

- accuracy/Micro F1 and Macro F1;
- per-class precision, recall, and F1;
- confusion matrix;
- per-language and per-metric breakdowns;
- paired bootstrap confidence intervals;
- McNemar tests for paired label predictions;
- latency, API calls, token usage, and measured cost.

Use a stable unique `sample_id` or strict positional alignment. Never join solely
on a known non-unique tuple.

## 13. Error analysis taxonomy

Annotate a stratified sample of at least 50--80 failures:

- candidate-generation miss;
- wrong ToC section;
- exact page vs adjacent page;
- SASB index/reference page;
- evidence spanning pages;
- topical mention without metric evidence;
- table extraction or reading-order failure;
- OCR failure;
- missing value, unit, denominator, scope, period, or disaggregation;
- full vs partial boundary;
- ambiguous or conflicting gold;
- model JSON/normalization failure.

Report counts, representative cases, and ideally two annotators plus agreement.

## 14. Reproducibility requirements

Every run receives a unique run ID and writes:

```text
runs/<run-id>/
  config.json
  environment.txt
  input_manifest.csv
  retrieval/*.jsonl
  reranking/*.jsonl
  predictions.csv
  metrics.json
  timing.csv
  errors.jsonl
  README.md
```

The configuration must include Git commit, data hashes, prompt hashes, model
identities, random seeds where meaningful, and all thresholds. See
`EXPERIMENT_LOG.md` for the human-readable record.

## 15. Implementation order

1. Confirm Task 1 data contract and official evaluator.
2. Implement ingestion, page-number mapping, and caching.
3. Implement BM25 and dense baselines.
4. Add fusion, ToC prior, and unit/table rules.
5. Evaluate candidate recall before spending VLM calls.
6. Add cached Top-10 VLM reranking.
7. Connect Top-1/Top-3 results to the existing Task 2 verifier.
8. Run controlled ablations and statistical tests.
9. Complete stratified error analysis.
10. Update the paper only with results reproducible from committed configs and
    public-safe artifacts.

## 16. Definition of done

Task 1 is complete only when:

- the official input/gold mapping is documented;
- every prediction maps to the intended PDF and page numbering scheme;
- baseline and VLM configurations are frozen and reproducible;
- retrieval and end-to-end metrics are generated automatically;
- no test-gold-guided selection occurs in the blind pipeline;
- error analysis and cost/latency are reported;
- public artifacts exclude restricted PDFs and secrets;
- the paper distinguishes official, anomaly-filtered, and diagnostic results.
