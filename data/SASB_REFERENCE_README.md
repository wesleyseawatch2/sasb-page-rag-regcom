# SASB Reference Used by This Package

This lightweight package does not include the full original SASB Standards documents.

In this experiment, SASB requirements are provided as structured row-level metadata in the dataset CSV files:

```text
data/datasets/all_subtask2_answer_sheet.csv
data/datasets/all_subtask2_dataset.csv
```

The pipeline reads the following SASB-related fields from each row:

```text
topic
metric_code
metric_description
sasb_category
sasb_unit_of_measure
sasb_key_terms
sasb_what_counts
```

These fields define the metric intent, disclosure category, required unit of measure, key terms, and what counts as sufficient disclosure for the given page-level verification task.

## Inference Setting

During inference, the system uses:

```text
CSV row-level SASB metadata
+ single-page PDF text
```

It does not retrieve from the full SASB Standards PDF or an external SASB database.

## Paper Wording

Recommended wording:

```text
The SASB requirements are provided as structured row-level metadata, including metric description, disclosure category, unit of measure, key terms, and what-counts specification. The system does not retrieve from the full SASB standards document during inference.
```

Chinese explanation:

```text
SASB 規定以結構化欄位形式提供在每筆資料中，包括 metric description、category、unit、key terms 和 what_counts；系統推論時沒有另外查完整 SASB 標準文件。
```

## Optional Extension

If full SASB Standards documents are later added, they can be placed in this folder and connected to the OrchestratorAgent through a retrieval stage. That extension is not part of the current final experiment.

