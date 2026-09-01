"""Build a stratified, human-review-only error-analysis sample (no self-annotated answers).

Reads artifacts/metrics/task1_neighbor2_per_query.csv (produced by aggregate_neighbor2.py; no API
calls, no runs/ writes) and stratified-samples across (language, mechanical transition category)
cells to a target sample size (default 65, inside the 50-80 range requested in
docs/TASK1_VLM_PLAN.md Section 13). The "category" column is a mechanically derived Hit@1
transition label (candidate_recall_miss / vlm_improved_top1 / vlm_degraded_top1 / both_top1 /
neither_top1 / empty_gold) -- a fact about the data, not a judgment call. The qualitative
error-taxonomy columns (Section 13 of the plan) are left BLANK for a human annotator; this script
never fills them in.

Usage:
    py src/task1/build_error_review_sample.py
    py src/task1/build_error_review_sample.py --target-size 65 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

# From docs/TASK1_VLM_PLAN.md Section 13. Provided as a reference list only (as a comment row and
# in the accompanying docs), never used to pre-fill an answer.
ERROR_TAXONOMY = [
    "candidate_generation_miss",
    "wrong_toc_section",
    "exact_vs_adjacent_page",
    "sasb_index_reference_page",
    "evidence_spanning_pages",
    "topical_mention_without_metric_evidence",
    "table_extraction_or_reading_order_failure",
    "ocr_failure",
    "missing_value_unit_denominator_scope_period_disaggregation",
    "full_vs_partial_boundary",
    "ambiguous_or_conflicting_gold",
    "model_json_or_normalization_failure",
    "other",
]

ANNOTATION_COLUMNS = [
    "annotator_initials",
    "error_taxonomy_category",  # pick one of ERROR_TAXONOMY; left blank here
    "is_prediction_actually_correct",  # human judgment; left blank here
    "correct_page_if_different_from_gold",
    "notes",
]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stratified_sample(rows: list[dict[str, str]], target_size: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        cells[(row["lang"], row["category"])].append(row)

    total = len(rows)
    if total == 0:
        return []

    quotas: dict[tuple[str, str], int] = {}
    for key, cell_rows in cells.items():
        share = len(cell_rows) / total
        quotas[key] = max(1, math.floor(target_size * share))

    selected: list[dict[str, str]] = []
    for key, cell_rows in cells.items():
        quota = min(quotas[key], len(cell_rows))
        selected.extend(rng.sample(cell_rows, quota))

    if len(selected) > target_size:
        selected = rng.sample(selected, target_size)
    elif len(selected) < target_size:
        remaining = [row for cell_rows in cells.values() for row in cell_rows if row not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: target_size - len(selected)])

    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("artifacts/metrics/task1_neighbor2_per_query.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("docs/TASK1_ERROR_REVIEW_SAMPLE.csv"))
    parser.add_argument("--target-size", type=int, default=65)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_rows(args.input)
    sample = stratified_sample(rows, args.target_size, args.seed)

    context_columns = [
        "sample_id",
        "lang",
        "report_stem",
        "topic",
        "metric_code",
        "gold_pages",
        "category",
        "candidate_recall_at10",
        "fused_top5",
        "vlm_top5",
        "no_relevant_page",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [f"# valid error_taxonomy_category values: {'|'.join(ERROR_TAXONOMY)}"]
        )
        writer.writerow(context_columns + ANNOTATION_COLUMNS)
        for row in sample:
            writer.writerow([row.get(col, "") for col in context_columns] + [""] * len(ANNOTATION_COLUMNS))

    by_cell: dict[tuple[str, str], int] = defaultdict(int)
    for row in sample:
        by_cell[(row["lang"], row["category"])] += 1
    print(f"sampled {len(sample)} / {len(rows)} available rows -> {args.output}")
    for key in sorted(by_cell):
        print(f"  {key[0]:>10s} / {key[1]:<22s} {by_cell[key]}")


if __name__ == "__main__":
    main()
