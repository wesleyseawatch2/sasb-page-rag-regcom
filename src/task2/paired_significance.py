"""Paired significance tests for Task 2 model predictions.

The test is deliberately prediction-level rather than aggregate-only: model
CSV files are joined by the task row identity, and each bootstrap resample
contains the same rows for both systems.  This script does not call a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


LABELS = ("yes", "yes but not complete", "no")
KEY_FIELDS = (
    "lang",
    "cid",
    "topic",
    "metric_description",
    "metric_code",
    "page",
    "file_stem",
)


def canonical_label(value: str) -> str:
    value = (value or "").strip().lower().replace("_", " ")
    if value in {"yes but not complete", "yes-but-not-complete", "partial"}:
        return "yes but not complete"
    return value


def read_predictions(path: Path) -> dict[tuple[str, ...], tuple[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[tuple[str, ...], tuple[str, str]] = {}
    occurrences: dict[tuple[str, ...], int] = {}
    for row in rows:
        base_key = tuple((row.get(field) or "").strip() for field in KEY_FIELDS)
        occurrence = occurrences.get(base_key, 0)
        occurrences[base_key] = occurrence + 1
        # Some official rows are exact duplicates.  Preserve their stable
        # occurrence order instead of collapsing them during a paired join.
        key = base_key + (str(occurrence),)
        if key in result:
            raise ValueError(f"duplicate row key in {path}: {key}")
        truth = canonical_label(row.get("label", ""))
        pred = canonical_label(row.get("pred_label", ""))
        result[key] = (truth, pred)
    return result


def macro_f1(truth: Iterable[str], pred: Iterable[str]) -> float:
    truth = list(truth)
    pred = list(pred)
    scores = []
    for label in LABELS:
        tp = sum(t == label and p == label for t, p in zip(truth, pred))
        fp = sum(t != label and p == label for t, p in zip(truth, pred))
        fn = sum(t == label and p != label for t, p in zip(truth, pred))
        denominator = 2 * tp + fp + fn
        scores.append((2 * tp / denominator) if denominator else 0.0)
    return sum(scores) / len(scores)


def exact_mcnemar_p(discordant_a: int, discordant_b: int) -> float:
    """Two-sided exact McNemar p-value using the smaller tail."""
    n = discordant_a + discordant_b
    if n == 0:
        return 1.0
    k = min(discordant_a, discordant_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def bootstrap_pair(
    truth: list[str],
    pred_a: list[str],
    pred_b: list[str],
    seed: int = 20260901,
    resamples: int = 10000,
) -> dict[str, list[float] | float]:
    import numpy as np

    n = len(truth)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, n, size=(resamples, n))
    truth_array = np.asarray(truth)
    a_array = np.asarray(pred_a)
    b_array = np.asarray(pred_b)
    accuracy_deltas = (
        (truth_array[samples] == a_array[samples]).mean(axis=1)
        - (truth_array[samples] == b_array[samples]).mean(axis=1)
    )
    macro_deltas = np.zeros(resamples, dtype=float)
    for label in LABELS:
        truth_label = truth_array == label
        a_label = a_array == label
        b_label = b_array == label
        t = truth_label[samples]
        a = a_label[samples]
        b = b_label[samples]
        tp_a = (t & a).sum(axis=1)
        fp_a = (~t & a).sum(axis=1)
        fn_a = (t & ~a).sum(axis=1)
        tp_b = (t & b).sum(axis=1)
        fp_b = (~t & b).sum(axis=1)
        fn_b = (t & ~b).sum(axis=1)
        den_a = 2 * tp_a + fp_a + fn_a
        den_b = 2 * tp_b + fp_b + fn_b
        f1_a = np.divide(2 * tp_a, den_a, out=np.zeros(resamples), where=den_a != 0)
        f1_b = np.divide(2 * tp_b, den_b, out=np.zeros(resamples), where=den_b != 0)
        macro_deltas += (f1_a - f1_b) / len(LABELS)

    def interval(values) -> list[float]:
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]

    return {
        "accuracy_delta": float(np.mean(accuracy_deltas)),
        "accuracy_ci95": interval(accuracy_deltas),
        "macro_f1_delta": float(np.mean(macro_deltas)),
        "macro_f1_ci95": interval(macro_deltas),
        "resamples": resamples,
        "seed": seed,
    }


def compare(name_a: str, rows_a: dict[tuple[str, ...], tuple[str, str]], name_b: str, rows_b: dict[tuple[str, ...], tuple[str, str]]) -> dict:
    keys = sorted(set(rows_a) & set(rows_b))
    if len(keys) != len(rows_a) or len(keys) != len(rows_b):
        raise ValueError(f"row mismatch for {name_a} vs {name_b}: {len(rows_a)} vs {len(rows_b)}; overlap {len(keys)}")
    truth = [rows_a[key][0] for key in keys]
    pred_a = [rows_a[key][1] for key in keys]
    pred_b = [rows_b[key][1] for key in keys]
    if any(label not in LABELS for label in truth):
        raise ValueError("unexpected gold label")
    a_only = sum(t == a and t != b for t, a, b in zip(truth, pred_a, pred_b))
    b_only = sum(t != a and t == b for t, a, b in zip(truth, pred_a, pred_b))
    return {
        "n": len(keys),
        "model_a": name_a,
        "model_b": name_b,
        "accuracy_a": sum(t == a for t, a in zip(truth, pred_a)) / len(keys),
        "accuracy_b": sum(t == b for t, b in zip(truth, pred_b)) / len(keys),
        "macro_f1_a": macro_f1(truth, pred_a),
        "macro_f1_b": macro_f1(truth, pred_b),
        "mcnemar": {
            "a_only_correct": a_only,
            "b_only_correct": b_only,
            "p_two_sided_exact": exact_mcnemar_p(a_only, b_only),
        },
        "paired_bootstrap": bootstrap_pair(truth, pred_a, pred_b),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True, help="CSV containing the gold label column")
    parser.add_argument("--model", action="append", nargs=2, metavar=("NAME", "CSV"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    truth_rows = read_predictions(args.truth)
    loaded = [(name, read_predictions(Path(path))) for name, path in args.model]
    # Prediction exports intentionally omit the gold label.  Join it from the
    # shared task dataset using the same occurrence-aware row identity.
    prediction_keys = set().union(*(set(rows) for _, rows in loaded))
    missing_truth = prediction_keys - set(truth_rows)
    if missing_truth:
        raise ValueError(f"{len(missing_truth)} prediction rows are absent from the truth CSV")
    truth_rows = {key: value for key, value in truth_rows.items() if key in prediction_keys}
    for _, rows in loaded:
        if set(rows) != set(truth_rows):
            raise ValueError(f"truth/prediction row mismatch: {len(truth_rows)} vs {len(rows)}")
        for key, (truth, pred) in list(rows.items()):
            rows[key] = (truth_rows[key][0], pred)
    comparisons = []
    for i, (name_a, rows_a) in enumerate(loaded):
        for name_b, rows_b in loaded[i + 1 :]:
            comparisons.append(compare(name_a, rows_a, name_b, rows_b))
    # Holm adjustment controls the family-wise error rate across the six
    # pairwise model comparisons while preserving their paired design.
    ordered = sorted(range(len(comparisons)), key=lambda i: comparisons[i]["mcnemar"]["p_two_sided_exact"])
    adjusted = [1.0] * len(comparisons)
    running = 0.0
    for rank, index in enumerate(ordered):
        raw = comparisons[index]["mcnemar"]["p_two_sided_exact"]
        corrected = min(1.0, raw * (len(comparisons) - rank))
        running = max(running, corrected)
        adjusted[index] = running
    for index, comparison in enumerate(comparisons):
        comparison["mcnemar"]["p_holm"] = adjusted[index]
        comparison["mcnemar"]["significant_after_holm_alpha_0.05"] = adjusted[index] < 0.05
    output = {
        "description": "Paired Task 2 model comparisons on identical row identities; exact McNemar and paired bootstrap.",
        "key_fields": list(KEY_FIELDS),
        "comparisons": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
