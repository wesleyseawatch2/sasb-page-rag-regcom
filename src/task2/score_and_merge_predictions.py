from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from glm_rag_predict_english import compact, evaluate, key, read_csv, write_csv


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "datasets" / "all" / "test_answer_sheet.csv")
    parser.add_argument("--truth", type=Path, default=ROOT / "data" / "datasets" / "all" / "test_truth.csv")
    parser.add_argument("--out-predictions", type=Path, required=True)
    parser.add_argument("--out-metrics", type=Path, required=True)
    parser.add_argument("--exclude-langs", nargs="*", default=["chinese"])
    args = parser.parse_args()

    exclude = {lang.strip().lower() for lang in args.exclude_langs if lang.strip()}
    pred_rows = []
    sources = []
    for path in args.predictions:
        rows = read_csv(path)
        pred_rows.extend(rows)
        sources.append(
            {
                "path": str(path),
                "rows": len(rows),
                "langs": dict(Counter(compact(row.get("lang")).lower() for row in rows)),
            }
        )

    test_rows = [
        row for row in read_csv(args.test)
        if compact(row.get("lang")).lower() not in exclude
    ]
    truth_rows = [
        row for row in read_csv(args.truth)
        if compact(row.get("lang")).lower() not in exclude
    ]

    counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    pred_by_occurrence = {}
    for row in pred_rows:
        if compact(row.get("lang")).lower() in exclude:
            continue
        base = key(row)
        occurrence = counts[base]
        counts[base] += 1
        pred_by_occurrence[(*base, occurrence)] = row

    ordered_rows = []
    missing = []
    extra_keys = set(pred_by_occurrence)
    test_counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    for test_row in test_rows:
        base = key(test_row)
        occurrence = test_counts[base]
        test_counts[base] += 1
        lookup_key = (*base, occurrence)
        pred = pred_by_occurrence.get(lookup_key)
        if pred is None:
            missing.append(lookup_key)
            continue
        out_row = dict(test_row)
        out_row["pred_label"] = pred.get("pred_label", "")
        out_row["category_match"] = pred.get("category_match", "")
        out_row["unit_match"] = pred.get("unit_match", "")
        ordered_rows.append(out_row)
        extra_keys.discard(lookup_key)

    fieldnames = list(test_rows[0].keys()) if test_rows else []
    for field in ["pred_label", "category_match", "unit_match"]:
        if field not in fieldnames:
            fieldnames.append(field)
    write_csv(args.out_predictions, ordered_rows, fieldnames)

    metrics = evaluate(ordered_rows, truth_rows)
    by_lang = {}
    for lang in sorted({compact(row.get("lang")).lower() for row in ordered_rows}):
        pred_subset = [row for row in ordered_rows if compact(row.get("lang")).lower() == lang]
        truth_subset = [row for row in truth_rows if compact(row.get("lang")).lower() == lang]
        lang_metrics = evaluate(pred_subset, truth_subset)
        by_lang[lang] = {
            "evaluated": lang_metrics["evaluated"],
            "accuracy": round(lang_metrics["accuracy"], 4),
            "micro_f1": round(lang_metrics["micro_f1"], 4),
            "macro_f1": round(lang_metrics["macro_f1"], 4),
        }

    metrics.update(
        {
            "sources": sources,
            "expected_truth_rows": len(truth_rows),
            "written_prediction_rows": len(ordered_rows),
            "missing_count": len(missing),
            "extra_count": len(extra_keys),
            "by_lang": by_lang,
            "predictions_path": str(args.out_predictions),
        }
    )
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "evaluated": metrics["evaluated"],
            "accuracy": round(metrics["accuracy"], 4),
            "micro_f1": round(metrics["micro_f1"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "missing_count": len(missing),
            "extra_count": len(extra_keys),
            "out_predictions": str(args.out_predictions),
            "out_metrics": str(args.out_metrics),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
