"""Aggregate per-language end-to-end Task 1 diagnostic outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


INPUT_RATE = 0.20
OUTPUT_RATE = 1.25


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def aggregate(rows: list[dict[str, Any]], cache_paths: list[Path]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("trace_status") == "complete"]
    nonempty = [row for row in completed if row.get("gold_pages")]
    scored = [row for row in completed if row.get("gold_page_labels")]
    correct = sum(row.get("pred_label") in set(row.get("gold_page_labels", [])) for row in scored)
    hits = sum(bool(row.get("retrieval_hit")) for row in nonempty)
    input_tokens = output_tokens = 0
    latencies: list[float] = []
    for path in cache_paths:
        for trace in read_jsonl(path):
            usage = trace.get("usage", {})
            input_tokens += int(usage.get("input_tokens", 0) or 0)
            output_tokens += int(usage.get("output_tokens", 0) or 0)
            latencies.append(float(trace.get("latency_seconds", 0.0) or 0.0))

    def language_metrics(language_rows: list[dict[str, Any]]) -> dict[str, Any]:
        done = [row for row in language_rows if row.get("trace_status") == "complete"]
        positive = [row for row in done if row.get("gold_pages")]
        label_scored = [row for row in done if row.get("gold_page_labels")]
        return {
            "queries": len(language_rows),
            "completed": len(done),
            "retrieval_hit_at_1": sum(bool(row.get("retrieval_hit")) for row in positive) / len(positive) if positive else 0.0,
            "selected_page_label_scored": len(label_scored),
            "selected_page_label_accuracy": (
                sum(row.get("pred_label") in set(row.get("gold_page_labels", [])) for row in label_scored) / len(label_scored)
                if label_scored else 0.0
            ),
            "pred_label_counts": dict(Counter(row.get("pred_label", "") for row in done)),
        }

    per_language = {
        language: language_metrics([row for row in rows if row.get("lang") == language])
        for language in sorted({str(row.get("lang", "")) for row in rows})
    }
    return {
        "model": "gpt-5.4-nano",
        "queries": len(rows),
        "completed": len(completed),
        "errors": len(rows) - len(completed),
        "retrieval_hit_at_1": hits / len(nonempty) if nonempty else 0.0,
        "selected_page_label_scored": len(scored),
        "selected_page_label_accuracy": correct / len(scored) if scored else 0.0,
        "pred_label_counts": dict(Counter(row.get("pred_label", "") for row in completed)),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": input_tokens / 1_000_000 * INPUT_RATE + output_tokens / 1_000_000 * OUTPUT_RATE,
        "latency_mean_seconds": statistics.fmean(latencies) if latencies else 0.0,
        "latency_p95_seconds": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0.0,
        "per_language": per_language,
        "note": "Diagnostic proxy; official Task 1 query-level gold/evaluator is not confirmed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, nargs="+", required=True)
    parser.add_argument("--caches", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for path in args.outputs:
        rows.extend(read_jsonl(path))
    rows.sort(key=lambda row: (str(row.get("lang", "")), str(row.get("sample_id", ""))))
    write_jsonl(args.output, rows)
    metrics = aggregate(rows, args.caches)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
