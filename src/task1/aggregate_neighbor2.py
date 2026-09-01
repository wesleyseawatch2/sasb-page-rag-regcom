"""API-free aggregator for the task1-neighbor2 (top-20, neighbor-expanded) VLM rerank run.

Reads only cached/local files under a run directory (default: runs/task1-neighbor2) and the
task2-style pricing convention already used in docs/EXPERIMENT_LOG.md. Makes no network or API
calls. Never writes into the run directory itself -- all outputs go under artifacts/metrics/, so
this script is safe to run repeatedly while another process is still appending to the VLM cache
file.

The ranking reconstruction (candidate_key -> page, order preserved, omitted candidates appended
in original candidate order) and the metric formulas (hit@k, reciprocal rank, near@1 = top-1
within one page of any gold page) are reimplemented here to exactly match
`normalize_vlm_ranking` / `hit_at` / `reciprocal_rank` / `command_evaluate` in
src/task1/task1_pipeline.py (read for reference only; not imported or modified, and not executed).

Usage:
    py src/task1/aggregate_neighbor2.py
    py src/task1/aggregate_neighbor2.py --run-dir runs/task1-neighbor2 --expected-total 490
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Same nano rate convention already used in docs/EXPERIMENT_LOG.md ("End-to-end VLM label
# diagnostic" entry: "$0.20/M input, $1.25/M output"). Not independently re-verified against a
# live pricing page here (that would require a network call); flagged as an approximation that
# ignores any cached-token discount.
NANO_INPUT_USD_PER_M = 0.20
NANO_OUTPUT_USD_PER_M = 1.25


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unparsed = 0
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A concurrently-written file can have a truncated final line; skip it rather
                # than crash, since a rerun after the writer finishes will pick it up cleanly.
                unparsed += 1
    if unparsed:
        print(f"warning: skipped {unparsed} unparsed trailing line(s) in {path}")
    return rows


def hit_at(predicted: list[int], gold: set[int], k: int) -> float:
    return float(bool(set(predicted[:k]) & gold))


def reciprocal_rank(predicted: list[int], gold: set[int]) -> float:
    for rank, page in enumerate(predicted, 1):
        if page in gold:
            return 1.0 / rank
    return 0.0


def near_at_1(predicted: list[int], gold: set[int]) -> float:
    return float(bool(predicted and any(abs(predicted[0] - page) <= 1 for page in gold)))


def normalize_vlm_ranking(ranked: list[dict[str, Any]], candidate_pages: list[int]) -> list[int]:
    """Reproduces task1_pipeline.normalize_vlm_ranking, returning a flat page-order list."""
    key_to_page = {f"C{index:02d}": page for index, page in enumerate(candidate_pages, 1)}
    allowed = set(candidate_pages)
    ordered: list[int] = []
    seen: set[int] = set()
    for item in ranked:
        key = str(item.get("candidate_key", "")).upper()
        if key in key_to_page:
            page = key_to_page[key]
        else:
            try:
                page = int(item["page"])
            except (KeyError, TypeError, ValueError):
                continue
        if page not in allowed or page in seen:
            continue
        seen.add(page)
        ordered.append(page)
    for page in candidate_pages:
        if page not in seen:
            ordered.append(page)
    return ordered


def load_retrieval(path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if not sample_id:
            continue
        by_id[sample_id] = {
            "lang": row.get("lang", ""),
            "cid": row.get("cid", ""),
            "report_stem": row.get("report_stem", ""),
            "topic": row.get("topic", ""),
            "metric_code": row.get("metric_code", ""),
            "gold_pages": sorted({int(p) for p in row.get("gold_pages", [])}),
            "fused_pages": [int(item["page"]) for item in row.get("fused", [])],
        }
    return by_id


def load_vlm_cache(paths: list[Path]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            sample_id = row.get("sample_id")
            if sample_id:
                by_id[sample_id] = row  # later occurrence wins, matching the pipeline's own cache merge
    return by_id


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def build_summary(
    retrieval: dict[str, dict[str, Any]],
    vlm_cache: dict[str, dict[str, Any]],
    expected_total: int,
) -> dict[str, Any]:
    per_query: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    pending_sample_ids: list[str] = []
    latencies: list[float] = []
    usage_totals = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
    model_ids: Counter[str] = Counter()

    for sample_id, meta in retrieval.items():
        cache_row = vlm_cache.get(sample_id)
        if cache_row is None:
            pending_sample_ids.append(sample_id)
            continue
        status = cache_row.get("status")
        if status == "error":
            errors.append(
                {
                    "sample_id": sample_id,
                    "lang": meta["lang"],
                    "error": cache_row.get("error", "unknown error"),
                }
            )
            continue
        if status != "complete":
            pending_sample_ids.append(sample_id)
            continue

        candidate_pages = [int(p) for p in cache_row.get("candidate_pages", [])]
        ranked = cache_row.get("prediction", {}).get("ranked_pages", [])
        vlm_pages = normalize_vlm_ranking(ranked, candidate_pages)
        no_relevant_page = bool(cache_row.get("prediction", {}).get("no_relevant_page"))

        gold = set(meta["gold_pages"])
        fused_pages = meta["fused_pages"]
        fused_h1 = hit_at(fused_pages, gold, 1) if gold else None
        vlm_h1 = hit_at(vlm_pages, gold, 1) if gold else None
        recall10 = bool(gold & set(fused_pages[:10])) if gold else None
        if not gold:
            category = "empty_gold"
        elif not recall10:
            category = "candidate_recall_miss"
        elif not fused_h1 and vlm_h1:
            category = "vlm_improved_top1"
        elif fused_h1 and not vlm_h1:
            category = "vlm_degraded_top1"
        elif fused_h1 and vlm_h1:
            category = "both_top1"
        else:
            category = "neither_top1"

        latencies.append(float(cache_row.get("latency_seconds", 0.0) or 0.0))
        usage = cache_row.get("usage", {}) or {}
        usage_totals["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
        usage_totals["cached_tokens"] += int(
            (usage.get("input_tokens_details", {}) or {}).get("cached_tokens", 0) or 0
        )
        usage_totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        model_ids[str(cache_row.get("model", "unknown"))] += 1

        row = {
            "sample_id": sample_id,
            "lang": meta["lang"],
            "report_stem": meta["report_stem"],
            "topic": meta["topic"],
            "metric_code": meta["metric_code"],
            "gold_pages": "|".join(map(str, meta["gold_pages"])),
            "has_gold": bool(gold),
            "candidate_recall_at10": recall10,
            "fused_top1": fused_pages[0] if fused_pages else None,
            "fused_top5": "|".join(map(str, fused_pages[:5])),
            "fused_hit_at_1": fused_h1,
            "fused_hit_at_5": hit_at(fused_pages, gold, 5) if gold else None,
            "fused_hit_at_10": hit_at(fused_pages, gold, 10) if gold else None,
            "fused_near_at_1": near_at_1(fused_pages, gold) if gold else None,
            "fused_reciprocal_rank": reciprocal_rank(fused_pages, gold) if gold else None,
            "vlm_top1": vlm_pages[0] if vlm_pages else None,
            "vlm_top5": "|".join(map(str, vlm_pages[:5])),
            "vlm_hit_at_1": vlm_h1,
            "vlm_hit_at_5": hit_at(vlm_pages, gold, 5) if gold else None,
            "vlm_hit_at_10": hit_at(vlm_pages, gold, 10) if gold else None,
            "vlm_near_at_1": near_at_1(vlm_pages, gold) if gold else None,
            "vlm_reciprocal_rank": reciprocal_rank(vlm_pages, gold) if gold else None,
            "no_relevant_page": no_relevant_page,
            "category": category,
        }
        per_query.append(row)

    def block_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
        nonempty = [r for r in rows if r["has_gold"]]
        empty = [r for r in rows if not r["has_gold"]]
        if not nonempty:
            return {
                "evaluated_non_empty_gold": 0,
                "excluded_empty_gold": len(empty),
                "hit_at_1": 0.0,
                "hit_at_5": 0.0,
                "hit_at_10": 0.0,
                "near_at_1": 0.0,
                "mrr": 0.0,
            }
        return {
            "evaluated_non_empty_gold": len(nonempty),
            "excluded_empty_gold": len(empty),
            "hit_at_1": statistics.fmean(r[f"{prefix}_hit_at_1"] for r in nonempty),
            "hit_at_5": statistics.fmean(r[f"{prefix}_hit_at_5"] for r in nonempty),
            "hit_at_10": statistics.fmean(r[f"{prefix}_hit_at_10"] for r in nonempty),
            "near_at_1": statistics.fmean(r[f"{prefix}_near_at_1"] for r in nonempty),
            "mrr": statistics.fmean(r[f"{prefix}_reciprocal_rank"] for r in nonempty),
        }

    def candidate_recall(rows: list[dict[str, Any]]) -> float:
        nonempty = [r for r in rows if r["has_gold"]]
        return statistics.fmean(bool(r["candidate_recall_at10"]) for r in nonempty) if nonempty else 0.0

    overall = {
        "fused_baseline": block_metrics(per_query, "fused"),
        "vlm_reranked": block_metrics(per_query, "vlm"),
        "candidate_recall_at10": candidate_recall(per_query),
        "no_relevant_page_rate": statistics.fmean(bool(r["no_relevant_page"]) for r in per_query)
        if per_query
        else 0.0,
        "categories": dict(Counter(r["category"] for r in per_query)),
    }

    per_language: dict[str, Any] = {}
    for language in sorted({r["lang"] for r in per_query}):
        lang_rows = [r for r in per_query if r["lang"] == language]
        per_language[language] = {
            "fused_baseline": block_metrics(lang_rows, "fused"),
            "vlm_reranked": block_metrics(lang_rows, "vlm"),
            "candidate_recall_at10": candidate_recall(lang_rows),
        }

    n_complete = len(per_query)
    n_errors = len(errors)
    n_pending = len(pending_sample_ids)
    is_complete = n_pending == 0 and (n_complete + n_errors) >= expected_total

    cost_usd = (
        usage_totals["input_tokens"] / 1_000_000.0 * NANO_INPUT_USD_PER_M
        + usage_totals["output_tokens"] / 1_000_000.0 * NANO_OUTPUT_USD_PER_M
    )

    summary = {
        "run_status": {
            "expected_total_queries": expected_total,
            "complete_rows": n_complete,
            "error_rows": n_errors,
            "pending_rows": n_pending,
            "is_complete": is_complete,
            "note": (
                "COMPLETE: safe to cite as a frozen diagnostic snapshot."
                if is_complete
                else "IN PROGRESS: another process is still writing this cache; rerun this "
                "script after it finishes before citing these numbers as final."
            ),
        },
        "overall": overall,
        "per_language": per_language,
        "api_usage": {
            "models": dict(model_ids),
            "total_input_tokens": usage_totals["input_tokens"],
            "total_cached_input_tokens": usage_totals["cached_tokens"],
            "total_output_tokens": usage_totals["output_tokens"],
            "estimated_cost_usd": round(cost_usd, 4),
            "cost_rate_note": (
                f"Approximate, using the same ${NANO_INPUT_USD_PER_M}/M input and "
                f"${NANO_OUTPUT_USD_PER_M}/M output nano-rate convention already used in "
                "docs/EXPERIMENT_LOG.md; cached-token discount not modeled."
            ),
            "mean_latency_seconds": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "median_latency_seconds": round(statistics.median(latencies), 3) if latencies else 0.0,
            "p95_latency_seconds": round(percentile(latencies, 95), 3) if latencies else 0.0,
        },
        "errors": errors[:50],
        "pending_sample_ids_preview": pending_sample_ids[:20],
        "per_query": per_query,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/task1-neighbor2"))
    parser.add_argument("--retrieval", type=Path, default=None, help="Defaults to <run-dir>/retrieval.jsonl")
    parser.add_argument(
        "--vlm-cache",
        type=Path,
        nargs="*",
        default=None,
        help="Defaults to <run-dir>/vlm-full20.jsonl if present, else <run-dir>/vlm-full20-cache.jsonl",
    )
    parser.add_argument("--expected-total", type=int, default=490)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/metrics/task1_neighbor2_summary.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("artifacts/metrics/task1_neighbor2_per_query.csv"))
    args = parser.parse_args()

    retrieval_path = args.retrieval or (args.run_dir / "retrieval.jsonl")
    if args.vlm_cache:
        vlm_paths = args.vlm_cache
    else:
        preferred = args.run_dir / "vlm-full20.jsonl"
        fallback = args.run_dir / "vlm-full20-cache.jsonl"
        vlm_paths = [preferred] if preferred.exists() else [fallback]

    retrieval = load_retrieval(retrieval_path)
    vlm_cache = load_vlm_cache(vlm_paths)
    summary = build_summary(retrieval, vlm_cache, args.expected_total)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["per_query"][0].keys()) if summary["per_query"] else [])
        writer.writeheader()
        for row in summary["per_query"]:
            writer.writerow(row)

    printable = {k: v for k, v in summary.items() if k != "per_query"}
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
