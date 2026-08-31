"""Compare fused retrieval and cached VLM rankings without making API calls."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pages(row: dict[str, Any], field: str) -> list[int]:
    return [int(item["page"]) for item in row.get(field, []) if item.get("page") is not None]


def first_ranked_page(row: dict[str, Any], field: str) -> int | None:
    ranked = pages(row, field)
    return ranked[0] if ranked else None


def reciprocal_rank(row: dict[str, Any], field: str, gold: set[int]) -> float:
    for rank, page in enumerate(pages(row, field), 1):
        if page in gold:
            return 1.0 / rank
    return 0.0


def hit(row: dict[str, Any], field: str, gold: set[int], k: int) -> bool:
    return bool(set(pages(row, field)[:k]) & gold)


def summarize_rows(
    baseline_rows: Iterable[dict[str, Any]], vlm_rows: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    baseline = {row["sample_id"]: row for row in baseline_rows}
    vlm = {row["sample_id"]: row for row in vlm_rows}
    records: list[dict[str, Any]] = []
    for sample_id, vrow in vlm.items():
        brow = baseline.get(sample_id)
        if brow is None:
            continue
        gold = {int(page) for page in brow.get("gold_pages", [])}
        base_candidates = pages(brow, "fused")[:10]
        base_h1 = hit(brow, "fused", gold, 1) if gold else False
        vlm_h1 = hit(vrow, "vlm_ranked", gold, 1) if gold else False
        if not gold:
            category = "empty_gold"
        elif not (gold & set(base_candidates)):
            category = "candidate_recall_miss"
        elif not base_h1 and vlm_h1:
            category = "vlm_improved_top1"
        elif base_h1 and not vlm_h1:
            category = "vlm_degraded_top1"
        elif base_h1 and vlm_h1:
            category = "both_top1"
        else:
            category = "neither_top1"
        prediction = vrow.get("vlm_trace", {}).get("prediction", {})
        records.append(
            {
                "sample_id": sample_id,
                "lang": brow.get("lang", ""),
                "cid": brow.get("cid", ""),
                "topic": brow.get("topic", ""),
                "metric_code": brow.get("metric_code", ""),
                "gold_pages": sorted(gold),
                "baseline_top5": pages(brow, "fused")[:5],
                "vlm_top5": pages(vrow, "vlm_ranked")[:5],
                "candidate_recall_at10": bool(gold & set(base_candidates)),
                "baseline_hit_at_1": base_h1,
                "vlm_hit_at_1": vlm_h1,
                "baseline_mrr": reciprocal_rank(brow, "fused", gold) if gold else 0.0,
                "vlm_mrr": reciprocal_rank(vrow, "vlm_ranked", gold) if gold else 0.0,
                "no_relevant_page": bool(prediction.get("no_relevant_page")),
                "category": category,
            }
        )

    def metric(rows: list[dict[str, Any]], key: str) -> float:
        return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0

    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        nonempty = [row for row in rows if row["gold_pages"]]
        return {
            "queries": len(rows),
            "nonempty_gold": len(nonempty),
            "empty_gold": len(rows) - len(nonempty),
            "candidate_recall_at10": metric(nonempty, "candidate_recall_at10"),
            "baseline_hit_at_1": metric(nonempty, "baseline_hit_at_1"),
            "vlm_hit_at_1": metric(nonempty, "vlm_hit_at_1"),
            "baseline_mrr": sum(row["baseline_mrr"] for row in nonempty) / len(nonempty) if nonempty else 0.0,
            "vlm_mrr": sum(row["vlm_mrr"] for row in nonempty) / len(nonempty) if nonempty else 0.0,
            "no_relevant_page_true_rate_empty": metric(
                [row for row in rows if not row["gold_pages"]], "no_relevant_page"
            ),
            "no_relevant_page_true_rate_nonempty": metric(nonempty, "no_relevant_page"),
        }

    per_language: dict[str, Any] = {}
    for language in sorted({row["lang"] for row in records}):
        per_language[language] = metrics([row for row in records if row["lang"] == language])
    return {
        "overall": metrics(records),
        "per_language": per_language,
        "categories": dict(Counter(row["category"] for row in records)),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--vlm", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    vlm_rows: list[dict[str, Any]] = []
    for path in args.vlm:
        vlm_rows.extend(read_jsonl(path))
    summary = summarize_rows(read_jsonl(args.baseline), vlm_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
