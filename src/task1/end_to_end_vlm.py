"""Run a cached Task 1 end-to-end diagnostic after VLM page reranking.

The upstream reranker chooses a page from a full report.  This script sends
only that selected page (one image plus compact text) to a second VLM call and
asks for the Task 1 label: ``yes``, ``yes but not complete``, or ``no``.
It deliberately reports this as a diagnostic because the official Task 1
query-level gold/evaluation contract has not been released in this package.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import fitz
from dotenv import load_dotenv

from task1_pipeline import source_pdf_stem


LABELS = ("yes", "yes but not complete", "no")
POSITIVE_LABELS = {"yes", "yes but not complete"}
INPUT_RATE = 0.20  # USD per 1M input tokens for gpt-5.4-nano
OUTPUT_RATE = 1.25  # USD per 1M output tokens for gpt-5.4-nano


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def parse_jsonish(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    if "yes but not complete" in text or "partial" in text or "incomplete" in text:
        return "yes but not complete"
    if text in {"yes", "y", "true", "relevant", "complete"}:
        return "yes"
    if text in {"no", "n", "false", "irrelevant", "not relevant"}:
        return "no"
    return "no"


def normalize_match(value: Any, label: str) -> str:
    if label == "no":
        return "N/A"
    text = str(value or "").strip().lower()
    if text in {"yes", "y", "true", "match", "compatible"}:
        return "yes"
    if text in {"no", "n", "false", "mismatch", "incompatible"}:
        return "no"
    return "N/A"


def compact(value: Any, limit: int = 6000) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + " …"


def render_page_data_url(pdf_path: Path, page_number: int, dpi: int) -> str:
    with fitz.open(str(pdf_path)) as doc:
        page = doc[page_number - 1]
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        data = pixmap.tobytes("jpeg", jpg_quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def extract_page_text(pdf_path: Path, page_number: int, limit: int = 6000) -> str:
    with fitz.open(str(pdf_path)) as doc:
        if page_number < 1 or page_number > len(doc):
            return ""
        return compact(doc[page_number - 1].get_text("text"), limit)


def final_prompt(row: dict[str, Any], page: int, page_text: str, upstream: dict[str, Any]) -> str:
    prediction = upstream.get("prediction", {}) if isinstance(upstream, dict) else {}
    hint = ""
    if prediction:
        ranked_pages = prediction.get("ranked_pages") or []
        top_type = ranked_pages[0].get("evidence_type", "") if ranked_pages else ""
        hint = (
            f"An upstream page-reranker assigned evidence_type={top_type!r} "
            f"and no_relevant_page={bool(prediction.get('no_relevant_page'))}. Treat this only as a hint; verify independently."
        )
    return f"""You are the final page-level compliance verifier for a SASB metric.
Judge only the single supplied PDF page. Return JSON only.

Metric code: {row.get('metric_code', '')}
Topic: {row.get('topic', '')}
Metric: {row.get('metric_description', '')}
Expected answer/value: {row.get('expected_value', '')}
Expected unit: {row.get('expected_unit', '')}
SASB category: {row.get('sasb_category', '')}
Required unit of measure: {row.get('sasb_unit_of_measure', '')}
What counts: {compact(row.get('sasb_what_counts', ''), 3500)}
Selected PDF page index: {page}
{hint}

Page text (may be incomplete because of PDF extraction):
{page_text}

Decision rules:
- yes: the page directly discloses the requested metric with sufficiently complete evidence.
- yes but not complete: the page is relevant but partial, proxy-only, missing required scope/value/unit, or incomplete.
- no: the metric is absent, only an index/reference is shown, or the page is unrelated.
- category_match and unit_match must be yes/no when applicable and N/A for no.

Required JSON shape:
{{"pred_label":"yes|yes but not complete|no","category_match":"yes|no|N/A","unit_match":"yes|no|N/A","evidence":"short grounded phrase","reason":"short explanation"}}
"""


def call_openai(
    row: dict[str, Any],
    page: int,
    page_text: str,
    upstream: dict[str, Any],
    model: str,
    dpi: int,
    image_detail: str,
    max_output_tokens: int,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from openai import OpenAI

    pdf_path = Path(str(row["pdf_path"]))
    content: list[dict[str, Any]] = [{"type": "input_text", "text": final_prompt(row, page, page_text, upstream)}]
    content.append({"type": "input_image", "image_url": render_page_data_url(pdf_path, page, dpi), "detail": image_detail})
    started = time.perf_counter()
    response = OpenAI(timeout=timeout).responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        max_output_tokens=max_output_tokens,
        store=False,
    )
    elapsed = time.perf_counter() - started
    parsed = parse_jsonish(response.output_text)
    usage = getattr(response, "usage", None)
    usage_dict = usage.model_dump() if usage and hasattr(usage, "model_dump") else {}
    return parsed, {
        "response_id": response.id,
        "model": response.model,
        "latency_seconds": round(elapsed, 3),
        "usage": usage_dict,
    }


def selected_page(row: dict[str, Any]) -> int | None:
    ranked = row.get("vlm_ranked") or []
    if ranked and ranked[0].get("page") is not None:
        return int(ranked[0]["page"])
    fused = row.get("fused") or []
    return int(fused[0]["page"]) if fused and fused[0].get("page") is not None else None


def load_truth(path: Path) -> dict[tuple[str, ...], set[str]]:
    truth: dict[tuple[str, ...], set[str]] = defaultdict(set)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            stem = source_pdf_stem(row.get("file_stem", ""))
            key = (
                str(row.get("lang", "")).strip(),
                str(row.get("cid", "")).strip(),
                stem,
                str(row.get("topic", "")).strip(),
                str(row.get("metric_description", "")).strip(),
                str(row.get("metric_code", "")).strip(),
                str(row.get("page", "")).strip(),
            )
            truth[key].add(str(row.get("label", "")).strip().lower())
    return truth


def truth_labels(row: dict[str, Any], page: int | None, truth: dict[tuple[str, ...], set[str]]) -> set[str]:
    if page is None:
        return set()
    key = (
        str(row.get("lang", "")).strip(),
        str(row.get("cid", "")).strip(),
        str(row.get("report_stem", "")).strip(),
        str(row.get("topic", "")).strip(),
        str(row.get("metric_description", "")).strip(),
        str(row.get("metric_code", "")).strip(),
        str(page),
    )
    return set(truth.get(key, set()))


def cost_from_usage(usage: dict[str, Any]) -> float:
    if not usage:
        return 0.0
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    return input_tokens / 1_000_000 * INPUT_RATE + output_tokens / 1_000_000 * OUTPUT_RATE


def filter_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.languages:
        allowed = {lang.lower() for lang in args.languages}
        rows = [row for row in rows if str(row.get("lang", "")).lower() in allowed]
    if args.max_per_language:
        kept: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for row in rows:
            lang = str(row.get("lang", "")).lower()
            if counts[lang] >= args.max_per_language:
                continue
            kept.append(row)
            counts[lang] += 1
        rows = kept
    if args.limit > 0:
        rows = rows[: args.limit]
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(args.env_file, override=True)
    retrieval_rows: list[dict[str, Any]] = []
    for path in args.retrieval:
        retrieval_rows.extend(read_jsonl(path))
    rows = filter_rows(retrieval_rows, args)
    truth = load_truth(args.truth) if args.truth and args.truth.exists() else {}
    cached = {row["sample_id"]: row for row in read_jsonl(args.cache)} if args.cache.exists() else {}
    output_rows: list[dict[str, Any]] = []
    total_cost = 0.0
    calls = 0
    errors = 0
    for position, row in enumerate(rows, 1):
        page = selected_page(row)
        upstream = row.get("vlm_trace", {})
        trace = cached.get(row["sample_id"])
        if trace is None:
            if page is None or not row.get("pdf_path"):
                trace = {"sample_id": row["sample_id"], "status": "error", "error": "missing selected page or PDF"}
            else:
                try:
                    page_text = extract_page_text(Path(row["pdf_path"]), page)
                    parsed, metadata = call_openai(
                        row, page, page_text, upstream, args.model, args.dpi,
                        args.image_detail, args.max_output_tokens, args.api_timeout,
                    )
                    trace = {
                        "sample_id": row["sample_id"], "status": "complete", "selected_page": page,
                        "prediction": parsed, **metadata,
                    }
                    calls += 1
                    total_cost += cost_from_usage(metadata.get("usage", {}))
                    if total_cost > args.max_cost_usd:
                        raise RuntimeError(
                            f"run cost guard exceeded: ${total_cost:.4f} > ${args.max_cost_usd:.2f}"
                        )
                    args.cache.parent.mkdir(parents=True, exist_ok=True)
                    with args.cache.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                except Exception as exc:
                    errors += 1
                    trace = {
                        "sample_id": row["sample_id"], "status": "error", "selected_page": page,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    # A cost-guard exception must stop before another request.
                    if "cost guard exceeded" in str(exc):
                        break
        prediction = trace.get("prediction", {}) if isinstance(trace, dict) else {}
        label = normalize_label(prediction.get("pred_label")) if trace.get("status") == "complete" else "no"
        category = normalize_match(prediction.get("category_match"), label) if trace.get("status") == "complete" else "N/A"
        unit = normalize_match(prediction.get("unit_match"), label) if trace.get("status") == "complete" else "N/A"
        gold = truth_labels(row, page, truth)
        output_rows.append({
            "sample_id": row["sample_id"], "lang": row.get("lang", ""), "cid": row.get("cid", ""),
            "report_stem": row.get("report_stem", ""), "topic": row.get("topic", ""),
            "metric_code": row.get("metric_code", ""), "selected_page": page,
            "pred_label": label, "category_match": category, "unit_match": unit,
            "gold_page_labels": sorted(gold), "gold_pages": row.get("gold_pages", []),
            "retrieval_hit": bool(page is not None and page in {int(p) for p in row.get("gold_pages", [])}),
            "trace_status": trace.get("status", ""), "trace": trace,
        })
        if position % 10 == 0 or position == len(rows):
            print(f"classified {position}/{len(rows)} calls={calls} cost=${total_cost:.4f}", flush=True)
    # Cached rows may have been present before this process, so compute usage
    # and metrics from all completed traces rather than only new calls.
    completed = [item for item in output_rows if item["trace_status"] == "complete"]
    nonempty = [item for item in completed if item["gold_pages"]]
    page_label_scored = [item for item in completed if item["gold_page_labels"]]
    label_correct = sum(item["pred_label"] in set(item["gold_page_labels"]) for item in page_label_scored)
    retrieval_hits = sum(bool(item["retrieval_hit"]) for item in nonempty)
    metrics = {
        "model": args.model, "queries_requested": len(rows), "completed": len(completed),
        "errors": errors, "new_api_calls": calls, "new_estimated_cost_usd": round(total_cost, 6),
        "retrieval_hit_at_1": retrieval_hits / len(nonempty) if nonempty else 0.0,
        "selected_page_label_scored": len(page_label_scored),
        "selected_page_label_accuracy": label_correct / len(page_label_scored) if page_label_scored else 0.0,
        "end_to_end_proxy_correct": sum(
            item["pred_label"] in set(item["gold_page_labels"])
            if item["gold_page_labels"] else item["pred_label"] == "no"
            for item in completed
        ),
        "end_to_end_proxy_accuracy": (
            sum(
                item["pred_label"] in set(item["gold_page_labels"])
                if item["gold_page_labels"] else item["pred_label"] == "no"
                for item in completed
            ) / len(completed) if completed else 0.0
        ),
        "latency_mean_seconds": statistics.fmean(
            float(item["trace"].get("latency_seconds", 0.0)) for item in completed
        ) if completed else 0.0,
        "pred_label_counts": dict(Counter(item["pred_label"] for item in completed)),
        "note": "Diagnostic proxy; official Task 1 query-level gold/evaluator is not confirmed.",
    }
    write_jsonl(args.output, output_rows)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, nargs="+", required=True, help="One or more VLM-reranked JSONL files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--truth", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--languages", nargs="+", default=[])
    parser.add_argument("--max-per-language", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=96)
    parser.add_argument("--image-detail", choices=["low", "high"], default="low")
    parser.add_argument("--max-output-tokens", type=int, default=300)
    parser.add_argument("--api-timeout", type=float, default=90.0)
    parser.add_argument("--max-cost-usd", type=float, default=8.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
