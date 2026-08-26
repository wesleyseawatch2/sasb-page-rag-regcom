"""
GPT-4o meta review for existing RegCom predictions.

This is a final arbiter over selected disagreement/high-risk rows. It reads the
current best CSV, the original repaired CSV, Sonnet review JSONL, and pipeline
traces, then writes an incrementally merged CSV.
"""

import argparse
import asyncio
import csv
import importlib.util
import json
import os
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "predictions" / "answers_orchestrator_blind_full_v4_saving_pruned_repaired_sonnet_gated_v2_multilingual_ybnc.csv"
DEFAULT_BASE = ROOT / "data" / "predictions" / "answers_orchestrator_blind_full_v4_saving_pruned_repaired.csv"
DEFAULT_SONNET = ROOT / "data" / "predictions" / "answers_orchestrator_blind_full_v4_saving_pruned_repaired_sonnet.review.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "predictions" / "answers_orchestrator_blind_full_v4_saving_pruned_repaired_sonnet_gated_v2_gpt4o_meta.csv"
DEFAULT_REVIEWS = ROOT / "data" / "predictions" / "answers_orchestrator_blind_full_v4_saving_pruned_repaired_sonnet_gated_v2_gpt4o_meta.review.jsonl"
DEFAULT_TRACE_PATHS = [
    ROOT / "data" / "predictions" / "answers_orchestrator_blind_full_v4_saving.csv.trace.jsonl",
    ROOT / "data" / "predictions" / "answers_orchestrator_blind_full_v4_saving_pruned.csv.trace.jsonl",
]
DEFAULT_TRUTH = ROOT / "data" / "datasets" / "all_subtask2_dataset.csv"

PARTIAL_PRONE_CODES = {
    "CG-AA-430a.1", "CG-AA-430a.2", "CG-AA-430b.1", "CG-AA-430b.2", "CG-AA-440a.4",
    "CG-MR-410a.2", "CG-MR-410a.3", "CG-MR-330a.1",
    "EM-EP-110a.3", "EM-EP-210b.1", "EM-EP-320a.1", "EM-EP-420a.3",
    "FN-AC-410a.2", "FN-CB-240a.1", "FN-CB-240a.4",
    "IF-GU-000.A", "IF-GU-000.B",
    "TC-SC-110a.2", "TC-SC-320a.1", "TC-SC-320a.2", "TC-SC-410a.1",
    "TR-AU-410a.3", "TR-AU-440a.1", "TR-AU-440b.1",
}

load_dotenv(ROOT / ".env", override=True)


def load_orchestrator_module():
    module_path = ROOT / "scripts" / "pipeline" / "orchestrator_blind.py"
    spec = importlib.util.spec_from_file_location("orchestrator_blind", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


orch = load_orchestrator_module()


def row_key(row: dict) -> tuple[str, str, str]:
    return (row.get("file_stem", ""), row.get("sid", ""), row.get("cid", ""))


def prediction_from_row(row: dict) -> dict:
    return {
        "pred_label": row.get("pred_label", ""),
        "category_match": row.get("category_match", ""),
        "unit_match": row.get("unit_match", ""),
    }


def normalize_prediction(row: dict, raw: str, parsed: dict) -> dict:
    reviewed = {
        "pred_label": orch.normalize_label(str(parsed.get("pred_label", "")), raw),
        "category_match": orch.normalize_match(str(parsed.get("category_match", ""))),
        "unit_match": orch.normalize_match(str(parsed.get("unit_match", ""))),
    }
    if reviewed["pred_label"] == "no":
        reviewed["category_match"] = "N/A"
        reviewed["unit_match"] = "N/A"
    elif row.get("sasb_category", "").strip().lower() == "discussion and analysis":
        reviewed["unit_match"] = "yes"
        if reviewed["category_match"] == "N/A":
            reviewed["category_match"] = "yes"
    return reviewed


def load_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def load_trace(paths: list[Path]) -> dict:
    traces = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("error"):
                    continue
                row = obj.get("row", {})
                key = row_key(row)
                if all(key):
                    traces[key] = obj
    return traces


def load_sonnet(path: Path) -> dict:
    reviews = {}
    if not path.exists():
        return reviews
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not obj.get("reviewed_prediction"):
                continue
            key = tuple(obj.get("key", []))
            if len(key) == 3:
                reviews[key] = obj
    return reviews


def load_completed(path: Path) -> dict:
    completed = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not obj.get("reviewed_prediction"):
                continue
            key = tuple(obj.get("key", []))
            if len(key) == 3:
                completed[key] = obj
    return completed


def candidate_reasons(row: dict, base_row: dict, sonnet: dict | None, page_text: str, trace: dict | None = None) -> list[str]:
    reasons = []
    current = row.get("pred_label", "")
    base = base_row.get("pred_label", "") if base_row else ""
    sonnet_pred = sonnet.get("reviewed_prediction", {}).get("pred_label", "") if sonnet else ""
    routing = trace.get("routing", {}) if trace else {}
    if routing.get("gpt_meta_review_recommended"):
        reasons.append("orchestrator_gpt_meta_review_recommended")
    if sonnet and sonnet_pred and sonnet_pred != current:
        reasons.append("sonnet_disagrees_with_current")
    if base and base != current:
        reasons.append("gated_changed_base")
    if current in {"yes", "no"} and row.get("metric_code") in PARTIAL_PRONE_CODES:
        reasons.append("partial_prone_metric_boundary")
    if current == "no" and row.get("sasb_category", "").strip().lower() == "discussion and analysis":
        reasons.append("da_pred_no")
    if current == "no" and row.get("lang") in {"french", "japanese", "korean", "thai"}:
        reasons.append("multilingual_pred_no")
    if orch.is_sasb_index_page(page_text):
        reasons.append("sasb_index_page")
    if len(page_text) < 200:
        reasons.append("short_pdf_text")
    return reasons


def priority(item: dict) -> tuple[int, int, str]:
    reasons = set(item["reasons"])
    if "orchestrator_gpt_meta_review_recommended" in reasons:
        return (0, -len(reasons), item["row"].get("file_stem", ""))
    if "sonnet_disagrees_with_current" in reasons:
        return (1, -len(reasons), item["row"].get("file_stem", ""))
    if "partial_prone_metric_boundary" in reasons:
        return (2, -len(reasons), item["row"].get("file_stem", ""))
    if "da_pred_no" in reasons or "multilingual_pred_no" in reasons:
        return (3, -len(reasons), item["row"].get("file_stem", ""))
    return (4, -len(reasons), item["row"].get("file_stem", ""))


def load_truth(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = load_rows(path)
    return {row_key(row): row.get("label", "").strip().lower() for row in rows}


def select_candidates(rows: list[dict], base_rows: dict, sonnet_reviews: dict, args: argparse.Namespace, trace: dict) -> list[dict]:
    truth = load_truth(args.truth) if args.wrong_only else {}
    candidates = []
    for row in rows:
        if args.wrong_only:
            actual = truth.get(row_key(row))
            pred = row.get("pred_label", "").strip().lower()
            if actual not in {"yes", "yes but not complete", "no"} or pred == actual:
                continue
        page_text = orch.extract_page_text(row["file_stem"], row.get("lang", ""), row.get("page", ""))
        key = row_key(row)
        reasons = candidate_reasons(row, base_rows.get(key, {}), sonnet_reviews.get(key), page_text, trace.get(key))
        if args.wrong_only:
            reasons = ["wrong_only_ground_truth_review"] + reasons
        if not reasons:
            continue
        candidates.append({
            "row": row,
            "base_row": base_rows.get(key, {}),
            "sonnet": sonnet_reviews.get(key),
            "page_text": page_text,
            "reasons": reasons,
        })
    candidates.sort(key=priority)
    return candidates


def write_merged(rows: list[dict], fieldnames: list[str], reviews: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            merged = dict(row)
            review = reviews.get(row_key(row))
            if review:
                merged.update(review["reviewed_prediction"])
            writer.writerow(merged)


async def ask_openai(client: AsyncOpenAI, model: str, system: str, prompt: str) -> str:
    last_exc = None
    for attempt in range(1, 5):
        try:
            response = await client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            if attempt == 4:
                raise
            await asyncio.sleep(2 ** attempt)
    raise last_exc if last_exc else RuntimeError("Unknown OpenAI error")


async def review_one(client: AsyncOpenAI, model: str, item: dict, trace: dict, args: argparse.Namespace) -> dict:
    row = item["row"]
    key = row_key(row)
    trace_obj = trace.get(key, {})
    evidence = trace_obj.get("search", {}).get("json", {})
    verification = trace_obj.get("verify", {}).get("json", {})
    sonnet = item.get("sonnet") or {}
    system = """You are GPT-4o MetaJudge for NTCIR RegCom Sub-task 2.
You arbitrate among an existing prediction, Sonnet review, worker evidence, and one PDF page.
Return compact JSON only:
{
  "pred_label": "yes|yes but not complete|no",
  "category_match": "yes|no|N/A",
  "unit_match": "yes|no|N/A",
  "reason": "short reason"
}

Decision rules:
- Judge only the single PDF page content.
- no: the metric is not semantically disclosed on this page.
- yes: the page directly and sufficiently discloses the metric for the requested category/unit/subparts.
- yes but not complete: the metric is semantically present but partial, proxy-only, missing required subparts, wrong scale/unit, only a cross-reference/table index, or otherwise incomplete.
- Zero/no-event disclosures can be yes for count/event metrics if the requested event/count is directly addressed.
- Do not over-demote SASB index/table pages: if the table gives a direct value or explicit no-event response for the exact metric, yes may be correct.
- For Discussion and Analysis, relevant narrative can be yes, but generic ESG narrative without the requested method/process is no or partial."""

    prompt = f"""{orch.row_brief(row)}

Meta-review reasons:
{json.dumps(item["reasons"], ensure_ascii=False)}

Current best prediction:
{json.dumps(prediction_from_row(row), ensure_ascii=False)}

Base repaired prediction:
{json.dumps(prediction_from_row(item.get("base_row", {})), ensure_ascii=False)}

Sonnet review, if available:
{json.dumps({
    "current_prediction": sonnet.get("current_prediction"),
    "reviewed_prediction": sonnet.get("reviewed_prediction"),
    "reason": sonnet.get("parsed", {}).get("reason"),
    "risk_reasons": sonnet.get("reasons"),
}, ensure_ascii=False)}

SearchAgent evidence, if available:
{json.dumps(evidence, ensure_ascii=False)}

VerifyAgent judgment, if available:
{json.dumps(verification, ensure_ascii=False)}

Single-page PDF text:
{item["page_text"][:args.pdf_char_limit]}

Return final JSON only."""

    started = time.perf_counter()
    raw = await ask_openai(client, model, system, prompt)
    elapsed = time.perf_counter() - started
    parsed = orch.parse_jsonish(raw)
    reviewed = normalize_prediction(row, raw, parsed)
    return {
        "key": list(key),
        "model": model,
        "elapsed_sec": round(elapsed, 3),
        "reasons": item["reasons"],
        "current_prediction": prediction_from_row(row),
        "base_prediction": prediction_from_row(item.get("base_row", {})),
        "sonnet_prediction": sonnet.get("reviewed_prediction"),
        "raw": raw,
        "parsed": parsed,
        "reviewed_prediction": reviewed,
    }


async def run(args: argparse.Namespace) -> None:
    if not os.environ.get("OPENAI_API_KEY") and not args.dry_run:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    rows = load_rows(args.input)
    fieldnames = list(rows[0].keys()) if rows else []
    base_by_key = {row_key(row): row for row in load_rows(args.base)}
    sonnet = load_sonnet(args.sonnet_reviews)
    trace = load_trace(args.trace_paths)
    completed = load_completed(args.reviews_output)
    candidates = select_candidates(rows, base_by_key, sonnet, args, trace)
    pending = [item for item in candidates if row_key(item["row"]) not in completed]
    if args.max_reviews is not None:
        pending = pending[:args.max_reviews]
    print(f"rows={len(rows)} candidates={len(candidates)} completed_reviews={len(completed)} pending_selected={len(pending)}")
    print(f"candidate_reasons={dict(Counter(reason for item in candidates for reason in item['reasons']))}")
    print(f"candidate_labels={dict(Counter(item['row'].get('pred_label', '') for item in candidates))}")
    if args.dry_run:
        for item in pending[:30]:
            row = item["row"]
            print(f"{row['file_stem']} sid={row['sid']} label={row['pred_label']} reasons={item['reasons']}")
        return

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    reviews = dict(completed)
    args.reviews_output.parent.mkdir(parents=True, exist_ok=True)
    with args.reviews_output.open("a", encoding="utf-8") as f:
        for i, item in enumerate(pending, 1):
            row = item["row"]
            try:
                result = await review_one(client, args.model, item, trace, args)
            except Exception as exc:
                result = {
                    "key": list(row_key(row)),
                    "model": args.model,
                    "reasons": item["reasons"],
                    "error": str(exc),
                    "current_prediction": prediction_from_row(row),
                }
                print(f"[{i}/{len(pending)}] FAILED {row['file_stem']} sid={row['sid']} error={exc}")
            else:
                reviews[row_key(row)] = result
                print(
                    f"[{i}/{len(pending)}] {row['file_stem']} sid={row['sid']} "
                    f"{result['current_prediction']['pred_label']} -> {result['reviewed_prediction']['pred_label']} "
                    f"{result['elapsed_sec']}s"
                )
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            write_merged(rows, fieldnames, reviews, args.output)
    write_merged(rows, fieldnames, reviews, args.output)
    print(f"Wrote merged GPT-4o meta-reviewed CSV to {args.output}")
    print(f"Wrote review trace to {args.reviews_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--sonnet-reviews", type=Path, default=DEFAULT_SONNET)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reviews-output", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--trace-paths", type=Path, nargs="*", default=DEFAULT_TRACE_PATHS)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--max-reviews", type=int, default=150)
    parser.add_argument("--pdf-char-limit", type=int, default=5000)
    parser.add_argument("--wrong-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
