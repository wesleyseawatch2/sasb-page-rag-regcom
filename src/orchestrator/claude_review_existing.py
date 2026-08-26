"""
Second-stage Claude review for an existing RegCom prediction CSV.

This script does not run the OpenAI orchestrator pipeline. It reads an existing
prediction file, selects high-risk rows, asks Claude to review only those rows,
and writes a merged CSV incrementally.
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

from google import genai as _genai
from google.genai import types as _genai_types
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "predictions" / "answers_orchestrator_blind_full_v4_saving_pruned_repaired.csv"
DEFAULT_OUTPUT = ROOT / "data" / "predictions" / "answers_orchestrator_blind_full_v4_saving_pruned_repaired_claude.csv"
DEFAULT_REVIEWS = ROOT / "data" / "predictions" / "answers_orchestrator_blind_full_v4_saving_pruned_repaired_claude.review.jsonl"
DEFAULT_TRACE_PATHS = [
    ROOT / "data" / "predictions" / "answers_orchestrator_blind_full_v4_saving.csv.trace.jsonl",
    ROOT / "data" / "predictions" / "answers_orchestrator_blind_full_v4_saving_pruned.csv.trace.jsonl",
]

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


def risk_reasons(row: dict, page_text: str, include_multilingual_no: bool, trace: dict | None = None) -> list[str]:
    reasons = []
    label = row.get("pred_label", "")
    routing = trace.get("routing", {}) if trace else {}
    selected = set(routing.get("selected_agents", []))
    policy = routing.get("review_policy", {})
    if "ClaudeConflictReviewAgent" in selected or policy.get("claude") in {"risk_only", "always"}:
        reasons.append("orchestrator_claude_review_recommended")
    if label == "yes but not complete":
        reasons.append("boundary_yes_but_not_complete")
    if row.get("unit_match") == "no":
        reasons.append("unit_mismatch")
    if row.get("category_match") == "no":
        reasons.append("category_mismatch")
    if row.get("sasb_category", "").strip().lower() == "discussion and analysis" and label == "no":
        reasons.append("da_pred_no")
    if include_multilingual_no and row.get("lang") in {"french", "japanese", "korean", "thai"} and label == "no":
        reasons.append("multilingual_pred_no")
    if len(page_text) < 200:
        reasons.append("short_pdf_text")
    if orch.is_sasb_index_page(page_text):
        reasons.append("sasb_index_page")
    return reasons


def priority(row: dict, reasons: list[str]) -> tuple[int, int, str]:
    if "orchestrator_claude_review_recommended" in reasons:
        return (0, len(reasons), row.get("file_stem", ""))
    high = {
        "boundary_yes_but_not_complete",
        "unit_mismatch",
        "category_mismatch",
        "sasb_index_page",
    }
    if any(reason in high for reason in reasons):
        return (1, len(reasons), row.get("file_stem", ""))
    if "da_pred_no" in reasons:
        return (2, len(reasons), row.get("file_stem", ""))
    if "multilingual_pred_no" in reasons:
        return (3, len(reasons), row.get("file_stem", ""))
    return (4, len(reasons), row.get("file_stem", ""))


def load_trace(trace_paths: list[Path]) -> dict:
    traces = {}
    for path in trace_paths:
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


def load_completed_reviews(path: Path) -> dict:
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
            key = tuple(obj.get("key", []))
            if len(key) == 3 and obj.get("reviewed_prediction"):
                completed[key] = obj
    return completed


def select_candidates(rows: list[dict], args: argparse.Namespace, trace: dict) -> list[dict]:
    candidates = []
    seen = set()
    for row in rows:
        key = row_key(row)
        if key in seen:
            continue
        seen.add(key)
        page_text = orch.extract_page_text(row["file_stem"], row.get("lang", ""), row.get("page", ""))
        reasons = risk_reasons(row, page_text, args.include_multilingual_no, trace.get(key))
        if not reasons:
            continue
        candidates.append({
            "row": row,
            "page_text": page_text,
            "reasons": reasons,
            "priority": priority(row, reasons),
        })
    candidates.sort(key=lambda item: (item["priority"][0], -item["priority"][1], item["priority"][2]))
    return candidates


def write_merged_csv(base_rows: list[dict], fieldnames: list[str], reviews: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in base_rows:
            merged = dict(row)
            review = reviews.get(row_key(row))
            if review:
                merged.update(review["reviewed_prediction"])
            writer.writerow(merged)


async def ask_claude(client: "_genai.Client", model: str, prompt: str, system: str) -> str:
    last_exc = None
    for attempt in range(1, 5):
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=_genai_types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.0,
                    max_output_tokens=500,
                ),
            )
            return response.text or ""
        except Exception as exc:
            last_exc = exc
            if attempt == 4:
                raise
            await asyncio.sleep(2 ** attempt)
    raise last_exc if last_exc else RuntimeError("Unknown Gemini error")


async def review_one(client: "_genai.Client", model: str, item: dict, trace: dict, args: argparse.Namespace) -> dict:
    row = item["row"]
    key = row_key(row)
    trace_obj = trace.get(key, {})
    evidence = trace_obj.get("search", {}).get("json", {})
    verification = trace_obj.get("verify", {}).get("json", {})
    system = """You are Claude ConflictReviewAgent for NTCIR RegCom Sub-task 2.
Review one high-risk existing prediction.
Return compact JSON only:
{
  "pred_label": "yes|yes but not complete|no",
  "category_match": "yes|no|N/A",
  "unit_match": "yes|no|N/A",
  "reason": "short reason"
}

Rules:
- Judge only the single PDF page content provided.
- This is page-level metric verification, not a full-report SASB audit.
- If the metric is not semantically mentioned on the page: no / N/A / N/A.
- Zero/no-event statements count as disclosure for event/count metrics.
- For Discussion and Analysis metrics, relevant narrative can be yes.
- Use yes but not complete for partial, proxy, incomplete, or wrong-unit disclosure.
- Do not use ground truth or assume anything outside the page."""

    prompt = f"""{orch.row_brief(row)}

High-risk reasons:
{json.dumps(item["reasons"], ensure_ascii=False)}

Current prediction:
{json.dumps(prediction_from_row(row), ensure_ascii=False)}

SearchAgent evidence, if available:
{json.dumps(evidence, ensure_ascii=False)}

VerifyAgent judgment, if available:
{json.dumps(verification, ensure_ascii=False)}

Single-page PDF text:
{item["page_text"][:args.pdf_char_limit]}

Review this row and return final JSON only."""

    started = time.perf_counter()
    raw = await ask_claude(client, model, prompt, system)
    elapsed = time.perf_counter() - started
    parsed = orch.parse_jsonish(raw)
    reviewed = normalize_prediction(row, raw, parsed)
    return {
        "key": list(key),
        "reasons": item["reasons"],
        "model": model,
        "elapsed_sec": round(elapsed, 3),
        "current_prediction": prediction_from_row(row),
        "raw": raw,
        "parsed": parsed,
        "reviewed_prediction": reviewed,
    }


async def run(args: argparse.Namespace) -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    rows = list(csv.DictReader(args.input.open(encoding="utf-8-sig")))
    fieldnames = list(rows[0].keys()) if rows else []
    trace = load_trace(args.trace_paths)
    completed = load_completed_reviews(args.reviews_output)
    candidates = select_candidates(rows, args, trace)
    pending = [item for item in candidates if row_key(item["row"]) not in completed]
    if args.max_reviews is not None:
        pending = pending[:args.max_reviews]

    print(f"rows={len(rows)} candidates={len(candidates)} completed_reviews={len(completed)} pending_selected={len(pending)}")
    print(f"candidate_reasons={dict(Counter(reason for item in candidates for reason in item['reasons']))}")
    print(f"candidate_labels={dict(Counter(item['row'].get('pred_label', '') for item in candidates))}")
    if args.dry_run:
        for item in pending[:25]:
            row = item["row"]
            print(f"{row['file_stem']} sid={row['sid']} label={row['pred_label']} reasons={item['reasons']}")
        return

    client = _genai.Client(api_key=api_key)
    args.reviews_output.parent.mkdir(parents=True, exist_ok=True)
    reviews = dict(completed)
    with args.reviews_output.open("a", encoding="utf-8") as f:
        for i, item in enumerate(pending, 1):
            try:
                result = await review_one(client, args.model, item, trace, args)
            except Exception as exc:
                row = item["row"]
                result = {
                    "key": list(row_key(row)),
                    "reasons": item["reasons"],
                    "model": args.model,
                    "error": str(exc),
                    "current_prediction": prediction_from_row(row),
                }
                print(f"[{i}/{len(pending)}] FAILED {row['file_stem']} sid={row['sid']} error={exc}")
            else:
                reviews[row_key(item["row"])] = result
                print(
                    f"[{i}/{len(pending)}] {item['row']['file_stem']} sid={item['row']['sid']} "
                    f"{result['current_prediction']['pred_label']} -> {result['reviewed_prediction']['pred_label']} "
                    f"{result['elapsed_sec']}s"
                )
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            write_merged_csv(rows, fieldnames, reviews, args.output)

    write_merged_csv(rows, fieldnames, reviews, args.output)
    print(f"Wrote merged Claude-reviewed CSV to {args.output}")
    print(f"Wrote review trace to {args.reviews_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reviews-output", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--trace-paths", type=Path, nargs="*", default=DEFAULT_TRACE_PATHS)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
    parser.add_argument("--max-reviews", type=int, default=120)
    parser.add_argument("--pdf-char-limit", type=int, default=5000)
    parser.add_argument("--include-multilingual-no", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
