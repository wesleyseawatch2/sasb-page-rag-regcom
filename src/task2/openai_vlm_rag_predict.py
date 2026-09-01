"""Task 2 multimodal RAG prediction with GPT-5.4 Mini.

This runner keeps the existing TF-IDF/label-balanced example retrieval used by
the text-only baselines, then adds a rendered image of the target PDF page to
the Responses API request.  The image is deliberately sent only for the target
page (not for every few-shot example) to control cost while still allowing the
model to read charts, tables, and layout that PDF text extraction misses.

The default invocation is a 24-row pilot.  Use ``--limit 0`` only after the
pilot succeeds and the API usage/cost look safe.  Every completed request is
written to a JSONL trace and can be resumed without duplicate calls.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from glm_rag_predict_english import (
    build_prompt,
    compact,
    derive_matches,
    evaluate,
    filter_languages,
    key,
    label_prior,
    normalize_label,
    normalize_match,
    parse_jsonish,
    read_csv,
    retrieval_text,
    select_examples,
    write_csv,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import fitz


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "datasets" / "all"
OUT_DIR = ROOT / "data" / "predictions" / "openai_vlm_rag"

# Official GPT-5.4 Mini text-token prices at the time this runner was written.
# The guard is intentionally configurable and is not used to claim a fixed
# invoice because image-token accounting can vary by request.
INPUT_USD_PER_M = 0.75
OUTPUT_USD_PER_M = 4.50


def read_trace(path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            index = item.get("index")
            prediction = item.get("prediction")
            if isinstance(index, int) and isinstance(prediction, dict):
                completed[index] = item
    return completed


def append_trace(path: Path, item: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def render_page_data_url(pdf_path: Path, page: str, dpi: int) -> str:
    page_number = int(float(page))
    if page_number < 1:
        raise ValueError(f"invalid PDF page number: {page}")
    with fitz.open(str(pdf_path)) as document:
        if page_number > document.page_count:
            raise ValueError(f"page {page_number} exceeds {document.page_count} pages in {pdf_path.name}")
        pixmap = document[page_number - 1].get_pixmap(dpi=dpi, alpha=False)
        data = pixmap.tobytes("jpeg", jpg_quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def source_pdf_path(row: dict[str, Any]) -> Path:
    file_stem = compact(row.get("file_stem"))
    lang = compact(row.get("lang")) or "english"
    # ``extract_page_text`` already applies the participant ``stem_page``
    # convention.  Reuse that same source-stem helper without depending on a
    # private symbol from the older script.
    import re

    source_stem = re.sub(r"(.+)_([0-9]+)$", r"\1", file_stem)
    return ROOT / "Training Set" / "PDF" / lang / f"{source_stem}.pdf"


def extract_repo_page_text(row: dict[str, Any]) -> str:
    """Extract text from the full report using the repository's PDF layout."""
    pdf_path = source_pdf_path(row)
    page_number = int(float(compact(row.get("page"))))
    if not pdf_path.exists() or page_number < 1:
        return ""
    with fitz.open(str(pdf_path)) as document:
        if page_number > document.page_count:
            return ""
        return document[page_number - 1].get_text("text").strip()


def build_multimodal_input(
    row: dict[str, Any],
    page_text: str,
    examples: list[dict[str, Any]],
    metric_prior: dict[str, int],
    pdf_char_limit: int,
    pdf_path: Path,
    page: str,
    dpi: int,
    image_detail: str,
) -> list[dict[str, Any]]:
    text_messages = build_prompt(row, page_text, examples, metric_prior, pdf_char_limit)
    system = text_messages[0]["content"]
    user = text_messages[1]["content"]
    multimodal_user: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                user
                + "\n\nThe following image is the rendered target PDF page. "
                "Use it to inspect charts, tables, labels, and layout that may be missing from the extracted text. "
                "When text and image disagree because extraction is incomplete, inspect the image directly."
            ),
        },
        {
            "type": "input_image",
            "image_url": render_page_data_url(pdf_path, page, dpi),
            "detail": image_detail,
        },
    ]
    # Keep a single user message for broad SDK compatibility and make the
    # image inclusion explicit in the traceable prompt.
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": system + "\n\n" + multimodal_user[0]["text"]},
                multimodal_user[1],
            ],
        }
    ]


def call_openai(
    client: Any,
    messages: list[dict[str, Any]],
    model: str,
    max_output_tokens: int,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    started = time.perf_counter()
    response = client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=max_output_tokens,
        store=False,
    )
    elapsed = time.perf_counter() - started
    usage = getattr(response, "usage", None)
    usage_dict = usage.model_dump() if usage and hasattr(usage, "model_dump") else {}
    return response.output_text, {
        "response_id": getattr(response, "id", ""),
        "model": getattr(response, "model", model),
        "latency_seconds": round(elapsed, 3),
        "usage": usage_dict,
    }


def usage_cost(usage: dict[str, Any]) -> float:
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    return input_tokens / 1_000_000 * INPUT_USD_PER_M + output_tokens / 1_000_000 * OUTPUT_USD_PER_M


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # ``all_subtask2_dataset.csv`` is the released Subtask 2 training split;
    # after excluding Chinese it yields the same 753-row pool used by the
    # existing paper baselines.  ``training_combined.csv`` also contains
    # auxiliary rows and would change the comparison denominator.
    parser.add_argument("--train", type=Path, default=DATA_DIR / "all_subtask2_dataset.csv")
    parser.add_argument("--test", type=Path, default=DATA_DIR / "test_answer_sheet.csv")
    parser.add_argument("--truth", type=Path, default=DATA_DIR / "test_truth.csv")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--limit", type=int, default=24, help="Rows to run; 0 means all retained test rows")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=300)
    parser.add_argument("--pdf-char-limit", type=int, default=5000)
    parser.add_argument("--dpi", type=int, default=96)
    parser.add_argument("--image-detail", choices=["low", "high"], default="low")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-cost-usd", type=float, default=1.0)
    parser.add_argument("--tag", default="gpt54mini_vlm")
    parser.add_argument("--include-langs", nargs="*", default=[])
    parser.add_argument("--exclude-langs", nargs="*", default=["chinese"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=True)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key and not args.dry_run:
        raise RuntimeError("OPENAI_API_KEY is not configured in the environment or .env")

    train_rows = filter_languages(read_csv(args.train), args.include_langs, args.exclude_langs)
    test_rows = filter_languages(read_csv(args.test), args.include_langs, args.exclude_langs)
    truth_rows = filter_languages(read_csv(args.truth), args.include_langs, args.exclude_langs)
    if args.limit > 0:
        test_rows = test_rows[: args.limit]
    if not test_rows:
        raise RuntimeError("No test rows remain after language/limit filtering")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{len(test_rows)}_{args.tag}"
    predictions_path = args.out_dir / f"predictions_{suffix}.csv"
    trace_path = args.out_dir / f"trace_{suffix}.jsonl"
    metrics_path = args.out_dir / f"metrics_{suffix}.json"

    print(f"extracting {len(train_rows)} training and {len(test_rows)} target page texts...", flush=True)
    train_page_texts = [
        extract_repo_page_text(row)
        for row in train_rows
    ]
    test_page_texts = [
        extract_repo_page_text(row)
        for row in test_rows
    ]

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
    train_matrix = vectorizer.fit_transform(
        [retrieval_text(row, page_text) for row, page_text in zip(train_rows, train_page_texts)]
    )
    test_matrix = vectorizer.transform(
        [retrieval_text(row, page_text) for row, page_text in zip(test_rows, test_page_texts)]
    )

    completed = read_trace(trace_path) if args.resume else {}
    if trace_path.exists() and not args.resume:
        trace_path.unlink()
    output_by_index: dict[int, dict[str, Any]] = {}
    for index, item in completed.items():
        if 1 <= index <= len(test_rows):
            out_row = dict(test_rows[index - 1])
            out_row.update(item.get("prediction", {}))
            output_by_index[index] = out_row

    client = None
    if not args.dry_run:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=args.timeout)

    trace_lock = threading.Lock()
    print_lock = threading.Lock()
    cost_lock = threading.Lock()
    total_cost = 0.0
    new_calls = 0
    errors: list[dict[str, Any]] = []
    stop_event = threading.Event()

    def process_one(index: int, row: dict[str, Any], page_text: str) -> tuple[int, dict[str, Any], dict[str, Any]]:
        nonlocal total_cost, new_calls
        if stop_event.is_set():
            raise RuntimeError("cost guard reached before request")
        similarities = cosine_similarity(test_matrix[index - 1], train_matrix).ravel()
        examples = select_examples(row, similarities, train_rows, train_page_texts, args.top_k)
        metric_prior = label_prior(row, train_rows)
        pdf_path = source_pdf_path(row)
        messages = build_multimodal_input(
            row, page_text, examples, metric_prior, args.pdf_char_limit, pdf_path,
            compact(row.get("page")), args.dpi, args.image_detail,
        )
        if args.dry_run:
            raw = json.dumps({"pred_label": "no", "category_match": "N/A", "unit_match": "N/A", "reason": "dry run"})
            metadata = {"model": args.model, "usage": {}, "latency_seconds": 0.0}
        else:
            assert client is not None
            raw, metadata = call_openai(client, messages, args.model, args.max_output_tokens, args.timeout)
        parsed = parse_jsonish(raw)
        pred_label = normalize_label(parsed.get("pred_label", raw))
        category_match = normalize_match(parsed.get("category_match", ""))
        unit_match = normalize_match(parsed.get("unit_match", ""))
        if pred_label == "no":
            category_match, unit_match = "N/A", "N/A"
        elif category_match == "N/A" or unit_match == "N/A":
            fallback_category, fallback_unit = derive_matches(row, pred_label)
            category_match = fallback_category if category_match == "N/A" else category_match
            unit_match = fallback_unit if unit_match == "N/A" else unit_match
        prediction = {
            "pred_label": pred_label,
            "category_match": category_match,
            "unit_match": unit_match,
        }
        trace = {
            "index": index,
            "key": key(row),
            "status": "complete",
            "model": args.model,
            "image": {"pdf": str(pdf_path), "page": compact(row.get("page")), "dpi": args.dpi, "detail": args.image_detail},
            "retrieved_examples": [
                {"similarity": item["similarity"], "file_stem": item["row"].get("file_stem", ""),
                 "sid": item["row"].get("sid", ""), "label": item["row"].get("label", ""),
                 "same_metric": item["same_metric"]}
                for item in examples
            ],
            "raw": raw,
            "parsed": parsed,
            "prediction": prediction,
            **metadata,
        }
        with cost_lock:
            total_cost += usage_cost(metadata.get("usage", {}))
            new_calls += 1
            if total_cost > args.max_cost_usd:
                stop_event.set()
        return index, {**row, **prediction}, trace

    pending = [
        (index, row, page_text)
        for index, (row, page_text) in enumerate(zip(test_rows, test_page_texts), 1)
        if index not in completed
    ]
    done_count = len(output_by_index)
    if pending:
        workers = max(1, args.concurrency)
        print(f"running {len(pending)} pending rows with concurrency={workers}, model={args.model}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one, index, row, page_text): (index, row) for index, row, page_text in pending}
            for future in as_completed(futures):
                index, row = futures[future]
                try:
                    index, output_row, trace = future.result()
                except Exception as exc:
                    error = {"index": index, "key": key(row), "status": "error", "error": f"{type(exc).__name__}: {exc}"}
                    errors.append(error)
                    append_trace(trace_path, error, trace_lock)
                    with print_lock:
                        print(f"ERROR index={index} {row.get('file_stem')} {row.get('sid')}: {exc}", flush=True)
                    continue
                output_by_index[index] = output_row
                append_trace(trace_path, trace, trace_lock)
                done_count += 1
                with print_lock:
                    print(f"{done_count}/{len(test_rows)} index={index} {row.get('file_stem')} {row.get('sid')} -> {output_row.get('pred_label')} cost=${total_cost:.4f}", flush=True)

    fieldnames = list(test_rows[0].keys())
    for field in ["pred_label", "category_match", "unit_match"]:
        if field not in fieldnames:
            fieldnames.append(field)
    output_rows = [output_by_index[index] for index in sorted(output_by_index)]
    write_csv(predictions_path, output_rows, fieldnames)
    metrics = evaluate(output_rows, truth_rows)
    metrics.update({
        "model": args.model,
        "multimodal": True,
        "image_detail": args.image_detail,
        "dpi": args.dpi,
        "limit": args.limit,
        "requested_rows": len(test_rows),
        "completed_rows": len(output_rows),
        "error_rows": len(errors),
        "new_api_calls": new_calls,
        "estimated_cost_usd": round(total_cost, 6),
        "max_cost_usd": args.max_cost_usd,
        "top_k": args.top_k,
        "concurrency": args.concurrency,
        "predictions_path": str(predictions_path),
        "trace_path": str(trace_path),
        "note": "Task 2 multimodal diagnostic; image is the target page only, few-shot examples remain text snippets.",
    })
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics.get(k) for k in ["evaluated", "accuracy", "micro_f1", "macro_f1", "completed_rows", "error_rows", "estimated_cost_usd"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
