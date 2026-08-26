from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from glm_rag_predict_english import (
    DATA_DIR,
    OUT_DIR as GLM_OUT_DIR,
    append_jsonl,
    build_prompt,
    compact,
    derive_matches,
    evaluate,
    extract_page_text,
    filter_languages,
    key,
    label_prior,
    load_trace_predictions,
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


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = GLM_OUT_DIR.parent / "gemini_25_pro_rag"


def gemini_payload(messages: list[dict], temperature: float, top_p: float, max_output_tokens: int) -> dict:
    system = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
    user = "\n\n".join(message["content"] for message in messages if message["role"] != "system")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
        },
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    return payload


def call_gemini(
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict],
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    timeout: int,
    retries: int,
) -> tuple[str, dict]:
    encoded_model = urllib.parse.quote(model, safe="")
    url = f"{base_url.rstrip('/')}/models/{encoded_model}:generateContent?key={urllib.parse.quote(api_key)}"
    payload = gemini_payload(messages, temperature, top_p, max_output_tokens)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    last_error = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            obj = json.loads(raw)
            parts = obj["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts).strip()
            return text, obj
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"Gemini API HTTP {exc.code}: {body}")
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= retries:
                raise last_error from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            last_error = RuntimeError(f"Gemini API request failed: {exc}")
            if attempt >= retries:
                raise last_error from exc
        time.sleep(min(30, 2**attempt))
    raise last_error or RuntimeError("Gemini API request failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--test", type=Path, default=DATA_DIR / "test_answer_sheet.csv")
    parser.add_argument("--truth", type=Path, default=DATA_DIR / "test_truth.csv")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--base-url", default=os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--pdf-char-limit", type=int, default=5000)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--tag", default="no_zh_gemini25pro_top3")
    parser.add_argument("--include-langs", nargs="*", default=[])
    parser.add_argument("--exclude-langs", nargs="*", default=["chinese"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=True)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key and not args.dry_run:
        raise RuntimeError("GEMINI_API_KEY is not configured in .env")

    train_rows = filter_languages(read_csv(args.train), args.include_langs, args.exclude_langs)
    test_rows = filter_languages(read_csv(args.test), args.include_langs, args.exclude_langs)
    truth_rows = filter_languages(read_csv(args.truth), args.include_langs, args.exclude_langs)
    if args.limit > 0:
        test_rows = test_rows[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix_count = len(test_rows) if args.limit <= 0 else args.limit
    suffix = f"{suffix_count}_{args.tag}" if args.tag else str(suffix_count)
    predictions_path = args.out_dir / f"predictions_{suffix}.csv"
    trace_path = args.out_dir / f"trace_{suffix}.jsonl"
    metrics_path = args.out_dir / f"metrics_{suffix}.json"

    train_page_texts = [
        extract_page_text(compact(row.get("file_stem")), compact(row.get("lang")) or "english", compact(row.get("page")))
        for row in train_rows
    ]
    test_page_texts = [
        extract_page_text(compact(row.get("file_stem")), compact(row.get("lang")) or "english", compact(row.get("page")))
        for row in test_rows
    ]

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
    train_docs = [retrieval_text(row, page_text) for row, page_text in zip(train_rows, train_page_texts)]
    test_docs = [retrieval_text(row, page_text) for row, page_text in zip(test_rows, test_page_texts)]
    train_matrix = vectorizer.fit_transform(train_docs)
    test_matrix = vectorizer.transform(test_docs)

    completed = load_trace_predictions(trace_path) if args.resume else {}
    if trace_path.exists() and not args.resume:
        trace_path.unlink()

    output_by_index = {}
    started = time.time()
    trace_lock = threading.Lock()
    print_lock = threading.Lock()

    for idx, row in enumerate(test_rows, 1):
        if idx not in completed:
            continue
        out_row = dict(row)
        out_row.update(completed[idx])
        output_by_index[idx] = out_row
        print(f"{idx}/{len(test_rows)} {row.get('file_stem')} {row.get('sid')} -> {out_row.get('pred_label')} (resumed)", flush=True)

    def process_one(idx: int, row: dict, page_text: str) -> tuple[int, dict]:
        similarities = cosine_similarity(test_matrix[idx - 1], train_matrix).ravel()
        examples = select_examples(row, similarities, train_rows, train_page_texts, args.top_k)
        metric_prior = label_prior(row, train_rows)
        messages = build_prompt(row, page_text, examples, metric_prior, args.pdf_char_limit)

        if args.dry_run:
            raw = json.dumps({"pred_label": "no", "category_match": "N/A", "unit_match": "N/A", "reason": "dry run"})
            response_obj = {}
        else:
            raw, response_obj = call_gemini(
                api_key,
                args.base_url,
                args.model,
                messages,
                args.temperature,
                args.top_p,
                args.max_output_tokens,
                args.timeout,
                args.retries,
            )

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

        out_row = dict(row)
        out_row.update(
            {
                "pred_label": pred_label,
                "category_match": category_match,
                "unit_match": unit_match,
            }
        )
        trace_obj = {
            "index": idx,
            "key": key(row),
            "retrieved_examples": [
                {
                    "similarity": item["similarity"],
                    "file_stem": item["row"].get("file_stem", ""),
                    "sid": item["row"].get("sid", ""),
                    "cid": item["row"].get("cid", ""),
                    "label": item["row"].get("label", ""),
                    "same_metric": item["same_metric"],
                }
                for item in examples
            ],
            "metric_label_prior": metric_prior,
            "raw": raw,
            "parsed": parsed,
            "prediction": {
                "pred_label": pred_label,
                "category_match": category_match,
                "unit_match": unit_match,
            },
            "usage": response_obj.get("usageMetadata", {}),
        }
        with trace_lock:
            append_jsonl(trace_path, trace_obj)
        return idx, out_row

    pending = [
        (idx, row, page_text)
        for idx, (row, page_text) in enumerate(zip(test_rows, test_page_texts), 1)
        if idx not in completed
    ]
    done_count = len(output_by_index)
    if pending:
        workers = max(1, args.concurrency)
        print(f"running {len(pending)} pending rows with concurrency={workers}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_one, idx, row, page_text): (idx, row)
                for idx, row, page_text in pending
            }
            for future in as_completed(futures):
                idx, row = futures[future]
                try:
                    idx, out_row = future.result()
                except Exception as exc:
                    with print_lock:
                        print(f"ERROR index={idx} {row.get('file_stem')} {row.get('sid')}: {exc}", flush=True)
                    continue
                output_by_index[idx] = out_row
                done_count += 1
                with print_lock:
                    print(
                        f"{done_count}/{len(test_rows)} index={idx} "
                        f"{row.get('file_stem')} {row.get('sid')} -> {out_row.get('pred_label')}",
                        flush=True,
                    )

    fieldnames = list(test_rows[0].keys()) if test_rows else []
    for field in ["pred_label", "category_match", "unit_match"]:
        if field not in fieldnames:
            fieldnames.append(field)
    output_rows = [output_by_index[idx] for idx in sorted(output_by_index)]
    write_csv(predictions_path, output_rows, fieldnames)

    metrics = evaluate(output_rows, truth_rows)
    metrics.update(
        {
            "model": args.model,
            "limit": args.limit,
            "top_k": args.top_k,
            "tag": args.tag,
            "resume": args.resume,
            "retries": args.retries,
            "concurrency": args.concurrency,
            "runtime_sec": round(time.time() - started, 2),
            "predictions_path": str(predictions_path),
            "trace_path": str(trace_path),
        }
    )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ["evaluated", "accuracy", "micro_f1", "macro_f1"]}, indent=2))
    print(f"wrote {predictions_path}")
    print(f"wrote {trace_path}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
