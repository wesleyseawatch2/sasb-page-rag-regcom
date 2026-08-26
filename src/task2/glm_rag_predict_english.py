from __future__ import annotations

import argparse
import csv
import http.client
import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "datasets" / "english_prediction"
OUT_DIR = ROOT / "data" / "predictions" / "english_prediction_glm_rag"
PAGES_DIR = ROOT / "pages"
SOURCE_PDF_DIR = ROOT / "Training Set" / "PDF"

LABELS = ["yes", "yes but not complete", "no"]
MATCHES = ["yes", "no", "N/A"]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_trace_predictions(path: Path) -> dict[int, dict]:
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
            idx = obj.get("index")
            prediction = obj.get("prediction")
            if isinstance(idx, int) and isinstance(prediction, dict):
                completed[idx] = prediction
    return completed


def compact(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def source_pdf_stem(file_stem: str) -> str:
    match = re.match(r"(.+)_([0-9]+)$", str(file_stem))
    return match.group(1) if match else str(file_stem)


def extract_page_text(file_stem: str, lang: str = "english", page: str = "") -> str:
    single_page_pdf = PAGES_DIR / f"{file_stem}.pdf"
    try:
        if single_page_pdf.exists() and single_page_pdf.stat().st_size > 0:
            with fitz.open(str(single_page_pdf)) as doc:
                return "\n".join(p.get_text("text") for p in doc).strip()
    except Exception:
        pass

    try:
        source_pdf = SOURCE_PDF_DIR / lang / f"{source_pdf_stem(file_stem)}.pdf"
        page_index = int(float(page)) - 1
        if source_pdf.exists() and page_index >= 0:
            with fitz.open(str(source_pdf)) as doc:
                if page_index < doc.page_count:
                    return doc[page_index].get_text("text").strip()
    except Exception:
        pass
    return ""


def retrieval_text(row: dict, page_text: str = "") -> str:
    fields = [
        "topic",
        "metric_description",
        "metric_code",
        "sid",
        "sasb_category",
        "sasb_unit_of_measure",
        "sasb_key_terms",
        "sasb_what_counts",
    ]
    return "\n".join([compact(row.get(field)) for field in fields] + [compact(page_text[:1500])])


def row_brief(row: dict) -> str:
    return "\n".join(
        [
            f"lang: {compact(row.get('lang'))}",
            f"cid: {compact(row.get('cid'))}",
            f"file_stem: {compact(row.get('file_stem'))}",
            f"page: {compact(row.get('page'))}",
            f"metric_code: {compact(row.get('metric_code') or row.get('sid'))}",
            f"topic: {compact(row.get('topic'))}",
            f"metric_description: {compact(row.get('metric_description'))}",
            f"sasb_category: {compact(row.get('sasb_category'))}",
            f"sasb_unit_of_measure: {compact(row.get('sasb_unit_of_measure'))}",
            f"sasb_key_terms: {compact(row.get('sasb_key_terms'))}",
            f"sasb_what_counts: {compact(row.get('sasb_what_counts'))}",
        ]
    )


def derive_matches(row: dict, label: str) -> tuple[str, str]:
    label = normalize_label(label)
    if label == "no":
        return "N/A", "N/A"
    if compact(row.get("sasb_category")).lower() == "discussion and analysis":
        return "yes", "yes"
    if label == "yes":
        return "yes", "yes"
    return "yes", "no"


def normalize_label(value: str) -> str:
    text = compact(value).lower()
    text = text.replace("_", " ").replace("-", " ")
    if text in LABELS:
        return text
    if "yes but not complete" in text or "partial" in text or "incomplete" in text:
        return "yes but not complete"
    if re.search(r"\byes\b", text):
        return "yes"
    return "no"


def normalize_match(value: str) -> str:
    text = compact(value).lower()
    if text in {"n/a", "na", "not applicable", "none", ""}:
        return "N/A"
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return "N/A"


def parse_jsonish(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


def metric_id(row: dict) -> str:
    return compact(row.get("metric_code") or row.get("sid"))


def evidence_snippet(row: dict, page_text: str, max_chars: int = 1200) -> str:
    text = compact(page_text)
    if len(text) <= max_chars:
        return text

    terms = []
    terms.extend([term.strip() for term in compact(row.get("sasb_key_terms")).split("|") if term.strip()])
    terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9&/-]{3,}", compact(row.get("metric_description"))))
    terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9&/-]{3,}", compact(row.get("topic"))))
    terms = [term.lower() for term in terms if len(term) >= 4]

    if not terms:
        return text[:max_chars]

    best_start = 0
    best_score = -1
    window = max_chars
    stride = max(300, window // 3)
    lower = text.lower()
    for start in range(0, max(1, len(text) - window + 1), stride):
        chunk = lower[start : start + window]
        score = sum(1 for term in terms if term in chunk)
        if score > best_score:
            best_score = score
            best_start = start
    return text[best_start : best_start + max_chars]


def label_prior(test_row: dict, train_rows: list[dict]) -> dict:
    same_metric = [
        normalize_label(row.get("label", ""))
        for row in train_rows
        if metric_id(row) == metric_id(test_row)
    ]
    labels = same_metric or [normalize_label(row.get("label", "")) for row in train_rows]
    counts = Counter(labels)
    return {label: counts.get(label, 0) for label in LABELS}


def select_examples(
    test_row: dict,
    similarities,
    train_rows: list[dict],
    train_page_texts: list[str],
    top_k: int,
) -> list[dict]:
    same_metric_indices = [
        i for i, row in enumerate(train_rows)
        if metric_id(row) == metric_id(test_row)
    ]
    all_indices = list(range(len(train_rows)))
    selected = []
    used = set()

    # Label-balanced first: the top similar same-metric example for each class.
    for label in LABELS:
        candidates = [
            i for i in same_metric_indices
            if normalize_label(train_rows[i].get("label", "")) == label
        ]
        if not candidates:
            candidates = [
                i for i in all_indices
                if normalize_label(train_rows[i].get("label", "")) == label
            ]
        if candidates:
            best = max(candidates, key=lambda i: similarities[i])
            selected.append(best)
            used.add(best)
        if len(selected) >= top_k:
            break

    # Fill any gaps with the most similar same-metric examples, then global.
    for pool in [same_metric_indices, all_indices]:
        for i in sorted(pool, key=lambda j: similarities[j], reverse=True):
            if i not in used:
                selected.append(i)
                used.add(i)
            if len(selected) >= top_k:
                break
        if len(selected) >= top_k:
            break

    return [
        {
            "row": train_rows[i],
            "page_text": train_page_texts[i],
            "similarity": float(similarities[i]),
            "same_metric": metric_id(train_rows[i]) == metric_id(test_row),
        }
        for i in selected[:top_k]
    ]


def build_prompt(
    test_row: dict,
    test_page_text: str,
    examples: list[dict],
    metric_prior: dict,
    pdf_char_limit: int,
) -> list[dict]:
    system = """You are RegComAgent for SASB page-level verification.
Judge only the single PDF page provided. This is page-level dataset matching,
not a full-report SASB audit.
Use the retrieved few-shot examples to calibrate this dataset's yes / yes but
not complete / no label boundary.

Return compact JSON only:
{
  "pred_label": "yes|yes but not complete|no",
  "category_match": "yes|no|N/A",
  "unit_match": "yes|no|N/A",
  "evidence": "short page evidence",
  "reason": "short reason"
}

Rules:
- pred_label=yes if the page contains relevant disclosure for the requested
  SASB metric at page level.
- pred_label=yes but not complete if the page is relevant but partial, proxy-only, incomplete, or wrong-unit.
- pred_label=no if the exact metric is not semantically disclosed.
- If pred_label=no, category_match=N/A and unit_match=N/A.
- Treat sasb_what_counts as guidance, not a strict checklist. Do not require
  every sub-bullet on one page.
- For Discussion and Analysis metrics with relevant narrative, prefer yes;
  category_match=yes and unit_match=yes.
- Zero/no-event statements can satisfy quantitative event/count metrics."""

    example_blocks = []
    for i, item in enumerate(examples, 1):
        row = item["row"]
        category_match, unit_match = derive_matches(row, row.get("label", ""))
        example_blocks.append(
            f"""Example {i} similarity={item['similarity']:.4f} same_metric={item['same_metric']}
Task:
{row_brief(row)}
Evidence snippet:
{evidence_snippet(row, item['page_text'], 1200)}
Gold output:
{{"pred_label": "{normalize_label(row.get('label', ''))}", "category_match": "{category_match}", "unit_match": "{unit_match}"}}"""
        )

    user = f"""Retrieved few-shot examples:

{chr(10).join(example_blocks)}

Training label distribution for this metric:
{json.dumps(metric_prior, ensure_ascii=False)}

Use the distribution only as calibration. The target page evidence still decides.

Now predict this test row.

Task:
{row_brief(test_row)}

PDF page text:
{compact(test_page_text[:pdf_char_limit])}

Return JSON only."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_glm(
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict],
    temperature: float,
    timeout: int,
    retries: int,
) -> tuple[str, dict]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.8,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= retries:
                raise RuntimeError(f"GLM API HTTP {exc.code}: {body}") from exc
            time.sleep(min(30, 2 ** attempt))
        except (urllib.error.URLError, http.client.RemoteDisconnected, TimeoutError, socket.timeout) as exc:
            if attempt >= retries:
                raise RuntimeError(f"GLM API request failed after {retries + 1} attempts: {exc}") from exc
            time.sleep(min(30, 2 ** attempt))
    obj = json.loads(raw)
    content = obj["choices"][0]["message"]["content"]
    return content, obj


def key(row: dict) -> tuple[str, str, str]:
    return compact(row.get("file_stem")), compact(row.get("sid")), compact(row.get("cid"))


def add_occurrence(rows: list[dict]) -> dict[tuple[str, str, str, int], dict]:
    counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    out = {}
    for row in rows:
        base = key(row)
        occurrence = counts[base]
        counts[base] += 1
        out[(*base, occurrence)] = row
    return out


def evaluate(pred_rows: list[dict], truth_rows: list[dict]) -> dict:
    pred_by_key = add_occurrence(pred_rows)
    truth_by_key = add_occurrence(truth_rows)
    y_true = []
    y_pred = []
    for k, pred in pred_by_key.items():
        truth = truth_by_key.get(k)
        if not truth:
            continue
        y_true.append(normalize_label(truth.get("label", "")))
        y_pred.append(normalize_label(pred.get("pred_label", "")))
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    return {
        "evaluated": len(y_true),
        "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0,
        "micro_f1": f1_score(y_true, y_pred, labels=LABELS, average="micro", zero_division=0) if y_true else 0.0,
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0) if y_true else 0.0,
        "truth_counts": dict(Counter(y_true)),
        "pred_counts": dict(Counter(y_pred)),
        "classification_report": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": {
            "labels": LABELS,
            "rows_gold_cols_pred": matrix.tolist(),
        },
    }


def filter_languages(rows: list[dict], include: list[str], exclude: list[str]) -> list[dict]:
    include_set = {lang.strip().lower() for lang in include if lang.strip()}
    exclude_set = {lang.strip().lower() for lang in exclude if lang.strip()}
    filtered = []
    for row in rows:
        lang = compact(row.get("lang")).lower()
        if include_set and lang not in include_set:
            continue
        if exclude_set and lang in exclude_set:
            continue
        filtered.append(row)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--test", type=Path, default=DATA_DIR / "test_answer_sheet.csv")
    parser.add_argument("--truth", type=Path, default=DATA_DIR / "test_truth.csv")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--base-url", default=os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--pdf-char-limit", type=int, default=5000)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--tag", default="tuned")
    parser.add_argument("--include-langs", nargs="*", default=[])
    parser.add_argument("--exclude-langs", nargs="*", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=True)
    api_key = os.environ.get("GLM_API_KEY", "").strip()
    if not api_key and not args.dry_run:
        raise RuntimeError("GLM_API_KEY is not configured in .env")

    train_rows = read_csv(args.train)
    test_rows = read_csv(args.test)
    truth_rows = read_csv(args.truth)

    train_rows = filter_languages(train_rows, args.include_langs, args.exclude_langs)
    test_rows = filter_languages(test_rows, args.include_langs, args.exclude_langs)
    truth_rows = filter_languages(truth_rows, args.include_langs, args.exclude_langs)

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
            raw, response_obj = call_glm(
                api_key,
                args.base_url,
                args.model,
                messages,
                args.temperature,
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
            "usage": response_obj.get("usage", {}),
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
                        print(
                            f"ERROR index={idx} {row.get('file_stem')} {row.get('sid')}: {exc}",
                            flush=True,
                        )
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
