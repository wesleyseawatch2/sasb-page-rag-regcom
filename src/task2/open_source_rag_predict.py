from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from glm_rag_predict_english import (
    DATA_DIR,
    LABELS,
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


OUT_DIR = GLM_OUT_DIR.parent / "open_source_rag_baseline"


class LocalChatModel:
    def __init__(
        self,
        model_name: str,
        device_map: str,
        max_input_tokens: int,
        trust_remote_code: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Please install torch and transformers before running this baseline.") from exc

        self.torch = torch
        self.max_input_tokens = max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        kwargs = {
            "torch_dtype": "auto",
            "trust_remote_code": trust_remote_code,
        }
        if device_map:
            kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if not device_map:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _prompt(self, messages: list[dict]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages) + "\n\nASSISTANT:\n"

    def _device(self):
        if hasattr(self.model, "device") and str(self.model.device) != "meta":
            return self.model.device
        return next(self.model.parameters()).device

    def generate(
        self,
        messages: list[dict],
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> tuple[str, dict]:
        prompt = self._prompt(messages)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = {k: v.to(self._device()) for k, v in inputs.items()}
        do_sample = temperature > 0
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs.update({"temperature": temperature, "top_p": top_p})

        started = time.time()
        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, **kwargs)
        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        usage = {
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(generated.shape[0]),
            "generation_sec": round(time.time() - started, 2),
        }
        return raw, usage


def model_tag(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--test", type=Path, default=DATA_DIR / "test_answer_sheet.csv")
    parser.add_argument("--truth", type=Path, default=DATA_DIR / "test_truth.csv")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-input-tokens", type=int, default=10000)
    parser.add_argument("--pdf-char-limit", type=int, default=5000)
    parser.add_argument("--tag", default="")
    parser.add_argument("--include-langs", nargs="*", default=[])
    parser.add_argument("--exclude-langs", nargs="*", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_rows = filter_languages(read_csv(args.train), args.include_langs, args.exclude_langs)
    test_rows = filter_languages(read_csv(args.test), args.include_langs, args.exclude_langs)
    truth_rows = filter_languages(read_csv(args.truth), args.include_langs, args.exclude_langs)
    if args.limit > 0:
        test_rows = test_rows[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix_count = len(test_rows) if args.limit <= 0 else args.limit
    tag = args.tag or model_tag(args.model)
    suffix = f"{suffix_count}_{tag}" if tag else str(suffix_count)
    predictions_path = args.out_dir / f"predictions_{suffix}.csv"
    trace_path = args.out_dir / f"trace_{suffix}.jsonl"
    metrics_path = args.out_dir / f"metrics_{suffix}.json"

    print("extracting page text...", flush=True)
    train_page_texts = [
        extract_page_text(compact(row.get("file_stem")), compact(row.get("lang")) or "english", compact(row.get("page")))
        for row in train_rows
    ]
    test_page_texts = [
        extract_page_text(compact(row.get("file_stem")), compact(row.get("lang")) or "english", compact(row.get("page")))
        for row in test_rows
    ]

    print("building tf-idf retrieval index...", flush=True)
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
    train_docs = [retrieval_text(row, page_text) for row, page_text in zip(train_rows, train_page_texts)]
    test_docs = [retrieval_text(row, page_text) for row, page_text in zip(test_rows, test_page_texts)]
    train_matrix = vectorizer.fit_transform(train_docs)
    test_matrix = vectorizer.transform(test_docs)

    completed = load_trace_predictions(trace_path) if args.resume else {}
    if trace_path.exists() and not args.resume:
        trace_path.unlink()

    model = None
    if not args.dry_run:
        print(f"loading local open-source model: {args.model}", flush=True)
        model = LocalChatModel(
            model_name=args.model,
            device_map=args.device_map,
            max_input_tokens=args.max_input_tokens,
            trust_remote_code=args.trust_remote_code,
        )

    started = time.time()
    output_by_index = {}
    for idx, row in enumerate(test_rows, 1):
        if idx in completed:
            out_row = dict(row)
            out_row.update(completed[idx])
            output_by_index[idx] = out_row
            print(f"{idx}/{len(test_rows)} {row.get('file_stem')} {row.get('sid')} -> {out_row.get('pred_label')} (resumed)", flush=True)

    pending = [
        (idx, row, page_text)
        for idx, (row, page_text) in enumerate(zip(test_rows, test_page_texts), 1)
        if idx not in completed
    ]

    for offset, (idx, row, page_text) in enumerate(pending, 1):
        similarities = cosine_similarity(test_matrix[idx - 1], train_matrix).ravel()
        examples = select_examples(row, similarities, train_rows, train_page_texts, args.top_k)
        metric_prior = label_prior(row, train_rows)
        messages = build_prompt(row, page_text, examples, metric_prior, args.pdf_char_limit)

        if args.dry_run:
            raw = json.dumps({"pred_label": "no", "category_match": "N/A", "unit_match": "N/A", "reason": "dry run"})
            usage = {}
        else:
            assert model is not None
            raw, usage = model.generate(
                messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
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
        output_by_index[idx] = out_row

        append_jsonl(
            trace_path,
            {
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
                "usage": usage,
            },
        )
        done_count = len(completed) + offset
        print(f"{done_count}/{len(test_rows)} index={idx} {row.get('file_stem')} {row.get('sid')} -> {pred_label}", flush=True)

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
            "tag": tag,
            "resume": args.resume,
            "truth_counts": metrics.get("truth_counts", {}),
            "pred_counts": metrics.get("pred_counts", {}),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "max_input_tokens": args.max_input_tokens,
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
