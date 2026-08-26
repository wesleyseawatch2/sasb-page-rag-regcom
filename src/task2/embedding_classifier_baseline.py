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
    compact,
    derive_matches,
    evaluate,
    extract_page_text,
    filter_languages,
    normalize_label,
    read_csv,
    retrieval_text,
    write_csv,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "predictions" / "embedding_classifier_baseline"


def safe_tag(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def label_to_matches(row: dict, pred_label: str) -> tuple[str, str]:
    return derive_matches(row, pred_label)


def classification_text(row: dict, page_text: str, pdf_char_limit: int) -> str:
    return retrieval_text(row, page_text[:pdf_char_limit])


def encode_sentence_transformer(
    model_name: str,
    train_texts: list[str],
    test_texts: list[str],
    batch_size: int,
    local_files_only: bool,
    max_length: int,
):
    try:
        from sentence_transformers import SentenceTransformer

        kwargs = {}
        if local_files_only:
            kwargs["local_files_only"] = True
        try:
            model = SentenceTransformer(model_name, **kwargs)
        except TypeError:
            model = SentenceTransformer(model_name)

        train_x = model.encode(
            train_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        test_x = model.encode(
            test_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return train_x, test_x
    except Exception as exc:
        print(f"SentenceTransformer loader failed, falling back to HF mean pooling: {exc}", flush=True)
        return encode_hf_mean_pooling(
            model_name=model_name,
            train_texts=train_texts,
            test_texts=test_texts,
            batch_size=batch_size,
            local_files_only=local_files_only,
            max_length=max_length,
        )


def encode_hf_mean_pooling(
    model_name: str,
    train_texts: list[str],
    test_texts: list[str],
    batch_size: int,
    local_files_only: bool,
    max_length: int,
):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer, BertModel, PreTrainedTokenizerFast

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    except Exception:
        tokenizer_path = Path(model_name) / "tokenizer.json"
        if not tokenizer_path.exists():
            raise
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_path),
            unk_token="<unk>",
            sep_token="</s>",
            pad_token="<pad>",
            cls_token="<s>",
            mask_token="<mask>",
            bos_token="<s>",
            eos_token="</s>",
        )

    config_path = Path(model_name) / "config.json"
    model_type = ""
    if config_path.exists():
        try:
            model_type = json.loads(config_path.read_text(encoding="utf-8")).get("model_type", "")
        except json.JSONDecodeError:
            model_type = ""
    if model_type == "bert":
        model = BertModel.from_pretrained(model_name, local_files_only=local_files_only)
    else:
        model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    def encode(texts: list[str], name: str):
        vectors = []
        total = len(texts)
        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = F.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy())
            print(f"{name}: {min(start + batch_size, total)}/{total}", flush=True)
        import numpy as np

        return np.vstack(vectors)

    return encode(train_texts, "encode train"), encode(test_texts, "encode test")


def fit_predict(
    encoder: str,
    train_texts: list[str],
    test_texts: list[str],
    y_train: list[str],
    model_name: str,
    batch_size: int,
    local_files_only: bool,
    max_length: int,
):
    if encoder == "tfidf":
        clf = make_pipeline(
            TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True),
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"),
        )
        clf.fit(train_texts, y_train)
        return clf.predict(test_texts), clf.predict_proba(test_texts), list(clf.classes_)

    train_x, test_x = encode_sentence_transformer(
        model_name=model_name,
        train_texts=train_texts,
        test_texts=test_texts,
        batch_size=batch_size,
        local_files_only=local_files_only,
        max_length=max_length,
    )
    clf = make_pipeline(
        StandardScaler(),
        Normalizer(),
        LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", multi_class="auto"),
    )
    clf.fit(train_x, y_train)
    return clf.predict(test_x), clf.predict_proba(test_x), list(clf.classes_)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--test", type=Path, default=DATA_DIR / "test_answer_sheet.csv")
    parser.add_argument("--truth", type=Path, default=DATA_DIR / "test_truth.csv")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--encoder", choices=["sentence-transformer", "tfidf"], default="sentence-transformer")
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pdf-char-limit", type=int, default=5000)
    parser.add_argument("--tag", default="")
    parser.add_argument("--include-langs", nargs="*", default=[])
    parser.add_argument("--exclude-langs", nargs="*", default=[])
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    started = time.time()
    train_rows = filter_languages(read_csv(args.train), args.include_langs, args.exclude_langs)
    test_rows = filter_languages(read_csv(args.test), args.include_langs, args.exclude_langs)
    truth_rows = filter_languages(read_csv(args.truth), args.include_langs, args.exclude_langs)
    if args.limit > 0:
        test_rows = test_rows[: args.limit]

    print(f"train rows: {len(train_rows)}", flush=True)
    print(f"test rows: {len(test_rows)}", flush=True)
    print("extracting page text...", flush=True)
    train_page_texts = [
        extract_page_text(compact(row.get("file_stem")), compact(row.get("lang")) or "english", compact(row.get("page")))
        for row in train_rows
    ]
    test_page_texts = [
        extract_page_text(compact(row.get("file_stem")), compact(row.get("lang")) or "english", compact(row.get("page")))
        for row in test_rows
    ]

    train_texts = [
        classification_text(row, page_text, args.pdf_char_limit)
        for row, page_text in zip(train_rows, train_page_texts)
    ]
    test_texts = [
        classification_text(row, page_text, args.pdf_char_limit)
        for row, page_text in zip(test_rows, test_page_texts)
    ]
    y_train = [normalize_label(row.get("label", "")) for row in train_rows]

    print(f"training {args.encoder} classifier...", flush=True)
    pred_labels, pred_probs, class_order = fit_predict(
        encoder=args.encoder,
        train_texts=train_texts,
        test_texts=test_texts,
        y_train=y_train,
        model_name=args.model,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
        max_length=args.max_length,
    )

    output_rows = []
    for row, pred_label, probs in zip(test_rows, pred_labels, pred_probs):
        pred_label = normalize_label(str(pred_label))
        category_match, unit_match = label_to_matches(row, pred_label)
        out_row = dict(row)
        out_row.update(
            {
                "pred_label": pred_label,
                "category_match": category_match,
                "unit_match": unit_match,
            }
        )
        for cls, prob in zip(class_order, probs):
            out_row[f"score_{safe_tag(cls)}"] = round(float(prob), 6)
        output_rows.append(out_row)

    suffix_count = len(test_rows) if args.limit <= 0 else args.limit
    tag = args.tag or f"{args.encoder}_{safe_tag(args.model)}"
    suffix = f"{suffix_count}_{tag}" if tag else str(suffix_count)
    predictions_path = args.out_dir / f"predictions_{suffix}.csv"
    metrics_path = args.out_dir / f"metrics_{suffix}.json"

    fieldnames = list(test_rows[0].keys()) if test_rows else []
    for field in ["pred_label", "category_match", "unit_match"]:
        if field not in fieldnames:
            fieldnames.append(field)
    for cls in class_order:
        score_field = f"score_{safe_tag(cls)}"
        if score_field not in fieldnames:
            fieldnames.append(score_field)
    write_csv(predictions_path, output_rows, fieldnames)

    metrics = evaluate(output_rows, truth_rows)
    metrics.update(
        {
            "encoder": args.encoder,
            "model": args.model if args.encoder == "sentence-transformer" else "tfidf",
            "classifier": "LogisticRegression(class_weight=balanced)",
            "limit": args.limit,
            "pdf_char_limit": args.pdf_char_limit,
            "max_length": args.max_length,
            "tag": tag,
            "train_counts": dict(Counter(y_train)),
            "runtime_sec": round(time.time() - started, 2),
            "predictions_path": str(predictions_path),
        }
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: metrics[k] for k in ["evaluated", "accuracy", "micro_f1", "macro_f1"]}, indent=2))
    print(f"wrote {predictions_path}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
