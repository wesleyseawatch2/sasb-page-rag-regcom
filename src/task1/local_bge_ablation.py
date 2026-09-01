"""Local A/B experiment with BGE-M3 retrieval and BGE multilingual reranking.

This script deliberately makes no API calls.  It reuses the completed 490-query
diagnostic inputs, recomputes the lexical/rule lanes, adds a BGE-M3 dense lane,
then reranks the fused Top-N text candidates with BGE-Reranker-v2-M3.  The
embedding and reranking stages use a small process pool; each worker loads one
model and limits its PyTorch intra-op threads to avoid CPU oversubscription.

Example (from the repository root)::

    py src/task1/local_bge_ablation.py --workers 2

The first run downloads the two Hugging Face models and can take a while on a
CPU-only machine.  Per-report BGE page embeddings are cached, so subsequent
runs only encode new/changed reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

# Import the pipeline's tested lexical/ranking helpers instead of duplicating
# their scoring definitions.
from task1_pipeline import (
    build_lexical_index,
    hit_at,
    rankings_from_scores,
    reciprocal_rank,
    reciprocal_rank_fusion,
    unit_rule_score,
    weighted_query_text,
)


ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def short_hash(*parts: object) -> str:
    value = "\x1f".join(compact(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def embedding_text(model_name: str, text: object, role: str) -> str:
    """Add the task prefix expected by instruction-tuned embedding models.

    BGE-M3 is used without a prefix, while the multilingual E5 family was
    trained with explicit ``query:``/``passage:`` prefixes.  Keeping this
    normalization in one place lets the same A/B runner compare both models
    without silently using the wrong E5 input format.
    """
    value = unicodedata.normalize("NFKC", str(text or "")).strip()
    if "e5" in model_name.lower():
        prefix = "query: " if role == "query" else "passage: "
        return prefix + value
    return value


def configure_torch_threads(thread_count: int) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch

        torch.set_num_threads(max(1, int(thread_count)))
        torch.set_num_interop_threads(1)
    except Exception:
        # The actual model import below will provide a useful error if torch is
        # unavailable; this helper should not hide it.
        pass


def _encode_embedding_shard(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker: load BGE-M3 once, encode all pages and assigned query texts."""
    configure_torch_threads(int(payload["torch_threads"]))
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("Install sentence-transformers to run BGE-M3") from exc

    device = str(payload.get("device", "cpu"))
    model = SentenceTransformer(payload["model_name"], device=device)
    max_length = int(payload["max_length"])
    if max_length > 0:
        model.max_seq_length = max_length
    batch_size = int(payload["batch_size"])
    cache_dir = Path(payload["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", payload["model_name"])
    page_limit = int(payload["page_char_limit"])
    query_embeddings: dict[str, np.ndarray] = {}
    page_files: list[str] = []
    page_count = 0
    cached_pages = 0

    for group in payload["groups"]:
        key = tuple(group["key"])
        pages = group["pages"]
        texts = [embedding_text(payload["model_name"], str(row.get("text", ""))[:page_limit], "passage") for row in pages]
        digest = short_hash(payload["model_name"], *key, group.get("pdf_sha256", ""), page_limit, max_length)
        cache_file = cache_dir / f"{model_slug[:24]}__{digest}__{page_limit}__{max_length}.npy"
        if cache_file.exists():
            loaded = np.load(cache_file, allow_pickle=False)
            if len(loaded) == len(texts):
                page_vectors = np.asarray(loaded)
                cached_pages += len(texts)
            else:
                page_vectors = np.asarray(
                    model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size)
                )
                np.save(cache_file, page_vectors)
        else:
            page_vectors = np.asarray(
                model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size)
            )
            np.save(cache_file, page_vectors)
        page_files.append(str(cache_file))
        page_count += len(texts)

        query_texts = group["query_texts"]
        if query_texts:
            vectors = np.asarray(
                model.encode(
                    [embedding_text(payload["model_name"], item["text"], "query") for item in query_texts],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=batch_size,
                )
            )
            for item, vector in zip(query_texts, vectors):
                query_embeddings[str(item["sample_id"])] = np.asarray(vector, dtype=np.float32)

    return {
        "query_embeddings": query_embeddings,
        "page_files": page_files,
        "page_count": page_count,
        "cached_pages": cached_pages,
    }


def _rerank_shard(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Worker: score query/page pairs with BGE-Reranker-v2-M3."""
    configure_torch_threads(int(payload["torch_threads"]))
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("Install transformers and torch to run the BGE reranker") from exc

    model_name = payload["model_name"]
    device = str(payload.get("device", "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to(device)
    batch_size = int(payload["batch_size"])
    max_length = int(payload["max_length"])
    output: list[dict[str, Any]] = []
    for row in payload["rows"]:
        candidates = row["candidates"]
        query_text = str(row["query_text"])
        scores: list[float] = []
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            passages = [str(item.get("text", "")) for item in batch]
            encoded = tokenizer(
                [query_text] * len(passages),
                passages,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            if device != "cpu":
                encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded).logits
            scores.extend([float(value) for value in logits.reshape(-1).cpu().tolist()])
        order = sorted(range(len(candidates)), key=lambda index: (-scores[index], int(candidates[index]["page"])))
        ranked = [
            {
                "page": int(candidates[index]["page"]),
                "score": round(scores[index], 8),
                "rank": rank,
            }
            for rank, index in enumerate(order, 1)
        ]
        output.append({"sample_id": row["sample_id"], "ranking": ranked})
    return output


def split_shards(items: list[dict[str, Any]], workers: int, weight_key: str | None = None) -> list[list[dict[str, Any]]]:
    workers = max(1, min(int(workers), len(items) or 1))
    shards: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    loads = [0] * workers
    ordered = sorted(items, key=lambda item: -(int(item.get(weight_key, 1)) if weight_key else 1))
    for item in ordered:
        index = min(range(workers), key=lambda candidate: loads[candidate])
        shards[index].append(item)
        loads[index] += int(item.get(weight_key, 1)) if weight_key else 1
    return [shard for shard in shards if shard]


def metric_summary(rows: list[dict[str, Any]], field: str) -> dict[str, float | int]:
    usable = [row for row in rows if row.get("gold_pages") and not row.get("error")]
    values: dict[str, list[float]] = {"h1": [], "h5": [], "h10": [], "near1": [], "mrr": []}
    for row in usable:
        gold = {int(page) for page in row["gold_pages"]}
        predicted = [int(item["page"]) for item in row.get(field, [])]
        values["h1"].append(hit_at(predicted, gold, 1))
        values["h5"].append(hit_at(predicted, gold, 5))
        values["h10"].append(hit_at(predicted, gold, 10))
        values["near1"].append(float(predicted and any(abs(predicted[0] - page) <= 1 for page in gold)))
        values["mrr"].append(reciprocal_rank(predicted, gold))
    return {
        "evaluated_non_empty_gold": len(usable),
        "excluded_empty_gold": sum(not row.get("gold_pages") for row in rows),
        "hit_at_1": statistics.fmean(values["h1"]) if usable else 0.0,
        "hit_at_5": statistics.fmean(values["h5"]) if usable else 0.0,
        "hit_at_10": statistics.fmean(values["h10"]) if usable else 0.0,
        "near_at_1": statistics.fmean(values["near1"]) if usable else 0.0,
        "mrr": statistics.fmean(values["mrr"]) if usable else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, default=Path("runs/task1-dense-optimized/retrieval.jsonl"))
    parser.add_argument("--pages", type=Path, default=Path("runs/task1-json-source-v2/pages.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("runs/task1-bge-ablation/results.jsonl"))
    parser.add_argument("--metrics-output", type=Path, default=Path("artifacts/metrics/task1_bge_ablation.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/task1-bge-m3"))
    parser.add_argument("--bge-model", default="BAAI/bge-m3")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for each worker (cpu or cuda). Use one worker for a single GPU.",
    )
    parser.add_argument("--torch-threads", type=int, default=0, help="Per-worker torch threads (0 = cpu_count/workers).")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--bge-max-length", type=int, default=1024)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--page-char-limit", type=int, default=6000)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--max-queries", type=int, default=0, help="Debug pilot; 0 runs all queries.")
    parser.add_argument("--skip-reranker", action="store_true")
    args = parser.parse_args()

    retrieval_rows = read_jsonl(args.retrieval)
    if args.max_queries > 0:
        retrieval_rows = retrieval_rows[: args.max_queries]
    page_rows = read_jsonl(args.pages)
    pages_by_report: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for page in page_rows:
        pages_by_report[(str(page["lang"]), str(page["report_stem"]))].append(page)
    for rows in pages_by_report.values():
        rows.sort(key=lambda row: int(row["page"]))

    # Rebuild the lexical lanes exactly from the page text, then add BGE as a
    # fourth lane.  This avoids treating the old MiniLM lane as part of BGE.
    base_rows: list[dict[str, Any]] = []
    lexical_cache: dict[tuple[str, str], dict[str, Any]] = {}
    text_cache: dict[tuple[str, str], list[str]] = {}
    group_records: dict[tuple[str, str], dict[str, Any]] = {}
    for row in retrieval_rows:
        key = (str(row["lang"]), str(row["report_stem"]))
        pages = pages_by_report.get(key, [])
        if not pages:
            base_rows.append({**row, "error": "no_indexed_pages"})
            continue
        texts = text_cache.setdefault(
            key, [str(page.get("text", ""))[: args.page_char_limit] for page in pages]
        )
        if key not in lexical_cache:
            lexical_cache[key] = build_lexical_index(texts, 50000, 100000)
        index = lexical_cache[key]
        query_text = str(row.get("query_text", "")) or weighted_query_text(row)
        from sklearn.metrics.pairwise import cosine_similarity

        word_query = index["word_vectorizer"].transform([query_text])
        tfidf = cosine_similarity(word_query, index["word_matrix"]).ravel()
        char_query = index["char_vectorizer"].transform([unicodedata.normalize("NFKC", query_text).lower()])
        char_tfidf = cosine_similarity(char_query, index["char_matrix"]).ravel()
        rule = np.asarray([unit_rule_score(row, text) for text in texts], dtype=float)
        rankings: dict[str, list[dict[str, Any]]] = {
            "tfidf": rankings_from_scores(tfidf, pages),
            "char_tfidf": rankings_from_scores(char_tfidf, pages),
        }
        rule_ranking = rankings_from_scores(rule, pages, positive_only=True)
        if rule_ranking:
            rankings["rules"] = rule_ranking
        record = group_records.setdefault(
            key,
            {
                "key": list(key),
                "pdf_sha256": pages[0].get("pdf_sha256", ""),
                "pages": pages,
                "query_texts": [],
            },
        )
        record["query_texts"].append({"sample_id": row["sample_id"], "text": query_text})
        base_rows.append({**row, "_pages": pages, "_texts": texts, "_rankings": rankings, "_query_text": query_text})

    workers = max(1, int(args.workers))
    cpu_count = os.cpu_count() or 1
    torch_threads = int(args.torch_threads) if args.torch_threads > 0 else max(1, cpu_count // workers)
    groups = list(group_records.values())
    # Weight by page count so long reports are balanced across workers.
    for group in groups:
        group["_missing_weight"] = len(group["pages"])
    embed_shards = split_shards(groups, workers, weight_key="_missing_weight")
    payloads = [
        {
            "model_name": args.bge_model,
            "device": args.device,
            "cache_dir": str(args.cache_dir.resolve()),
            "page_char_limit": args.page_char_limit,
            "max_length": args.bge_max_length,
            "batch_size": args.embedding_batch_size,
            "torch_threads": torch_threads,
            "groups": shard,
        }
        for shard in embed_shards
    ]
    print(f"BGE-M3 embedding: {len(groups)} reports / {sum(len(g['pages']) for g in groups)} pages, {len(payloads)} workers")
    query_vectors: dict[str, np.ndarray] = {}
    cached_pages = 0
    with ProcessPoolExecutor(max_workers=len(payloads)) as pool:
        futures = [pool.submit(_encode_embedding_shard, payload) for payload in payloads]
        for future in as_completed(futures):
            result = future.result()
            query_vectors.update(result["query_embeddings"])
            cached_pages += int(result["cached_pages"])
    print(f"BGE-M3 page embeddings ready; reused {cached_pages} cached page vectors")

    # Load the per-report vectors written by workers and make BGE rankings.
    vector_by_group: dict[tuple[str, str], np.ndarray] = {}
    for group in groups:
        digest = short_hash(args.bge_model, *group["key"], group.get("pdf_sha256", ""), args.page_char_limit, args.bge_max_length)
        model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.bge_model)
        cache_file = args.cache_dir / f"{model_slug[:24]}__{digest}__{args.page_char_limit}__{args.bge_max_length}.npy"
        vector_by_group[tuple(group["key"])] = np.asarray(np.load(cache_file, allow_pickle=False))

    output_rows: list[dict[str, Any]] = []
    for row in base_rows:
        if row.get("error"):
            output_rows.append(row)
            continue
        key = (str(row["lang"]), str(row["report_stem"]))
        page_vectors = vector_by_group[key]
        query_vector = query_vectors[str(row["sample_id"])]
        bge_scores = page_vectors @ query_vector
        rankings = dict(row["_rankings"])
        rankings["bge_m3"] = rankings_from_scores(bge_scores, row["_pages"])
        fused = reciprocal_rank_fusion(
            rankings,
            {"tfidf": 1.0, "char_tfidf": 0.02, "rules": 0.5, "bge_m3": 1.0},
            60,
        )
        page_by_number = {int(page["page"]): page for page in row["_pages"]}
        bge_pool = [
            {**item, "text": page_by_number[int(item["page"])].get("text", "")}
            for item in fused[: args.candidate_k]
        ]
        output_rows.append(
            {
                **{key: value for key, value in row.items() if not key.startswith("_")},
                "bge_rankings": {lane: values[: args.candidate_k] for lane, values in rankings.items()},
                "bge_fused": fused[: args.candidate_k],
                "bge_candidate_pool": bge_pool,
            }
        )

    metrics: dict[str, Any] = {
        "config": {
            "bge_model": args.bge_model,
            "reranker_model": args.reranker_model,
            "workers": workers,
            "device": args.device,
            "torch_threads_per_worker": torch_threads,
            "page_char_limit": args.page_char_limit,
            "candidate_k": args.candidate_k,
            "reranker": not args.skip_reranker,
        },
        "baseline_miniLM_existing": metric_summary(output_rows, "fused"),
        "bge_fused": metric_summary(output_rows, "bge_fused"),
        "bge_candidate_recall_at10": metric_summary(output_rows, "bge_fused")["hit_at_10"],
        "rows": len(output_rows),
        "bge_page_vectors_cached": cached_pages,
    }

    if not args.skip_reranker:
        rerank_inputs = [
            {
                "sample_id": row["sample_id"],
                "query_text": row.get("_query_text", row.get("query_text", "")),
                "candidates": row.get("bge_candidate_pool", []),
            }
            for row in output_rows
            if row.get("bge_candidate_pool")
        ]
        # Keep reranking independent of API calls and use multiple processes.
        rerank_shards = split_shards(rerank_inputs, workers, weight_key=None)
        rerank_payloads = [
            {
                "model_name": args.reranker_model,
                "batch_size": args.reranker_batch_size,
                "max_length": args.reranker_max_length,
            "torch_threads": torch_threads,
                "device": args.device,
                "rows": shard,
            }
            for shard in rerank_shards
        ]
        print(f"BGE reranking: {len(rerank_inputs)} queries x {args.candidate_k} candidates, {len(rerank_payloads)} workers")
        reranked_by_id: dict[str, list[dict[str, Any]]] = {}
        with ProcessPoolExecutor(max_workers=len(rerank_payloads)) as pool:
            futures = [pool.submit(_rerank_shard, payload) for payload in rerank_payloads]
            for future in as_completed(futures):
                for item in future.result():
                    reranked_by_id[str(item["sample_id"])] = item["ranking"]
        for row in output_rows:
            ranking = reranked_by_id.get(str(row.get("sample_id")))
            if ranking is not None:
                row["bge_reranked"] = ranking
        metrics["bge_reranked"] = metric_summary(output_rows, "bge_reranked")

    # Remove temporary full page text from persisted output while retaining the
    # ranking fields and enough metadata for reproducibility.
    for row in output_rows:
        for item in row.get("bge_candidate_pool", []):
            item.pop("text", None)
    write_jsonl(args.output, output_rows)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # On Windows, ProcessPoolExecutor uses spawn.  Keep imports and worker
    # functions module-level so every child can import this file cleanly.
    main()
