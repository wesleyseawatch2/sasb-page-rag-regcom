from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import fitz
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


POSITIVE_LABELS = {"yes", "yes but not complete"}


def compact(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def source_pdf_stem(file_stem: str) -> str:
    match = re.match(r"(.+)_([0-9]+)$", compact(file_stem))
    return match.group(1) if match else compact(file_stem)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stable_id(*parts: object) -> str:
    text = "\x1f".join(compact(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_pdf(pdf_root: Path, lang: str, stem: str) -> Path | None:
    direct = pdf_root / lang / f"{stem}.pdf"
    if direct.exists():
        return direct
    matches = list(pdf_root.glob(f"*/{stem}.pdf"))
    return matches[0] if len(matches) == 1 else None


def query_key(row: dict[str, str]) -> tuple[str, ...]:
    # This reconstructs a report-level retrieval query from the page-level data.
    # It is an analysis key, not a claim about the unreleased official Task 1 key.
    return (
        compact(row.get("lang", "")),
        compact(row.get("cid", "")),
        source_pdf_stem(row.get("file_stem", "")),
        compact(row.get("topic", "")),
        compact(row.get("metric_description", "")),
        compact(row.get("metric_code", "")),
        compact(row.get("answer_value", "")),
        compact(row.get("answer_unit", "")),
    )


def build_queries(rows: list[dict[str, str]], pdf_root: Path) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[query_key(row)].append(row)

    queries: list[dict[str, Any]] = []
    for key, members in groups.items():
        first = members[0]
        lang, cid, report_stem, topic, metric, metric_code, value, unit = key
        pdf_path = find_pdf(pdf_root, lang, report_stem)
        labels = sorted({compact(row.get("label", "")).lower() for row in members})
        gold_pages = sorted(
            {
                int(float(row["page"]))
                for row in members
                if compact(row.get("label", "")).lower() in POSITIVE_LABELS
                and compact(row.get("page", ""))
            }
        )
        query_text = " ".join(
            compact(first.get(field, ""))
            for field in (
                "topic",
                "metric_description",
                "metric_code",
                "sasb_category",
                "sasb_unit_of_measure",
                "sasb_key_terms",
                "sasb_what_counts",
            )
        )
        queries.append(
            {
                "sample_id": stable_id(*key),
                "lang": lang,
                "cid": cid,
                "report_stem": report_stem,
                "topic": topic,
                "metric_description": metric,
                "metric_code": metric_code,
                "expected_value": value,
                "expected_unit": unit,
                "sasb_category": compact(first.get("sasb_category", "")),
                "sasb_unit_of_measure": compact(first.get("sasb_unit_of_measure", "")),
                "sasb_key_terms": compact(first.get("sasb_key_terms", "")),
                "sasb_what_counts": compact(first.get("sasb_what_counts", "")),
                "query_text": query_text,
                "gold_pages": gold_pages,
                "gold_labels": labels,
                "gold_conflict": len(labels) > 1,
                "pdf_path": str(pdf_path.resolve()) if pdf_path else "",
                "pdf_found": bool(pdf_path),
            }
        )
    return sorted(queries, key=lambda row: (row["lang"], row["report_stem"], row["sample_id"]))


def page_quality(text: str) -> list[str]:
    flags: list[str] = []
    if not text.strip():
        flags.append("empty_text")
    elif len(text) < 100:
        flags.append("short_text")
    if len(re.findall(r"\d", text)) >= 20:
        flags.append("number_dense")
    if text.count("|") >= 8 or text.count("\t") >= 8:
        flags.append("table_like")
    return flags


def index_pdf(pdf_path: Path, lang: str, report_stem: str) -> list[dict[str, Any]]:
    pdf_hash = sha256_file(pdf_path)
    pages: list[dict[str, Any]] = []
    with fitz.open(str(pdf_path)) as doc:
        for index, page in enumerate(doc):
            text = page.get_text("text").strip()
            pages.append(
                {
                    "report_stem": report_stem,
                    "lang": lang,
                    "pdf_path": str(pdf_path.resolve()),
                    "pdf_sha256": pdf_hash,
                    "pdf_page_index": index,
                    "page": index + 1,
                    "text": text,
                    "char_count": len(text),
                    "numeric_token_count": len(re.findall(r"\b\d[\d.,%]*\b", text)),
                    "quality_flags": page_quality(text),
                }
            )
    return pages


def command_prepare(args: argparse.Namespace) -> None:
    truth = read_csv(args.truth)
    queries = build_queries(truth, args.pdf_root)
    if args.max_queries:
        queries = queries[: args.max_queries]
    write_jsonl(args.queries_out, queries)

    reports: dict[tuple[str, str], Path] = {}
    for query in queries:
        if query["pdf_found"]:
            reports[(query["lang"], query["report_stem"])] = Path(query["pdf_path"])
    page_rows: list[dict[str, Any]] = []
    for index, ((lang, stem), path) in enumerate(sorted(reports.items()), 1):
        print(f"[index {index}/{len(reports)}] {lang}/{stem}")
        page_rows.extend(index_pdf(path, lang, stem))
    write_jsonl(args.pages_out, page_rows)

    audit = {
        "source_truth": str(args.truth.resolve()),
        "queries": len(queries),
        "reports": len(reports),
        "pages": len(page_rows),
        "missing_pdfs": sum(not row["pdf_found"] for row in queries),
        "empty_gold_queries": sum(not row["gold_pages"] for row in queries),
        "conflicting_gold_queries": sum(row["gold_conflict"] for row in queries),
        "note": "Queries are reconstructed from Subtask 2 truth and are not asserted to be the official Task 1 release.",
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


def unit_rule_score(query: dict[str, Any], text: str) -> float:
    haystack = text.lower()
    terms = re.split(r"[|;,/]", f"{query.get('metric_code', '')}|{query.get('expected_unit', '')}|{query.get('sasb_unit_of_measure', '')}")
    terms = [compact(term).lower() for term in terms if len(compact(term)) >= 2 and compact(term).lower() != "n/a"]
    if not terms:
        return 0.0
    return sum(term in haystack for term in terms) / len(terms)


def rankings_from_scores(
    scores: np.ndarray, pages: list[dict[str, Any]], positive_only: bool = False
) -> list[dict[str, Any]]:
    order = [int(i) for i in np.argsort(-scores) if not positive_only or scores[int(i)] > 0]
    return [
        {"page": int(pages[i]["page"]), "score": round(float(scores[i]), 8), "rank": rank}
        for rank, i in enumerate(order, 1)
    ]


def reciprocal_rank_fusion(
    rankings: dict[str, list[dict[str, Any]]], weights: dict[str, float], k: int
) -> list[dict[str, Any]]:
    fused: dict[int, float] = defaultdict(float)
    lane_ranks: dict[int, dict[str, int]] = defaultdict(dict)
    for lane, rows in rankings.items():
        weight = weights.get(lane, 1.0)
        for row in rows:
            page = int(row["page"])
            rank = int(row["rank"])
            fused[page] += weight / (k + rank)
            lane_ranks[page][lane] = rank
    ordered = sorted(fused, key=lambda page: (-fused[page], page))
    return [
        {
            "page": page,
            "score": round(fused[page], 10),
            "rank": rank,
            "lane_ranks": lane_ranks[page],
        }
        for rank, page in enumerate(ordered, 1)
    ]


def dense_scores(query_text: str, texts: list[str], model_name: str) -> np.ndarray | None:
    if not model_name or model_name.lower() == "none":
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers for dense retrieval") from exc
    model = SentenceTransformer(model_name)
    embeddings = model.encode([query_text, *texts], normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings[1:]) @ np.asarray(embeddings[0])


def command_retrieve(args: argparse.Namespace) -> None:
    queries = read_jsonl(args.queries)
    all_pages = read_jsonl(args.pages)
    pages_by_report: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in all_pages:
        pages_by_report[page["report_stem"]].append(page)

    outputs: list[dict[str, Any]] = []
    for position, query in enumerate(queries, 1):
        pages = sorted(pages_by_report.get(query["report_stem"], []), key=lambda row: row["page"])
        if not pages:
            outputs.append({**query, "error": "no_indexed_pages", "rankings": {}, "fused": []})
            continue
        texts = [page["text"][: args.page_char_limit] for page in pages]
        query_text = query["query_text"]
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, max_features=args.max_features
        )
        matrix = vectorizer.fit_transform([query_text, *texts])
        tfidf = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        rule = np.asarray([unit_rule_score(query, text) for text in texts], dtype=float)
        rankings = {"tfidf": rankings_from_scores(tfidf, pages)}
        rule_ranking = rankings_from_scores(rule, pages, positive_only=True)
        if rule_ranking:
            rankings["rules"] = rule_ranking
        dense = dense_scores(query_text, texts, args.dense_model)
        if dense is not None:
            rankings["dense"] = rankings_from_scores(dense, pages)
        weights = {"tfidf": args.tfidf_weight, "rules": args.rule_weight, "dense": args.dense_weight}
        fused = reciprocal_rank_fusion(rankings, weights, args.rrf_k)
        outputs.append(
            {
                **query,
                "retrieval_config": {
                    "dense_model": args.dense_model,
                    "rrf_k": args.rrf_k,
                    "weights": weights,
                    "page_char_limit": args.page_char_limit,
                },
                "rankings": {lane: rows[: args.keep_rankings] for lane, rows in rankings.items()},
                "fused": fused[: args.top_k],
            }
        )
        if position % 25 == 0 or position == len(queries):
            print(f"retrieved {position}/{len(queries)}")
    write_jsonl(args.output, outputs)


def hit_at(predicted: list[int], gold: set[int], k: int) -> float:
    return float(bool(set(predicted[:k]) & gold))


def reciprocal_rank(predicted: list[int], gold: set[int]) -> float:
    for rank, page in enumerate(predicted, 1):
        if page in gold:
            return 1.0 / rank
    return 0.0


def command_evaluate(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.predictions)
    positive = [row for row in rows if row.get("gold_pages") and not row.get("error")]
    per_query: list[dict[str, Any]] = []
    for row in positive:
        gold = {int(page) for page in row["gold_pages"]}
        source = row.get(args.ranking_field, [])
        predicted = [int(item["page"]) for item in source]
        exact = hit_at(predicted, gold, 1)
        near = float(bool(predicted and any(abs(predicted[0] - page) <= 1 for page in gold)))
        per_query.append(
            {
                "sample_id": row["sample_id"],
                "lang": row["lang"],
                "report_stem": row["report_stem"],
                "gold_pages": "|".join(map(str, sorted(gold))),
                "pred_top1": predicted[0] if predicted else "",
                "hit_at_1": exact,
                "hit_at_5": hit_at(predicted, gold, 5),
                "hit_at_10": hit_at(predicted, gold, 10),
                "near_at_1": near,
                "reciprocal_rank": reciprocal_rank(predicted, gold),
            }
        )
    metrics = {
        "ranking_field": args.ranking_field,
        "evaluated_non_empty_gold": len(per_query),
        "hit_at_1": statistics.fmean(row["hit_at_1"] for row in per_query) if per_query else 0.0,
        "hit_at_5": statistics.fmean(row["hit_at_5"] for row in per_query) if per_query else 0.0,
        "hit_at_10": statistics.fmean(row["hit_at_10"] for row in per_query) if per_query else 0.0,
        "near_at_1": statistics.fmean(row["near_at_1"] for row in per_query) if per_query else 0.0,
        "mrr": statistics.fmean(row["reciprocal_rank"] for row in per_query) if per_query else 0.0,
        "excluded_empty_gold": sum(not row.get("gold_pages") for row in rows),
        "errors": sum(bool(row.get("error")) for row in rows),
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_csv(args.per_query_out, per_query)
    print(json.dumps(metrics, indent=2))


def command_export_task2(args: argparse.Namespace) -> None:
    """Export retrieved pages in the existing Subtask 2 answer-sheet schema."""
    rows = read_jsonl(args.predictions)
    exported: list[dict[str, Any]] = []
    for row in rows:
        ranking = row.get(args.ranking_field, [])
        for evidence_rank, candidate in enumerate(ranking[: args.evidence_pages], 1):
            page = int(candidate["page"])
            exported.append(
                {
                    "sample_id": row["sample_id"],
                    "evidence_rank": evidence_rank,
                    "retrieval_score": candidate.get("score", candidate.get("relevance", "")),
                    "lang": row["lang"],
                    "cid": row["cid"],
                    "sid": "",
                    "topic": row["topic"],
                    "metric_description": row["metric_description"],
                    "metric_code": row["metric_code"],
                    "page": page,
                    "file_stem": f"{row['report_stem']}_{page:03d}",
                    "sasb_category": row["sasb_category"],
                    "sasb_unit_of_measure": row["sasb_unit_of_measure"],
                    "sasb_key_terms": row["sasb_key_terms"],
                    "sasb_what_counts": row["sasb_what_counts"],
                    "pred_label": "",
                    "category_match": "",
                    "unit_match": "",
                }
            )
    write_csv(args.output, exported)
    print(f"exported {len(exported)} page-metric rows to {args.output}")


def render_page_data_url(pdf_path: Path, page_number: int, dpi: int) -> str:
    with fitz.open(str(pdf_path)) as doc:
        page = doc[page_number - 1]
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        data = pixmap.tobytes("jpeg", jpg_quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def parse_jsonish(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def rerank_prompt(query: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    candidate_lines = "\n".join(
        f"- {key}: PDF page index {item['page']}; fused rank {item['rank']}; score {item['score']}"
        for key, item in ((f"C{index:02d}", item) for index, item in enumerate(candidates, 1))
    )
    return f"""You rerank candidate ESG report pages for one SASB metric.
Judge only the supplied candidate page images. Do not infer evidence from the
metric description alone. An index page that merely names a metric is not a
direct disclosure. Return JSON only.

Metric code: {query.get('metric_code', '')}
Topic: {query.get('topic', '')}
Metric: {query.get('metric_description', '')}
Expected category: {query.get('sasb_category', '')}
Expected unit: {query.get('sasb_unit_of_measure', '')}
What counts: {query.get('sasb_what_counts', '')}

Candidates (use these opaque candidate keys, not printed page numbers), in the
same order as the following images:
{candidate_lines}

Required JSON shape:
{{
  "ranked_pages": [
    {{"candidate_key": "C01", "relevance": 0.0, "evidence_type": "direct|partial|index_reference|topical_only|unrelated"}}
  ],
  "no_relevant_page": false
}}
Return exactly one compact JSON object. Include all candidates, in ranked order.
Return candidate_key values exactly as listed; never return a printed page number.
Do not include a reason or any other fields.
"""


def call_openai_vlm(
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    model: str,
    dpi: int,
    image_detail: str,
    max_output_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the openai package to execute VLM reranking") from exc
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    content: list[dict[str, Any]] = [{"type": "input_text", "text": rerank_prompt(query, candidates)}]
    for index, candidate in enumerate(candidates, 1):
        # Bind the opaque PDF index to the image immediately before it. This
        # prevents the model from returning a printed-page number read from
        # the document instead of the candidate identifier used for scoring.
        content.append(
            {
                "type": "input_text",
                "text": f"Candidate key: C{index:02d}. PDF page index: {int(candidate['page'])}. The next image is this candidate.",
            }
        )
        content.append(
            {
                "type": "input_image",
                "image_url": render_page_data_url(Path(query["pdf_path"]), int(candidate["page"]), dpi),
                "detail": image_detail,
            }
        )
    client = OpenAI()
    started = time.perf_counter()
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        max_output_tokens=max_output_tokens,
        store=False,
    )
    elapsed = time.perf_counter() - started
    parsed = parse_jsonish(response.output_text)
    usage = getattr(response, "usage", None)
    metadata = {
        "response_id": response.id,
        "model": response.model,
        "latency_seconds": round(elapsed, 3),
        "usage": usage.model_dump() if usage and hasattr(usage, "model_dump") else {},
    }
    return parsed, metadata


def normalize_vlm_ranking(
    ranked: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep only candidate IDs and append omitted candidates in fused order."""
    candidate_pages = [int(item["page"]) for item in candidates]
    key_to_page = {f"C{index:02d}": page for index, page in enumerate(candidate_pages, 1)}
    allowed = set(candidate_pages)
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in ranked:
        key = str(item.get("candidate_key", "")).upper()
        if key in key_to_page:
            page = key_to_page[key]
        else:
            try:
                page = int(item["page"])
            except (KeyError, TypeError, ValueError):
                continue
        if page not in allowed or page in seen:
            continue
        seen.add(page)
        try:
            relevance = float(item.get("relevance", 0.0))
        except (TypeError, ValueError):
            relevance = 0.0
        normalized.append(
            {
                "page": page,
                "relevance": relevance,
                "evidence_type": str(item.get("evidence_type", "unrated")),
            }
        )
    for page in candidate_pages:
        if page not in seen:
            normalized.append({"page": page, "relevance": -1.0, "evidence_type": "unrated"})
    return normalized


def command_rerank(args: argparse.Namespace) -> None:
    load_dotenv(args.env_file)
    rows = read_jsonl(args.retrieval)
    completed: dict[str, dict[str, Any]] = {}
    if args.cache.exists():
        completed = {row["sample_id"]: row for row in read_jsonl(args.cache) if row.get("sample_id")}
    output_rows: list[dict[str, Any]] = []
    cache_handle = None
    if args.execute_api:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        cache_handle = args.cache.open("a", encoding="utf-8")
    try:
        for position, row in enumerate(rows, 1):
            candidates = list(row.get("fused", []))[: args.candidates]
            if row["sample_id"] in completed:
                trace = completed[row["sample_id"]]
            elif not args.execute_api:
                trace = {
                    "sample_id": row["sample_id"],
                    "status": "dry_run",
                    "model": args.model,
                    "candidate_pages": [item["page"] for item in candidates],
                    "prompt": rerank_prompt(row, candidates),
                }
            else:
                try:
                    parsed, metadata = call_openai_vlm(
                        row,
                        candidates,
                        args.model,
                        args.dpi,
                        args.image_detail,
                        args.max_output_tokens,
                    )
                    trace = {
                        "sample_id": row["sample_id"],
                        "status": "complete",
                        "candidate_pages": [item["page"] for item in candidates],
                        "prediction": parsed,
                        **metadata,
                    }
                    assert cache_handle is not None
                    cache_handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    cache_handle.flush()
                except Exception as exc:
                    # Keep a malformed/failed response from aborting a long run.
                    # Do not cache failures: a later run should be able to retry.
                    trace = {
                        "sample_id": row["sample_id"],
                        "status": "error",
                        "model": args.model,
                        "candidate_pages": [item["page"] for item in candidates],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            ranked = normalize_vlm_ranking(
                trace.get("prediction", {}).get("ranked_pages", []), candidates
            )
            output_rows.append({**row, "vlm_trace": trace, "vlm_ranked": ranked})
            if position % 10 == 0 or position == len(rows):
                print(f"reranked {position}/{len(rows)}")
    finally:
        if cache_handle:
            cache_handle.close()
    write_jsonl(args.output, output_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traceable Task 1 retrieval and VLM reranking pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Reconstruct queries and index full PDFs")
    prepare.add_argument("--truth", type=Path, required=True)
    prepare.add_argument("--pdf-root", type=Path, required=True)
    prepare.add_argument("--queries-out", type=Path, required=True)
    prepare.add_argument("--pages-out", type=Path, required=True)
    prepare.add_argument("--audit-out", type=Path, required=True)
    prepare.add_argument("--max-queries", type=int, default=0)
    prepare.set_defaults(func=command_prepare)

    retrieve = sub.add_parser("retrieve", help="Run local retrieval and RRF")
    retrieve.add_argument("--queries", type=Path, required=True)
    retrieve.add_argument("--pages", type=Path, required=True)
    retrieve.add_argument("--output", type=Path, required=True)
    retrieve.add_argument("--dense-model", default="none")
    retrieve.add_argument("--tfidf-weight", type=float, default=1.0)
    retrieve.add_argument("--dense-weight", type=float, default=1.0)
    retrieve.add_argument("--rule-weight", type=float, default=0.5)
    retrieve.add_argument("--rrf-k", type=int, default=60)
    retrieve.add_argument("--top-k", type=int, default=10)
    retrieve.add_argument("--keep-rankings", type=int, default=20)
    retrieve.add_argument("--page-char-limit", type=int, default=6000)
    retrieve.add_argument("--max-features", type=int, default=50000)
    retrieve.set_defaults(func=command_retrieve)

    evaluate = sub.add_parser("evaluate", help="Evaluate a ranking field")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--ranking-field", default="fused", choices=["fused", "vlm_ranked"])
    evaluate.add_argument("--metrics-out", type=Path, required=True)
    evaluate.add_argument("--per-query-out", type=Path, required=True)
    evaluate.set_defaults(func=command_evaluate)

    export = sub.add_parser("export-task2", help="Export Top-K pages for the existing Task 2 verifier")
    export.add_argument("--predictions", type=Path, required=True)
    export.add_argument("--ranking-field", default="fused", choices=["fused", "vlm_ranked"])
    export.add_argument("--evidence-pages", type=int, default=1)
    export.add_argument("--output", type=Path, required=True)
    export.set_defaults(func=command_export_task2)

    rerank = sub.add_parser("rerank", help="Dry-run or execute cached OpenAI VLM reranking")
    rerank.add_argument("--retrieval", type=Path, required=True)
    rerank.add_argument("--output", type=Path, required=True)
    rerank.add_argument("--cache", type=Path, required=True)
    rerank.add_argument("--env-file", type=Path, default=Path(".env"))
    rerank.add_argument("--model", default=os.getenv("OPENAI_VLM_MODEL", "gpt-5.4-nano"))
    rerank.add_argument("--candidates", type=int, default=10)
    rerank.add_argument("--dpi", type=int, default=96)
    rerank.add_argument("--image-detail", choices=["low", "high"], default="low")
    rerank.add_argument("--max-output-tokens", type=int, default=900)
    rerank.add_argument("--execute-api", action="store_true")
    rerank.set_defaults(func=command_rerank)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
