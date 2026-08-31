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
import unicodedata
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


def _first_json_value(row: dict[str, Any], *names: str) -> str:
    """Return the first non-empty value among schema aliases."""
    for name in names:
        value = row.get(name)
        if value is not None and compact(value):
            return str(value)
    return ""


def normalize_json_truth_row(row: dict[str, Any], language: str = "") -> dict[str, str]:
    """Normalize the participant JSON schemas to the pipeline CSV schema.

    The six language files use slightly different field names and, in a few
    cases, contain both original and normalized spellings.  Keeping this
    conversion here lets us audit the downloaded Test Set directly instead of
    relying on a separately generated CSV.
    """
    lang = _first_json_value(row, "lang", "language") or language
    metric_code = _first_json_value(row, "metric_code", "code", "Code")
    if not metric_code:
        # Korean rows use the SASB metric description as ``sid``; Chinese
        # rows use a numeric sid and do not expose a metric code in the JSON.
        sid = _first_json_value(row, "sid", "SID")
        if re.search(r"[A-Za-z]", sid):
            metric_code = sid
    return {
        "lang": lang.lower(),
        "cid": _first_json_value(row, "cid", "CID"),
        "sid": _first_json_value(row, "sid", "SID"),
        "topic": _first_json_value(row, "topic", "Topic"),
        "metric_description": _first_json_value(
            row, "metric_description", "metric", "Metric"
        ),
        "metric_code": metric_code,
        "page": _first_json_value(
            row,
            "page",
            "Page",
            "pdf_page",
            "PDF_Page",
            "file_page_number",
            "document_page_number",
        ),
        "file_stem": _first_json_value(row, "file_stem", "file stem"),
        "label": _first_json_value(row, "label", "Label"),
        "answer_value": _first_json_value(
            row,
            "answer_value",
            "value",
            "Value",
            "confirmed value",
            "Confirmed value",
            "confirmed_value",
        ),
        "answer_unit": _first_json_value(
            row,
            "answer_unit",
            "unit",
            "Unit",
            "confirmed unit",
            "Confirmed Unit",
            "confirmed_unit",
        ),
        "complete": _first_json_value(row, "complete", "Complete"),
        "sasb_category": _first_json_value(
            row, "sasb_category", "SASB Category", "category", "Category"
        ),
        "sasb_unit_of_measure": _first_json_value(
            row,
            "sasb_unit_of_measure",
            "sasb unit of measurement",
            "SASB Unit of Measurement",
        ),
        "sasb_key_terms": _first_json_value(row, "sasb_key_terms", "SASB Key Terms"),
        "sasb_what_counts": _first_json_value(
            row, "sasb_what_counts", "SASB What Counts"
        ),
    }


def read_json_truth(path: Path, language: str = "") -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    inferred_language = language or path.stem.removesuffix("_test").lower()
    return [normalize_json_truth_row(row, inferred_language) for row in payload]


def read_truth(path: Path) -> list[dict[str, str]]:
    """Read CSV truth or a directory/file of language-specific Test JSONs."""
    if path.is_dir():
        files = sorted(path.glob("*_test.json"))
        if not files:
            raise FileNotFoundError(f"No *_test.json files found under {path}")
        rows: list[dict[str, str]] = []
        for file in files:
            rows.extend(read_json_truth(file))
        return rows
    if path.suffix.lower() == ".json":
        return read_json_truth(path)
    return read_csv(path)


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


def query_key(row: dict[str, str], include_answer_variant: bool = True) -> tuple[str, ...]:
    # This reconstructs a report-level retrieval query from the page-level data.
    # It is an analysis key, not a claim about the unreleased official Task 1 key.
    key = (
        compact(row.get("lang", "")),
        compact(row.get("cid", "")),
        source_pdf_stem(row.get("file_stem", "")),
        compact(row.get("topic", "")),
        compact(row.get("metric_description", "")),
        compact(row.get("metric_code", "")),
    )
    if include_answer_variant:
        return key + (compact(row.get("answer_value", "")), compact(row.get("answer_unit", "")))
    return key + ("", "")


def build_queries(
    rows: list[dict[str, str]], pdf_root: Path, include_answer_variant: bool = True
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[query_key(row, include_answer_variant)].append(row)

    queries: list[dict[str, Any]] = []
    for key, members in groups.items():
        first = members[0]
        lang, cid, report_stem, topic, metric, metric_code, value, unit = key
        pdf_path = find_pdf(pdf_root, lang, report_stem)
        labels = sorted({compact(row.get("label", "")).lower() for row in members})
        answer_values = sorted({compact(row.get("answer_value", "")) for row in members if compact(row.get("answer_value", ""))})
        answer_units = sorted({compact(row.get("answer_unit", "")) for row in members if compact(row.get("answer_unit", ""))})
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
                "expected_value": " | ".join(answer_values) if answer_values else value,
                "expected_unit": " | ".join(answer_units) if answer_units else unit,
                "answer_variants": [{"value": v, "unit": u} for v, u in sorted({(compact(row.get("answer_value", "")), compact(row.get("answer_unit", ""))) for row in members})],
                "sasb_category": compact(first.get("sasb_category", "")),
                "sasb_unit_of_measure": compact(first.get("sasb_unit_of_measure", "")),
                "sasb_key_terms": compact(first.get("sasb_key_terms", "")),
                "sasb_what_counts": compact(first.get("sasb_what_counts", "")),
                "query_text": query_text,
                "gold_pages": gold_pages,
                "gold_labels": labels,
                # A query normally contains both relevant and irrelevant
                # pages. Keep this as a mixed-page diagnostic, not an
                # annotation-conflict claim.
                "mixed_page_labels": len(labels) > 1,
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
    truth = read_truth(args.truth)
    include_answer_variant = args.query_group == "answer_variant"
    queries = build_queries(truth, args.pdf_root, include_answer_variant)
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

    page_label_groups: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in truth:
        page_label_groups[query_key(row, include_answer_variant) + (compact(row.get("page", "")),)].add(
            compact(row.get("label", "")).lower()
        )
    audit = {
        "source_truth": str(args.truth.resolve()),
        "query_group": args.query_group,
        "queries": len(queries),
        "reports": len(reports),
        "pages": len(page_rows),
        "missing_pdfs": sum(not row["pdf_found"] for row in queries),
        "empty_gold_queries": sum(not row["gold_pages"] for row in queries),
        "mixed_page_label_queries": sum(row["mixed_page_labels"] for row in queries),
        "same_page_label_conflict_groups": sum(
            len(labels) > 1 for labels in page_label_groups.values()
        ),
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


def build_candidate_pool(
    rankings: dict[str, list[dict[str, Any]]],
    fused: list[dict[str, Any]],
    per_lane_k: int,
) -> list[dict[str, Any]]:
    """Union the strongest pages from each lane for downstream reranking."""
    if per_lane_k <= 0:
        return []
    fused_by_page = {int(item["page"]): item for item in fused}
    lane_ranks: dict[int, dict[str, int]] = defaultdict(dict)
    for lane, rows in rankings.items():
        for item in rows[:per_lane_k]:
            lane_ranks[int(item["page"])][lane] = int(item["rank"])
    # Interleave lanes so a downstream budget of (for example) 30 images sees
    # lexical, character, rule, and dense candidates instead of only the first
    # lane's consensus pages. The pool itself can be larger than that budget.
    ordered: list[int] = []
    seen: set[int] = set()
    lane_items = list(rankings.items())
    for rank_index in range(per_lane_k):
        for _, rows in lane_items:
            if rank_index >= len(rows):
                continue
            page = int(rows[rank_index]["page"])
            if page not in seen:
                ordered.append(page)
                seen.add(page)
    return [
        {
            "page": page,
            "score": round(float(fused_by_page.get(page, {}).get("score", 0.0)), 10),
            "rank": rank,
            "lane_ranks": lane_ranks[page],
            "in_fused": page in fused_by_page,
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


def load_dense_model(model_name: str) -> Any | None:
    if not model_name or model_name.lower() == "none":
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers for dense retrieval") from exc
    return SentenceTransformer(model_name)


def weighted_query_text(query: dict[str, Any]) -> str:
    """Prioritize metric identity over long explanatory metadata."""
    weighted_fields = (
        ("metric_code", 4),
        ("metric_description", 3),
        ("topic", 2),
        ("sasb_key_terms", 2),
        ("sasb_unit_of_measure", 1),
        ("sasb_category", 1),
    )
    parts: list[str] = []
    for field, weight in weighted_fields:
        value = compact(query.get(field, ""))
        if value:
            parts.extend([value] * weight)
    return " ".join(parts) or compact(query.get("query_text", ""))


def build_lexical_index(
    texts: list[str], max_features: int, char_max_features: int
) -> dict[str, Any]:
    """Fit reusable word and character indexes once per report."""
    # Keep image-only or broken-text reports in the index. The sentinel gets
    # no overlap with normal queries, while avoiding sklearn's empty-vocabulary
    # exception and preserving those pages for later OCR/VLM handling.
    safe_texts = [text if text.strip() else "__empty_page__" for text in texts]
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, max_features=max_features
    )
    word_matrix = word_vectorizer.fit_transform(safe_texts)
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        sublinear_tf=True,
        max_features=char_max_features,
    )
    char_matrix = char_vectorizer.fit_transform(safe_texts)
    return {
        "word_vectorizer": word_vectorizer,
        "word_matrix": word_matrix,
        "char_vectorizer": char_vectorizer,
        "char_matrix": char_matrix,
    }


def command_retrieve(args: argparse.Namespace) -> None:
    queries = read_jsonl(args.queries)
    all_pages = read_jsonl(args.pages)
    pages_by_report: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for page in all_pages:
        pages_by_report[(page["lang"], page["report_stem"])].append(page)

    outputs: list[dict[str, Any]] = []
    lexical_cache: dict[tuple[str, str], dict[str, Any]] = {}
    dense_model = load_dense_model(args.dense_model)
    dense_current_key: tuple[str, str] | None = None
    dense_current_embeddings: np.ndarray | None = None
    dense_cache_dir = args.dense_cache_dir
    for position, query in enumerate(queries, 1):
        report_key = (query["lang"], query["report_stem"])
        pages = sorted(pages_by_report.get(report_key, []), key=lambda row: row["page"])
        if not pages:
            outputs.append({**query, "error": "no_indexed_pages", "rankings": {}, "fused": []})
            continue
        texts = [page["text"][: args.page_char_limit] for page in pages]
        if report_key not in lexical_cache:
            lexical_cache[report_key] = build_lexical_index(
                texts, args.max_features, args.char_max_features
            )
        index = lexical_cache[report_key]
        query_text = (
            query.get("query_text", "")
            if args.query_text_mode == "raw"
            else weighted_query_text(query)
        )
        word_query = index["word_vectorizer"].transform([query_text])
        tfidf = cosine_similarity(word_query, index["word_matrix"]).ravel()
        char_query = index["char_vectorizer"].transform(
            [unicodedata.normalize("NFKC", query_text).lower()]
        )
        char_tfidf = cosine_similarity(char_query, index["char_matrix"]).ravel()
        rule = np.asarray([unit_rule_score(query, text) for text in texts], dtype=float)
        rankings = {"tfidf": rankings_from_scores(tfidf, pages)}
        rankings["char_tfidf"] = rankings_from_scores(char_tfidf, pages)
        rule_ranking = rankings_from_scores(rule, pages, positive_only=True)
        if rule_ranking:
            rankings["rules"] = rule_ranking
        dense = None
        if dense_model is not None:
            if dense_current_key != report_key:
                dense_current_key = report_key
                dense_current_embeddings = None
                dense_texts = [
                    unicodedata.normalize("NFKC", text)[: args.dense_page_char_limit]
                    for text in texts
                ]
                pdf_hash = str(pages[0].get("pdf_sha256", ""))[:12]
                model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.dense_model)
                cache_name = re.sub(
                    r"[^A-Za-z0-9_.-]+",
                    "_",
                    f"{model_slug}__{query['lang']}__{query['report_stem']}__{pdf_hash}",
                )
                cache_file = dense_cache_dir / f"{cache_name}__{args.dense_page_char_limit}.npy"
                if cache_file.exists():
                    loaded = np.load(cache_file)
                    dense_current_embeddings = loaded if len(loaded) == len(texts) else None
                if dense_current_embeddings is None:
                    dense_current_embeddings = np.asarray(
                        dense_model.encode(
                            dense_texts,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            batch_size=args.dense_batch_size,
                        )
                    )
                    dense_cache_dir.mkdir(parents=True, exist_ok=True)
                    np.save(cache_file, dense_current_embeddings)
            query_embedding = np.asarray(
                dense_model.encode(
                    [query_text],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=1,
                )[0]
            )
            assert dense_current_embeddings is not None
            dense = dense_current_embeddings @ query_embedding
        if dense is not None:
            rankings["dense"] = rankings_from_scores(dense, pages)
        weights = {
            "tfidf": args.tfidf_weight,
            "char_tfidf": args.char_weight,
            "rules": args.rule_weight,
            "dense": args.dense_weight,
        }
        fused = reciprocal_rank_fusion(rankings, weights, args.rrf_k)
        candidate_pool = build_candidate_pool(rankings, fused, args.candidate_pool_k)
        outputs.append(
            {
                **query,
                "retrieval_config": {
                    "dense_model": args.dense_model,
                    "rrf_k": args.rrf_k,
                    "weights": weights,
                    "page_char_limit": args.page_char_limit,
                    "char_max_features": args.char_max_features,
                    "query_text_mode": args.query_text_mode,
                },
                "rankings": {lane: rows[: args.keep_rankings] for lane, rows in rankings.items()},
                "fused": fused[: args.top_k],
                "candidate_pool": candidate_pool,
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
    api_timeout: float,
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
    client = OpenAI(timeout=api_timeout)
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
    if args.languages:
        allowed = {language.lower() for language in args.languages}
        rows = [row for row in rows if str(row.get("lang", "")).lower() in allowed]
    if args.max_per_language:
        kept: list[dict[str, Any]] = []
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            language = str(row.get("lang", "")).lower()
            if counts[language] >= args.max_per_language:
                continue
            kept.append(row)
            counts[language] += 1
        rows = kept
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
            candidates = list(row.get(args.ranking_field, []))[: args.candidates]
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
                        args.api_timeout,
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
    prepare.add_argument(
        "--truth",
        type=Path,
        required=True,
        help="CSV truth, one Test JSON, or a directory containing *_test.json files.",
    )
    prepare.add_argument("--pdf-root", type=Path, required=True)
    prepare.add_argument("--queries-out", type=Path, required=True)
    prepare.add_argument("--pages-out", type=Path, required=True)
    prepare.add_argument("--audit-out", type=Path, required=True)
    prepare.add_argument("--max-queries", type=int, default=0)
    prepare.add_argument(
        "--query-group",
        choices=["report_metric", "answer_variant"],
        default="report_metric",
        help="Group by report/metric (recommended) or preserve answer-value variants.",
    )
    prepare.set_defaults(func=command_prepare)

    retrieve = sub.add_parser("retrieve", help="Run local retrieval and RRF")
    retrieve.add_argument("--queries", type=Path, required=True)
    retrieve.add_argument("--pages", type=Path, required=True)
    retrieve.add_argument("--output", type=Path, required=True)
    retrieve.add_argument("--dense-model", default="none")
    retrieve.add_argument("--dense-batch-size", type=int, default=32)
    retrieve.add_argument("--dense-page-char-limit", type=int, default=1500)
    retrieve.add_argument("--dense-cache-dir", type=Path, default=Path("cache/task1-dense"))
    retrieve.add_argument("--tfidf-weight", type=float, default=1.0)
    retrieve.add_argument("--char-weight", type=float, default=0.02)
    retrieve.add_argument("--dense-weight", type=float, default=1.0)
    retrieve.add_argument("--rule-weight", type=float, default=0.5)
    retrieve.add_argument("--rrf-k", type=int, default=60)
    retrieve.add_argument("--top-k", type=int, default=10)
    retrieve.add_argument(
        "--candidate-pool-k",
        type=int,
        default=0,
        help="Union this many pages per lane for optional VLM reranking (0 disables).",
    )
    retrieve.add_argument("--keep-rankings", type=int, default=20)
    retrieve.add_argument("--page-char-limit", type=int, default=6000)
    retrieve.add_argument("--max-features", type=int, default=50000)
    retrieve.add_argument("--char-max-features", type=int, default=100000)
    retrieve.add_argument(
        "--query-text-mode",
        choices=["raw", "weighted_fields"],
        default="raw",
        help="Use the original query string (recommended) or weighted fields.",
    )
    retrieve.set_defaults(func=command_retrieve)

    evaluate = sub.add_parser("evaluate", help="Evaluate a ranking field")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument(
        "--ranking-field", default="fused", choices=["fused", "candidate_pool", "vlm_ranked"]
    )
    evaluate.add_argument("--metrics-out", type=Path, required=True)
    evaluate.add_argument("--per-query-out", type=Path, required=True)
    evaluate.set_defaults(func=command_evaluate)

    export = sub.add_parser("export-task2", help="Export Top-K pages for the existing Task 2 verifier")
    export.add_argument("--predictions", type=Path, required=True)
    export.add_argument(
        "--ranking-field", default="fused", choices=["fused", "candidate_pool", "vlm_ranked"]
    )
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
    rerank.add_argument(
        "--api-timeout",
        type=float,
        default=90.0,
        help="Per-request timeout in seconds; timed-out rows remain retryable.",
    )
    rerank.add_argument(
        "--languages",
        nargs="+",
        default=[],
        help="Optional language allow-list for pilot runs (e.g. english thai).",
    )
    rerank.add_argument(
        "--max-per-language",
        type=int,
        default=0,
        help="Optional cap per language after filtering; 0 means no cap.",
    )
    rerank.add_argument(
        "--ranking-field", choices=["fused", "candidate_pool"], default="fused"
    )
    rerank.add_argument("--execute-api", action="store_true")
    rerank.set_defaults(func=command_rerank)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
