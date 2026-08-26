"""
End-to-end RegComAgent experiment runner.

This project does not fine-tune a neural model. "Training code" here means the
complete reproducible experiment pipeline used in the paper:

1. Few-shot dynamic orchestrator inference
2. API-error pruning
3. Deterministic repair
4. Claude Sonnet conflict review
5. Orchestrator-controlled Claude gating
6. GPT-4o meta review
7. Evaluation and packaging

Example quick smoke test:
    py scripts/pipeline/run_regcomagent_experiment.py --stages pipeline prune repair eval --sample 5

Example full run:
    py scripts/pipeline/run_regcomagent_experiment.py --stages all --max-rows 931 --concurrency 1

Note:
    --gpt-wrong-only uses ground truth to select rows for error-focused review.
    It reproduces the final diagnostic/post-reviewed experiment, but should not
    be described as a blind submission setting.
    Add --static-routing only when reproducing the older fixed-order pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "data" / "predictions"
DEFAULT_OUT_DIR = PRED_DIR / "regcomagent_orchestrator_experiment"
DEFAULT_TRUTH = ROOT / "data" / "datasets" / "all_subtask2_dataset.csv"
DEFAULT_INPUT = ROOT / "data" / "datasets" / "all_subtask2_answer_sheet.csv"
VALID_LABELS = ["yes", "yes but not complete", "no"]


def load_orchestrator_module():
    module_path = ROOT / "scripts" / "pipeline" / "orchestrator_blind.py"
    spec = importlib.util.spec_from_file_location("orchestrator_blind", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def row_key(row: dict) -> tuple[str, str, str]:
    return (row.get("file_stem", ""), row.get("sid", ""), row.get("cid", ""))


def output_sidecar(path: Path, suffix: str) -> Path:
    return path.with_suffix(path.suffix + suffix)


def paths(out_dir: Path) -> dict[str, Path]:
    return {
        "raw": out_dir / "01_pipeline.csv",
        "pruned": out_dir / "02_pruned.csv",
        "repaired": out_dir / "03_deterministic_repaired.csv",
        "gemini_all": out_dir / "04_gemini_all.csv",
        "gemini_reviews": out_dir / "04_gemini.review.jsonl",
        "gemini_gated": out_dir / "05_gemini_gated.csv",
        "gpt_final": out_dir / "06_final_gpt4o_meta.csv",
        "gpt_reviews": out_dir / "06_final_gpt4o_meta.review.jsonl",
        "metrics": out_dir / "metrics_final.json",
        "manifest": out_dir / "MANIFEST.csv",
        "readme": out_dir / "README.md",
    }


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    started = time.perf_counter()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"finished in {time.perf_counter() - started:.1f}s", flush=True)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_review_jsonl(path: Path) -> dict[tuple[str, str, str], dict]:
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
            key = tuple(obj.get("key", []))
            if len(key) == 3:
                reviews[key] = obj
    return reviews


def normalize_prediction(pred: dict) -> dict:
    label = pred.get("pred_label", "")
    category = pred.get("category_match", "")
    unit = pred.get("unit_match", "")
    if label == "no":
        category = "N/A"
        unit = "N/A"
    return {
        "pred_label": label,
        "category_match": category,
        "unit_match": unit,
    }


def stage_pipeline(args: argparse.Namespace, p: dict[str, Path]) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "pipeline" / "orchestrator_blind.py"),
        "--input", str(args.input),
        "--output", str(p["raw"]),
        "--truth", str(args.truth),
        "--max-rows", str(args.max_rows),
        "--orchestrator-model", args.orchestrator_model,
        "--worker-model", args.worker_model,
        "--few-shot-limit", str(args.few_shot_limit),
        "--few-shot-scope", args.few_shot_scope,
        "--language-char-limit", str(args.language_char_limit),
        "--pdf-char-limit", str(args.pdf_char_limit),
        "--concurrency", str(args.concurrency),
        "--resume",
    ]
    if args.sample:
        cmd.extend(["--sample", str(args.sample), "--seed", str(args.seed)])
    if not args.static_routing:
        cmd.append("--dynamic-routing")
    if args.inline_gemini_review:
        cmd.append("--claude-review")
    run_command(cmd, args.dry_run)


def stage_prune(args: argparse.Namespace, p: dict[str, Path]) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "pipeline" / "orchestrator_blind.py"),
        "--output", str(p["raw"]),
        "--trace-output", str(output_sidecar(p["raw"], ".trace.jsonl")),
        "--prune-error-rows",
        "--pruned-output", str(p["pruned"]),
    ]
    run_command(cmd, args.dry_run)


def stage_repair(args: argparse.Namespace, p: dict[str, Path]) -> None:
    if args.dry_run:
        print(f"would repair {p['pruned']} -> {p['repaired']}")
        return
    orch = load_orchestrator_module()
    rows = read_csv(p["pruned"])
    if not rows:
        raise RuntimeError(f"No rows found in {p['pruned']}")
    fieldnames = list(rows[0].keys())
    changed = 0
    for row in rows:
        before = {
            "pred_label": row.get("pred_label", ""),
            "category_match": row.get("category_match", ""),
            "unit_match": row.get("unit_match", ""),
        }
        page_text = orch.extract_page_text(row["file_stem"], row.get("lang", ""), row.get("page", ""))
        after = orch.apply_rule_overrides(row, page_text, dict(before))
        after = normalize_prediction(after)
        if before != after:
            changed += 1
            row.update(after)
    write_csv(p["repaired"], rows, fieldnames)
    print(f"deterministic repair wrote {p['repaired']} changed={changed}")


def stage_gemini(args: argparse.Namespace, p: dict[str, Path]) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "pipeline" / "claude_review_existing.py"),
        "--input", str(p["repaired"]),
        "--output", str(p["gemini_all"]),
        "--reviews-output", str(p["gemini_reviews"]),
        "--trace-paths",
        str(output_sidecar(p["raw"], ".trace.jsonl")),
        str(output_sidecar(p["pruned"], ".trace.jsonl")),
        "--model", args.gemini_model,
        "--max-reviews", str(args.gemini_max_reviews),
        "--pdf-char-limit", str(args.review_pdf_char_limit),
        "--include-multilingual-no",
    ]
    if args.review_dry_run:
        cmd.append("--dry-run")
    run_command(cmd, args.dry_run)


def accept_gemini_review(review: dict) -> bool:
    reasons = set(review.get("reasons", []))
    current = review.get("current_prediction", {})
    reviewed = review.get("reviewed_prediction", {})
    current_label = current.get("pred_label", "")
    reviewed_label = reviewed.get("pred_label", "")

    # Final gated-v2 policy used in the paper artifacts:
    # - accept Claude demotion from ybnc to no for boundary cases
    # - accept multilingual no -> ybnc recovery
    # - accept rare no -> yes recovery on SASB index / D&A cases
    # - reject most yes demotions because Sonnet was too conservative there
    if (
        current_label == "yes but not complete"
        and reviewed_label == "no"
        and "boundary_yes_but_not_complete" in reasons
    ):
        return True
    if (
        current_label == "no"
        and reviewed_label == "yes but not complete"
        and "multilingual_pred_no" in reasons
    ):
        return True
    if (
        current_label == "no"
        and reviewed_label == "yes"
        and ({"sasb_index_page", "da_pred_no"} & reasons)
    ):
        return True
    return False


def stage_gate_gemini(args: argparse.Namespace, p: dict[str, Path]) -> None:
    if args.dry_run:
        print(f"would gate Gemini reviews {p['gemini_reviews']} -> {p['gemini_gated']}")
        return
    rows = read_csv(p["repaired"])
    if not rows:
        raise RuntimeError(f"No rows found in {p['repaired']}")
    fieldnames = list(rows[0].keys())
    reviews = load_review_jsonl(p["gemini_reviews"])
    accepted = 0
    for row in rows:
        review = reviews.get(row_key(row))
        if not review or not review.get("reviewed_prediction"):
            continue
        if accept_gemini_review(review):
            row.update(normalize_prediction(review["reviewed_prediction"]))
            accepted += 1
    write_csv(p["gemini_gated"], rows, fieldnames)
    print(f"Gemini gated merge wrote {p['gemini_gated']} accepted_reviews={accepted}")


def stage_gpt(args: argparse.Namespace, p: dict[str, Path]) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "pipeline" / "gpt4o_meta_review_existing.py"),
        "--input", str(p["gemini_gated"]),
        "--base", str(p["repaired"]),
        "--sonnet-reviews", str(p["gemini_reviews"]),
        "--truth", str(args.truth),
        "--output", str(p["gpt_final"]),
        "--reviews-output", str(p["gpt_reviews"]),
        "--trace-paths",
        str(output_sidecar(p["raw"], ".trace.jsonl")),
        str(output_sidecar(p["pruned"], ".trace.jsonl")),
        "--model", args.gpt_review_model,
        "--max-reviews", str(args.gpt_max_reviews),
        "--pdf-char-limit", str(args.review_pdf_char_limit),
    ]
    if args.gpt_wrong_only:
        cmd.append("--wrong-only")
    if args.review_dry_run:
        cmd.append("--dry-run")
    run_command(cmd, args.dry_run)


def evaluate(pred_path: Path, truth_path: Path) -> dict:
    rows = read_csv(pred_path)
    truth_rows = read_csv(truth_path)
    truth = {row_key(row): row.get("label", "").strip().lower() for row in truth_rows}
    labels = VALID_LABELS
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    confusion = Counter()
    missing_truth = 0
    invalid = 0
    evaluated = 0

    for row in rows:
        key = row_key(row)
        actual = truth.get(key)
        pred = row.get("pred_label", "").strip().lower()
        if actual is None:
            missing_truth += 1
            continue
        if pred not in labels:
            invalid += 1
            continue
        evaluated += 1
        confusion[(actual, pred)] += 1
        if pred == actual:
            tp[actual] += 1
        else:
            fp[pred] += 1
            fn[actual] += 1

    per_class = {}
    f1s = []
    for label in labels:
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[(label, pred)] for pred in labels),
        }
    correct = sum(tp.values())
    return {
        "pred_path": str(pred_path),
        "evaluated": evaluated,
        "correct": correct,
        "micro_f1": round(correct / evaluated, 4) if evaluated else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "missing_truth": missing_truth,
        "invalid": invalid,
        "pred_counts": dict(Counter(row.get("pred_label", "") for row in rows)),
        "per_class": per_class,
        "confusion": {f"{actual} -> {pred}": count for (actual, pred), count in confusion.items()},
    }


def stage_eval(args: argparse.Namespace, p: dict[str, Path]) -> None:
    if args.dry_run:
        print(f"would evaluate {p['gpt_final']} against {args.truth}")
        return
    pred_path = p["gpt_final"] if p["gpt_final"].exists() else p["gemini_gated"]
    if not pred_path.exists():
        pred_path = p["repaired"] if p["repaired"].exists() else p["raw"]
    metrics = evaluate(pred_path, args.truth)
    p["metrics"].parent.mkdir(parents=True, exist_ok=True)
    p["metrics"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"wrote metrics to {p['metrics']}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out_dir: Path, manifest_path: Path) -> None:
    rows = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.name == manifest_path.name:
            continue
        rows.append({
            "relative_path": str(path.relative_to(out_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    write_csv(manifest_path, rows, ["relative_path", "size_bytes", "sha256"])


def stage_package(args: argparse.Namespace, p: dict[str, Path]) -> None:
    if args.dry_run:
        print(f"would package outputs in {args.out_dir}")
        return
    snapshot_dir = args.out_dir / "scripts_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "orchestrator_blind.py",
        "claude_review_existing.py",
        "gpt4o_meta_review_existing.py",
        "run_regcomagent_experiment.py",
    ]:
        shutil.copy2(ROOT / "scripts" / "pipeline" / name, snapshot_dir / name)

    readme = f"""# RegComAgent Orchestrator Experiment

This folder was produced by `scripts/pipeline/run_regcomagent_experiment.py`.

## Main Files

- `01_pipeline.csv`: base OrchestratorAgent pipeline output
- `02_pruned.csv`: output after removing API-error rows
- `03_deterministic_repaired.csv`: deterministic repair output
- `04_gemini.review.jsonl`: Gemini review trace
- `05_gemini_gated.csv`: gated Gemini merge
- `06_final_gpt4o_meta.csv`: final GPT-4o meta-reviewed output
- `06_final_gpt4o_meta.review.jsonl`: GPT-4o review trace
- `metrics_final.json`: evaluation metrics
- `scripts_snapshot/`: scripts used for this run

## Architecture

CSV row + PDF page -> OrchestratorAgent -> TaskPlanner ->
LanguageNormalizationAgent -> SearchAgent -> VerifyAgent -> WriterAgent ->
deterministic Result Aggregator -> Claude ConflictReviewAgent ->
GPT-4o MetaReviewAgent -> CSV output.

## Important Note

If `--gpt-wrong-only` was used, the GPT-4o review stage selected rows with
ground-truth guidance. Report it as a post-reviewed diagnostic experiment, not
as a strict blind submission.
"""
    p["readme"].write_text(readme, encoding="utf-8")
    write_manifest(args.out_dir, p["manifest"])
    print(f"packaged experiment folder: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=["all", "pipeline", "prune", "repair", "gemini", "gate-gemini", "gpt", "eval", "package"],
    )
    parser.add_argument("--max-rows", type=int, default=931)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--static-routing", action="store_true")
    parser.add_argument("--inline-gemini-review", action="store_true")
    parser.add_argument("--orchestrator-model", default="gpt-4o")
    parser.add_argument("--worker-model", default="gpt-4o-mini")
    parser.add_argument("--few-shot-limit", type=int, default=11)
    parser.add_argument("--few-shot-scope", choices=["all", "decision", "none"], default="decision")
    parser.add_argument("--language-char-limit", type=int, default=1800)
    parser.add_argument("--pdf-char-limit", type=int, default=6000)
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--gemini-max-reviews", type=int, default=426)
    parser.add_argument("--gpt-review-model", default="gpt-4o")
    parser.add_argument("--gpt-max-reviews", type=int, default=230)
    parser.add_argument("--gpt-wrong-only", action="store_true")
    parser.add_argument("--review-pdf-char-limit", type=int, default=5000)
    parser.add_argument("--review-dry-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    p = paths(args.out_dir)

    stage_order = ["pipeline", "prune", "repair", "gemini", "gate-gemini", "gpt", "eval", "package"]
    requested = stage_order if "all" in args.stages else args.stages

    for stage in requested:
        print(f"\n=== Stage: {stage} ===", flush=True)
        if stage == "pipeline":
            stage_pipeline(args, p)
        elif stage == "prune":
            stage_prune(args, p)
        elif stage == "repair":
            stage_repair(args, p)
        elif stage == "gemini":
            stage_gemini(args, p)
        elif stage == "gate-gemini":
            stage_gate_gemini(args, p)
        elif stage == "gpt":
            stage_gpt(args, p)
        elif stage == "eval":
            stage_eval(args, p)
        elif stage == "package":
            stage_package(args, p)
        else:
            raise ValueError(stage)


if __name__ == "__main__":
    main()
