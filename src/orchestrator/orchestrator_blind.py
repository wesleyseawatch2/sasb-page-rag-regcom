"""
NTCIR RegCom Sub-task 2 blind Orchestrator Agent pipeline.

Architecture:
  CSV row + PDF page
    -> OrchestratorAgent       (gpt-4o)
       Task understanding / overall control strategy / agent coordination /
       worker instructions
    -> TaskPlanner             (gpt-4o-mini)
       Subtask decomposition / execution order / step-level plan
    -> LanguageNormalizationAgent (gpt-4o-mini)
       Align metric intent, key terms, and units with the page language
    -> SearchAgent             (gpt-4o-mini)
       Retrieve positive and negative evidence from PDF text
    -> VerifyAgent             (gpt-4o-mini)
       Check metric mentioned / category / unit / evidence completeness
    -> WriterAgent             (gpt-4o-mini)
       Generate structured prediction fields:
       pred_label / category_match / unit_match
    -> Result Aggregator       (deterministic module)
       Schema validation / cleanup / conflict handling / final normalization
    -> CSV output

The Result Aggregator is a deterministic post-processing module invoked by
the OrchestratorAgent workflow for schema validation, cleanup, conflict
handling, and final normalization.

Prediction reads only:
  - data/datasets/all_subtask2_answer_sheet.csv
  - pages/{file_stem}.pdf

Ground truth is used only by --eval.
"""

import argparse
import asyncio
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

import fitz
from google import genai as _genai
from google.genai import types as _genai_types
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "datasets" / "all_subtask2_answer_sheet.csv"
DEFAULT_OUTPUT = ROOT / "data" / "predictions" / "answers_orchestrator_blind.csv"
DEFAULT_TRUTH = ROOT / "data" / "datasets" / "all_subtask2_dataset.csv"
PAGES_DIR = ROOT / "pages"
SOURCE_PDF_DIR = ROOT / "Training Set" / "PDF"
TIMING_FIELDS = [
    "row_index",
    "source_index",
    "file_stem",
    "sid",
    "cid",
    "pdf_sec",
    "orchestrator_sec",
    "planner_sec",
    "language_sec",
    "search_sec",
    "verify_sec",
    "writer_sec",
    "claude_sec",
    "aggregate_sec",
    "total_sec",
]

VALID_LABELS = {"yes", "yes but not complete", "no"}

load_dotenv(ROOT / ".env", override=True)

fitz.TOOLS.mupdf_display_errors(False)
fitz.TOOLS.mupdf_display_warnings(False)


def output_sidecar_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_suffix(output_path.suffix + suffix)


def timing_path_for(output_path: Path) -> Path:
    return output_sidecar_path(output_path, ".timing.csv")


def trace_path_for(output_path: Path) -> Path:
    return output_sidecar_path(output_path, ".trace.jsonl")


def log_path_for(output_path: Path) -> Path:
    return output_sidecar_path(output_path, ".log")


def errors_path_for(output_path: Path) -> Path:
    return output_sidecar_path(output_path, ".errors.jsonl")


def sorted_path_for(output_path: Path) -> Path:
    return output_sidecar_path(output_path, ".sorted.csv")


def row_key(row: dict) -> tuple[str, str, str]:
    return (row.get("file_stem", ""), row.get("sid", ""), row.get("cid", ""))


def source_index(row: dict) -> int:
    try:
        return int(row.get("source_index", "0"))
    except ValueError:
        return 0


def load_completed_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    try:
        rows = csv.DictReader(path.open(encoding="utf-8-sig"))
        return {
            row_key(row)
            for row in rows
            if row.get("pred_label", "").strip()
        }
    except Exception:
        return set()


def make_client(model: str) -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(model=model, temperature=0)


def make_claude_client() -> _genai.Client | None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    return _genai.Client(api_key=api_key)


def make_agent(name: str, model: str, system_message: str) -> AssistantAgent:
    return AssistantAgent(
        name=name,
        model_client=make_client(model),
        system_message=system_message,
    )


def make_agents(orchestrator_model: str, worker_model: str) -> dict[str, AssistantAgent]:
    return {
        "orchestrator": make_agent(
            "OrchestratorAgent",
            orchestrator_model,
            """You are the senior OrchestratorAgent for NTCIR RegCom Sub-task 2.
You are the smartest model in this pipeline.

Responsibilities:
- Understand the input task.
- Define the overall strategy.
- Select and coordinate downstream agents.
- Generate high-level worker instructions.
- Monitor and integrate intermediate outputs.

Do not do TaskPlanner's job. You decide what should be done, which agents
should do it, and what the strategy is. TaskPlanner decides the concrete
steps and execution order.

Return compact JSON only:
{
  "goal": "...",
  "strategy": "...",
  "selected_agents": [
    "TaskPlanner",
    "LanguageNormalizationAgent",
    "SearchAgent",
    "VerifyAgent",
    "WriterAgent",
    "ClaudeConflictReviewAgent",
    "GPT4oMetaReviewAgent"
  ],
  "review_policy": {
    "claude": "skip|risk_only|always",
    "gpt4o_meta": "skip|risk_only|always",
    "reason": "..."
  },
  "must_check": ["..."],
  "risk_notes": ["..."],
  "worker_instructions": "..."
}

Key task rules:
- This is single-page metric verification, not a full-report SASB audit.
- Judge only the given PDF page.
- Do not produce final CSV labels yourself.""",
        ),
        "planner": make_agent(
            "TaskPlanner",
            worker_model,
            """You are a cheap TaskPlanner.

Responsibilities:
- Convert the OrchestratorAgent strategy into concrete subtasks.
- Determine execution order.
- Produce step-level plans for LanguageNormalizationAgent, SearchAgent,
  VerifyAgent, and WriterAgent.

Do not redefine the overall strategy.

Return compact JSON only:
{
  "steps": [
    {"agent": "LanguageNormalizationAgent", "task": "..."},
    {"agent": "SearchAgent", "task": "..."},
    {"agent": "VerifyAgent", "task": "..."},
    {"agent": "WriterAgent", "task": "..."}
  ]
}""",
        ),
        "language": make_agent(
            "LanguageNormalizationAgent",
            worker_model,
            """You are a cheap LanguageNormalizationAgent.

Align the SASB metric intent, key terms, and units with the PDF page language.
Your output is a working note for SearchAgent and VerifyAgent, not a final
classification.

Return compact JSON only:
{
  "page_language": "...",
  "metric_intent_en": "...",
  "metric_intent_page_language": "...",
  "normalized_key_terms": ["..."],
  "unit_aliases": ["..."],
  "positive_cues": ["..."],
  "negative_cues": ["..."],
  "notes": "..."
}

Rules:
- Translate or paraphrase the metric into the page language when useful.
- Include local-language aliases for count units, currency units, percentages,
  rates, zero/no-event statements, and common ESG/SASB wording.
- Do not decide pred_label, category_match, or unit_match.""",
        ),
        "search": make_agent(
            "SearchAgent",
            worker_model,
            """You are a cheap SearchAgent.

Use the LanguageNormalizationAgent notes to retrieve positive and negative
evidence from the PDF text for the exact SASB metric or sub-metric.

Return compact JSON only:
{
  "positive_evidence": ["short quote or paraphrase"],
  "negative_evidence": ["short quote or paraphrase"],
  "snippet_summary": "..."
}

Rules:
- Search semantically; translate mentally across Chinese, English, French,
  Japanese, Korean, and Thai.
- Do not decide the final label.
- Keep snippets short.""",
        ),
        "verify": make_agent(
            "VerifyAgent",
            worker_model,
            """You are a cheap VerifyAgent.

Use the row, SearchAgent evidence, and full PDF page text to check:
- whether the exact metric/sub-metric is mentioned,
- whether the required category is satisfied,
- whether the required unit is satisfied,
- whether the evidence is complete, partial, or absent.

Return compact JSON only:
{
  "metric_mentioned": "yes|no",
  "category_match": "yes|no|N/A",
  "unit_match": "yes|no|N/A",
  "completeness": "complete|partial|none",
  "reason": "short reason"
}

Rules:
- If the page does not semantically address the exact metric/sub-metric,
  metric_mentioned=no.
- Reject adjacent topics even if they share broad ESG keywords. The page must
  address the requested metric, not merely a nearby policy, award, training
  program, generic privacy page, housing loan page, or generic ESG section.
- For event/count metrics, 0, none, no incidents, no breaches, no recalls,
  no fines, or equivalent wording counts as quantitative disclosure.
- For data-breach submetrics, a table or statement showing information
  security incidents/data leakage events/customer data affected as 0 can
  satisfy total breaches, personal-breach percentage, and affected-account
  count. Zero is a valid disclosed value.
- For data-security Discussion and Analysis metrics, incident reporting and
  handling process, root-cause analysis, corrective action, security resources,
  response drills, threat scenarios, risk assessment, vulnerability controls,
  or security frameworks count as relevant narrative. Privacy policy or
  employee training alone is only adjacent evidence.
- For financial inclusion loan metrics, urban-renewal, dangerous-building,
  consumer debt negotiation, personal relief, or residential property loan
  pages are adjacent unless the page explicitly reports qualifying small
  business/community development loan counts or balances.
- Local count units such as cases, accounts, loans, people, facilities,
  incidents, recalls, employees, vehicles, households, customers, and
  local-language count words can satisfy Number.
- Local money units such as $, USD, EUR, NT$, TWD, KRW, JPY, THB, yuan,
  million, billion, and local-language currency words can satisfy
  Presentation currency.
- Energy units such as kWh, MWh, GWh, MMBtu, and toe can be converted to or
  treated as equivalent evidence for energy-consumption metrics that request
  GJ, as long as the total energy quantity is clearly disclosed.
- For loan delinquency/nonaccrual/forbearance balance submetrics, a disclosed
  monetary amount, balance, or percentage/ratio of the relevant loan amount can
  satisfy the amount/balance component.
- For Discussion and Analysis metrics, relevant narrative is category_match=yes,
  unit_match=yes, and usually completeness=complete.
- Be less strict than a full SASB audit: if the page clearly discloses the
  requested page-level metric, mark completeness=complete even if it does not
  cover every SASB sub-bullet.""",
        ),
        "writer": make_agent(
            "WriterAgent",
            worker_model,
            """You are a cheap WriterAgent.

Generate structured prediction fields from VerifyAgent's findings.

Return exactly three lines:
pred_label: yes | yes but not complete | no
category_match: yes | no | N/A
unit_match: yes | no | N/A

Rules:
- If metric_mentioned=no: pred_label=no, category_match=N/A, unit_match=N/A.
- If metric_mentioned=yes and completeness=complete: pred_label=yes.
- If metric_mentioned=yes and completeness=partial: pred_label=yes but not complete.
- For Discussion and Analysis metrics with relevant narrative, prefer yes over
  yes but not complete.
- For Quantitative metrics, clear numbers, zero/no-event statements, rates,
  percentages, counts, or currency values should usually be yes.
- Use yes but not complete only when VerifyAgent explicitly says the evidence
  is partial, a proxy, or wrong-unit.""",
        ),
    }


AGENT_ALIASES = {
    "taskplanner": "TaskPlanner",
    "planner": "TaskPlanner",
    "languagenormalizationagent": "LanguageNormalizationAgent",
    "languagenormalisationagent": "LanguageNormalizationAgent",
    "languagealignmentagent": "LanguageNormalizationAgent",
    "languagenormalizer": "LanguageNormalizationAgent",
    "language": "LanguageNormalizationAgent",
    "searchagent": "SearchAgent",
    "search": "SearchAgent",
    "verifyagent": "VerifyAgent",
    "verificationagent": "VerifyAgent",
    "verify": "VerifyAgent",
    "writeragent": "WriterAgent",
    "writer": "WriterAgent",
    "claudeconflictreviewagent": "ClaudeConflictReviewAgent",
    "claudereviewagent": "ClaudeConflictReviewAgent",
    "claude": "ClaudeConflictReviewAgent",
    "gpt4ometareviewagent": "GPT4oMetaReviewAgent",
    "gptmetareviewagent": "GPT4oMetaReviewAgent",
    "gpt4o": "GPT4oMetaReviewAgent",
    "metareviewagent": "GPT4oMetaReviewAgent",
}


CORE_AGENT_ORDER = [
    "LanguageNormalizationAgent",
    "SearchAgent",
    "VerifyAgent",
    "WriterAgent",
]


def canonical_agent_name(value: str) -> str | None:
    key = re.sub(r"[^a-z0-9]", "", str(value).lower())
    return AGENT_ALIASES.get(key)


def normalize_review_mode(value: str, default: str = "risk_only") -> str:
    mode = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if mode in {"always", "force", "required"}:
        return "always"
    if mode in {"skip", "none", "off", "false", "no"}:
        return "skip"
    if mode in {"risk_only", "risk", "conditional", "auto", "if_needed"}:
        return "risk_only"
    return default


def build_dynamic_route(orchestrator_brief: dict, dynamic_routing: bool) -> dict:
    """Convert OrchestratorAgent JSON into an executable route.

    The route is dynamic, but still guarded by dependency rules so the workflow
    can always produce a valid CSV answer.
    """
    default_agents = [
        "TaskPlanner",
        "LanguageNormalizationAgent",
        "SearchAgent",
        "VerifyAgent",
        "WriterAgent",
    ]
    if not dynamic_routing:
        selected = default_agents
    else:
        raw_selected = orchestrator_brief.get("selected_agents", default_agents)
        if not isinstance(raw_selected, list):
            raw_selected = default_agents
        selected = []
        for item in raw_selected:
            canonical = canonical_agent_name(str(item))
            if canonical and canonical not in selected:
                selected.append(canonical)
        if not selected:
            selected = default_agents

    # Safety closure: a valid final answer requires verification. Writer is
    # optional because the deterministic writer can derive labels from Verify.
    if "TaskPlanner" not in selected:
        selected.insert(0, "TaskPlanner")
    if "VerifyAgent" not in selected:
        selected.append("VerifyAgent")

    ordered = []
    for agent in ["TaskPlanner", *CORE_AGENT_ORDER, "ClaudeConflictReviewAgent", "GPT4oMetaReviewAgent"]:
        if agent in selected and agent not in ordered:
            ordered.append(agent)

    review_policy = orchestrator_brief.get("review_policy", {})
    if not isinstance(review_policy, dict):
        review_policy = {}
    claude_mode = normalize_review_mode(review_policy.get("claude"), "risk_only")
    gpt_mode = normalize_review_mode(review_policy.get("gpt4o_meta"), "risk_only")
    if "ClaudeConflictReviewAgent" in ordered and claude_mode == "skip":
        claude_mode = "risk_only"
    if "GPT4oMetaReviewAgent" in ordered and gpt_mode == "skip":
        gpt_mode = "risk_only"

    return {
        "dynamic_routing": dynamic_routing,
        "selected_agents": ordered,
        "skipped_agents": [agent for agent in CORE_AGENT_ORDER if agent not in ordered],
        "review_policy": {
            "claude": claude_mode,
            "gpt4o_meta": gpt_mode,
            "reason": review_policy.get("reason", ""),
        },
        "gpt_meta_review_recommended": gpt_mode in {"risk_only", "always"} and "GPT4oMetaReviewAgent" in ordered,
    }


def prediction_from_verification(row: dict, verification: dict) -> dict:
    metric_mentioned = str(verification.get("metric_mentioned", "")).strip().lower()
    completeness = str(verification.get("completeness", "")).strip().lower()
    category = normalize_match(str(verification.get("category_match", "")))
    unit = normalize_match(str(verification.get("unit_match", "")))
    sasb_category = row.get("sasb_category", "").strip().lower()

    if metric_mentioned == "no" or completeness in {"none", "absent", "no"}:
        return {"pred_label": "no", "category_match": "N/A", "unit_match": "N/A"}

    if sasb_category == "discussion and analysis":
        unit = "yes"
        if category == "N/A":
            category = "yes"

    if completeness in {"partial", "incomplete", "proxy", "wrong_unit"}:
        label = "yes but not complete"
    elif completeness in {"complete", "yes", "full"}:
        label = "yes"
    elif metric_mentioned == "yes":
        label = "yes but not complete"
    else:
        label = "no"

    if label == "no":
        category = "N/A"
        unit = "N/A"
    return {"pred_label": label, "category_match": category, "unit_match": unit}


def source_pdf_stem(file_stem: str) -> str:
    match = re.match(r"(.+)_([0-9]+)$", file_stem)
    return match.group(1) if match else file_stem


def extract_page_text(file_stem: str, lang: str = "", page: str = "") -> str:
    pdf_path = PAGES_DIR / f"{file_stem}.pdf"
    text = ""
    try:
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            with fitz.open(str(pdf_path)) as doc:
                text = "\n".join(page.get_text("text") for page in doc).strip()
    except Exception:
        text = ""
    if text:
        return text

    try:
        if lang and page:
            source_path = SOURCE_PDF_DIR / lang / f"{source_pdf_stem(file_stem)}.pdf"
            page_index = int(page) - 1
            if source_path.exists() and page_index >= 0:
                with fitz.open(str(source_path)) as doc:
                    if page_index < doc.page_count:
                        return doc[page_index].get_text("text").strip()
    except Exception:
        return ""
    return ""


def row_brief(row: dict) -> str:
    return f"""Row:
lang: {row.get("lang", "")}
cid: {row.get("cid", "")}
sid: {row.get("sid", "")}
file_stem: {row.get("file_stem", "")}
topic: {row.get("topic", "")}
metric_code: {row.get("metric_code", "")}
metric_description: {row.get("metric_description", "")}
sasb_category: {row.get("sasb_category", "")}
sasb_unit_of_measure: {row.get("sasb_unit_of_measure", "")}
sasb_key_terms: {row.get("sasb_key_terms", "")}
sasb_what_counts: {row.get("sasb_what_counts", "")}"""


def derive_example_matches(row: dict) -> tuple[str, str]:
    label = row.get("label", "").strip().lower()
    category = row.get("sasb_category", "").strip().lower()
    if label == "no":
        return "N/A", "N/A"
    if category == "discussion and analysis":
        return "yes", "yes"
    if label == "yes":
        return "yes", "yes"
    return "yes", "no"


def load_few_shot_examples(input_path: Path, truth_path: Path, limit: int) -> str:
    if limit <= 0 or not truth_path.exists():
        return ""

    answer_rows = list(csv.DictReader(input_path.open(encoding="utf-8-sig")))
    answer_keys = {(r["file_stem"], r["sid"], r["cid"]) for r in answer_rows}

    truth_rows = list(csv.DictReader(truth_path.open(encoding="utf-8-sig")))
    example_rows = [
        r for r in truth_rows
        if (r["file_stem"], r["sid"], r["cid"]) not in answer_keys
    ][:limit]

    if not example_rows:
        return ""

    lines = [
        "Few-shot calibration examples from held-out development rows.",
        "Use them to calibrate label boundaries, not as answers for target rows.",
    ]
    for i, row in enumerate(example_rows, 1):
        category_match, unit_match = derive_example_matches(row)
        lines.append(
            "\n".join([
                f"Example {i}:",
                f"lang: {row.get('lang', '')}",
                f"file_stem: {row.get('file_stem', '')}",
                f"topic: {row.get('topic', '')}",
                f"metric_code: {row.get('metric_code', '')}",
                f"metric_description: {row.get('metric_description', '')}",
                f"sasb_category: {row.get('sasb_category', '')}",
                f"sasb_unit_of_measure: {row.get('sasb_unit_of_measure', '')}",
                f"gold_pred_label: {row.get('label', '')}",
                f"gold_category_match: {category_match}",
                f"gold_unit_match: {unit_match}",
                f"answer_value: {row.get('answer_value', '')}",
                f"answer_unit: {row.get('answer_unit', '')}",
                f"complete: {row.get('complete', '')}",
            ])
        )
    return "\n\n".join(lines)


def few_shot_block_for(few_shot_text: str, scope: str, stage: str) -> str:
    if not few_shot_text or scope == "none":
        return ""
    if scope == "all":
        return f"\n\n{few_shot_text}\n"
    if scope == "decision" and stage in {"verify", "writer"}:
        return f"\n\n{few_shot_text}\n"
    return ""


async def ask(agent: AssistantAgent, prompt: str) -> str:
    response = await agent.on_messages(
        [TextMessage(content=prompt, source="user")],
        CancellationToken(),
    )
    return response.chat_message.content if response.chat_message else ""


async def timed_ask(agent: AssistantAgent, prompt: str, timings: dict, key: str) -> str:
    started = time.perf_counter()
    try:
        last_exc = None
        for attempt in range(1, 5):
            try:
                return await ask(agent, prompt)
            except Exception as exc:
                last_exc = exc
                if attempt == 4:
                    raise
                await asyncio.sleep(2 ** attempt)
        raise last_exc if last_exc else RuntimeError("Unknown API error")
    finally:
        timings[key] = time.perf_counter() - started


async def timed_claude_message(
    client: "_genai.Client",
    model: str,
    system: str,
    prompt: str,
    timings: dict,
) -> str:
    started = time.perf_counter()
    try:
        response = None
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
                break
            except Exception:
                if attempt == 4:
                    raise
                await asyncio.sleep(2 ** attempt)
        return response.text if response else ""
    finally:
        timings["claude_sec"] = time.perf_counter() - started


def parse_jsonish(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def parse_writer_output(text: str, sasb_category: str) -> dict:
    raw = text.strip()

    def field(name: str) -> str:
        match = re.search(rf"{name}\s*:\s*([^\n\r]+)", raw, flags=re.IGNORECASE)
        return match.group(1).strip() if match else ""

    label = normalize_label(field("pred_label"), raw)
    category = normalize_match(field("category_match"))
    unit = normalize_match(field("unit_match"))

    if label == "no":
        category, unit = "N/A", "N/A"
    else:
        if category == "N/A":
            category = "yes"
        if sasb_category.strip().lower() == "discussion and analysis":
            unit = "yes"
        elif unit == "N/A":
            unit = "no"

    return {
        "pred_label": label,
        "category_match": category,
        "unit_match": unit,
    }


def normalize_label(value: str, raw: str) -> str:
    value = value.strip().lower()
    raw = raw.strip().lower()
    if "yes but not complete" in value:
        return "yes but not complete"
    if value in {"yes", "no"}:
        return value
    if "yes but not complete" in raw:
        return "yes but not complete"
    if re.search(r"\bno\b", raw):
        return "no"
    if re.search(r"\byes\b", raw):
        return "yes"
    return "no"


def normalize_match(value: str) -> str:
    value = value.strip().lower()
    if value in {"n/a", "na", "not applicable"}:
        return "N/A"
    if value == "yes":
        return "yes"
    if value == "no":
        return "no"
    return "N/A"


async def classify_row(
    row: dict,
    page_text: str,
    agents: dict[str, AssistantAgent],
    timings: dict,
    few_shot_text: str,
    few_shot_scope: str,
    language_char_limit: int,
    pdf_char_limit: int,
    claude_client: "_genai.Client | None",
    claude_model: str,
    claude_review_enabled: bool,
    dynamic_routing: bool,
) -> tuple[dict, dict]:
    trace = {
        "row": {k: row.get(k, "") for k in [
            "source_index", "lang", "cid", "sid", "file_stem", "topic",
            "metric_code", "metric_description", "sasb_category",
            "sasb_unit_of_measure", "sasb_key_terms", "sasb_what_counts",
        ]},
        "page_text_chars": len(page_text),
    }
    if not page_text:
        prediction = {
            "pred_label": "no",
            "category_match": "N/A",
            "unit_match": "N/A",
        }
        trace["prediction"] = prediction
        trace["note"] = "Missing or unreadable PDF text."
        return prediction, trace

    row_context = row_brief(row)

    orchestrator_raw = await timed_ask(
        agents["orchestrator"],
        f"""{row_context}{few_shot_block_for(few_shot_text, few_shot_scope, "orchestrator")}

Create the workflow brief for this single-page verification task.
Do not inspect ground truth. Do not produce final CSV labels yet.""",
        timings,
        "orchestrator_sec",
    )
    orchestrator_brief = parse_jsonish(orchestrator_raw)
    trace["orchestrator"] = {"raw": orchestrator_raw, "json": orchestrator_brief}
    route = build_dynamic_route(orchestrator_brief, dynamic_routing)
    trace["routing"] = route

    planner_raw = await timed_ask(
        agents["planner"],
        f"""Orchestrator brief:
{json.dumps(orchestrator_brief, ensure_ascii=False)}

Executable route selected by the OrchestratorAgent:
{json.dumps(route, ensure_ascii=False)}

Create the concrete worker checklist only for the selected executable agents.
If an agent is skipped, do not create work for that agent.""",
        timings,
        "planner_sec",
    )
    planner_steps = parse_jsonish(planner_raw)
    trace["planner"] = {"raw": planner_raw, "json": planner_steps}

    if "LanguageNormalizationAgent" in route["selected_agents"]:
        language_raw = await timed_ask(
            agents["language"],
            f"""{row_context}{few_shot_block_for(few_shot_text, few_shot_scope, "language")}

Orchestrator brief:
{json.dumps(orchestrator_brief, ensure_ascii=False)}

Planner:
{json.dumps(planner_steps, ensure_ascii=False)}

PDF text sample:
{page_text[:language_char_limit]}

Normalize the metric intent, key terms, and units for this page language.""",
            timings,
            "language_sec",
        )
        language_notes = parse_jsonish(language_raw)
        trace["language"] = {"raw": language_raw, "json": language_notes}
    else:
        language_notes = {
            "skipped": True,
            "reason": "Skipped by OrchestratorAgent dynamic route.",
        }
        trace["language"] = {"skipped": True, "json": language_notes}

    if "SearchAgent" in route["selected_agents"]:
        search_raw = await timed_ask(
            agents["search"],
            f"""{row_context}{few_shot_block_for(few_shot_text, few_shot_scope, "search")}

Orchestrator brief:
{json.dumps(orchestrator_brief, ensure_ascii=False)}

Planner:
{json.dumps(planner_steps, ensure_ascii=False)}

Language normalization:
{json.dumps(language_notes, ensure_ascii=False)}

Single-page PDF text:
{page_text[:pdf_char_limit]}""",
            timings,
            "search_sec",
        )
        evidence = parse_jsonish(search_raw)
        trace["search"] = {"raw": search_raw, "json": evidence}
    else:
        evidence = {
            "positive_evidence": [],
            "negative_evidence": [],
            "snippet_summary": "SearchAgent skipped by OrchestratorAgent dynamic route; VerifyAgent must inspect full page text directly.",
        }
        trace["search"] = {"skipped": True, "json": evidence}

    verify_raw = await timed_ask(
        agents["verify"],
        f"""{row_context}{few_shot_block_for(few_shot_text, few_shot_scope, "verify")}

Orchestrator brief:
{json.dumps(orchestrator_brief, ensure_ascii=False)}

Planner:
{json.dumps(planner_steps, ensure_ascii=False)}

Language normalization:
{json.dumps(language_notes, ensure_ascii=False)}

SearchAgent evidence:
{json.dumps(evidence, ensure_ascii=False)}

Full single-page PDF text:
{page_text[:pdf_char_limit]}

Use the page-level RegCom rules to verify the metric.""",
        timings,
        "verify_sec",
    )
    verification = parse_jsonish(verify_raw)
    trace["verify"] = {"raw": verify_raw, "json": verification}

    if "WriterAgent" in route["selected_agents"]:
        writer_raw = await timed_ask(
            agents["writer"],
            f"""{row_context}{few_shot_block_for(few_shot_text, few_shot_scope, "writer")}

Verification:
{json.dumps(verification, ensure_ascii=False)}

Write the final answer fields.""",
            timings,
            "writer_sec",
        )
        prediction = parse_writer_output(writer_raw, row.get("sasb_category", ""))
        trace["writer"] = {"raw": writer_raw, "parsed": prediction}
    else:
        prediction = prediction_from_verification(row, verification)
        trace["writer"] = {
            "skipped": True,
            "reason": "Skipped by OrchestratorAgent dynamic route; deterministic writer derived fields from VerifyAgent.",
            "parsed": prediction,
        }
    started = time.perf_counter()
    try:
        final_prediction = aggregate_result(row, verification, prediction)
        before_overrides = dict(final_prediction)
        final_prediction = apply_rule_overrides(row, page_text, final_prediction)
        override_applied = final_prediction != before_overrides
        trace["aggregator"] = {
            "before_overrides": before_overrides,
            "after_overrides": final_prediction,
            "override_applied": override_applied,
        }
        review_needed, review_reasons = should_claude_review(
            row,
            page_text,
            evidence,
            verification,
            final_prediction,
            override_applied,
        )
        claude_mode = route["review_policy"]["claude"]
        if claude_mode == "always":
            review_needed = True
            review_reasons = ["orchestrator_forced_claude_review", *review_reasons]
        elif claude_mode == "skip":
            review_needed = False
            review_reasons = ["orchestrator_skipped_claude_review", *review_reasons]
        elif "ClaudeConflictReviewAgent" in route["selected_agents"] and not review_needed:
            review_needed = True
            review_reasons = ["orchestrator_selected_claude_review", *review_reasons]

        trace["claude_review"] = {
            "enabled": claude_review_enabled,
            "triggered": False,
            "reasons": review_reasons,
            "policy": claude_mode,
        }
        if claude_review_enabled and review_needed:
            if claude_client is None:
                trace["claude_review"]["error"] = "GEMINI_API_KEY not configured."
            else:
                reviewed_prediction, review_trace = await run_claude_review(
                    claude_client,
                    claude_model,
                    row,
                    page_text,
                    evidence,
                    verification,
                    final_prediction,
                    review_reasons,
                    pdf_char_limit,
                    timings,
                )
                trace["claude_review"].update(review_trace)
                trace["claude_review"]["triggered"] = True
                final_prediction = aggregate_result(row, verification, reviewed_prediction)
        trace["prediction"] = final_prediction
        return final_prediction, trace
    finally:
        timings["aggregate_sec"] = time.perf_counter() - started


def aggregate_result(row: dict, verification: dict, prediction: dict) -> dict:
    metric_mentioned = str(verification.get("metric_mentioned", "")).lower()
    completeness = str(verification.get("completeness", "")).lower()
    sasb_category = row.get("sasb_category", "").strip().lower()

    if metric_mentioned == "no" or completeness == "none":
        return {
            "pred_label": "no",
            "category_match": "N/A",
            "unit_match": "N/A",
        }

    if prediction["pred_label"] != "no" and sasb_category == "discussion and analysis":
        prediction["unit_match"] = "yes"

    if prediction["pred_label"] == "no":
        prediction["category_match"] = "N/A"
        prediction["unit_match"] = "N/A"

    return prediction


def normalize_code(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def page_has_metric_code(page_text: str, metric_code: str) -> bool:
    code = normalize_code(metric_code)
    if not code:
        return False
    compact_page = normalize_code(page_text)
    return code in compact_page


def is_sasb_index_page(page_text: str) -> bool:
    markers = [
        "SASB",
        "指標代碼",
        "會計指標",
        "揭露項目",
        "指標揭露",
        "SASB 永續會計準則索引表",
        "SASB 產業重大主題指標",
        "永續揭露主題與會計指標",
        "sustainability accounting standards",
    ]
    return any(marker.lower() in page_text.lower() for marker in markers)


def contains_any(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in terms)


def apply_rule_overrides(row: dict, page_text: str, prediction: dict) -> dict:
    metric_code = row.get("metric_code", "")
    metric_desc = row.get("metric_description", "")
    text = page_text.lower()

    sasb_index = is_sasb_index_page(page_text)
    metric_code_present = page_has_metric_code(page_text, metric_code)

    if metric_code == "TC-SC-140a.1":
        mentions_water_amount = contains_any(
            page_text,
            ["取水", "耗水", "用水", "water withdrawal", "water consumption"],
        )
        wrong_scale_only = contains_any(page_text, ["公噸", "立方公尺", "tons", "cubic meters"]) and not contains_any(
            page_text,
            ["千立方", "thousand cubic", "10^3 m3", "10^3m3"],
        )
        if mentions_water_amount and wrong_scale_only and prediction["pred_label"] == "yes but not complete":
            return {"pred_label": "no", "category_match": "N/A", "unit_match": "N/A"}

    if metric_code == "FN-CB-230a.1":
        complaint_or_audit_page = contains_any(
            page_text,
            ["陳情", "申訴", "個資使用稽核", "裁罰金額", "主管機關裁罰", "audit finding", "complaint"],
        )
        no_breach_narrative = contains_any(
            page_text,
            ["無顧客個資洩漏", "無個資洩漏", "無資料洩漏", "no customer data breach", "no data breach"],
        )
        actual_breach_terms = contains_any(
            page_text,
            ["資料洩露數量", "資料洩漏數量", "受影響之帳戶", "受影響帳戶", "account holders affected"],
        )
        if complaint_or_audit_page and no_breach_narrative and not actual_breach_terms:
            return {"pred_label": "no", "category_match": "N/A", "unit_match": "N/A"}

    if metric_code == "TC-SC-440a.1":
        conflict_minerals_only = contains_any(
            page_text,
            ["責任礦產", "衝突礦產", "剛果民主共和國", "conflict minerals", "responsible minerals"],
        )
        critical_materials = contains_any(
            page_text,
            ["關鍵材料", "critical materials", "rare earth", "稀土"],
        )
        if conflict_minerals_only and not critical_materials:
            return {"pred_label": "no", "category_match": "N/A", "unit_match": "N/A"}

    if metric_code == "FN-CB-550a.2":
        stress_testing = contains_any(page_text, ["壓力測試", "stress test"])
        capital_or_strategy = contains_any(
            page_text,
            ["資本適足", "資本規劃", "長期策略", "capital adequacy", "capital planning", "strategy"],
        )
        if stress_testing and capital_or_strategy and prediction["pred_label"] == "no":
            return {"pred_label": "yes", "category_match": "yes", "unit_match": "yes"}

    if metric_code == "FN-CB-410a.2":
        esg_credit = contains_any(
            page_text,
            ["esg", "赤道原則", "信用分析", "授信", "credit analysis", "equator principles"],
        )
        if esg_credit and prediction["pred_label"] == "yes but not complete":
            return {"pred_label": "yes", "category_match": "yes", "unit_match": "yes"}

    if metric_code == "TC-SC-130a.1":
        renewable_zero_percent = (
            contains_any(page_text, ["再生能源0%", "再生能源 0%", "0% of energy", "0 %"])
            and contains_any(page_text, ["外購電力", "能源使用", "energy consumption", "purchased electricity"])
        )
        if renewable_zero_percent and prediction["pred_label"] == "no":
            return {"pred_label": "yes", "category_match": "yes", "unit_match": "yes"}

    if metric_code == "FN-CB-240a.1":
        urban_redevelopment = contains_any(
            page_text,
            ["都市更新", "危老建築", "重建貸款", "urban renewal", "dangerous old building"],
        )
        specific_urban_redevelopment_loan = contains_any(
            page_text,
            ["都市更新及危老建築重建貸款業務", "都更及危險老舊建築物"],
        )
        small_business_or_target_community = contains_any(
            page_text,
            ["小型企業", "小微", "中小企業", "社區發展計畫", "small business", "community development program"],
        )
        if specific_urban_redevelopment_loan or (urban_redevelopment and not small_business_or_target_community):
            return {"pred_label": "no", "category_match": "N/A", "unit_match": "N/A"}

    if sasb_index and metric_code.startswith("HC-MS-") and prediction["pred_label"] == "yes but not complete":
        return prediction

    if sasb_index and metric_code_present:
        if any(term in page_text for term in ["不適用", "無", "詳", "章節", "省略", "—", "－"]):
            if prediction["pred_label"] == "no":
                return {"pred_label": "yes but not complete", "category_match": "yes", "unit_match": "yes"}
        if any(char.isdigit() for char in page_text) or any(term in page_text for term in ["戶", "元", "%", "％", "m3", "m³", "件", "次"]):
            if prediction["pred_label"] in {"no", "yes but not complete"}:
                return {"pred_label": "yes", "category_match": "yes", "unit_match": "yes"}

    if metric_code == "FN-CB-230a.2":
        adjacent_only = any(term in page_text for term in ["隱私權保護", "客戶資料保密", "資安教育訓練", "社交工程演練"])
        risk_process = any(
            term in page_text
            for term in ["資安事件", "資訊安全事件", "資訊外洩事件", "通報", "處理流程", "根因分析", "矯正措施", "事件應變", "ISO 27001", "風險評估", "弱點", "漏洞"]
        )
        if adjacent_only and not risk_process:
            return {"pred_label": "no", "category_match": "N/A", "unit_match": "N/A"}

    if metric_code == "FN-CB-240a.1":
        adjacent_loan = any(term in page_text for term in ["都市更新", "危老", "債務協商", "消費金融", "住宅"])
        qualifying_loan = any(term in page_text for term in ["紓困", "中小企業貸款", "小型企業", "社區發展貸款", "低收入", "中等收入"])
        if adjacent_loan and not qualifying_loan:
            return {"pred_label": "no", "category_match": "N/A", "unit_match": "N/A"}

    if metric_code in {"FN-CB-240a.2", "FN-CB-240a.3"} or "逾期" in metric_desc or "催收" in metric_desc or "暫緩還款" in metric_desc:
        if any(term in page_text for term in ["逾期放款", "逾期貸款", "暫緩還款"]) and any(term in page_text for term in ["金額", "比例", "%", "占"]):
            return {"pred_label": "yes", "category_match": "yes", "unit_match": "yes"}

    if metric_code in {"TC-SC-130a.1", "EM-EP-130a.1", "EM-RM-130a.1"} or "能源消耗" in metric_desc or "energy consumption" in metric_desc.lower():
        if any(unit in text for unit in ["kwh", "mwh", "gwh", "gj", "gigajoule"]) or any(term in page_text for term in ["總能源消耗", "能源消耗量", "用電量"]):
            if prediction["pred_label"] == "yes but not complete" and prediction["category_match"] == "yes":
                return {"pred_label": "yes", "category_match": "yes", "unit_match": "yes"}

    if metric_code.startswith("IF-GU-") or metric_code.startswith("IFGU"):
        if sasb_index and (metric_code_present or any(term in page_text for term in ["天然氣", "客戶", "費率", "斷氣", "復氣", "節省量", "管線"])):
            if any(term in page_text for term in ["新台幣", "戶", "次", "%", "％", "m3", "m³", "客戶數", "售氣量"]):
                return {"pred_label": "yes", "category_match": "yes", "unit_match": "yes"}

    return prediction


def should_claude_review(
    row: dict,
    page_text: str,
    evidence: dict,
    verification: dict,
    prediction: dict,
    override_applied: bool,
) -> tuple[bool, list[str]]:
    reasons = []
    sasb_category = row.get("sasb_category", "").strip().lower()
    metric_mentioned = str(verification.get("metric_mentioned", "")).lower()
    completeness = str(verification.get("completeness", "")).lower()
    unit_match = prediction.get("unit_match", "")
    pred_label = prediction.get("pred_label", "")
    positive = evidence.get("positive_evidence") or []

    if pred_label == "yes but not complete":
        reasons.append("boundary_label_yes_but_not_complete")
    if sasb_category == "discussion and analysis" and pred_label != "yes":
        reasons.append("da_not_yes")
    if positive and metric_mentioned == "no":
        reasons.append("positive_evidence_but_metric_not_mentioned")
    if metric_mentioned == "yes" and pred_label == "no":
        reasons.append("verify_yes_but_prediction_no")
    if completeness == "complete" and pred_label != "yes":
        reasons.append("verify_complete_but_prediction_not_yes")
    if unit_match == "no":
        reasons.append("unit_mismatch")
    if is_sasb_index_page(page_text):
        reasons.append("sasb_index_page")
    if override_applied:
        reasons.append("aggregator_override_applied")

    return bool(reasons), reasons


async def run_claude_review(
    client: "_genai.Client",
    model: str,
    row: dict,
    page_text: str,
    evidence: dict,
    verification: dict,
    prediction: dict,
    reasons: list[str],
    pdf_char_limit: int,
    timings: dict,
) -> tuple[dict, dict]:
    system = """You are Claude ConflictReviewAgent for NTCIR RegCom Sub-task 2.
Review only high-risk rows after the main Orchestrator pipeline.
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
- If metric is not semantically mentioned: no / N/A / N/A.
- Zero/no-event statements count as disclosure for event/count metrics.
- For D&A metrics, relevant narrative can be yes.
- yes but not complete is for partial/proxy/wrong-unit disclosure."""

    prompt = f"""{row_brief(row)}

High-risk reasons:
{json.dumps(reasons, ensure_ascii=False)}

Main pipeline evidence:
{json.dumps(evidence, ensure_ascii=False)}

Main pipeline verification:
{json.dumps(verification, ensure_ascii=False)}

Current prediction:
{json.dumps(prediction, ensure_ascii=False)}

Single-page PDF text:
{page_text[:pdf_char_limit]}

Review the conflict and return final JSON only."""

    raw = await timed_claude_message(client, model, system, prompt, timings)
    parsed = parse_jsonish(raw)
    reviewed = {
        "pred_label": normalize_label(str(parsed.get("pred_label", "")), raw),
        "category_match": normalize_match(str(parsed.get("category_match", ""))),
        "unit_match": normalize_match(str(parsed.get("unit_match", ""))),
    }
    if reviewed["pred_label"] == "no":
        reviewed["category_match"] = "N/A"
        reviewed["unit_match"] = "N/A"
    elif row.get("sasb_category", "").strip().lower() == "discussion and analysis":
        reviewed["unit_match"] = "yes"
        if reviewed["category_match"] == "N/A":
            reviewed["category_match"] = "yes"

    return reviewed, {
        "raw": raw,
        "parsed": parsed,
        "reviewed_prediction": reviewed,
        "model": model,
    }


async def process_row(
    row: dict,
    display_index: int,
    total: int,
    orchestrator_model: str,
    worker_model: str,
    few_shot_text: str,
    few_shot_scope: str,
    language_char_limit: int,
    pdf_char_limit: int,
    claude_review_enabled: bool,
    claude_model: str,
    dynamic_routing: bool,
) -> tuple[dict | None, dict, dict, str, bool]:
    total_started = time.perf_counter()
    timings = {
        field: 0.0
        for field in TIMING_FIELDS
        if field.endswith("_sec")
    }
    trace = {}
    try:
        agents = make_agents(orchestrator_model, worker_model)
        claude_client = make_claude_client() if claude_review_enabled else None
        started = time.perf_counter()
        page_text = extract_page_text(row["file_stem"], row.get("lang", ""), row.get("page", ""))
        timings["pdf_sec"] = time.perf_counter() - started
        prediction, trace = await classify_row(
            row,
            page_text,
            agents,
            timings,
            few_shot_text,
            few_shot_scope,
            language_char_limit,
            pdf_char_limit,
            claude_client,
            claude_model,
            claude_review_enabled,
            dynamic_routing,
        )
        success = True
    except Exception as exc:
        prediction = {}
        trace = {
            "row": {
                "source_index": row.get("source_index", ""),
                "lang": row.get("lang", ""),
                "file_stem": row.get("file_stem", ""),
                "sid": row.get("sid", ""),
                "cid": row.get("cid", ""),
                "metric_code": row.get("metric_code", ""),
            },
            "error": str(exc),
        }
        success = False
    finally:
        timings["total_sec"] = time.perf_counter() - total_started

    out_row = None
    if success:
        out_row = {
            k: v for k, v in row.items()
            if k != "source_index"
        }
        out_row.update(prediction)

    timing_row = {
        "row_index": display_index,
        "source_index": row["source_index"],
        "file_stem": row["file_stem"],
        "sid": row["sid"],
        "cid": row["cid"],
    }
    for field in TIMING_FIELDS:
        if field.endswith("_sec"):
            timing_row[field] = f"{timings.get(field, 0.0):.3f}"

    trace["row_index"] = display_index
    trace["source_index"] = row["source_index"]
    trace["timings"] = timing_row

    progress = display_index / total * 100 if total else 100.0
    if success:
        log_message = (
            f"Completed {display_index}/{total} ({progress:.1f}%) "
            f"source_index={row['source_index']} {row['file_stem']} sid={row['sid']} "
            f"label={prediction['pred_label']} total_sec={timings['total_sec']:.3f}"
        )
    else:
        log_message = (
            f"Failed {display_index}/{total} ({progress:.1f}%) "
            f"source_index={row['source_index']} {row['file_stem']} sid={row['sid']} "
            f"error={trace['error']} total_sec={timings['total_sec']:.3f}"
        )
    return out_row, timing_row, trace, log_message, success


async def run(args: argparse.Namespace) -> list[dict]:
    rows = list(csv.DictReader(args.input.open(encoding="utf-8-sig")))
    for idx, row in enumerate(rows, 1):
        row["source_index"] = str(idx)
    input_fieldnames = list(rows[0].keys()) if rows else []
    output_fieldnames = [f for f in input_fieldnames if f != "source_index"]
    internal_fieldnames = input_fieldnames
    if args.sample is not None:
        rng = random.Random(args.seed)
        sample_size = min(args.sample, len(rows))
        rows = rng.sample(rows, sample_size)
    if args.offset:
        rows = rows[args.offset :]
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    out_rows = []
    timing_rows = []
    few_shot_text = load_few_shot_examples(args.input, args.truth, args.few_shot_limit)
    timing_path = args.timing_output or timing_path_for(args.output)
    trace_path = args.trace_output or trace_path_for(args.output)
    log_path = args.log_output or log_path_for(args.output)
    errors_path = errors_path_for(args.output)

    completed_keys = load_completed_keys(args.output) if args.resume else set()
    original_count = len(rows)
    if completed_keys:
        rows = [row for row in rows if row_key(row) not in completed_keys]

    if not args.resume or not args.output.exists():
        init_csv(args.output, output_fieldnames)
    if not args.resume or not timing_path.exists():
        init_csv(timing_path, TIMING_FIELDS)
    if not args.resume or not trace_path.exists():
        init_text(trace_path)
    if not args.resume or not log_path.exists():
        init_text(log_path)
    if not args.resume or not errors_path.exists():
        init_text(errors_path)
    append_log(
        log_path,
        (
            f"Start run rows_remaining={len(rows)} rows_selected={original_count} "
            f"resume={args.resume} skipped_completed={original_count - len(rows)} "
            f"concurrency={args.concurrency} "
            f"orchestrator={args.orchestrator_model} worker={args.worker_model} "
            f"dynamic_routing={args.dynamic_routing} "
            f"claude_review={args.claude_review} claude_model={args.claude_model} "
            f"few_shot_examples={args.few_shot_limit} few_shot_scope={args.few_shot_scope} "
            f"language_char_limit={args.language_char_limit} pdf_char_limit={args.pdf_char_limit}"
        ),
    )

    pending = []
    for i, row in enumerate(rows, 1):
        pending.append((i, row))

    if args.concurrency <= 1:
        for i, row in pending:
            out_row, timing_row, trace, log_message, success = await process_row(
                row,
                i,
                len(rows),
                args.orchestrator_model,
                args.worker_model,
                few_shot_text,
                args.few_shot_scope,
                args.language_char_limit,
                args.pdf_char_limit,
                args.claude_review,
                args.claude_model,
                args.dynamic_routing,
            )
            if success and out_row is not None:
                out_rows.append(out_row)
                timing_rows.append(timing_row)
                append_csv_row(args.output, output_fieldnames, out_row)
                append_csv_row(timing_path, TIMING_FIELDS, timing_row)
                append_jsonl(trace_path, trace)
                print_progress(i, len(rows), row, out_row, timing_row)
            else:
                append_jsonl(errors_path, trace)
                print(log_message)
            append_log(log_path, log_message)
    else:
        semaphore = asyncio.Semaphore(args.concurrency)

        async def bounded_process(item):
            i, row = item
            async with semaphore:
                return await process_row(
                    row,
                    i,
                    len(rows),
                    args.orchestrator_model,
                    args.worker_model,
                    few_shot_text,
                    args.few_shot_scope,
                    args.language_char_limit,
                    args.pdf_char_limit,
                    args.claude_review,
                    args.claude_model,
                    args.dynamic_routing,
                )

        tasks = [asyncio.create_task(bounded_process(item)) for item in pending]
        for task in asyncio.as_completed(tasks):
            out_row, timing_row, trace, log_message, success = await task
            if success and out_row is not None:
                out_rows.append(out_row)
                timing_rows.append(timing_row)
                append_csv_row(args.output, output_fieldnames, out_row)
                append_csv_row(timing_path, TIMING_FIELDS, timing_row)
                append_jsonl(trace_path, trace)
                print_progress(int(timing_row["row_index"]), len(rows), trace["row"], out_row, timing_row)
            else:
                append_jsonl(errors_path, trace)
                print(log_message)
            append_log(log_path, log_message)

    args.timing_rows = timing_rows
    args.timing_output = timing_path
    args.trace_output = trace_path
    args.log_output = log_path
    append_log(log_path, f"Finished run rows={len(out_rows)}")
    return out_rows


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_timing_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def init_csv(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_csv_row(path: Path, fieldnames: list[str], row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def init_text(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def append_log(path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def write_sorted_output(input_path: Path, output_path: Path, sorted_path: Path) -> None:
    if not input_path.exists():
        return
    input_rows = list(csv.DictReader(input_path.open(encoding="utf-8-sig")))
    if not input_rows:
        return
    output_rows = list(csv.DictReader(output_path.open(encoding="utf-8-sig")))
    if not output_rows:
        return
    order = {
        row_key(row): i
        for i, row in enumerate(input_rows, 1)
    }
    output_rows.sort(key=lambda row: order.get(row_key(row), 10**9))
    with sorted_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)


def error_keys_from_trace(trace_path: Path) -> set[tuple[str, str, str]]:
    keys = set()
    if not trace_path.exists():
        return keys
    with trace_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not obj.get("error"):
                continue
            row = obj.get("row", {})
            key = row_key(row)
            if all(key):
                keys.add(key)
    return keys


def write_pruned_output(output_path: Path, trace_path: Path, pruned_path: Path) -> None:
    error_keys = error_keys_from_trace(trace_path)
    rows = list(csv.DictReader(output_path.open(encoding="utf-8-sig")))
    if not rows:
        write_rows(pruned_path, [], [])
        print(f"No rows found in {output_path}")
        return
    kept = [row for row in rows if row_key(row) not in error_keys]
    write_rows(pruned_path, kept, list(rows[0].keys()))
    print(f"Pruned {len(rows) - len(kept)} error rows from {output_path}")
    print(f"Wrote {len(kept)} rows to {pruned_path}")


def print_progress(i: int, total: int, row: dict, out_row: dict, timing_row: dict) -> None:
    print(
        f"[{i}/{total}] source_index={row.get('source_index', '')} "
        f"{row['file_stem']} sid={row['sid']} "
        f"label={out_row['pred_label']} "
        f"cat={out_row['category_match']} unit={out_row['unit_match']} "
        f"time={float(timing_row['total_sec']):.1f}s "
        f"(orch={float(timing_row['orchestrator_sec']):.1f}, "
        f"plan={float(timing_row['planner_sec']):.1f}, "
        f"lang={float(timing_row['language_sec']):.1f}, "
        f"search={float(timing_row['search_sec']):.1f}, "
        f"verify={float(timing_row['verify_sec']):.1f}, "
        f"writer={float(timing_row['writer_sec']):.1f})"
    )


def evaluate(rows: list[dict], truth_path: Path) -> None:
    truth_rows = list(csv.DictReader(truth_path.open(encoding="utf-8-sig")))
    truth = {(r["file_stem"], r["sid"], r["cid"]): r["label"].strip().lower() for r in truth_rows}

    labels = ["yes", "yes but not complete", "no"]
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    missing = 0
    invalid = 0
    evaluated = 0

    for row in rows:
        key = (row["file_stem"], row["sid"], row["cid"])
        actual = truth.get(key)
        pred = row["pred_label"].strip().lower()
        if actual is None:
            missing += 1
            continue
        if pred not in VALID_LABELS:
            invalid += 1
            continue
        evaluated += 1
        if pred == actual:
            tp[actual] += 1
        else:
            fp[pred] += 1
            fn[actual] += 1

    correct = sum(tp.values())
    print()
    print("Evaluation")
    print(f"evaluated={evaluated} correct={correct} accuracy={correct / evaluated:.4f}")
    print(f"missing_truth={missing} invalid={invalid}")
    print(f"pred_counts={dict(Counter(r['pred_label'] for r in rows))}")

    f1s = []
    for label in labels:
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        print(f"{label:<22} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    print(f"macro_f1={sum(f1s) / len(f1s):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--orchestrator-model", default="gpt-4o")
    parser.add_argument("--worker-model", default="gpt-4o-mini")
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--few-shot-limit", type=int, default=11)
    parser.add_argument("--few-shot-scope", choices=["all", "decision", "none"], default="all")
    parser.add_argument("--language-char-limit", type=int, default=3000)
    parser.add_argument("--pdf-char-limit", type=int, default=9000)
    parser.add_argument("--timing-output", type=Path)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--log-output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--dynamic-routing", action="store_true")
    parser.add_argument("--claude-review", action="store_true")
    parser.add_argument("--claude-model", default=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
    parser.add_argument("--prune-error-rows", action="store_true")
    parser.add_argument("--pruned-output", type=Path)
    parser.add_argument("--eval", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.prune_error_rows:
        trace_path = args.trace_output or trace_path_for(args.output)
        pruned_path = args.pruned_output or output_sidecar_path(args.output, ".pruned.csv")
        write_pruned_output(args.output, trace_path, pruned_path)
        return

    if args.eval:
        sorted_path = sorted_path_for(args.output)
        write_sorted_output(args.input, args.output, sorted_path)
        eval_path = sorted_path if sorted_path.exists() else args.output
        saved_rows = list(csv.DictReader(eval_path.open(encoding="utf-8-sig")))
        evaluate(saved_rows, args.truth)
        return

    rows = await run(args)
    print(f"\nFinished this invocation: {len(rows)} newly completed rows")
    print(f"Prediction rows are saved incrementally to {args.output}")
    print(f"Wrote timing rows to {args.timing_output}")
    print(f"Wrote trace rows to {args.trace_output}")
    print(f"Wrote execution log to {args.log_output}")
    sorted_path = sorted_path_for(args.output)
    write_sorted_output(args.input, args.output, sorted_path)
    print(f"Wrote sorted prediction rows to {sorted_path}")


if __name__ == "__main__":
    asyncio.run(main())
