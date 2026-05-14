"""Curate high-quality grading assets for the finance-QA RAG smoke run.

Earlier iterations hit four classes of mismatch between LLM-emitted
scenarios and the rubric / interpolator contracts:

1. ``${setup_data.queries.X.query}`` interpolation in the request body
   never resolves — the trace interpolator only reads from ``captures``
   (keyed by trace step id), not ``setup_data``. So *every value the
   request actually needs* must be inline in the scenario YAML.

2. ``_resolve_gold_path`` (used by ``oracle_set_overlap``,
   ``validate_curated_gold``) walks ``setup_data`` directly: paths must
   NOT carry a leading ``setup_data.`` segment.

3. Rubric ``target`` paths resolve against ``captures``, where each
   step's response lives at ``captures[<step_id>]`` with keys ``status``
   / ``headers`` / ``body``. So the right path is ``<step_id>.body.X``,
   not ``trace.<step_id>.response.body.X``.

4. ``BehavioralEquivalence`` compares ``target`` (resolved via captures)
   against ``expected`` taken AS A LITERAL VALUE — there is no
   path-vs-path comparison. Cross-step equivalence must therefore route
   through an LLM judge (``llm_judge_semantic_eq``) with a canonical
   gold in setup_data.

The 18 scenarios below are written with all four constraints honored:

- Trace bodies are fully inline (question text + search_results list);
  no ``${setup_data...}`` placeholders.
- Rubric paths into setup_data drop the ``setup_data.`` prefix.
- Rubric ``target`` paths use ``<step_id>.body.<field>``.
- Cross-step equivalence scenarios use ``llm_judge_semantic_eq``
  pointed at a curated gold answer in setup_data.

Setup_data files written (loader keys by file stem):

- ``queries.json`` (still emitted, indexed for documentation, not
  referenced by traces).
- ``gold_answers.json``: dict keyed by scenario ``id`` with ``answer``,
  ``alt_ans``, ``answer_type``, ``question_type``. ``alt_ans`` doubles
  as the false-premise rubric's ``expected_falsity_path`` target.
- ``gold_supports.json``: dict keyed by scenario ``id`` listing the
  passage_ids that genuinely support the gold answer. ``oracle_set_overlap``
  uses this as ``gold_set_path``.

Rubric signature reference:
- ``literal_match(target, expected)``
- ``regex_match(target, pattern)``
- ``schema_match(target, must_have_fields, field_types?)``
- ``numeric_range(target, min_value, max_value)``
- ``subset_match(target, acceptable_source, acceptable_key?, min_overlap?)``
- ``behavioral_equivalence(target, expected, case_sensitive?)``
- ``oracle_set_overlap(target, gold_set_path, target_key?, min_recall?, top_k?)``
- ``llm_judge_coverage(target, must_contain_facts, judge_context_path?, strictness?, router)``
- ``llm_judge_semantic_eq(target, gold_path, alt_path?, strictness?, router)``
- ``llm_judge_false_premise(target, expected_falsity_path?, router)``
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

_WORKTREE = Path(__file__).resolve().parent.parent
COURSE_ID = "course_f918e889a33c"
COURSE_ROOT = _WORKTREE / "workspaces" / "outcome" / COURSE_ID
SETUP_DIR = COURSE_ROOT / "private" / "grader" / "_setup"
SCENARIOS_DIR = COURSE_ROOT / "private" / "grader" / "scenarios"


# ---------------- inline scenario data ----------------
#
# Each entry below carries:
#   - scenario id (used for filenames + as the key in gold_answers /
#     gold_supports)
#   - question text + search_results that the trace will POST inline
#   - gold answer + alt_ans (used by llm_judge_semantic_eq and
#     llm_judge_false_premise)
#   - gold_supports: passage_ids that support the gold answer (used by
#     oracle_set_overlap)

SCENARIOS = [
    {
        "id": "happy_valid_q1",
        "category": "happy_path",
        "question": "What was Acme Corp's Q3 2024 total revenue?",
        "search_results": [
            {"passage_id": "acme_q3_24_rev", "text": "Acme Corp reported Q3 2024 total revenue of $12.4 billion, up 8% year over year."},
            {"passage_id": "acme_q3_24_margin", "text": "Operating margin in the quarter was 18.6%, supported by supply-chain normalization."},
            {"passage_id": "acme_q2_24_rev", "text": "Acme Corp's Q2 2024 revenue totaled $11.9 billion, a slight decline versus Q1 2024."},
            {"passage_id": "rival_q3_24", "text": "Competitor Helix Industries reported Q3 2024 revenue of $9.1 billion."},
        ],
        "gold_answer": "$12.4 billion",
        "alt_ans": ["12.4 billion dollars", "$12.4B", "USD 12.4 billion"],
        "gold_supports": ["acme_q3_24_rev"],
    },
    {
        "id": "happy_valid_q2",
        "category": "happy_path",
        "question": "What operating margin did GlobalBank report for fiscal 2023?",
        "search_results": [
            {"passage_id": "gb_fy23_margin", "text": "GlobalBank disclosed a fiscal 2023 operating margin of 31.2%, supported by lower credit provisions."},
            {"passage_id": "gb_fy23_rev", "text": "Fiscal 2023 total revenue at GlobalBank rose to $48.2 billion."},
            {"passage_id": "industry_avg_margin", "text": "Industry-wide operating margins for diversified global banks averaged 26.4% in fiscal 2023."},
        ],
        "gold_answer": "31.2%",
        "alt_ans": ["31.2 percent", "thirty-one point two percent"],
        "gold_supports": ["gb_fy23_margin"],
    },
    {
        "id": "happy_valid_q3",
        "category": "happy_path",
        "question": "How much free cash flow did NorthStar Energy generate in 2024?",
        "search_results": [
            {"passage_id": "ns_2024_fcf", "text": "NorthStar Energy generated free cash flow of $2.8 billion in 2024, after capital expenditures of $1.6 billion."},
            {"passage_id": "ns_2024_capex", "text": "Capital expenditures in 2024 were $1.6 billion, weighted toward upstream sustaining investment."},
            {"passage_id": "ns_2024_ebitda", "text": "Adjusted EBITDA for 2024 reached $5.4 billion, a record for NorthStar Energy."},
        ],
        "gold_answer": "$2.8 billion",
        "alt_ans": ["2.8 billion dollars", "$2.8B"],
        "gold_supports": ["ns_2024_fcf"],
    },
    {
        "id": "happy_valid_q4",
        "category": "happy_path",
        "question": "What was Vertex Industries' net debt at the end of 2023?",
        "search_results": [
            {"passage_id": "vtx_2023_net_debt", "text": "Vertex Industries closed 2023 with net debt of $7.8 billion, after applying $0.9 billion of cash to scheduled debt repayments."},
            {"passage_id": "vtx_2023_gross_debt", "text": "Gross debt at year-end 2023 was $9.4 billion, broadly flat versus the prior year."},
            {"passage_id": "vtx_2023_leverage", "text": "Vertex Industries' net leverage ratio finished 2023 at 2.3x adjusted EBITDA."},
        ],
        "gold_answer": "$7.8 billion",
        "alt_ans": ["7.8 billion dollars", "$7.8B"],
        "gold_supports": ["vtx_2023_net_debt"],
    },
    {
        "id": "happy_valid_q5",
        "category": "happy_path",
        "question": "What did Sentinel Insurance report as diluted EPS for Q2 2024?",
        "search_results": [
            {"passage_id": "sen_q2_24_eps", "text": "Sentinel Insurance reported Q2 2024 diluted earnings per share of $3.42, up from $2.91 a year earlier."},
            {"passage_id": "sen_q2_24_rev", "text": "Net written premiums in Q2 2024 grew 6% year over year to $5.6 billion."},
            {"passage_id": "sen_q2_24_book", "text": "Book value per share at Sentinel Insurance ended Q2 2024 at $78.40."},
        ],
        "gold_answer": "$3.42",
        "alt_ans": ["3.42", "$3.42 per share"],
        "gold_supports": ["sen_q2_24_eps"],
    },
    {
        "id": "boundary_single_passage",
        "category": "boundary",
        "question": "What was Harbor Logistics' Q1 2024 revenue?",
        "search_results": [
            {"passage_id": "p1", "text": "Harbor Logistics reported Q1 2024 revenue of $3.2 billion, up 4% year over year."},
        ],
        "gold_answer": "$3.2 billion",
        "alt_ans": ["3.2 billion dollars", "$3.2B"],
        "gold_supports": ["p1"],
    },
    {
        "id": "boundary_conflicting_passages",
        "category": "boundary",
        "question": "What was Meridian Capital's net income in Q4 2023?",
        "search_results": [
            {"passage_id": "m1", "text": "Meridian Capital reported Q4 2023 net income of $480 million."},
            {"passage_id": "m2", "text": "Meridian Capital Q4 2023 net income was $520 million, according to the press release."},
        ],
        # Either reported figure is a valid extractive answer; the
        # passages disagree but both are sourced. We accept either.
        "gold_answer": "$480 million",
        "alt_ans": [
            "$520 million",
            "Meridian Capital reported Q4 2023 net income of $480 million.",
            "Meridian Capital Q4 2023 net income was $520 million.",
        ],
        "gold_supports": ["m1", "m2"],
    },
    {
        "id": "boundary_two_passages",
        "category": "boundary",
        "question": "How many full-time employees did Vector Technologies report at the end of 2024?",
        "search_results": [
            {"passage_id": "v1", "text": "Vector Technologies ended 2024 with 44,200 full-time employees, a modest decrease from 45,100 a year earlier."},
            {"passage_id": "v2", "text": "Vector Technologies opened three new regional engineering hubs during 2024."},
        ],
        "gold_answer": "44,200 full-time employees",
        "alt_ans": ["44,200", "44200 full-time employees"],
        "gold_supports": ["v1"],
    },
    {
        "id": "malformed_missing_question",
        "category": "malformed_input",
        "question": None,  # Sentinel — assembler omits the field entirely.
        "search_results": [{"passage_id": "p1", "text": "Some text."}],
        "expect_status_4xx": True,
    },
    {
        "id": "malformed_empty_search_results",
        "category": "malformed_input",
        "question": "What was the operating margin?",
        "search_results": [],  # empty list — runtime will likely 4xx.
        "expect_status_4xx": True,
    },
    {
        "id": "malformed_bad_passage_shape",
        "category": "malformed_input",
        "question": "Anything?",
        # Item missing the required `text` field.
        "search_results": [{"passage_id": "p1"}],
        "expect_status_4xx": True,
    },
    {
        "id": "out_of_scope_q_fp_0001",
        "category": "out_of_scope",
        "question": "When did Acme Corp file for Chapter 11 bankruptcy in 2024?",
        "search_results": [
            {"passage_id": "acme_cash", "text": "Acme Corp reported a cash balance of $4.2 billion at quarter end and reaffirmed its full-year guidance."},
            {"passage_id": "acme_rating", "text": "A leading rating agency affirmed Acme Corp's investment-grade credit rating with a stable outlook."},
            {"passage_id": "acme_buyback", "text": "The Acme Corp board authorized an additional $1.5 billion share repurchase program."},
        ],
        "gold_answer": "The provided passages do not support a 2024 Chapter 11 filing; the service should abstain.",
        "alt_ans": [
            "Acme Corp did not file for bankruptcy in 2024. The passages report strong cash position, an affirmed investment-grade credit rating, and an active share buyback — all inconsistent with a Chapter 11 filing.",
        ],
        "gold_supports": [],
        "is_false_premise": True,
    },
    {
        "id": "out_of_scope_q_fp_0002",
        "category": "out_of_scope",
        "question": "What was GlobalBank's IPO price when it went public in 2024?",
        "search_results": [
            {"passage_id": "gb_q4_24_results", "text": "GlobalBank reported Q4 2024 net income of $7.1 billion."},
            {"passage_id": "gb_dividend_history", "text": "GlobalBank has paid a quarterly dividend for more than 25 consecutive years."},
            {"passage_id": "gb_buyback", "text": "Under its long-running share repurchase program, GlobalBank repurchased $4.8 billion of common stock in 2024."},
        ],
        "gold_answer": "GlobalBank's 2024 IPO is not supported by the passages; the service should abstain.",
        "alt_ans": [
            "GlobalBank has been publicly traded for decades; no 2024 IPO occurred. The passages discuss Q4 results, a 25-year dividend history, and a long-running share buyback program — all inconsistent with a 2024 IPO.",
        ],
        "gold_supports": [],
        "is_false_premise": True,
    },
    {
        "id": "out_of_scope_insufficient_evidence",
        "category": "out_of_scope",
        "question": "What was Quantum Trading's CEO's 2024 total compensation?",
        "search_results": [
            {"passage_id": "qt1", "text": "Quantum Trading discussed expansion into two new European cities."},
            {"passage_id": "qt2", "text": "Trading volumes grew 12% year over year, according to a Q3 update."},
        ],
        "gold_answer": "The passages do not state Quantum Trading's CEO compensation; the service should abstain.",
        "alt_ans": [
            "The provided passages discuss expansion and trading volumes, but do not state the CEO's 2024 total compensation; abstention is appropriate.",
        ],
        "gold_supports": [],
        "is_false_premise": True,
    },
    {
        "id": "idempotency_same_request_twice",
        "category": "idempotency",
        "question": "Did BlueArc raise full-year guidance?",
        "search_results": [
            {"passage_id": "p1", "text": "BlueArc raised its full-year revenue guidance to a range of $4.4-4.5 billion, from $4.2-4.3 billion previously."},
        ],
        "gold_answer": "Yes — BlueArc raised its full-year revenue guidance to a range of $4.4-4.5 billion, up from $4.2-4.3 billion previously.",
        "alt_ans": [
            "BlueArc raised full-year revenue guidance to $4.4-4.5 billion from $4.2-4.3 billion.",
            "Yes, guidance was raised to $4.4-4.5 billion.",
        ],
        "gold_supports": ["p1"],
    },
    {
        "id": "adversarial_distractor_injection",
        "category": "adversarial",
        "question": "What dividend did Harbor declare?",
        "search_results": [
            {"passage_id": "x1", "text": "Harbor discussed vessel utilization and the macro environment in its earnings call."},
            {"passage_id": "s1", "text": "Harbor declared a quarterly dividend of $0.42 per share, payable next month."},
            {"passage_id": "x2", "text": "Analysts asked about refinancing plans and drydock timing."},
        ],
        "gold_answer": "$0.42 per share",
        "alt_ans": ["a quarterly dividend of $0.42 per share", "$0.42"],
        "gold_supports": ["s1"],
    },
    {
        "id": "adversarial_reordered_passages",
        "category": "adversarial",
        "question": "What was Crestline's Q1 2024 revenue?",
        "search_results": [
            {"passage_id": "s3", "text": "Crestline expanded into one new market in Q1 2024."},
            {"passage_id": "s2", "text": "Crestline did not adjust its full-year guidance."},
            {"passage_id": "s1", "text": "Crestline reported Q1 2024 revenue of $2.1 billion, up 5% year over year."},
        ],
        "gold_answer": "$2.1 billion",
        "alt_ans": ["2.1 billion dollars", "$2.1B"],
        "gold_supports": ["s1"],
    },
    {
        "id": "adversarial_question_variation",
        "category": "adversarial",
        "question": "How much did Apex Manufacturing spend on capex in 2024?",
        "search_results": [
            {"passage_id": "p1", "text": "Apex Manufacturing reported capital expenditures of $920 million in 2024."},
        ],
        "gold_answer": "$920 million",
        "alt_ans": ["920 million dollars", "$920M"],
        "gold_supports": ["p1"],
    },
]


# ---------------- assembled scenario YAML ----------------


def _common_schema_match(step_id: str) -> dict:
    return {
        "kind": "schema_match",
        "target": f"{step_id}.body",
        "must_have_fields": ["answer", "citations", "abstained"],
    }


def _scenario_to_yaml_payload(s: dict) -> dict:
    """Render one scenario row into the runtime scenario YAML shape.

    Rubric set depends on the category:
      - happy_path / boundary / idempotency / adversarial: schema_match
        + llm_judge_semantic_eq + oracle_set_overlap (when gold_supports
        non-empty).
      - malformed_input: numeric_range on response.status only.
      - out_of_scope (false_premise + insufficient_evidence): schema_match
        + literal_match(abstained == True) + llm_judge_false_premise.
    """
    sid = s["id"]
    step_id = f"call_{sid}"
    category = s["category"]
    rubrics: list[dict] = []
    quality_bar_ids: list[str] = []

    # Build the request body.
    if s.get("expect_status_4xx"):
        # Malformed input: trace asserts 4xx.
        body: dict = {}
        if s["question"] is not None:
            body["question"] = s["question"]
        body["search_results"] = s["search_results"]
        trace_step = {
            "id": step_id,
            "method": "POST",
            "path": "/finance/answer",
            "body": body,
        }
        rubrics.append({
            "kind": "numeric_range",
            "target": f"{step_id}.status",
            "min_value": 400,
            "max_value": 499,
        })
        quality_bar_ids.append("finance_answer_schema_conformance")
        scenario = {
            "id": sid,
            "description": f"{sid}: malformed request must produce a 4xx response.",
            "category": category,
            "quality_bar_ids": quality_bar_ids,
            "trace": [trace_step],
            "rubrics": rubrics,
        }
        return scenario

    body = {"question": s["question"], "search_results": s["search_results"]}
    trace_step = {
        "id": step_id,
        "method": "POST",
        "path": "/finance/answer",
        "body": body,
    }
    rubrics.append(_common_schema_match(step_id))
    quality_bar_ids.append("finance_answer_schema_conformance")

    if s.get("is_false_premise"):
        rubrics.append({
            "kind": "literal_match",
            "target": f"{step_id}.body.abstained",
            "expected": True,
        })
        rubrics.append({
            "kind": "llm_judge_false_premise",
            "target": f"{step_id}.body.answer",
            # llm_judge_false_premise uses a MERGED context where
            # setup_data sits under the ``setup_data`` key — paths into
            # setup_data must carry the prefix.
            "expected_falsity_path": f"setup_data.gold_answers.{sid}.alt_ans",
        })
        quality_bar_ids.append("false_premise_abstention_precision")
    else:
        rubrics.append({
            "kind": "llm_judge_semantic_eq",
            "target": f"{step_id}.body.answer",
            # llm_judge_semantic_eq uses a MERGED context where
            # setup_data sits under the ``setup_data`` key — paths into
            # setup_data must carry the prefix.
            "gold_path": f"setup_data.gold_answers.{sid}.answer",
            "alt_path": f"setup_data.gold_answers.{sid}.alt_ans",
            "strictness": "lenient",
        })
        quality_bar_ids.append("finance_answer_faithfulness")
        if s.get("gold_supports"):
            rubrics.append({
                "kind": "oracle_set_overlap",
                "target": f"{step_id}.body.citations",
                "gold_set_path": f"gold_supports.{sid}",
                "min_recall": 0.5,
            })
            quality_bar_ids.append("citation_set_overlap")
        if category in ("idempotency", "adversarial"):
            quality_bar_ids.append("extractive_stub_resistance")

    scenario = {
        "id": sid,
        "description": f"{sid}: curated finance-QA scenario.",
        "category": category,
        "quality_bar_ids": quality_bar_ids,
        "trace": [trace_step],
        "rubrics": rubrics,
    }
    return scenario


def _setup_data_payloads():
    """Build the setup_data files: gold_answers + gold_supports + queries.

    ``gold_answers`` keys by scenario_id (matches rubric ``gold_path`` /
    ``expected_falsity_path``). ``gold_supports`` keys by scenario_id as
    well (matches ``gold_set_path``).
    """
    gold_answers: dict[str, dict] = {}
    gold_supports: dict[str, list[str]] = {}
    queries_index: dict[str, dict] = {}
    for s in SCENARIOS:
        sid = s["id"]
        if s.get("expect_status_4xx"):
            continue
        gold_answers[sid] = {
            "answer": s["gold_answer"],
            "alt_ans": s["alt_ans"],
            "answer_type": "false_premise" if s.get("is_false_premise") else "valid",
            "question_type": s["category"],
        }
        gold_supports[sid] = s["gold_supports"]
        queries_index[sid] = {
            "query": s["question"],
            "domain": "finance",
            "question_type": s["category"],
        }
    return queries_index, gold_answers, gold_supports


def main() -> int:
    if not COURSE_ROOT.exists():
        print(f"ERROR: course root not found: {COURSE_ROOT}")
        return 1

    # ---- setup_data ----
    SETUP_DIR.mkdir(parents=True, exist_ok=True)
    # Remove any stale JSONL form so the loader doesn't surface it as raw text.
    jsonl_path = SETUP_DIR / "queries.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()
    # The runtime keeps ``search_results_index`` around even though our
    # scenarios no longer reference it via interpolation — wipe it so a
    # stale UUID-keyed index from an earlier CRAG load can't confuse a
    # debugger.
    sri_path = SETUP_DIR / "search_results_index.json"
    if sri_path.exists():
        sri_path.unlink()
    queries_index, gold_answers, gold_supports = _setup_data_payloads()
    (SETUP_DIR / "queries.json").write_text(json.dumps(queries_index, indent=2))
    (SETUP_DIR / "gold_answers.json").write_text(json.dumps(gold_answers, indent=2))
    (SETUP_DIR / "gold_supports.json").write_text(json.dumps(gold_supports, indent=2))

    # ---- scenarios ----
    if SCENARIOS_DIR.exists():
        for child in SCENARIOS_DIR.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    for idx, s in enumerate(SCENARIOS, start=1):
        filename = f"{idx:02d}_{s['id']}.yaml"
        payload = _scenario_to_yaml_payload(s)
        (SCENARIOS_DIR / filename).write_text(
            yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
        )

    print(
        f"Curated assets written:\n"
        f"  setup_data : {SETUP_DIR}\n"
        f"     - queries.json ({len(queries_index)} entries)\n"
        f"     - gold_answers.json ({len(gold_answers)} entries)\n"
        f"     - gold_supports.json ({len(gold_supports)} entries)\n"
        f"  scenarios : {SCENARIOS_DIR} ({len(SCENARIOS)} files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
