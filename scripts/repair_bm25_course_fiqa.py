"""Second-pass repair: swap the BM25 course's grader bundle from
synthesized-corpus (CRAG queries + LLM-imagined passages) to a real
BeIR/fiqa-2018 retrieval task.

BeIR/fiqa ships:
- 57,638 real financial Stack Exchange passages (corpus)
- 6,648 real questions
- 1,706 test-split qrels (relevance judgments)

For each scenario we build a 10-passage pool:
- 1-3 gold passages (per qrels) — the only ones that actually answer the query
- 7-9 hard distractors selected by token overlap with the query (corpus
  passages that LOOK relevant lexically but aren't gold per qrels)

Token-overlap retrieval (V3) should now fail most happy_path scenarios
because the distractors look just as relevant lexically as the gold.
The learner has to do something better — TF-IDF, BM25 IDF weighting,
or dense embeddings — to discriminate.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset

COURSE_ID = "course_f918e889a33c"
WORKSPACE = REPO_ROOT / "workspaces" / "outcome" / COURSE_ID
SETUP_DIR = WORKSPACE / "private" / "grader" / "_setup"
SCENARIOS_DIR = WORKSPACE / "private" / "grader" / "scenarios"

POOL_SIZE = 10  # total passages per scenario retrieval pool


def _tokenize(text: str) -> set[str]:
    stop = {"a", "an", "and", "are", "as", "at", "be", "by", "did", "do",
            "does", "for", "from", "had", "has", "have", "how", "in", "is",
            "it", "its", "of", "on", "or", "that", "the", "their", "there",
            "to", "was", "were", "what", "when", "where", "which", "who",
            "why", "will", "with", "i", "my", "you", "your"}
    return {tok for tok in re.findall(r"[a-zA-Z0-9]+", text.lower())
            if tok not in stop and len(tok) > 2}


def main() -> None:
    print("Loading BeIR/fiqa corpus + queries + qrels...")
    corpus_ds = load_dataset("BeIR/fiqa", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/fiqa", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/fiqa-qrels", split="test")

    # Index queries and qrels.
    queries = {str(r["_id"]): r["text"] for r in queries_ds}
    qrels: dict[str, list[str]] = defaultdict(list)
    for r in qrels_ds:
        if int(r["score"]) > 0:
            qrels[str(r["query-id"])].append(str(r["corpus-id"]))

    # Index corpus as a list with id mapping. We'll do a token-inverted
    # index over the full corpus so we can find topical distractors fast.
    print(f"Indexing {len(corpus_ds)} corpus passages by token...")
    corpus_by_id: dict[str, dict] = {}
    inverted: dict[str, set[str]] = defaultdict(set)
    for row in corpus_ds:
        cid = str(row["_id"])
        text = (row.get("title", "") + " " + row.get("text", "")).strip()
        corpus_by_id[cid] = {"_id": cid, "title": row.get("title", ""), "text": row.get("text", "")}
        for tok in _tokenize(text):
            inverted[tok].add(cid)
    print(f"inverted index size: {len(inverted)} unique tokens")

    # Pick 18 test queries that have qrels and a reasonable number of
    # corpus passages with token overlap (so we can build a 10-passage pool).
    print("Selecting 18 queries with usable qrels + hard-distractor candidates...")
    eligible_qids: list[str] = []
    for qid, gold_ids in qrels.items():
        if qid not in queries:
            continue
        if not gold_ids:
            continue
        if any(g not in corpus_by_id for g in gold_ids):
            continue
        eligible_qids.append(qid)
        if len(eligible_qids) >= 200:  # take a generous pool, narrow below
            break

    # For each candidate, score by how many topical distractors are available.
    selected: list[tuple[str, list[str], list[str]]] = []  # (qid, gold_ids, distractor_ids)
    for qid in eligible_qids:
        gold_ids = qrels[qid][:3]  # cap golds at 3 to keep recall test tractable
        q_tokens = _tokenize(queries[qid])
        if not q_tokens:
            continue
        # Score corpus passages by token overlap; exclude golds.
        scored: dict[str, int] = defaultdict(int)
        for tok in q_tokens:
            for cid in inverted.get(tok, ()):
                if cid in gold_ids:
                    continue
                scored[cid] += 1
        if not scored:
            continue
        top_distractors = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
        distractor_ids = [cid for cid, _ in top_distractors[: POOL_SIZE - len(gold_ids)]]
        if len(distractor_ids) < POOL_SIZE - len(gold_ids):
            continue
        selected.append((qid, gold_ids, distractor_ids))
        if len(selected) == 20:
            break

    print(f"selected {len(selected)} queries with gold + ≥{POOL_SIZE} pool size")

    # --- Write _setup files ---
    queries_jsonl_lines: list[str] = []
    gold_answers: dict[str, dict] = {}
    search_results_index: dict[str, list[dict]] = {}

    for qid, gold_ids, distractor_ids in selected:
        query_text = queries[qid]
        # Pool: golds first, then distractors. The trace runner sees the
        # full pool — order doesn't matter to the learner's retriever.
        pool_passages = []
        for cid in gold_ids + distractor_ids:
            p = corpus_by_id[cid]
            pool_passages.append({
                "passage_id": p["_id"],
                "text": p["text"],
                "title": p["title"],
                "source": "beir/fiqa",
            })
        # The "answer" for fiqa qrels isn't a short span — qrels just
        # marks which passages are relevant. We use the gold passage's
        # text snippet as the reference answer for the LLM judge.
        # Fiqa Q&As are full-paragraph answers; we take the first
        # sentence of the highest-scoring gold passage as the reference.
        gold_first = corpus_by_id[gold_ids[0]]["text"]
        first_sent = re.split(r"(?<=[.!?])\s", gold_first, maxsplit=1)[0]
        gold_answer_text = first_sent.strip()[:300]

        full_row = {
            "query_id": qid,
            "query": query_text,
            "answer": gold_answer_text,
            "alt_ans": [],
            "search_results": pool_passages,
            "domain": "finance",
            "question_type": "natural",
            "answer_type": "valid",
        }
        queries_jsonl_lines.append(json.dumps(full_row, sort_keys=True))
        gold_answers[qid] = {
            "query": query_text,
            "answer": gold_answer_text,
            "alt_ans": [],
            "answer_type": "valid",
            "question_type": "natural",
            "gold_passages": gold_ids,
        }
        search_results_index[qid] = pool_passages

    # Wipe + write canonical files.
    for stale in ("queries.json", "gold_supports.json", "queries.jsonl",
                  "gold_answers.json", "search_results_index.json"):
        p = SETUP_DIR / stale
        if p.exists():
            p.unlink()

    (SETUP_DIR / "queries.jsonl").write_text("\n".join(queries_jsonl_lines) + "\n")
    (SETUP_DIR / "gold_answers.json").write_text(
        json.dumps(gold_answers, indent=2, sort_keys=True)
    )
    (SETUP_DIR / "search_results_index.json").write_text(
        json.dumps(search_results_index, indent=2, sort_keys=True)
    )
    print(f"wrote canonical _setup files referencing {len(selected)} real BeIR/fiqa queries")

    # --- Rewrite scenarios to reference real fiqa qids ---
    qids = list(gold_answers.keys())
    assert len(qids) >= 12

    # Wipe existing scenarios.
    for old in SCENARIOS_DIR.glob("*.yaml"):
        old.unlink()

    def scenario(filename: str, content: str) -> None:
        (SCENARIOS_DIR / filename).write_text(content)

    # 5 happy_path
    for i, qid in enumerate(qids[:5], start=1):
        sid = f"happy_valid_q{i}"
        scenario(f"{i:02d}_{sid}.yaml", "\n".join([
            f"id: {sid}",
            f"description: '{sid}: real BeIR/fiqa query, retrieval pool is 1-3 gold + 7-9 topical distractors from the actual corpus.'",
            "category: happy_path",
            "quality_bar_ids:",
            "- finance_answer_schema_conformance",
            "- finance_answer_faithfulness",
            "- citation_set_overlap",
            "trace:",
            f"- id: call_{sid}",
            "  method: POST",
            "  path: /finance/answer",
            "  body:",
            f"    question: \"${{setup_data.gold_answers.{qid}.query}}\"",
            f"    search_results: \"${{setup_data.search_results_index.{qid}}}\"",
            f"  capture: call_{sid}",
            "rubrics:",
            "- kind: schema_match",
            f"  target: call_{sid}.body",
            "  must_have_fields: [\"answer\", \"citations\", \"abstained\"]",
            "- kind: llm_judge_semantic_eq",
            f"  target: call_{sid}.body.answer",
            f"  gold_path: setup_data.gold_answers.{qid}.answer",
            "  strictness: lenient",
            "- kind: oracle_set_overlap",
            f"  target: call_{sid}.body.citations",
            f"  gold_set_path: setup_data.gold_answers.{qid}.gold_passages",
            "  min_recall: 0.5",
        ]) + "\n")

    # 3 boundary
    boundary_names = ["single_passage", "two_passages", "conflicting_passages"]
    for i, qid in enumerate(qids[5:8], start=1):
        sid = f"boundary_{boundary_names[i-1]}"
        scenario(f"{5+i:02d}_{sid}.yaml", "\n".join([
            f"id: {sid}",
            f"description: '{sid}: edge case on real fiqa retrieval pool.'",
            "category: boundary",
            "quality_bar_ids:",
            "- finance_answer_schema_conformance",
            "- finance_answer_faithfulness",
            "trace:",
            f"- id: call_{sid}",
            "  method: POST",
            "  path: /finance/answer",
            "  body:",
            f"    question: \"${{setup_data.gold_answers.{qid}.query}}\"",
            f"    search_results: \"${{setup_data.search_results_index.{qid}}}\"",
            f"  capture: call_{sid}",
            "rubrics:",
            "- kind: schema_match",
            f"  target: call_{sid}.body",
            "  must_have_fields: [\"answer\", \"citations\", \"abstained\"]",
            "- kind: llm_judge_semantic_eq",
            f"  target: call_{sid}.body.answer",
            f"  gold_path: setup_data.gold_answers.{qid}.answer",
            "  strictness: lenient",
        ]) + "\n")

    # 3 malformed_input (unchanged — framework checks)
    malformed = [
        ("malformed_missing_question", '{"question": "", "search_results": []}'),
        ("malformed_empty_search_results", '{"question": "What is a 401k?", "search_results": []}'),
        ("malformed_bad_passage_shape", '{"question": "What is a 401k?", "search_results": [{"passage_id": "", "text": "x"}]}'),
    ]
    for i, (sid, body_json) in enumerate(malformed, start=1):
        scenario(f"{8+i:02d}_{sid}.yaml", "\n".join([
            f"id: {sid}",
            f"description: '{sid}: framework validation should reject malformed input with 4xx.'",
            "category: malformed_input",
            "quality_bar_ids:",
            "- finance_answer_schema_conformance",
            "trace:",
            f"- id: call_{sid}",
            "  method: POST",
            "  path: /finance/answer",
            f"  body: {body_json}",
            f"  capture: call_{sid}",
            "rubrics:",
            "- kind: numeric_range",
            f"  target: call_{sid}.status",
            "  min_value: 400",
            "  max_value: 499",
        ]) + "\n")

    # 3 out_of_scope: drop gold passages from pool; learner must abstain
    for i, qid in enumerate(qids[8:11], start=1):
        sid = f"out_of_scope_q_fp_{i:04d}" if i <= 2 else "out_of_scope_insufficient_evidence"
        pool = search_results_index[qid]
        gold_passage_ids = set(gold_answers[qid]["gold_passages"])
        distractors_only = [p for p in pool if p["passage_id"] not in gold_passage_ids]
        scenario(f"{11+i:02d}_{sid}.yaml", "\n".join([
            f"id: {sid}",
            f"description: '{sid}: gold passages removed; service must abstain.'",
            "category: out_of_scope",
            "quality_bar_ids:",
            "- finance_answer_schema_conformance",
            "- false_premise_abstention_precision",
            "trace:",
            f"- id: call_{sid}",
            "  method: POST",
            "  path: /finance/answer",
            "  body:",
            f"    question: {json.dumps(queries[qid])}",
            f"    search_results: {json.dumps(distractors_only)}",
            f"  capture: call_{sid}",
            "rubrics:",
            "- kind: schema_match",
            f"  target: call_{sid}.body",
            "  must_have_fields: [\"answer\", \"citations\", \"abstained\"]",
            "- kind: literal_match",
            f"  target: call_{sid}.body.abstained",
            "  expected: true",
        ]) + "\n")

    # 1 idempotency
    qid = qids[11]
    scenario("15_idempotency_same_request_twice.yaml", "\n".join([
        "id: idempotency_same_request_twice",
        "description: 'idempotency_same_request_twice: two identical requests produce identical answers.'",
        "category: idempotency",
        "quality_bar_ids:",
        "- finance_answer_schema_conformance",
        "trace:",
        "- id: first",
        "  method: POST",
        "  path: /finance/answer",
        "  body:",
        f"    question: \"${{setup_data.gold_answers.{qid}.query}}\"",
        f"    search_results: \"${{setup_data.search_results_index.{qid}}}\"",
        "  capture: first",
        "- id: second",
        "  method: POST",
        "  path: /finance/answer",
        "  body:",
        f"    question: \"${{setup_data.gold_answers.{qid}.query}}\"",
        f"    search_results: \"${{setup_data.search_results_index.{qid}}}\"",
        "  capture: second",
        "rubrics:",
        "- kind: behavioral_equivalence",
        "  target: first.body.answer",
        "  expected_path: second.body.answer",
    ]) + "\n")

    # 3 adversarial: reorder, distractor injection, question paraphrase
    adv_names = ["distractor_injection", "reordered_passages", "question_variation"]
    for i, qid in enumerate(qids[12:15], start=1):
        sid = f"adversarial_{adv_names[i-1]}"
        pool = search_results_index[qid]
        if adv_names[i-1] == "reordered_passages":
            adv_pool = list(reversed(pool))
        elif adv_names[i-1] == "distractor_injection":
            # Add 5 extra random corpus passages from the inverted index
            # (passages that share at least one token with the query).
            q_tokens = _tokenize(queries[qid])
            extra_ids: set[str] = set()
            for tok in q_tokens:
                for cid in list(inverted.get(tok, ()))[:5]:
                    if cid not in [p["passage_id"] for p in pool]:
                        extra_ids.add(cid)
                        if len(extra_ids) >= 5:
                            break
                if len(extra_ids) >= 5:
                    break
            extra = [{"passage_id": cid, "text": corpus_by_id[cid]["text"],
                      "title": corpus_by_id[cid]["title"], "source": "beir/fiqa-injected"}
                     for cid in extra_ids]
            adv_pool = pool + extra
        else:
            adv_pool = pool
        scenario(f"{15+i:02d}_{sid}.yaml", "\n".join([
            f"id: {sid}",
            f"description: '{sid}: real fiqa query under adversarial pool manipulation.'",
            "category: adversarial",
            "quality_bar_ids:",
            "- finance_answer_schema_conformance",
            "- finance_answer_faithfulness",
            "trace:",
            f"- id: call_{sid}",
            "  method: POST",
            "  path: /finance/answer",
            "  body:",
            f"    question: {json.dumps(queries[qid])}",
            f"    search_results: {json.dumps(adv_pool)}",
            f"  capture: call_{sid}",
            "rubrics:",
            "- kind: schema_match",
            f"  target: call_{sid}.body",
            "  must_have_fields: [\"answer\", \"citations\", \"abstained\"]",
            "- kind: llm_judge_semantic_eq",
            f"  target: call_{sid}.body.answer",
            f"  gold_path: setup_data.gold_answers.{qid}.answer",
            "  strictness: lenient",
        ]) + "\n")

    print(f"wrote 18 scenarios referencing real BeIR/fiqa queries with hard topical distractors")


if __name__ == "__main__":
    main()
