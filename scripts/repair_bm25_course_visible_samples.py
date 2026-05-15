"""Populate ``public/examples/sample_queries.json`` with real BeIR/fiqa
search_results so a learner can develop and test retrieval locally
WITHOUT having to download the dataset themselves.

We pick 5 fiqa queries that AREN'T used in the hidden 18 scenarios
(so the learner can't just copy outputs) and ship them with their
full retrieval pool (1-3 gold + 7-9 distractors). This gives the
learner ~50 real passages to develop against.

The visible queries' gold answers are also shipped so the learner can
verify their retrieval/extraction is finding the right passage and
extracting the right span before submitting.
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
SAMPLES_PATH = WORKSPACE / "public" / "examples" / "sample_queries.json"


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

    queries = {str(r["_id"]): r["text"] for r in queries_ds}
    qrels: dict[str, list[str]] = defaultdict(list)
    for r in qrels_ds:
        if int(r["score"]) > 0:
            qrels[str(r["query-id"])].append(str(r["corpus-id"]))

    # Read hidden grader's qids so we don't overlap.
    hidden_qids: set[str] = set()
    hidden_setup_path = WORKSPACE / "private" / "grader" / "_setup" / "gold_answers.json"
    if hidden_setup_path.exists():
        hidden_setup = json.loads(hidden_setup_path.read_text())
        hidden_qids = set(hidden_setup.keys())
        print(f"hidden grader uses {len(hidden_qids)} qids — visible samples will avoid them")

    corpus_by_id: dict[str, dict] = {}
    inverted: dict[str, set[str]] = defaultdict(set)
    print("Indexing corpus...")
    for row in corpus_ds:
        cid = str(row["_id"])
        text = (row.get("title", "") + " " + row.get("text", "")).strip()
        corpus_by_id[cid] = {"_id": cid, "title": row.get("title", ""), "text": row.get("text", "")}
        for tok in _tokenize(text):
            inverted[tok].add(cid)

    # Pick 5 visible queries: have qrels, not used in hidden scenarios.
    visible: list[dict] = []
    for qid, gold_ids in qrels.items():
        if qid in hidden_qids:
            continue
        if qid not in queries or not gold_ids:
            continue
        if any(g not in corpus_by_id for g in gold_ids):
            continue
        q_tokens = _tokenize(queries[qid])
        if not q_tokens:
            continue
        scored: dict[str, int] = defaultdict(int)
        for tok in q_tokens:
            for cid in inverted.get(tok, ()):
                if cid in gold_ids:
                    continue
                scored[cid] += 1
        if not scored:
            continue
        gold_capped = gold_ids[:2]
        top_distractors = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
        distractor_ids = [cid for cid, _ in top_distractors[: 10 - len(gold_capped)]]
        if len(distractor_ids) < 10 - len(gold_capped):
            continue

        pool = []
        for cid in gold_capped + distractor_ids:
            p = corpus_by_id[cid]
            pool.append({
                "passage_id": p["_id"],
                "text": p["text"],
                "title": p["title"],
                "source": "beir/fiqa",
            })
        # Pick the first gold passage's first sentence as the expected
        # answer (matches the hidden grader's convention).
        gold_text = corpus_by_id[gold_capped[0]]["text"]
        first_sent = re.split(r"(?<=[.!?])\s", gold_text, maxsplit=1)[0]
        visible.append({
            "query_id": qid,
            "question": queries[qid],
            "expected_answer": first_sent.strip()[:280],
            "gold_passage_ids": gold_capped,
            "search_results": pool,
            "domain": "finance",
        })
        if len(visible) == 5:
            break

    print(f"selected {len(visible)} visible sample queries with real corpus pools")
    SAMPLES_PATH.write_text(json.dumps(visible, indent=2))
    print(f"wrote {SAMPLES_PATH}")

    # Sync to active enrollment workspaces.
    for enrollment in (REPO_ROOT / "learner_workspaces").glob("enrollment_*"):
        target = enrollment / "workspace" / "public" / "examples" / "sample_queries.json"
        if target.parent.exists():
            target.write_text(json.dumps(visible, indent=2))
            print(f"synced -> {target}")


if __name__ == "__main__":
    main()
