"""Oracle-comparison rubrics for the scenario-rubric library.

This module hosts rubrics that compare a learner-produced list against
a gold list loaded from ``setup_data``. The canonical use case is RAG
retrieval evaluation: "did the learner's top-k retrieval surface the
gold-labelled relevant documents for this query?".

The rubrics here follow the contract in
``app.services.scenario_rubrics_base``:
- Inputs come from a ``RubricContext`` (captures + setup_data).
- The outcome is always encoded in a ``Verdict``; pass/fail logic is
  never communicated by raising.
- Exceptions are reserved for genuine misconfiguration that the rubric
  framework cannot mask.
"""
from __future__ import annotations

from typing import Any

from app.services.scenario_rubrics_base import (
    Rubric,
    RubricContext,
    Verdict,
    register_rubric,
    resolve_path,
)


@register_rubric
class OracleSetOverlap(Rubric):
    """Recall of a learner-produced list against a gold set.

    Computes ``len(target ∩ gold) / len(gold)`` and compares it to
    ``min_recall``. Designed for RAG retrieval evaluation: the gold
    set is the doc_ids that *should* appear in the learner's top-k
    retrieval result for the current query, and ``target`` is the
    learner's retrieval response.

    Configuration:
      ``target``: dotted path inside ``ctx.captures``. May resolve to
        a list of plain values OR a list of dicts (in which case
        ``target_key`` extracts the comparison value from each).

      ``target_key``: when ``target`` is a list of dicts, the dict key
        whose value should be compared against the gold set.

      ``gold_set_path``: dotted path resolved from ``ctx.setup_data``
        (NOT ``captures``) to a list of gold values. The runner is
        expected to populate ``setup_data`` with whatever shape the
        path references. Paths may be nested
        (e.g. ``"gold_qa.q1.expected_doc_ids"``); the runner or scenario
        author is responsible for keying gold sets by scenario.

      ``min_recall``: minimum fraction of gold items that must appear
        in the target list. Default ``0.5``.

      ``top_k``: when set, only the first ``top_k`` entries of the
        target list are considered (recall@k). The slice is taken
        *before* ``target_key`` extraction, so it counts entries of
        the raw target list.

    Semantics:
      - PASS when ``recall >= min_recall`` and the gold set is non-empty.
      - FAIL with a recall diagnostic when ``recall < min_recall``.
      - ABSTAIN when the gold set is empty or the ``gold_set_path``
        cannot be resolved; these are configuration problems, not
        learner failures.
      - FAIL with ``{"missing_path": <target>}`` when the learner's
        response shape is missing the target list (KeyError /
        IndexError raised by ``resolve_path``).
    """

    name = "oracle_set_overlap"

    def __init__(
        self,
        *,
        target: str,
        gold_set_path: str,
        target_key: str | None = None,
        min_recall: float = 0.5,
        top_k: int | None = None,
    ) -> None:
        self.target = target
        self.target_key = target_key
        self.gold_set_path = gold_set_path
        self.min_recall = min_recall
        self.top_k = top_k

    def judge(self, ctx: RubricContext) -> Verdict:
        # Resolve the gold set. We accept BOTH path conventions:
        #   - bare path (``"gold_supports.q1"``) — walks ctx.setup_data
        #     directly (the original convention of this rubric).
        #   - prefixed path (``"setup_data.gold_supports.q1"``) — matches
        #     the merged-context convention used by ``llm_judge_*``
        #     rubrics. The LLM tends to emit the prefixed form, so we
        #     accept it here for cross-rubric consistency.
        # If the gold set is missing or empty, the rubric can't render a
        # verdict on the learner.
        gold_path = self.gold_set_path
        if gold_path.startswith("setup_data."):
            gold_path = gold_path[len("setup_data.") :]
        try:
            gold_raw = resolve_path(ctx.setup_data, gold_path)
        except (KeyError, IndexError):
            return Verdict(
                status="abstain",
                rationale=(
                    f"gold path not found in setup_data: {self.gold_set_path}"
                ),
            )

        gold_list = list(gold_raw) if gold_raw is not None else []
        if len(gold_list) == 0:
            return Verdict(
                status="abstain",
                rationale="gold set is empty; cannot evaluate recall",
            )

        # Resolve the learner's target list. A missing path means the
        # learner's response didn't include the expected shape — that
        # IS a learner failure, distinct from a config issue.
        try:
            target_raw = resolve_path(ctx.captures, self.target)
        except (KeyError, IndexError):
            return Verdict(
                status="fail",
                rationale=(
                    f"target path not present in captures: {self.target}"
                ),
                diagnostic={"missing_path": self.target},
            )

        # Validate target SHAPE before indexing. A bad learner payload
        # (string instead of list, dicts missing the configured key,
        # unhashable extracted values) must produce a FAIL verdict, not
        # a raised exception that crashes the scenario runner.
        # Validation order per element when ``target_key`` is set:
        #   1. element must be a dict
        #   2. dict must contain ``target_key``
        #   3. ``item[target_key]`` must be hashable (set membership)
        # The first check that fails wins; later checks are skipped for
        # that element.
        if not isinstance(target_raw, list):
            target_type = type(target_raw).__name__
            return Verdict(
                status="fail",
                rationale=(
                    f"target is not a list (learner returned {target_type})"
                ),
                diagnostic={
                    "target_path": self.target,
                    "target_type": target_type,
                },
            )

        target_list: list[Any] = list(target_raw)
        if self.top_k is not None:
            target_list = target_list[: self.top_k]

        invalid_reasons: list[dict[str, Any]] = []
        target_values: list[Any] = []
        if self.target_key is not None:
            for idx, item in enumerate(target_list):
                if not isinstance(item, dict):
                    invalid_reasons.append(
                        {
                            "index": idx,
                            "reason": "not_a_dict",
                            "element_type": type(item).__name__,
                        }
                    )
                    continue
                if self.target_key not in item:
                    invalid_reasons.append(
                        {
                            "index": idx,
                            "reason": "missing_target_key",
                            "target_key": self.target_key,
                        }
                    )
                    continue
                value = item[self.target_key]
                try:
                    hash(value)
                except TypeError:
                    invalid_reasons.append(
                        {
                            "index": idx,
                            "reason": "unhashable_value",
                            "value_type": type(value).__name__,
                        }
                    )
                    continue
                target_values.append(value)
        else:
            for idx, value in enumerate(target_list):
                try:
                    hash(value)
                except TypeError:
                    invalid_reasons.append(
                        {
                            "index": idx,
                            "reason": "unhashable_value",
                            "value_type": type(value).__name__,
                        }
                    )
                    continue
                target_values.append(value)

        invalid_count = len(invalid_reasons)
        # If every element was invalid we cannot compute a recall at
        # all — fail with the diagnostic. Otherwise proceed with the
        # valid subset and expose the invalid count.
        if target_list and not target_values:
            return Verdict(
                status="fail",
                rationale=(
                    f"all {invalid_count} target element(s) were invalid "
                    f"(no valid entries to compare against gold)"
                ),
                diagnostic={
                    "target_path": self.target,
                    "target_key": self.target_key,
                    "invalid_elements": invalid_count,
                    "invalid_reasons": invalid_reasons,
                },
            )

        target_set = set(target_values)
        gold_set = set(gold_list)

        matched = sorted(target_set & gold_set, key=lambda v: gold_list.index(v))
        missed = [v for v in gold_list if v not in target_set]

        recall = len(matched) / len(gold_list)
        diagnostic: dict[str, Any] = {
            "recall": recall,
            "min_recall": self.min_recall,
            "gold_size": len(gold_list),
            "matched": list(matched),
            "missed": list(missed),
        }
        if invalid_count:
            diagnostic["invalid_elements"] = invalid_count
            diagnostic["invalid_reasons"] = invalid_reasons

        if recall >= self.min_recall:
            return Verdict(
                status="pass",
                rationale=(
                    f"recall {recall:.2f} ≥ threshold {self.min_recall:.2f} "
                    f"({len(matched)}/{len(gold_list)} gold items found)"
                ),
                diagnostic=diagnostic,
            )
        return Verdict(
            status="fail",
            rationale=(
                f"recall {recall:.2f} < threshold {self.min_recall:.2f} "
                f"({len(matched)}/{len(gold_list)} gold items found)"
            ),
            diagnostic=diagnostic,
        )
