"""Tests for the ``start_node`` parameter on the LangGraph executor —
lets us resume a blocked workflow from a specific node (e.g.
``reviewer_code`` after a domain-grounding judge fix) without
re-running already-passed authoring nodes."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.services.langgraph_assignment_graph import LangGraphAssignmentGraph


# ---------------- argument plumbing ----------------


def test_execute_defaults_to_authoring_runtime() -> None:
    """No start_node → executor begins at ``authoring_runtime`` (current
    behavior must not regress)."""
    graph = LangGraphAssignmentGraph(
        sandbox_runner=MagicMock(),
        test_authoring_service=MagicMock(),
        baseline_verifier=MagicMock(),
    )
    starting_nodes: list[str] = []

    def fake_invoke(node_name, state):
        starting_nodes.append(node_name)
        # Mark this node's execution and short-circuit
        from app.domain.workflow import WorkflowNodeExecution, WorkflowNodeKind, WorkflowNodeStatus

        kind = graph._kind_for_node_name(node_name)
        from datetime import UTC, datetime
        from uuid import uuid4

        state["node_executions"].append(
            WorkflowNodeExecution(
                node_id=f"node_{uuid4().hex[:8]}",
                kind=kind,
                attempt=1,
                iteration=1,
                status=WorkflowNodeStatus.passed,
                summary="stub",
                findings=[],
                created_at=datetime.now(UTC),
            )
        )
        return state

    graph._invoke_node = fake_invoke  # type: ignore[assignment]
    graph._next_node = lambda node_name, state: None  # type: ignore[assignment]

    run = _stub_run_with_spec()
    graph.execute(run)
    assert starting_nodes[0] == "authoring_runtime"


def test_execute_with_start_node_jumps_to_that_node() -> None:
    """``start_node="reviewer_code"`` → executor begins at reviewer_code,
    skipping the authoring nodes that already passed in a prior run."""
    graph = LangGraphAssignmentGraph(
        sandbox_runner=MagicMock(),
        test_authoring_service=MagicMock(),
        baseline_verifier=MagicMock(),
    )
    starting_nodes: list[str] = []

    def fake_invoke(node_name, state):
        starting_nodes.append(node_name)
        from app.domain.workflow import WorkflowNodeExecution, WorkflowNodeStatus

        kind = graph._kind_for_node_name(node_name)
        from datetime import UTC, datetime
        from uuid import uuid4

        state["node_executions"].append(
            WorkflowNodeExecution(
                node_id=f"node_{uuid4().hex[:8]}",
                kind=kind,
                attempt=1,
                iteration=1,
                status=WorkflowNodeStatus.passed,
                summary="stub",
                findings=[],
                created_at=datetime.now(UTC),
            )
        )
        return state

    graph._invoke_node = fake_invoke  # type: ignore[assignment]
    graph._next_node = lambda node_name, state: None  # type: ignore[assignment]

    run = _stub_run_with_spec()
    graph.execute(run, start_node="reviewer_code")
    assert starting_nodes[0] == "reviewer_code"


def test_execute_rejects_unknown_start_node() -> None:
    graph = LangGraphAssignmentGraph(
        sandbox_runner=MagicMock(),
        test_authoring_service=MagicMock(),
        baseline_verifier=MagicMock(),
    )
    run = _stub_run_with_spec()
    with pytest.raises(ValueError) as ei:
        graph.execute(run, start_node="totally_made_up_node")
    assert "totally_made_up_node" in str(ei.value)


# ---------------- helpers ----------------


def _stub_run_with_spec():
    """Minimal WorkflowRun stub the executor needs to enter the loop.
    Real construction is heavy; we only need .artifacts.task_agent_spec
    non-None and a node_executions list."""
    from types import SimpleNamespace

    run = SimpleNamespace()
    run.id = "run_test"
    run.title = "stub"
    run.artifacts = SimpleNamespace(
        task_agent_spec=SimpleNamespace(),  # any non-None
        node_executions=[],
        workspace_snapshot=None,
    )
    # The executor calls run.model_copy(deep=True) in state setup;
    # patch a stub that returns self so the test doesn't need full pydantic.
    run.model_copy = lambda *, deep=False: run
    return run
