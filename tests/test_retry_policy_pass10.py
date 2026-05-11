"""Pass 10 Job B: one global retry policy of 5 attempts for everyone.

Replaces the prior shared-codebase-vs-non-shared branching. The graph
exposes a single integer ``max_authoring_attempts`` (default 5) and no
spec-based branching helpers.
"""

from __future__ import annotations

import unittest

from app.services.docker_sandbox_runner import DockerSandboxRunner
from app.services.langgraph_assignment_graph import LangGraphAssignmentGraph


class RetryPolicyPass10Tests(unittest.TestCase):
    def test_constructor_default_max_authoring_attempts_is_5(self) -> None:
        graph = LangGraphAssignmentGraph(DockerSandboxRunner())
        self.assertEqual(graph.max_authoring_attempts, 5)

    def test_no_module_level_max_authoring_attempts_function(self) -> None:
        import app.services.langgraph_assignment_graph as module

        self.assertFalse(
            hasattr(module, "max_authoring_attempts")
            and callable(getattr(module, "max_authoring_attempts"))
            and not isinstance(getattr(module, "max_authoring_attempts"), int),
            msg="Pass 10 Job B removes the module-level max_authoring_attempts(spec) helper.",
        )

    def test_no_spec_branching_helpers_on_graph(self) -> None:
        graph = LangGraphAssignmentGraph(DockerSandboxRunner())
        self.assertFalse(
            hasattr(graph, "_max_authoring_attempts_for_spec"),
            msg="Pass 10 Job B removes the spec-branching helper from the graph.",
        )
        self.assertFalse(
            hasattr(graph, "_state_max_authoring_attempts"),
            msg="Pass 10 Job B removes the state-level spec-branching helper from the graph.",
        )

    def test_policy_reports_constructor_max_authoring_attempts(self) -> None:
        graph = LangGraphAssignmentGraph(
            DockerSandboxRunner(),
            max_authoring_attempts=5,
        )
        policy = graph.policy()
        self.assertEqual(policy.max_authoring_attempts, 5)


if __name__ == "__main__":
    unittest.main()
