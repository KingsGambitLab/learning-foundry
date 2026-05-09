from __future__ import annotations

from app.domain.workflow import FailureContext, WorkflowNodeExecution, WorkflowRun
from app.services.learner_brief_builder import ensure_task_agent_deliverable_briefs


class TaskAgentRepairService:
    def apply(
        self,
        run: WorkflowRun,
        latest_node: WorkflowNodeExecution,
        failure_context: FailureContext | None = None,
    ) -> tuple[WorkflowRun, bool, str]:
        spec = run.artifacts.task_agent_spec
        if spec is None:
            return run, False, "No assignment spec is available to repair."

        changed = False
        notes: list[str] = []
        issue_codes = {issue.code for issue in (failure_context.validation_issues if failure_context is not None else [])}
        finding_text = " ".join(
            f"{finding.title} {finding.detail}"
            for finding in (failure_context.findings if failure_context is not None else latest_node.findings)
        ).lower()

        if (
            {
                "missing_deliverable_learning_outcomes",
                "missing_learner_brief",
                "missing_learner_starter_surface",
                "missing_public_checks",
            }
            & issue_codes
        ) or any(
            phrase in finding_text
            for phrase in (
                "learner brief",
                "learning outcome",
                "starter surface",
                "visible learner checks",
                "public checks",
            )
        ):
            ensure_task_agent_deliverable_briefs(spec, overwrite=True)
            changed = True
            notes.append("Rebuilt learner briefs, public checks, and derived deliverable guidance from the current spec.")

        if "missing_editable_files" in issue_codes and not spec.runtime_dependencies.editable_files:
            app_service = next(
                (service for service in spec.project_contract.runtime_plan.services if service.service_id == "app"),
                None,
            )
            spec.runtime_dependencies.editable_files = [
                app_service.entrypoint_path if app_service is not None and app_service.entrypoint_path else "app.py"
            ]
            changed = True
            notes.append("Restored the default editable file list for the learner workspace.")

        if (
            {"missing_visible_check_command", "public_checks_without_command"} & issue_codes
            and spec.runtime_dependencies.visible_check_command is None
        ):
            spec.runtime_dependencies.visible_check_command = "python checks/run_visible_checks.py"
            changed = True
            notes.append("Restored the learner-visible check command.")

        if not changed:
            return run, False, "No deterministic spec repairs were needed for the current failure packet."
        ensure_task_agent_deliverable_briefs(spec, overwrite=True)
        return run, True, " ".join(notes)
