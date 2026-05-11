"""Pin the boot-stage diagnostic surface for HTTP healthcheck timeouts.

When `_wait_for_http` times out polling a healthcheck URL, the harness
must surface the LAST observed HTTP response (status + body) — not just
"timed out, last error: None". Without this, a stack like Python+Uvicorn
that responds 501 Not Implemented on every /health poll leaves the
boot-stage summary showing only the Uvicorn startup banner ("Application
startup complete. ..."), burying the real diagnostic (501) deep inside
stdout_tail.

These tests pin two pieces of the fix:
  1. `_wait_for_http` includes the last non-2xx response status (and a
     short body excerpt) in the `LearnerStudioError` raised on timeout.
  2. `_summarize_stage_failure(failed_stage=boot)` surfaces that
     "Last HTTP response: ..." line as the headline teaser when present,
     instead of taking the last 3 stderr lines (which for Uvicorn is the
     success banner).
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.services.docker_sandbox_runner import DockerSandboxRunner, SandboxFailureStage
from app.services.learner_studio_service import LearnerStudioError, LearnerStudioService


class FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class WaitForHttpLastResponseTests(unittest.TestCase):
    def test_timeout_error_includes_last_5xx_status_and_body(self) -> None:
        """A 501 Not Implemented on every /health poll must surface in
        the timeout error: status code AND a short body excerpt.
        """
        service = LearnerStudioService()
        service.start_timeout_s = 0.05  # fast timeout for the test

        fake_response = FakeResponse(
            status_code=501,
            text='{"detail":"Not Implemented"}',
        )

        with (
            patch(
                "app.services.learner_studio_service.httpx.get",
                return_value=fake_response,
            ),
            patch.object(service, "_container_running", return_value=True),
            patch.object(service, "_container_logs", return_value=""),
        ):
            with self.assertRaises(LearnerStudioError) as ctx:
                service._wait_for_http(
                    "http://127.0.0.1:18001/health",
                    container_name="test_container",
                )

        message = str(ctx.exception)
        self.assertIn("501", message,
                      f"Timeout error must include the last 5xx status code; got: {message!r}")
        self.assertIn("Last HTTP response", message,
                      f"Timeout error must use the 'Last HTTP response' marker; got: {message!r}")
        # Body excerpt should be present so the model sees the actual response payload.
        self.assertIn("Not Implemented", message,
                      f"Timeout error must include the response body excerpt; got: {message!r}")

    def test_timeout_error_truncates_long_response_body(self) -> None:
        """Body excerpt must be capped so the headline stays readable."""
        service = LearnerStudioService()
        service.start_timeout_s = 0.05

        long_body = "X" * 1000
        fake_response = FakeResponse(status_code=503, text=long_body)

        with (
            patch(
                "app.services.learner_studio_service.httpx.get",
                return_value=fake_response,
            ),
            patch.object(service, "_container_running", return_value=True),
            patch.object(service, "_container_logs", return_value=""),
        ):
            with self.assertRaises(LearnerStudioError) as ctx:
                service._wait_for_http(
                    "http://127.0.0.1:18001/health",
                    container_name="test_container",
                )

        message = str(ctx.exception)
        # 1000-char body must be truncated; full body would bloat the headline.
        self.assertLess(
            len(message),
            500,
            f"Truncated error message should stay short; got {len(message)} chars",
        )
        self.assertIn("503", message)


class BootStageHttpResponseHeadlineTests(unittest.TestCase):
    def test_boot_failure_summary_surfaces_last_http_response_when_present(self) -> None:
        """When `_wait_for_http` raised a timeout error containing
        "Last HTTP response: 501 ...", the boot-stage headline must
        surface THAT line — not the Uvicorn startup banner from stderr.
        """
        runner = DockerSandboxRunner()

        error_text = (
            "Timed out waiting for 'http://127.0.0.1:18001/health' during 'boot'. "
            "Last HTTP response: 501 {\"detail\":\"Not Implemented\"}. "
            "Last error: None"
        )
        # Uvicorn's stderr tail is the success banner — it does NOT signal
        # the real failure. Older summarizers would pick this as the
        # teaser, hiding the 501.
        uvicorn_stderr = "\n".join([
            "INFO:     Started server process [1]",
            "INFO:     Waiting for application startup.",
            "INFO:     Application startup complete.",
            "INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)",
        ])

        summary = runner._summarize_stage_failure(
            deliverable_id="deliverable_1",
            failed_stage=SandboxFailureStage.boot,
            error_text=error_text,
            logs=uvicorn_stderr,
            default="boot failed",
        )

        self.assertIn("deliverable_1 failed during boot", summary)
        self.assertIn("501", summary,
                      f"Boot summary must surface the 501 status; got: {summary!r}")
        self.assertNotIn(
            "Uvicorn running",
            summary,
            f"Boot summary must NOT collapse to the Uvicorn success banner when "
            f"a Last HTTP response is available; got: {summary!r}",
        )

    def test_boot_failure_summary_falls_back_to_stderr_when_no_http_response(self) -> None:
        """If no HTTP response was recorded (e.g., connection refused
        every poll), the boot summary keeps the existing stderr-tail
        behavior so we don't regress the existing diagnostic path.
        """
        runner = DockerSandboxRunner()

        error_text = (
            "Timed out waiting for 'http://127.0.0.1:18001/health' during 'boot'. "
            "Last error: [Errno 61] Connection refused"
        )
        stderr = (
            "2026-05-11T10:00:00Z ERROR HikariPool-1 - Exception during pool init.\n"
            "org.postgresql.util.PSQLException: Connection to postgres:5432 refused"
        )

        summary = runner._summarize_stage_failure(
            deliverable_id="deliverable_1",
            failed_stage=SandboxFailureStage.boot,
            error_text=error_text,
            logs=stderr,
            default="boot failed",
        )

        self.assertIn("deliverable_1 failed during boot", summary)
        self.assertTrue(
            "PSQLException" in summary or "HikariPool" in summary,
            f"Boot summary must keep stderr-tail behavior when no HTTP "
            f"response is recorded; got: {summary!r}",
        )


if __name__ == "__main__":
    unittest.main()
