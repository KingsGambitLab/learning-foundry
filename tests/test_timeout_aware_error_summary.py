"""Pin: when `_wait_for_http` times out and the container is still
running, the stage-failure summary must surface the timeout headline —
not the last 3 lines of stderr, which for slow installs are just
progress messages that misleadingly look like the failure.

Observed today on course_632fbd0012ac (Rails 8 brief): Rails `bundle
install` is heavy (~50 gems, native extensions). The harness's
`_wait_for_http` polls /health with a 90s deadline. While bundler is
still working at second 91, the harness gives up. `_wait_for_http`
correctly raises `LearnerStudioError("Timed out waiting for '...'
during 'install'. Last error: ...")`.

`_summarize_stage_failure` then receives:
  error_text = "Timed out waiting for 'http://127.0.0.1:54123/health' during 'install'. Last error: ..."
  logs      = container stderr ("Installing rails-html-sanitizer 1.7.0 | Fetching rdoc 7.2.0 | Installing rdoc 7.2.0")

Today's code prefers `logs` over `error_text` at the tail-line teaser
step. The timeout message gets thrown away, and the headline reads:

  "deliverable_1 failed during install:
   Installing rails-html-sanitizer 1.7.0 | Fetching rdoc 7.2.0 | Installing rdoc 7.2.0"

This misleads the reader (operator OR repair LLM): the gem-install
lines look like the failure, but they're just where install happened
to be when the timeout fired. The actual failure is "harness gave up
too early on a healthy container."

This test pins the fix: a "Timed out waiting for" line in `error_text`
must be surfaced as the headline teaser — same shape as the existing
"Last HTTP response:" handling for boot-stage HTTP 5xx failures.
"""

from __future__ import annotations

import unittest

from app.services.docker_sandbox_runner import DockerSandboxRunner, SandboxFailureStage


class TimeoutAwareErrorSummaryTests(unittest.TestCase):
    def test_install_timeout_surfaces_timeout_headline_not_stderr_tail(self) -> None:
        """The Rails case: install stage, error_text says 'Timed out
        waiting', logs are gem-install progress.
        """
        runner = DockerSandboxRunner()
        summary = runner._summarize_stage_failure(
            deliverable_id="deliverable_1",
            failed_stage=SandboxFailureStage.install,
            error_text=(
                "Timed out waiting for 'http://127.0.0.1:54123/health' during 'install'. "
                "Last error: None"
            ),
            logs=(
                "Installing rails-html-sanitizer 1.7.0\n"
                "Fetching rdoc 7.2.0\n"
                "Installing rdoc 7.2.0\n"
            ),
            default="install failed",
        )
        self.assertIn("Timed out waiting", summary,
                      f"Install-stage summary must surface the timeout headline; "
                      f"got: {summary!r}")
        self.assertIn("install", summary)
        # The bundler progress lines must NOT be the headline teaser.
        self.assertNotIn(
            "Installing rdoc",
            summary,
            f"Install-stage summary must NOT show the last bundler progress "
            f"line as the failure — that's misleading. Got: {summary!r}",
        )

    def test_verify_timeout_surfaces_timeout_headline(self) -> None:
        runner = DockerSandboxRunner()
        summary = runner._summarize_stage_failure(
            deliverable_id="deliverable_2",
            failed_stage=SandboxFailureStage.verify,
            error_text=(
                "Timed out waiting for 'http://127.0.0.1:60001/health' during 'verify'. "
                "Last error: [Errno 61] Connection refused"
            ),
            logs="some verify-stage output that's not the actual failure",
            default="verify failed",
        )
        self.assertIn("Timed out waiting", summary)
        self.assertIn("verify", summary)

    def test_boot_timeout_keeps_existing_http_response_priority(self) -> None:
        """Regression guard: when error_text has BOTH 'Last HTTP response'
        and 'Timed out waiting', the HTTP response stays as headline
        (matches the partial-starter 501 case we fixed earlier today).
        """
        runner = DockerSandboxRunner()
        summary = runner._summarize_stage_failure(
            deliverable_id="deliverable_1",
            failed_stage=SandboxFailureStage.boot,
            error_text=(
                "Timed out waiting for 'http://127.0.0.1:54123/health' to respond. "
                "Last error: None Last HTTP response: 501 Not Implemented"
            ),
            logs="INFO:     Application startup complete.",
            default="boot failed",
        )
        # The Last HTTP response wins for boot-stage when present.
        self.assertIn("501", summary)
        self.assertIn("Not Implemented", summary)

    def test_no_timeout_no_http_falls_back_to_stderr_tail(self) -> None:
        """When neither 'Timed out waiting' nor 'Last HTTP response'
        is in the error_text, the existing stderr-tail behavior is
        preserved (e.g., toolchain mismatch failures).
        """
        runner = DockerSandboxRunner()
        summary = runner._summarize_stage_failure(
            deliverable_id="deliverable_1",
            failed_stage=SandboxFailureStage.install,
            error_text="Container stopped during 'install' with exit code 1",
            logs=(
                "go: downloading github.com/jackc/pgx v5.6.0\n"
                "go: github.com/rogpeppe/go-internal@v1.14.1 requires go >= 1.23 (running go 1.22.4)\n"
            ),
            default="install failed",
        )
        # Toolchain-mismatch case: surface the stderr tail.
        self.assertIn("requires go >= 1.23", summary)


if __name__ == "__main__":
    unittest.main()
