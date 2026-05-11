from __future__ import annotations

import unittest

from app.domain.sandbox import SandboxFailureStage
from app.services.docker_sandbox_runner import DockerSandboxRunner


class DockerSandboxRunnerTests(unittest.TestCase):
    def test_boot_failure_summary_prefers_useful_container_log_line(self) -> None:
        runner = DockerSandboxRunner()

        summary = runner._summarize_stage_failure(
            deliverable_id="deliverable_1",
            failed_stage=SandboxFailureStage.boot,
            error_text=(
                "Timed out waiting for 'http://127.0.0.1:18001/health' during 'boot'. "
                "Last error: [Errno 61] Connection refused"
            ),
            logs=(
                "2026-05-11T10:00:00Z ERROR com.zaxxer.hikari.pool.HikariPool: "
                "HikariPool-1 - Exception during pool initialization.\n"
                "org.postgresql.util.PSQLException: Connection to postgres:5432 refused"
            ),
            default="boot failed",
        )

        self.assertEqual(
            summary,
            (
                "deliverable_1 failed during boot: "
                "2026-05-11T10:00:00Z ERROR com.zaxxer.hikari.pool.HikariPool: "
                "HikariPool-1 - Exception during pool initialization."
            ),
        )


if __name__ == "__main__":
    unittest.main()
