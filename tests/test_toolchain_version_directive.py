"""Pin the language-agnostic toolchain-version-mismatch directive.

Every modern stack reports the same pattern when the runtime version
shipped by the Dockerfile base image is older than what a transitive
dependency requires:

  - Go:    `requires go >= 1.23 (running go 1.22.4)`
  - Python: `requires-python >= 3.12` vs `python:3.11-slim` base image
  - Node:  `engines.node >= 20` vs `FROM node:18`
  - Java:  `requires JDK 21` vs `FROM eclipse-temurin:17-jre`
  - Rust:  `requires rustc >= 1.78` vs `FROM rust:1.75`

In every case the canonical fix is the SAME: bump the Dockerfile `FROM`
line (and any manifest version constraint) to the version the
dependency requires. Pinning transitive deps to lower versions is the
wrong move — toolchain requirements are non-negotiable upstream
signals.

Without this directive, the model rewrites manifests in circles. The
Go validation run repeatedly hit `requires go >= 1.23 (running go
1.22.4)` and never bumped `FROM golang:1.22.4-bookworm` across 5
retries.

These tests pin: BOTH the progressive shared-repo prompt AND the
per-deliverable repair prompt include a language-agnostic toolchain
version-mismatch directive.
"""

from __future__ import annotations

import inspect
import unittest


class ToolchainVersionDirectiveTests(unittest.TestCase):
    def test_repo_authoring_prompt_includes_toolchain_version_mismatch_directive(self) -> None:
        """The shared/progressive repo authoring prompt must teach the
        model to fix toolchain-version mismatches by bumping the
        Dockerfile base image, not by pinning transitive deps lower.
        """
        from app.services.openai_repo_authoring import OpenAIStarterRepoAuthoringService

        source = inspect.getsource(OpenAIStarterRepoAuthoringService)

        # Directive must use language-agnostic phrasing.
        self.assertIn(
            "toolchain version mismatch",
            source.lower(),
            "Repo authoring prompts must include a language-agnostic "
            "toolchain version-mismatch directive.",
        )
        # The fix must point at the Dockerfile base image, not transitive pins.
        self.assertIn(
            "Dockerfile",
            source,
            "Toolchain directive must reference the Dockerfile.",
        )
        # The directive must explicitly forbid pinning transitive deps lower.
        self.assertTrue(
            "do not pin transitive" in source.lower()
            or "do not attempt to pin" in source.lower()
            or "do not downgrade" in source.lower(),
            "Toolchain directive must forbid pinning transitive deps to "
            "lower versions as an escape valve.",
        )

    def test_toolchain_directive_appears_in_both_progressive_and_repair_prompts(self) -> None:
        """Both the progressive shared-repo authoring path AND the
        per-deliverable repair path see toolchain-version failures.
        Both prompts must carry the directive — otherwise repair loops
        re-introduce the mismatch the progressive path fixed.
        """
        from app.services import openai_repo_authoring

        source = inspect.getsource(openai_repo_authoring)
        # Heuristic: count how many times the directive appears. It must
        # show up at LEAST twice (one per prompt site).
        occurrences = source.lower().count("toolchain version mismatch")
        self.assertGreaterEqual(
            occurrences,
            2,
            f"Toolchain version-mismatch directive must appear in both "
            f"prompt sites (progressive + repair); found {occurrences} "
            f"occurrence(s).",
        )


class LockfileIntegrityDirectiveTests(unittest.TestCase):
    """Language-agnostic lockfile-hash-mismatch directive.

    Every modern ecosystem with a lockfile reports the same failure
    pattern when the lockfile's content hashes don't match the upstream
    artifacts:

      - Go:     `checksum mismatch` / `bits may have been replaced`
      - npm:    `EINTEGRITY: sha512-... integrity checksum failed`
      - pnpm:   `ERR_PNPM_LOCKFILE_BREAKING_CHANGE` / lockfile integrity
      - Yarn:   `Hash mismatch detected for ...`
      - Cargo:  `the lockfile ... is corrupt`
      - Poetry: `Lock file is not compatible`
      - pip:    `THESE PACKAGES DO NOT MATCH THE HASHES`

    The canonical fix is identical: regenerate the lockfile from the
    manifest inside `install.sh` using the language's refresh command.
    Hashes can NEVER be authored by hand — they come from the registry.

    Observed in `course_d540fbc15802` (Go URL shortener final run):
    retries 6-12 ALL failed with `bits may have been replaced` errors
    because the model kept hand-authoring stale `go.sum` entries.
    """

    def test_repo_authoring_prompt_includes_lockfile_integrity_directive(self) -> None:
        from app.services.openai_repo_authoring import OpenAIStarterRepoAuthoringService

        source = inspect.getsource(OpenAIStarterRepoAuthoringService)
        lowered = source.lower()

        self.assertIn(
            "lockfile integrity",
            lowered,
            "Repo authoring prompts must include a language-agnostic "
            "lockfile-integrity (hash mismatch) directive.",
        )
        # The directive must forbid hand-authoring hashes.
        self.assertTrue(
            "do not author" in lowered
            or "cannot be authored" in lowered
            or "never be authored" in lowered
            or "must come from" in lowered,
            "Lockfile directive must forbid hand-authoring lockfile "
            "content hashes — they must come from the registry.",
        )
        # The directive must point at install.sh / a regeneration command.
        self.assertIn(
            "regenerate",
            lowered,
            "Lockfile directive must direct the model to REGENERATE the "
            "lockfile (not hand-edit it).",
        )

    def test_lockfile_integrity_directive_appears_in_both_progressive_and_repair_prompts(self) -> None:
        from app.services import openai_repo_authoring

        source = inspect.getsource(openai_repo_authoring)
        occurrences = source.lower().count("lockfile integrity")
        self.assertGreaterEqual(
            occurrences,
            2,
            f"Lockfile-integrity directive must appear in both prompt sites "
            f"(progressive + repair); found {occurrences} occurrence(s).",
        )


if __name__ == "__main__":
    unittest.main()
