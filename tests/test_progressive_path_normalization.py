"""Pin defensive normalization of file paths returned by the progressive
shared-repo authoring prompt.

The prompt tells the model the shared repo root is named `starter`
(`shared_repo_root: "starter"`). For some stacks — observed clearly in
Go where the project convention is `module starter` with files under
the module root — the model prefixes every authored path with `starter/`
on the assumption that those paths are relative to the WORKSPACE root,
not the shared starter root.

The application writer expects paths to be relative to the STARTER
ROOT (`public/starter/`), so a model-returned path `starter/cmd/server/
main.go` gets written to `public/starter/starter/cmd/server/main.go` —
one directory too deep. The reviewer then correctly reports the files
as missing at the expected location, triggering a regeneration loop
that re-introduces the same error.

Observed in `course_9c165891055e` (Go URL shortener):
  - public/starter/Dockerfile           ✓ (no prefix)
  - public/starter/.coursegen/runtime/  ✓ (no prefix)
  - public/starter/starter/cmd/...      ✗ (model added extra `starter/`)
  - public/starter/starter/internal/... ✗ (same)

This test pins: `_normalize_relative_path` strips a leading `starter/`
prefix defensively, so the model's misinterpretation no longer blocks
the run.
"""

from __future__ import annotations

import unittest

from app.services.openai_repo_authoring import OpenAIStarterRepoAuthoringService


class ProgressivePathNormalizationTests(unittest.TestCase):
    def _service(self) -> OpenAIStarterRepoAuthoringService:
        # `_normalize_relative_path` is a pure helper — no client wiring
        # needed. The bare instance with __new__ avoids dragging the
        # full config machinery in.
        return OpenAIStarterRepoAuthoringService.__new__(
            OpenAIStarterRepoAuthoringService
        )

    def test_strips_leading_starter_prefix(self) -> None:
        service = self._service()
        self.assertEqual(
            service._normalize_relative_path("starter/cmd/server/main.go"),
            "cmd/server/main.go",
            "A leading `starter/` prefix must be stripped — it's the shared "
            "repo root and paths should be relative to it, not duplicate it.",
        )

    def test_strips_leading_starter_prefix_for_go_module_layout(self) -> None:
        """Specific to the Go-module-name-shadows-repo-root quirk."""
        service = self._service()
        self.assertEqual(
            service._normalize_relative_path("starter/internal/http/router.go"),
            "internal/http/router.go",
        )
        self.assertEqual(
            service._normalize_relative_path("starter/go.mod"),
            "go.mod",
        )

    def test_preserves_paths_without_the_prefix(self) -> None:
        service = self._service()
        # These are the canonical, correctly-formed paths.
        self.assertEqual(
            service._normalize_relative_path("Dockerfile"),
            "Dockerfile",
        )
        self.assertEqual(
            service._normalize_relative_path("cmd/server/main.go"),
            "cmd/server/main.go",
        )
        self.assertEqual(
            service._normalize_relative_path(".coursegen/runtime/install.sh"),
            ".coursegen/runtime/install.sh",
        )

    def test_does_not_false_strip_starter_lookalike(self) -> None:
        """`starterly/foo.go` should NOT have `starter` stripped — the
        normalizer must only strip the literal `starter/` directory
        prefix, not any path that begins with the substring `starter`.
        """
        service = self._service()
        # Paths must end up matching `is_repo_contract_path`; pick a
        # path that survives normalization without the prefix-strip.
        # Two cases: prefix-only-without-slash, and substring-only.
        self.assertEqual(
            service._normalize_relative_path("starters_helper/foo.go"),
            "starters_helper/foo.go",
            "The normalizer must match `starter/` exactly (with the "
            "trailing slash), not the substring `starter`.",
        )

    def test_just_starter_alone_returns_none(self) -> None:
        """A path of literally `starter` (or `starter/`) is meaningless
        after the prefix strip — it leaves an empty string. Must
        return None rather than write to the directory itself.
        """
        service = self._service()
        self.assertIsNone(service._normalize_relative_path("starter"))
        self.assertIsNone(service._normalize_relative_path("starter/"))


if __name__ == "__main__":
    unittest.main()
