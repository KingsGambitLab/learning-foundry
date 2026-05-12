from __future__ import annotations

import unittest

from app.services.failure_context_builder import _excerpt


class FailureContextExcerptTests(unittest.TestCase):
    def test_excerpt_keeps_8kb_tail_by_default(self) -> None:
        """Real Go/Maven/Buildkit failures spill 5-20KB of stderr. The error
        sits in the tail (last ~200-500 chars). Defaulting to 8KB keeps the
        full diagnostic in the prompt instead of clipping mid-stack-trace.
        """
        # 20KB of filler followed by the real error at the very end.
        filler = "\n".join(f"go: downloading dep-{i}" for i in range(2000))
        canonical = (
            "go: github.com/rogpeppe/go-internal@v1.14.1 requires go >= 1.23 "
            "(running go 1.22.4)"
        )
        text = filler + "\n" + canonical
        self.assertGreater(len(text), 20_000)

        excerpt = _excerpt(text)

        self.assertIsNotNone(excerpt)
        # Stripped + "..." prefix. The body should be ~8000 chars.
        # Tolerate the "..." sentinel.
        self.assertLessEqual(len(excerpt), 8000 + 3)
        self.assertGreaterEqual(len(excerpt), 8000 - 3)
        # The canonical diagnostic at the END of input must survive.
        self.assertIn("requires go >= 1.23", excerpt)
        # And the very-old filler at the start must be dropped.
        self.assertNotIn("dep-0\n", excerpt)
        self.assertNotIn("dep-100\n", excerpt)

    def test_excerpt_returns_text_unchanged_when_under_budget(self) -> None:
        short = "go: requires go >= 1.23\n"
        self.assertEqual(_excerpt(short), short.strip())

    def test_excerpt_returns_none_for_empty_input(self) -> None:
        self.assertIsNone(_excerpt(""))
        self.assertIsNone(_excerpt("   \n\n"))


if __name__ == "__main__":
    unittest.main()
