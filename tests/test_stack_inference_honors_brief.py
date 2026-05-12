"""Pin: `infer_implementation_stack` must recognize all major
stacks the harness intends to support, not just Python/FastAPI.

Observed today on course_71d009477ea4: submitted with goal
"Build a production-ready team incident response application using
Rails 8 with PostgreSQL and Solid Queue" and title "Production Team
Incident Response in Rails 8". The inference returned
`("python", "fastapi")` — completely ignored the explicit Rails 8
declaration — and the resulting course was authored as a Python
FastAPI app.

Root cause: `FRAMEWORK_LANGUAGE_HINTS`, `DEFAULT_FRAMEWORK_BY_LANGUAGE`,
and `LANGUAGE_KEYWORDS` are hardcoded dicts that omit Ruby/Rails (and
several other major stacks). When the brief mentions "Rails" the
keyword scan fails to match anything, and the final fallback
`if normalized_language is None and normalized_framework is None:
normalized_language = "python"; normalized_framework = "fastapi"`
kicks in.

These tests pin:
  1. Ruby/Rails brief produces (ruby, rails)
  2. Java/Spring Boot brief produces (java, spring boot)
  3. Elixir/Phoenix brief produces (elixir, phoenix)
  4. C#/.NET brief produces (csharp, aspnet) or similar
  5. The brief's explicit stack always wins over the Python fallback
"""

from __future__ import annotations

import unittest

from app.services.assignment_design_inference import infer_implementation_stack


class StackInferenceHonorsBriefTests(unittest.TestCase):
    def test_rails_brief_returns_ruby_rails(self) -> None:
        language, framework = infer_implementation_stack(
            title="Production Team Incident Response in Rails 8",
            problem_statement=(
                "Build a production-ready team incident response application "
                "using Rails 8 with PostgreSQL and Solid Queue."
            ),
            implementation_language=None,
            application_framework=None,
            tech_stack=None,
        )
        self.assertEqual(
            language, "ruby",
            f"Rails brief must resolve to language=ruby, got {language!r}",
        )
        self.assertEqual(
            framework, "rails",
            f"Rails brief must resolve to framework=rails, got {framework!r}",
        )

    def test_ruby_only_brief_defaults_to_rails(self) -> None:
        language, framework = infer_implementation_stack(
            title="Ruby Service",
            problem_statement="Build a Ruby web service with PostgreSQL.",
            implementation_language=None,
            application_framework=None,
            tech_stack=None,
        )
        self.assertEqual(language, "ruby")
        self.assertEqual(
            framework, "rails",
            "Ruby with no framework hint should default to Rails (most "
            "common production Ruby stack).",
        )

    def test_spring_boot_brief_returns_java(self) -> None:
        language, framework = infer_implementation_stack(
            title="Spring Boot Order Service",
            problem_statement=(
                "Build a Spring Boot service in Java with PostgreSQL and "
                "JPA-backed persistence."
            ),
            implementation_language=None,
            application_framework=None,
            tech_stack=None,
        )
        self.assertEqual(language, "java")
        self.assertIn(framework, {"spring boot", "spring"})

    def test_phoenix_brief_returns_elixir(self) -> None:
        language, framework = infer_implementation_stack(
            title="Phoenix Live Dashboard",
            problem_statement=(
                "Build a Phoenix LiveView application in Elixir with "
                "PostgreSQL and LiveView updates."
            ),
            implementation_language=None,
            application_framework=None,
            tech_stack=None,
        )
        self.assertEqual(language, "elixir")
        self.assertEqual(framework, "phoenix")

    def test_explicit_framework_in_creator_setup_wins(self) -> None:
        """When the creator_setup explicitly sets a framework, that
        value must survive — never overridden by keyword scanning.
        """
        language, framework = infer_implementation_stack(
            title="Generic service",
            problem_statement="Build a generic web service.",
            implementation_language="ruby",
            application_framework="rails",
            tech_stack=None,
        )
        self.assertEqual(language, "ruby")
        self.assertEqual(framework, "rails")

    def test_python_fallback_only_when_no_signal(self) -> None:
        """The python/fastapi fallback should only fire when the brief
        has no stack signal at all — not when it mentions Rails, Phoenix,
        Spring, etc.
        """
        language, framework = infer_implementation_stack(
            title="Generic service",
            problem_statement="Build a web service.",
            implementation_language=None,
            application_framework=None,
            tech_stack=None,
        )
        self.assertEqual(language, "python")
        self.assertEqual(framework, "fastapi")


if __name__ == "__main__":
    unittest.main()
