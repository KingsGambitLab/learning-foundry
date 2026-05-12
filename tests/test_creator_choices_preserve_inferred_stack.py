"""Pin: `_apply_creator_choices_to_design_spec` must NOT overwrite
inferred stack values with None when the creator didn't supply them.

Observed today on course_632fbd0012ac (Rails 8 brief): the OpenAI
planner's `_normalize_raw_plan` ran `infer_assignment_design` which
correctly resolved `implementation_language="ruby",
application_framework="rails"` from the brief. Then
`CourseGenerationService._normalize_plan` ran
`_apply_creator_choices_to_design_spec(shared_design_spec,
creator_choices)`. `creator_choices` was an EMPTY
`CreatorCourseSetupChoices` (the API request didn't include
`creator_setup`), so every field was `None`. The function
unconditionally overrode the inferred values with those `None`s,
yielding `implementation_language=None`,
`application_framework=None` on the persisted spec — even though the
workspace was correctly authored as Rails.

Root cause: `_apply_creator_choices_to_design_spec` uses
`design_spec.runtime_dependencies.model_copy(update={
"implementation_language": creator_choices.implementation_language,
... })` — an unconditional override. When the creator didn't pick a
stack, the override is None, destroying the inferred value.

Fix: prefer the creator's choice if set, else preserve the inferred
spec value. `creator_choices.implementation_language or
design_spec.runtime_dependencies.implementation_language`.
"""

from __future__ import annotations

import unittest

from app.domain.course import CreatorCourseSetupChoices
from app.domain.registry import StarterType
from app.services.assignment_design_inference import infer_assignment_design
from app.services.course_generation_service import CourseGenerationService


class CreatorChoicesPreserveInferredStackTests(unittest.TestCase):
    def test_apply_empty_creator_choices_preserves_inferred_stack(self) -> None:
        """When creator_choices has all-None fields (the default when
        the API request omits creator_setup), the inferred stack values
        MUST survive unchanged.
        """
        inference = infer_assignment_design(
            title="Production Team Incident Response in Rails 8",
            problem_statement=(
                "Build a production-ready team incident response application "
                "using Rails 8 with PostgreSQL and Solid Queue."
            ),
        )
        design_spec = inference.design_spec
        self.assertIsNotNone(design_spec)
        self.assertEqual(design_spec.runtime_dependencies.implementation_language, "ruby")
        self.assertEqual(design_spec.runtime_dependencies.application_framework, "rails")

        empty_choices = CreatorCourseSetupChoices(
            starter_type=StarterType.partial,
            implementation_language=None,
            language_version=None,
            application_framework=None,
            framework_version=None,
            package_manager=None,
            primary_database=None,
            primary_database_version=None,
            cache_backend=None,
            cache_backend_version=None,
        )

        service = CourseGenerationService.__new__(CourseGenerationService)
        adjusted = service._apply_creator_choices_to_design_spec(
            design_spec, empty_choices
        )

        self.assertIsNotNone(adjusted)
        self.assertEqual(
            adjusted.runtime_dependencies.implementation_language,
            "ruby",
            "Empty creator_choices must preserve the inferred "
            "implementation_language='ruby' — not overwrite with None.",
        )
        self.assertEqual(
            adjusted.runtime_dependencies.application_framework,
            "rails",
            "Empty creator_choices must preserve the inferred "
            "application_framework='rails' — not overwrite with None.",
        )

    def test_apply_non_empty_creator_choices_overrides_inferred_stack(self) -> None:
        """When the creator explicitly picks a stack, that choice
        SHOULD win over the inferred one. (Preserves the original
        intent of the override.)
        """
        inference = infer_assignment_design(
            title="A Service",
            problem_statement="Build a service using Rails.",
        )
        design_spec = inference.design_spec
        self.assertEqual(design_spec.runtime_dependencies.implementation_language, "ruby")

        explicit_choices = CreatorCourseSetupChoices(
            starter_type=StarterType.partial,
            implementation_language="python",
            application_framework="fastapi",
        )

        service = CourseGenerationService.__new__(CourseGenerationService)
        adjusted = service._apply_creator_choices_to_design_spec(
            design_spec, explicit_choices
        )
        self.assertEqual(
            adjusted.runtime_dependencies.implementation_language,
            "python",
            "Explicit creator_choices.implementation_language must win.",
        )
        self.assertEqual(
            adjusted.runtime_dependencies.application_framework,
            "fastapi",
        )


if __name__ == "__main__":
    unittest.main()
