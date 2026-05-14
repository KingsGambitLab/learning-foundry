import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.learner_studio_service import (
    _hash_learner_studio_inputs,
    _learner_studio_image_inputs,
    default_learner_studio_image,
)


class LearnerStudioImageTagTest(unittest.TestCase):
    def test_tag_has_hash_suffix(self) -> None:
        tag = default_learner_studio_image()
        self.assertTrue(tag.startswith("course-gen-learner-studio:"))
        suffix = tag.split(":", 1)[1]
        self.assertEqual(len(suffix), 12)
        # All hex
        int(suffix, 16)

    def test_inputs_include_dockerfile_and_extension_sources(self) -> None:
        inputs = _learner_studio_image_inputs()
        rel_names = {p.name for p in inputs}
        self.assertIn("learner-studio.Dockerfile", rel_names)
        self.assertIn("package.json", rel_names)
        self.assertIn("extension.ts", rel_names)

    def test_inputs_exclude_node_modules_and_dist(self) -> None:
        inputs = _learner_studio_image_inputs()
        for p in inputs:
            parts = set(p.parts)
            self.assertNotIn("node_modules", parts)
            self.assertNotIn("dist", parts)
            self.assertNotIn("out", parts)
            self.assertNotIn("test-out", parts)
            self.assertNotEqual(p.suffix, ".vsix")

    def test_hash_changes_when_a_tracked_file_changes(self) -> None:
        # Temporarily mock `_learner_studio_image_inputs` to point at two known files
        # so we can mutate one and confirm the hash output changes.
        with tempfile.TemporaryDirectory() as td:
            f1 = Path(td) / "a.txt"
            f1.write_text("hello")
            f2 = Path(td) / "b.txt"
            f2.write_text("world")
            with patch(
                "app.services.learner_studio_service._learner_studio_image_inputs",
                return_value=[f1, f2],
            ):
                hash_before = _hash_learner_studio_inputs()
                f1.write_text("hello!")
                hash_after = _hash_learner_studio_inputs()
            self.assertNotEqual(hash_before, hash_after)


if __name__ == "__main__":
    unittest.main()
