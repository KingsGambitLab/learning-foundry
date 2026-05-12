"""Pass-4 Job C: dashboard form + JS must use the collapsed `starter_type`
enum (`empty | partial`). Legacy `bare_stub`, `partial_implementation`,
`working_buggy`, and `working_suboptimal` strings would be rejected by the
API after Pass 1, so neither the HTML form nor the JS may emit them.
"""

from __future__ import annotations

import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_HTML = REPO_ROOT / "app" / "templates" / "dashboard.html"
DASHBOARD_JS = REPO_ROOT / "app" / "static" / "dashboard.js"


class _StarterTypeRadioParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.radios: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "input":
            return
        attrs_dict = {key: (value or "") for key, value in attrs}
        if attrs_dict.get("type") != "radio":
            return
        if attrs_dict.get("name") != "starter_type":
            return
        self.radios.append(attrs_dict)


class DashboardStarterTypeFormTests(unittest.TestCase):
    def test_dashboard_html_has_two_radios_with_collapsed_enum_values(self) -> None:
        html = DASHBOARD_HTML.read_text(encoding="utf-8")
        parser = _StarterTypeRadioParser()
        parser.feed(html)
        values = [radio.get("value") for radio in parser.radios]
        self.assertEqual(
            sorted(values),
            ["empty", "partial"],
            "Dashboard form must expose exactly two `starter_type` radio buttons "
            "with values `empty` and `partial` (matching the collapsed enum).",
        )
        # Exactly one default-checked radio, and it must be `partial`.
        checked = [
            radio.get("value")
            for radio in parser.radios
            if "checked" in radio
        ]
        self.assertEqual(
            checked,
            ["partial"],
            "Default-checked starter_type radio must be `partial`.",
        )

    def test_dashboard_html_does_not_reference_legacy_starter_type_values(self) -> None:
        html = DASHBOARD_HTML.read_text(encoding="utf-8")
        for legacy in ("bare_stub", "partial_implementation", "working_buggy", "working_suboptimal"):
            self.assertNotIn(
                legacy,
                html,
                f"Dashboard HTML must not reference the retired starter_type value `{legacy}`.",
            )

    def test_dashboard_js_does_not_reference_legacy_starter_type_values(self) -> None:
        js = DASHBOARD_JS.read_text(encoding="utf-8")
        for legacy in ("bare_stub", "partial_implementation", "working_buggy", "working_suboptimal"):
            self.assertNotIn(
                legacy,
                js,
                f"Dashboard JS must not reference the retired starter_type value `{legacy}`.",
            )

    def test_dashboard_js_friendly_label_map_uses_collapsed_enum(self) -> None:
        """The friendly label map in dashboard.js must use only the two new
        starter_type keys."""
        js = DASHBOARD_JS.read_text(encoding="utf-8")
        # Find the friendlyStarterType labels block.
        match = re.search(
            r"function friendlyStarterType[^}]*?const labels = \{([^}]*)\}",
            js,
            re.DOTALL,
        )
        self.assertIsNotNone(
            match,
            "Could not locate `friendlyStarterType` label map in dashboard.js.",
        )
        labels_block = match.group(1)
        self.assertIn("empty:", labels_block)
        self.assertIn("partial:", labels_block)


if __name__ == "__main__":
    unittest.main()
