from __future__ import annotations

import json
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
DRAFT_TIMELINE_TEMPLATE_PATH = APP_ROOT / "templates" / "draft_timeline.html"
DRAFT_TIMELINE_STATE_PLACEHOLDER = "__DRAFT_TIMELINE_STATE_JSON__"
DRAFT_TIMELINE_ASSET_VERSION_PLACEHOLDER = "__DRAFT_TIMELINE_ASSET_VERSION__"


def _asset_version(*paths: Path) -> str:
    latest = max(int(path.stat().st_mtime_ns) for path in paths)
    return str(latest)


def build_draft_timeline_state(*, draft_id: str | None = None) -> dict:
    return {
        "draft_id": draft_id,
        "timeline_url_template": "/v1/course-runs/{course_run_id}/timeline",
        "dashboard_url": "/create-course",
    }


def render_draft_timeline_page(state: dict) -> str:
    payload = json.dumps(state).replace("</", "<\\/")
    template = DRAFT_TIMELINE_TEMPLATE_PATH.read_text(encoding="utf-8")
    asset_version = _asset_version(
        APP_ROOT / "static" / "app-shell.css",
        APP_ROOT / "static" / "draft-timeline.css",
        APP_ROOT / "static" / "draft-timeline.js",
    )
    return (
        template
        .replace(DRAFT_TIMELINE_STATE_PLACEHOLDER, payload)
        .replace(DRAFT_TIMELINE_ASSET_VERSION_PLACEHOLDER, asset_version)
    )
