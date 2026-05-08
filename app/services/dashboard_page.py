from __future__ import annotations

import json
from pathlib import Path

from app.domain.course import CourseGenerationStatus

APP_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_TEMPLATE_PATH = APP_ROOT / "templates" / "dashboard.html"
DASHBOARD_STATE_PLACEHOLDER = "__DASHBOARD_STATE_JSON__"
DASHBOARD_ASSET_VERSION_PLACEHOLDER = "__DASHBOARD_ASSET_VERSION__"


def _asset_version(*paths: Path) -> str:
    latest = max(int(path.stat().st_mtime_ns) for path in paths)
    return str(latest)


def build_dashboard_state(*, generation_status: CourseGenerationStatus) -> dict:
    return {
        "generation_status": generation_status.model_dump(mode="json"),
        "docs_url": "/docs",
        "generate_url": "/v1/course-runs/generate-async",
        "create_revision_url_template": "/v1/course-runs/{course_run_id}/create-revision-async",
        "materialize_url_template": "/v1/course-runs/{course_run_id}/materialize-async",
        "publish_url_template": "/v1/course-runs/{course_run_id}/publish-async",
        "suggest_outcomes_url": "/v1/course-generation/suggest-outcomes",
        "creator_assets_url": "/v1/creator-assets",
        "status_url": "/v1/course-generation/status",
        "reset_local_url": "/v1/course-runs/reset-local",
    }


def render_author_dashboard(state: dict) -> str:
    payload = json.dumps(state).replace("</", "<\\/")
    template = DASHBOARD_TEMPLATE_PATH.read_text(encoding="utf-8")
    asset_version = _asset_version(
        APP_ROOT / "static" / "app-shell.css",
        APP_ROOT / "static" / "dashboard.css",
        APP_ROOT / "static" / "dashboard.js",
    )
    return (
        template
        .replace(DASHBOARD_STATE_PLACEHOLDER, payload)
        .replace(DASHBOARD_ASSET_VERSION_PLACEHOLDER, asset_version)
    )
