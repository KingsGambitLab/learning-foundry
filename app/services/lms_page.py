from __future__ import annotations

import json
from pathlib import Path

from app.domain.learner import LearnerEnrollmentList, PublishedCourseCatalog

APP_ROOT = Path(__file__).resolve().parents[1]
LMS_TEMPLATE_PATH = APP_ROOT / "templates" / "lms_home.html"
LMS_COURSES_TEMPLATE_PATH = APP_ROOT / "templates" / "lms_courses.html"
LMS_STATE_PLACEHOLDER = "__LMS_STATE_JSON__"
LMS_ASSET_VERSION_PLACEHOLDER = "__LMS_ASSET_VERSION__"


def _asset_version(*paths: Path) -> str:
    latest = max(int(path.stat().st_mtime_ns) for path in paths)
    return str(latest)


def build_lms_state(*, catalog: PublishedCourseCatalog, enrollments: LearnerEnrollmentList) -> dict:
    return {
        "catalog": catalog.model_dump(mode="json"),
        "enrollments": enrollments.model_dump(mode="json"),
        "catalog_url": "/v1/lms/catalog",
        "enrollments_url": "/v1/lms/enrollments",
        "create_course_url": "/create-course",
    }


def render_lms_home(state: dict) -> str:
    payload = json.dumps(state).replace("</", "<\\/")
    template = LMS_TEMPLATE_PATH.read_text(encoding="utf-8")
    asset_version = _asset_version(
        APP_ROOT / "static" / "app-shell.css",
        APP_ROOT / "static" / "lms.css",
        APP_ROOT / "static" / "lms.js",
    )
    return (
        template
        .replace(LMS_STATE_PLACEHOLDER, payload)
        .replace(LMS_ASSET_VERSION_PLACEHOLDER, asset_version)
    )


def render_lms_courses_page(state: dict) -> str:
    payload = json.dumps(state).replace("</", "<\\/")
    template = LMS_COURSES_TEMPLATE_PATH.read_text(encoding="utf-8")
    asset_version = _asset_version(
        APP_ROOT / "static" / "app-shell.css",
        APP_ROOT / "static" / "lms.css",
        APP_ROOT / "static" / "lms-courses.js",
    )
    return (
        template
        .replace(LMS_STATE_PLACEHOLDER, payload)
        .replace(LMS_ASSET_VERSION_PLACEHOLDER, asset_version)
    )
