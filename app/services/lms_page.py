from __future__ import annotations

import json
from pathlib import Path

from app.domain.learner import LearnerEnrollmentList, PublishedCourseCatalog

APP_ROOT = Path(__file__).resolve().parents[1]
LMS_TEMPLATE_PATH = APP_ROOT / "templates" / "lms_home.html"
LMS_STATE_PLACEHOLDER = "__LMS_STATE_JSON__"


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
    return template.replace(LMS_STATE_PLACEHOLDER, payload)
