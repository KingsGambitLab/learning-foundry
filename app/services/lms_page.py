from __future__ import annotations

import html as _html
import json
from pathlib import Path

from app.domain.learner import LearnerEnrollmentList, PublishedCourseCatalog

APP_ROOT = Path(__file__).resolve().parents[1]
LMS_TEMPLATE_PATH = APP_ROOT / "templates" / "lms_home.html"
LMS_COURSES_TEMPLATE_PATH = APP_ROOT / "templates" / "lms_courses.html"
LMS_STATE_PLACEHOLDER = "__LMS_STATE_JSON__"
LMS_ASSET_VERSION_PLACEHOLDER = "__LMS_ASSET_VERSION__"
LMS_NAV_PLACEHOLDER = "__LMS_NAV__"

BRAND_NAME = "Scaler Labs"
# Scaler.com favicon — used as the brand mark in the topbar.
BRAND_ICON_URL = "https://www.scaler.com/favicon.ico"


def _asset_version(*paths: Path) -> str:
    latest = max(int(path.stat().st_mtime_ns) for path in paths)
    return str(latest)


def _render_nav(user: dict | None) -> str:
    """Role-aware top nav.

    - One primary tab: Labs (the merged enrolled + catalog list).
    - Course builder shows only for creators.
    - No "Learner LMS" / "API docs" links.
    - Right side: signed-in identity + Log out, or a Log in button.
    """
    role = (user or {}).get("role")
    links = ['<a class="topnav-link active" href="/courses">Labs</a>']
    if role == "creator":
        links.append('<a class="topnav-link" href="/create-course">Course builder</a>')
    if user:
        who = _html.escape(str(user.get("display_name") or user.get("email") or "Account"))
        links.append(
            f'<span class="topnav-user">{who}</span>'
            '<form method="post" action="/auth/logout" class="topnav-logout">'
            '<button type="submit" class="topnav-link topnav-logout-btn">Log out</button>'
            "</form>"
        )
    else:
        links.append('<a class="topnav-link" href="/login">Log in</a>')
    return (
        '<nav class="topnav" aria-label="Primary">' + "".join(links) + "</nav>"
    )


def build_lms_state(
    *,
    catalog: PublishedCourseCatalog,
    enrollments: LearnerEnrollmentList,
    user: dict | None = None,
) -> dict:
    return {
        "catalog": catalog.model_dump(mode="json"),
        "enrollments": enrollments.model_dump(mode="json"),
        "catalog_url": "/v1/lms/catalog",
        "enrollments_url": "/v1/lms/enrollments",
        "create_course_url": "/create-course",
        "user": user,
    }


def _render(template_path: Path, state: dict, *asset_paths: Path) -> str:
    payload = json.dumps(state).replace("</", "<\\/")
    template = template_path.read_text(encoding="utf-8")
    asset_version = _asset_version(*asset_paths)
    return (
        template
        .replace(LMS_STATE_PLACEHOLDER, payload)
        .replace(LMS_ASSET_VERSION_PLACEHOLDER, asset_version)
        .replace(LMS_NAV_PLACEHOLDER, _render_nav(state.get("user")))
        .replace("__BRAND_NAME__", _html.escape(BRAND_NAME))
        .replace("__BRAND_ICON_URL__", BRAND_ICON_URL)
    )


def render_lms_home(state: dict) -> str:
    return _render(
        LMS_TEMPLATE_PATH,
        state,
        APP_ROOT / "static" / "app-shell.css",
        APP_ROOT / "static" / "lms.css",
        APP_ROOT / "static" / "lms.js",
    )


def render_lms_courses_page(state: dict) -> str:
    return _render(
        LMS_COURSES_TEMPLATE_PATH,
        state,
        APP_ROOT / "static" / "app-shell.css",
        APP_ROOT / "static" / "lms.css",
        APP_ROOT / "static" / "lms-courses.js",
    )
