from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
DOCS_TEMPLATE_PATH = APP_ROOT / "templates" / "docs.html"
DOCS_STATE_PLACEHOLDER = "__DOCS_STATE_JSON__"


def build_docs_state(*, openapi_schema: dict) -> dict:
    paths = openapi_schema.get("paths", {})
    schemas = openapi_schema.get("components", {}).get("schemas", {})
    method_names = {"get", "post", "put", "patch", "delete", "options", "head"}
    sections = Counter()
    operation_count = 0

    for path_item in paths.values():
        for method_name, operation in path_item.items():
            if method_name.lower() not in method_names or not isinstance(operation, dict):
                continue
            operation_count += 1
            for tag in operation.get("tags", ["system"]):
                sections[tag] += 1

    return {
        "info": openapi_schema.get("info", {}),
        "openapi_url": "/openapi.json",
        "path_count": len(paths),
        "operation_count": operation_count,
        "schema_count": len(schemas),
        "tag_count": len(sections),
        "sections": [
            {"name": name, "operations": count}
            for name, count in sections.most_common()
        ],
    }


def render_docs_page(state: dict) -> str:
    payload = json.dumps(state).replace("</", "<\\/")
    template = DOCS_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.replace(DOCS_STATE_PLACEHOLDER, payload)
