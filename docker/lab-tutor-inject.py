#!/usr/bin/env python3
"""Inject the Lab Tutor widget into code-server's workbench.html at container start.

Gated on LAB_TUTOR_BASE_URL — if unset or empty, this is a no-op.
Idempotent: a sentinel HTML comment marks an already-patched file.

TODO (production): Instead of stripping the CSP entirely, narrow it to allow
only the FastAPI backend origin (e.g. http://host.docker.internal:8012).
The current approach of removing the CSP meta tag is a dev-path shortcut.
"""
from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path

SENTINEL = "<!-- lab-tutor-injected -->"

# Candidate paths for code-server's workbench.html. The first one that exists wins.
# The path varies slightly between code-server versions; covers the common ones.
CANDIDATES = [
    "/usr/lib/code-server/lib/vscode/out/vs/code/browser/workbench/workbench.html",
    "/usr/lib/code-server/out/vs/code/browser/workbench/workbench.html",
    "/usr/local/lib/code-server/lib/vscode/out/vs/code/browser/workbench/workbench.html",
]


def find_workbench() -> Path | None:
    # Try the known candidates first.
    for p in CANDIDATES:
        if Path(p).is_file():
            return Path(p)
    # Fallback: search under /usr/lib/code-server.
    base = Path("/usr/lib/code-server")
    if base.exists():
        for p in base.rglob("workbench.html"):
            if p.is_file():
                return p
    return None


def main() -> int:
    base_url = (os.environ.get("LAB_TUTOR_BASE_URL") or "").strip()
    if not base_url:
        print("[lab-tutor-inject] LAB_TUTOR_BASE_URL unset; skipping.", flush=True)
        return 0

    title = (os.environ.get("LAB_TUTOR_ASSIGNMENT_TITLE") or "").strip()

    workbench = find_workbench()
    if workbench is None:
        print("[lab-tutor-inject] workbench.html not found; skipping.", file=sys.stderr, flush=True)
        return 0  # do not fail the container start

    content = workbench.read_text(encoding="utf-8")
    if SENTINEL in content:
        print(f"[lab-tutor-inject] already patched: {workbench}", flush=True)
        return 0

    # Strip the existing CSP meta tag (dev convenience).
    # Match: <meta http-equiv="Content-Security-Policy" ... > with any attributes.
    new_content, csp_count = re.subn(
        r"<meta[^>]*http-equiv=[\"']Content-Security-Policy[\"'][^>]*/?>",
        "",
        content,
        flags=re.IGNORECASE,
    )

    # Build the injection. HTML-escape the title for attribute context.
    base_url_attr = html.escape(base_url, quote=True)
    title_attr = html.escape(title, quote=True)
    injection = (
        f'\n  {SENTINEL}\n'
        f'  <link rel="stylesheet" href="{base_url_attr}/static/lab-tutor.css">\n'
        f'  <script\n'
        f'    src="{base_url_attr}/static/lab-tutor.js"\n'
        f'    data-base-url="{base_url_attr}"\n'
        f'    data-assignment-title="{title_attr}"\n'
        f'    defer\n'
        f'  ></script>\n'
    )

    # Insert before </head>. If </head> is missing, append at end (defensive).
    if "</head>" in new_content:
        new_content = new_content.replace("</head>", injection + "</head>", 1)
    else:
        new_content = new_content + injection

    workbench.write_text(new_content, encoding="utf-8")
    print(
        f"[lab-tutor-inject] patched {workbench} "
        f"(stripped {csp_count} CSP meta; base_url={base_url}; title={title!r})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
