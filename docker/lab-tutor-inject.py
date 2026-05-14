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
    session_id = (os.environ.get("LAB_TUTOR_SESSION_ID") or "").strip()

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
    session_attr = html.escape(session_id, quote=True)
    injection = (
        f'\n  {SENTINEL}\n'
        f'  <link rel="stylesheet" href="{base_url_attr}/static/lab-tutor.css">\n'
        f'  <script\n'
        f'    src="{base_url_attr}/static/lab-tutor.js"\n'
        f'    data-base-url="{base_url_attr}"\n'
        f'    data-assignment-title="{title_attr}"\n'
        f'    data-session-id="{session_attr}"\n'
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

    # The CSP that actually blocks our cross-origin script is set as an HTTP
    # response header from server-main.js, not the workbench.html meta tag.
    # Rename the header key so the browser ignores it. Targeted to one file
    # (server-main.js) so extension-side CSP handling stays intact.
    server_main_candidates = [
        Path("/usr/lib/code-server/lib/vscode/out/server-main.js"),
        Path("/usr/lib/code-server/out/server-main.js"),
    ]
    server_main = next((p for p in server_main_candidates if p.is_file()), None)
    if server_main is None:
        for p in Path("/usr/lib/code-server").rglob("server-main.js"):
            if p.is_file():
                server_main = p
                break
    if server_main is not None:
        smcontent = server_main.read_text(encoding="utf-8")
        if '"X-Lt-Disabled-Csp"' in smcontent:
            print(f"[lab-tutor-inject] server-main.js already patched: {server_main}", flush=True)
        else:
            patched, n = re.subn(
                r'"Content-Security-Policy"',
                '"X-Lt-Disabled-Csp"',
                smcontent,
            )
            if n > 0:
                server_main.write_text(patched, encoding="utf-8")
                print(
                    f"[lab-tutor-inject] renamed {n} CSP header occurrences in {server_main}",
                    flush=True,
                )
            else:
                print(
                    f"[lab-tutor-inject] no CSP header occurrences found in {server_main}",
                    file=sys.stderr,
                    flush=True,
                )
    else:
        print("[lab-tutor-inject] server-main.js not found; CSP header NOT patched", file=sys.stderr, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
