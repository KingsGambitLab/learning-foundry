"""Offline visible self-check. No network, no real LLM, no submission.

Starts a tiny in-process LLM stub and points LAB_LLM_BASE_URL at it so
your S8 path runs offline, then drives the visible samples through your
app via FastAPI's TestClient and checks the DETERMINISTIC fields
(action / abstained / citations / redactions). The hidden grader uses
different conversations from the same distribution.
"""
from __future__ import annotations

import json
import os
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = pathlib.Path(__file__).resolve().parents[2]


class _Stub(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0) or 0))
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "content": "Here's how to do that, based on our help docs.",
            "usage": {"input_tokens": 50, "output_tokens": 20},
            "model": "stub",
        }).encode())

    def log_message(self, *a):  # silence
        pass


def _start_stub() -> str:
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}"


def main() -> int:
    os.environ["LAB_LLM_BASE_URL"] = _start_stub()
    os.environ.setdefault("LAB_LLM_TOKEN", "stub-token")

    import sys
    sys.path.insert(0, str(HERE))
    from app import app  # noqa: E402
    from fastapi.testclient import TestClient  # noqa: E402

    data = json.loads((HERE / "public/examples/sample_conversations.json").read_text())
    kb = data["kb_articles"]
    client = TestClient(app)
    passed = failed = 0
    for c in data["cases"]:
        body = {"message": c["message"], "conversation_id": c["name"],
                "history": c.get("history", []), "kb_articles": kb}
        r = client.post("/support/answer", json=body)
        ok = r.status_code == 200
        j = r.json() if ok else {}
        e = c["expect"]
        if "action" in e:
            ok = ok and j.get("action") == e["action"]
        if "abstained" in e:
            ok = ok and j.get("abstained") == e["abstained"]
        if "citations_include" in e:
            ok = ok and e["citations_include"] in (j.get("citations") or [])
        if "redactions_min" in e:
            ok = ok and int(j.get("redactions", 0)) >= e["redactions_min"]
        print(("PASS" if ok else "FAIL"), c["name"], "->",
              {k: j.get(k) for k in ("action", "citations", "redactions", "abstained")})
        passed += ok
        failed += not ok
    print(f"\n{passed} passed / {failed} failed (visible samples)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
