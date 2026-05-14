import { describe, it, before, after } from "node:test";
import { strict as assert } from "node:assert";
import * as http from "node:http";
import { AddressInfo } from "node:net";
import { TutorClient } from "../src/services/tutor-client";

describe("TutorClient", () => {
  let server: http.Server;
  let baseUrl: string;
  const captured: { url?: string; body?: string } = {};

  before(async () => {
    server = http.createServer((req, res) => {
      let body = "";
      req.on("data", (c) => { body += c.toString(); });
      req.on("end", () => {
        captured.url = req.url;
        captured.body = body;
        res.setHeader("content-type", "application/json");
        if (req.url === "/v1/tutor/submit") {
          res.end(JSON.stringify({
            test_results: { passed: true, details: "all green" },
            viva_questions: [{ prompt: "Q1" }, { prompt: "Q2" }],
          }));
        } else {
          res.end(JSON.stringify({ reply: "hi back", hint_tier: null }));
        }
      });
    });
    await new Promise<void>((r) => server.listen(0, r));
    const port = (server.address() as AddressInfo).port;
    baseUrl = `http://127.0.0.1:${port}`;
  });

  after(() => server.close());

  it("POSTs to /v1/tutor/chat with session id and message (no title)", async () => {
    const client = new TutorClient(baseUrl, "sess-123");
    const reply = await client.chat("hello");
    assert.equal(captured.url, "/v1/tutor/chat");
    // assignment_title is undefined → JSON.stringify omits the key entirely.
    assert.deepEqual(JSON.parse(captured.body!), {
      session_id: "sess-123",
      message: "hello",
    });
    assert.equal(reply, "hi back");
  });

  it("POSTs assignment_title when client is constructed with a title", async () => {
    const client = new TutorClient(baseUrl, "sess-456", "Build a REST API");
    const reply = await client.chat("stuck on routing");
    assert.equal(captured.url, "/v1/tutor/chat");
    assert.deepEqual(JSON.parse(captured.body!), {
      session_id: "sess-456",
      message: "stuck on routing",
      assignment_title: "Build a REST API",
    });
    assert.equal(reply, "hi back");
  });

  it("submit POSTs to /v1/tutor/submit and returns parsed body", async () => {
    const client = new TutorClient(baseUrl, "sess-123");
    const result = await client.submit("code goes here");
    assert.equal(captured.url, "/v1/tutor/submit");
    assert.deepEqual(JSON.parse(captured.body!), {
      session_id: "sess-123",
      code_snapshot: "code goes here",
    });
    assert.equal(result.test_results.passed, true);
    assert.equal(result.test_results.details, "all green");
    assert.equal(result.viva_questions.length, 2);
    assert.equal(result.viva_questions[0].prompt, "Q1");
  });
});
