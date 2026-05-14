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
        res.end(JSON.stringify({ reply: "hi back", hint_tier: null }));
      });
    });
    await new Promise<void>((r) => server.listen(0, r));
    const port = (server.address() as AddressInfo).port;
    baseUrl = `http://127.0.0.1:${port}`;
  });

  after(() => server.close());

  it("POSTs to /v1/tutor/chat with session id and message", async () => {
    const client = new TutorClient(baseUrl, "sess-123");
    const reply = await client.chat("hello");
    assert.equal(captured.url, "/v1/tutor/chat");
    assert.deepEqual(JSON.parse(captured.body!), {
      session_id: "sess-123",
      message: "hello",
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
    // Server above returns the same body for every call; the test verifies plumbing,
    // not response shape — that's covered by the integration test in Task 11.
    assert.equal(typeof result, "object");
  });
});
