import { ChatReply, SubmitResult } from "../types";

export class TutorClient {
  constructor(
    private readonly baseUrl: string,
    private readonly sessionId: string,
  ) {}

  async chat(message: string): Promise<string> {
    const data = await this.post<ChatReply>("/v1/tutor/chat", {
      session_id: this.sessionId,
      message,
    });
    return data.reply;
  }

  async submit(codeSnapshot: string): Promise<SubmitResult> {
    return this.post<SubmitResult>("/v1/tutor/submit", {
      session_id: this.sessionId,
      code_snapshot: codeSnapshot,
    });
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(this.baseUrl.replace(/\/$/, "") + path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      throw new Error(`Tutor service ${res.status} on ${path}`);
    }
    return (await res.json()) as T;
  }
}
