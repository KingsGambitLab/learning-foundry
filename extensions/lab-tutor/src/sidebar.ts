import * as fs from "fs";
import * as vscode from "vscode";

export class TutorSidebarProvider implements vscode.WebviewViewProvider {
  public static readonly viewId = "labTutor.chat";
  private cachedTemplate?: string;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly onUserMessage: (text: string) => Promise<string>,
    private readonly assignmentTitle?: string,
  ) {}

  resolveWebviewView(view: vscode.WebviewView): void {
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.extensionUri, "media")],
    };
    view.webview.html = this.html(view.webview);

    view.webview.onDidReceiveMessage(async (msg) => {
      if (msg?.type !== "send" || typeof msg.text !== "string") return;
      try {
        const reply = await this.onUserMessage(msg.text);
        view.webview.postMessage({ type: "reply", text: reply });
      } catch (err) {
        const text = err instanceof Error ? err.message : String(err);
        view.webview.postMessage({ type: "error", text });
      }
    });

    const welcomeLead = this.assignmentTitle
      ? `**Working on ${this.assignmentTitle}.**`
      : `**Lab tutor ready.**`;
    const welcomeBody = this.assignmentTitle
      ? `Tell me where you're stuck — design, code, or testing.`
      : `Ask anything. I won't give you the answer, but I'll help you find it.`;
    const welcome = `${welcomeLead}\n\n${welcomeBody}`;
    // Post after a tick so the webview script is listening.
    setTimeout(() => {
      view.webview.postMessage({ type: "welcome", text: welcome });
      view.webview.postMessage({ type: "subtitle", text: this.assignmentTitle ?? "" });
    }, 0);
  }

  private html(webview: vscode.Webview): string {
    const nonce = nonceOf();
    const scriptUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.extensionUri, "media", "sidebar.js"),
    );
    if (this.cachedTemplate === undefined) {
      this.cachedTemplate = fs.readFileSync(
        vscode.Uri.joinPath(this.extensionUri, "media", "sidebar.html").fsPath,
        "utf8",
      );
    }
    return this.cachedTemplate
      .replaceAll("${nonce}", nonce)
      .replaceAll("${scriptUri}", scriptUri.toString());
  }
}

function nonceOf(): string {
  const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  let s = "";
  for (let i = 0; i < 32; i++) s += chars[Math.floor(Math.random() * chars.length)];
  return s;
}
