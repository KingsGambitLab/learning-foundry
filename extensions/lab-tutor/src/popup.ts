import * as vscode from "vscode";
import { VivaQuestion } from "./types";
import { escapeHtml } from "./util/html";

export function showVivaPopup(questions: VivaQuestion[]): void {
  const panel = vscode.window.createWebviewPanel(
    "labTutorViva",
    "Lab tutor — viva",
    vscode.ViewColumn.Beside,
    { enableScripts: false, retainContextWhenHidden: true },
  );
  const items = questions
    .map(
      (q, i) =>
        `<li><strong>Q${i + 1}.</strong> ${escapeHtml(q.prompt)}</li>`,
    )
    .join("\n");
  panel.webview.html = `<!DOCTYPE html>
<html><body style="font-family: var(--vscode-font-family); padding: 16px;">
  <h2>Defend two design choices</h2>
  <ol>${items}</ol>
  <p style="opacity: 0.7;">(Stub — recording UI lands in Phase 4.)</p>
</body></html>`;
}

