import * as vscode from "vscode";
import { HintBudget } from "./state/hint-budget";
import { TutorStatusBar } from "./status-bar";
import { TutorSidebarProvider } from "./sidebar";
import { registerSubmitCommand } from "./submit-command";
import { TutorClient } from "./services/tutor-client";

export function activate(context: vscode.ExtensionContext): void {
  const cfg = vscode.workspace.getConfiguration("labTutor");
  const baseUrl =
    process.env.LAB_TUTOR_BASE_URL ||
    cfg.get<string>("baseUrl") ||
    "http://localhost:8000";
  const sessionId =
    process.env.LAB_TUTOR_SESSION_ID ||
    cfg.get<string>("sessionId") ||
    "dev-session";
  const client = new TutorClient(baseUrl, sessionId);

  const budget = new HintBudget(4);
  const statusBar = new TutorStatusBar(budget);
  context.subscriptions.push(statusBar);

  const sidebar = new TutorSidebarProvider(
    context.extensionUri,
    (text) => client.chat(text),
  );
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      TutorSidebarProvider.viewId,
      sidebar,
    ),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("lab.openTutor", async () => {
      await vscode.commands.executeCommand("workbench.view.extension.labTutor");
    }),
  );

  registerSubmitCommand(context, () => client.submit(currentEditorContent()));
}

function currentEditorContent(): string {
  return vscode.window.activeTextEditor?.document.getText() ?? "";
}

export function deactivate(): void {
  // No-op.
}
