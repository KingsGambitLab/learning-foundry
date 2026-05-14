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
  const assignmentTitle =
    process.env.LAB_TUTOR_ASSIGNMENT_TITLE ||
    cfg.get<string>("assignmentTitle") ||
    undefined;
  const client = new TutorClient(baseUrl, sessionId, assignmentTitle);

  const budget = new HintBudget(4);
  const statusBar = new TutorStatusBar(budget);
  context.subscriptions.push(statusBar);

  const sidebar = new TutorSidebarProvider(
    context.extensionUri,
    (text) => client.chat(text),
    assignmentTitle,
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

  // Open the tutor pane immediately so the learner doesn't have to discover it.
  // The setTimeout ensures the view provider is registered before the focus command fires.
  setTimeout(() => {
    vscode.commands.executeCommand("labTutor.chat.focus").then(
      undefined,
      () => {
        // Fallback for older VS Code/code-server versions
        vscode.commands.executeCommand("workbench.view.extension.labTutor");
      },
    );
  }, 500);
}

function currentEditorContent(): string {
  return vscode.window.activeTextEditor?.document.getText() ?? "";
}

export function deactivate(): void {
  // No-op.
}
