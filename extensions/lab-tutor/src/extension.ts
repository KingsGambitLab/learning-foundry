import * as vscode from "vscode";
import { HintBudget } from "./state/hint-budget";
import { TutorStatusBar } from "./status-bar";
import { TutorSidebarProvider } from "./sidebar";

export function activate(context: vscode.ExtensionContext): void {
  const budget = new HintBudget(4);
  const statusBar = new TutorStatusBar(budget);
  context.subscriptions.push(statusBar);

  const sidebar = new TutorSidebarProvider(
    context.extensionUri,
    async (text) => `(stub) Got: ${text}`,
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
}

export function deactivate(): void {
  // No-op.
}
