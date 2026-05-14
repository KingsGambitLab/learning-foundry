import * as vscode from "vscode";
import { showVivaPopup } from "./popup";
import { SubmitResult } from "./types";

export function registerSubmitCommand(
  context: vscode.ExtensionContext,
  submit: () => Promise<SubmitResult>,
): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("lab.submitAssignment", async () => {
      await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: "Submitting..." },
        async () => {
          try {
            const result = await submit();
            if (!result.test_results.passed) {
              vscode.window.showWarningMessage(
                `Tests failed: ${result.test_results.details}`,
              );
              return;
            }
            vscode.window.showInformationMessage("Tests passed. Time to defend.");
            showVivaPopup(result.viva_questions);
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            vscode.window.showErrorMessage(`Submit failed: ${msg}`);
          }
        },
      );
    }),
  );
}
