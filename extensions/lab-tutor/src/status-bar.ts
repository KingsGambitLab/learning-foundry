import * as vscode from "vscode";
import { HintBudget } from "./state/hint-budget";

export class TutorStatusBar {
  private readonly item: vscode.StatusBarItem;

  constructor(private readonly budget: HintBudget) {
    this.item = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100,
    );
    this.item.command = "lab.openTutor";
    this.setState("watching");
  }

  setState(state: "watching" | "coaching" | "idle" | "reviewing"): void {
    const stateLabel: Record<typeof state, string> = {
      watching: "$(eye) Lab tutor: watching",
      coaching: "$(comment-discussion) Lab tutor: coaching",
      idle: "$(clock) Lab tutor: idle",
      reviewing: "$(checklist) Lab tutor: reviewing",
    };
    this.item.text = `${stateLabel[state]} — ${this.budget.label}`;
    this.item.tooltip = "Open lab tutor";
    this.item.show();
  }

  dispose(): void {
    this.item.dispose();
  }
}
