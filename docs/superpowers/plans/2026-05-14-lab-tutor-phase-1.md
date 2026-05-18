# Lab Tutor — Phase 1 (Skeleton) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the end-to-end skeleton of the Lab Tutor — a VS Code extension that ships into the cloud-hosted VS Code instances launched by this app, plus a `/v1/tutor/*` FastAPI surface that returns canned responses. No Copilot integration, no event listeners, no real LLM calls yet — just the bones that prove the wiring works.

**Architecture:**
- VS Code extension lives at `extensions/lab-tutor/` (TypeScript, bundled with esbuild, packaged as `.vsix`).
- Extension is pre-installed into the `learner-studio` Docker image at build time and loaded via `code-server --extensions-dir`.
- Backend lives under `app/` as: `app/api/tutor.py` (FastAPI router), `app/services/tutor_service.py` (stub responses), `app/domain/tutor.py` (dataclasses).
- `LearnerStudioService.launch_editor()` and `seed_workspace_from_snapshot()` are extended to seed `.vscode/settings.json` with the tutor base URL — no refactors to their core flow.

**Tech Stack:** TypeScript 5 / esbuild / `@vscode/vsce` (extension); Python 3.12 / FastAPI / `unittest` (backend); Docker (existing `learner-studio.Dockerfile`).

---

## Scope and Open Questions (surface before executing)

This plan covers **Phase 1 only** from the handover doc (skeleton). Phases 2–5 (reactive popups, Copilot hook integration, submit/viva, hardening) need their own plans once Phase 1 lands.

**Execution discipline (maintainer instruction):** treat the handover doc as the source of truth for feature requirements and presentation. For any technical change during execution, do the research first and run the choice through `/codex:adversarial-review` before committing the implementation. This is especially load-bearing on Tasks 7–10 (where we touch existing services) and on any Phase 3 design work.

**Resolved decisions:**

1. **Use `code-server` (Coder's distribution).** Keep existing infra; do not migrate to Microsoft's `code serve-web`. The handover §3.1 cites Copilot licensing on code-server as the reason to switch — that constraint needs fresh verification before Phase 3, not now. Open follow-ups for Phase 3: (a) verify whether Copilot can be officially used on the current code-server build; (b) if not, decide whether sideloading is acceptable or whether rehearsal mode hooks must come from a different agent surface. Phase 1 is hook-agnostic.

**Open questions from §8 that later phases must resolve:**

2. **Hook service auth model.** Trust-by-container-locality vs signed requests. Blocks Phase 3.
3. **Weak-prompt rubric v1.** Blocks Phase 3 (rehearsal mode).
4. **Inline completions policy.** Tab-completion bypasses chat hooks. Blocks Phase 3.

**Constraint from memory:** "Don't edit core course-gen for coding-question paths — new judge/experience capability plugs in as addition, not refactor." User has explicitly placed this code under `app/`, so we're allowed to add — but we still avoid refactoring existing services. We extend `learner_package_runtime.seed_workspace_from_snapshot` minimally and add `--extensions-dir` to the `LearnerStudioService` launch args; everything else is new files.

---

## File Structure

**New files (extension):**
- `extensions/lab-tutor/package.json` — manifest + contributions + scripts
- `extensions/lab-tutor/tsconfig.json` — TS compiler config
- `extensions/lab-tutor/.vscodeignore` — packaging exclusions
- `extensions/lab-tutor/esbuild.mjs` — bundler config
- `extensions/lab-tutor/src/extension.ts` — activation entry
- `extensions/lab-tutor/src/status-bar.ts` — status-bar item
- `extensions/lab-tutor/src/sidebar.ts` — `TutorSidebarProvider` (webview)
- `extensions/lab-tutor/src/submit-command.ts` — `lab.submitAssignment` command
- `extensions/lab-tutor/src/popup.ts` — viva popup webview helper
- `extensions/lab-tutor/src/services/tutor-client.ts` — HTTP client to backend
- `extensions/lab-tutor/src/state/hint-budget.ts` — pure-logic hint-budget tracker (testable without VS Code)
- `extensions/lab-tutor/media/sidebar.html` — sidebar webview HTML
- `extensions/lab-tutor/media/sidebar.js` — sidebar webview script
- `extensions/lab-tutor/media/icon.svg` — activity-bar icon
- `extensions/lab-tutor/test/hint-budget.test.ts` — node:test unit test
- `extensions/lab-tutor/test/tutor-client.test.ts` — node:test unit test
- `extensions/lab-tutor/README.md` — one-paragraph note (so vsce doesn't warn)
- `extensions/lab-tutor/CHANGELOG.md` — one-line entry (so vsce doesn't warn)

**New files (backend):**
- `app/api/tutor.py` — FastAPI router for `/v1/tutor/*`
- `app/services/tutor_service.py` — `TutorService` with canned-response methods
- `app/domain/tutor.py` — request/response dataclasses + Pydantic schemas
- `tests/test_tutor_routes.py` — FastAPI route tests (unittest + `TestClient`)
- `tests/test_tutor_service.py` — service unit tests
- `tests/test_learner_studio_env_vars.py` — verifies the launcher passes `LAB_TUTOR_BASE_URL` + `LAB_TUTOR_SESSION_ID` to the container

**Modified files:**
- `docker/learner-studio.Dockerfile` — copy + build + pre-install the extension into a fixed `--extensions-dir`
- `app/services/learner_studio_service.py` — pass `--extensions-dir /opt/lab-tutor/extensions` to code-server AND pass `LAB_TUTOR_BASE_URL` + `LAB_TUTOR_SESSION_ID` as container env vars (REVISED 2026-05-14 — see Task 8 note below)
- `extensions/lab-tutor/src/extension.ts` — read tutor base URL + session id from `process.env.LAB_TUTOR_*` first; fall back to `labTutor.*` workspace config for dev-only override
- `app/main.py` — include `tutor_router`
- `.gitignore` — add `extensions/lab-tutor/node_modules/`, `extensions/lab-tutor/dist/`, `extensions/lab-tutor/*.vsix`

**REMOVED from scope (2026-05-14):** seeding `.vscode/settings.json` from `seed_workspace_from_snapshot`. Per `/codex:adversarial-review` findings (see Task 8 note below), that approach creates an ownership conflict on a learner-editable file, can erase JSONC-formatted user settings, and races on concurrent launches. The env-var approach in Task 9 replaces it.

---

## Task 1: Extension scaffold + bundler

**Files:**
- Create: `extensions/lab-tutor/package.json`
- Create: `extensions/lab-tutor/tsconfig.json`
- Create: `extensions/lab-tutor/.vscodeignore`
- Create: `extensions/lab-tutor/esbuild.mjs`
- Create: `extensions/lab-tutor/README.md`
- Create: `extensions/lab-tutor/CHANGELOG.md`
- Create: `extensions/lab-tutor/src/extension.ts`
- Modify: `.gitignore`

- [ ] **Step 1: Verify Node is available**

Run: `node --version && npm --version`
Expected: Node 20+ and npm 10+ (matches the Docker image; if missing locally, install via `nvm install 20`).

- [ ] **Step 2: Create `extensions/lab-tutor/package.json`**

```json
{
  "name": "lab-tutor",
  "displayName": "Lab tutor",
  "description": "Embedded tutor for graded coding assignments.",
  "publisher": "scaler",
  "version": "0.1.0",
  "engines": { "vscode": "^1.85.0" },
  "categories": ["Other"],
  "activationEvents": ["onStartupFinished"],
  "main": "./dist/extension.js",
  "contributes": {
    "viewsContainers": {
      "activitybar": [
        { "id": "labTutor", "title": "Lab tutor", "icon": "media/icon.svg" }
      ]
    },
    "views": {
      "labTutor": [
        { "id": "labTutor.chat", "name": "Tutor", "type": "webview" }
      ]
    },
    "commands": [
      { "command": "lab.submitAssignment", "title": "Submit assignment", "category": "Lab tutor", "icon": "$(cloud-upload)" },
      { "command": "lab.openTutor", "title": "Open lab tutor", "category": "Lab tutor" }
    ],
    "menus": {
      "editor/title": [
        { "command": "lab.submitAssignment", "group": "navigation" }
      ]
    },
    "configuration": {
      "title": "Lab tutor",
      "properties": {
        "labTutor.baseUrl": {
          "type": "string",
          "default": "http://localhost:8000",
          "description": "Base URL of the tutor backend service."
        },
        "labTutor.sessionId": {
          "type": "string",
          "default": "",
          "description": "Assignment session id. Seeded by the launcher."
        }
      }
    }
  },
  "scripts": {
    "build": "node esbuild.mjs",
    "watch": "node esbuild.mjs --watch",
    "package": "npm run build && vsce package --no-dependencies -o lab-tutor.vsix",
    "test": "tsc -p tsconfig.test.json && node --test 'test-out/**/*.test.js'"
  },
  "devDependencies": {
    "@types/node": "^20.11.0",
    "@types/vscode": "^1.85.0",
    "@vscode/vsce": "^3.0.0",
    "esbuild": "^0.24.0",
    "typescript": "^5.3.3"
  }
}
```

- [ ] **Step 3: Create `extensions/lab-tutor/tsconfig.json`**

```json
{
  "compilerOptions": {
    "module": "commonjs",
    "target": "ES2022",
    "outDir": "out",
    "lib": ["ES2022"],
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "resolveJsonModule": true,
    "sourceMap": true,
    "rootDir": "src"
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "out", "test"]
}
```

- [ ] **Step 4: Create `extensions/lab-tutor/tsconfig.test.json`**

```json
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "outDir": "test-out",
    "rootDir": ".",
    "noEmit": false
  },
  "include": ["src/**/*", "test/**/*"],
  "exclude": ["node_modules", "dist", "out"]
}
```

- [ ] **Step 5: Create `extensions/lab-tutor/esbuild.mjs`**

```js
import { build, context } from "esbuild";

const watch = process.argv.includes("--watch");
const opts = {
  entryPoints: ["src/extension.ts"],
  bundle: true,
  outfile: "dist/extension.js",
  external: ["vscode"],
  format: "cjs",
  platform: "node",
  target: "node20",
  sourcemap: true,
  logLevel: "info",
};

if (watch) {
  const ctx = await context(opts);
  await ctx.watch();
} else {
  await build(opts);
}
```

- [ ] **Step 6: Create `extensions/lab-tutor/.vscodeignore`**

```
.vscode/**
.vscode-test/**
src/**
test/**
test-out/**
out/**
node_modules/**
.gitignore
tsconfig*.json
esbuild.mjs
**/*.map
```

- [ ] **Step 7: Create `extensions/lab-tutor/README.md`**

```markdown
# Lab tutor

Embedded tutor for graded coding assignments. Coaches the use of agentic coding tools while preserving the learner's judgment development.
```

- [ ] **Step 8: Create `extensions/lab-tutor/CHANGELOG.md`**

```markdown
# Changelog

## 0.1.0

- Scaffold: activation, sidebar, status bar, submit command stubs.
```

- [ ] **Step 9: Create `extensions/lab-tutor/src/extension.ts` (minimal placeholder)**

```ts
import * as vscode from "vscode";

export function activate(_context: vscode.ExtensionContext): void {
  // Real activation wired in Task 2.
}

export function deactivate(): void {
  // No-op.
}
```

- [ ] **Step 10: Append to `.gitignore` at repo root**

Add these lines at the end of the existing `.gitignore`:

```
extensions/*/node_modules/
extensions/*/dist/
extensions/*/out/
extensions/*/test-out/
extensions/*/*.vsix
```

- [ ] **Step 11: Install deps and verify build**

Run: `cd extensions/lab-tutor && npm install && npm run build && ls dist/extension.js`
Expected: `dist/extension.js` exists; no errors.

- [ ] **Step 12: Commit**

```bash
git add extensions/lab-tutor .gitignore
git commit -m "feat(lab-tutor): scaffold extension with esbuild + TS toolchain"
```

---

## Task 2: Status-bar item + activation

**Files:**
- Create: `extensions/lab-tutor/src/status-bar.ts`
- Create: `extensions/lab-tutor/src/state/hint-budget.ts`
- Create: `extensions/lab-tutor/test/hint-budget.test.ts`
- Modify: `extensions/lab-tutor/src/extension.ts`

- [ ] **Step 1: Write failing test for `HintBudget` (pure logic)**

Create `extensions/lab-tutor/test/hint-budget.test.ts`:

```ts
import { describe, it } from "node:test";
import { strict as assert } from "node:assert";
import { HintBudget } from "../src/state/hint-budget";

describe("HintBudget", () => {
  it("starts with the given capacity and zero consumed", () => {
    const b = new HintBudget(4);
    assert.equal(b.remaining, 4);
    assert.equal(b.consumed, 0);
  });

  it("decrements remaining on consume", () => {
    const b = new HintBudget(4);
    b.consume();
    assert.equal(b.remaining, 3);
    assert.equal(b.consumed, 1);
  });

  it("clamps at zero and reports exhausted", () => {
    const b = new HintBudget(1);
    b.consume();
    b.consume();
    assert.equal(b.remaining, 0);
    assert.equal(b.exhausted, true);
  });

  it("formats a human label", () => {
    const b = new HintBudget(4);
    b.consume();
    assert.equal(b.label, "Hints: 3/4");
  });
});
```

- [ ] **Step 2: Run test, verify it fails**

Run: `cd extensions/lab-tutor && npm test`
Expected: FAIL — `Cannot find module '../src/state/hint-budget'`.

- [ ] **Step 3: Implement `HintBudget`**

Create `extensions/lab-tutor/src/state/hint-budget.ts`:

```ts
export class HintBudget {
  private _consumed = 0;
  constructor(private readonly capacity: number) {}

  get remaining(): number {
    return Math.max(0, this.capacity - this._consumed);
  }

  get consumed(): number {
    return Math.min(this.capacity, this._consumed);
  }

  get exhausted(): boolean {
    return this.remaining === 0;
  }

  get label(): string {
    return `Hints: ${this.remaining}/${this.capacity}`;
  }

  consume(): void {
    this._consumed += 1;
  }
}
```

- [ ] **Step 4: Run test, verify it passes**

Run: `cd extensions/lab-tutor && npm test`
Expected: 4 tests passing.

- [ ] **Step 5: Implement status-bar item**

Create `extensions/lab-tutor/src/status-bar.ts`:

```ts
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
```

- [ ] **Step 6: Wire activation in `src/extension.ts`**

Replace `extensions/lab-tutor/src/extension.ts` with:

```ts
import * as vscode from "vscode";
import { HintBudget } from "./state/hint-budget";
import { TutorStatusBar } from "./status-bar";

export function activate(context: vscode.ExtensionContext): void {
  const budget = new HintBudget(4);
  const statusBar = new TutorStatusBar(budget);
  context.subscriptions.push(statusBar);

  context.subscriptions.push(
    vscode.commands.registerCommand("lab.openTutor", async () => {
      await vscode.commands.executeCommand("workbench.view.extension.labTutor");
    }),
  );
}

export function deactivate(): void {
  // No-op.
}
```

- [ ] **Step 7: Verify build still works**

Run: `cd extensions/lab-tutor && npm run build`
Expected: no errors; `dist/extension.js` regenerated.

- [ ] **Step 8: Commit**

```bash
git add extensions/lab-tutor/src extensions/lab-tutor/test
git commit -m "feat(lab-tutor): status bar + hint budget"
```

---

## Task 3: Sidebar webview

**Files:**
- Create: `extensions/lab-tutor/src/sidebar.ts`
- Create: `extensions/lab-tutor/media/sidebar.html`
- Create: `extensions/lab-tutor/media/sidebar.js`
- Create: `extensions/lab-tutor/media/icon.svg`
- Modify: `extensions/lab-tutor/src/extension.ts`
- Modify: `extensions/lab-tutor/.vscodeignore` (re-include `media/`)

- [ ] **Step 1: Create the activity-bar icon**

Create `extensions/lab-tutor/media/icon.svg`:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
  <path d="M12 2L1 7l11 5 9-4.09V17h2V7L12 2zm0 10.18L4.21 8 12 4.18 19.79 8 12 12.18zM5 13.18v4L12 21l7-3.82v-4l-7 3.82L5 13.18z"/>
</svg>
```

- [ ] **Step 2: Create the sidebar HTML shell**

Create `extensions/lab-tutor/media/sidebar.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';" />
  <title>Tutor</title>
  <style>
    body { font-family: var(--vscode-font-family); padding: 12px; margin: 0; color: var(--vscode-foreground); }
    #log { display: flex; flex-direction: column; gap: 8px; margin-bottom: 12px; max-height: 60vh; overflow-y: auto; }
    .msg { padding: 8px 10px; border-radius: 6px; line-height: 1.4; }
    .msg.user { background: var(--vscode-input-background); }
    .msg.tutor { background: var(--vscode-editor-inactiveSelectionBackground); }
    #row { display: flex; gap: 6px; }
    #input { flex: 1; padding: 6px; background: var(--vscode-input-background); color: var(--vscode-input-foreground); border: 1px solid var(--vscode-input-border, transparent); border-radius: 4px; }
    button { padding: 6px 12px; background: var(--vscode-button-background); color: var(--vscode-button-foreground); border: none; border-radius: 4px; cursor: pointer; }
    button:hover { background: var(--vscode-button-hoverBackground); }
  </style>
</head>
<body>
  <div id="log"></div>
  <div id="row">
    <input id="input" type="text" placeholder="Ask the tutor..." />
    <button id="send">Send</button>
  </div>
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>
```

- [ ] **Step 3: Create the sidebar script**

Create `extensions/lab-tutor/media/sidebar.js`:

```js
(function () {
  const vscode = acquireVsCodeApi();
  const log = document.getElementById("log");
  const input = document.getElementById("input");
  const send = document.getElementById("send");

  function append(role, text) {
    const el = document.createElement("div");
    el.className = "msg " + role;
    el.textContent = text;
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
  }

  function submit() {
    const text = input.value.trim();
    if (!text) return;
    append("user", text);
    vscode.postMessage({ type: "send", text });
    input.value = "";
  }

  send.addEventListener("click", submit);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); });

  window.addEventListener("message", (event) => {
    const m = event.data;
    if (m.type === "reply") append("tutor", m.text);
    if (m.type === "error") append("tutor", "Error: " + m.text);
  });
})();
```

- [ ] **Step 4: Implement `TutorSidebarProvider`**

Create `extensions/lab-tutor/src/sidebar.ts`:

```ts
import * as vscode from "vscode";

export class TutorSidebarProvider implements vscode.WebviewViewProvider {
  public static readonly viewId = "labTutor.chat";
  private view?: vscode.WebviewView;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly onUserMessage: (text: string) => Promise<string>,
  ) {}

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
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
  }

  private html(webview: vscode.Webview): string {
    const nonce = nonceOf();
    const scriptUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.extensionUri, "media", "sidebar.js"),
    );
    const tpl = require("fs").readFileSync(
      vscode.Uri.joinPath(this.extensionUri, "media", "sidebar.html").fsPath,
      "utf8",
    );
    return tpl
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
```

- [ ] **Step 5: Register provider in activation**

Update `extensions/lab-tutor/src/extension.ts`:

```ts
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
```

- [ ] **Step 6: Update `.vscodeignore` to include `media/`**

Verify `media/**` is NOT excluded (it isn't by default — but confirm by inspecting the file).

- [ ] **Step 7: Build**

Run: `cd extensions/lab-tutor && npm run build`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add extensions/lab-tutor/src extensions/lab-tutor/media
git commit -m "feat(lab-tutor): sidebar webview with stub echo"
```

---

## Task 4: Submit command

**Files:**
- Create: `extensions/lab-tutor/src/submit-command.ts`
- Create: `extensions/lab-tutor/src/popup.ts`
- Modify: `extensions/lab-tutor/src/extension.ts`

- [ ] **Step 1: Implement the viva popup helper**

Create `extensions/lab-tutor/src/popup.ts`:

```ts
import * as vscode from "vscode";

export interface VivaQuestion {
  prompt: string;
}

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

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[c]!);
}
```

- [ ] **Step 2: Implement the submit command**

Create `extensions/lab-tutor/src/submit-command.ts`:

```ts
import * as vscode from "vscode";
import { showVivaPopup, VivaQuestion } from "./popup";

export interface SubmitResult {
  test_results: { passed: boolean; details: string };
  viva_questions: VivaQuestion[];
}

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
```

- [ ] **Step 3: Wire the command in activation**

Update `extensions/lab-tutor/src/extension.ts` — add the import and call inside `activate`:

```ts
import { registerSubmitCommand } from "./submit-command";
```

Then inside `activate`, after the `lab.openTutor` registration, add:

```ts
registerSubmitCommand(context, async () => ({
  test_results: { passed: true, details: "stub" },
  viva_questions: [
    { prompt: "Explain why you chose this data structure." },
    { prompt: "Walk through your error handling." },
  ],
}));
```

- [ ] **Step 4: Build**

Run: `cd extensions/lab-tutor && npm run build`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add extensions/lab-tutor/src
git commit -m "feat(lab-tutor): submit command + viva popup stub"
```

---

## Task 5: HTTP tutor client (extension side)

**Files:**
- Create: `extensions/lab-tutor/src/services/tutor-client.ts`
- Create: `extensions/lab-tutor/test/tutor-client.test.ts`
- Modify: `extensions/lab-tutor/src/extension.ts`

- [ ] **Step 1: Write failing test for `TutorClient.chat`**

Create `extensions/lab-tutor/test/tutor-client.test.ts`:

```ts
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
```

- [ ] **Step 2: Run test, verify it fails**

Run: `cd extensions/lab-tutor && npm test`
Expected: FAIL — `Cannot find module '../src/services/tutor-client'`.

- [ ] **Step 3: Implement `TutorClient`**

Create `extensions/lab-tutor/src/services/tutor-client.ts`:

```ts
export interface ChatReply {
  reply: string;
  hint_tier: number | null;
}

export interface SubmitReply {
  test_results: { passed: boolean; details: string };
  viva_questions: { prompt: string }[];
}

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

  async submit(codeSnapshot: string): Promise<SubmitReply> {
    return this.post<SubmitReply>("/v1/tutor/submit", {
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
```

- [ ] **Step 4: Run test, verify it passes**

Run: `cd extensions/lab-tutor && npm test`
Expected: both `TutorClient` tests pass (plus the four `HintBudget` tests from Task 2).

- [ ] **Step 5: Wire `TutorClient` into activation**

Update `extensions/lab-tutor/src/extension.ts` — replace the body of `activate`:

```ts
import * as vscode from "vscode";
import { HintBudget } from "./state/hint-budget";
import { TutorStatusBar } from "./status-bar";
import { TutorSidebarProvider } from "./sidebar";
import { registerSubmitCommand } from "./submit-command";
import { TutorClient } from "./services/tutor-client";

export function activate(context: vscode.ExtensionContext): void {
  const cfg = vscode.workspace.getConfiguration("labTutor");
  const baseUrl = cfg.get<string>("baseUrl") ?? "http://localhost:8000";
  const sessionId = cfg.get<string>("sessionId") ?? "dev-session";
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
```

- [ ] **Step 6: Build**

Run: `cd extensions/lab-tutor && npm run build`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add extensions/lab-tutor/src extensions/lab-tutor/test
git commit -m "feat(lab-tutor): HTTP client wired to sidebar and submit"
```

---

## Task 6: Backend domain types and service

**Files:**
- Create: `app/domain/tutor.py`
- Create: `app/services/tutor_service.py`
- Create: `tests/test_tutor_service.py`

- [ ] **Step 1: Write failing test for `TutorService`**

Create `tests/test_tutor_service.py`:

```python
import unittest

from app.domain.tutor import TutorChatRequest, TutorSubmitRequest
from app.services.tutor_service import TutorService, _CHAT_PREVIEW_LIMIT


class TutorServiceTest(unittest.TestCase):
    def test_chat_echoes_truncated_message(self) -> None:
        svc = TutorService()
        reply = svc.chat(TutorChatRequest(session_id="s1", message="hello"))
        self.assertEqual(reply.reply, "(stub) Got: hello")
        self.assertIsNone(reply.hint_tier)

    def test_chat_truncates_long_message(self) -> None:
        svc = TutorService()
        msg = "x" * 200
        reply = svc.chat(TutorChatRequest(session_id="s1", message=msg))
        prefix_len = len("(stub) Got: ")
        self.assertLessEqual(len(reply.reply), _CHAT_PREVIEW_LIMIT + prefix_len)

    def test_submit_returns_two_viva_questions(self) -> None:
        svc = TutorService()
        result = svc.submit(
            TutorSubmitRequest(session_id="s1", code_snapshot="code")
        )
        self.assertEqual(result.test_results["passed"], True)
        self.assertEqual(len(result.viva_questions), 2)
        for q in result.viva_questions:
            self.assertTrue(q.prompt)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python -m unittest tests.test_tutor_service -v`
Expected: FAIL — `ModuleNotFoundError: app.domain.tutor`.

- [ ] **Step 3: Implement the domain types**

Create `app/domain/tutor.py`:

```python
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TutorChatRequest(BaseModel):
    session_id: str
    message: str


class TutorChatResponse(BaseModel):
    reply: str
    hint_tier: int | None = None


class TutorSubmitRequest(BaseModel):
    session_id: str
    code_snapshot: str


class TutorVivaQuestion(BaseModel):
    prompt: str


class TutorSubmitResponse(BaseModel):
    test_results: dict[str, Any] = Field(default_factory=dict)
    viva_questions: list[TutorVivaQuestion] = Field(default_factory=list)
```

- [ ] **Step 4: Implement the service with canned responses**

Create `app/services/tutor_service.py`:

```python
from __future__ import annotations

from app.domain.tutor import (
    TutorChatRequest,
    TutorChatResponse,
    TutorSubmitRequest,
    TutorSubmitResponse,
    TutorVivaQuestion,
)

_CHAT_PREVIEW_LIMIT = 80


class TutorService:
    """Phase 1 stub. Returns canned responses; real RAG/judge wiring is Phase 2."""

    def chat(self, req: TutorChatRequest) -> TutorChatResponse:
        preview = req.message[:_CHAT_PREVIEW_LIMIT]
        return TutorChatResponse(reply=f"(stub) Got: {preview}", hint_tier=None)

    def submit(self, req: TutorSubmitRequest) -> TutorSubmitResponse:
        return TutorSubmitResponse(
            test_results={"passed": True, "details": "stub"},
            viva_questions=[
                TutorVivaQuestion(prompt="Explain why you chose this data structure."),
                TutorVivaQuestion(prompt="Walk through your error handling."),
            ],
        )
```

- [ ] **Step 5: Run test, verify it passes**

Run: `python -m unittest tests.test_tutor_service -v`
Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/domain/tutor.py app/services/tutor_service.py tests/test_tutor_service.py
git commit -m "feat(tutor): domain types + stub service"
```

---

## Task 7: FastAPI router for `/v1/tutor/*`

**Files:**
- Create: `app/api/tutor.py`
- Create: `tests/test_tutor_routes.py`
- Modify: `app/main.py` (register router + add `tutor_service` to lifespan)
- Modify: `app/domain/tutor.py` (add `Field(min_length=1)` constraints to request types)

**Pattern used:** request-scoped DI via `app.state`; no parallel `*Schema` layer — domain types are Pydantic `BaseModel`s and are used directly as request/response types (same pattern as `app/api/routes.py`).

- [x] **Step 1: Confirm where routers are mounted**

`app/main.py` is the mount point. Existing services are initialized in the `@asynccontextmanager async def lifespan(app)` function via `if not hasattr(app.state, ...)` guards, then assigned to `app.state.*`. Routes access them via request-scoped getters like:
```python
def _workflow_service(request: Request) -> WorkflowService:
    return request.app.state.workflow_service
```

- [x] **Step 2: Write failing test for the routes**

Create `tests/test_tutor_routes.py`. Since `TestClient` does not trigger lifespan, tests set `app.state.tutor_service` directly in `setUp` (same pattern used by `test_api.py`, `test_draft_timeline.py`):

```python
import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.services.tutor_service import TutorService


class TutorRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        app.state.tutor_service = TutorService()
        self.client = TestClient(app)

    def test_chat_returns_canned_reply(self) -> None:
        resp = self.client.post(
            "/v1/tutor/chat",
            json={"session_id": "s1", "message": "hello"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["reply"], "(stub) Got: hello")
        self.assertIsNone(body["hint_tier"])

    def test_submit_returns_two_viva_questions(self) -> None:
        resp = self.client.post(
            "/v1/tutor/submit",
            json={"session_id": "s1", "code_snapshot": "x"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["test_results"]["passed"])
        self.assertEqual(len(body["viva_questions"]), 2)
        for q in body["viva_questions"]:
            self.assertTrue(q["prompt"])

    def test_chat_rejects_missing_session_id(self) -> None:
        resp = self.client.post("/v1/tutor/chat", json={"message": "hi"})
        self.assertEqual(resp.status_code, 422)

    def test_chat_rejects_empty_session_id(self) -> None:
        resp = self.client.post(
            "/v1/tutor/chat",
            json={"session_id": "", "message": "hello"},
        )
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 4: Implement the router**

`app/domain/tutor.py` — add `Field(min_length=1)` constraints to request types (validation lives on the domain type, not a parallel schema):

```python
class TutorChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)

class TutorSubmitRequest(BaseModel):
    session_id: str = Field(min_length=1)
    code_snapshot: str
```

`app/api/tutor.py` — request-scoped DI, domain types used directly, no parallel `*Schema` layer:

```python
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.domain.tutor import (
    TutorChatRequest,
    TutorChatResponse,
    TutorSubmitRequest,
    TutorSubmitResponse,
)
from app.services.tutor_service import TutorService


def _tutor_service(request: Request) -> TutorService:
    return request.app.state.tutor_service


router = APIRouter(prefix="/v1/tutor", tags=["tutor"])


@router.post("/chat", response_model=TutorChatResponse)
def chat(
    req: TutorChatRequest,
    svc: TutorService = Depends(_tutor_service),
) -> TutorChatResponse:
    return svc.chat(req)


@router.post("/submit", response_model=TutorSubmitResponse)
def submit(
    req: TutorSubmitRequest,
    svc: TutorService = Depends(_tutor_service),
) -> TutorSubmitResponse:
    return svc.submit(req)
```

- [x] **Step 5: Register the router + initialize service in lifespan**

`app/main.py`:
```python
from app.services.tutor_service import TutorService
# ...
from app.api.tutor import router as tutor_router
# ...

# In lifespan, before yield:
if not hasattr(app.state, "tutor_service"):
    app.state.tutor_service = TutorService()

# After app = FastAPI(...):
app.include_router(tutor_router)
```

- [x] **Step 6: Run the new test, verify it passes**

Run: `python -m unittest tests.test_tutor_routes -v`
Expected: 4 tests pass.

- [x] **Step 7: Re-run the full backend test suite to catch regressions**

Run: `python -m unittest discover -s tests 2>&1 | tail -5`
Expected: existing pass count + 4 new passes; no new failures.

- [x] **Step 8: Commit**

```bash
git add app/api/tutor.py app/main.py app/domain/tutor.py tests/test_tutor_routes.py docs/superpowers/plans/2026-05-14-lab-tutor-phase-1.md
git commit -m "refactor(tutor): align router with codebase DI + drop parallel schemas"
```

---

## Task 8: ~~Seed `.vscode/settings.json` into the workspace~~ — RETIRED 2026-05-14

> **This task was retired before implementation.** `/codex:adversarial-review` flagged three blocking design issues with merging tutor metadata into `.vscode/settings.json`:
>
> 1. **Ownership conflict.** The function's existing contract is skip-if-exists (no clobber). Special-casing settings.json with merge-and-overwrite either silently overrides learner edits to `labTutor.*` (bad UX) or leaves `sessionId` stale on session rotation (functionally broken). No policy works in both directions because settings.json is jointly owned by the learner and the launcher.
> 2. **JSONC corruption.** VS Code settings files are conventionally JSONC (comments, trailing commas). The proposed `json.loads` + fallback-to-`{}` would treat ordinary user-authored settings as corrupt and silently erase them.
> 3. **Race on concurrent launches.** Read-modify-write isn't atomic; two launches against the same workspace last-write-win, pinning one editor to the other session's id. Visible breakage, hard to diagnose.
>
> **Replacement design (now in Task 9):** the launcher passes `LAB_TUTOR_BASE_URL` and `LAB_TUTOR_SESSION_ID` as environment variables on the code-server container. The extension reads them from `process.env.*` at activation, with `labTutor.*` workspace config kept only as a dev-time override. Launcher-owned ephemeral state belongs in launcher-owned ephemeral storage (env vars), not in a learner-editable file.
>
> Task 9 below has been updated to include this env-var change. No code from Task 8's original design lands.

**Original Task 8 content (kept below for historical reference only — DO NOT IMPLEMENT):**

- [ ] **(retired) Step 1: Read the current seeding signature**

Run: `grep -n "def seed_workspace_from_snapshot" app/services/learner_package_runtime.py`
Read the function definition and the immediately surrounding 40 lines so you understand its inputs (workspace root, snapshot) and where it writes files. Note the existing pattern for paths under `.coursegen/` — the tutor file lives at `.vscode/settings.json`, parallel to that.

- [ ] **Step 2: Write failing test**

Create `tests/test_tutor_workspace_seeding.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from app.services.learner_package_runtime import seed_workspace_from_snapshot


class TutorWorkspaceSeedingTest(unittest.TestCase):
    def test_seeds_lab_tutor_settings_json(self) -> None:
        # Minimal stub snapshot — the seeding code reads docs + files off this object.
        # If the existing function calls more attributes, mirror them as MagicMock returns.
        snapshot = MagicMock()
        snapshot.snapshot_id = "snap-1"
        snapshot.files = []
        snapshot.readme_markdown = ""
        snapshot.project_brief_markdown = ""
        snapshot.deliverables_markdown = ""
        snapshot.review_areas = []

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seed_workspace_from_snapshot(
                root,
                snapshot,
                tutor_base_url="http://lab-tutor.svc:8000",
                tutor_session_id="sess-abc",
            )
            settings_path = root / ".vscode" / "settings.json"
            self.assertTrue(settings_path.exists(), "settings.json must be seeded")
            settings = json.loads(settings_path.read_text())
            self.assertEqual(settings["labTutor.baseUrl"], "http://lab-tutor.svc:8000")
            self.assertEqual(settings["labTutor.sessionId"], "sess-abc")

    def test_preserves_existing_settings_json(self) -> None:
        snapshot = MagicMock()
        snapshot.snapshot_id = "snap-1"
        snapshot.files = []
        snapshot.readme_markdown = ""
        snapshot.project_brief_markdown = ""
        snapshot.deliverables_markdown = ""
        snapshot.review_areas = []

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            vsdir = root / ".vscode"
            vsdir.mkdir()
            (vsdir / "settings.json").write_text(json.dumps({"editor.formatOnSave": True}))

            seed_workspace_from_snapshot(
                root,
                snapshot,
                tutor_base_url="http://lab-tutor.svc:8000",
                tutor_session_id="sess-abc",
            )
            settings = json.loads((vsdir / "settings.json").read_text())
            self.assertTrue(settings["editor.formatOnSave"])
            self.assertEqual(settings["labTutor.baseUrl"], "http://lab-tutor.svc:8000")


if __name__ == "__main__":
    unittest.main()
```

**Note:** if `MagicMock` for `snapshot` causes the existing seed code to crash (e.g., the function iterates `snapshot.files` expecting a real type), replace the mock with a real instance of the snapshot dataclass from `app.domain` — `grep -rn "class.*Snapshot" app/domain/` to find it. Use whichever matches the function's real parameter type.

- [ ] **Step 3: Run the new test, verify it fails**

Run: `python -m unittest tests.test_tutor_workspace_seeding -v`
Expected: FAIL — either `TypeError: seed_workspace_from_snapshot() got an unexpected keyword argument 'tutor_base_url'` (signature mismatch) or `settings.json must be seeded`.

- [ ] **Step 4: Extend `seed_workspace_from_snapshot`**

Open `app/services/learner_package_runtime.py`. Extend the signature with two new keyword-only parameters and add the seeding logic at the end of the function (after all existing writes). Adapt to the existing style:

```python
def seed_workspace_from_snapshot(
    workspace_root: Path,
    snapshot,                       # keep existing type annotation
    *,
    tutor_base_url: str | None = None,
    tutor_session_id: str | None = None,
) -> None:
    # ... all existing logic untouched ...

    if tutor_base_url is not None or tutor_session_id is not None:
        _merge_vscode_settings(
            workspace_root / ".vscode" / "settings.json",
            {
                "labTutor.baseUrl": tutor_base_url,
                "labTutor.sessionId": tutor_session_id,
            },
        )


def _merge_vscode_settings(path: Path, updates: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}
    if not isinstance(existing, dict):
        existing = {}
    for k, v in updates.items():
        if v is None:
            continue
        existing[k] = v
    path.write_text(json.dumps(existing, indent=2) + "\n")
```

Add `import json` at the top of the file if not already imported.

- [ ] **Step 5: Run the new test, verify it passes**

Run: `python -m unittest tests.test_tutor_workspace_seeding -v`
Expected: 2 tests pass.

- [ ] **Step 6: Run the existing seeding tests to catch regressions**

Run: `python -m unittest tests.test_publish_snapshot_seed_files tests.test_learner_package_runtime -v`
Expected: previously-passing tests still pass (the new parameters are keyword-only with defaults, so existing callers are unaffected).

- [ ] **Step 7: Thread the params from `LMSService` / launcher (only if existing callers can supply them now)**

Run: `grep -rn "seed_workspace_from_snapshot" app/` to locate every caller.

- For Phase 1 we leave existing callers unchanged (the new params default to `None`, which keeps current behavior). The launcher will pass them in **Task 9** after we know the base URL it computes. No changes here.

- [ ] **Step 8: Commit**

```bash
git add app/services/learner_package_runtime.py tests/test_tutor_workspace_seeding.py
git commit -m "feat(learner-package): seed labTutor base URL into .vscode/settings.json"
```

---

## Task 9: Pass tutor URL + session id through `LearnerStudioService.launch_editor` — REVISED 2026-05-14

> **Revised design** (after Task 8's adversarial review). Launcher passes `LAB_TUTOR_BASE_URL` and `LAB_TUTOR_SESSION_ID` as **container environment variables** at `docker run` time. The extension reads them from `process.env.LAB_TUTOR_*` at activation, with `labTutor.*` workspace config retained only as a dev-time override. Nothing is written to `.vscode/settings.json` from the launcher.

**Files:**
- Modify: `app/services/learner_studio_service.py` — pass `-e LAB_TUTOR_BASE_URL=... -e LAB_TUTOR_SESSION_ID=...` on the docker invocation, and add `--extensions-dir /opt/lab-tutor/extensions` to the code-server command.
- Modify: `extensions/lab-tutor/src/extension.ts` — read `process.env.LAB_TUTOR_*` first; fall back to `vscode.workspace.getConfiguration("labTutor")`.
- Create: `tests/test_learner_studio_env_vars.py` — verifies the docker invocation carries the env vars and the `--extensions-dir` flag.

**Below this revised header, the original step list is preserved for context but should be re-read against the new design — most original steps still apply with `tutor_base_url` and `tutor_session_id` flowing as env vars rather than as kwargs to `seed_workspace_from_snapshot`.**

- [ ] **Step 1: Read the launcher**

Run: `grep -n "def launch_editor\|seed_workspace_from_snapshot\|extensions-dir\|user-data-dir" app/services/learner_studio_service.py`
Read the surrounding 60 lines so you understand: (a) which seeding call (if any) it makes, (b) where the code-server invocation is constructed, (c) what info the launcher already has about session id and tutor base URL.

If the launcher does not currently call `seed_workspace_from_snapshot` itself (that call may live in `LMSService._ensure_workspace_seeded` per the explore agent's findings), then this task instead lives in `LMSService` — apply the same pattern there.

- [ ] **Step 2: Write a failing test that captures the seeded settings**

Append to `tests/test_learner_studio_service.py` (or, if the seeding happens in `LMSService`, to `tests/test_lms_service.py`):

```python
def test_launch_editor_seeds_tutor_settings(self) -> None:
    # Use the existing test setUp/factories — DO NOT duplicate fixture setup.
    # Goal: assert that after launch_editor (mocked Docker), the workspace contains
    # .vscode/settings.json with labTutor.baseUrl populated to the configured value.
    ...
```

Concrete shape depends on the existing test conventions in that file — read 1–2 existing test methods first and follow their fixture/mocking style. The assertion that must pass:

```python
settings = json.loads(
    (workspace_root / ".vscode" / "settings.json").read_text()
)
self.assertTrue(settings["labTutor.baseUrl"].startswith("http"))
self.assertEqual(settings["labTutor.sessionId"], expected_session_id)
```

- [ ] **Step 3: Run the new test, verify it fails**

Run the specific new test method: `python -m unittest tests.test_learner_studio_service.<ClassName>.test_launch_editor_seeds_tutor_settings -v`
Expected: FAIL.

- [ ] **Step 4: Wire the parameters into the launcher**

In `app/services/learner_studio_service.py`:

1. Add a constructor parameter (or class attribute) `tutor_base_url: str` with a default sourced from an env var:

```python
import os
# inside __init__:
self._tutor_base_url = tutor_base_url or os.environ.get(
    "LAB_TUTOR_BASE_URL", "http://localhost:8000"
)
```

2. In `launch_editor` (or in the seeding helper it calls), pass through:

```python
seed_workspace_from_snapshot(
    workspace_root,
    snapshot,
    tutor_base_url=self._tutor_base_url,
    tutor_session_id=session_id,  # whatever variable already identifies the session
)
```

3. Also append `--extensions-dir /opt/lab-tutor/extensions` to the existing code-server command list (so the pre-installed extension in Task 10's Docker image is discovered). Find the existing args list (the explore agent identified it around `learner_studio_service.py` line ~132) and append:

```python
code_server_cmd = [
    *existing_args,
    "--extensions-dir", "/opt/lab-tutor/extensions",
]
```

- [ ] **Step 5: Update the launcher's lifespan wiring**

In `app/main.py`, where `LearnerStudioService` is constructed (the explore agent identified it in the lifespan setup at lines 70–80), pass `tutor_base_url` if you want non-default behavior. For Phase 1 the env-var default is enough — no `main.py` change strictly required.

- [ ] **Step 6: Run the new test, verify it passes**

Run: `python -m unittest tests.test_learner_studio_service.<ClassName>.test_launch_editor_seeds_tutor_settings -v`
Expected: pass.

- [ ] **Step 7: Re-run the full test suite**

Run: `python -m unittest discover -s tests 2>&1 | tail -10`
Expected: existing tests still pass.

- [ ] **Step 8: Commit**

```bash
git add app/services/learner_studio_service.py tests/test_learner_studio_service.py
git commit -m "feat(learner-studio): seed tutor URL into workspace + load extensions-dir"
```

---

## Task 10: Bake the extension into the Docker image — REVISED 2026-05-14

> **Revised design** (after `/codex:adversarial-review`). Two blockers in the original plan:
> 1. `LearnerStudioService._ensure_image()` only builds when the tag is absent. With a static `course-gen-learner-studio:latest` tag, every change to the Dockerfile or `extensions/lab-tutor/**` is silently ignored after the first build. Learners would run stale extension code with no warning.
> 2. The build context is the repo root with no `.dockerignore`, so Docker tars ~161MB (incl. ~155MB of `extensions/lab-tutor/node_modules`) on every build. Performance and a trust-boundary concern: local learner data under `learner_workspaces/`, secrets in `.claude/`, etc., would be sent to the daemon.
>
> **Revised approach:**
> - Add a repo-root `.dockerignore` excluding `.git`, `.claude`, `logs/`, `tmp/`, `learner_workspaces/`, `workspaces/`, `extensions/*/node_modules`, `extensions/*/dist`, `extensions/*/test-out`, `extensions/*/*.vsix`.
> - Multi-stage Dockerfile with the build stage split into `COPY package*.json` → `npm ci` → `COPY` source so the dependency layer caches independently of source edits.
> - Derive the image tag from a SHA-1 over the contents of `docker/learner-studio.Dockerfile` and every tracked file under `extensions/lab-tutor/` (excluding `node_modules`, `dist`, `test-out`, `*.vsix`). New helper `_compute_learner_studio_image_tag()` in `app/services/learner_studio_service.py`. Tag format: `course-gen-learner-studio:<12-char-hash>`. `_ensure_image` continues to check-then-build, but a fresh tag implies a fresh build whenever inputs change.
> - `default_learner_studio_image()` returns the hashed tag when invoked; tests that want a deterministic image name can construct the service with `image_name=...` explicitly (existing behavior preserved).

**Files:**
- Create: `.dockerignore` at the repo root.
- Modify: `docker/learner-studio.Dockerfile` (multi-stage with layer-cache split + extension install).
- Modify: `app/services/learner_studio_service.py` — content-hash-based image tag + helper function.
- Create: `tests/test_learner_studio_image_tag.py` — verifies the hash changes when a file changes.

**The original Step 1+ content below remains for context but is superseded by the revised design.**

- [ ] **Step 1: Read the current Dockerfile**

Run: `cat docker/learner-studio.Dockerfile`
Note: the image already includes Node + npm + code-server.

- [ ] **Step 2: Add extension build + install stages**

Append a build stage that compiles + packages the extension, and install the resulting `.vsix` into `/opt/lab-tutor/extensions` in the final image. If the Dockerfile is single-stage, refactor to multi-stage:

```dockerfile
# --- Stage: build the lab-tutor extension --------------------------------
FROM node:20-bookworm-slim AS lab-tutor-build
WORKDIR /build
COPY extensions/lab-tutor ./
RUN npm install --no-audit --no-fund \
 && npm run package \
 && ls -la /build/lab-tutor.vsix

# --- Final image (preserve existing FROM/USER/WORKDIR/installs) ----------
# ...existing FROM and code-server install...
COPY --from=lab-tutor-build /build/lab-tutor.vsix /opt/lab-tutor/lab-tutor.vsix
RUN mkdir -p /opt/lab-tutor/extensions \
 && code-server --extensions-dir /opt/lab-tutor/extensions \
                --install-extension /opt/lab-tutor/lab-tutor.vsix
```

Confirm two things while editing:
- The COPY path in the build stage is correct relative to the Docker build context. If the existing build context is the repo root (verify by running `grep -rn "docker build\|context:" docker/ docker-compose.yml 2>/dev/null` — search for compose files first), then `COPY extensions/lab-tutor ./` works as written. Otherwise, adjust the build context in `docker-compose.yml` or the call site.
- The user `code-server --install-extension` runs as must have write perms to `/opt/lab-tutor/extensions`. If the final image switches to a non-root user before this line, move the install to before the user switch, or `chown` the directory.

- [ ] **Step 3: Smoke-build the image**

Run: `docker build -f docker/learner-studio.Dockerfile -t learner-studio:test .`
Expected: build succeeds; final image contains `/opt/lab-tutor/extensions/scaler.lab-tutor-0.1.0/` (or similar).

If you don't have Docker available locally, run: `docker --version` to confirm. If absent, document this step as deferred and create a follow-up TODO; mention in the commit message.

- [ ] **Step 4: Verify the extension installed inside the image**

Run: `docker run --rm learner-studio:test ls /opt/lab-tutor/extensions/`
Expected: a directory containing `package.json` and `dist/extension.js`.

- [ ] **Step 5: Commit**

```bash
git add docker/learner-studio.Dockerfile
git commit -m "feat(docker): pre-install lab-tutor extension into learner-studio image"
```

---

## Task 11: End-to-end smoke verification

**Files:** (no code changes — verification only)

- [ ] **Step 1: Start the backend**

Run: `uvicorn app.main:app --reload --port 8000` (in one terminal)
Expected: FastAPI logs `Uvicorn running on http://127.0.0.1:8000`.

- [ ] **Step 2: Hit the tutor routes with curl**

```bash
curl -s -X POST http://localhost:8000/v1/tutor/chat \
  -H 'content-type: application/json' \
  -d '{"session_id":"smoke","message":"hello"}'

curl -s -X POST http://localhost:8000/v1/tutor/submit \
  -H 'content-type: application/json' \
  -d '{"session_id":"smoke","code_snapshot":"print(1)"}'
```

Expected: `chat` returns `{"reply":"(stub) Got: hello","hint_tier":null}`; `submit` returns the test result + two viva prompts.

- [ ] **Step 3: Launch a learner workspace through the existing flow**

Use the existing learner-studio path that calls `LearnerStudioService.launch_editor` (the `/v1/learner/*` API surfaces). Pick the smallest existing way to drive this — check `scripts/` or the test files for a working integration entrypoint, or use the LMS UI route documented in the README.

- [ ] **Step 4: Open the launched code-server in a browser**

Confirm three things:
1. The "Lab tutor" icon appears in the activity bar; clicking it shows the sidebar.
2. Typing "hello" in the sidebar and pressing Enter shows a `(stub) Got: hello` reply.
3. A "Submit assignment" button appears in the editor title bar; clicking it shows the viva popup with two questions.

- [ ] **Step 5: Verify the seeded settings**

Inside the running container (or by inspecting the mounted workspace from the host):

```bash
cat <workspace_root>/.vscode/settings.json
```

Expected: `labTutor.baseUrl` and `labTutor.sessionId` are populated.

- [ ] **Step 6: Document the smoke results**

Create `extensions/lab-tutor/SMOKE.md` with a one-page log: timestamp, image tag built, the three things observed in step 4, and any deviations. Commit it.

```bash
git add extensions/lab-tutor/SMOKE.md
git commit -m "docs(lab-tutor): record Phase 1 smoke verification"
```

---

## Definition of done for this plan

- [ ] `npm test` in `extensions/lab-tutor/` reports all unit tests passing.
- [ ] `python -m unittest discover -s tests` passes with no new failures.
- [ ] `docker build -f docker/learner-studio.Dockerfile -t learner-studio:test .` succeeds.
- [ ] A learner workspace launched through the existing path shows the sidebar, status bar, submit button, and sidebar chat returns the canned reply.
- [ ] `.vscode/settings.json` in the seeded workspace contains `labTutor.baseUrl` and `labTutor.sessionId`.
- [ ] No regressions in pre-existing tests.

---

## Self-review notes

- **Spec coverage:** Handover §5.1 (extension surfaces: sidebar, status bar, submit, popup) covered by Tasks 2–4 + 7. Event listeners (terminal, inactivity, save, paste detection) intentionally deferred to Phase 2 per the handover Phase 1 scope. Hook config (§5.2) deferred to Phase 3. Local hook service (§5.3) deferred to Phase 3. Egress proxy (§5.5) deferred to Phase 5. Git server-side hooks (§5.6) deferred to Phase 4. Audit log tables (§6) deferred — Phase 1 uses no persistence; routes are stateless. This is in scope-line with the handover's Phase 1 description.
- **Architectural decision logged:** Continue with `code-server`. Phase 3 (Copilot hooks) must verify Copilot's licensing/support story on code-server before scheduling; that verification should run through `/codex:adversarial-review`.
- **No placeholders.** Every code step contains the actual code; every command is runnable.
- **Type consistency:** `ChatReply` and `ChatResponseSchema` both have `{reply: string, hint_tier: int | null}`. `SubmitReply` and `SubmitResponseSchema` both have `{test_results, viva_questions}` with matching `prompt` field. `TutorChatRequest`/`TutorSubmitRequest` dataclass field names (`session_id`, `message`, `code_snapshot`) match the Pydantic `*Schema` and the TS client's POST bodies.
- **Memory respected:** No core course-gen refactors. The only modifications to existing files are: (a) one optional kw-arg added to `seed_workspace_from_snapshot`, (b) one launch-arg added to `LearnerStudioService`, (c) one `include_router` line in `app/main.py`. Everything else is new files.
