/* lab-tutor.js — Lab Tutor floating chat widget
   Vanilla JS, no framework, no bundler. ES2020.

   Two invocation modes:
     1. <script src="/static/lab-tutor.js" data-*="..." defer>
        Reads data attributes on the script tag; shows immediately.
     2. Programmatic via window.__labTutorMount({ assignmentTitle, sessionId, baseUrl })
        and window.__labTutorUnmount()
        Used by the LMS SPA to show/hide per course toggle. */
(function () {
  "use strict";

  // ── Config from script tag ────────────────────────────────────────────────
  const me = document.currentScript
    || document.querySelector('script[src*="lab-tutor.js"]');

  const scriptCfg = {
    baseUrl: (me && me.dataset.baseUrl) || "",
    assignmentTitle: (me && me.dataset.assignmentTitle) || "",
    sessionId:
      (me && me.dataset.sessionId) ||
      ("lms-" + Math.random().toString(36).slice(2)),
    enrollmentId: (me && me.dataset.enrollmentId) || "",
  };

  // ── Persistence ───────────────────────────────────────────────────────────
  const STORAGE_VERSION = 1;
  const HISTORY_MAX = 50;

  function storageKey(cfg) {
    const id = cfg.enrollmentId || cfg.sessionId || "anon";
    return `lab_tutor_history.v${STORAGE_VERSION}.${id}`;
  }

  function loadHistory(cfg) {
    try {
      const raw = localStorage.getItem(storageKey(cfg));
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed.filter(
        (m) => m && (m.role === "user" || m.role === "tutor") && typeof m.text === "string"
      );
    } catch {
      return [];
    }
  }

  function saveHistory(cfg, history) {
    try {
      const trimmed = history.slice(-HISTORY_MAX);
      localStorage.setItem(storageKey(cfg), JSON.stringify(trimmed));
    } catch {
      /* quota or disabled — silent */
    }
  }

  // ── Inject stylesheet once ────────────────────────────────────────────────
  if (!document.querySelector('link[href*="lab-tutor.css"]')) {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = (scriptCfg.baseUrl || "") + "/static/lab-tutor.css";
    document.head.appendChild(link);
  }

  // ── Mermaid helpers ───────────────────────────────────────────────────────
  // Lazy-load mermaid.min.js once. Returns a promise that resolves with the
  // global `mermaid` object. Falls back to null on load failure.
  let mermaidPromise = null;
  function loadMermaid() {
    if (mermaidPromise) return mermaidPromise;
    mermaidPromise = new Promise((resolve) => {
      if (window.mermaid) return resolve(window.mermaid);
      const script = document.createElement("script");
      script.src = (scriptCfg.baseUrl || "") + "/static/vendor/mermaid.min.js";
      script.onload = () => {
        try {
          window.mermaid.initialize({
            startOnLoad: false,
            theme: "default",
            securityLevel: "loose",  // we control the input via the tutor's reply
            fontFamily: "inherit",
          });
          resolve(window.mermaid);
        } catch (e) {
          console.warn("[lab-tutor] mermaid init failed:", e);
          resolve(null);
        }
      };
      script.onerror = () => {
        console.warn("[lab-tutor] mermaid script load failed");
        resolve(null);
      };
      document.head.appendChild(script);
    });
    return mermaidPromise;
  }

  let mermaidIdCounter = 0;
  async function renderMermaidInto(parent, code) {
    // Show a small "Rendering diagram…" placeholder while we load+parse.
    const placeholder = document.createElement("div");
    placeholder.className = "lt-mermaid lt-mermaid--loading";
    placeholder.textContent = "Rendering diagram…";
    parent.appendChild(placeholder);

    const mermaid = await loadMermaid();
    if (!mermaid) {
      // Fallback: show the original mermaid source as a code block.
      placeholder.className = "lt-mermaid lt-mermaid--failed";
      placeholder.textContent = "";
      const pre = document.createElement("pre");
      pre.textContent = code;
      placeholder.appendChild(pre);
      return;
    }
    try {
      const id = "lt-mermaid-" + (++mermaidIdCounter);
      const { svg } = await mermaid.render(id, code);
      placeholder.className = "lt-mermaid";
      placeholder.textContent = "";
      // svg is a trusted-ish string from mermaid; we control the input by virtue
      // of it coming from the tutor's reply (our own backend). Use a sandboxed
      // container so any quirks stay isolated.
      const wrap = document.createElement("div");
      wrap.className = "lt-mermaid-svg";
      wrap.innerHTML = svg;
      placeholder.appendChild(wrap);
    } catch (e) {
      console.warn("[lab-tutor] mermaid render error:", e);
      placeholder.className = "lt-mermaid lt-mermaid--failed";
      placeholder.textContent = "";
      const note = document.createElement("div");
      note.className = "lt-mermaid-error";
      note.textContent = "Couldn't render diagram. Source:";
      placeholder.appendChild(note);
      const pre = document.createElement("pre");
      pre.textContent = code;
      placeholder.appendChild(pre);
    }
  }

  // ── SVG helpers ───────────────────────────────────────────────────────────
  function chatIcon() {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "none");
    svg.setAttribute("aria-hidden", "true");
    svg.innerHTML = `<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><circle cx="9" cy="10" r="1" fill="currentColor"/><circle cx="12" cy="10" r="1" fill="currentColor"/><circle cx="15" cy="10" r="1" fill="currentColor"/>`;
    return svg;
  }

  function closeIcon() {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "none");
    svg.setAttribute("aria-hidden", "true");
    svg.innerHTML = `<line x1="18" y1="6" x2="6" y2="18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><line x1="6" y1="6" x2="18" y2="18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>`;
    return svg;
  }

  function expandIcon() {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "none");
    svg.setAttribute("aria-hidden", "true");
    // Four corner arrows pointing outward — "maximise" glyph
    svg.innerHTML = `<polyline points="4 10 4 4 10 4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><polyline points="20 10 20 4 14 4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><polyline points="4 14 4 20 10 20" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><polyline points="20 14 20 20 14 20" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>`;
    return svg;
  }

  function collapseIcon() {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "none");
    svg.setAttribute("aria-hidden", "true");
    // Four corner arrows pointing inward — "minimise" glyph
    svg.innerHTML = `<polyline points="10 4 10 10 4 10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><polyline points="14 4 14 10 20 10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><polyline points="10 20 10 14 4 14" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><polyline points="14 20 14 14 20 14" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>`;
    return svg;
  }

  // ── Widget factory ────────────────────────────────────────────────────────
  // Called once; returns mount/unmount/update handles.
  function createWidget(initialCfg) {
    let cfg = Object.assign({}, scriptCfg, initialCfg || {});
    let panelOpen = false;
    let welcomeShown = false;
    let thinking = false;
    let history = [];

    // Root container
    const root = document.createElement("div");
    root.className = "lt-root";

    // Bubble
    const bubble = document.createElement("button");
    bubble.className = "lt-bubble";
    bubble.type = "button";
    bubble.setAttribute("aria-label", "Open Lab Tutor");
    bubble.appendChild(chatIcon());

    const dot = document.createElement("span");
    dot.className = "lt-dot";
    dot.setAttribute("aria-hidden", "true");
    bubble.appendChild(dot);

    // Panel
    const panel = document.createElement("div");
    panel.className = "lt-panel lt-panel--hidden";
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", "Lab Tutor");

    // Header
    const header = document.createElement("div");
    header.className = "lt-header";

    const headerTitle = document.createElement("span");
    headerTitle.className = "lt-header-title";
    headerTitle.textContent = cfg.assignmentTitle || "Lab Tutor";

    const expandBtn = document.createElement("button");
    expandBtn.className = "lt-icon-btn lt-expand";
    expandBtn.type = "button";
    expandBtn.setAttribute("aria-label", "Maximise Lab Tutor");
    expandBtn.appendChild(expandIcon());

    const closeBtn = document.createElement("button");
    closeBtn.className = "lt-icon-btn lt-close";
    closeBtn.type = "button";
    closeBtn.setAttribute("aria-label", "Close Lab Tutor");
    closeBtn.appendChild(closeIcon());

    const headerActions = document.createElement("div");
    headerActions.className = "lt-header-actions";
    headerActions.appendChild(expandBtn);
    headerActions.appendChild(closeBtn);

    header.appendChild(headerTitle);
    header.appendChild(headerActions);

    let expanded = false;
    function toggleExpanded(force) {
      const next = typeof force === "boolean" ? force : !expanded;
      if (next === expanded) return;
      expanded = next;
      panel.classList.toggle("lt-panel--expanded", expanded);
      // Swap the button icon + label so the affordance reflects the next action.
      expandBtn.innerHTML = "";
      expandBtn.appendChild(expanded ? collapseIcon() : expandIcon());
      expandBtn.setAttribute(
        "aria-label",
        expanded ? "Minimise Lab Tutor" : "Maximise Lab Tutor"
      );
    }
    expandBtn.addEventListener("click", () => toggleExpanded());

    // Message log
    const log = document.createElement("div");
    log.className = "lt-log";
    log.setAttribute("aria-live", "polite");
    log.setAttribute("aria-label", "Conversation");

    // Input row
    const inputRow = document.createElement("div");
    inputRow.className = "lt-input-row";

    const input = document.createElement("input");
    input.className = "lt-input";
    input.type = "text";
    input.placeholder = "Ask something...";
    input.setAttribute("aria-label", "Message");

    const sendBtn = document.createElement("button");
    sendBtn.className = "lt-send";
    sendBtn.type = "button";
    sendBtn.textContent = "Send";

    inputRow.appendChild(input);
    inputRow.appendChild(sendBtn);

    panel.appendChild(header);
    panel.appendChild(log);
    panel.appendChild(inputRow);

    root.appendChild(bubble);
    root.appendChild(panel);

    // ── Welcome card ────────────────────────────────────────────────────────
    function buildWelcomeCard() {
      const card = document.createElement("div");
      card.className = "lt-welcome";

      const p = document.createElement("p");
      p.style.margin = "0 0 8px";

      if (cfg.assignmentTitle) {
        const strong = document.createElement("strong");
        strong.textContent = "Working on " + cfg.assignmentTitle + ".";
        p.appendChild(strong);
        p.appendChild(document.createTextNode(
          " Tell me where you're stuck — design, code, or testing."
        ));
      } else {
        const strong = document.createElement("strong");
        strong.textContent = "Ask anything.";
        p.appendChild(strong);
        p.appendChild(document.createTextNode(
          " I won't give you the answer — but I'll help you find it."
        ));
      }

      const chips = document.createElement("div");
      chips.className = "lt-chips";

      const chipLabels = [
        "I'm stuck",
        "Why isn't this working?",
        "Walk me through the design",
      ];
      for (const label of chipLabels) {
        const chip = document.createElement("button");
        chip.className = "lt-chip";
        chip.type = "button";
        chip.textContent = label;
        chip.addEventListener("click", () => {
          input.value = label;
          input.focus();
        });
        chips.appendChild(chip);
      }

      card.appendChild(p);
      card.appendChild(chips);
      return card;
    }

    // ── Render helpers ───────────────────────────────────────────────────────
    function scrollToBottom() {
      log.scrollTop = log.scrollHeight;
    }

    function appendUser(text, { persist = true } = {}) {
      const wrap = document.createElement("div");
      wrap.className = "lt-msg lt-msg--user";
      const bub = document.createElement("div");
      bub.className = "lt-msg-bubble";
      bub.textContent = text;
      wrap.appendChild(bub);
      log.appendChild(wrap);
      scrollToBottom();
      if (persist) {
        history.push({ role: "user", text });
        saveHistory(cfg, history);
      }
    }

    // Render a text segment with **bold** parsing into a container element.
    // Does NOT use innerHTML on untrusted content.
    function appendParagraphsBold(container, text) {
      const bub = document.createElement("div");
      bub.className = "lt-msg-bubble";
      const segments = text.split(/(\*\*[^*]+\*\*)/g);
      for (const seg of segments) {
        if (seg.startsWith("**") && seg.endsWith("**")) {
          const strong = document.createElement("strong");
          strong.textContent = seg.slice(2, -2);
          bub.appendChild(strong);
        } else {
          bub.appendChild(document.createTextNode(seg));
        }
      }
      container.appendChild(bub);
    }

    function appendTutor(text, { persist = true } = {}) {
      const wrap = document.createElement("div");
      wrap.className = "lt-msg lt-msg--tutor";

      // Split on mermaid fences. Even indices = text, odd indices = mermaid code.
      const fence = /```mermaid\n([\s\S]*?)\n```/g;
      let lastIndex = 0;
      let m;
      let any = false;
      while ((m = fence.exec(text)) !== null) {
        any = true;
        const before = text.slice(lastIndex, m.index);
        if (before.trim()) appendParagraphsBold(wrap, before);
        const host = document.createElement("div");
        host.className = "lt-mermaid-host";
        wrap.appendChild(host);
        // Fire-and-forget; the placeholder appears synchronously.
        renderMermaidInto(host, m[1]);
        lastIndex = m.index + m[0].length;
      }
      const tail = text.slice(lastIndex);
      if (!any || tail.trim()) {
        appendParagraphsBold(wrap, tail);
      }

      log.appendChild(wrap);
      scrollToBottom();
      if (persist) {
        history.push({ role: "tutor", text });
        saveHistory(cfg, history);
      }
    }

    let thinkingEl = null;

    function appendThinking() {
      if (thinkingEl) return;
      const wrap = document.createElement("div");
      wrap.className = "lt-thinking";
      wrap.setAttribute("aria-label", "Thinking");
      for (let i = 0; i < 3; i++) {
        const d = document.createElement("span");
        d.className = "lt-dot-bounce";
        wrap.appendChild(d);
      }
      thinkingEl = wrap;
      log.appendChild(wrap);
      scrollToBottom();
    }

    function removeThinking() {
      if (thinkingEl) {
        thinkingEl.remove();
        thinkingEl = null;
      }
    }

    // ── Open / close ─────────────────────────────────────────────────────────
    function openPanel() {
      panelOpen = true;
      panel.classList.remove("lt-panel--hidden");
      bubble.setAttribute("aria-label", "Close Lab Tutor");
      dot.classList.add("lt-dot--hidden");

      if (!welcomeShown) {
        welcomeShown = true;
        // Hydrate from localStorage on first open
        const saved = loadHistory(cfg);
        if (saved.length > 0) {
          history = saved;
          for (const msg of saved) {
            if (msg.role === "user") {
              appendUser(msg.text, { persist: false });
            } else {
              appendTutor(msg.text, { persist: false });
            }
          }
          // Returning learner — skip the welcome card and chips
        } else {
          log.appendChild(buildWelcomeCard());
          scrollToBottom();
        }
      }

      input.focus();
    }

    function closePanel() {
      panelOpen = false;
      panel.classList.add("lt-panel--hidden");
      bubble.setAttribute("aria-label", "Open Lab Tutor");
    }

    // ── Send message ──────────────────────────────────────────────────────────
    // When called with no argument, reads from the input field (normal user flow).
    // When called with a string, uses that text directly (programmatic / intercept flow).
    async function sendMessage(externalText) {
      const text = (externalText !== undefined ? externalText : input.value).trim();
      if (!text || thinking) return;

      input.value = "";
      sendBtn.disabled = true;
      thinking = true;

      appendUser(text);
      appendThinking();

      try {
        const res = await fetch(cfg.baseUrl + "/v1/tutor/chat", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            session_id: cfg.sessionId,
            message: text,
            assignment_title: cfg.assignmentTitle || null,
          }),
        });

        removeThinking();

        if (!res.ok) {
          let detail = "Something went wrong. Please try again.";
          try {
            const payload = await res.json();
            if (payload?.detail) {
              detail = typeof payload.detail === "string"
                ? payload.detail
                : JSON.stringify(payload.detail);
            }
          } catch (_) {
            // keep default
          }
          appendTutor("⚠️ " + detail, { persist: false });
        } else {
          const data = await res.json();
          appendTutor(data.reply || "(no reply)");
        }
      } catch (_err) {
        removeThinking();
        appendTutor(
          "⚠️ Couldn’t reach the tutor. Check your connection and try again.",
          { persist: false }
        );
      } finally {
        thinking = false;
        sendBtn.disabled = false;
        input.focus();
      }
    }

    // ── Events ────────────────────────────────────────────────────────────────
    bubble.addEventListener("click", () => {
      if (panelOpen) closePanel();
      else openPanel();
    });

    closeBtn.addEventListener("click", closePanel);
    sendBtn.addEventListener("click", sendMessage);

    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    const escHandler = (e) => {
      if (e.key === "Escape" && panelOpen) closePanel();
    };
    document.addEventListener("keydown", escHandler);

    // ── Public API ────────────────────────────────────────────────────────────
    return {
      mount() {
        if (!document.body.contains(root)) {
          document.body.appendChild(root);
        }
        root.style.display = "";
        console.info("[lab-tutor] persistence key:", storageKey(cfg));
      },
      unmount() {
        closePanel();
        root.style.display = "none";
      },
      open: openPanel,
      ask: sendMessage,
      update(newCfg) {
        if (newCfg.assignmentTitle !== undefined) {
          cfg.assignmentTitle = newCfg.assignmentTitle;
          headerTitle.textContent = newCfg.assignmentTitle || "Lab Tutor";
        }
        if (newCfg.sessionId !== undefined) {
          cfg.sessionId = newCfg.sessionId;
        }
        if (newCfg.baseUrl !== undefined) {
          cfg.baseUrl = newCfg.baseUrl;
        }
        if (newCfg.enrollmentId !== undefined) {
          cfg.enrollmentId = newCfg.enrollmentId;
        }
      },
      destroy() {
        document.removeEventListener("keydown", escHandler);
        root.remove();
      },
    };
  }

  // ── Boot ──────────────────────────────────────────────────────────────────
  // Create the widget singleton and expose the programmatic API.
  // If we were loaded via a <script data-*> tag (standalone mode), mount immediately.
  // If loaded programmatically from lms.js, the caller calls window.__labTutorMount().
  let widget = null;

  function ensureWidget() {
    if (!widget) {
      widget = createWidget();
    }
    return widget;
  }

  // Standalone mode: script tag has data attributes → mount immediately
  const isStandalone = me && (
    me.dataset.assignmentTitle !== undefined ||
    me.dataset.sessionId !== undefined
  );

  if (isStandalone) {
    if (document.body) {
      ensureWidget().mount();
    } else {
      document.addEventListener("DOMContentLoaded", () => ensureWidget().mount());
    }
  }

  // Programmatic API for lms.js SPA integration
  window.__labTutorMount = function (cfg) {
    const w = ensureWidget();
    if (cfg) w.update(cfg);
    w.mount();
  };

  window.__labTutorUnmount = function () {
    if (widget) widget.unmount();
  };

  window.__labTutorUpdate = function (cfg) {
    if (widget) widget.update(cfg);
  };

  // Open the tutor panel (mount first if needed). Safe to call repeatedly.
  window.__labTutorOpen = function () {
    const w = ensureWidget();
    w.mount();
    w.open();
  };

  // Programmatically submit a message as if the learner typed it and pressed Enter.
  // The text is persisted to history and the /v1/tutor/chat flow fires normally.
  window.__labTutorAskAs = function (prompt) {
    const w = ensureWidget();
    w.mount();
    w.open();
    w.ask(prompt);
  };

  // ── Agent panel intercept ─────────────────────────────────────────────────
  // Hooks code-server's "Build with Agent" input. On every submit the prompt
  // is sent to POST /v1/tutor/triage. The judge returns "tutor" (broad,
  // do-the-assignment-for-me prompts) or "agent" (focused tool use). We
  // route accordingly; fail-open to "agent" on any error.
  (function setupAgentIntercept() {
    let replaying = false; // true while we are synthetically replaying to the agent
    let triagePending = false; // debounce: one in-flight triage at a time

    const SELECTORS = [
      'textarea[placeholder*="Describe what to build" i]',
      '[contenteditable="true"][placeholder*="Describe what to build" i]',
      '[aria-label*="Describe what to build" i]',
      '[aria-label*="agent" i][role="textbox"]',
      '.chat-input-container textarea',
    ];

    function findAgentInput() {
      for (const sel of SELECTORS) {
        const el = document.querySelector(sel);
        if (el && el.offsetParent !== null) return { el, sel };
      }
      return null;
    }

    function readPrompt(input) {
      // 1. Plain form inputs
      if (input.tagName === "TEXTAREA" || input.tagName === "INPUT") {
        return input.value || "";
      }
      // 2. Standard contenteditable — try the outer element's text first
      const directText = (input.innerText || input.textContent || "").trim();
      if (directText) return directText;
      // 3. Monaco / VS Code editor: visible content lives in .view-lines
      //    (each line is a .view-line element). innerText of the wrapper
      //    is empty because Monaco renders into a virtual viewport.
      const viewLines = input.querySelector(".view-lines");
      if (viewLines) {
        const lines = Array.from(viewLines.querySelectorAll(".view-line"))
          .map((l) => l.innerText || l.textContent || "");
        const joined = lines.join("\n").trim();
        if (joined) return joined;
      }
      // 4. Any inner contenteditable
      const editable = input.querySelector('[contenteditable="true"]');
      if (editable) {
        const t = (editable.innerText || editable.textContent || "").trim();
        if (t) return t;
      }
      // 5. As a last resort, ask the Monaco JS API directly. We pick the
      //    editor whose DOM node is contained by our hooked element.
      if (typeof window.monaco !== "undefined" && window.monaco.editor && window.monaco.editor.getEditors) {
        try {
          for (const ed of window.monaco.editor.getEditors()) {
            const dom = ed.getDomNode && ed.getDomNode();
            if (dom && input.contains(dom) && ed.getValue) {
              const v = ed.getValue();
              if (v) return v;
            }
          }
        } catch (_e) { /* ignore */ }
      }
      return "";
    }

    function clearPrompt(input) {
      if (input.tagName === "TEXTAREA" || input.tagName === "INPUT") {
        input.value = "";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      } else {
        input.textContent = "";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      }
    }

    function findSendButton(input) {
      return (
        input.closest("form")?.querySelector('button[type="submit"]')
        || input.parentElement?.querySelector('button[aria-label*="send" i]')
        || input.parentElement?.querySelector('button[title*="send" i]')
        || input.parentElement?.parentElement?.querySelector('button[aria-label*="send" i]')
        || null
      );
    }

    function replayToAgent(input) {
      // Synthetically re-submit so the agent handles it normally.
      replaying = true;
      const btn = findSendButton(input);
      if (btn) {
        btn.click();
      } else {
        input.dispatchEvent(
          new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true })
        );
      }
      setTimeout(() => { replaying = false; }, 250);
    }

    async function intercept(prompt, input) {
      if (triagePending) return; // already classifying a prompt — skip
      triagePending = true;

      let action = "agent";
      let reason = "triage skipped";

      try {
        const cfg = (widget && widget._cfg) || scriptCfg;
        const sessionId = (cfg && cfg.sessionId) || scriptCfg.sessionId;
        const assignmentTitle = (cfg && cfg.assignmentTitle) || scriptCfg.assignmentTitle || null;
        const baseUrl = (cfg && cfg.baseUrl) || scriptCfg.baseUrl || "";

        const res = await fetch(baseUrl + "/v1/tutor/triage", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, prompt, assignment_title: assignmentTitle }),
          signal: AbortSignal.timeout ? AbortSignal.timeout(15000) : undefined,
        });

        if (res.ok) {
          const data = await res.json();
          action = (data.action === "tutor" || data.action === "agent") ? data.action : "agent";
          reason = data.reason || "";
          console.info("[lab-tutor] triage:", action, "—", reason);
        } else {
          console.warn("[lab-tutor] triage HTTP error", res.status, "— defaulting to agent");
          action = "agent";
        }
      } catch (err) {
        console.warn("[lab-tutor] triage fetch failed:", err, "— defaulting to agent");
        action = "agent";
      } finally {
        triagePending = false;
      }

      if (action === "tutor") {
        // Route to the tutor widget and clear the agent input.
        if (typeof window.__labTutorOpen === "function") window.__labTutorOpen();
        if (typeof window.__labTutorAskAs === "function") window.__labTutorAskAs(prompt);
        clearPrompt(input);
      } else {
        // Let the agent answer — replay the submit with the original text intact.
        replayToAgent(input);
      }
    }

    // Track the currently-hooked agent input so the document/window-level
    // listeners know which element they're guarding.
    let hookedInput = null;

    function isInsideAgent(target) {
      if (!hookedInput || !target) return false;
      if (target === hookedInput) return true;
      if (typeof target.contains === "function" && hookedInput.contains(target)) return true;
      // contenteditable / Monaco sometimes routes events to an inner node;
      // walk the path upward.
      let n = target;
      while (n) {
        if (n === hookedInput) return true;
        n = n.parentElement;
      }
      return false;
    }

    function maybeIntercept(reason) {
      if (replaying) {
        console.info("[lab-tutor] skipping — replaying");
        return false;
      }
      if (!hookedInput) {
        console.info("[lab-tutor] skipping — no hooked input");
        return false;
      }
      const prompt = readPrompt(hookedInput).trim();
      if (prompt.length < 1) {
        console.warn(
          "[lab-tutor] empty prompt — readPrompt couldn't extract text from the agent input.",
          "Element:", hookedInput,
          "outerHTML preview:", (hookedInput.outerHTML || "").slice(0, 200)
        );
        return false;
      }
      console.info("[lab-tutor] intercepting via", reason, "— prompt:", prompt.slice(0, 60));
      // intercept is async; fire-and-forget (errors are caught inside).
      intercept(prompt, hookedInput).catch((err) => {
        console.warn("[lab-tutor] intercept unexpected error:", err);
      });
      return true;
    }

    function hookInput(input, selector) {
      if (input.__labTutorHooked) return;
      input.__labTutorHooked = true;
      hookedInput = input;
      console.info("[lab-tutor] hooked agent input via", selector);

      // Capture-phase keydown at the document level — fires BEFORE any
      // handler bound inside the editor (Monaco/CodeMirror typically bind
      // on their own root or on window without capture).
      document.addEventListener("keydown", (e) => {
        if (replaying) return;
        if (e.key !== "Enter" || e.shiftKey || e.isComposing) return;
        if (!isInsideAgent(e.target)) return;
        console.info("[lab-tutor] caught Enter on agent input");
        e.preventDefault();
        e.stopImmediatePropagation();
        e.stopPropagation();
        maybeIntercept("Enter");
      }, true);

      // contenteditable editors fire `beforeinput` with insertParagraph
      // for the Enter key. Catch that too — some implementations swallow
      // the keydown.
      document.addEventListener("beforeinput", (e) => {
        if (replaying) return;
        if (e.inputType !== "insertParagraph" && e.inputType !== "insertLineBreak") return;
        if (!isInsideAgent(e.target)) return;
        console.info("[lab-tutor] caught beforeinput", e.inputType);
        e.preventDefault();
        e.stopImmediatePropagation();
        e.stopPropagation();
        maybeIntercept("beforeinput");
      }, true);

      // Send-button click — broad search at intercept time so we tolerate
      // the button being re-rendered or appearing later.
      document.addEventListener("click", (e) => {
        if (replaying) return;
        const btn = e.target && e.target.closest && e.target.closest("button");
        if (!btn) return;
        const label = (btn.getAttribute("aria-label") || btn.getAttribute("title") || btn.textContent || "").toLowerCase();
        if (!/send|submit|build/.test(label)) return;
        // Only intercept if the click is near (or inside) the agent input's container.
        const root = hookedInput && hookedInput.closest('[role="textbox"], form, .chat-input-container');
        if (root && root.contains(btn)) {
          // ok, this is the agent's send button
        } else if (hookedInput && hookedInput.parentElement && hookedInput.parentElement.parentElement && hookedInput.parentElement.parentElement.contains(btn)) {
          // ok
        } else {
          return;
        }
        const prompt = readPrompt(hookedInput).trim();
        if (prompt.length < 1) return;
        console.info("[lab-tutor] caught send-button click — label:", label);
        e.preventDefault();
        e.stopImmediatePropagation();
        e.stopPropagation();
        maybeIntercept("click");
      }, true);
    }

    function tryHook() {
      const m = findAgentInput();
      if (m) hookInput(m.el, m.sel);
    }

    tryHook();
    const observer = new MutationObserver(() => tryHook());
    observer.observe(document.body, { childList: true, subtree: true });
  })();
})();
