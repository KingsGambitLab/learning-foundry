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
  };

  // ── Inject stylesheet once ────────────────────────────────────────────────
  if (!document.querySelector('link[href*="lab-tutor.css"]')) {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = (scriptCfg.baseUrl || "") + "/static/lab-tutor.css";
    document.head.appendChild(link);
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

  // ── Widget factory ────────────────────────────────────────────────────────
  // Called once; returns mount/unmount/update handles.
  function createWidget(initialCfg) {
    let cfg = Object.assign({}, scriptCfg, initialCfg || {});
    let panelOpen = false;
    let welcomeShown = false;
    let thinking = false;

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

    const closeBtn = document.createElement("button");
    closeBtn.className = "lt-close";
    closeBtn.type = "button";
    closeBtn.setAttribute("aria-label", "Close Lab Tutor");
    closeBtn.appendChild(closeIcon());

    header.appendChild(headerTitle);
    header.appendChild(closeBtn);

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

    function appendUser(text) {
      const wrap = document.createElement("div");
      wrap.className = "lt-msg lt-msg--user";
      const bub = document.createElement("div");
      bub.className = "lt-msg-bubble";
      bub.textContent = text;
      wrap.appendChild(bub);
      log.appendChild(wrap);
      scrollToBottom();
    }

    function appendTutor(text) {
      const wrap = document.createElement("div");
      wrap.className = "lt-msg lt-msg--tutor";
      const bub = document.createElement("div");
      bub.className = "lt-msg-bubble";
      // Parse **bold** segments without using innerHTML on untrusted content
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
      wrap.appendChild(bub);
      log.appendChild(wrap);
      scrollToBottom();
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
        log.appendChild(buildWelcomeCard());
        scrollToBottom();
      }

      input.focus();
    }

    function closePanel() {
      panelOpen = false;
      panel.classList.add("lt-panel--hidden");
      bubble.setAttribute("aria-label", "Open Lab Tutor");
    }

    // ── Send message ──────────────────────────────────────────────────────────
    async function sendMessage() {
      const text = input.value.trim();
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
          appendTutor("⚠️ " + detail);
        } else {
          const data = await res.json();
          appendTutor(data.reply || "(no reply)");
        }
      } catch (_err) {
        removeThinking();
        appendTutor(
          "⚠️ Couldn’t reach the tutor. Check your connection and try again."
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
      },
      unmount() {
        closePanel();
        root.style.display = "none";
      },
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

  // ── Shared panel helpers exposed on window ────────────────────────────────
  // These are used by the agent rehearsal interception block below.
  // They call into the widget's internal openPanel / appendTutor functions
  // by routing through the createWidget return value's internal closures.
  // We expose them lazily so they work regardless of whether the widget
  // was mounted via standalone or programmatic mode.

  window.__labTutorOpen = function () {
    const w = ensureWidget();
    // Access the internal openPanel via mount (which calls it on next open)
    // We need to reach inside — call mount() which ensures the root is in DOM,
    // then trigger openPanel via the bubble click simulation.
    w.mount();
    // Trigger open if not already open by dispatching a click on the bubble.
    // The widget internally tracks panelOpen; we check via aria-label on the bubble.
    const bubble = document.querySelector(".lt-bubble");
    if (bubble && bubble.getAttribute("aria-label") === "Open Lab Tutor") {
      bubble.click();
    }
  };

  window.__labTutorAppendTutorMessage = function (text) {
    // Ensure the widget exists and is mounted before appending.
    ensureWidget().mount();
    // Find the log element directly in the DOM.
    const logEl = document.querySelector(".lt-log");
    if (!logEl) return;
    const wrap = document.createElement("div");
    wrap.className = "lt-msg lt-msg--tutor";
    const bub = document.createElement("div");
    bub.className = "lt-msg-bubble";
    // Parse **bold** segments and > quoted lines the same way appendTutor does.
    const lines = text.split("\n");
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (line.startsWith("> ")) {
        // Blockquote styling for > lines
        const blockquote = document.createElement("blockquote");
        blockquote.style.cssText = "margin:4px 0 4px 8px;padding-left:8px;border-left:3px solid rgba(255,255,255,0.3);opacity:0.85;font-style:italic;";
        blockquote.textContent = line.slice(2);
        bub.appendChild(blockquote);
      } else {
        // Parse **bold** segments
        const segments = line.split(/(\*\*[^*]+\*\*)/g);
        for (const seg of segments) {
          if (seg.startsWith("**") && seg.endsWith("**")) {
            const strong = document.createElement("strong");
            strong.textContent = seg.slice(2, -2);
            bub.appendChild(strong);
          } else {
            bub.appendChild(document.createTextNode(seg));
          }
        }
      }
      // Add line break between lines (but not after the last one)
      if (i < lines.length - 1) {
        bub.appendChild(document.createElement("br"));
      }
    }
    wrap.appendChild(bub);
    logEl.appendChild(wrap);
    logEl.scrollTop = logEl.scrollHeight;
  };

  // ---- Agent prompt-rehearsal interception ------------------------------
  (function setupAgentRehearsal() {
    let pending = null;
    let replaying = false;

    const SELECTORS = [
      // Most specific: placeholder text from Cursor/code-server "Build with Agent"
      'textarea[placeholder*="Describe what to build" i]',
      '[contenteditable="true"][placeholder*="Describe what to build" i]',
      '[aria-label*="Describe what to build" i]',
      '[aria-label*="agent" i][role="textbox"]',
      // Fallback: any chat panel input
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
      if (input.tagName === "TEXTAREA" || input.tagName === "INPUT") {
        return input.value || "";
      }
      return input.innerText || input.textContent || "";
    }

    function findSendButton(input) {
      const candidates = [
        input.closest('form') && input.closest('form').querySelector('button[type="submit"]'),
        input.parentElement && input.parentElement.querySelector('button[aria-label*="send" i]'),
        input.parentElement && input.parentElement.querySelector('button[title*="send" i]'),
        input.parentElement && input.parentElement.parentElement && input.parentElement.parentElement.querySelector('button[aria-label*="send" i]'),
      ];
      return candidates.find(Boolean) || null;
    }

    async function rehearse(prompt) {
      try {
        const cfg = ensureWidget()._cfg || scriptCfg;
        const res = await fetch(scriptCfg.baseUrl + "/v1/tutor/rehearse", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            session_id: scriptCfg.sessionId,
            prompt,
            assignment_title: scriptCfg.assignmentTitle || null,
          }),
        });
        if (!res.ok) return { verdict: "ok", message: "" };
        return await res.json();
      } catch (e) {
        return { verdict: "ok", message: "" };
      }
    }

    async function intercept(prompt, input, sendBtn) {
      if (replaying) return false;
      if (pending) return true; // an intercept is already in flight; swallow this submit
      pending = (async () => {
        try {
          const data = await rehearse(prompt);
          if (data.verdict === "rehearsal") {
            if (typeof window.__labTutorOpen === "function") window.__labTutorOpen();
            const quoted = prompt.length > 200 ? prompt.slice(0, 200) + "…" : prompt;
            const body = data.message + "\n\nYou were about to ask the agent:\n> " + quoted;
            if (typeof window.__labTutorAppendTutorMessage === "function") {
              window.__labTutorAppendTutorMessage(body);
            }
            return true;
          }
          // verdict ok: replay the submit
          replaying = true;
          try {
            if (sendBtn) {
              sendBtn.click();
            } else {
              // Best-effort fallback: dispatch a new Enter keydown
              input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true }));
            }
          } finally {
            // Reset on next tick so the replay listener sees it
            setTimeout(function () { replaying = false; }, 50);
          }
          return false;
        } finally {
          pending = null;
        }
      })();
      return pending;
    }

    function hookInput(input, selector) {
      if (input.__labTutorHooked) return;
      input.__labTutorHooked = true;
      console.info("[lab-tutor] hooked agent input via", selector);
      const sendBtn = findSendButton(input);

      input.addEventListener("keydown", async function (e) {
        if (replaying) return;
        if (e.key !== "Enter" || e.shiftKey) return;
        const prompt = readPrompt(input).trim();
        if (prompt.length < 5) return;
        e.preventDefault();
        e.stopImmediatePropagation();
        await intercept(prompt, input, sendBtn);
      }, true);

      if (sendBtn && !sendBtn.__labTutorHooked) {
        sendBtn.__labTutorHooked = true;
        sendBtn.addEventListener("click", async function (e) {
          if (replaying) return;
          const prompt = readPrompt(input).trim();
          if (prompt.length < 5) return;
          e.preventDefault();
          e.stopImmediatePropagation();
          await intercept(prompt, input, sendBtn);
        }, true);
      }
    }

    function tryHook() {
      const m = findAgentInput();
      if (m) hookInput(m.el, m.sel);
    }

    // Initial attempt + observer for lazy-mounted panels
    tryHook();
    const observer = new MutationObserver(function () { tryHook(); });
    observer.observe(document.body, { childList: true, subtree: true });
  })();
})();
