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

    function appendTutor(text, { persist = true } = {}) {
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
})();
