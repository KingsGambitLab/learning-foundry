(function () {
  const vscode = acquireVsCodeApi();
  const log = document.getElementById("log");
  const input = document.getElementById("input");
  const send = document.getElementById("send");
  const dot = document.getElementById("dot");
  const subtitle = document.getElementById("subtitle");

  // State machine: engaging vs idle
  let engagingTimer = null;

  function setEngaging() {
    document.body.dataset.state = "engaging";
    dot.title = "engaging";
    if (engagingTimer !== null) {
      clearTimeout(engagingTimer);
      engagingTimer = null;
    }
  }

  function scheduleIdle() {
    if (engagingTimer !== null) clearTimeout(engagingTimer);
    engagingTimer = setTimeout(function () {
      document.body.dataset.state = "idle";
      dot.title = "idle";
      engagingTimer = null;
    }, 60000);
  }

  // Bold text parser — no innerHTML
  function parseBold(container, text) {
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    for (const part of parts) {
      if (part.startsWith("**") && part.endsWith("**")) {
        const strong = document.createElement("strong");
        strong.textContent = part.slice(2, -2);
        container.appendChild(strong);
      } else {
        container.appendChild(document.createTextNode(part));
      }
    }
  }

  function appendMsg(role, text, parseBoldText) {
    const el = document.createElement("div");
    el.className = "msg " + role;
    if (parseBoldText) {
      parseBold(el, text);
    } else {
      el.textContent = text;
    }
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
    return el;
  }

  function appendWelcome(text) {
    // Split on first blank line to get lead vs body
    const parts = text.split(/\n\n([\s\S]*)/);
    const leadText = parts[0];
    const bodyText = parts[1] || "";

    const card = document.createElement("div");
    card.className = "welcome-card";

    const lead = document.createElement("span");
    lead.className = "lead";
    parseBold(lead, leadText);
    card.appendChild(lead);

    if (bodyText) {
      const body = document.createElement("span");
      body.className = "body";
      body.textContent = bodyText;
      card.appendChild(body);
    }

    log.appendChild(card);
    log.scrollTop = log.scrollHeight;

    // Render quick-start chips after the welcome card
    appendChips();
  }

  const CHIPS = ["I'm stuck", "Why isn't this working?", "Walk me through the design"];

  function appendChips() {
    const container = document.createElement("div");
    container.id = "chips";

    for (const label of CHIPS) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip";
      btn.textContent = label;
      btn.addEventListener("click", function () {
        input.value = btn.textContent.trim();
        input.focus();
      });
      container.appendChild(btn);
    }

    log.appendChild(container);
    log.scrollTop = log.scrollHeight;
  }

  function submit() {
    const text = input.value.trim();
    if (!text) return;

    // Remove chips if still visible
    const chips = document.getElementById("chips");
    if (chips) chips.remove();

    appendMsg("user", text, false);
    vscode.postMessage({ type: "send", text });
    input.value = "";

    // Switch to engaging state
    setEngaging();
  }

  send.addEventListener("click", submit);
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter") submit();
  });

  window.addEventListener("message", function (event) {
    const m = event.data;
    if (m.type === "welcome") {
      appendWelcome(m.text);
    } else if (m.type === "subtitle") {
      if (m.text) {
        subtitle.textContent = m.text;
      }
    } else if (m.type === "reply") {
      appendMsg("tutor", m.text, true);
      // Start 60s idle timer after reply arrives
      scheduleIdle();
    } else if (m.type === "error") {
      appendMsg("tutor", "Error: " + m.text, false);
      scheduleIdle();
    }
  });
})();
