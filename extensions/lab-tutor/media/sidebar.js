(function () {
  const vscode = acquireVsCodeApi();
  const log = document.getElementById("log");
  const input = document.getElementById("input");
  const send = document.getElementById("send");

  function append(role, text, parseBold = false) {
    const el = document.createElement("div");
    el.className = "msg " + role;
    if (parseBold) {
      // Render **bold** without using innerHTML.
      const parts = text.split(/(\*\*[^*]+\*\*)/g);
      for (const part of parts) {
        if (part.startsWith("**") && part.endsWith("**")) {
          const strong = document.createElement("strong");
          strong.textContent = part.slice(2, -2);
          el.appendChild(strong);
        } else {
          el.appendChild(document.createTextNode(part));
        }
      }
    } else {
      el.textContent = text;
    }
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
    if (m.type === "welcome") append("tutor", m.text, true);
    else if (m.type === "reply") append("tutor", m.text);
    else if (m.type === "error") append("tutor", "Error: " + m.text);
  });
})();
