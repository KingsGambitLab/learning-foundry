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
