(() => {
  const stateElement = document.getElementById("draft-timeline-state");
  const state = stateElement?.textContent ? JSON.parse(stateElement.textContent) : {};

  const message = document.getElementById("timeline-message");
  const title = document.getElementById("timeline-title");
  const subtitle = document.getElementById("timeline-subtitle");
  const draftIdInput = document.getElementById("timeline-draft-id");
  const loadButton = document.getElementById("timeline-load-button");
  const refreshToggle = document.getElementById("timeline-refresh-toggle");
  const refreshNow = document.getElementById("timeline-refresh-now");
  const refreshStatus = document.getElementById("timeline-refresh-status");
  const refreshCopy = document.getElementById("timeline-refresh-copy");
  const meta = document.getElementById("timeline-meta");
  const summaryGrid = document.getElementById("timeline-summary-grid");
  const stream = document.getElementById("timeline-stream");
  const backLink = document.getElementById("timeline-back-link");
  const autoRefreshIntervalMs = 5000;
  let autoRefreshHandle = null;
  let lastLoadedDraftId = null;
  let loadInFlight = false;

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function setMessage(kind, text) {
    message.className = "message";
    if (!text) {
      message.textContent = "";
      return;
    }
    message.textContent = text;
    message.classList.add("visible", kind);
  }

  function pill(text) {
    return `<span class="small-pill">${escapeHtml(text)}</span>`;
  }

  function formatDate(value) {
    try {
      return new Date(value).toLocaleString();
    } catch (_error) {
      return value || "";
    }
  }

  function extractDetail(response, fallback = "Request failed.") {
    return response.text().then((text) => {
      try {
        const payload = JSON.parse(text);
        return payload.detail || payload.message || fallback;
      } catch (_error) {
        return text || fallback;
      }
    });
  }

  function readDraftIdFromUrl() {
    const url = new URL(window.location.href);
    return url.searchParams.get("draft");
  }

  function writeDraftIdToUrl(draftId) {
    const url = new URL(window.location.href);
    if (draftId) {
      url.searchParams.set("draft", draftId);
    } else {
      url.searchParams.delete("draft");
    }
    window.history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
  }

  function readLiveFromUrl() {
    const url = new URL(window.location.href);
    return url.searchParams.get("live") === "1";
  }

  function writeLiveToUrl(enabled) {
    const url = new URL(window.location.href);
    if (enabled) {
      url.searchParams.set("live", "1");
    } else {
      url.searchParams.delete("live");
    }
    window.history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
  }

  function renderSummary(timeline) {
    const run = timeline.course_run;
    const items = timeline.items || [];
    const latest = items.length ? items[items.length - 1] : null;
    summaryGrid.innerHTML = [
      stat("Draft id", run.id),
      stat("Stage", run.stage),
      stat("Status", run.status),
      stat("Deliverables", String(run.deliverable_count)),
      stat("Timeline items", String(items.length)),
      stat("Latest activity", latest ? formatDate(latest.created_at) : "No activity yet"),
      stat("Shared workflow", timeline.shared_workflow_run_id || "None"),
      stat("Linked workflows", String((timeline.linked_workflow_run_ids || []).length)),
    ].join("");
  }

  function stat(label, value) {
    return `
      <div class="review-item timeline-stat">
        <span class="timeline-stat-label">${escapeHtml(label)}</span>
        <span class="timeline-stat-value">${escapeHtml(value)}</span>
      </div>
    `;
  }

  function renderItems(timeline) {
    const items = timeline.items || [];
    if (!items.length) {
      stream.innerHTML = `<div class="review-item"><p>No stored activity yet for this draft.</p></div>`;
      return;
    }
    stream.innerHTML = items.map((item) => {
      const payload = item.payload && Object.keys(item.payload).length
        ? `<details><summary>View payload</summary><pre>${escapeHtml(JSON.stringify(item.payload, null, 2))}</pre></details>`
        : "";
      return `
        <article class="timeline-card ${escapeHtml(item.source_kind.replaceAll("_", "-"))}">
          <div class="timeline-card-header">
            <div>
              <h3>${escapeHtml(item.title)}</h3>
              <div class="timeline-card-meta">
                ${pill(item.source_kind.replaceAll("_", " "))}
                ${pill(item.source_title)}
                ${item.stage ? pill(`stage: ${item.stage}`) : ""}
                ${item.status ? pill(`status: ${item.status}`) : ""}
                ${item.sequence_no ? pill(`seq ${item.sequence_no}`) : ""}
                ${item.attempt ? pill(`attempt ${item.attempt}`) : ""}
              </div>
            </div>
            <span class="status-pill">${escapeHtml(formatDate(item.created_at))}</span>
          </div>
          ${item.detail ? `<p class="timeline-card-detail">${escapeHtml(item.detail)}</p>` : ""}
          ${payload}
        </article>
      `;
    }).join("");
  }

  function setAutoRefreshUi(enabled) {
    refreshStatus.textContent = enabled ? "Auto refresh on" : "Auto refresh off";
    refreshToggle.textContent = enabled ? "Pause auto refresh" : "Start auto refresh";
    refreshCopy.textContent = enabled
      ? "Refreshing every 5 seconds while this tab is open so you can follow agent progress live."
      : "Turn on auto refresh to follow agent progress while the draft is building or reviewing.";
  }

  function stopAutoRefresh() {
    if (autoRefreshHandle !== null) {
      window.clearInterval(autoRefreshHandle);
      autoRefreshHandle = null;
    }
    setAutoRefreshUi(false);
    writeLiveToUrl(false);
  }

  function startAutoRefresh() {
    if (!lastLoadedDraftId && !draftIdInput.value.trim()) {
      setMessage("error", "Load a draft before starting auto refresh.");
      return;
    }
    if (autoRefreshHandle !== null) {
      window.clearInterval(autoRefreshHandle);
    }
    autoRefreshHandle = window.setInterval(() => {
      if (document.hidden) return;
      const draftId = lastLoadedDraftId || draftIdInput.value.trim();
      if (!draftId || loadInFlight) return;
      loadTimeline(draftId, { background: true });
    }, autoRefreshIntervalMs);
    setAutoRefreshUi(true);
    writeLiveToUrl(true);
  }

  async function loadTimeline(draftId, options = {}) {
    if (!draftId) {
      setMessage("error", "Enter a draft id first.");
      return;
    }
    const urlTemplate = state.timeline_url_template || "/v1/course-runs/{course_run_id}/timeline";
    if (loadInFlight) return;
    loadInFlight = true;
    loadButton.disabled = true;
    refreshNow.disabled = true;
    if (!options.background) {
      setMessage("info", "Loading draft timeline…");
    }
    try {
      const response = await fetch(urlTemplate.replace("{course_run_id}", encodeURIComponent(draftId)));
      if (!response.ok) {
        throw new Error(await extractDetail(response, "Could not load the draft timeline."));
      }
      const timeline = await response.json();
      title.textContent = timeline.course_run.title || "Draft timeline";
      subtitle.textContent = `Showing course events, workflow authoring activity, workflow events, and node executions for ${timeline.course_run.id}.`;
      meta.innerHTML = [
        pill(`updated ${formatDate(timeline.course_run.updated_at)}`),
        pill(`AI spend ${Number(timeline.course_run.ai_usage?.estimated_cost_usd || 0).toFixed(4)} USD`),
      ].join("");
      renderSummary(timeline);
      renderItems(timeline);
      backLink.href = `${state.dashboard_url || "/create-course"}?draft=${encodeURIComponent(timeline.course_run.id)}&tab=drafts`;
      draftIdInput.value = timeline.course_run.id;
      writeDraftIdToUrl(timeline.course_run.id);
       lastLoadedDraftId = timeline.course_run.id;
      document.title = `${timeline.course_run.title} · Draft Timeline`;
      if (!options.background) {
        setMessage("success", `Loaded ${timeline.items.length} timeline item${timeline.items.length === 1 ? "" : "s"}.`);
      }
    } catch (error) {
      setMessage("error", error instanceof Error ? error.message : "Could not load the draft timeline.");
      summaryGrid.innerHTML = `<div class="review-item"><p>Timeline unavailable.</p></div>`;
      stream.innerHTML = `<div class="review-item"><p>We could not load this draft.</p></div>`;
    } finally {
      loadButton.disabled = false;
      refreshNow.disabled = false;
      loadInFlight = false;
    }
  }

  loadButton?.addEventListener("click", () => {
    loadTimeline(draftIdInput.value.trim());
  });

  draftIdInput?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      loadTimeline(draftIdInput.value.trim());
    }
  });

  refreshToggle?.addEventListener("click", () => {
    if (autoRefreshHandle !== null) {
      stopAutoRefresh();
      return;
    }
    startAutoRefresh();
  });

  refreshNow?.addEventListener("click", () => {
    const draftId = draftIdInput.value.trim();
    loadTimeline(draftId);
  });

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && autoRefreshHandle !== null) {
      const draftId = lastLoadedDraftId || draftIdInput.value.trim();
      if (draftId && !loadInFlight) {
        loadTimeline(draftId, { background: true });
      }
    }
  });

  const initialDraftId = state.draft_id || readDraftIdFromUrl();
  setAutoRefreshUi(false);
  if (initialDraftId) {
    draftIdInput.value = initialDraftId;
    loadTimeline(initialDraftId);
    if (readLiveFromUrl()) {
      startAutoRefresh();
    }
  } else {
    setMessage("info", "Paste a draft id to inspect its flow.");
  }
})();
