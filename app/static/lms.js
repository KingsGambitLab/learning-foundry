(() => {
  const stateElement = document.getElementById("lms-state");
  const state = stateElement?.textContent ? JSON.parse(stateElement.textContent) : {};

  const pageMessage = document.getElementById("page-message");
  const toastRegion = document.getElementById("toast-region");
  const learnerShell = document.getElementById("learner-shell");
  const learnView = document.getElementById("learn-view");
  const coursesView = document.getElementById("courses-view");
  const learnerFocus = document.getElementById("learner-focus");
  const deliverablesPanel = document.getElementById("deliverables-panel");
  const deliverablesTitle = document.getElementById("deliverables-title");
  const deliverablesCaption = document.getElementById("deliverables-caption");
  const deliverablesBody = document.getElementById("deliverables-body");
  const submissionHistory = document.getElementById("submission-history");
  const submissionHistoryBody = document.getElementById("submission-history-body");
  const enrollmentList = document.getElementById("enrollment-list");
  const catalogBody = document.getElementById("catalog-body");
  const catalogCaption = document.getElementById("catalog-caption");
  const catalogList = document.getElementById("catalog-list");
  const catalogToggle = document.getElementById("catalog-toggle");

  const currentUrl = new URL(window.location.href);
  const selectionFromUrl = currentUrl.searchParams.get("enrollment");
  const uiState = {
    currentExperience: null,
    currentEnrollmentId: selectionFromUrl,
    catalogExpanded: !((state.enrollments?.enrollments || []).length > 0),
    busyAction: null,
    busyTarget: null,
    pageMessage: null,
    workspaceFeedback: null,
    submissionFeedback: null,
    writeupExpanded: false,
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function titleCase(value) {
    return String(value || "")
      .replaceAll("_", " ")
      .replaceAll("-", " ")
      .replace(/\s+/g, " ")
      .trim()
      .replace(/\b\w/g, (char) => char.toUpperCase());
  }

  function formatDate(value) {
    if (!value) return "";
    try {
      return new Date(value).toLocaleString([], {
        month: "short",
        day: "numeric",
        year: "numeric",
        hour: "numeric",
        minute: "2-digit",
      });
    } catch (_error) {
      return value;
    }
  }

  function formatRelative(value) {
    if (!value) return "Recently";
    try {
      const date = new Date(value);
      const diffMs = date.getTime() - Date.now();
      const absSeconds = Math.round(Math.abs(diffMs) / 1000);
      const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
      if (absSeconds < 60) {
        return formatter.format(Math.round(diffMs / 1000), "second");
      }
      const absMinutes = Math.round(absSeconds / 60);
      if (absMinutes < 60) {
        return formatter.format(Math.round(diffMs / 60000), "minute");
      }
      const absHours = Math.round(absMinutes / 60);
      if (absHours < 48) {
        return formatter.format(Math.round(diffMs / 3600000), "hour");
      }
      const absDays = Math.round(absHours / 24);
      return formatter.format(Math.round(diffMs / 86400000), "day");
    } catch (_error) {
      return formatDate(value);
    }
  }

  function percent(value) {
    return `${Math.round((value || 0) * 100)}%`;
  }

  function existingEnrollmentForCourse(courseRunId) {
    return (state.enrollments?.enrollments || []).find((item) => item.course_run_id === courseRunId) || null;
  }

  function selectedEnrollmentSummary() {
    const enrollments = state.enrollments?.enrollments || [];
    if (uiState.currentEnrollmentId) {
      const selected = enrollments.find((item) => item.id === uiState.currentEnrollmentId);
      if (selected) return selected;
    }
    return enrollments[0] || null;
  }

  function computeCourseProgress(summaryOrExperience) {
    const deliverables = summaryOrExperience?.deliverables || [];
    const enrollment = summaryOrExperience?.enrollment || summaryOrExperience || {};
    const total = deliverables.length || Number(enrollment.deliverable_count || 0);
    const completed = deliverables.length
      ? deliverables.filter((deliverable) => deliverable.status === "passed").length
      : Number(enrollment.completed_deliverable_count || 0);
    const currentIndex = deliverables.length
      ? (
        deliverables.find((deliverable) => deliverable.deliverable_id === enrollment.current_deliverable_id)?.deliverable_index
        || 0
      )
      : Number(enrollment.current_deliverable_index || 0);
    const positionPercent = total
      ? Math.round(((enrollment.status === "completed" ? total : currentIndex || completed) / total) * 100)
      : 0;
    const completionPercent = total ? Math.round((completed / total) * 100) : 0;
    return {
      total,
      completed,
      currentIndex,
      positionPercent: total ? Math.max(0, Math.min(positionPercent, 100)) : 0,
      completionPercent: total ? Math.max(0, Math.min(completionPercent, 100)) : 0,
    };
  }

  function learnerReadyReason(course) {
    if (course.supported_for_lms) {
      return "Includes a shared learner workspace, visible checks, and assignment review.";
    }
    return course.support_reason || "This course is being prepared and is not ready for learners yet.";
  }

  function courseStatusCopy(summary) {
    const labels = {
      active: "In progress",
      completed: "Completed",
      blocked: "Blocked",
    };
    return labels[summary?.status] || titleCase(summary?.status);
  }

  function courseStatusKind(summary) {
    const kinds = {
      active: "in-progress",
      completed: "passed",
      blocked: "blocked",
    };
    return kinds[summary?.status] || "neutral";
  }

  function deliverableStatusCopy(deliverable, enrollment) {
    if (enrollment?.status === "completed" && deliverable.status === "passed") {
      return "Completed";
    }
    if (enrollment?.current_deliverable_id === deliverable.deliverable_id && enrollment?.status !== "completed") {
      return "In progress";
    }
    const labels = {
      available: "Ready now",
      passed: "Passed",
      locked: "Locked",
    };
    return labels[deliverable.status] || titleCase(deliverable.status);
  }

  function deliverableStatusKind(deliverable, enrollment) {
    if (enrollment?.current_deliverable_id === deliverable.deliverable_id && enrollment?.status !== "completed") {
      return "in-progress";
    }
    const kinds = {
      available: "ready",
      passed: "passed",
      locked: "locked",
    };
    return kinds[deliverable.status] || "neutral";
  }

  function workspaceStatusCopy(status) {
    const labels = {
      starting: "Starting",
      running: "Running",
      stopped: "Stopped",
      failed: "Failed",
    };
    return labels[status] || titleCase(status);
  }

  function workspaceStatusKind(status) {
    const kinds = {
      starting: "warning",
      running: "running",
      stopped: "stopped",
      failed: "failed",
    };
    return kinds[status] || "neutral";
  }

  function renderStatusPill(kind, label) {
    return `<span class="status-pill ${escapeHtml(kind)}">${escapeHtml(label)}</span>`;
  }

  function renderInfoPill(label, value) {
    return `<span class="info-pill"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></span>`;
  }

  function syncUrlState() {
    const url = new URL(window.location.href);
    if (uiState.currentEnrollmentId) {
      url.searchParams.set("enrollment", uiState.currentEnrollmentId);
    } else {
      url.searchParams.delete("enrollment");
    }
    url.searchParams.delete("tab");
    window.history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
  }

  const submitProgressSteps = [
    {
      title: "Starting review",
      message: "Preparing your shared project files for the hidden grader.",
      delayMs: 0,
    },
    {
      title: "Booting review sandbox",
      message: "Starting the app sandbox and wiring up the grader checks.",
      delayMs: 1500,
    },
    {
      title: "Running hidden checks",
      message: "Review is in progress now. First boots can take a little longer.",
      delayMs: 7000,
    },
    {
      title: "Still reviewing",
      message: "We are waiting for your project app to come up and finish the deliverable checks.",
      delayMs: 18000,
    },
  ];

  let submitProgressTimeout = null;

  function openWorkspaceUrl(url) {
    if (!url) {
      return false;
    }
    window.location.assign(url);
    return true;
  }

  function stopSubmitProgress() {
    if (submitProgressTimeout !== null) {
      window.clearTimeout(submitProgressTimeout);
      submitProgressTimeout = null;
    }
  }

  function advanceSubmitProgress(stepIndex) {
    const step = submitProgressSteps[stepIndex];
    if (!step || !isBusy("submit")) {
      return;
    }
    uiState.submissionFeedback = {
      kind: "info",
      title: step.title,
      message: step.message,
    };
    renderAll();
    const nextStep = submitProgressSteps[stepIndex + 1];
    if (!nextStep) {
      submitProgressTimeout = null;
      return;
    }
    const delay = Math.max(0, nextStep.delayMs - step.delayMs);
    submitProgressTimeout = window.setTimeout(() => advanceSubmitProgress(stepIndex + 1), delay);
  }

  function startSubmitProgress() {
    stopSubmitProgress();
    advanceSubmitProgress(0);
  }

  function setBusy(action, target = null) {
    uiState.busyAction = action;
    uiState.busyTarget = target;
  }

  function clearBusy() {
    uiState.busyAction = null;
    uiState.busyTarget = null;
  }

  function isBusy(action, target = null) {
    return uiState.busyAction === action && (target === null || uiState.busyTarget === target);
  }

  function setPageMessage(kind, text) {
    uiState.pageMessage = text ? { kind, text } : null;
    renderPageMessage();
  }

  function renderPageMessage() {
    if (!uiState.pageMessage) {
      pageMessage.className = "message";
      pageMessage.textContent = "";
      return;
    }
    pageMessage.className = `message visible ${uiState.pageMessage.kind}`;
    pageMessage.textContent = uiState.pageMessage.text;
  }

  function showToast(kind, title, text) {
    const toast = document.createElement("div");
    toast.className = `toast ${kind}`;
    toast.innerHTML = `<strong>${escapeHtml(title)}</strong><p>${escapeHtml(text)}</p>`;
    toastRegion.appendChild(toast);
    window.setTimeout(() => {
      toast.remove();
    }, 4800);
  }

  function updateDocumentTitle() {
    const experience = uiState.currentExperience;
    if (!experience?.enrollment) {
      document.title = "Learner LMS · Course Gen Codex";
      return;
    }
    const courseTitle = experience.enrollment.course_title || "Learner LMS";
    document.title = `${courseTitle} · Learner LMS`;
  }

  function renderChrome() {
    learnerShell.classList.add("no-sidebar");
  }

  async function readResponseError(response) {
    const raw = await response.text();
    let detail = raw || `Request failed (${response.status}).`;
    try {
      const parsed = raw ? JSON.parse(raw) : null;
      if (parsed && parsed.detail !== undefined) {
        detail = typeof parsed.detail === "string" ? parsed.detail : JSON.stringify(parsed.detail);
      }
    } catch (_error) {
      // Keep the raw text fallback when the payload is not JSON.
    }
    return {
      status: response.status,
      detail,
      raw,
    };
  }

  function normalizeError(action, error) {
    if (error && typeof error === "object" && "message" in error && typeof error.message === "string" && !("detail" in error)) {
      return { message: error.message, detail: error.message };
    }

    const detail = typeof error?.detail === "string"
      ? error.detail
      : (error?.message || "Something went wrong.");

    if (action === "workspace") {
      if (/timed out waiting/i.test(detail) || /connection refused/i.test(detail)) {
        return {
          message: "Workspace isn't running yet. We couldn't reach your VS Code session.",
          detail,
        };
      }
      return {
        message: "We couldn't start your workspace right now.",
        detail,
      };
    }

    if (action === "submit") {
      if (/timed out waiting/i.test(detail) || /connection refused/i.test(detail)) {
        return {
          message: "We couldn't reach your project app for review.",
          detail,
        };
      }
      return {
        message: "We couldn't submit this project for review right now.",
        detail,
      };
    }

    if (action === "enroll") {
      if (/not ready/i.test(detail) || /prepared/i.test(detail) || /learner/i.test(detail)) {
        return {
          message: "This course is still being prepared and is not ready for learners yet.",
          detail,
        };
      }
      return {
        message: "We couldn't create your enrollment right now.",
        detail,
      };
    }

    if (action === "load") {
      return {
        message: "We couldn't load this learner progress right now.",
        detail,
      };
    }

    if (action === "catalog") {
      return {
        message: "We couldn't refresh the published course catalog.",
        detail,
      };
    }

    if (action === "enrollments") {
      return {
        message: "We couldn't refresh your course list right now.",
        detail,
      };
    }

    return {
      message: detail,
      detail,
    };
  }

  function safeMarkdownHref(href) {
    const candidate = String(href || "").trim();
    if (!candidate) return null;
    if (candidate.startsWith("#") || candidate.startsWith("/")) return candidate;
    try {
      const parsed = new URL(candidate, window.location.origin);
      if (parsed.protocol === "http:" || parsed.protocol === "https:" || parsed.protocol === "mailto:") {
        return candidate;
      }
    } catch (_error) {
      return null;
    }
    return null;
  }

  function inlineFormat(text) {
    let out = text;
    out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
    out = out.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    out = out.replace(/__([^_\n]+)__/g, "<strong>$1</strong>");
    out = out.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
    out = out.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_match, label, href) => {
      const safeHref = safeMarkdownHref(href);
      if (!safeHref) return label;
      return `<a href="${safeHref}" target="_blank" rel="noopener noreferrer">${label}</a>`;
    });
    return out;
  }

  function renderMarkdown(markdown) {
    const source = escapeHtml(markdown || "No project brief yet.");
    const lines = source.replace(/\r\n/g, "\n").split("\n");
    const html = [];
    let paragraph = [];
    let listItems = [];
    let orderedItems = [];
    let codeBlock = [];
    let inCodeBlock = false;

    function flushParagraph() {
      if (!paragraph.length) return;
      const joined = paragraph.join(" ");
      const boldOnly = joined.match(/^\s*\*\*(.+)\*\*\s*:?\s*$/);
      if (boldOnly) {
        html.push(`<h4>${inlineFormat(boldOnly[1])}</h4>`);
      } else {
        html.push(`<p>${inlineFormat(joined)}</p>`);
      }
      paragraph = [];
    }

    function flushList() {
      if (!listItems.length) return;
      html.push(`<ul>${listItems.map((item) => `<li>${inlineFormat(item)}</li>`).join("")}</ul>`);
      listItems = [];
    }

    function flushOrderedList() {
      if (!orderedItems.length) return;
      html.push(`<ol>${orderedItems.map((item) => `<li>${inlineFormat(item)}</li>`).join("")}</ol>`);
      orderedItems = [];
    }

    function flushCodeBlock() {
      if (!codeBlock.length) return;
      html.push(`<pre><code>${codeBlock.join("\n")}</code></pre>`);
      codeBlock = [];
    }

    for (const rawLine of lines) {
      const line = rawLine.trimEnd();

      if (line.startsWith("```")) {
        flushParagraph();
        flushList();
        flushOrderedList();
        if (inCodeBlock) {
          flushCodeBlock();
        }
        inCodeBlock = !inCodeBlock;
        continue;
      }

      if (inCodeBlock) {
        codeBlock.push(line);
        continue;
      }

      const trimmed = line.trim();

      if (!trimmed) {
        flushParagraph();
        flushList();
        flushOrderedList();
        continue;
      }

      const headingMatch = trimmed.match(/^(#{1,4})\s+(.*)$/);
      if (headingMatch) {
        flushParagraph();
        flushList();
        flushOrderedList();
        const level = Math.min(4, headingMatch[1].length);
        html.push(`<h${level}>${inlineFormat(headingMatch[2])}</h${level}>`);
        continue;
      }

      const orderedMatch = trimmed.match(/^\d+\.\s+(.*)$/);
      if (orderedMatch) {
        flushParagraph();
        flushList();
        orderedItems.push(orderedMatch[1]);
        continue;
      }

      const listMatch = trimmed.match(/^[-*•]\s+(.*)$/);
      if (listMatch) {
        flushParagraph();
        flushOrderedList();
        listItems.push(listMatch[1]);
        continue;
      }

      paragraph.push(trimmed);
    }

    flushParagraph();
    flushList();
    flushOrderedList();
    flushCodeBlock();

    return html.join("");
  }

  function renderFeedbackBanner(feedback) {
    if (!feedback) return "";
    return `
      <div class="feedback-banner ${escapeHtml(feedback.kind || "info")}">
        <strong>${escapeHtml(feedback.title)}</strong>
        <p>${escapeHtml(feedback.message)}</p>
        ${feedback.detail ? `
          <details>
            <summary>View error details</summary>
            <pre>${escapeHtml(feedback.detail)}</pre>
          </details>
        ` : ""}
      </div>
    `;
  }

  function renderLearnerGuidance(feedback) {
    if (!feedback) return "";
    const strengths = Array.isArray(feedback.strengths) ? feedback.strengths.filter(Boolean) : [];
    const whyItMatters = Array.isArray(feedback.why_it_matters) ? feedback.why_it_matters.filter(Boolean) : [];
    const likelyRootCause = Array.isArray(feedback.likely_root_cause) ? feedback.likely_root_cause.filter(Boolean) : [];
    const investigationSteps = Array.isArray(feedback.investigation_steps) ? feedback.investigation_steps.filter(Boolean) : [];
    return `
      <details class="review-guidance" open>
        <summary>Tech lead feedback</summary>
        ${feedback.learner_feedback ? `<p class="review-guidance-summary">${escapeHtml(feedback.learner_feedback)}</p>` : ""}
        ${feedback.fundamental_gap ? `
          <div class="review-guidance-section">
            <h5>Fundamental gap</h5>
            <p>${escapeHtml(feedback.fundamental_gap)}</p>
          </div>
        ` : ""}
        ${strengths.length ? `
          <div class="review-guidance-section">
            <h5>What already looks strong</h5>
            <ul>${strengths.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
          </div>
        ` : ""}
        ${whyItMatters.length ? `
          <div class="review-guidance-section">
            <h5>Why it matters</h5>
            <ul>${whyItMatters.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
          </div>
        ` : ""}
        ${likelyRootCause.length ? `
          <div class="review-guidance-section">
            <h5>Likely root cause</h5>
            <ul>${likelyRootCause.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
          </div>
        ` : ""}
        ${investigationSteps.length ? `
          <div class="review-guidance-section">
            <h5>Where to investigate next</h5>
            <ol>${investigationSteps.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ol>
          </div>
        ` : ""}
      </details>
    `;
  }

  function renderWorkspaceStatusInline(experience) {
    const session = experience?.workspace_session;
    if (!experience) {
      return "";
    }

    let note = "Workspace opens the shared project with your saved edits.";
    if (session?.status === "running") {
      note = "Workspace is running with your saved files ready to resume.";
    } else if (session?.status === "starting") {
      note = "Workspace is starting up for the shared project.";
    } else if (session) {
      note = "Workspace is idle with your saved files ready to reopen.";
    }

    return `
      <div class="workspace-inline-meta">
        ${renderStatusPill(session ? workspaceStatusKind(session.status) : "neutral", session ? workspaceStatusCopy(session.status) : "Idle")}
        ${session?.host_port ? renderInfoPill("Port", String(session.host_port)) : ""}
        <span class="workspace-inline-note">${escapeHtml(note)}</span>
      </div>
    `;
  }

  function renderFocus() {
    const experience = uiState.currentExperience;
    const selectedSummary = selectedEnrollmentSummary();

    if (!experience && isBusy("open-enrollment")) {
      learnerFocus.innerHTML = `
        <div class="hero-copy">
          <p class="eyebrow">Continue where you left off</p>
          <h1>Loading your latest course...</h1>
          <p class="focus-subcopy">We are fetching your project brief, workspace status, and review history.</p>
        </div>
      `;
      return;
    }

    if (!experience && selectedSummary) {
      learnerFocus.innerHTML = `
        <div class="hero-copy">
          <p class="eyebrow">Continue where you left off</p>
          <h1>${escapeHtml(selectedSummary.course_title)}</h1>
          <p class="focus-subtitle">Loading your project brief...</p>
        </div>
      `;
      return;
    }

    if (!experience) {
      const catalog = state.catalog?.courses || [];
      const readyCourses = catalog.filter((course) => course.supported_for_lms).length;
      learnerFocus.innerHTML = `
        <div class="hero-copy empty-state">
          <p class="eyebrow">Learner LMS</p>
          <h1>Pick a published course and start building.</h1>
          <p>Enroll once and we will keep your shared workspace, project brief, and review history pinned here. ${escapeHtml(String(readyCourses))} learner-ready course${readyCourses === 1 ? "" : "s"} below.</p>
          <div class="focus-actions">
            <a class="button primary" href="#catalog-panel">Browse published courses ↓</a>
          </div>
        </div>
      `;
      return;
    }

    const enrollment = experience.enrollment;
    const progress = computeCourseProgress(experience);
    const deliverables = experience.deliverables || [];
    const latestSubmission = experience.latest_assignment_submission;
    const latestReport = experience.latest_assignment_report;
    const writeup = experience.project_brief_markdown || "No project brief yet.";
    const session = experience.workspace_session;
    const launchLabel = isBusy("workspace")
      ? "Starting workspace..."
      : (session?.editor_url ? "Resume in workspace" : "Open VS Code workspace");
    const submitLabel = isBusy("submit") ? "Submitting..." : "Submit project for review";
    const canLaunchWorkspace = enrollment.status !== "completed";
    const canSubmit = enrollment.status !== "completed";

    const workspaceRunning = experience?.workspace_session?.status === "running";
    const workspaceAction = session?.editor_url
      ? `
        <a class="button primary" href="${escapeHtml(session.editor_url)}">${escapeHtml(launchLabel)}</a>
      `
      : `
        <button
          class="button primary"
          type="button"
          data-action="launch-workspace"
          ${canLaunchWorkspace ? "" : "disabled"}
        >${escapeHtml(launchLabel)}</button>
      `;

    const eyebrowText = enrollment.status === "completed"
      ? "Course complete"
      : latestSubmission
        ? "Resume your project"
        : `Project review areas: ${progress.total}`;

    const visibleFiles = [...new Set(deliverables.flatMap((item) => item.visible_files || []))];
    const visibleFilesPreview = visibleFiles.slice(0, 3);
    const visibleFilesExtra = Math.max(0, visibleFiles.length - visibleFilesPreview.length);
    const visibleFilesText = visibleFilesPreview.length
      ? `${visibleFilesPreview.map((f) => `<code>${escapeHtml(f)}</code>`).join(", ")}${visibleFilesExtra ? ` <span class="quickref-more">+${visibleFilesExtra} more</span>` : ""}`
      : `<span class="quickref-empty">No visible files listed yet</span>`;
    const latestReviewText = latestSubmission
      ? `${latestSubmission.passed_tests}/${latestSubmission.total_tests} checks passed`
      : "Not submitted yet";

    learnerFocus.innerHTML = `
      <div class="focus-layout">
        <div class="focus-main">
          <p class="course-chip">${escapeHtml(enrollment.course_title)}</p>
          <p class="eyebrow">${escapeHtml(eyebrowText)}</p>
          <h1>${escapeHtml(enrollment.course_title)}</h1>
          <p class="focus-subcopy">${escapeHtml(enrollment.course_summary || "Build the shared project in one workspace and use the deliverable scorecard to see what still needs work.")}</p>

          <dl class="deliverable-quickref" aria-label="Project at a glance">
            <div class="quickref-row">
              <dt>Files to edit</dt>
              <dd>${visibleFilesText}</dd>
            </div>
            <div class="quickref-row">
              <dt>Run visible checks</dt>
              <dd>Inside the VS Code workspace, while you iterate.</dd>
            </div>
            <div class="quickref-row">
              <dt>Submit</dt>
              <dd>Use <strong>Submit project for review</strong> below — runs the hidden grader.</dd>
            </div>
            <div class="quickref-row">
              <dt>Latest review</dt>
              <dd>${escapeHtml(latestReviewText)}</dd>
            </div>
          </dl>

          <div class="writeup-shell">
            <div class="experience-section-header">
              <div>
                <p class="card-eyebrow">What we are building</p>
                <h3>Project brief</h3>
              </div>
              ${uiState.writeupExpanded ? `
                <button class="button subtle" type="button" data-toggle-writeup="collapse">Collapse brief</button>
              ` : ""}
            </div>
            <div class="writeup-frame ${uiState.writeupExpanded ? "" : "is-collapsed"}">
              <div class="writeup-rendered">${renderMarkdown(writeup)}</div>
              ${uiState.writeupExpanded ? "" : `
                <button class="writeup-expand" type="button" data-toggle-writeup="expand">
                  <span>Read full brief</span>
                </button>
              `}
            </div>
          </div>

          <ol class="learner-flow" aria-label="Project workflow">
            <li class="learner-flow-step ${workspaceRunning ? "is-active" : ""}">
              <span class="learner-flow-index">1</span>
              <div class="learner-flow-body">
                <div class="learner-flow-head">
                  <h3>Open your workspace</h3>
                  ${renderWorkspaceStatusInline(experience)}
                </div>
                <p>Cloud VS Code with the shared project files and your saved edits.</p>
                <div class="focus-actions">
                  ${workspaceAction}
                </div>
                ${renderFeedbackBanner(uiState.workspaceFeedback)}
              </div>
            </li>
            <li class="learner-flow-step">
              <span class="learner-flow-index">2</span>
              <div class="learner-flow-body">
                <div class="learner-flow-head">
                  <h3>Run the visible checks</h3>
                  <span class="info-pill"><strong>In the workspace</strong></span>
                </div>
                <p>Iterate locally with the visible tests. They are a subset of the real review run.</p>
              </div>
            </li>
            <li class="learner-flow-step ${canSubmit ? "" : "is-disabled"}">
              <span class="learner-flow-index">3</span>
              <div class="learner-flow-body">
                <div class="learner-flow-head">
                  <h3>Submit the project</h3>
                  <span class="info-pill warn"><strong>Hidden grader</strong></span>
                </div>
                <p>The hidden grader is deeper than the visible checks. Feedback comes back grouped by deliverable.</p>
                <div class="focus-actions">
                  <button
                    class="button"
                    type="button"
                    data-action="submit-project"
                    ${canSubmit ? "" : "disabled"}
                  >${escapeHtml(submitLabel)}</button>
                </div>
                ${renderFeedbackBanner(uiState.submissionFeedback)}
              </div>
            </li>
          </ol>
        </div>
      </div>
    `;
  }

  function renderExperience() {
    const experience = uiState.currentExperience;
    if (!experience) {
      deliverablesPanel.classList.add("hidden");
      deliverablesBody.innerHTML = '<p class="empty">Open a course to see its deliverables.</p>';
      submissionHistory.classList.add("hidden");
      submissionHistoryBody.innerHTML = "<p class=\"empty\">Open a course to see its review history.</p>";
      return;
    }

    const enrollment = experience.enrollment;
    const progress = computeCourseProgress(experience);
    const deliverables = experience.deliverables || [];
    const latestReport = experience.latest_assignment_report;
    const latestSubmission = experience.latest_assignment_submission;

    deliverablesPanel.classList.remove("hidden");
    submissionHistory.classList.remove("hidden");
    deliverablesTitle.textContent = `${progress.total} deliverable${progress.total === 1 ? "" : "s"}`;
    deliverablesCaption.textContent = latestReport
      ? `${latestReport.passed_tests}/${latestReport.total_tests} hidden checks are passing in the latest review run.`
      : "Submit the full project to get a scorecard for each deliverable.";

    deliverablesBody.innerHTML = deliverables.map((deliverable) => {
      const latestGrade = deliverable.latest_submission;
      const statusLabel = latestGrade
        ? (latestGrade.status === "passed" ? "Ready" : "Needs work")
        : "Not reviewed";
      return `
        <div class="deliverable-row">
          <span class="deliverable-row-index">${escapeHtml(String(deliverable.deliverable_index))}</span>
          <div class="deliverable-row-copy">
            <h4>${escapeHtml(deliverable.title)}</h4>
            <p>${escapeHtml(deliverable.objective || "")}</p>
            <div class="deliverable-row-meta">
              ${renderStatusPill(latestGrade ? (latestGrade.status === "passed" ? "passed" : "blocked") : "neutral", statusLabel)}
              ${latestGrade ? renderInfoPill("Last review", `${latestGrade.passed_tests}/${latestGrade.total_tests}`) : ""}
            </div>
          </div>
        </div>
      `;
    }).join("");

    const latestCard = latestSubmission ? `
      <div class="latest-grade-card ${latestSubmission.status === "passed" ? "passed" : "needs-work"}">
        <p class="card-eyebrow">Latest project review</p>
        <div class="latest-grade-row">
          <strong>${escapeHtml(`${latestSubmission.passed_tests}/${latestSubmission.total_tests} tests passed`)}</strong>
          ${renderStatusPill(latestSubmission.status === "passed" ? "passed" : "blocked", titleCase(latestSubmission.status))}
        </div>
        <p class="latest-grade-meta">${escapeHtml(`Pass rate ${percent(latestSubmission.pass_rate)} · Submitted ${formatDate(latestSubmission.created_at)}`)}</p>
      </div>
    ` : `
      <div class="latest-grade-card empty">
        <p class="card-eyebrow">Latest project review</p>
        <p>Submit the shared project to log your first review result.</p>
      </div>
    `;

    const historyByAttempt = [];
    const seenAttempts = new Set();
    for (const submission of experience.submissions || []) {
      const key = submission.submission_group_id || submission.id;
      if (seenAttempts.has(key)) continue;
      seenAttempts.add(key);
      historyByAttempt.push(submission);
    }

    if (!historyByAttempt.length) {
      submissionHistoryBody.innerHTML = `
        <div class="submission-state">
          ${latestCard}
          <p>No prior submissions yet. Use <strong>Submit project for review</strong> to log your first result.</p>
        </div>
      `;
      return;
    }

    submissionHistoryBody.innerHTML = `
      <div class="submission-state">
        ${latestCard}
        <h3>All project submissions</h3>
        <p>Each submission reviews the whole project, then groups the findings by deliverable.</p>
        ${latestReport ? `
          <div class="submission-list review-area-list">
            ${latestReport.review_areas.map((reviewArea) => `
              <div class="submission-item">
                <strong>${escapeHtml(reviewArea.title)}</strong>
                <p>${escapeHtml(reviewArea.objective)}</p>
                <div class="submission-item-meta">
                  ${renderStatusPill(reviewArea.grade_report.status === "passed" ? "passed" : "blocked", reviewArea.grade_report.status === "passed" ? "Ready" : "Needs work")}
                  ${renderInfoPill("Checks", `${reviewArea.grade_report.passed_tests}/${reviewArea.grade_report.total_tests}`)}
                </div>
                ${reviewArea.feedback && reviewArea.grade_report.status !== "passed" ? renderLearnerGuidance(reviewArea.feedback) : ""}
              </div>
            `).join("")}
          </div>
        ` : ""}
        <div class="submission-list">
          ${historyByAttempt.map((submission) => `
            <div class="submission-item">
              <strong>${escapeHtml(formatDate(submission.created_at))}</strong>
              <p>${escapeHtml(titleCase(submission.status))} · ${escapeHtml(`${submission.passed_tests}/${submission.total_tests} checks passed`)}</p>
              <div class="submission-item-meta">
                ${renderStatusPill(submission.status === "passed" ? "passed" : "blocked", titleCase(submission.status))}
                ${renderInfoPill("Pass rate", percent(submission.pass_rate))}
                ${renderInfoPill("Submitted", formatDate(submission.created_at))}
              </div>
            </div>
          `).join("")}
        </div>
      </div>
    `;
  }

  function renderEnrollments(payload) {
    if (!enrollmentList) return;
    const enrollments = payload?.enrollments || [];
    if (!enrollments.length) {
      enrollmentList.innerHTML = `
        <div class="summary-card empty-state">
          <h3>No courses yet</h3>
          <p>Choose a learner-ready course below and we will pin it here once you enroll.</p>
        </div>
      `;
      return;
    }

    enrollmentList.innerHTML = enrollments.map((item) => {
      const selected = item.id === uiState.currentEnrollmentId;
      const progress = computeCourseProgress(item);
      const summaryLine = item.status === "completed"
        ? "Project complete"
        : `${item.completed_deliverable_count}/${item.deliverable_count} deliverables ready in the latest review`;
      return `
        <div class="course-row ${selected ? "is-selected" : ""}">
          <div class="course-row-main">
            <div class="course-row-meta">
              ${renderStatusPill(courseStatusKind(item), courseStatusCopy(item))}
              ${renderInfoPill("Deliverables", `${item.completed_deliverable_count}/${item.deliverable_count}`)}
            </div>
            <h3>${escapeHtml(item.course_title)}</h3>
            <p>${escapeHtml(summaryLine)}</p>
            <div class="micro-progress"><span style="width: ${progress.positionPercent}%"></span></div>
          </div>
          <div class="course-row-side">
            <div class="course-row-meta">
              ${renderInfoPill("Updated", formatRelative(item.updated_at))}
            </div>
            <button
              class="button ${selected ? "subtle" : ""}"
              type="button"
              data-open-enrollment="${escapeHtml(item.id)}"
              ${(selected || isBusy("open-enrollment", item.id)) ? "disabled" : ""}
            >${escapeHtml(isBusy("open-enrollment", item.id) ? "Opening..." : (selected ? "Viewing now" : "Open learner progress"))}</button>
          </div>
        </div>
      `;
    }).join("");
  }

  function renderCatalog(payload) {
    if (!catalogList) return;
    const courses = payload?.courses || [];
    const enrolledCourseIds = new Set((state.enrollments?.enrollments || []).map((item) => item.course_run_id));
    const readyCourses = courses.filter((course) => course.supported_for_lms).length;

    catalogCaption.textContent = (state.enrollments?.enrollments || []).length
      ? `Browse ${courses.length} published course${courses.length === 1 ? "" : "s"} when you are ready for something new. ${readyCourses} ${readyCourses === 1 ? "is" : "are"} learner-ready today.`
      : `Start with one of ${readyCourses} learner-ready published course${readyCourses === 1 ? "" : "s"}, then return here whenever you want another codebase.`;

    catalogBody.classList.toggle("hidden", !uiState.catalogExpanded);
    catalogToggle.textContent = uiState.catalogExpanded
      ? ((state.enrollments?.enrollments || []).length ? "Hide catalog" : "Collapse")
      : `Browse ${courses.length} published course${courses.length === 1 ? "" : "s"}`;

    if (!courses.length) {
      catalogList.innerHTML = '<div class="summary-card empty-state"><h3>No published courses yet</h3><p>Publish a course from the builder first, then learners can enroll from here.</p></div>';
      return;
    }

    catalogList.innerHTML = courses.map((course) => {
      const existingEnrollment = existingEnrollmentForCourse(course.course_run_id);
      const isBlocked = !course.supported_for_lms;
      const buttonLabel = existingEnrollment
        ? "Open learner progress"
        : (isBusy("enroll", course.course_run_id) ? "Enrolling..." : "Enroll and start");
      return `
        <div class="catalog-card ${isBlocked ? "is-blocked" : ""}">
          <div class="catalog-card-main">
            <div class="catalog-card-meta">
              ${renderStatusPill(course.supported_for_lms ? "ready" : "not-ready", course.supported_for_lms ? "Ready to enroll" : "Not learner-ready")}
              ${renderInfoPill("Deliverables", String(course.deliverable_count))}
              ${renderInfoPill("Published", formatDate(course.published_at))}
            </div>
            <h3>${escapeHtml(course.title)}</h3>
            <p>${escapeHtml(course.summary)}</p>
            <p class="catalog-helper ${isBlocked ? "warning" : ""}">
              ${escapeHtml(isBlocked
                ? (existingEnrollment
                  ? `${learnerReadyReason(course)} Reopen your existing attempt from My courses.`
                  : learnerReadyReason(course))
                : (enrolledCourseIds.has(course.course_run_id)
                  ? "You are already enrolled in this course."
                  : "You can enroll now and jump straight into the shared project workspace."))}
            </p>
          </div>
          <div class="catalog-card-footer">
            ${existingEnrollment && !isBlocked ? `
              <button class="button" type="button" data-open-enrollment="${escapeHtml(existingEnrollment.id)}">Open learner progress</button>
            ` : `
              <button
                class="button ${course.supported_for_lms ? "primary" : ""}"
                type="button"
                data-enroll-course="${escapeHtml(course.course_run_id)}"
                ${course.supported_for_lms ? "" : "disabled"}
                title="${course.supported_for_lms ? "" : "This course is still being prepared by the author."}"
              >${escapeHtml(course.supported_for_lms ? buttonLabel : "Preparing for learners")}</button>
            `}
          </div>
        </div>
      `;
    }).join("");
  }

  function renderAll() {
    renderChrome();
    renderPageMessage();
    renderFocus();
    renderExperience();
    updateDocumentTitle();
  }

  async function refreshCatalog() {
    const response = await fetch(state.catalog_url);
    if (!response.ok) {
      throw normalizeError("catalog", await readResponseError(response));
    }
    state.catalog = await response.json();
    renderCatalog(state.catalog);
  }

  async function refreshEnrollments() {
    const response = await fetch(state.enrollments_url);
    if (!response.ok) {
      throw normalizeError("enrollments", await readResponseError(response));
    }
    state.enrollments = await response.json();
    renderEnrollments(state.enrollments);
  }

  async function loadEnrollment(enrollmentId) {
    if (!enrollmentId) return null;
    setBusy("open-enrollment", enrollmentId);
    renderAll();

    try {
      const response = await fetch(`${state.enrollments_url}/${encodeURIComponent(enrollmentId)}/experience`);
      if (!response.ok) {
        throw await readResponseError(response);
      }
      const payload = await response.json();
      uiState.currentExperience = payload;
      uiState.currentEnrollmentId = enrollmentId;
      uiState.workspaceFeedback = null;
      uiState.writeupExpanded = false;
      setPageMessage(null, "");
      syncUrlState();
      return payload;
    } catch (error) {
      const friendly = normalizeError("load", error);
      setPageMessage("error", friendly.message);
      throw friendly;
    } finally {
      clearBusy();
      renderAll();
    }
  }

  async function handleEnroll(courseRunId) {
    setBusy("enroll", courseRunId);
    renderAll();
    try {
      const response = await fetch(state.enrollments_url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ course_run_id: courseRunId }),
      });
      if (!response.ok) {
        throw await readResponseError(response);
      }
      const enrollment = await response.json();
      uiState.submissionFeedback = null;
      uiState.workspaceFeedback = null;
      await refreshEnrollments();
      await loadEnrollment(enrollment.id);
      syncUrlState();
      showToast("success", "Enrollment created", "Your project workspace is ready to open.");
    } catch (error) {
      const friendly = normalizeError("enroll", error);
      setPageMessage("error", friendly.message);
    } finally {
      clearBusy();
      renderAll();
    }
  }

  async function handleLaunchWorkspace() {
    const experience = uiState.currentExperience;
    if (!experience?.enrollment?.id) {
      return;
    }

    const currentSession = experience.workspace_session;
    if (currentSession?.editor_url && currentSession.status === "running") {
      openWorkspaceUrl(currentSession.editor_url);
      return;
    }

    uiState.workspaceFeedback = null;
    setBusy("workspace");
    renderAll();
    try {
      const response = await fetch(`${state.enrollments_url}/${encodeURIComponent(experience.enrollment.id)}/workspace`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        throw await readResponseError(response);
      }
      await refreshEnrollments();
      await loadEnrollment(experience.enrollment.id);
      const refreshedSession = uiState.currentExperience?.active_deliverable?.workspace_session;
      const effectiveSession = uiState.currentExperience?.workspace_session || refreshedSession;
      if (effectiveSession?.editor_url) {
        openWorkspaceUrl(effectiveSession.editor_url);
      } else {
        showToast("success", "Workspace ready", "Your coding workspace is running for this enrollment.");
      }
    } catch (error) {
      const friendly = normalizeError("workspace", error);
      uiState.workspaceFeedback = {
        kind: "error",
        title: "Workspace isn't running",
        message: friendly.message,
        detail: friendly.detail,
      };
    } finally {
      clearBusy();
      renderAll();
    }
  }

  async function handleSubmitDeliverable() {
    const experience = uiState.currentExperience;
    if (!experience?.enrollment?.id) {
      return;
    }

    setBusy("submit");
    startSubmitProgress();
    renderAll();

    try {
      const response = await fetch(`${state.enrollments_url}/${encodeURIComponent(experience.enrollment.id)}/submit`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        throw await readResponseError(response);
      }
      const gradedExperience = await response.json();
      const latestSubmission = gradedExperience.latest_assignment_submission;
      stopSubmitProgress();

      uiState.submissionFeedback = {
        kind: latestSubmission?.status === "passed" ? "success" : "error",
        title: latestSubmission?.status === "passed" ? "Project reviewed" : "Project needs another pass",
        message: latestSubmission
          ? `${latestSubmission.passed_tests}/${latestSubmission.total_tests} checks passed.`
          : "Grading finished.",
      };

      await refreshEnrollments();
      await loadEnrollment(experience.enrollment.id);

      if (latestSubmission) {
        showToast(
          latestSubmission.status === "passed" ? "success" : "info",
          "Review finished",
          `${latestSubmission.passed_tests}/${latestSubmission.total_tests} checks passed.`
        );
      }
    } catch (error) {
      stopSubmitProgress();
      const friendly = normalizeError("submit", error);
      uiState.submissionFeedback = {
        kind: "error",
        title: "Grading didn't complete",
        message: friendly.message,
        detail: friendly.detail,
      };
    } finally {
      stopSubmitProgress();
      clearBusy();
      renderAll();
    }
  }

  if (catalogToggle) {
    catalogToggle.addEventListener("click", () => {
      uiState.catalogExpanded = !uiState.catalogExpanded;
      renderCatalog(state.catalog);
    });
  }

  document.addEventListener("click", async (event) => {
    const target = event.target instanceof Element
      ? event.target.closest("[data-enroll-course],[data-open-enrollment],[data-action],[data-toggle-writeup]")
      : null;
    if (!(target instanceof HTMLElement)) {
      return;
    }

    if (target.dataset.toggleWriteup) {
      uiState.writeupExpanded = !uiState.writeupExpanded;
      renderFocus();
      return;
    }

    if (target.dataset.enrollCourse) {
      await handleEnroll(target.dataset.enrollCourse);
      return;
    }

    if (target.dataset.openEnrollment) {
      await loadEnrollment(target.dataset.openEnrollment);
      syncUrlState();
      renderAll();
      return;
    }

    const action = target.dataset.action;
    if (action === "launch-workspace") {
      await handleLaunchWorkspace();
      return;
    }

    if (action === "submit-project") {
      await handleSubmitDeliverable();
      return;
    }
  });

  renderAll();

  const initialEnrollment = selectedEnrollmentSummary();
  if (initialEnrollment?.id) {
    loadEnrollment(initialEnrollment.id).catch(() => {
      renderAll();
    });
  }

  Promise.allSettled([refreshCatalog(), refreshEnrollments()]).then(async (results) => {
    const catalogFailure = results[0].status === "rejected" ? results[0].reason : null;
    const enrollmentFailure = results[1].status === "rejected" ? results[1].reason : null;

    if (catalogFailure || enrollmentFailure) {
      setPageMessage("error", catalogFailure?.message || enrollmentFailure?.message || "We couldn't refresh the learner home.");
      renderAll();
      return;
    }

    const nextEnrollment = selectedEnrollmentSummary();
    if (nextEnrollment?.id && nextEnrollment.id !== uiState.currentEnrollmentId) {
      try {
        await loadEnrollment(nextEnrollment.id);
      } catch (_error) {
        renderAll();
      }
      return;
    }

    renderAll();
  });
})();
