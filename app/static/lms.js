(() => {
  const stateElement = document.getElementById("lms-state");
  const state = stateElement?.textContent ? JSON.parse(stateElement.textContent) : {};

  const pageMessage = document.getElementById("page-message");
  const toastRegion = document.getElementById("toast-region");
  const learnerShell = document.getElementById("learner-shell");
  const learnView = document.getElementById("learn-view");
  const coursesView = document.getElementById("courses-view");
  const courseSidebar = document.getElementById("course-sidebar");
  const learnerFocus = document.getElementById("learner-focus");
  const experiencePanel = document.getElementById("experience-panel");
  const experienceTitle = document.getElementById("experience-title");
  const experienceCaption = document.getElementById("experience-caption");
  const moduleListTitle = document.getElementById("module-list-title");
  const moduleTimeline = document.getElementById("module-timeline");
  const submissionHistory = document.getElementById("submission-history");
  const submissionHistoryBody = document.getElementById("submission-history-body");
  const enrollmentList = document.getElementById("enrollment-list");
  const catalogBody = document.getElementById("catalog-body");
  const catalogCaption = document.getElementById("catalog-caption");
  const catalogList = document.getElementById("catalog-list");
  const catalogToggle = document.getElementById("catalog-toggle");

  const currentUrl = new URL(window.location.href);
  const selectionFromUrl = currentUrl.searchParams.get("enrollment");
  const initialProgressFromUrl = currentUrl.searchParams.get("progress");

  const uiState = {
    currentExperience: null,
    currentEnrollmentId: selectionFromUrl,
    sidebarCollapsed: initialProgressFromUrl === "hidden",
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
    const modules = summaryOrExperience?.modules || [];
    const enrollment = summaryOrExperience?.enrollment || summaryOrExperience || {};
    const total = modules.length || Number(enrollment.module_count || 0);
    const completed = modules.length
      ? modules.filter((module) => module.status === "passed").length
      : Number(enrollment.completed_module_count || 0);
    const currentIndex = modules.length
      ? (modules.find((module) => module.module_id === enrollment.current_module_id)?.module_index || 0)
      : Number(enrollment.current_module_index || 0);
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
      return "Includes learner workspaces, graded checkpoints, and module progression.";
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

  function moduleStatusCopy(module, enrollment) {
    if (enrollment?.status === "completed" && module.status === "passed") {
      return "Completed";
    }
    if (enrollment?.current_module_id === module.module_id && enrollment?.status !== "completed") {
      return "In progress";
    }
    const labels = {
      available: "Ready now",
      passed: "Passed",
      locked: "Locked",
    };
    return labels[module.status] || titleCase(module.status);
  }

  function moduleStatusKind(module, enrollment) {
    if (enrollment?.current_module_id === module.module_id && enrollment?.status !== "completed") {
      return "in-progress";
    }
    const kinds = {
      available: "ready",
      passed: "passed",
      locked: "locked",
    };
    return kinds[module.status] || "neutral";
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
    if (uiState.sidebarCollapsed) {
      url.searchParams.set("progress", "hidden");
    } else {
      url.searchParams.delete("progress");
    }
    window.history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
  }

  function toggleSidebar(forceValue = null) {
    if (!uiState.currentExperience) {
      return;
    }
    uiState.sidebarCollapsed = typeof forceValue === "boolean" ? forceValue : !uiState.sidebarCollapsed;
    syncUrlState();
    renderAll();
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
    const moduleTitle = experience.active_module?.title;
    document.title = moduleTitle
      ? `${moduleTitle} · ${courseTitle}`
      : `${courseTitle} · Learner LMS`;
  }

  function renderChrome() {
    const sidebarAvailable = Boolean(uiState.currentExperience);

    learnerShell.classList.toggle("sidebar-hidden", sidebarAvailable && uiState.sidebarCollapsed);
    learnerShell.classList.toggle("no-sidebar", !sidebarAvailable);
    courseSidebar.classList.toggle("hidden", !sidebarAvailable);
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
          message: "We couldn't reach your module app for grading.",
          detail,
        };
      }
      return {
        message: "We couldn't submit this module for grading right now.",
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

  function inlineCode(text) {
    return text.replace(/`([^`]+)`/g, "<code>$1</code>");
  }

  function renderMarkdown(markdown) {
    const source = escapeHtml(markdown || "No module writeup yet.");
    const lines = source.replace(/\r\n/g, "\n").split("\n");
    const html = [];
    let paragraph = [];
    let listItems = [];
    let orderedItems = [];
    let codeBlock = [];
    let inCodeBlock = false;

    function flushParagraph() {
      if (!paragraph.length) return;
      html.push(`<p>${inlineCode(paragraph.join(" "))}</p>`);
      paragraph = [];
    }

    function flushList() {
      if (!listItems.length) return;
      html.push(`<ul>${listItems.map((item) => `<li>${inlineCode(item)}</li>`).join("")}</ul>`);
      listItems = [];
    }

    function flushOrderedList() {
      if (!orderedItems.length) return;
      html.push(`<ol>${orderedItems.map((item) => `<li>${inlineCode(item)}</li>`).join("")}</ol>`);
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
        html.push(`<h${level}>${inlineCode(headingMatch[2])}</h${level}>`);
        continue;
      }

      const orderedMatch = trimmed.match(/^\d+\.\s+(.*)$/);
      if (orderedMatch) {
        flushParagraph();
        flushList();
        orderedItems.push(orderedMatch[1]);
        continue;
      }

      const listMatch = trimmed.match(/^[-*]\s+(.*)$/);
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

  function renderWorkspaceStatusInline(experience) {
    const activeModule = experience?.active_module;
    const session = activeModule?.workspace_session;

    if (!activeModule) {
      return "";
    }

    let note = "Workspace starts on this module starter when you open VS Code.";
    if (session?.status === "running") {
      note = "Workspace is running with your saved files ready to resume.";
    } else if (session?.status === "starting") {
      note = "Workspace is starting up for this module.";
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
          <p class="focus-subcopy">We are fetching your current module, workspace status, and progress history.</p>
        </div>
      `;
      return;
    }

    if (!experience && selectedSummary) {
      learnerFocus.innerHTML = `
        <div class="hero-copy">
          <p class="eyebrow">Continue where you left off</p>
          <h1>${escapeHtml(selectedSummary.course_title)}</h1>
          <p class="focus-subtitle">${selectedSummary.current_module_title
            ? `Module ${selectedSummary.current_module_index} · ${escapeHtml(selectedSummary.current_module_title)}`
            : "Loading your current module..."}</p>
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
          <p>Enroll once and we will keep your workspace, grading history, and next module pinned here. ${escapeHtml(String(readyCourses))} learner-ready course${readyCourses === 1 ? "" : "s"} below.</p>
          <div class="focus-actions">
            <a class="button primary" href="#catalog-panel">Browse published courses ↓</a>
          </div>
        </div>
      `;
      return;
    }

    const enrollment = experience.enrollment;
    const activeModule = experience.active_module;
    const progress = computeCourseProgress(experience);
    const currentSubmission = activeModule.latest_submission;
    const writeup = activeModule.content_markdown || activeModule.starter_readme || "No module writeup yet.";
    const session = activeModule.workspace_session;
    const launchLabel = isBusy("workspace")
      ? "Starting workspace..."
      : (session?.editor_url ? "Resume in workspace" : "Open VS Code workspace");
    const submitLabel = isBusy("submit") ? "Submitting..." : "Submit for grading";
    const canLaunchWorkspace = Boolean(enrollment.current_module_id) && enrollment.status !== "completed";
    const canSubmit = enrollment.current_module_id && enrollment.status !== "completed";

    const visibleFilesCount = activeModule.visible_files.length;
    const latestScoreCopy = currentSubmission
      ? `${currentSubmission.passed_tests}/${currentSubmission.total_tests} tests`
      : "Not submitted yet";
    const workspaceRunning = activeModule?.workspace_session?.status === "running";

    const eyebrowText = enrollment.status === "completed"
      ? "Course complete"
      : currentSubmission
        ? `Module ${activeModule.module_index} · resume`
        : `Module ${activeModule.module_index} of ${progress.total}`;

    learnerFocus.innerHTML = `
      <div class="focus-layout">
        <div class="focus-main">
          <p class="course-chip">${escapeHtml(enrollment.course_title)}</p>
          <p class="eyebrow">${escapeHtml(eyebrowText)}</p>
          <h1>${escapeHtml(activeModule.title)}</h1>
          <p class="focus-subcopy">${escapeHtml(enrollment.course_summary || "Keep building through the published learner modules with a persistent workspace and graded checkpoints.")}</p>

          <div class="writeup-shell">
            <div class="experience-section-header">
              <div>
                <p class="card-eyebrow">What to build</p>
                <h3>Module brief</h3>
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

          <ol class="learner-flow" aria-label="Module flow">
            <li class="learner-flow-step ${workspaceRunning ? "is-active" : ""}">
              <span class="learner-flow-index">1</span>
              <div class="learner-flow-body">
                <div class="learner-flow-head">
                  <h3>Open your workspace</h3>
                  ${renderWorkspaceStatusInline(experience)}
                </div>
                <p>Cloud VS Code with the module starter and your saved edits.</p>
                <div class="focus-actions">
                  <button
                    class="button primary"
                    type="button"
                    data-action="launch-workspace"
                    ${canLaunchWorkspace ? "" : "disabled"}
                  >${escapeHtml(launchLabel)}</button>
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
                <p>Iterate locally with the visible tests. They're a subset of the real grader.</p>
              </div>
            </li>
            <li class="learner-flow-step ${canSubmit ? "" : "is-disabled"}">
              <span class="learner-flow-index">3</span>
              <div class="learner-flow-body">
                <div class="learner-flow-head">
                  <h3>Submit for grading</h3>
                  <span class="info-pill warn"><strong>Hidden grader</strong></span>
                </div>
                <p>The hidden grader is deeper than the visible checks. Passing unlocks the next module.</p>
                <div class="focus-actions">
                  <button
                    class="button"
                    type="button"
                    data-action="submit-module"
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
      experiencePanel.classList.add("hidden");
      moduleTimeline.innerHTML = '<div class="empty">Open a course to see its module ladder.</div>';
      submissionHistory.classList.add("hidden");
      submissionHistoryBody.innerHTML = "<p class=\"empty\">Open a course to see its grading history.</p>";
      return;
    }

    const enrollment = experience.enrollment;
    const progress = computeCourseProgress(experience);

    experiencePanel.classList.remove("hidden");
    submissionHistory.classList.remove("hidden");
    experienceTitle.textContent = enrollment.course_title;
    experienceCaption.textContent = `${progress.completed}/${progress.total} modules passed. The current module stays highlighted until you submit and unlock the next one.`;
    moduleListTitle.textContent = `Module ladder · ${progress.total} total`;

    moduleTimeline.innerHTML = experience.modules.map((module) => {
      const isCurrent = enrollment.current_module_id === module.module_id && enrollment.status !== "completed";
      const statusLabel = isCurrent
        ? "Current"
        : module.status === "locked"
          ? "Locked"
          : module.status === "passed"
            ? "Passed"
            : "Ready";
      return `
        <div class="module-row ${isCurrent ? "is-current" : ""} ${module.status === "locked" ? "is-locked" : ""}">
          <span class="module-row-index">${escapeHtml(String(module.module_index))}</span>
          <div class="module-row-copy">
            <h4>${escapeHtml(module.title)}</h4>
            <div class="module-row-meta">
              ${renderStatusPill(moduleStatusKind(module, enrollment), statusLabel)}
              ${module.latest_submission ? renderInfoPill("Last grade", `${module.latest_submission.passed_tests}/${module.latest_submission.total_tests}`) : ""}
            </div>
          </div>
        </div>
      `;
    }).join("");

    const allSubmissions = experience.modules
      .filter((module) => module.latest_submission)
      .map((module) => ({
        ...module.latest_submission,
        moduleTitle: module.title,
        moduleIndex: module.module_index,
      }))
      .sort((left, right) => new Date(right.created_at).getTime() - new Date(left.created_at).getTime());

    const activeModule = experience.active_module;
    const latestSubmission = activeModule?.latest_submission;
    const latestCard = latestSubmission ? `
      <div class="latest-grade-card ${latestSubmission.status === "passed" ? "passed" : "needs-work"}">
        <p class="card-eyebrow">Latest grade · this module</p>
        <div class="latest-grade-row">
          <strong>${escapeHtml(`${latestSubmission.passed_tests}/${latestSubmission.total_tests} tests passed`)}</strong>
          ${renderStatusPill(latestSubmission.status === "passed" ? "passed" : "blocked", titleCase(latestSubmission.status))}
        </div>
        <p class="latest-grade-meta">${escapeHtml(`Pass rate ${percent(latestSubmission.pass_rate)} · Submitted ${formatDate(latestSubmission.created_at)}`)}</p>
      </div>
    ` : `
      <div class="latest-grade-card empty">
        <p class="card-eyebrow">Latest grade · this module</p>
        <p>Submit the current module to log your first grading result.</p>
      </div>
    `;

    if (!allSubmissions.length) {
      submissionHistoryBody.innerHTML = `
        <div class="submission-state">
          ${latestCard}
          <p>No prior submissions yet. Use <strong>Submit for grading</strong> on the current module to log your first result and unlock the next step.</p>
        </div>
      `;
      return;
    }

    submissionHistoryBody.innerHTML = `
      <div class="submission-state">
        ${latestCard}
        <h3>All submissions</h3>
        <p>Each module keeps its most recent grading run here so you can see what passed and what unlocked next.</p>
        <div class="submission-list">
          ${allSubmissions.map((submission) => `
            <div class="submission-item">
              <strong>Module ${escapeHtml(String(submission.moduleIndex))} · ${escapeHtml(submission.moduleTitle)}</strong>
              <p>${escapeHtml(titleCase(submission.status))} · ${escapeHtml(`${submission.passed_tests}/${submission.total_tests} tests passed`)}</p>
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
      const summaryLine = item.current_module_title
        ? `Module ${item.current_module_index} · ${item.current_module_title}`
        : "Course complete";
      return `
        <div class="course-row ${selected ? "is-selected" : ""}">
          <div class="course-row-main">
            <div class="course-row-meta">
              ${renderStatusPill(courseStatusKind(item), courseStatusCopy(item))}
              ${renderInfoPill("Modules", `${item.completed_module_count}/${item.module_count}`)}
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
              ${renderInfoPill("Modules", String(course.module_count))}
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
                  : "You can enroll now and jump straight into the current module workspace."))}
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
      uiState.sidebarCollapsed = false;
      syncUrlState();
      showToast("success", "Enrollment created", "Your current module is ready to open in the workspace.");
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

    uiState.workspaceFeedback = null;
    setBusy("workspace");
    renderAll();
    try {
      const response = await fetch(`${state.enrollments_url}/${encodeURIComponent(experience.enrollment.id)}/workspace`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ module_id: experience.active_module.module_id }),
      });
      if (!response.ok) {
        throw await readResponseError(response);
      }
      await refreshEnrollments();
      await loadEnrollment(experience.enrollment.id);
      const refreshedSession = uiState.currentExperience?.active_module?.workspace_session;
      if (refreshedSession?.editor_url) {
        const editorTab = window.open(refreshedSession.editor_url, "_blank", "noopener,noreferrer");
        showToast(
          editorTab ? "success" : "info",
          "Workspace ready",
          editorTab
            ? "Opening your VS Code workspace now."
            : "Your browser blocked the new tab. Use the workspace button again to retry."
        );
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

  async function handleSubmitModule() {
    const experience = uiState.currentExperience;
    if (!experience?.enrollment?.id) {
      return;
    }

    uiState.submissionFeedback = null;
    setBusy("submit");
    renderAll();

    try {
      const response = await fetch(`${state.enrollments_url}/${encodeURIComponent(experience.enrollment.id)}/submit`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ module_id: experience.active_module.module_id }),
      });
      if (!response.ok) {
        throw await readResponseError(response);
      }
      const gradedExperience = await response.json();
      const latestSubmission = gradedExperience.active_module.latest_submission;
      const nextModuleId = gradedExperience.enrollment.current_module_id;
      const unlockedNext = Boolean(
        latestSubmission?.status === "passed" &&
        nextModuleId &&
        nextModuleId !== gradedExperience.active_module.module_id
      );

      uiState.submissionFeedback = {
        kind: latestSubmission?.status === "passed" ? "success" : "error",
        title: latestSubmission?.status === "passed" ? "Submission graded" : "Submission needs another pass",
        message: latestSubmission
          ? `${latestSubmission.passed_tests}/${latestSubmission.total_tests} tests passed${unlockedNext ? ". The next module is now available." : "."}`
          : "Grading finished.",
      };

      await refreshEnrollments();
      await loadEnrollment(experience.enrollment.id);

      if (unlockedNext) {
        const nextModule = uiState.currentExperience?.modules?.find((module) => module.module_id === nextModuleId);
        showToast(
          "success",
          "Module unlocked",
          nextModule ? `Next up: ${nextModule.title}.` : "Your next module is ready."
        );
      } else if (latestSubmission) {
        showToast(
          latestSubmission.status === "passed" ? "success" : "info",
          "Grading finished",
          `${latestSubmission.passed_tests}/${latestSubmission.total_tests} tests passed.`
        );
      }
    } catch (error) {
      const friendly = normalizeError("submit", error);
      uiState.submissionFeedback = {
        kind: "error",
        title: "Grading didn't complete",
        message: friendly.message,
        detail: friendly.detail,
      };
    } finally {
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
      ? event.target.closest("[data-enroll-course],[data-open-enrollment],[data-action],[data-toggle-writeup],[data-toggle-progress]")
      : null;
    if (!(target instanceof HTMLElement)) {
      return;
    }

    if (target.hasAttribute("data-toggle-progress")) {
      toggleSidebar();
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
      uiState.sidebarCollapsed = false;
      syncUrlState();
      renderAll();
      return;
    }

    const action = target.dataset.action;
    if (action === "launch-workspace") {
      await handleLaunchWorkspace();
      return;
    }

    if (action === "submit-module") {
      await handleSubmitModule();
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
