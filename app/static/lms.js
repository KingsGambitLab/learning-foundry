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

  // Display-only: strip a leading marketing prefix ("Production" /
  // "Production-Quality ") from course/lab titles. Does NOT mutate data.
  function cleanTitle(t) {
    return String(t ?? "").replace(/^\s*Production(?:-Quality)?\s+/i, "").trim();
  }

  // 15+/18 (>=15 marks) is treated as "solved"/green in the UI. The
  // backend status stays authoritative for data; this is render-only.
  const GOOD_ACCURACY_MARKS = 15;

  // Concept-level hint per failing rubric — bridges the gap between
  // "what's wrong" (the worked example) and the IR/technique a learner
  // should reach for. Deliberately a CONCEPT nudge, never the answer.
  const RUBRIC_HINTS = {
    llm_judge_semantic_eq:
      "Checks your answer means the same as the reference. If it's off-topic, that's usually a retrieval problem — you answered from the wrong source, not a wording problem.",
    llm_judge_false_premise:
      "This question can't be answered from the provided knowledge base — the service must abstain. The skill: verify the question's core premise is actually grounded in the sources before you answer.",
    literal_match:
      "An exact-value check (often abstained==true on unanswerable questions). Detect when no provided source supports the question's premise.",
    oracle_set_overlap:
      "Citation recall: a source that supports the answer is missing from your `citations`. Revisit how you decide which sources to cite.",
    subset_match:
      "Citation precision: you cited a source that isn't an accepted supporting source. Only cite sources that actually support the answer.",
    schema_match:
      "Your response shape doesn't match the contract — check the required fields and their types.",
    behavioral_equivalence:
      "Your output shifts (or stays templated) under reordered/distractor inputs. Make the answer depend on evidence content, not position or fixed patterns.",
    extractive_stub_resistance:
      "Your output shifts (or stays templated) under reordered/distractor inputs. Make the answer depend on evidence content, not position or fixed patterns.",
    regex_match:
      "The field doesn't match the required format. Compare your value against the expected pattern and fix the shape, not just the content.",
    numeric_range:
      "A numeric field is out of the allowed range. Check the bound shown in Expected and clamp/compute the value accordingly.",
    llm_judge_coverage:
      "Your answer is missing key points it must convey. Make sure each required fact from the evidence is actually stated.",
    llm_judge_false_premise:
      "The question's premise isn't supported by the evidence — the service must abstain/refuse rather than answer.",
  };
  // Target-specific hints: many distinct skills grade via the same
  // rubric kind (literal_match on action vs redactions vs abstained),
  // so a kind-only hint is misleading. failing_rubric is now a label
  // like "literal_match on action" — key the hint on kind + target.
  const TARGET_HINTS = {
    "literal_match on action":
      "Your routing decision was wrong. Decide the policy: should this be answered, asked-to-clarify, escalated to a human, or refused?",
    "literal_match on abstained":
      "You should have refused/abstained — the request can't be answered from what's provided (off-scope or no supporting evidence).",
    "literal_match on redactions":
      "PII wasn't handled. Detect and redact emails / phones / cards / SSNs in echoed content and report how many you redacted.",
    "literal_match on escalation_reason":
      "When you escalate, you must include a clear escalation_reason.",
  };
  function hintForRubric(label) {
    if (!label) return "";
    const s = String(label).trim();
    if (TARGET_HINTS[s]) return TARGET_HINTS[s];
    const kind = s.split(" on ")[0].trim();
    return RUBRIC_HINTS[kind] ||
      "Re-read this scenario's Expected vs Your output above and adjust the step that produced the difference.";
  }

  // Plain-English name for the failing check. The internal rubric label
  // ("oracle_set_overlap on citations") still drives hintForRubric, but
  // learners never see rubric-kind jargon. Keyed on kind + target;
  // unknown combos degrade to a readable field-oriented phrase.
  const CHECK_LABELS = {
    "oracle_set_overlap on citations": "Citations — a required source is missing",
    "subset_match on citations": "Citations — an unsupported source was cited",
    "literal_match on action": "Routing decision (answer / clarify / escalate / refuse)",
    "behavioral_equivalence on action": "Routing decision stability",
    "literal_match on abstained": "Abstain / refuse decision",
    "llm_judge_false_premise on abstained": "Abstain on an unanswerable question",
    "literal_match on redactions": "PII redaction count",
    "numeric_range on redactions": "PII redaction count",
    "literal_match on escalation_reason": "Escalation reason",
    "regex_match on escalation_reason": "Escalation reason",
    "llm_judge_semantic_eq on reply": "Answer quality (meaning)",
    "llm_judge_coverage on reply": "Answer completeness",
  };
  function friendlyCheckLabel(label) {
    if (!label) return "";
    const s = String(label).trim();
    if (CHECK_LABELS[s]) return CHECK_LABELS[s];
    const [kind, field] = s.split(" on ").map((x) => x && x.trim());
    if (kind && kind.indexOf("schema_match") === 0) return "Response shape (contract)";
    if (!field || field === "response") {
      if (kind && kind.indexOf("schema_match") === 0) return "Response shape (contract)";
      return "Response check";
    }
    if (kind === "regex_match") return `Format of \`${field}\``;
    if (kind === "numeric_range") return `\`${field}\` out of allowed range`;
    return `\`${field}\` is incorrect`;
  }

  function isSolved(passed, total, backendStatus) {
    const p = Number(passed || 0);
    return backendStatus === "passed" || p >= GOOD_ACCURACY_MARKS || (total && p === Number(total));
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
    return course.support_reason || "This lab is being prepared and is not ready for learners yet.";
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

  function deliverableSolved(deliverable) {
    // Display-only: a deliverable is "solved"/green when its latest
    // submission clears the >=15 marks bar (or is a full backend pass).
    // Defensive: fall back to false if the summary lacks per-mark data.
    const sub = deliverable?.latest_submission;
    if (!sub) return false;
    return isSolved(sub.passed_tests, sub.total_tests, sub.status);
  }

  function deliverableStatusCopy(deliverable, enrollment) {
    if (enrollment?.status === "completed" && deliverable.status === "passed") {
      return "Completed";
    }
    if (deliverableSolved(deliverable)) {
      return "Solved";
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
    if (deliverableSolved(deliverable)) {
      return "passed";
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
      message: "Preparing your shared project files for the learner review run.",
      delayMs: 0,
    },
    {
      title: "Booting review sandbox",
      message: "Starting the app sandbox and wiring up the grader checks.",
      delayMs: 1500,
    },
    {
      title: "Running review checks",
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
    window.open(url, "_blank", "noopener,noreferrer");
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
      document.title = "Scaler Labs";
      return;
    }
    const courseTitle = cleanTitle(experience.enrollment.course_title) || "Learner LMS";
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
          message: "This lab is still being prepared and is not ready for learners yet.",
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
        message: "We couldn't refresh the published lab catalog.",
        detail,
      };
    }

    if (action === "enrollments") {
      return {
        message: "We couldn't refresh your lab list right now.",
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

  // Shared trusted VS Code glyph (used on the workspace launch control
  // and via the `:vscode:` markdown shortcode). Static SVG — safe to
  // inject after escapeHtml since it is ours, not user input.
  const VSCODE_GLYPH =
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false" ' +
    'style="width:15px;height:15px;vertical-align:-2px;margin-right:6px" ' +
    'fill="currentColor"><path d="M17.6 2.3 9.9 9.6 5.3 6.1 3 7.2v9.6l2.3 1.1 ' +
    '4.6-3.5 7.7 7.3L21 20V4l-3.4-1.7Zm.4 4.9v9.6l-5.2-4.8 5.2-4.8ZM6 9.1l2.7 ' +
    '2.9L6 14.9V9.1Z"/></svg>';

  function inlineFormat(text) {
    let out = text;
    out = out.replace(/:vscode:/g, VSCODE_GLYPH);
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

  function renderLearnerGuidance(feedback, opts) {
    if (!feedback) return "";
    const strengths = Array.isArray(feedback.strengths) ? feedback.strengths.filter(Boolean) : [];
    const whyItMatters = Array.isArray(feedback.why_it_matters) ? feedback.why_it_matters.filter(Boolean) : [];
    const likelyRootCause = Array.isArray(feedback.likely_root_cause) ? feedback.likely_root_cause.filter(Boolean) : [];
    const investigationSteps = Array.isArray(feedback.investigation_steps) ? feedback.investigation_steps.filter(Boolean) : [];
    // When the deliverable already passes, improvements are optional —
    // render a calm neutral panel (never the red "needs work" treatment)
    // and soften the wording so it reads as refinement, not failure.
    const isReady = Boolean(opts && opts.isReady);
    const panelClass = isReady
      ? " is-ready"
      : (opts && opts.isStrong ? " is-strong" : "");
    const summaryLabel = isReady ? "Optional improvements" : "Summary feedback";
    const rootCauseLabel = isReady ? "Optional refinements" : "Likely root cause";
    return `
      <details class="review-guidance${panelClass}"${isReady ? "" : " open"}>
        <summary>${summaryLabel}</summary>
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
            <h5>${rootCauseLabel}</h5>
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

  function summarizeDiagnostics(diagnostics) {
    // Compress the per-rubric diagnostic list into a SHORT one-line
    // headline + a cleaned-up tail. The dominant noise pattern when a
    // scenario fails early is N rubrics independently reporting
    // "<some.path> not found in captures" — all cascading from the
    // same first failure. Detect that cluster, surface one path with
    // "+N more", and keep other rubric kinds intact.
    if (!Array.isArray(diagnostics) || !diagnostics.length) return { headline: "", details: [] };
    const cleanedRaw = diagnostics.map((d) => String(d).trim()).filter(Boolean);
    if (!cleanedRaw.length) return { headline: "", details: [] };
    // Exact dedupe: when two rubrics emit identical diagnostic text
    // (e.g. two ``schema_match`` instances both report
    // "target dict failed schema check"), show it once.
    const seen = new Set();
    const cleaned = cleanedRaw.filter((d) => {
      if (seen.has(d)) return false;
      seen.add(d);
      return true;
    });

    // Group "not found in captures" / "not present in captures" diagnostics.
    const missingPathRe = /not (?:found|present) in captures/i;
    const missing = cleaned.filter((d) => missingPathRe.test(d));
    const other = cleaned.filter((d) => !missingPathRe.test(d));

    const details = [...other];
    if (missing.length) {
      const first = missing[0];
      if (missing.length === 1) {
        details.push(first);
      } else {
        details.push(`${first} (+${missing.length - 1} more missing path${missing.length - 1 === 1 ? "" : "s"})`);
      }
    }
    return { headline: details[0] || cleaned[0], details };
  }

  function parseCourseSummary(text) {
    // Course summaries follow a stable shape:
    //
    //   <overview paragraph(s)>
    //
    //   Skills you'll learn:
    //   - skill bullet 1       OR    1. skill bullet 1
    //   - skill bullet 2              2. skill bullet 2
    //   ...
    //
    //   <optional trailing paragraph(s)>
    //
    // We pull skills from the bullet block ONLY — lines that don't
    // start with a bullet marker (``-``/``*``/``•``) or a numeric
    // ordinal (``1.``/``2.``) are skipped, so post-bullet paragraphs
    // like "Graded against 18 hidden scenarios..." don't pollute the
    // skill list.
    if (!text) return { overview: "", skills: [] };
    const trimmed = String(text).trim();
    const headerMatch = trimmed.match(/^([\s\S]*?)\n+\s*skills you('?ll)? learn:?\s*\n([\s\S]*)$/i);
    if (!headerMatch) {
      return { overview: trimmed, skills: [] };
    }
    const overview = headerMatch[1].trim();
    const skillsBlock = headerMatch[3] || "";

    const bulletPattern = /^\s*(?:[-*•]|\d+[.)])\s+/;
    const skills = [];
    for (const rawLine of skillsBlock.split(/\n/)) {
      const line = rawLine.trim();
      if (!line) continue;
      if (!bulletPattern.test(line)) {
        // First non-bullet line ends the skills block — anything after
        // is descriptive trailing prose, not a skill.
        break;
      }
      const cleaned = line.replace(bulletPattern, "").trim();
      if (cleaned) skills.push(cleaned);
    }
    return { overview, skills };
  }

  function shortenOverview(text, maxSentences = 1) {
    // The first 1-2 sentences carry the elevator pitch; later sentences
    // ("It is interesting because...") are usually marketing prose.
    // Trim conservatively so we never cut mid-sentence.
    if (!text) return "";
    const sentences = text.match(/[^.!?]+[.!?]+(?:\s|$)/g);
    if (!sentences || !sentences.length) return text.trim();
    return sentences.slice(0, maxSentences).join("").trim();
  }

  function skillTagLabel(skill) {
    // The bullet sentence is long. Show just the headline noun phrase
    // so the tag stays compact; the full sentence is preserved in the
    // hover title.
    if (!skill) return "";
    const cleaned = skill.trim();
    // Strategy 1 (most reliable): courses written as
    //   ``Name - Description`` / ``Name — Description`` / ``Name: Description``
    // — split on the FIRST separator and use the left as the label.
    // Cap the label at 80 chars so a hyphen buried deep in prose can't
    // sneak through; that's enough room for any reasonable skill name
    // (longest seen in published courses: ~60 chars).
    const sepMatch = cleaned.match(/^(.{1,80}?)\s+[-–—:]\s+/);
    if (sepMatch && sepMatch[1]) {
      return sepMatch[1].trim();
    }
    // Strategy 2 (no explicit separator): cut at the first connector
    // word — these mark the end of the skill's headline noun phrase
    // and the start of its explanation. The character class includes
    // ``.`` so prefixes like ``Foo (Okapi)`` don't break the regex.
    const connectorMatch = cleaned.match(
      /^([\w\s\-/().'&]+?)\s+(?:for|using|across|with|so|that|including|which|where|based|of|to|via|by|when)\b/i
    );
    if (connectorMatch && connectorMatch[1]) {
      return connectorMatch[1].trim();
    }
    // Strategy 3 (fallback): first 4 words.
    const words = cleaned.split(/\s+/);
    if (words.length <= 4) return cleaned;
    return words.slice(0, 4).join(" ");
  }

  function renderSkillTags(skills) {
    if (!skills || !skills.length) return "";
    const chips = skills.map((skill) => {
      const label = skillTagLabel(skill);
      // Full sentence stays accessible as the hover tooltip — no data lost.
      return `<span class="skill-tag" title="${escapeHtml(skill)}">${escapeHtml(label)}</span>`;
    });
    return `<div class="skill-tags" aria-label="Skills you'll learn">${chips.join("")}</div>`;
  }

  function renderTestResults(gradeReport) {
    // Render per-test results from a DeliverableGradeReport. Outcome-mode
    // graders set ``feedback=None`` on the ReviewArea — the actionable
    // signal lives in each TestGradeResult's diagnostics. We keep the
    // top-level "N checks need attention" container open by default but
    // collapse each individual scenario's diagnostics behind a
    // per-row <details>, with the headline (first / deduped diagnostic)
    // shown alongside the scenario name. Cascading "not found in
    // captures" diagnostics get folded into one line "+N more".
    const results = Array.isArray(gradeReport?.results) ? gradeReport.results : [];
    if (!results.length) return "";
    const failed = results.filter((r) => r.status !== "passed");
    const passed = results.filter((r) => r.status === "passed");

    const renderOne = (result) => {
      const diagnostics = Array.isArray(result.diagnostics) ? result.diagnostics.filter(Boolean) : [];
      const { headline, details } = summarizeDiagnostics(diagnostics);
      const statusKind = result.status === "passed" ? "passed" : "blocked";
      const hasMoreDetails = details.length > 1;
      const countLabel = details.length
        ? `${details.length} diagnostic${details.length === 1 ? "" : "s"}`
        : "";
      // Worked example (failed scenarios only): the actual question,
      // the expected/gold reference, the learner's own output, and
      // which rubric failed — so a learner fixes in one pass instead
      // of guessing against hidden inputs.
      const exRow = (label, val) =>
        val
          ? `<div class="we-row"><span class="we-label">${escapeHtml(label)}</span><code class="we-val">${escapeHtml(String(val))}</code></div>`
          : "";
      const isPassed = result.status === "passed";
      const workedExample =
        !isPassed &&
        (result.example_question || result.example_expected || result.example_actual || result.failing_rubric)
          ? `<div class="test-result-example">
              ${exRow("What failed", friendlyCheckLabel(result.failing_rubric))}
              ${exRow("Question", result.example_question)}
              ${exRow("Expected", result.example_expected)}
              ${exRow("Your output", result.example_actual)}
              <div class="we-hint"><span class="we-label">How to think about it</span><span class="we-hint-text">${escapeHtml(hintForRubric(result.failing_rubric))}</span></div>
            </div>`
          // Passing scenarios: a positive worked example — the question
          // and the learner's own response that satisfied every check.
          // No Expected/failing-check/hint (nothing failed).
          : isPassed && (result.example_question || result.example_actual)
          ? `<div class="test-result-example test-result-example-passed">
              ${exRow("Question", result.example_question)}
              ${exRow("Your output", result.example_actual)}
            </div>`
          : "";
      // Each test is a <details> whose summary carries the head row + the
      // one-line headline. The diagnostic count sits on the right of the
      // head row (uses the previously-dead space) and doubles as the
      // expand affordance. Body is the full deduped diagnostic list.
      // A row is expandable when it has a body: extra diagnostics or a
      // worked example (failing OR the new passing positive example).
      const expandable = hasMoreDetails || !!workedExample;
      return `
        <li class="test-result test-result-${escapeHtml(statusKind)}">
          <details class="test-result-row${expandable ? " is-expandable" : ""}">
            <summary class="test-result-summary-row">
              <div class="test-result-head">
                <span class="test-result-head-left">
                  ${renderStatusPill(statusKind, titleCase(result.status))}
                  <strong class="test-result-name">${escapeHtml(result.test_id)}</strong>
                  ${result.kind ? `<span class="test-result-kind">${escapeHtml(result.kind)}</span>` : ""}
                </span>
                <span class="test-result-meta">
                  ${countLabel ? `<span class="test-result-count">${escapeHtml(countLabel)}</span>` : ""}
                  ${expandable ? `<span class="test-result-toggle" aria-hidden="true"></span>` : ""}
                </span>
              </div>
              ${headline ? `<p class="test-result-headline">${escapeHtml(headline)}</p>` : ""}
            </summary>
            ${hasMoreDetails ? `
              <ul class="test-result-diagnostics">
                ${details.map((d) => `<li>${escapeHtml(d)}</li>`).join("")}
              </ul>
            ` : ""}
            ${workedExample}
          </details>
        </li>
      `;
    };

    // Per-scenario detail collapses by default. The headline summary
    // (rendered by ``renderLearnerGuidance`` from the populated
    // ``feedback`` object) is the primary view; drill into the full
    // list only when the learner wants to see every scenario.
    return `
      ${failed.length ? `
        <details class="review-guidance">
          <summary>${escapeHtml(`See all ${failed.length} failing check${failed.length === 1 ? "" : "s"}`)}</summary>
          <ul class="test-result-list">${failed.map(renderOne).join("")}</ul>
        </details>
      ` : ""}
      ${passed.length ? `
        <details class="review-guidance is-strong">
          <summary>${escapeHtml(`See all ${passed.length} passing check${passed.length === 1 ? "" : "s"}`)}</summary>
          <ul class="test-result-list">${passed.map(renderOne).join("")}</ul>
        </details>
      ` : ""}
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
          <h1>Loading your latest lab...</h1>
          <p class="focus-subcopy">We are fetching your project brief, workspace status, and review history.</p>
        </div>
      `;
      return;
    }

    if (!experience && selectedSummary) {
      learnerFocus.innerHTML = `
        <div class="hero-copy">
          <p class="eyebrow">Continue where you left off</p>
          <h1>${escapeHtml(cleanTitle(selectedSummary.course_title))}</h1>
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
          <h1>Pick a published lab and start building.</h1>
          <p>Enroll once and we will keep your shared workspace, project brief, and review history pinned here. ${escapeHtml(String(readyCourses))} learner-ready lab${readyCourses === 1 ? "" : "s"} below.</p>
          <div class="focus-actions">
            <a class="button primary" href="/courses">Browse published labs</a>
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

    // Workspace control reuses the shared VS Code glyph (module const).
    const labelWithIcon = VSCODE_GLYPH + escapeHtml(launchLabel);
    const workspaceRunning = experience?.workspace_session?.status === "running";
    const workspaceAction = session?.editor_url
      ? `
        <a class="button primary" href="${escapeHtml(session.editor_url)}" target="_blank" rel="noopener noreferrer">${labelWithIcon}</a>
      `
      : `
        <button
          class="button primary"
          type="button"
          data-action="launch-workspace"
          ${canLaunchWorkspace ? "" : "disabled"}
        >${labelWithIcon}</button>
      `;

    const eyebrowText = enrollment.status === "completed"
      ? "Lab complete"
      : latestSubmission
        ? "Resume your project"
        : `Project review areas: ${progress.total}`;

    const latestReviewText = latestSubmission
      ? `${latestSubmission.passed_tests}/${latestSubmission.total_tests} checks passed`
      : "Not submitted yet";

    const parsedSummary = parseCourseSummary(enrollment.course_summary);
    const shortOverview = shortenOverview(parsedSummary.overview)
      || "Build the shared project in one workspace and use the deliverable scorecard to see what still needs work.";

    learnerFocus.innerHTML = `
      <div class="focus-layout">
        <div class="focus-main">
          <p class="course-chip">${escapeHtml(cleanTitle(enrollment.course_title))}</p>
          <p class="eyebrow">${escapeHtml(eyebrowText)}</p>
          <h1>${escapeHtml(cleanTitle(enrollment.course_title))}</h1>
          <p class="focus-subcopy">${escapeHtml(shortOverview)}</p>
          ${renderSkillTags(parsedSummary.skills)}

          <dl class="deliverable-quickref" aria-label="Project at a glance">
            <div class="quickref-row">
              <dt>Run visible checks</dt>
              <dd>Inside the VS Code workspace, while you iterate.</dd>
            </div>
            <div class="quickref-row">
              <dt>Submit</dt>
              <dd>Use <strong>Submit project for review</strong> below — runs the full learner review checks.</dd>
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
                  <span class="info-pill warn"><strong>Full review</strong></span>
                </div>
                <p>Submit the whole project to run the learner review checks. Feedback comes back grouped by deliverable.</p>
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
      deliverablesBody.innerHTML = '<p class="empty">Open a lab to see its deliverables.</p>';
      submissionHistory.classList.add("hidden");
      submissionHistoryBody.innerHTML = "<p class=\"empty\">Open a lab to see its review history.</p>";
      return;
    }

    const enrollment = experience.enrollment;
    const progress = computeCourseProgress(experience);
    const deliverables = experience.deliverables || [];
    const latestReport = experience.latest_assignment_report;
    const latestSubmission = experience.latest_assignment_submission;

    // Outcome-mode courses always ship a single ``outcome_main``
    // deliverable; in that case the per-deliverable Project Review
    // panel is redundant with the header (which already shows the
    // latest score) and the Summary feedback panel below (which
    // surfaces the actionable detail). Hide it so the page reads
    // cleanly. Multi-deliverable legacy courses keep the panel
    // because it's the only place the per-deliverable scorecard
    // lives.
    if (deliverables.length <= 1) {
      deliverablesPanel.classList.add("hidden");
    } else {
      deliverablesPanel.classList.remove("hidden");
    }
    submissionHistory.classList.remove("hidden");
    deliverablesTitle.textContent = `${progress.total} deliverable${progress.total === 1 ? "" : "s"}`;
    deliverablesCaption.textContent = latestReport
      ? `${latestReport.passed_tests}/${latestReport.total_tests} review checks are passing in the latest review run.`
      : "Submit the full project to get a scorecard for each deliverable.";

    deliverablesBody.innerHTML = deliverables.map((deliverable) => {
      const latestGrade = deliverable.latest_submission;
      const latestGradeSolved = latestGrade
        ? isSolved(latestGrade.passed_tests, latestGrade.total_tests, latestGrade.status)
        : false;
      const statusLabel = latestGrade
        ? (latestGradeSolved ? "Solved" : "Needs work")
        : "Not reviewed";
      return `
        <div class="deliverable-row">
          <span class="deliverable-row-index">${escapeHtml(String(deliverable.deliverable_index))}</span>
          <div class="deliverable-row-copy">
            <h4>${escapeHtml(deliverable.title)}</h4>
            <p>${escapeHtml(deliverable.objective || "")}</p>
            <div class="deliverable-row-meta">
              ${renderStatusPill(latestGrade ? (latestGradeSolved ? "passed" : "blocked") : "neutral", statusLabel)}
              ${latestGrade ? renderInfoPill("Last review", `${latestGrade.passed_tests}/${latestGrade.total_tests}`) : ""}
            </div>
          </div>
        </div>
      `;
    }).join("");

    const latestSubmissionSolved = latestSubmission
      ? isSolved(latestSubmission.passed_tests, latestSubmission.total_tests, latestSubmission.status)
      : false;
    const latestCard = latestSubmission ? `
      <div class="latest-grade-card ${latestSubmissionSolved ? "passed" : "needs-work"}">
        <p class="card-eyebrow">Latest project review</p>
        <div class="latest-grade-row">
          <strong>${escapeHtml(`${latestSubmission.passed_tests}/${latestSubmission.total_tests} tests passed`)}</strong>
          ${renderStatusPill(latestSubmissionSolved ? "passed" : "blocked", latestSubmissionSolved ? "Solved" : titleCase(latestSubmission.status))}
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
            ${latestReport.review_areas.map((reviewArea) => {
              const gr = reviewArea.grade_report;
              // Tri-state status: passed (100%), strong (>=83%), needs-work.
              // 83% matches the user's "15+/18 turns green" calibration —
              // captures "good solution" without requiring perfection.
              const passRate = gr.total_tests ? (gr.passed_tests / gr.total_tests) : 0;
              const isPassed = isSolved(gr.passed_tests, gr.total_tests, gr.status);
              const isStrong = !isPassed && passRate >= 0.83;
              const pillKind = isPassed ? "passed" : (isStrong ? "passed" : "blocked");
              const pillLabel = isPassed ? "Ready" : (isStrong ? "Strong" : "Needs work");
              return `
              <div class="submission-item ${isStrong ? "is-strong" : ""} ${isPassed ? "is-ready" : ""}">
                <strong>${escapeHtml(reviewArea.title)}</strong>
                <p>${escapeHtml(reviewArea.objective)}</p>
                <div class="submission-item-meta">
                  ${renderStatusPill(pillKind, pillLabel)}
                  ${renderInfoPill("Checks", `${gr.passed_tests}/${gr.total_tests}`)}
                </div>
                ${reviewArea.feedback ? renderLearnerGuidance(reviewArea.feedback, { isStrong, isReady: isPassed }) : ""}
                ${renderTestResults(gr)}
              </div>
              `;
            }).join("")}
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
          <h3>No labs yet</h3>
          <p>Choose a learner-ready lab below and we will pin it here once you enroll.</p>
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
            <h3>${escapeHtml(cleanTitle(item.course_title))}</h3>
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
            <h3>${escapeHtml(cleanTitle(course.title))}</h3>
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
      syncLabTutor();
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
      syncLabTutor();
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

  // ── Lab Tutor widget integration ──────────────────────────────────────────
  // Dynamically injects lab-tutor.js and mounts/unmounts the floating chat
  // widget based on the active enrollment's course lab_tutor_enabled flag.

  let labTutorScriptInjected = false;

  function labTutorEnabledForCurrentEnrollment() {
    const experience = uiState.currentExperience;
    if (!experience?.enrollment?.course_run_id) return false;
    const courseRunId = experience.enrollment.course_run_id;
    const catalog = state.catalog?.courses || [];
    const course = catalog.find((c) => c.course_run_id === courseRunId);
    return course?.lab_tutor_enabled === true;
  }

  function syncLabTutor() {
    const enabled = labTutorEnabledForCurrentEnrollment();
    if (!enabled) {
      if (typeof window.__labTutorUnmount === "function") {
        window.__labTutorUnmount();
      }
      return;
    }

    const experience = uiState.currentExperience;
    const enrollmentId = experience?.enrollment?.id || "anon";
    const courseTitle = experience?.enrollment?.course_title || "";

    if (!labTutorScriptInjected) {
      labTutorScriptInjected = true;
      const script = document.createElement("script");
      script.src = "/static/lab-tutor.js";
      script.dataset.sessionId = "lms-" + enrollmentId;
      script.dataset.assignmentTitle = courseTitle;
      script.dataset.enrollmentId = enrollmentId;
      script.addEventListener("load", () => {
        if (typeof window.__labTutorMount === "function") {
          window.__labTutorMount({
            sessionId: "lms-" + enrollmentId,
            assignmentTitle: courseTitle,
            enrollmentId: enrollmentId,
          });
        }
      });
      document.head.appendChild(script);
    } else if (typeof window.__labTutorMount === "function") {
      window.__labTutorMount({
        sessionId: "lms-" + enrollmentId,
        assignmentTitle: courseTitle,
        enrollmentId: enrollmentId,
      });
    }
  }

  renderAll();
  syncLabTutor();

  const initialEnrollment = selectedEnrollmentSummary();
  if (initialEnrollment?.id) {
    loadEnrollment(initialEnrollment.id).then(() => {
      syncLabTutor();
    }).catch(() => {
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
        syncLabTutor();
      } catch (_error) {
        renderAll();
      }
      return;
    }

    syncLabTutor();
    renderAll();
  });
})();
