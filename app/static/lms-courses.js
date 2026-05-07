(() => {
  const stateElement = document.getElementById("lms-state");
  const state = stateElement?.textContent ? JSON.parse(stateElement.textContent) : {};

  const pageMessage = document.getElementById("page-message");
  const enrollmentList = document.getElementById("enrollment-list");
  const catalogList = document.getElementById("catalog-list");
  const catalogCaption = document.getElementById("catalog-caption");
  const myCoursesPane = document.getElementById("my-courses-pane");
  const allCoursesPane = document.getElementById("all-courses-pane");
  const tabButtons = Array.from(document.querySelectorAll("[data-courses-tab]"));

  const url = new URL(window.location.href);
  const initialTab = url.searchParams.get("view") === "all" ? "all" : "my";

  const uiState = {
    activeTab: initialTab,
    busyAction: null,
    busyTarget: null,
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
      if (absSeconds < 60) return formatter.format(Math.round(diffMs / 1000), "second");
      const absMinutes = Math.round(absSeconds / 60);
      if (absMinutes < 60) return formatter.format(Math.round(diffMs / 60000), "minute");
      const absHours = Math.round(absMinutes / 60);
      if (absHours < 48) return formatter.format(Math.round(diffMs / 3600000), "hour");
      const absDays = Math.round(absHours / 24);
      return formatter.format(Math.round(diffMs / 86400000), "day");
    } catch (_error) {
      return formatDate(value);
    }
  }

  function computeProgress(item) {
    const total = Number(item.module_count || 0);
    const completed = Number(item.completed_module_count || 0);
    const positionPercent = total
      ? Math.round(((item.status === "completed" ? total : Number(item.current_module_index || 0) || completed) / total) * 100)
      : 0;
    return { total, completed, positionPercent: Math.max(0, Math.min(positionPercent, 100)) };
  }

  function renderStatusPill(kind, label) {
    return `<span class="status-pill ${escapeHtml(kind)}">${escapeHtml(label)}</span>`;
  }

  function renderInfoPill(label, value) {
    return `<span class="info-pill"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></span>`;
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

  function existingEnrollmentForCourse(courseRunId) {
    return (state.enrollments?.enrollments || []).find((item) => item.course_run_id === courseRunId) || null;
  }

  function setActiveTab(next) {
    if (next !== "my" && next !== "all") return;
    uiState.activeTab = next;
    tabButtons.forEach((btn) => {
      const active = btn.dataset.coursesTab === next;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", String(active));
    });
    myCoursesPane.classList.toggle("hidden", next !== "my");
    allCoursesPane.classList.toggle("hidden", next !== "all");

    const u = new URL(window.location.href);
    if (next === "all") {
      u.searchParams.set("view", "all");
    } else {
      u.searchParams.delete("view");
    }
    window.history.replaceState({}, "", `${u.pathname}${u.search}${u.hash}`);
  }

  function setMessage(kind, text) {
    if (!text) {
      pageMessage.className = "message";
      pageMessage.textContent = "";
      return;
    }
    pageMessage.className = `message visible ${kind}`;
    pageMessage.textContent = text;
  }

  function renderEnrollments() {
    const enrollments = state.enrollments?.enrollments || [];
    if (!enrollments.length) {
      enrollmentList.innerHTML = `
        <div class="summary-card empty-state">
          <h3>No courses yet</h3>
          <p>Switch to <strong>All courses</strong> to enroll in something. Once you do, your active course will pin here.</p>
        </div>
      `;
      return;
    }

    enrollmentList.innerHTML = enrollments.map((item) => {
      const progress = computeProgress(item);
      const summaryLine = item.current_module_title
        ? `Module ${item.current_module_index} · ${item.current_module_title}`
        : "Course complete";
      return `
        <div class="course-row">
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
            <a class="button primary" href="/?enrollment=${escapeHtml(item.id)}">Resume</a>
          </div>
        </div>
      `;
    }).join("");
  }

  function renderCatalog() {
    const courses = state.catalog?.courses || [];
    const enrolled = new Set((state.enrollments?.enrollments || []).map((item) => item.course_run_id));
    const ready = courses.filter((c) => c.supported_for_lms).length;

    catalogCaption.textContent = courses.length
      ? `${courses.length} published course${courses.length === 1 ? "" : "s"} · ${ready} learner-ready today.`
      : "No published courses yet.";

    if (!courses.length) {
      catalogList.innerHTML = `<div class="summary-card empty-state"><h3>No published courses yet</h3><p>Publish a course from the builder first, then learners can enroll from here.</p></div>`;
      return;
    }

    catalogList.innerHTML = courses.map((course) => {
      const existing = existingEnrollmentForCourse(course.course_run_id);
      const isBlocked = !course.supported_for_lms;
      const isAlreadyEnrolled = enrolled.has(course.course_run_id);
      const reasonCopy = course.supported_for_lms
        ? (isAlreadyEnrolled
          ? "You are already enrolled in this course."
          : "Enroll now and jump straight into the current module workspace.")
        : (course.support_reason || "This course is being prepared and is not ready for learners yet.");
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
            <p class="catalog-helper ${isBlocked ? "warning" : ""}">${escapeHtml(reasonCopy)}</p>
          </div>
          <div class="catalog-card-footer">
            ${existing && !isBlocked
              ? `<a class="button" href="/?enrollment=${escapeHtml(existing.id)}">Resume learner progress</a>`
              : (isBusy("enroll", course.course_run_id)
                ? `<button class="button primary" type="button" disabled>Enrolling...</button>`
                : `<button class="button ${course.supported_for_lms ? "primary" : ""}" type="button" data-enroll-course="${escapeHtml(course.course_run_id)}" ${course.supported_for_lms ? "" : "disabled"}>${escapeHtml(course.supported_for_lms ? "Enroll and start" : "Preparing for learners")}</button>`)
            }
          </div>
        </div>
      `;
    }).join("");
  }

  function isBusy(action, target = null) {
    return uiState.busyAction === action && (target === null || uiState.busyTarget === target);
  }

  async function refreshCatalog() {
    const r = await fetch(state.catalog_url);
    if (r.ok) {
      state.catalog = await r.json();
      renderCatalog();
    }
  }

  async function refreshEnrollments() {
    const r = await fetch(state.enrollments_url);
    if (r.ok) {
      state.enrollments = await r.json();
      renderEnrollments();
      renderCatalog();
    }
  }

  async function handleEnroll(courseRunId) {
    uiState.busyAction = "enroll";
    uiState.busyTarget = courseRunId;
    renderCatalog();
    try {
      const r = await fetch(state.enrollments_url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ course_run_id: courseRunId }),
      });
      if (!r.ok) {
        const text = await r.text();
        throw new Error(text || "Could not enroll.");
      }
      const enrollment = await r.json();
      window.location.assign(`/?enrollment=${encodeURIComponent(enrollment.id)}`);
    } catch (e) {
      setMessage("error", e instanceof Error ? e.message : "Could not enroll.");
    } finally {
      uiState.busyAction = null;
      uiState.busyTarget = null;
      renderCatalog();
    }
  }

  document.addEventListener("click", (event) => {
    const target = event.target instanceof Element
      ? event.target.closest("[data-courses-tab],[data-enroll-course]")
      : null;
    if (!(target instanceof HTMLElement)) return;

    if (target.dataset.coursesTab) {
      setActiveTab(target.dataset.coursesTab);
      return;
    }
    if (target.dataset.enrollCourse) {
      handleEnroll(target.dataset.enrollCourse);
    }
  });

  setActiveTab(uiState.activeTab);
  renderEnrollments();
  renderCatalog();
  refreshEnrollments();
  refreshCatalog();
})();
