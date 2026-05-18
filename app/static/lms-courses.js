(() => {
  const stateElement = document.getElementById("lms-state");
  const state = stateElement?.textContent ? JSON.parse(stateElement.textContent) : {};

  const pageMessage = document.getElementById("page-message");
  const labsList = document.getElementById("labs-list");

  const uiState = { busyCourse: null };

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

  function setMessage(kind, text) {
    if (!text) {
      pageMessage.className = "message";
      pageMessage.textContent = "";
      return;
    }
    pageMessage.className = `message visible ${kind}`;
    pageMessage.textContent = text;
  }

  function enrollmentForCourse(courseRunId) {
    return (state.enrollments?.enrollments || []).find(
      (item) => item.course_run_id === courseRunId
    ) || null;
  }

  // One merged list: enrolled labs on top (Resume), everything else
  // below (Enrol). No tabs, no progress bar, no deliverable counts.
  function render() {
    const courses = state.catalog?.courses || [];
    const enrollments = state.enrollments?.enrollments || [];
    const enrolledByCourse = new Map(
      enrollments.map((e) => [e.course_run_id, e])
    );

    const enrolled = [];
    const available = [];
    for (const course of courses) {
      if (enrolledByCourse.has(course.course_run_id)) {
        enrolled.push(course);
      } else {
        available.push(course);
      }
    }

    if (!courses.length) {
      labsList.innerHTML = `<div class="summary-card empty-state"><h3>No labs yet</h3><p>Published labs will show up here.</p></div>`;
      return;
    }

    const cardFor = (course) => {
      const enrollment = enrolledByCourse.get(course.course_run_id) || null;
      const isEnrolled = Boolean(enrollment);
      const isReady = Boolean(course.supported_for_lms);
      const busy = uiState.busyCourse === course.course_run_id;

      // Whole card is a click target. data-* drive the click handler.
      const attrs = isEnrolled
        ? `data-go="/?enrollment=${escapeHtml(enrollment.id)}"`
        : (isReady && !busy
          ? `data-enroll="${escapeHtml(course.course_run_id)}"`
          : "");
      const cta = isEnrolled
        ? `<span class="button primary">Resume</span>`
        : (busy
          ? `<span class="button primary is-busy">Enrolling…</span>`
          : (isReady
            ? `<span class="button primary">Enrol</span>`
            : `<span class="button" aria-disabled="true">Preparing…</span>`));
      // The enrollment summary here only carries status /
      // completed_deliverable_count / deliverable_count — it does NOT
      // expose raw passed_tests, so the >=15-marks "Solved" rule can't
      // be evaluated per-mark. Conservatively show "Solved" only when
      // the enrollment is fully completed; otherwise keep "Enrolled".
      const isSolvedEnrollment = isEnrolled && enrollment.status === "completed";
      const stateLabel = isEnrolled
        ? (isSolvedEnrollment ? "Solved" : "Enrolled")
        : (isReady ? "Available" : "Preparing");
      const stateKind = isEnrolled ? "passed" : (isReady ? "ready" : "not-ready");

      return `
        <div class="lab-card ${isEnrolled ? "is-enrolled" : ""} ${!isReady && !isEnrolled ? "is-blocked" : ""} ${attrs ? "is-clickable" : ""}"
             ${attrs} role="${attrs ? "button" : ""}" ${attrs ? 'tabindex="0"' : ""}>
          <div class="lab-card-main">
            <span class="status-pill ${stateKind}">${escapeHtml(stateLabel)}</span>
            <h3>${escapeHtml(cleanTitle(course.title))}</h3>
            <p>${escapeHtml(course.summary)}</p>
          </div>
          <div class="lab-card-cta">${cta}</div>
        </div>
      `;
    };

    const sections = [];
    if (enrolled.length) {
      sections.push(
        `<h2 class="lab-section-title">In progress</h2>` +
        enrolled.map(cardFor).join("")
      );
    }
    if (available.length) {
      sections.push(
        `<h2 class="lab-section-title">Browse labs</h2>` +
        available.map(cardFor).join("")
      );
    }
    labsList.innerHTML = sections.join("");
  }

  async function refresh() {
    try {
      const [c, e] = await Promise.all([
        fetch(state.catalog_url, { credentials: "same-origin" }),
        fetch(state.enrollments_url, { credentials: "same-origin" }),
      ]);
      if (c.ok) state.catalog = await c.json();
      if (e.ok) state.enrollments = await e.json();
      render();
    } catch (_err) {
      /* keep server-rendered state */
    }
  }

  async function handleEnroll(courseRunId) {
    uiState.busyCourse = courseRunId;
    render();
    try {
      const r = await fetch(state.enrollments_url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ course_run_id: courseRunId }),
      });
      if (!r.ok) {
        const text = await r.text();
        throw new Error(text || "Could not enrol.");
      }
      const enrollment = await r.json();
      window.location.assign(`/?enrollment=${encodeURIComponent(enrollment.id)}`);
    } catch (e) {
      setMessage("error", e instanceof Error ? e.message : "Could not enrol.");
      uiState.busyCourse = null;
      render();
    }
  }

  function onActivate(card) {
    if (!(card instanceof HTMLElement)) return;
    const go = card.getAttribute("data-go");
    if (go) {
      window.location.assign(go);
      return;
    }
    const enroll = card.getAttribute("data-enroll");
    if (enroll) handleEnroll(enroll);
  }

  labsList.addEventListener("click", (event) => {
    const card = event.target instanceof Element
      ? event.target.closest(".lab-card.is-clickable")
      : null;
    if (card) onActivate(card);
  });

  labsList.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const card = event.target instanceof Element
      ? event.target.closest(".lab-card.is-clickable")
      : null;
    if (card) {
      event.preventDefault();
      onActivate(card);
    }
  });

  render();
  refresh();
})();
