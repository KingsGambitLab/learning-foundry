(() => {
      const stateElement = document.getElementById("dashboard-state");
      const state = stateElement?.textContent ? JSON.parse(stateElement.textContent) : {};
      let currentCourseRun = null;
      let currentReview = null;
      let currentEvents = [];
      let currentWorkflowDetails = {};
      let currentLearnerEval = null;
      let currentCreatorFeedback = [];
      let recentDraftRuns = [];
      let currentTab = "create";
      let draftPollHandle = null;
      let draftLoadInProgress = false;
      let pendingDraftId = null;
      const workflowDetailCache = new Map();
      const workflowRejectMode = new Set();
      const workflowCommentCache = new Map();

      const generationBanner = document.getElementById("generation-banner");
      const generationHeading = document.getElementById("generation-heading");
      const generationBadge = document.getElementById("generation-badge");
      const generationMessage = document.getElementById("generation-message");
      const generationMeta = document.getElementById("generation-meta");
      const generationSetupToggle = document.getElementById("generation-setup-toggle");
      const generationSetup = document.getElementById("generation-setup");
      const formMessage = document.getElementById("form-message");
      const workspaceLayout = document.querySelector(".workspace-layout");
      const workspaceRail = document.querySelector(".workspace-rail");
      const draftContextBar = document.getElementById("draft-context-bar");
      const allDraftsButton = document.getElementById("all-drafts-button");
      const draftSwitcher = document.getElementById("draft-switcher");
      const draftSwitcherTitle = document.getElementById("draft-switcher-title");
      const draftSwitcherList = document.getElementById("draft-switcher-list");
      const draftInboxPanel = document.getElementById("draft-inbox-panel");
      const workflowProgressTitle = document.getElementById("workflow-progress-title");
      const workflowProgressCopy = document.getElementById("workflow-progress-copy");
      const workflowProgressCount = document.getElementById("workflow-progress-count");
      const workflowReviewProgress = document.getElementById("workflow-review-progress");
      const workflowProgressBar = document.getElementById("workflow-progress-bar");
      const workflowProgressNext = document.getElementById("workflow-progress-next");
      const workflowSteps = Array.from(document.querySelectorAll("[data-workflow-step]"));
      const createTabButton = document.getElementById("tab-create");
      const draftsTabButton = document.getElementById("tab-drafts");
      const createTabPane = document.getElementById("tab-pane-create");
      const draftsTabPane = document.getElementById("tab-pane-drafts");
      const createDraftShortcut = document.getElementById("create-draft-shortcut");
      const draftSearchInput = document.getElementById("draft-search");
      const goalField = document.getElementById("goal");
      const goalCount = document.getElementById("goal-count");
      const outcomesCount = document.getElementById("outcomes-count");
      const generateButton = document.getElementById("generate-button");
      const suggestOutcomesButton = document.getElementById("suggest-outcomes-button");
      const creatorStep1Next = document.getElementById("creator-step-1-next");
      const creatorStep2Next = document.getElementById("creator-step-2-next");
      const creatorStep3Next = document.getElementById("creator-step-3-next");
      const creatorAddOutcome = document.getElementById("creator-add-outcome");
      const creatorOutcomesList = document.getElementById("creator-outcomes-list");
      const creatorPlanPreview = document.getElementById("creator-plan-preview");
      const creatorDataSourceCount = document.getElementById("creator-data-source-count");
      const creatorDataSourcePurpose = document.getElementById("creator-data-source-purpose");
      const creatorDataSourceFileInput = document.getElementById("creator-data-source-file");
      const creatorUploadDataSourceButton = document.getElementById("creator-upload-data-source");
      const creatorSelectedDataSources = document.getElementById("creator-selected-data-sources");
      const creatorAssetLibrary = document.getElementById("creator-asset-library");
      const creatorStepTitle = document.getElementById("creator-step-title");
      const creatorPanes = Array.from(document.querySelectorAll(".creator-pane"));
      const creatorStepper = document.getElementById("creator-stepper");
      const creatorState = {
        step: 1,
        goal: "",
        outcomes: [],
        choices: { starter_type: "partial_implementation", primary_database: "postgres", cache_backend: "redis", tech_stack: [], data_sources: [] },
        assets: [],
        plan: null,
      };
      const results = document.getElementById("results");
      const draftStageBadge = document.getElementById("draft-stage-badge");
      const draftStatusSummary = document.getElementById("draft-status-summary");
      const draftUnblockSummary = document.getElementById("draft-unblock-summary");
      const sourceBadge = document.getElementById("source-badge");
      const planSummary = document.getElementById("plan-summary");
      const moduleList = document.getElementById("module-list");
      const courseSummary = document.getElementById("course-summary");
      const reviewMetrics = document.getElementById("review-metrics");
      const progressTimeline = document.getElementById("progress-timeline");
      const reviewBlockers = document.getElementById("review-blockers");
      const reviewActions = document.getElementById("review-actions");
      const publishedVersions = document.getElementById("published-versions");
      const draftActivity = document.getElementById("draft-activity");
      const linkedWorkflows = document.getElementById("linked-workflows");
      const recentDrafts = document.getElementById("recent-drafts");
      const materializeButton = document.getElementById("materialize-button");
      const publishButton = document.getElementById("publish-button");
      const createRevisionButton = document.getElementById("create-revision-button");
      const resetLocalButton = document.getElementById("reset-local-button");
      const activeDraftStorageKey = "courseGenCurrentDraftId";
      const activeTabStorageKey = "courseGenActiveTab";
      const createRevisionUrlTemplate = state.create_revision_url_template || "/v1/course-runs/{course_run_id}/create-revision-async";
      const materializeUrlTemplate = state.materialize_url_template || "/v1/course-runs/{course_run_id}/materialize-async";
      const publishUrlTemplate = state.publish_url_template || "/v1/course-runs/{course_run_id}/publish-async";

      function setMessage(element, kind, text) {
        element.className = "message";
        if (!text) {
          element.textContent = "";
          return;
        }
        element.textContent = text;
        element.classList.add("visible", kind);
      }

      function pill(text) {
        return `<span class="small-pill">${escapeHtml(text)}</span>`;
      }

      function formatUsd(value) {
        const amount = Number(value || 0);
        if (!Number.isFinite(amount) || amount <= 0) return "$0.00";
        if (amount >= 100) return `$${amount.toFixed(0)}`;
        if (amount >= 10) return `$${amount.toFixed(2)}`;
        if (amount >= 1) return `$${amount.toFixed(3)}`;
        return `$${amount.toFixed(4)}`;
      }

      function usageSpendLabel(usage) {
        const spend = Number(usage?.estimated_cost_usd || 0);
        const requests = Number(usage?.request_count || 0);
        if (!requests) {
          return "AI spend: $0.00";
        }
        return `AI spend: ${formatUsd(spend)} across ${pluralize(requests, "request")}`;
      }

      function escapeHtml(value) {
        return String(value)
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#39;");
      }

      function readUrlState() {
        const url = new URL(window.location.href);
        return {
          draftId: url.searchParams.get("draft"),
          tab: url.searchParams.get("tab"),
        };
      }

      function writeUrlState(options = {}, historyMode = "replace") {
        const url = new URL(window.location.href);
        const draftId = options.draftId !== undefined ? options.draftId : currentCourseRun?.id || null;
        const tab = options.tab !== undefined ? options.tab : currentTab;
        if (draftId) {
          url.searchParams.set("draft", draftId);
        } else {
          url.searchParams.delete("draft");
        }
        if (tab && (tab !== "create" || draftId)) {
          url.searchParams.set("tab", tab);
        } else {
          url.searchParams.delete("tab");
        }
        const nextUrl = `${url.pathname}${url.search}${url.hash}`;
        if (historyMode === "push") {
          window.history.pushState({}, "", nextUrl);
          return;
        }
        window.history.replaceState({}, "", nextUrl);
      }

      function formatDate(value) {
        try {
          return new Date(value).toLocaleString();
        } catch (_error) {
          return value || "";
        }
      }

      function pluralize(count, singular, plural = `${singular}s`) {
        return `${count} ${count === 1 ? singular : plural}`;
      }

      function formatRelativeTime(value) {
        if (!value) return "—";
        try {
          const date = new Date(value);
          const diffMs = date.getTime() - Date.now();
          const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
          const absSec = Math.round(Math.abs(diffMs) / 1000);
          if (absSec < 60) return formatter.format(Math.round(diffMs / 1000), "second");
          const absMin = Math.round(absSec / 60);
          if (absMin < 60) return formatter.format(Math.round(diffMs / 60000), "minute");
          const absHr = Math.round(absMin / 60);
          if (absHr < 48) return formatter.format(Math.round(diffMs / 3600000), "hour");
          const absDay = Math.round(absHr / 24);
          return formatter.format(Math.round(diffMs / 86400000), "day");
        } catch (_e) {
          return formatDate(value);
        }
      }

      function draftIdentityBadges(run) {
        const tags = [];
        if (run.status === "published") tags.push(`<span class="draft-row-tag tag-live">Live</span>`);
        else if (run.stage === "ready_to_publish") tags.push(`<span class="draft-row-tag tag-ready">Ready</span>`);
        else if (run.active_operation) tags.push(`<span class="draft-row-tag tag-busy">${escapeHtml(friendlyOperation(run.active_operation))}</span>`);
        else if (run.stage === "blocked" || run.last_error) tags.push(`<span class="draft-row-tag tag-blocked">Blocked</span>`);
        else if (run.status === "awaiting_human" || run.stage === "awaiting_course_review") tags.push(`<span class="draft-row-tag tag-waiting">Needs you</span>`);
        return tags.join("");
      }

      function draftSnippet(run) {
        const goal = run.goal || run.summary || "";
        if (!goal) return "";
        const trimmed = goal.replace(/\s+/g, " ").trim();
        return trimmed.length > 90 ? `${trimmed.slice(0, 87)}…` : trimmed;
      }

      function buildDraftOptionMarkup(run, options = {}) {
        const summary = summarizeDraftListState(run);
        const isSelected = currentCourseRun?.id === run.id;
        const updated = formatDate(run.updated_at);
        const updatedRelative = formatRelativeTime(run.updated_at);
        if (options.compact) {
          return `
            <button
              class="draft-switcher-option ${isSelected ? "selected" : ""}"
              type="button"
              data-switch-course-run="${escapeHtml(run.id)}"
            >
              <span class="draft-switcher-option-title">${escapeHtml(run.title)}</span>
              <span class="draft-switcher-option-meta">${escapeHtml(summary.label)} • ${escapeHtml(updatedRelative)} • ${escapeHtml(formatUsd(run.ai_usage?.estimated_cost_usd || 0))}</span>
            </button>
          `;
        }
        const snippet = draftSnippet(run);
        return `
          <button
            class="draft-row ${isSelected ? "selected" : ""}"
            type="button"
            data-load-course-run="${escapeHtml(run.id)}"
            title="${escapeHtml(`Last updated ${updated}`)}"
          >
            <div class="draft-row-main">
              <div class="draft-row-heading">
                <h4>${escapeHtml(run.title)}</h4>
                <div class="draft-row-tags">
                  ${draftIdentityBadges(run)}
                </div>
              </div>
              <span class="badge ${escapeHtml(summary.kind)}">${escapeHtml(summary.label)}</span>
            </div>
            ${snippet ? `<p class="draft-row-snippet">${escapeHtml(snippet)}</p>` : ""}
            <div class="draft-row-meta">
              <span class="draft-row-updated">${escapeHtml(updatedRelative)}</span>
              <span><strong>${escapeHtml(String(run.deliverable_count))}</strong> deliverable${run.deliverable_count === 1 ? "" : "s"}</span>
              <span>${escapeHtml(usageSpendLabel(run.ai_usage))}</span>
              ${run.base_archetype ? `<span>${escapeHtml(friendlyBuildPattern(run.base_archetype))}</span>` : ""}
            </div>
          </button>
        `;
      }

      function renderDraftSwitcherMenu(runs) {
        if (!runs.length) {
          draftSwitcherList.innerHTML = `<div class="review-item"><p>No drafts yet. Start building and we will list them here.</p></div>`;
          return;
        }
        draftSwitcherList.innerHTML = runs.map((run) => buildDraftOptionMarkup(run, { compact: true })).join("");
      }

      function updateDocumentTitle() {
        if (currentTab === "drafts" && currentCourseRun?.title) {
          document.title = `${currentCourseRun.title} · Course Builder`;
          return;
        }
        if (draftLoadInProgress && pendingDraftId) {
          document.title = "Loading draft · Course Builder";
          return;
        }
        document.title = currentTab === "drafts"
          ? "Drafts · Course Builder"
          : "Create Course Draft";
      }

      function updateWorkspaceChrome() {
        const hasOpenDraft = Boolean(currentCourseRun);
        const pendingRun = recentDraftRuns.find((run) => run.id === pendingDraftId) || null;
        const hasPendingDraft = currentTab === "drafts" && !hasOpenDraft && (draftLoadInProgress || Boolean(pendingDraftId));
        const showInboxRail = currentTab === "drafts" && !hasOpenDraft && !hasPendingDraft;
        const focusedDraftView = currentTab === "drafts" && hasOpenDraft;
        const showDraftContext = currentTab === "drafts" && (hasOpenDraft || hasPendingDraft);

        workspaceLayout?.classList.toggle("has-rail", showInboxRail);
        workspaceLayout?.classList.toggle("is-loading-draft", hasPendingDraft);
        workspaceRail?.classList.toggle("hidden", !showInboxRail);
        draftContextBar?.classList.toggle("hidden", !showDraftContext);
        draftContextBar?.classList.toggle("loading", hasPendingDraft && !hasOpenDraft);
        draftInboxPanel?.classList.toggle("hidden", currentTab !== "drafts" || hasOpenDraft || hasPendingDraft);
        document.body.classList.toggle("dashboard-drafts-mode", currentTab === "drafts");
        document.body.classList.toggle("dashboard-create-mode", currentTab === "create");
        document.body.classList.toggle("dashboard-has-active-draft", focusedDraftView);
        document.body.classList.toggle("dashboard-draft-loading", hasPendingDraft);

        if (draftSwitcher instanceof HTMLDetailsElement && !hasOpenDraft) {
          draftSwitcher.open = false;
        }

        draftSwitcherTitle.textContent = hasOpenDraft
          ? (currentCourseRun.title || "Selected draft")
          : hasPendingDraft
            ? (pendingRun?.title || "Loading draft…")
            : "Selected draft";
        updateDocumentTitle();
      }

      function clearSelectedDraft(options = {}) {
        resetDraftSelection();
        window.localStorage.removeItem(activeDraftStorageKey);
        setActiveTab(options.tab || "drafts", { updateUrl: false });
        writeUrlState({ draftId: null, tab: options.tab || "drafts" }, options.historyMode || "push");
        renderRecentDrafts(recentDraftRuns);
        updateWorkspaceChrome();
      }

      function titleCase(value) {
        return String(value || "")
          .replaceAll("_", " ")
          .replaceAll("-", " ")
          .replace(/\s+/g, " ")
          .trim()
          .replace(/\b\w/g, (char) => char.toUpperCase());
      }

      function friendlyCourseShape(packageType) {
        const labels = {
          progressive_codebase_course: "One codebase across deliverables",
          survey_course: "Separate assignment per deliverable",
        };
        return labels[packageType] || titleCase(packageType);
      }

      function friendlyBuildPattern(archetypeId) {
        const labels = {
          task_agent_service: "Agentic system",
          grounded_rag_service: "Grounded RAG system",
          retrieval_service: "Retrieval system",
          stateful_service: "Stateful backend",
          protocol_server: "Protocol server",
        };
        return labels[archetypeId] || titleCase(archetypeId);
      }

      function friendlyUseCase(domainPack) {
        const labels = {
          support_triage: "Support triage",
          oncall_copilot: "On-call copilot",
          rfp_drafter: "RFP drafting",
          analyst_sql: "Analyst SQL",
          qbr_prep: "QBR prep",
          investment_memo: "Investment memo",
          clinical_case_triage: "Clinical case triage",
        };
        return labels[domainPack] || titleCase(domainPack);
      }

      function friendlyFocusArea(overlayId) {
        const labels = {
          productionization_overlay: "Production readiness",
          scale_slo_overlay: "Scale and SLOs",
          freshness_overlay: "Freshness",
          adversarial_overlay: "Adversarial robustness",
        };
        return labels[overlayId] || titleCase(overlayId);
      }

      function friendlyGate(gate) {
        const labels = {
          gate_1_spec_review: "Review assignment spec",
          gate_2_progression_review: "Review deliverables",
          gate_3_pre_publish: "Final assignment review",
        };
        return labels[gate] || titleCase(gate);
      }

      function friendlyStarterType(starterType) {
        const labels = {
          bare_stub: "Bare scaffold",
          partial_implementation: "Partial implementation",
          working_buggy: "Working but buggy",
          working_suboptimal: "Working but suboptimal",
        };
        return labels[starterType] || titleCase(starterType);
      }

      function friendlyRisk(riskClass) {
        const labels = {
          standard: "Standard risk",
          review_required: "Review required",
          high_stakes: "High stakes",
        };
        return labels[riskClass] || titleCase(riskClass);
      }

      function friendlyToolSafety(safety) {
        const labels = {
          read: "Read only",
          write: "Writes data",
          irreversible: "Irreversible",
        };
        return labels[safety] || titleCase(safety);
      }

      function friendlyOperation(operation) {
        const labels = {
          generation: "Building the first draft",
          revision: "Creating a new version draft",
          materialize: "Preparing the review package",
          publish: "Publishing this course version",
        };
        return labels[operation] || titleCase(operation);
      }

      function humanizeAuthorCopy(text) {
        if (!text) return "";
        return String(text)
          .replaceAll("linked assignment workflow", "assignment")
          .replaceAll("linked assignment workflows", "assignments")
          .replaceAll("HIL gates", "review steps")
          .replaceAll("gate_1_spec_review", friendlyGate("gate_1_spec_review"))
          .replaceAll("gate_2_progression_review", friendlyGate("gate_2_progression_review"))
          .replaceAll("gate_3_pre_publish", friendlyGate("gate_3_pre_publish"));
      }

      function friendlyEventTitle(eventType) {
        const labels = {
          course_run_created: "Draft created",
          course_generation_queued: "Draft queued",
          course_generation_started: "Draft build started",
          course_brief_generated: "Course plan drafted",
          course_generation_failed: "Draft build failed",
          course_revision_queued: "New version queued",
          course_revision_started: "Version cloning started",
          course_revision_completed: "New version draft ready",
          course_revision_created: "New version draft ready",
          course_revision_failed: "New version failed",
          course_materialize_queued: "Bundle queued",
          course_materialize_started: "Bundle build started",
          course_bundle_materialized: "Review bundle ready",
          course_materialize_completed: "Bundle build finished",
          course_materialize_failed: "Bundle build failed",
          course_publish_queued: "Publish queued",
          course_publish_started: "Publish started",
          course_run_published: "Course published",
          course_publish_completed: "Publish finished",
          course_publish_failed: "Publish failed",
          course_run_synced: "Draft updated",
        };
        return labels[eventType] || titleCase(eventType);
      }

      function formatEventMessage(event) {
        return event?.payload?.message
          || event?.payload?.error
          || event?.payload?.detail
          || "";
      }

      function approvedWorkflowMilestones(workflow) {
        if (!workflow) return [];
        if (workflow.status === "published") {
          return ["Assignment spec", "Module ladder", "Final assignment review"];
        }
        if (workflow.pending_gate === "gate_3_pre_publish") {
          return ["Assignment spec", "Module ladder"];
        }
        if (workflow.pending_gate === "gate_2_progression_review") {
          return ["Assignment spec"];
        }
        return [];
      }

      function buildApprovedSummary(courseRun, review) {
        const workflows = review?.linked_workflows || [];
        if (!workflows.length) {
          return "No assignment workflow has been created yet.";
        }
        if (workflows.length === 1) {
          const milestones = approvedWorkflowMilestones(workflows[0]);
          if (workflows[0].status === "published") {
            return "Assignment checks are complete and the linked workflow is published.";
          }
          if (milestones.length) {
            return `${milestones.join(", ")} approved so far.`;
          }
        }
        const readyModules = review?.counts?.ready_deliverables ?? 0;
        const totalModules = review?.counts?.total_deliverables ?? courseRun.deliverables.length;
        const publishedWorkflows = review?.counts?.published_workflow_runs ?? 0;
        return `${readyModules} of ${totalModules} deliverables are publish-ready; ${publishedWorkflows} linked assignment workflow${publishedWorkflows === 1 ? "" : "s"} published.`;
      }

      function findLastCompleted(events) {
        const meaningful = [...(events || [])].reverse().find((event) => (
          event.event_type.endsWith("_completed")
          || event.event_type === "course_brief_generated"
          || event.event_type === "course_run_published"
          || event.event_type === "course_run_created"
          || event.event_type === "course_bundle_materialized"
        ));
        return meaningful ? friendlyEventTitle(meaningful.event_type) : "Nothing completed yet";
      }

      function buildStatusModel(courseRun, review, events) {
        const workflows = review?.linked_workflows || [];
        const pendingWorkflow = workflows.find((workflow) => workflow.pending_gate);
        const latestEvent = events.length ? events[events.length - 1] : null;
        const nextAction = review?.next_actions?.[0] || "We will keep this draft moving and surface the next meaningful checkpoint here.";

        if (courseRun.active_operation) {
          return {
            owner: "Agent",
            currentTask: friendlyOperation(courseRun.active_operation),
            nextTask: nextAction,
            approvedSoFar: buildApprovedSummary(courseRun, review),
            lastCompleted: findLastCompleted(events),
            latestMessage: formatEventMessage(latestEvent) || "The agent is still working through the current step.",
          };
        }
        if (pendingWorkflow) {
          const gate = pendingWorkflow.pending_gate;
          const nextByGate = {
            gate_1_spec_review: "Once you approve this, we move into the deliverables review for the assignment.",
            gate_2_progression_review: "Once you approve this, we run the final assignment checks before publish.",
            gate_3_pre_publish: "Once you approve this, the linked assignment workflow can publish and the course can move toward publish.",
          };
          return {
            owner: "You",
            currentTask: friendlyGate(gate),
            nextTask: nextByGate[gate] || nextAction,
            approvedSoFar: buildApprovedSummary(courseRun, review),
            lastCompleted: findLastCompleted(events),
            latestMessage: "The agent has paused at a review gate and is waiting for your decision.",
          };
        }
        if (courseRun.stage === "ready_to_publish") {
          return {
            owner: "You",
            currentTask: "Decide whether to publish this version",
            nextTask: "When you publish, we create a learner snapshot and update new enrollments to point at this version.",
            approvedSoFar: buildApprovedSummary(courseRun, review),
            lastCompleted: findLastCompleted(events),
            latestMessage: "The linked assignment work is in good shape and the course is lined up for publish.",
          };
        }
        if (courseRun.status === "published") {
          return {
            owner: "Done",
            currentTask: "This version is published",
            nextTask: "Start a new version when you want to make the next learner-facing change.",
            approvedSoFar: "Learners are pinned to the published snapshot for this version.",
            lastCompleted: findLastCompleted(events),
            latestMessage: "This version is live for new enrollments.",
          };
        }
        if (courseRun.stage === "blocked" || courseRun.last_error) {
          return {
            owner: "You",
            currentTask: "Review the blocker and decide what to change",
            nextTask: nextAction,
            approvedSoFar: buildApprovedSummary(courseRun, review),
            lastCompleted: findLastCompleted(events),
            latestMessage: courseRun.last_error || "The draft hit a blocker that needs attention.",
          };
        }
        return {
          owner: "Agent",
          currentTask: "Preparing the draft for review",
          nextTask: nextAction,
          approvedSoFar: buildApprovedSummary(courseRun, review),
          lastCompleted: findLastCompleted(events),
          latestMessage: formatEventMessage(latestEvent) || "The agent is preparing the next review-ready state.",
        };
      }

      function buildUnblockModel(courseRun, review, events) {
        const workflows = review?.linked_workflows || [];
        const pendingWorkflow = workflows.find((workflow) => workflow.pending_gate);
        const latestEvent = events.length ? events[events.length - 1] : null;

        if (courseRun.active_operation) {
          return {
            reason: formatEventMessage(latestEvent) || `The agent is currently ${friendlyOperation(courseRun.active_operation).toLowerCase()}.`,
            unblock: "Nothing is needed from you right now. This draft will keep updating automatically.",
          };
        }

        if (pendingWorkflow) {
          const gate = pendingWorkflow.pending_gate;
          const whyByGate = {
            gate_1_spec_review: "The linked assignment finished authoring and reviewer checks, and it now needs your spec review before we continue.",
            gate_2_progression_review: "The assignment spec is approved, and the deliverable plan now needs your review before we continue.",
            gate_3_pre_publish: "The assignment has cleared earlier gates and needs a final review before it can publish.",
          };
          const unblockByGate = {
            gate_1_spec_review: "Read the assignment spec snapshot below. Approve if the contract, tools, deliverables, and grader checks match your intent. Otherwise request changes and describe what to fix.",
            gate_2_progression_review: "Review the proposed deliverables and review flow. Approve if the plan teaches the right engineering work. Otherwise request changes with a note.",
            gate_3_pre_publish: "Do a final pass on the assignment package. Approve if it is ready to publish, or request changes with the exact issue you want fixed.",
          };
          return {
            reason: whyByGate[gate] || "The agent is paused at a review step and needs your decision.",
            unblock: unblockByGate[gate] || "Use the review panel below to approve or request changes.",
          };
        }

        if (courseRun.stage === "ready_to_publish") {
          return {
            reason: "The course and linked assignment version have cleared the current review work.",
            unblock: "Publish this version when you are happy with the learner-facing result, or start a new version if you want more changes first.",
          };
        }

        if (courseRun.status === "published") {
          return {
            reason: "This version is already published and learner enrollments are pinned to its snapshot.",
            unblock: "Start a new version when you want to make the next learner-facing change.",
          };
        }

        if (courseRun.stage === "blocked" || courseRun.last_error) {
          return {
            reason: courseRun.last_error || "The draft hit a blocker during generation or review.",
            unblock: "This page updates automatically. Inspect the recent activity, then request changes or start a new version depending on what you want to fix.",
          };
        }

        return {
          reason: formatEventMessage(latestEvent) || "The agent is still preparing the next review-ready state.",
          unblock: "Nothing is needed from you yet. We will surface the next approval step as soon as it is ready.",
        };
      }

      function buildTimeline(courseRun, review) {
        const workflows = review?.linked_workflows || [];
        const stepStates = [
          { title: "Brief captured", detail: "Goal and outcomes are stored on the draft.", state: "done" },
          {
            title: "Course plan drafted",
            detail: courseRun.generated_plan ? "Deliverables and course shape are ready to read." : "We are still drafting the deliverable plan.",
            state: courseRun.generated_plan || courseRun.deliverables.length ? "done" : "current",
          },
          {
            title: "Assignment workflows prepared",
            detail: workflows.length ? "Linked assignment workflows exist for this draft." : "The draft has not created linked assignment workflows yet.",
            state: workflows.length ? "done" : (courseRun.active_operation ? "current" : "up-next"),
          },
          {
            title: "Reviews and approvals",
            detail: workflows.some((workflow) => workflow.pending_gate)
              ? "A linked assignment workflow is waiting on review."
              : "We are either checking the assignment work or this step is complete.",
            state: workflows.some((workflow) => workflow.pending_gate)
              ? "current"
              : ((review?.counts?.ready_deliverables ?? 0) > 0 || courseRun.stage === "ready_to_publish" || courseRun.status === "published")
                ? "done"
                : "up-next",
          },
          {
            title: "Learner version published",
            detail: courseRun.status === "published"
              ? "Learners are now pinned to the latest published snapshot."
              : "This happens after the review loop is complete and you choose to publish.",
            state: courseRun.status === "published"
              ? "done"
              : (courseRun.active_operation === "publish" || courseRun.stage === "ready_to_publish" ? "current" : "up-next"),
          },
        ];
        return stepStates;
      }

      function summarizeDraftListState(run) {
        if (run.active_operation || run.stage === "drafting" || run.status === "active") {
          return {
            label: run.active_operation ? friendlyOperation(run.active_operation) : "Building now",
            kind: "fallback",
            owner: "Agent",
            ownerCopy: "Agent is actively building this draft.",
            detailCopy: "Open the draft to follow progress and recent activity.",
          };
        }
        if (run.stage === "published" || run.status === "published") {
          return {
            label: "Published",
            kind: "live",
            owner: "Done",
            ownerCopy: "This version is already live for learners.",
            detailCopy: "Start a new version when you want to make learner-facing changes.",
          };
        }
        if (run.stage === "ready_to_publish") {
          return {
            label: "Ready to publish",
            kind: "live",
            owner: "You",
            ownerCopy: "You decide whether this version should publish now.",
            detailCopy: "Open the draft to do the final publish pass.",
          };
        }
        if (run.stage === "blocked" || run.status === "blocked") {
          return {
            label: "Blocked",
            kind: "fallback",
            owner: "You",
            ownerCopy: "You need to unblock the next step.",
            detailCopy: "Open the draft to review the blocker and decide what to fix.",
          };
        }
        if (run.status === "awaiting_human" || run.stage === "awaiting_course_review") {
          return {
            label: "Waiting on review",
            kind: "fallback",
            owner: "You",
            ownerCopy: "You have a review step waiting.",
            detailCopy: "Open the draft to approve it or request changes.",
          };
        }
        return {
          label: "Draft ready",
          kind: "active",
          owner: "Agent",
          ownerCopy: "The system is preparing the next authoring step.",
          detailCopy: "Open the draft to watch progress and recent activity.",
        };
      }

      function pickRestorableDraft(runs) {
        const rememberedDraftId = window.localStorage.getItem(activeDraftStorageKey);
        const rememberedDraft = runs.find((run) => run.id === rememberedDraftId);
        if (rememberedDraft) {
          return rememberedDraft;
        }
        return runs.find((run) => run.stage !== "published" && run.status !== "published") || runs[0] || null;
      }

      function toolCardClass(tool) {
        if (tool.safety === "write") {
          return "tool-card tool-card-write";
        }
        if (tool.safety === "irreversible" || tool.approval_required) {
          return "tool-card tool-card-irreversible";
        }
        return "tool-card tool-card-read";
      }

      function schemaFields(schema) {
        return Object.keys(schema?.properties || {});
      }

      function summarizeSchemaFields(schema) {
        const fields = schemaFields(schema);
        if (!fields.length) {
          return "No explicit fields listed";
        }
        if (fields.length <= 4) {
          return fields.join(", ");
        }
        return `${fields.slice(0, 4).join(", ")} +${fields.length - 4} more`;
      }

      function renderSpecSection(title, items) {
        return `
          <div class="review-spec-section">
            <h5>${escapeHtml(title)}</h5>
            <div class="review-list">
              ${items.join("")}
            </div>
          </div>
        `;
      }

      function summarizeToolPolicy(tools) {
        const writeCount = tools.filter((tool) => tool.safety === "write").length;
        const irreversibleCount = tools.filter((tool) => tool.safety === "irreversible" || tool.approval_required).length;
        const detail = [];
        if (writeCount) detail.push(`${writeCount} writes data`);
        if (irreversibleCount) detail.push(`${irreversibleCount} irreversible`);
        if (!detail.length) detail.push("read-only only");
        return `${pluralize(tools.length, "tool")} (${detail.join(", ")})`;
      }

      function renderSpecSummaryRow(label, value) {
        return `
          <div class="review-spec-summary-row">
            <span>${escapeHtml(label)}</span>
            <strong>${escapeHtml(value)}</strong>
          </div>
        `;
      }

      function reviewArtifactKind(detail) {
        if (detail?.artifacts?.task_agent_spec) return "task_agent_spec";
        if (detail?.artifacts?.blueprint) return "archetype_blueprint";
        return null;
      }

      function renderPlainReviewSummary(workflow, detail) {
        const spec = detail?.artifacts?.task_agent_spec;
        const blueprint = detail?.artifacts?.blueprint;
        const pendingGate = workflow.pending_gate || "gate_1_spec_review";

        if (!spec && blueprint) {
          const inputs = (blueprint.required_inputs || []).map((item) => titleCase(item)).slice(0, 3).join(", ");
          const starters = (blueprint.starter_types || []).map(friendlyStarterType).join(", ");
          return `
            <div class="review-plain-summary">
              <p class="review-plain-eyebrow">Course blueprint</p>
              <h5>${escapeHtml(blueprint.title || "Course blueprint")}</h5>
              <p>${escapeHtml(blueprint.summary || "")}</p>
              <ul>
                <li><strong>Build pattern:</strong> ${escapeHtml(friendlyBuildPattern(blueprint.archetype_id))}</li>
                <li><strong>Course shape:</strong> ${escapeHtml(friendlyCourseShape(blueprint.package_type))}</li>
                ${inputs ? `<li><strong>Required inputs:</strong> ${escapeHtml(inputs)}</li>` : ""}
                ${starters ? `<li><strong>Starter shapes:</strong> ${escapeHtml(starters)}</li>` : ""}
              </ul>
            </div>
          `;
        }
        if (!spec) {
          return `
            <div class="review-plain-summary">
              <p>Loading the review snapshot. This page will keep updating while the package is prepared.</p>
            </div>
          `;
        }

        const tools = spec.tool_registry?.tools || [];
        const writeCount = tools.filter((t) => t.safety === "write").length;
        const irreversibleCount = tools.filter((t) => t.safety === "irreversible" || t.approval_required).length;
        const behaviors = spec.behaviors || [];
        const qualities = spec.qualities || [];
        const modules = spec.modules || [];
        const eyebrowByGate = {
          gate_1_spec_review: "First review",
          gate_2_progression_review: "Module ladder check",
          gate_3_pre_publish: "Last check before publish",
        };
        const headlineByGate = {
          gate_1_spec_review: "Does this assignment shape match what you want learners to build?",
          gate_2_progression_review: "Do these deliverables cover the right engineering work?",
          gate_3_pre_publish: "Ready to ship this version?",
        };

        const moduleList = modules.length
          ? `<ul class="review-plain-modules">${modules.map((m, i) => `
              <li><strong>${i + 1}. ${escapeHtml(m.title)}</strong>${m.objective ? ` — ${escapeHtml(m.objective)}` : ""}</li>
            `).join("")}</ul>`
          : "";

        const facts = [];
        facts.push(`<li><strong>Deliverables:</strong> ${modules.length}</li>`);
        facts.push(`<li><strong>Tools the system can use:</strong> ${tools.length}${writeCount || irreversibleCount ? ` (${[writeCount && `${writeCount} write data`, irreversibleCount && `${irreversibleCount} irreversible`].filter(Boolean).join(", ")})` : ""}</li>`);
        facts.push(`<li><strong>Checks:</strong> ${behaviors.length} learner-visible · ${qualities.length} quality bars</li>`);

        return `
          <div class="review-plain-summary">
            <p class="review-plain-eyebrow">${escapeHtml(eyebrowByGate[pendingGate] || "Review")}</p>
            <h5>${escapeHtml(headlineByGate[pendingGate] || spec.title || "Review this step")}</h5>
            <p class="review-plain-spec-title"><strong>${escapeHtml(spec.title || "")}</strong>${spec.summary ? ` — ${escapeHtml(spec.summary)}` : ""}</p>
            <ul>${facts.join("")}</ul>
            ${pendingGate !== "gate_1_spec_review" ? moduleList : ""}
          </div>
        `;
      }

      function renderSpecSnapshot(workflow, detail) {
        const spec = detail?.artifacts?.task_agent_spec;
        const blueprint = detail?.artifacts?.blueprint;
        if (!spec && !blueprint) {
          return `
            <div class="review-item">
              <p>Review details are not available for this workflow yet. This page will keep updating before you decide on the step.</p>
            </div>
          `;
        }

        if (!spec && blueprint) {
          const summaryRows = [
            renderSpecSummaryRow("Build pattern", friendlyBuildPattern(blueprint.archetype_id)),
            renderSpecSummaryRow("Course shape", friendlyCourseShape(blueprint.package_type)),
            renderSpecSummaryRow("Use case", `${blueprint.recommended_domain_pack ? friendlyUseCase(blueprint.recommended_domain_pack) : "General"} • ${friendlyRisk(blueprint.risk_class)}`),
            renderSpecSummaryRow("Required inputs", pluralize((blueprint.required_inputs || []).length, "input")),
            renderSpecSummaryRow("Starter shape", pluralize((blueprint.starter_types || []).length, "starter type")),
            renderSpecSummaryRow(
              "Evaluation surface",
              [
                pluralize((blueprint.behavior_tests || []).length, "behavior check"),
                pluralize((blueprint.quality_tests || []).length, "quality bar"),
                pluralize((blueprint.judge_tests || []).length, "judge check"),
              ].join(" • "),
            ),
          ];
          const expandedSections = [
            renderSpecSection("Blueprint frame", [
              `<div class="review-item"><p><strong>Build pattern</strong></p><p>${escapeHtml(friendlyBuildPattern(blueprint.archetype_id))}</p></div>`,
              `<div class="review-item"><p><strong>Course shape</strong></p><p>${escapeHtml(friendlyCourseShape(blueprint.package_type))}</p></div>`,
              `<div class="review-item"><p><strong>Use case</strong></p><p>${escapeHtml(blueprint.recommended_domain_pack ? friendlyUseCase(blueprint.recommended_domain_pack) : "General")}</p></div>`,
              `<div class="review-item"><p><strong>Risk class</strong></p><p>${escapeHtml(friendlyRisk(blueprint.risk_class))}</p></div>`,
            ]),
            renderSpecSection("Required inputs", (blueprint.required_inputs || []).map((item) => `
              <div class="review-item">
                <p><strong>${escapeHtml(titleCase(item))}</strong></p>
              </div>
            `).concat(!(blueprint.required_inputs || []).length ? [`<div class="review-item"><p>No explicit required inputs listed.</p></div>`] : [])),
            renderSpecSection("Starter shapes", (blueprint.starter_types || []).map((starterType) => `
              <div class="review-item">
                <p><strong>${escapeHtml(friendlyStarterType(starterType))}</strong></p>
              </div>
            `).concat(!(blueprint.starter_types || []).length ? [`<div class="review-item"><p>No starter shape guidance listed.</p></div>`] : [])),
            renderSpecSection("Evaluation surface", [
              ...((blueprint.behavior_tests || []).map((item) => `
                <div class="review-item">
                  <p><strong>Behavior</strong></p>
                  <p>${escapeHtml(titleCase(item))}</p>
                </div>
              `)),
              ...((blueprint.quality_tests || []).map((item) => `
                <div class="review-item">
                  <p><strong>Quality</strong></p>
                  <p>${escapeHtml(titleCase(item))}</p>
                </div>
              `)),
              ...((blueprint.judge_tests || []).map((item) => `
                <div class="review-item">
                  <p><strong>Judge</strong></p>
                  <p>${escapeHtml(titleCase(item))}</p>
                </div>
              `)),
            ].concat(
              !((blueprint.behavior_tests || []).length + (blueprint.quality_tests || []).length + (blueprint.judge_tests || []).length)
                ? [`<div class="review-item"><p>No evaluation checks are listed yet.</p></div>`]
                : []
            )),
            renderSpecSection("Example projects and notes", [
              ...((blueprint.project_examples || []).map((item) => `
                <div class="review-item">
                  <p><strong>Example project</strong></p>
                  <p>${escapeHtml(item)}</p>
                </div>
              `)),
              ...((blueprint.notes || []).map((item) => `
                <div class="review-item">
                  <p><strong>Review note</strong></p>
                  <p>${escapeHtml(item)}</p>
                </div>
              `)),
            ].concat(!(blueprint.project_examples || []).length && !(blueprint.notes || []).length ? [`<div class="review-item"><p>No extra blueprint notes are attached yet.</p></div>`] : [])),
          ];

          return `
            <div class="review-spec-card">
              <div class="review-spec-header">
                <div>
                  <p class="review-kicker">Archetype blueprint</p>
                  <h5>${escapeHtml(blueprint.title)}</h5>
                  <p>${escapeHtml(blueprint.summary)}</p>
                </div>
              </div>
              <div class="review-spec-summary-list">
                ${summaryRows.join("")}
              </div>
              <details class="spec-details">
                <summary class="spec-details-toggle">View blueprint details</summary>
                <div class="review-spec-grid review-spec-grid-sections">
                  ${expandedSections.join("")}
                </div>
              </details>
            </div>
          `;
        }

        const endpointSummary = (spec.production_contract?.canonical_endpoints || [])
          .map((endpoint) => `${endpoint.method} ${endpoint.path}`);
        const tools = spec.tool_registry?.tools || [];
        const modules = spec.modules || [];
        const behaviors = spec.behaviors || [];
        const qualities = spec.qualities || [];
        const modes = (spec.supported_modes || []).map(titleCase);
        const pendingGate = workflow.pending_gate || "gate_1_spec_review";
        const reviewKicker = pendingGate === "gate_2_progression_review"
          ? "Module ladder snapshot"
          : pendingGate === "gate_3_pre_publish"
            ? "Final publish review"
            : "Assignment spec snapshot";
        const checksSummary = qualities.length
          ? `${pluralize(behaviors.length, "behavior check")} • ${pluralize(qualities.length, "quality bar")}`
          : `${pluralize(behaviors.length, "behavior check")} • quality bars defined`;
        const summaryRows = [
          renderSpecSummaryRow("Build pattern", friendlyBuildPattern(spec.archetype)),
          renderSpecSummaryRow("Use case", `${spec.domain_pack ? friendlyUseCase(spec.domain_pack) : "General"} • ${friendlyRisk(spec.risk_class)}`),
          renderSpecSummaryRow("Modes", modes.join(", ") || "No modes listed"),
          renderSpecSummaryRow("Tools", summarizeToolPolicy(tools)),
          renderSpecSummaryRow("Checks", checksSummary),
          renderSpecSummaryRow("Endpoints", pluralize(endpointSummary.length, "route")),
        ];
        const expandedSections = [
          renderSpecSection("Contract frame", [
            `<div class="review-item"><p><strong>Build pattern</strong></p><p>${escapeHtml(friendlyBuildPattern(spec.archetype))}</p></div>`,
            `<div class="review-item"><p><strong>Course shape</strong></p><p>${escapeHtml(friendlyCourseShape(spec.package_type))}</p></div>`,
            `<div class="review-item"><p><strong>Use case</strong></p><p>${escapeHtml(spec.domain_pack ? friendlyUseCase(spec.domain_pack) : "General")}</p></div>`,
            `<div class="review-item"><p><strong>Risk class</strong></p><p>${escapeHtml(friendlyRisk(spec.risk_class))}</p></div>`,
          ]),
          renderSpecSection("Contract schemas", [
            `<div class="review-item"><p><strong>Input schema</strong></p><p>${escapeHtml(summarizeSchemaFields(spec.task_schema))}</p></div>`,
            `<div class="review-item"><p><strong>Output schema</strong></p><p>${escapeHtml(summarizeSchemaFields(spec.output_schema))}</p></div>`,
            `<div class="review-item"><p><strong>Run state schema</strong></p><p>${escapeHtml(summarizeSchemaFields(spec.run_state_schema))}</p></div>`,
            `<div class="review-item"><p><strong>Trace schema</strong></p><p>${escapeHtml(summarizeSchemaFields(spec.trace_schema))}</p></div>`,
          ]),
          renderSpecSection("Tools and approval policy", tools.map((tool) => `
            <div class="review-item ${toolCardClass(tool)}">
              <p><strong>${escapeHtml(tool.id)}</strong></p>
              <p>${escapeHtml(tool.description)}</p>
              <div class="pill-row">
                ${pill(friendlyToolSafety(tool.safety))}
                ${tool.approval_required ? pill("Requires approval") : ""}
                ${tool.dry_run_supported ? pill("Dry run supported") : ""}
              </div>
            </div>
          `).concat(!tools.length ? [`<div class="review-item"><p>No tools listed.</p></div>`] : [])),
          renderSpecSection("Checks and quality bars", [
            ...behaviors.map((behavior) => `
              <div class="review-item">
                <p><strong>${escapeHtml(behavior.description)}</strong></p>
              </div>
            `),
            ...qualities.map((quality) => `
              <div class="review-item">
                <p><strong>${escapeHtml(quality.description)}</strong></p>
              </div>
            `),
          ]),
          renderSpecSection("Modes and endpoints", [
            `<div class="review-item"><p><strong>Supported modes</strong></p><p>${escapeHtml(modes.join(", ") || "No modes listed")}</p></div>`,
            `<div class="review-item"><p><strong>Endpoints</strong></p><p>${escapeHtml(endpointSummary.join(", ") || "No endpoints listed")}</p></div>`,
          ]),
        ];

        if (pendingGate !== "gate_1_spec_review") {
          expandedSections.push(renderSpecSection("Deliverables", modules.map((module, index) => `
            <div class="review-item">
              <p><strong>${escapeHtml(`${index + 1}. ${module.title}`)}</strong></p>
              <p>${escapeHtml(module.objective)}</p>
              <div class="pill-row">
                ${pill(friendlyStarterType(module.starter_type))}
                ${(module.overlay_ids || []).map((overlay) => pill(friendlyFocusArea(overlay))).join("")}
              </div>
            </div>
          `).concat(!modules.length ? [`<div class="review-item"><p>No deliverable plan is attached yet.</p></div>`] : [])));
          expandedSections.push(renderSpecSection("Progression gates", [
            ...behaviors
              .filter((behavior) => behavior.first_required_in)
              .map((behavior) => `
                <div class="review-item">
                  <p><strong>${escapeHtml(behavior.description)}</strong></p>
                  <p>${escapeHtml(`Required from ${behavior.first_required_in}`)}</p>
                </div>
              `),
            ...qualities
              .filter((quality) => quality.first_required_in)
              .map((quality) => `
                <div class="review-item">
                  <p><strong>${escapeHtml(quality.description)}</strong></p>
                  <p>${escapeHtml(`Required from ${quality.first_required_in}`)}</p>
                </div>
              `),
          ]));
        }

        return `
          <div class="review-spec-card">
            <div class="review-spec-header">
              <div>
                <p class="review-kicker">${escapeHtml(reviewKicker)}</p>
                <h5>${escapeHtml(spec.title)}</h5>
                <p>${escapeHtml(spec.summary)}</p>
              </div>
            </div>
            <div class="review-spec-summary-list">
              ${summaryRows.join("")}
            </div>
            <details class="spec-details">
              <summary class="spec-details-toggle">View full spec</summary>
              <div class="review-spec-grid review-spec-grid-sections">
                ${expandedSections.join("")}
              </div>
            </details>
          </div>
        `;
      }

      function renderGenerationStatus(status) {
        const isLive = Boolean(status && status.available && status.source === "openai_live");
        generationBanner.className = `generation-banner ${isLive ? "live" : "fallback"}`;
        if (isLive) {
          generationHeading.textContent = `AI planning live${status?.model_id ? ` · ${status.model_id}` : ""}`;
          generationBadge.textContent = "Live";
        } else if (status?.api_key_present === false) {
          generationHeading.textContent = "This app instance is using the fallback planner";
          generationBadge.textContent = "No key in server";
        } else if (status?.sdk_installed === false) {
          generationHeading.textContent = "This app instance cannot use OpenAI planning";
          generationBadge.textContent = "SDK missing";
        } else {
          generationHeading.textContent = "This app instance is using the fallback planner";
          generationBadge.textContent = "Fallback";
        }
        generationBadge.className = `badge ${isLive ? "live" : "fallback"}`;
        generationMessage.textContent = status?.message || "";

        generationMeta.innerHTML = [
          status?.provider ? pill(`Provider: ${status.provider}`) : "",
          status?.model_id ? pill(`Model: ${status.model_id}`) : "",
          status?.sdk_installed ? pill("SDK installed") : pill("SDK missing"),
          status?.api_key_present ? pill("API key detected") : pill("No API key in this server"),
          status?.env_file ? pill(`Env file: ${status.env_file}`) : pill("No env file configured"),
        ].filter(Boolean).join("");
        generationSetupToggle.classList.toggle("hidden", isLive);
        if (isLive) {
          generationSetup.classList.add("hidden");
          generationSetupToggle.textContent = "Show setup steps";
        }
      }

      function resetDraftSelection() {
        currentCourseRun = null;
        currentReview = null;
        currentEvents = [];
        currentWorkflowDetails = {};
        draftLoadInProgress = false;
        pendingDraftId = null;
        workflowRejectMode.clear();
        workflowCommentCache.clear();
        stopDraftPolling();
        results.classList.remove("visible");
        draftStageBadge.textContent = "Waiting";
        draftStageBadge.className = "badge fallback";
        draftStatusSummary.innerHTML = "";
        draftUnblockSummary.innerHTML = "";
        planSummary.innerHTML = "";
        moduleList.innerHTML = "";
        courseSummary.innerHTML = "";
        reviewMetrics.innerHTML = "";
        progressTimeline.innerHTML = "";
        reviewBlockers.innerHTML = "";
        reviewActions.innerHTML = "";
        publishedVersions.innerHTML = "";
        draftActivity.innerHTML = "";
        linkedWorkflows.innerHTML = "";
        sourceBadge.textContent = "Waiting";
        sourceBadge.className = "badge fallback";
        materializeButton.disabled = true;
        publishButton.disabled = true;
        createRevisionButton.disabled = true;
        renderWorkflowProgress();
        renderDraftSwitcherMenu(recentDraftRuns);
        updateWorkspaceChrome();
      }

      function updateBriefCounters() {
        const goalLength = goalField.value.trim().length;
        goalCount.textContent = `${goalLength} character${goalLength === 1 ? "" : "s"}`;
        outcomesCount.textContent = `${creatorState.outcomes.length} outcome${creatorState.outcomes.length === 1 ? "" : "s"}`;
      }

      function showCreatorStep(n) {
        creatorState.step = n;
        creatorPanes.forEach((pane) => {
          const idx = parseInt(pane.id.split("-").pop(), 10);
          const active = idx === n;
          pane.classList.toggle("active", active);
          if (active) {
            pane.removeAttribute("hidden");
          } else {
            pane.setAttribute("hidden", "");
          }
        });
        if (creatorStepper) {
          creatorStepper.querySelectorAll("[data-creator-step]").forEach((li) => {
            const idx = parseInt(li.dataset.creatorStep, 10);
            li.classList.remove("done", "current", "up-next");
            if (idx < n) li.classList.add("done");
            else if (idx === n) li.classList.add("current");
            else li.classList.add("up-next");
          });
        }
        const titles = {
          1: "Describe the system",
          2: "Refine outcomes",
          3: "Setup choices",
          4: "Review the proposed plan",
        };
        if (creatorStepTitle) creatorStepTitle.textContent = titles[n] || "";
      }

      function syncOutcomesFromInputs() {
        const inputs = creatorOutcomesList?.querySelectorAll("input[data-outcome-index]") || [];
        creatorState.outcomes = Array.from(inputs).map((el) => el.value.trim()).filter(Boolean);
      }

      function renderCreatorOutcomes() {
        if (!creatorOutcomesList) return;
        if (!creatorState.outcomes.length) {
          creatorOutcomesList.innerHTML = `
            <div class="creator-outcome-empty">
              <p class="field-hint">No outcomes yet. Add one or use “Suggest outcomes again”.</p>
            </div>
          `;
        } else {
          creatorOutcomesList.innerHTML = creatorState.outcomes.map((outcome, i) => `
            <div class="creator-outcome-row">
              <span class="creator-outcome-index">${i + 1}</span>
              <input type="text" data-outcome-index="${i}" value="${escapeHtml(outcome)}" placeholder="What will the learner be able to do?" />
              <button type="button" class="creator-outcome-remove" data-remove-outcome="${i}" aria-label="Remove outcome">×</button>
            </div>
          `).join("");
        }
        updateBriefCounters();
      }

      function readCreatorChoices() {
        const starter = document.querySelector('input[name="starter_type"]:checked');
        const dbSelect = document.getElementById("creator-database");
        const cacheSelect = document.getElementById("creator-cache");
        return {
          starter_type: starter?.value || "partial_implementation",
          primary_database: dbSelect?.value || null,
          cache_backend: cacheSelect?.value || null,
          tech_stack: [],
          data_sources: Array.isArray(creatorState.choices.data_sources)
            ? creatorState.choices.data_sources.map((source) => ({ ...source }))
            : [],
        };
      }

      function applyCreatorChoicesToInputs(choices) {
        if (!choices) return;
        creatorState.choices = {
          starter_type: choices.starter_type || "partial_implementation",
          primary_database: choices.primary_database || null,
          cache_backend: choices.cache_backend || null,
          tech_stack: Array.isArray(choices.tech_stack) ? [...choices.tech_stack] : [],
          data_sources: Array.isArray(choices.data_sources) ? choices.data_sources.map((source) => ({ ...source })) : [],
        };
        const starter = document.querySelector(`input[name="starter_type"][value="${choices.starter_type}"]`);
        if (starter) starter.checked = true;
        const dbSelect = document.getElementById("creator-database");
        if (dbSelect && choices.primary_database !== undefined) {
          dbSelect.value = choices.primary_database || "";
        }
        const cacheSelect = document.getElementById("creator-cache");
        if (cacheSelect && choices.cache_backend !== undefined) {
          cacheSelect.value = choices.cache_backend || "";
        }
        renderCreatorDataSources();
      }

      function friendlyDatabase(value) {
        const labels = { postgres: "PostgreSQL", mysql: "MySQL", sqlite: "SQLite", mongodb: "MongoDB" };
        if (!value) return "No database";
        return labels[value] || titleCase(value);
      }

      function friendlyCache(value) {
        const labels = { redis: "Redis", memcached: "Memcached" };
        if (!value) return "No cache";
        return labels[value] || titleCase(value);
      }

      function friendlyDataSourcePurpose(value) {
        const labels = {
          retrieval: "Retrieval",
          reference_data: "Reference data",
          seed_state: "Seed state",
          external_mock: "Mocked dependency data",
        };
        return labels[value] || titleCase(String(value || "reference_data").replaceAll("_", " "));
      }

      function formatBytes(bytes) {
        const value = Number(bytes || 0);
        if (!Number.isFinite(value) || value <= 0) return "0 B";
        if (value < 1024) return `${value} B`;
        if (value < 1024 * 1024) return `${(value / 1024).toFixed(value < 10 * 1024 ? 1 : 0)} KB`;
        return `${(value / (1024 * 1024)).toFixed(1)} MB`;
      }

      function selectedCreatorAssetIds() {
        return new Set(
          (creatorState.choices.data_sources || [])
            .map((source) => source.asset_id || source.id)
            .filter(Boolean),
        );
      }

      function renderCreatorDataSources() {
        const selectedSources = creatorState.choices.data_sources || [];
        const selectedIds = selectedCreatorAssetIds();
        if (creatorDataSourceCount) {
          creatorDataSourceCount.textContent = `${selectedSources.length} attached`;
        }
        if (creatorSelectedDataSources) {
          if (!selectedSources.length) {
            creatorSelectedDataSources.innerHTML = `
              <div class="creator-data-source-empty">
                <p class="field-hint">No uploaded data sources selected yet.</p>
              </div>
            `;
          } else {
            creatorSelectedDataSources.innerHTML = selectedSources.map((source) => `
              <article class="creator-data-source-card">
                <div class="creator-data-source-copy">
                  <strong>${escapeHtml(source.title || source.id)}</strong>
                  <p>${escapeHtml(source.workspace_path || "data/source.txt")}</p>
                  <div class="pill-row">
                    ${pill(friendlyDataSourcePurpose(source.purpose))}
                    ${source.format ? pill(source.format.toUpperCase()) : ""}
                    ${source.learner_visible ? pill("Learner visible") : ""}
                  </div>
                </div>
                <button type="button" class="button subtle" data-remove-data-source="${escapeHtml(source.asset_id || source.id)}">Remove</button>
              </article>
            `).join("");
          }
        }
        if (creatorAssetLibrary) {
          if (!creatorState.assets.length) {
            creatorAssetLibrary.innerHTML = "";
          } else {
            const items = creatorState.assets.map((asset) => {
              const assetId = asset.id;
              const selected = selectedIds.has(assetId);
              return `
                <article class="creator-asset-row">
                  <div class="creator-asset-copy">
                    <strong>${escapeHtml(asset.title || asset.file_name)}</strong>
                    <p>${escapeHtml(asset.file_name)} • ${escapeHtml(asset.workspace_path)} • ${escapeHtml(formatBytes(asset.size_bytes))}</p>
                  </div>
                  <div class="creator-asset-actions">
                    <button type="button" class="button subtle" data-toggle-data-source="${escapeHtml(assetId)}">${selected ? "Remove from course" : "Use in this course"}</button>
                  </div>
                </article>
              `;
            }).join("");
            creatorAssetLibrary.innerHTML = `
              <div class="creator-asset-library-header">
                <h4>Uploaded files</h4>
                <p class="field-hint">These files are stored locally and can be reused in later drafts.</p>
              </div>
              <div class="creator-asset-library-list">${items}</div>
            `;
          }
        }
      }

      function renderCreatorPlanPreview() {
        const plan = creatorState.plan;
        if (!creatorPlanPreview) return;
        if (!plan) {
          creatorPlanPreview.innerHTML = `<p class="field-hint">Generate a plan from step 3 first.</p>`;
          return;
        }
        const choices = plan.creator_choices || creatorState.choices;
        const choicePills = [
          pill(`Starter: ${friendlyStarterType(choices.starter_type)}`),
          pill(`Database: ${friendlyDatabase(choices.primary_database)}`),
          pill(`Cache: ${friendlyCache(choices.cache_backend)}`),
          ...(choices.data_sources && choices.data_sources.length ? [pill(`Data sources: ${choices.data_sources.length}`)] : []),
        ].join("");
        const dataSourceList = choices.data_sources && choices.data_sources.length
          ? `
            <div class="creator-plan-data-sources">
              <h4>Attached data sources</h4>
              <ul>
                ${choices.data_sources.map((source) => `<li><strong>${escapeHtml(source.title || source.id)}</strong> — <code>${escapeHtml(source.workspace_path || "data/source.txt")}</code></li>`).join("")}
              </ul>
            </div>
          `
          : "";
        const deliverableCards = (plan.deliverables || []).map((m, i) => `
          <article class="creator-plan-module">
            <header>
              <span class="creator-plan-module-index">${i + 1}</span>
              <h4>${escapeHtml(m.title)}</h4>
            </header>
            <p>${escapeHtml(m.summary || "")}</p>
            ${m.learning_outcomes && m.learning_outcomes.length ? `
              <p class="creator-plan-module-section-title">What this deliverable covers</p>
              <ul>${m.learning_outcomes.map((o) => `<li>${escapeHtml(o)}</li>`).join("")}</ul>
            ` : ""}
            ${m.creator_notes && m.creator_notes.length ? `
              <p class="creator-plan-module-section-title">Notes for you</p>
              <ul class="creator-plan-notes">${m.creator_notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul>
            ` : ""}
          </article>
        `).join("");
        creatorPlanPreview.innerHTML = `
          <div class="creator-plan-summary">
            <h3>${escapeHtml(plan.title)}</h3>
            ${plan.summary ? `<p class="creator-plan-summary-line">${escapeHtml(plan.summary)}</p>` : ""}
            ${plan.creator_summary ? `<p class="creator-plan-narrative">${escapeHtml(plan.creator_summary)}</p>` : ""}
            <div class="pill-row">${choicePills}</div>
            ${dataSourceList}
          </div>
          <div class="creator-plan-modules">${deliverableCards}</div>
          ${plan.notes && plan.notes.length ? `
            <aside class="creator-plan-footnotes">
              <h4>Notes about this plan</h4>
              <ul>${plan.notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul>
            </aside>
          ` : ""}
        `;
      }

      async function refreshCreatorAssets() {
        if (!state.creator_assets_url) return;
        try {
          const response = await fetch(state.creator_assets_url);
          if (!response.ok) {
            throw new Error(await extractDetail(response));
          }
          const payload = await response.json();
          creatorState.assets = Array.isArray(payload.assets) ? payload.assets : [];
          renderCreatorDataSources();
        } catch (error) {
          if (creatorAssetLibrary) {
            creatorAssetLibrary.innerHTML = `<p class="field-hint">${escapeHtml(error instanceof Error ? error.message : "Could not load uploaded files.")}</p>`;
          }
        }
      }

      function attachCreatorAsset(record) {
        const next = Array.isArray(creatorState.choices.data_sources)
          ? creatorState.choices.data_sources.filter((source) => (source.asset_id || source.id) !== record.id)
          : [];
        next.push({ ...record.data_source });
        creatorState.choices = { ...creatorState.choices, data_sources: next };
        renderCreatorDataSources();
      }

      function detachCreatorAsset(assetId) {
        creatorState.choices = {
          ...creatorState.choices,
          data_sources: (creatorState.choices.data_sources || []).filter((source) => (source.asset_id || source.id) !== assetId),
        };
        renderCreatorDataSources();
      }

      function defaultDataSourcePurpose() {
        const goal = (goalField?.value || creatorState.goal || "").toLowerCase();
        if (/(rag|retrieval|knowledge base|documents|corpus|search|citation)/.test(goal)) {
          return "retrieval";
        }
        return "reference_data";
      }

      async function uploadCreatorAssets() {
        if (!creatorDataSourceFileInput?.files?.length) {
          setMessage(formMessage, "error", "Choose at least one file to upload.");
          return;
        }
        const purpose = creatorDataSourcePurpose?.value || defaultDataSourcePurpose();
        const files = Array.from(creatorDataSourceFileInput.files);
        creatorUploadDataSourceButton.disabled = true;
        setMessage(formMessage, "info", `Uploading ${pluralize(files.length, "file")}…`);
        try {
          const uploaded = [];
          for (const file of files) {
            const content = await file.text();
            const response = await fetch(state.creator_assets_url, {
              method: "POST",
              headers: { "content-type": "application/json" },
              body: JSON.stringify({
                file_name: file.name,
                content,
                content_type: file.type || null,
                purpose,
                learner_visible: true,
              }),
            });
            if (!response.ok) {
              throw new Error(await extractDetail(response));
            }
            uploaded.push(await response.json());
          }
          for (const record of uploaded.reverse()) {
            creatorState.assets = [record, ...creatorState.assets.filter((asset) => asset.id !== record.id)];
            attachCreatorAsset(record);
          }
          if (creatorDataSourceFileInput) {
            creatorDataSourceFileInput.value = "";
          }
          setMessage(
            formMessage,
            "success",
            `Uploaded ${pluralize(uploaded.length, "file")} and added ${pluralize(uploaded.length, "data source")} to this course setup.`,
          );
        } catch (error) {
          setMessage(formMessage, "error", error instanceof Error ? error.message : "Could not upload the selected file.");
        } finally {
          creatorUploadDataSourceButton.disabled = false;
        }
      }

      async function fetchCreatorSuggestedOutcomes(opts = {}) {
        const goal = goalField.value.trim();
        if (goal.length < 10) {
          setMessage(formMessage, "error", "Add a more specific problem statement first.");
          return false;
        }
        creatorState.goal = goal;
        if (!opts.silentMessage) {
          setMessage(formMessage, "info", "Suggesting outcomes from your problem statement...");
        }
        try {
          const response = await fetch(state.suggest_outcomes_url, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ goal }),
          });
          if (!response.ok) {
            throw new Error(await extractDetail(response));
          }
          const payload = await response.json();
          const fresh = (payload.learning_outcomes || []).filter(Boolean);
          creatorState.outcomes = fresh.length
            ? fresh
            : [
                "Define a working contract for the system.",
                "Implement the core feature with sensible defaults.",
                "Add tests that match the contract.",
              ];
          if (!opts.silentMessage) {
            setMessage(
              formMessage,
              payload.source === "openai_live" ? "success" : "info",
              payload.source === "openai_live"
                ? "Suggested outcomes added. Edit them however you want."
                : "Suggested outcomes added from the fallback planner. Edit them however you want.",
            );
          }
          return true;
        } catch (error) {
          setMessage(formMessage, "error", error instanceof Error ? error.message : "Could not suggest outcomes.");
          return false;
        }
      }

      async function fetchCreatorPlan() {
        creatorState.choices = readCreatorChoices();
          setMessage(formMessage, "info", "Building a deliverable plan from your brief...");
        try {
          const response = await fetch("/v1/course-generation/creator-plan", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              goal: creatorState.goal,
              learning_outcomes: creatorState.outcomes,
              creator_choices: creatorState.choices,
            }),
          });
          if (!response.ok) {
            throw new Error(await extractDetail(response));
          }
          const payload = await response.json();
          creatorState.plan = payload.plan;
          if (payload.plan?.creator_choices) {
            applyCreatorChoicesToInputs(payload.plan.creator_choices);
          }
          if (payload.learning_outcomes && payload.learning_outcomes.length) {
            creatorState.outcomes = payload.learning_outcomes.filter(Boolean);
          }
          if (payload.status) {
            renderGenerationStatus(payload.status);
          }
          setMessage(
            formMessage,
            payload.source === "openai_live" ? "success" : "info",
            payload.source === "openai_live"
              ? "Deliverable plan ready. Review and create the draft when you're aligned."
              : "Deliverable plan ready (fallback planner). Review and create the draft when you're aligned.",
          );
          return true;
        } catch (error) {
          setMessage(formMessage, "error", error instanceof Error ? error.message : "Could not generate the deliverable plan.");
          return false;
        }
      }

      async function createDraftFromCreatorPlan() {
        if (!creatorState.plan) {
          setMessage(formMessage, "error", "Generate a plan first.");
          return;
        }
        generateButton.disabled = true;
        const originalLabel = generateButton.textContent;
        let createdCourseRun = null;
        generateButton.textContent = "Creating draft…";
        if (creatorPlanPreview) {
          creatorPlanPreview.classList.add("creator-plan-preview-creating");
        }
        setMessage(formMessage, "info", "Creating your draft from this plan…");
        try {
          const response = await fetch("/v1/course-runs/from-creator-plan-async", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ plan: creatorState.plan }),
          });
          if (!response.ok) {
            throw new Error(await extractDetail(response));
          }
          const payload = await response.json();
          createdCourseRun = payload.course_run;
          generateButton.textContent = "Opening draft…";
          await loadCourseDraft(createdCourseRun.id, {
            silentMessage: true,
            historyMode: "push",
            tabAfterLoad: "drafts",
            scrollToResult: true,
          });
          setMessage(
            formMessage,
            "success",
            `Draft created: “${createdCourseRun.title}”. Opened on the Drafts tab.`,
          );
          resetCreatorFlow();
        } catch (error) {
          const message = createdCourseRun
            ? "Your draft was created, but we couldn't open it yet. Your plan is still here so you can try again."
            : (error instanceof Error ? error.message : "Could not create the draft.");
          setMessage(formMessage, "error", message);
          if (creatorPlanPreview) {
            creatorPlanPreview.classList.remove("creator-plan-preview-creating");
          }
        } finally {
          generateButton.disabled = false;
          generateButton.textContent = originalLabel;
        }
      }

      function resetCreatorFlow() {
        creatorState.goal = "";
        creatorState.outcomes = [];
        creatorState.choices = {
          starter_type: "partial_implementation",
          primary_database: "postgres",
          cache_backend: "redis",
          tech_stack: [],
          data_sources: [],
        };
        creatorState.plan = null;
        if (goalField) goalField.value = "";
        if (creatorDataSourceFileInput) creatorDataSourceFileInput.value = "";
        if (creatorDataSourcePurpose) creatorDataSourcePurpose.value = defaultDataSourcePurpose();
        renderCreatorOutcomes();
        renderCreatorDataSources();
        if (creatorPlanPreview) {
          creatorPlanPreview.classList.remove("creator-plan-preview-creating");
          creatorPlanPreview.innerHTML = "";
        }
        showCreatorStep(1);
      }

      function renderWorkflowProgress(courseRun = currentCourseRun, review = currentReview, events = currentEvents) {
        const showingDraftContext = currentTab === "drafts" && Boolean(courseRun);
        const loadingDraftContext = currentTab === "drafts" && !courseRun && (draftLoadInProgress || Boolean(pendingDraftId));
        const readyModules = review?.counts?.ready_deliverables ?? 0;
        const totalModules = review?.counts?.total_deliverables ?? courseRun?.deliverables?.length ?? 0;
        const progressPercent = totalModules ? Math.min(100, Math.round((readyModules / totalModules) * 100)) : 0;
        const progressState = {
          brief: "current",
          plan: "up-next",
          review: "up-next",
          publish: "up-next",
        };

        if (!showingDraftContext) {
          if (loadingDraftContext) {
            workflowProgressTitle.textContent = "Loading selected draft";
            workflowProgressCopy.textContent = "Fetching the latest review state, workflow summaries, and activity for this draft.";
            workflowProgressNext.textContent = "Next: Review the loaded draft";
          } else if (currentTab === "drafts") {
            workflowProgressTitle.textContent = "Open a draft";
            workflowProgressCopy.textContent = "Pick a draft from the left rail to review progress, approvals, and recent activity.";
            workflowProgressNext.textContent = "Next: Select a draft or start a new brief";
          } else {
            workflowProgressTitle.textContent = "Start a new brief";
            workflowProgressCopy.textContent = "Write the goal and outcomes to create the first draft.";
            workflowProgressNext.textContent = "Next: Generate the course plan";
          }
          workflowProgressCount.textContent = "0 of 0 deliverables ready";
          workflowReviewProgress.classList.add("hidden");
          workflowReviewProgress.dataset.tone = "active";
          workflowProgressBar.style.width = "0%";
        } else {
          const statusModel = buildStatusModel(courseRun, review || {}, events || []);
          const nextTask = humanizeAuthorCopy(statusModel.nextTask);
          workflowProgressTitle.textContent = statusModel.currentTask;
          workflowProgressCopy.textContent = humanizeAuthorCopy(statusModel.latestMessage);
          workflowProgressCount.textContent = `${readyModules} of ${totalModules} deliverables ready`;
          workflowReviewProgress.classList.remove("hidden");
          workflowReviewProgress.dataset.tone = courseRun.status === "published"
            ? "live"
            : (courseRun.stage === "blocked" || courseRun.last_error)
              ? "blocked"
              : (courseRun.status === "awaiting_human" || courseRun.stage === "awaiting_course_review")
                ? "fallback"
                : "active";
          workflowProgressBar.style.width = `${progressPercent}%`;
          workflowProgressNext.textContent = `Next: ${nextTask}`;

          if (courseRun.status === "published") {
            progressState.brief = "done";
            progressState.plan = "done";
            progressState.review = "done";
            progressState.publish = "done";
          } else if (courseRun.stage === "ready_to_publish" || courseRun.active_operation === "publish") {
            progressState.brief = "done";
            progressState.plan = "done";
            progressState.review = "done";
            progressState.publish = "current";
          } else if (
            review?.linked_workflows?.length
            || courseRun.stage === "awaiting_course_review"
            || courseRun.status === "awaiting_human"
            || courseRun.stage === "blocked"
          ) {
            progressState.brief = "done";
            progressState.plan = "done";
            progressState.review = "current";
          } else if (
            courseRun.generated_plan
            || courseRun.deliverables.length
            || courseRun.active_operation === "generation"
            || courseRun.stage === "drafting"
            || courseRun.status === "active"
          ) {
            progressState.brief = "done";
            progressState.plan = "current";
          }
        }

        workflowSteps.forEach((step) => {
          const stepKey = step.dataset.workflowStep;
          const state = progressState[stepKey] || "up-next";
          step.classList.remove("done", "current", "up-next");
          step.classList.add(state);
        });
      }

      function setActiveTab(tabName, options = {}) {
        const showCreate = tabName === "create";
        currentTab = showCreate ? "create" : "drafts";
        window.localStorage.setItem(activeTabStorageKey, tabName);
        createTabButton.classList.toggle("active", showCreate);
        draftsTabButton.classList.toggle("active", !showCreate);
        createTabButton.setAttribute("aria-selected", showCreate ? "true" : "false");
        draftsTabButton.setAttribute("aria-selected", showCreate ? "false" : "true");
        createTabPane.classList.toggle("active", showCreate);
        draftsTabPane.classList.toggle("active", !showCreate);
        renderWorkflowProgress();
        updateWorkspaceChrome();
        if (currentCourseRun && currentTab === "drafts") {
          ensureDraftPolling(currentCourseRun.id);
        } else {
          stopDraftPolling();
        }
        if (options.updateUrl !== false) {
          writeUrlState({}, options.historyMode || "replace");
        }
      }

      function scrollDraftIntoView() {
        const focusTarget = document.getElementById("review-step-panel")
          || document.getElementById("draft-context-bar")
          || results;
        if (focusTarget instanceof HTMLElement) {
          focusTarget.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }

      function revealPanelFromHash(hash) {
        if (!hash || !hash.startsWith("#")) return;
        const target = document.querySelector(hash);
        if (!(target instanceof HTMLElement)) return;
        if (target instanceof HTMLDetailsElement) {
          target.open = true;
        }
      }

      function renderPersistedPlan(courseRun) {
        const plan = courseRun?.generated_plan;
        const status = courseRun?.generation_status;
        const source = courseRun?.generation_source || "deterministic_fallback";
        if (!plan) {
          sourceBadge.textContent = courseRun?.stage === "drafting" ? "Planning..." : "Waiting";
          sourceBadge.className = "badge fallback";
          planSummary.innerHTML = `
            <div class="summary-item">
              <h4>${escapeHtml(courseRun?.title || "Draft in progress")}</h4>
              <p>${escapeHtml(courseRun?.summary || "We are generating the course plan and linked workflows.")}</p>
            </div>
          `;
          moduleList.innerHTML = `<div class="review-item"><p>${escapeHtml(courseRun?.stage === "drafting" ? "We will fill in the deliverable plan here as soon as the draft is ready." : "No generated deliverable plan is stored on this draft yet.")}</p></div>`;
          return;
        }
        renderPlan(plan, source, status);
      }

      function renderPlan(plan, source, status) {
        sourceBadge.textContent = source === "openai_live"
          ? `Planned with ${status?.model_id || "OpenAI"}`
          : "Fallback plan";
        sourceBadge.className = `badge ${source === "openai_live" ? "live" : "fallback"}`;

        const focusAreas = [...new Set((plan.deliverables || []).flatMap((deliverable) => deliverable.overlays_hint || []))]
          .map((overlay) => pill(`Focus: ${friendlyFocusArea(overlay)}`))
          .join("");
        const noteItems = (plan.notes || []).map((note) => `<div class="review-item"><p>${escapeHtml(note)}</p></div>`).join("");
        planSummary.innerHTML = `
          <div class="summary-item">
            <h4>Deliverable plan</h4>
            <p>${escapeHtml(plan.summary)}</p>
            <div class="pill-row">
              ${pill(`Course shape: ${friendlyCourseShape(plan.package_type)}`)}
              ${plan.base_archetype_hint ? pill(`Build pattern: ${friendlyBuildPattern(plan.base_archetype_hint)}`) : ""}
              ${focusAreas}
            </div>
          </div>
          ${noteItems ? `<div class="summary-item"><h4>Why this plan</h4><div class="review-list">${noteItems}</div></div>` : ""}
        `;

        moduleList.innerHTML = (plan.deliverables || []).map((deliverable, index) => `
          <div class="module-item">
            <h4>${index + 1}. ${escapeHtml(deliverable.title)}</h4>
            <p>${escapeHtml(deliverable.summary || "")}</p>
            <div class="pill-row">
              ${deliverable.archetype_hint ? pill(`Build pattern: ${friendlyBuildPattern(deliverable.archetype_hint)}`) : ""}
              ${deliverable.domain_pack_hint ? pill(`Use case: ${friendlyUseCase(deliverable.domain_pack_hint)}`) : ""}
              ${(deliverable.overlays_hint || []).map((overlay) => pill(`Focus: ${friendlyFocusArea(overlay)}`)).join("")}
            </div>
          </div>
        `).join("");
      }

      function renderCourse(courseRun) {
        currentCourseRun = courseRun;
        window.localStorage.setItem(activeDraftStorageKey, courseRun.id);
        updateWorkspaceChrome();
        const overviewPills = [
          pill(`Last updated: ${formatDate(courseRun.updated_at)}`),
          pill(`${courseRun.deliverables.length} deliverable${courseRun.deliverables.length === 1 ? "" : "s"}`),
          pill(usageSpendLabel(courseRun.ai_usage)),
          courseRun.base_archetype ? pill(`Build pattern: ${friendlyBuildPattern(courseRun.base_archetype)}`) : "",
          courseRun.active_operation ? pill(`Working on: ${friendlyOperation(courseRun.active_operation)}`) : "",
        ].filter(Boolean).join("");
        const requestedOutcomes = (courseRun.requested_learning_outcomes || []).map((item) => pill(item)).join("");
        courseSummary.innerHTML = `
          <div class="summary-item">
            <h4>${escapeHtml(courseRun.title)}</h4>
            <p>${escapeHtml(courseRun.goal || courseRun.summary)}</p>
            <div class="pill-row">
              ${overviewPills}
            </div>
          </div>
          ${requestedOutcomes ? `
            <div class="summary-item">
              <h4>Learning outcomes</h4>
              <div class="pill-row">${requestedOutcomes}</div>
            </div>
          ` : ""}
        `;
        const isBusy = Boolean(courseRun.active_operation);
        materializeButton.disabled = courseRun.stage === "drafting" || isBusy;
        publishButton.disabled = courseRun.stage !== "ready_to_publish" || isBusy;
        createRevisionButton.disabled = courseRun.status !== "published" || isBusy;

        const isPublished = courseRun.status === "published";
        const isReadyToPublish = courseRun.stage === "ready_to_publish";
        publishButton.classList.toggle("primary", isReadyToPublish && !isBusy);
        publishButton.classList.toggle("subtle", !(isReadyToPublish && !isBusy));
        createRevisionButton.classList.toggle("primary", isPublished && !isBusy);
        createRevisionButton.classList.toggle("subtle", !(isPublished && !isBusy));
        materializeButton.classList.toggle("subtle", true);
        materializeButton.classList.toggle("primary", false);

        renderDraftSwitcherMenu(recentDraftRuns);
      }

      function renderDraftStatus(courseRun, review, events) {
        const workflows = review?.linked_workflows || [];
        const pendingGates = workflows.filter((workflow) => workflow.pending_gate);
        const latestEvent = events.length ? events[events.length - 1] : null;
        const activeOperation = courseRun.active_operation || null;
        const statusModel = buildStatusModel(courseRun, review, events);
        const unblockModel = buildUnblockModel(courseRun, review, events);

        let badgeText = "Draft ready";
        let badgeKind = "live";
        let headline = "The draft is ready for review.";
        let explanation = statusModel.latestMessage;

        if (activeOperation === "generation") {
          badgeText = "Building now";
          badgeKind = "fallback";
          headline = "We are building the draft right now.";
          explanation = latestEvent?.payload?.message || "The course plan and linked assignment workflows are still being generated. This view will keep updating.";
        } else if (activeOperation === "revision") {
          badgeText = "Cloning version";
          badgeKind = "fallback";
          headline = "We are preparing the new version draft now.";
          explanation = latestEvent?.payload?.message || "The published course is being cloned into a fresh draft with new linked workflow runs.";
        } else if (activeOperation === "materialize") {
          badgeText = "Preparing bundle";
          badgeKind = "fallback";
          headline = "We are building the review package now.";
          explanation = latestEvent?.payload?.message || "The course bundle and linked workflow outputs are being materialized for review.";
        } else if (activeOperation === "publish") {
          badgeText = "Publishing now";
          badgeKind = "fallback";
          headline = "We are publishing this course now.";
          explanation = latestEvent?.payload?.message || "We are creating the learner-facing snapshot and updating the published version for new enrollments.";
        } else if (courseRun.stage === "drafting" || courseRun.status === "active") {
          badgeText = "Building now";
          badgeKind = "fallback";
          headline = "We are building the draft right now.";
          explanation = latestEvent?.payload?.message || "The course plan and linked assignment workflows are still being generated. This view will keep updating.";
        } else if (pendingGates.length) {
          const gateLabels = pendingGates
            .map((workflow) => friendlyGate(workflow.pending_gate))
            .filter(Boolean);
          badgeText = "Waiting on you";
          badgeKind = "fallback";
          headline = "This draft is waiting for your review input.";
          explanation = `The linked workflow is paused for ${gateLabels.join(", ")}. Approve the step or request changes below.`;
        } else if (courseRun.stage === "blocked") {
          badgeText = "Blocked";
          badgeKind = "fallback";
          headline = "The draft hit an error.";
          explanation = courseRun.last_error || "Check the activity feed and latest review blockers to see what stopped generation.";
        } else if (courseRun.last_error) {
          badgeText = "Needs attention";
          badgeKind = "fallback";
          headline = "The last author action needs attention.";
          explanation = courseRun.last_error;
        } else if (courseRun.stage === "ready_to_publish") {
          badgeText = "Ready to publish";
          badgeKind = "live";
          headline = "Everything is lined up for publishing.";
          explanation = "Materialize the bundle if needed, then publish when you’re happy with the course.";
        }

        draftStageBadge.textContent = badgeText;
        draftStageBadge.className = `badge ${badgeKind}`;
        const ownerToneClass = statusModel.owner === "You"
          ? "owner-you"
          : statusModel.owner === "Done"
            ? "owner-done"
            : "owner-agent";
        draftStatusSummary.innerHTML = `
          <div class="summary-item focus-headline ${ownerToneClass}">
            <p class="owner-line">${escapeHtml(statusModel.owner === "You"
              ? "Waiting on you"
              : statusModel.owner === "Done"
                ? "All done"
                : "Agent is working")}</p>
            <h4>${escapeHtml(headline)}</h4>
            <p>${escapeHtml(explanation)}</p>
            <p class="focus-footnote"><strong>Up next:</strong> ${escapeHtml(statusModel.nextTask)}</p>
          </div>
        `;
        if (statusModel.owner === "Done") {
          draftUnblockSummary.innerHTML = `
            <div class="focus-callouts focus-callouts-single">
              <div class="focus-callout summary-item-emphasis">
                <span>What to do next</span>
                <p>${escapeHtml(unblockModel.unblock)}</p>
                <button type="button" class="button primary focus-callout-cta" data-focus-action="start-new-version">Start a new version</button>
              </div>
            </div>
          `;
        } else if (statusModel.owner === "You") {
          draftUnblockSummary.innerHTML = `
            <div class="focus-callouts">
              <div class="focus-callout summary-item-emphasis">
                <span>What to do</span>
                <p>${escapeHtml(unblockModel.unblock)}</p>
                <a class="button primary focus-callout-cta" href="#review-step-panel">Open the review step</a>
              </div>
              <div class="focus-callout">
                <span>Why we're waiting</span>
                <p>${escapeHtml(unblockModel.reason)}</p>
              </div>
            </div>
          `;
        } else {
          draftUnblockSummary.innerHTML = `
            <div class="focus-callouts focus-callouts-single">
              <div class="focus-callout">
                <span>What's happening now</span>
                <p>${escapeHtml(unblockModel.reason)}</p>
              </div>
            </div>
          `;
        }
      }

      function renderLearnerEvalSummary(evalReport) {
        const target = document.getElementById("learner-eval-summary");
        if (!target) return;
        if (!evalReport) {
          target.innerHTML = "";
          target.classList.add("hidden");
          return;
        }
        const deliverableResults = evalReport.deliverable_results || [];
        const passed = deliverableResults.filter((r) => r.progression_observed || r.good_attempt?.passed).length;
        const total = deliverableResults.length;
        const overall = evalReport.overall_status || "unknown";
        const tone = overall === "passed" ? "passed" : overall === "blocked" ? "blocked" : "neutral";
        target.classList.remove("hidden");
        target.innerHTML = `
          <div class="panel-header">
            <h3>Learner test pass</h3>
            <span class="badge ${tone === "passed" ? "live" : tone === "blocked" ? "fallback" : "fallback"}">${escapeHtml(titleCase(overall))}</span>
          </div>
          <div class="panel-body">
            <p class="learner-eval-line">${passed}/${total} deliverables cleared in the latest learner walkthrough.</p>
            <p class="learner-eval-meta">Run on ${escapeHtml(formatDate(evalReport.created_at))}</p>
            ${evalReport.notes && evalReport.notes.length ? `
              <ul class="learner-eval-notes">${evalReport.notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul>
            ` : ""}
          </div>
        `;
      }

      function renderActivity(events) {
        currentEvents = events;
        if (!events.length) {
          draftActivity.innerHTML = `<div class="review-item"><p>No activity recorded yet.</p></div>`;
          return;
        }
        draftActivity.innerHTML = [...events].slice(-8).reverse().map((event) => {
          const message = formatEventMessage(event) || "No extra detail recorded for this step.";
          return `
            <div class="review-item">
              <p><strong>${escapeHtml(friendlyEventTitle(event.event_type))}</strong></p>
              <p>${escapeHtml(formatDate(event.created_at))}</p>
              <p>${escapeHtml(message)}</p>
            </div>
          `;
        }).join("");
      }

      function shouldPollDraft(courseRun, review) {
        if (!courseRun) return false;
        return currentTab === "drafts";
      }

      async function fetchWorkflowDetails(workflows) {
        const pending = workflows.filter((workflow) => workflow.pending_gate);
        const details = {};
        await Promise.all(pending.map(async (workflow) => {
          if (workflowDetailCache.has(workflow.run_id)) {
            details[workflow.run_id] = workflowDetailCache.get(workflow.run_id);
            return;
          }
          const response = await fetch(`/v1/workflow-runs/${workflow.run_id}`);
          if (!response.ok) {
            return;
          }
          const payload = await response.json();
          workflowDetailCache.set(workflow.run_id, payload);
          details[workflow.run_id] = payload;
        }));
        return details;
      }

      function stopDraftPolling() {
        if (draftPollHandle !== null) {
          window.clearInterval(draftPollHandle);
          draftPollHandle = null;
        }
      }

      function ensureDraftPolling(courseRunId) {
        stopDraftPolling();
        draftPollHandle = window.setInterval(async () => {
          if (!currentCourseRun || currentCourseRun.id !== courseRunId) {
            stopDraftPolling();
            return;
          }
          try {
            const { courseRun, review } = await refreshDraftDetails(courseRunId, { silent: true });
            if (!shouldPollDraft(courseRun, review)) {
              stopDraftPolling();
            }
          } catch (_error) {
            stopDraftPolling();
          }
        }, 5000);
      }

      function renderReview(review, courseRun, events, workflowDetails = {}) {
        currentReview = review;
        const counts = review.counts || {};
        const workflows = review.linked_workflows || [];
        const pendingWorkflows = workflows.filter((workflow) => workflow.pending_gate);
        const statusModel = buildStatusModel(courseRun, review, events);
        const readyModules = counts.ready_deliverables ?? 0;
        const totalModules = counts.total_deliverables ?? courseRun.deliverables.length;
        const bundleCount = counts.deliverables_with_bundle ?? 0;
        const publishedAssignments = counts.published_workflow_runs ?? 0;

        reviewMetrics.innerHTML = `
          <div class="metric-item"><p><strong>Deliverables ready</strong><br />${readyModules} of ${totalModules}</p></div>
          <div class="metric-item"><p><strong>Review bundles</strong><br />${bundleCount} ready</p></div>
          <div class="metric-item"><p><strong>Assignment versions ready</strong><br />${publishedAssignments} published</p></div>
          <div class="metric-item"><p><strong>Pending on</strong><br />${escapeHtml(statusModel.owner)}</p></div>
        `;

        progressTimeline.innerHTML = buildTimeline(courseRun, review).map((step) => `
          <div class="timeline-item ${step.state}">
            <span class="timeline-dot"></span>
            <div>
              <h4>${escapeHtml(step.title)}</h4>
              <p>${escapeHtml(step.detail)}</p>
            </div>
          </div>
        `).join("");

        const blockers = review.blockers || [];
        reviewBlockers.innerHTML = blockers.length
          ? blockers.map((item) => `<div class="review-item"><p>${escapeHtml(humanizeAuthorCopy(item))}</p></div>`).join("")
          : `<div class="review-item"><p>No blockers are in the way right now.</p></div>`;

        const nextActions = review.next_actions || [];
        reviewActions.innerHTML = nextActions.length
          ? nextActions.map((item) => `<div class="review-item"><p>${escapeHtml(humanizeAuthorCopy(item))}</p></div>`).join("")
          : `<div class="review-item"><p>No next action is recorded yet.</p></div>`;

        const emptyReviewMarkup = (() => {
          if (courseRun.status === "published") {
            return `
              <div class="review-empty">
                <p class="review-empty-eyebrow">No review needed</p>
                <h4>This version is published</h4>
                <p>Learners are pinned to the published snapshot. Start a new version when you want learner-facing changes.</p>
              </div>
            `;
          }
          if (courseRun.stage === "ready_to_publish") {
            return `
              <div class="review-empty">
                <p class="review-empty-eyebrow">Ready to publish</p>
                <h4>No more reviews are pending</h4>
                <p>The linked assignment work is approved. Use <strong>Publish this version</strong> in the bar above when you're happy with the result.</p>
              </div>
            `;
          }
          if (courseRun.active_operation) {
            return `
              <div class="review-empty">
                <p class="review-empty-eyebrow">Agent is working</p>
                <h4>Nothing for you to review yet</h4>
                <p>${escapeHtml(humanizeAuthorCopy(statusModel.latestMessage))}</p>
                <p class="review-empty-hint">A review step will appear here as soon as the agent pauses for a decision.</p>
              </div>
            `;
          }
          if (courseRun.stage === "blocked" || courseRun.last_error) {
          return `
            <div class="review-empty review-empty-warn">
              <p class="review-empty-eyebrow">Blocked</p>
              <h4>The draft hit a problem</h4>
              <p>${escapeHtml(courseRun.last_error || "Open the activity feed to see what stopped the build.")}</p>
              <p class="review-empty-hint">This page updates automatically. Start a new version if you want to retry from a clean draft.</p>
            </div>
          `;
          }
          return `
            <div class="review-empty">
              <p class="review-empty-eyebrow">No reviews waiting</p>
              <h4>Nothing is queued for your review</h4>
              <p>The agent is preparing the next checkpoint. We'll surface it here as soon as it's ready.</p>
            </div>
          `;
        })();

        linkedWorkflows.innerHTML = pendingWorkflows.length
          ? pendingWorkflows.map((workflow) => {
              const summary = workflow.review_summary || null;
              const blockers = summary?.blockers || [];
              const pendingGate = workflow.pending_gate || null;
              const rejectMode = workflowRejectMode.has(workflow.run_id);
              const commentValue = workflowCommentCache.get(workflow.run_id) || "";
              const detail = workflowDetails[workflow.run_id] || currentWorkflowDetails[workflow.run_id] || workflowDetailCache.get(workflow.run_id);
              const artifactKind = reviewArtifactKind(detail);
              const hasReviewArtifact = Boolean(artifactKind);
              const reviewPromptByGate = {
                gate_1_spec_review: "Approve if the assignment contract, tools, checks, and endpoints match the learner task you want to ship.",
                gate_2_progression_review: "Approve if the deliverable plan teaches the work in the right order and the readiness checks make sense.",
                gate_3_pre_publish: "Approve if this assignment package is ready to publish for learners.",
              };
              const gatePrompt = pendingGate === "gate_1_spec_review" && artifactKind === "archetype_blueprint"
                ? "Approve if the blueprint chooses the right archetype, inputs, starter shape, and evaluation surface for the course you want to build next."
                : (reviewPromptByGate[pendingGate] || "Approve if this review step matches your intent, or request changes with the exact fix.");
              return `
                <div class="module-item review-workbench">
                  ${renderPlainReviewSummary(workflow, detail)}
                  <details class="review-technical-details">
                    <summary>Technical details</summary>
                    ${renderSpecSnapshot(workflow, detail)}
                  </details>
                  <div class="review-pane-footer">
                    <div class="review-footer-note ${blockers.length ? "warning" : ""}">
                      ${blockers.length
                        ? `<div class="review-list">${blockers.map((item) => `<div class="review-item"><p>${escapeHtml(item)}</p></div>`).join("")}</div>`
                        : `<p>${escapeHtml(
                            hasReviewArtifact
                              ? gatePrompt
                              : "Review details are still loading. This page will keep updating until the review package is ready."
                          )}</p>`}
                    </div>
                    <div class="workflow-action-bar">
                      <button
                        class="button primary"
                        type="button"
                        data-workflow-action="approve"
                        data-run-id="${escapeHtml(workflow.run_id)}"
                        data-gate="${escapeHtml(pendingGate)}"
                        ${hasReviewArtifact ? "" : "disabled"}
                      >
                        Approve
                      </button>
                      <button
                        class="button danger-outline"
                        type="button"
                        data-workflow-action="${rejectMode ? "submit-reject" : "open-reject"}"
                        data-run-id="${escapeHtml(workflow.run_id)}"
                        data-gate="${escapeHtml(pendingGate)}"
                        ${hasReviewArtifact ? "" : "disabled"}
                      >
                        ${rejectMode ? "Submit change request" : "Request changes"}
                      </button>
                    </div>
                    <div class="workflow-feedback ${rejectMode ? "" : "hidden"}">
                      <label for="workflow-comment-${escapeHtml(workflow.run_id)}">Reviewer note</label>
                      <textarea
                        id="workflow-comment-${escapeHtml(workflow.run_id)}"
                        data-workflow-comment
                        data-run-id="${escapeHtml(workflow.run_id)}"
                        placeholder="Describe the changes you want the next authoring step to make."
                      >${escapeHtml(commentValue)}</textarea>
                      <div class="workflow-feedback-actions">
                        <p class="field-hint">Explain what to fix before rerunning the authoring loop.</p>
                        <button
                          class="button subtle"
                          type="button"
                          data-workflow-action="cancel-reject"
                          data-run-id="${escapeHtml(workflow.run_id)}"
                          data-gate="${escapeHtml(pendingGate)}"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              `;
            }).join("")
          : emptyReviewMarkup;

        publishButton.disabled = review.stage !== "ready_to_publish" || Boolean(currentCourseRun?.active_operation);

        const reviewPanel = document.getElementById("review-step-panel");
        if (reviewPanel instanceof HTMLDetailsElement) {
          reviewPanel.open = pendingWorkflows.length > 0
            || courseRun.active_operation
            || courseRun.stage === "blocked";
        }
        const reviewTitle = document.getElementById("review-step-title");
        const reviewCopy = document.getElementById("review-step-copy");
        if (reviewTitle && reviewCopy) {
          if (pendingWorkflows.length > 0) {
            reviewTitle.textContent = "Decision needed";
            reviewCopy.textContent = "Approve if this matches what you want, or ask for specific changes below.";
          } else if (courseRun.status === "published") {
            reviewTitle.textContent = "Published";
            reviewCopy.textContent = "Learners are pinned to this version's snapshot.";
          } else if (courseRun.stage === "ready_to_publish") {
            reviewTitle.textContent = "Ready to publish";
            reviewCopy.textContent = "All checkpoints are approved. Use Publish this version when you're ready.";
          } else if (courseRun.active_operation) {
            reviewTitle.textContent = "Agent is working";
            reviewCopy.textContent = "Nothing to review yet. We will surface the next checkpoint here.";
          } else if (courseRun.stage === "blocked") {
            reviewTitle.textContent = "Blocked";
            reviewCopy.textContent = "The draft hit a problem. Use the activity feed below to see what happened.";
          } else {
            reviewTitle.textContent = "Review this step";
            reviewCopy.textContent = "Approve to keep the draft moving, or request changes with what you want fixed.";
          }
        }
        document.body.classList.toggle("dashboard-published-state", courseRun.status === "published");
      }

      function renderPublishedVersions(versionList) {
        const versions = versionList?.versions || [];
        if (!versions.length) {
          publishedVersions.innerHTML = `<div class="review-item"><p>No published versions yet. Once you publish, we will pin learners to an immutable snapshot here.</p></div>`;
          return;
        }

        publishedVersions.innerHTML = versions.map((version) => `
          <div class="module-item">
            <h4>v${escapeHtml(version.version)}</h4>
            <p>${escapeHtml(new Date(version.created_at).toLocaleString())}</p>
            <div class="pill-row">
              ${version.default_for_new_enrollments ? pill("Default for new enrollments") : ""}
              ${version.is_latest ? pill("Latest publish") : ""}
              ${pill(`${version.learner_count} learner${version.learner_count === 1 ? "" : "s"} pinned`)}
              ${pill(`${version.deliverable_count} deliverable${version.deliverable_count === 1 ? "" : "s"}`)}
              ${pill("New enrollments only")}
            </div>
            <div class="review-list">
              ${(version.changes || []).map((item) => `<div class="review-item"><p>${escapeHtml(item)}</p></div>`).join("")}
            </div>
          </div>
        `).join("");
      }

      async function submitWorkflowDecision(runId, gate, decision, comment) {
        const response = await fetch(`/v1/workflow-runs/${runId}/decisions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            gate,
            decision,
            comment: comment || null,
          }),
        });
        if (!response.ok) {
          throw new Error(await extractDetail(response));
        }
        return response.json();
      }

      async function refreshDraftDetails(courseRunId, options = {}) {
        const [courseResponse, creatorViewResponse, eventsResponse] = await Promise.all([
          fetch(`/v1/course-runs/${courseRunId}/sync`, { method: "POST" }),
          fetch(`/v1/course-runs/${courseRunId}/creator-view`),
          fetch(`/v1/course-runs/${courseRunId}/events`),
        ]);
        if (!courseResponse.ok) {
          const detail = await extractDetail(courseResponse);
          throw new Error(detail);
        }
        if (!creatorViewResponse.ok) {
          const detail = await extractDetail(creatorViewResponse);
          throw new Error(detail);
        }
        if (!eventsResponse.ok) {
          const detail = await extractDetail(eventsResponse);
          throw new Error(detail);
        }
        const courseRun = await courseResponse.json();
        const creatorView = await creatorViewResponse.json();
        const events = await eventsResponse.json();
        const review = creatorView.review || { linked_workflows: [], counts: {}, blockers: [], next_actions: [] };
        const versions = creatorView.published_versions || { versions: [] };
        currentLearnerEval = creatorView.latest_learner_evaluation || null;
        currentCreatorFeedback = creatorView.creator_feedback || [];
        const workflowDetails = options.includeWorkflowDetails === false
          ? {}
          : await fetchWorkflowDetails(review.linked_workflows || []);
        currentWorkflowDetails = workflowDetails;
        results.classList.add("visible");
        renderPersistedPlan(courseRun);
        renderCourse(courseRun);
        renderDraftStatus(courseRun, review, events);
        renderWorkflowProgress(courseRun, review, events);
        renderReview(review, courseRun, events, workflowDetails);
        renderPublishedVersions(versions);
        renderActivity(events);
        renderLearnerEvalSummary(currentLearnerEval);
        if (!options.silent && shouldPollDraft(courseRun, review)) {
          ensureDraftPolling(courseRun.id);
        }
        if (!options.silent && !shouldPollDraft(courseRun, review)) {
          stopDraftPolling();
        }
        return { courseRun, review, events, versions };
      }

      function renderRecentDrafts(runs) {
        const query = draftSearchInput.value.trim().toLowerCase();
        const visibleRuns = query
          ? runs.filter((run) => run.title.toLowerCase().includes(query))
          : runs;
        renderDraftSwitcherMenu(runs);

        if (!runs.length) {
          recentDrafts.innerHTML = `<div class="review-item"><p>No drafts yet. Start building above and it will show up here.</p></div>`;
          return;
        }

        if (!visibleRuns.length) {
          recentDrafts.innerHTML = `<div class="review-item"><p>No drafts match “${escapeHtml(query)}”.</p></div>`;
          return;
        }

        recentDrafts.innerHTML = visibleRuns.map((run) => buildDraftOptionMarkup(run)).join("");
      }

      async function refreshRecentDrafts() {
        const response = await fetch("/v1/course-runs");
        if (!response.ok) {
          throw new Error(await extractDetail(response));
        }
        const payload = await response.json();
        recentDraftRuns = payload.runs || [];
        renderRecentDrafts(recentDraftRuns);
        return recentDraftRuns;
      }

      async function loadCourseDraft(courseRunId, options = {}) {
        draftLoadInProgress = true;
        pendingDraftId = courseRunId;
        renderWorkflowProgress();
        updateWorkspaceChrome();
        if (!options.silentMessage) {
          setMessage(formMessage, "info", "Loading the selected draft...");
        }
        try {
          if (currentCourseRun?.id !== courseRunId) {
            workflowRejectMode.clear();
            workflowCommentCache.clear();
          }
          const [courseResponse] = await Promise.all([
            fetch(`/v1/course-runs/${courseRunId}`),
          ]);
          if (!courseResponse.ok) {
            throw new Error(await extractDetail(courseResponse));
          }
          const courseRun = await courseResponse.json();
          results.classList.add("visible");
          renderPersistedPlan(courseRun);
          renderCourse(courseRun);
          pendingDraftId = null;
          setActiveTab(options.tabAfterLoad || "drafts", { updateUrl: false });
          writeUrlState({ draftId: courseRunId, tab: options.tabAfterLoad || "drafts" }, options.historyMode || "push");
          if (options.scrollToResult) {
            scrollDraftIntoView();
          }
          if (!options.silentMessage) {
            setMessage(formMessage, "info", "Loading draft details...");
          }
          await refreshRecentDrafts();
          await refreshDraftDetails(courseRunId, { includeWorkflowDetails: options.includeWorkflowDetails });
          if (!options.silentMessage) {
            setMessage(formMessage, "success", "Draft loaded.");
          }
        } finally {
          draftLoadInProgress = false;
          if (currentCourseRun?.id !== courseRunId) {
            pendingDraftId = currentCourseRun?.id || null;
          }
          updateWorkspaceChrome();
          renderWorkflowProgress();
        }
      }

      async function extractDetail(response) {
        try {
          const payload = await response.json();
          if (payload.detail) return payload.detail;
          return JSON.stringify(payload);
        } catch (_error) {
          return `${response.status} ${response.statusText}`;
        }
      }

      creatorStep1Next?.addEventListener("click", async () => {
        creatorStep1Next.disabled = true;
        try {
          const ok = await fetchCreatorSuggestedOutcomes();
          if (ok) {
            renderCreatorOutcomes();
            showCreatorStep(2);
          }
        } finally {
          creatorStep1Next.disabled = false;
        }
      });

      creatorStep2Next?.addEventListener("click", () => {
        syncOutcomesFromInputs();
        if (!creatorState.outcomes.length) {
          setMessage(formMessage, "error", "Add at least one outcome before continuing.");
          return;
        }
        applyCreatorChoicesToInputs(creatorState.choices);
        if (creatorDataSourcePurpose && !(creatorState.choices.data_sources || []).length) {
          creatorDataSourcePurpose.value = defaultDataSourcePurpose();
        }
        showCreatorStep(3);
      });

      creatorStep3Next?.addEventListener("click", async () => {
        creatorStep3Next.disabled = true;
        try {
          const ok = await fetchCreatorPlan();
          if (ok) {
            renderCreatorPlanPreview();
            showCreatorStep(4);
          }
        } finally {
          creatorStep3Next.disabled = false;
        }
      });

      generateButton?.addEventListener("click", async (event) => {
        event.preventDefault();
        await createDraftFromCreatorPlan();
      });

      creatorAddOutcome?.addEventListener("click", () => {
        syncOutcomesFromInputs();
        creatorState.outcomes.push("");
        renderCreatorOutcomes();
        const inputs = creatorOutcomesList?.querySelectorAll("input[data-outcome-index]");
        inputs?.[inputs.length - 1]?.focus();
      });

      creatorOutcomesList?.addEventListener("click", (event) => {
        const target = event.target;
        if (!(target instanceof HTMLElement)) return;
        const removeBtn = target.closest("[data-remove-outcome]");
        if (removeBtn instanceof HTMLElement) {
          syncOutcomesFromInputs();
          const idx = parseInt(removeBtn.dataset.removeOutcome || "-1", 10);
          if (idx >= 0) {
            creatorState.outcomes.splice(idx, 1);
            renderCreatorOutcomes();
          }
        }
      });

      creatorOutcomesList?.addEventListener("input", (event) => {
        if (!(event.target instanceof HTMLInputElement)) return;
        const inputs = creatorOutcomesList.querySelectorAll("input[data-outcome-index]");
        outcomesCount.textContent = `${inputs.length} outcome${inputs.length === 1 ? "" : "s"}`;
      });

      creatorSelectedDataSources?.addEventListener("click", (event) => {
        const target = event.target;
        if (!(target instanceof HTMLElement)) return;
        const action = target.closest("[data-remove-data-source]");
        if (!(action instanceof HTMLElement)) return;
        const assetId = action.dataset.removeDataSource;
        if (!assetId) return;
        detachCreatorAsset(assetId);
      });

      creatorAssetLibrary?.addEventListener("click", (event) => {
        const target = event.target;
        if (!(target instanceof HTMLElement)) return;
        const action = target.closest("[data-toggle-data-source]");
        if (!(action instanceof HTMLElement)) return;
        const assetId = action.dataset.toggleDataSource;
        if (!assetId) return;
        const selectedIds = selectedCreatorAssetIds();
        if (selectedIds.has(assetId)) {
          detachCreatorAsset(assetId);
          return;
        }
        const asset = creatorState.assets.find((item) => item.id === assetId);
        if (!asset) {
          setMessage(formMessage, "error", "That uploaded file is no longer available.");
          return;
        }
        attachCreatorAsset(asset);
      });

      creatorUploadDataSourceButton?.addEventListener("click", async () => {
        await uploadCreatorAssets();
      });

      document.querySelectorAll("[data-creator-prev]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const target = parseInt(btn.dataset.creatorPrev, 10);
          if (creatorState.step === 2) syncOutcomesFromInputs();
          if (target) showCreatorStep(target);
        });
      });

      suggestOutcomesButton?.addEventListener("click", async () => {
        const hasOutcomes = creatorState.outcomes.length > 0;
        if (hasOutcomes && !window.confirm("Replace the current outcomes with a fresh AI suggestion?")) {
          return;
        }
        suggestOutcomesButton.disabled = true;
        try {
          const ok = await fetchCreatorSuggestedOutcomes();
          if (ok) {
            renderCreatorOutcomes();
          }
        } finally {
          suggestOutcomesButton.disabled = false;
        }
      });

      materializeButton.addEventListener("click", async () => {
        if (!currentCourseRun) return;
        materializeButton.disabled = true;
        setMessage(formMessage, "info", "Building the review package and following it in Drafts...");
        try {
          const response = await fetch(materializeUrlTemplate.replace("{course_run_id}", currentCourseRun.id), {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ overwrite: true }),
          });
          if (!response.ok) {
            throw new Error(await extractDetail(response));
          }
          const payload = await response.json();
          await refreshDraftDetails(payload.course_run.id);
          await refreshRecentDrafts();
          setMessage(formMessage, "success", "Review package build started. We’ll keep updating the Drafts tab until the files are ready.");
        } catch (error) {
          setMessage(formMessage, "error", error instanceof Error ? error.message : "Could not build the review package.");
        } finally {
          materializeButton.disabled = currentCourseRun?.stage === "drafting" || Boolean(currentCourseRun?.active_operation);
        }
      });

      publishButton.addEventListener("click", async () => {
        if (!currentCourseRun) return;
        publishButton.disabled = true;
        setMessage(formMessage, "info", "Starting publish and following it in Drafts...");
        try {
          const response = await fetch(publishUrlTemplate.replace("{course_run_id}", currentCourseRun.id), {
            method: "POST",
          });
          if (!response.ok) {
            throw new Error(await extractDetail(response));
          }
          const payload = await response.json();
          await refreshDraftDetails(payload.course_run.id);
          await refreshRecentDrafts();
          setMessage(formMessage, "success", "Publishing started. We’ll keep updating the Drafts tab until the new learner snapshot is live.");
        } catch (error) {
          setMessage(formMessage, "error", error instanceof Error ? error.message : "Could not publish the course.");
        } finally {
          publishButton.disabled = currentReview?.stage !== "ready_to_publish" || Boolean(currentCourseRun?.active_operation);
        }
      });

      createRevisionButton.addEventListener("click", async () => {
        if (!currentCourseRun) return;
        const confirmed = window.confirm("Start a new draft version from the current published course? Existing learners will stay pinned to the already published version.");
        if (!confirmed) return;
        createRevisionButton.disabled = true;
        setMessage(formMessage, "info", "Starting a new version draft and moving it into Drafts...");
        try {
          const response = await fetch(createRevisionUrlTemplate.replace("{course_run_id}", currentCourseRun.id), {
            method: "POST",
          });
          if (!response.ok) {
            throw new Error(await extractDetail(response));
          }
          const payload = await response.json();
          await loadCourseDraft(payload.course_run.id, {
            silentMessage: true,
            historyMode: "push",
            tabAfterLoad: "drafts",
          });
          setMessage(formMessage, "success", "New version draft started. We’ll keep updating the Drafts tab while the linked workflows are cloned and checked.");
        } catch (error) {
          setMessage(formMessage, "error", error instanceof Error ? error.message : "Could not create a new course version.");
        } finally {
          createRevisionButton.disabled = currentCourseRun?.status !== "published";
        }
      });

      linkedWorkflows.addEventListener("input", (event) => {
        const target = event.target;
        if (!(target instanceof HTMLTextAreaElement)) return;
        if (!target.matches("[data-workflow-comment]")) return;
        const runId = target.dataset.runId;
        if (!runId) return;
        workflowCommentCache.set(runId, target.value);
      });

      linkedWorkflows.addEventListener("click", async (event) => {
        const target = event.target;
        if (!(target instanceof HTMLElement)) return;

        const trigger = target.closest("[data-workflow-action]");
        if (!(trigger instanceof HTMLElement)) return;
        const action = trigger.dataset.workflowAction;
        if (!action || !currentCourseRun) return;

        const runId = trigger.dataset.runId;
        const gate = trigger.dataset.gate;
        if (!runId || !gate) return;
        const card = trigger.closest(".module-item");
        const noteField = card ? card.querySelector("[data-workflow-comment]") : null;
        const note = noteField instanceof HTMLTextAreaElement ? noteField.value.trim() : "";

        if (action === "open-reject") {
          workflowRejectMode.add(runId);
          renderReview(currentReview, currentCourseRun, currentEvents, currentWorkflowDetails);
          const refreshedNoteField = document.getElementById(`workflow-comment-${runId}`);
          if (refreshedNoteField instanceof HTMLTextAreaElement) {
            refreshedNoteField.focus();
          }
          return;
        }

        if (action === "cancel-reject") {
          workflowRejectMode.delete(runId);
          workflowCommentCache.delete(runId);
          renderReview(currentReview, currentCourseRun, currentEvents, currentWorkflowDetails);
          return;
        }

        if (action === "submit-reject" && !note) {
          setMessage(formMessage, "error", "Add a reviewer note before requesting changes.");
          return;
        }

        const decision = action === "submit-reject" ? "reject" : action;
        const button = trigger;
        button.setAttribute("disabled", "true");
        setMessage(
          formMessage,
          "info",
          `${decision === "approve" ? "Approving" : "Requesting changes for"} ${friendlyGate(gate)}...`,
        );

        try {
          await submitWorkflowDecision(runId, gate, decision, note);
          workflowDetailCache.delete(runId);
          workflowRejectMode.delete(runId);
          workflowCommentCache.delete(runId);
          await refreshDraftDetails(currentCourseRun.id);
          setMessage(
            formMessage,
            "success",
            `${decision === "approve" ? "Approved" : "Requested changes for"} ${friendlyGate(gate)}.`,
          );
        } catch (error) {
          setMessage(
            formMessage,
            "error",
            error instanceof Error ? error.message : "Could not apply the workflow gate decision.",
          );
        } finally {
          button.removeAttribute("disabled");
        }
      });

      recentDrafts.addEventListener("click", async (event) => {
        const target = event.target;
        if (!(target instanceof HTMLElement)) return;
        const trigger = target.closest("[data-load-course-run]");
        if (!(trigger instanceof HTMLElement)) return;
        const courseRunId = trigger.dataset.loadCourseRun;
        if (!courseRunId) return;
        trigger.setAttribute("disabled", "true");
        try {
          await loadCourseDraft(courseRunId, { scrollToResult: true });
        } catch (error) {
          setMessage(
            formMessage,
            "error",
            error instanceof Error ? error.message : "Could not load the selected course draft.",
          );
        } finally {
          trigger.removeAttribute("disabled");
        }
      });

      draftSwitcherList.addEventListener("click", async (event) => {
        const target = event.target;
        if (!(target instanceof HTMLElement)) return;
        const trigger = target.closest("[data-switch-course-run]");
        if (!(trigger instanceof HTMLElement)) return;
        const courseRunId = trigger.dataset.switchCourseRun;
        if (!courseRunId) return;
        trigger.setAttribute("disabled", "true");
        try {
          await loadCourseDraft(courseRunId, {
            silentMessage: true,
            historyMode: "push",
            tabAfterLoad: currentTab === "drafts" ? "drafts" : currentTab,
            scrollToResult: currentTab === "drafts",
          });
          if (draftSwitcher instanceof HTMLDetailsElement) {
            draftSwitcher.open = false;
          }
        } catch (error) {
          setMessage(
            formMessage,
            "error",
            error instanceof Error ? error.message : "Could not switch drafts.",
          );
        } finally {
          trigger.removeAttribute("disabled");
        }
      });

      draftSearchInput.addEventListener("input", () => {
        renderRecentDrafts(recentDraftRuns);
      });

      createDraftShortcut.addEventListener("click", () => {
        clearSelectedDraft({ historyMode: "push", tab: "create" });
        goalField.scrollIntoView({ behavior: "smooth", block: "start" });
      });

      generationSetupToggle.addEventListener("click", () => {
        const isHidden = generationSetup.classList.toggle("hidden");
        generationSetupToggle.textContent = isHidden ? "Show setup steps" : "Hide setup steps";
      });

      allDraftsButton.addEventListener("click", () => {
        clearSelectedDraft({ historyMode: "push", tab: "drafts" });
      });

      resetLocalButton.addEventListener("click", async () => {
        const confirmed = window.confirm("Clear all local drafts, workflows, bundles, and workspaces? This will give you a clean slate.");
        if (!confirmed) return;

        resetLocalButton.setAttribute("disabled", "true");
        setMessage(formMessage, "info", "Clearing all local drafts and generated artifacts...");
        try {
          const response = await fetch(state.reset_local_url, {
            method: "POST",
          });
          if (!response.ok) {
            throw new Error(await extractDetail(response));
          }
          const payload = await response.json();
          resetDraftSelection();
          window.localStorage.removeItem(activeDraftStorageKey);
          setActiveTab("create", { updateUrl: false });
          writeUrlState({ draftId: null, tab: "create" }, "replace");
          await refreshRecentDrafts();
          await refreshCreatorAssets();
          setMessage(
            formMessage,
            "success",
            `Cleared ${payload.deleted_course_runs} drafts and ${payload.deleted_workflow_runs} workflow runs.`,
          );
        } catch (error) {
          setMessage(
            formMessage,
            "error",
            error instanceof Error ? error.message : "Could not clear local drafts.",
          );
        } finally {
          resetLocalButton.removeAttribute("disabled");
        }
      });

      createTabButton.addEventListener("click", () => clearSelectedDraft({ historyMode: "push", tab: "create" }));
      draftsTabButton.addEventListener("click", () => setActiveTab("drafts", { historyMode: "push" }));
      goalField.addEventListener("input", updateBriefCounters);

      document.addEventListener("click", (event) => {
        const target = event.target;
        if (!(target instanceof Element)) return;
        const anchor = target.closest('a[href^="#"]');
        if (anchor instanceof HTMLAnchorElement) {
          revealPanelFromHash(anchor.getAttribute("href"));
        }
        const focusAction = target.closest("[data-focus-action]");
        if (focusAction instanceof HTMLElement) {
          const action = focusAction.dataset.focusAction;
          if (action === "start-new-version" && createRevisionButton && !createRevisionButton.disabled) {
            createRevisionButton.click();
          }
        }
      });

      renderCreatorOutcomes();
      if (creatorDataSourcePurpose && !creatorDataSourcePurpose.value) {
        creatorDataSourcePurpose.value = defaultDataSourcePurpose();
      }
      renderCreatorDataSources();
      showCreatorStep(1);
      updateBriefCounters();
      const urlState = readUrlState();
      pendingDraftId = urlState.draftId || null;
      renderWorkflowProgress();
      renderGenerationStatus(state.generation_status);
      const storedTab = window.localStorage.getItem(activeTabStorageKey) || "create";
      setActiveTab(
        urlState.tab || storedTab,
        { updateUrl: false },
      );
      refreshRecentDrafts().then(async (runs) => {
        const rememberedDraftId = urlState.draftId
          || (((urlState.tab || storedTab) === "drafts")
            ? window.localStorage.getItem(activeDraftStorageKey)
            : null);
        const draftToLoad = rememberedDraftId || null;
        if (draftToLoad) {
          try {
            await loadCourseDraft(draftToLoad, {
              silentMessage: true,
              historyMode: "replace",
              tabAfterLoad: urlState.tab || storedTab || "drafts",
            });
          } catch (_error) {
            window.localStorage.removeItem(activeDraftStorageKey);
            writeUrlState({ draftId: null, tab: urlState.tab || storedTab || "create" }, "replace");
          }
        } else {
          updateWorkspaceChrome();
        }
      }).catch((error) => {
        recentDrafts.innerHTML = `<div class="review-item"><p>${escapeHtml(error instanceof Error ? error.message : "Could not load recent drafts.")}</p></div>`;
      });
      refreshCreatorAssets();

      revealPanelFromHash(window.location.hash);

      window.addEventListener("popstate", async () => {
        const next = readUrlState();
        if (!next.draftId) {
          window.localStorage.removeItem(activeDraftStorageKey);
          resetDraftSelection();
          setActiveTab(next.tab || "create", { updateUrl: false });
          revealPanelFromHash(window.location.hash);
          return;
        }
        try {
          await loadCourseDraft(next.draftId, {
            silentMessage: true,
            historyMode: "replace",
            tabAfterLoad: next.tab || "drafts",
          });
        } catch (_error) {
          resetDraftSelection();
          writeUrlState({ draftId: null, tab: "create" }, "replace");
        }
      });

      window.addEventListener("hashchange", () => {
        revealPanelFromHash(window.location.hash);
      });
})();
