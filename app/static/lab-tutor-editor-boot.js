/* Lab Tutor bootstrap for the embedded code-server editor.
 *
 * nginx injects this (a same-origin, CSP 'self'-allowed script) into the
 * code-server page with data-editor-port=<dynamic port>. nginx only knows
 * the port, not which course/assignment the workspace belongs to, so the
 * old approach hard-coded a useless "Lab workspace" title. This script
 * asks the app to resolve the port -> the owning learner's enrollment,
 * then mounts the existing widget with the SAME session_id /
 * assignment_title the LMS-page widget uses (shared tutor history).
 *
 * On any failure it still mounts with a generic context so the bubble
 * always appears.
 */
(function () {
  "use strict";
  if (window.__labTutorBootStarted) return;
  window.__labTutorBootStarted = true;

  var me = document.currentScript;
  var port = (me && me.dataset && me.dataset.editorPort) || "";

  function mountWith(ctx) {
    var opts = {
      baseUrl: "",
      sessionId: (ctx && ctx.session_id) || ("editor-" + (port || "x")),
      assignmentTitle: (ctx && ctx.assignment_title) || "Lab workspace",
    };
    function doMount() {
      if (typeof window.__labTutorMount === "function") {
        window.__labTutorMount(opts);
      }
    }
    if (typeof window.__labTutorMount === "function") {
      doMount();
      return;
    }
    // Load the widget WITHOUT data-* attributes so it does not
    // standalone-auto-mount with empty context; we mount it ourselves
    // once it is ready, with the resolved context.
    var s = document.createElement("script");
    s.src = "/static/lab-tutor.js";
    s.addEventListener("load", doMount);
    s.addEventListener("error", function () {
      /* widget asset failed to load; nothing else to do */
    });
    (document.head || document.documentElement).appendChild(s);
  }

  var done = false;
  function finish(ctx) {
    if (done) return;
    done = true;
    mountWith(ctx);
  }

  try {
    fetch("/v1/tutor/editor-context?port=" + encodeURIComponent(port), {
      credentials: "same-origin",
      headers: { accept: "application/json" },
    })
      .then(function (r) {
        return r.ok ? r.json() : null;
      })
      .then(function (ctx) {
        finish(ctx);
      })
      .catch(function () {
        finish(null);
      });
    // Safety net: if the resolver hangs, still mount generically.
    setTimeout(function () {
      finish(null);
    }, 6000);
  } catch (_e) {
    finish(null);
  }
})();
