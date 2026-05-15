(() => {
  const stateElement = document.getElementById("docs-state");
  const state = stateElement?.textContent ? JSON.parse(stateElement.textContent) : {};
  const statsRoot = document.getElementById("docs-stats");
  const tagsRoot = document.getElementById("docs-tag-list");
  const panelTitle = document.getElementById("docs-panel-title");
  const loading = document.getElementById("swagger-loading");

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function renderStats() {
    if (!statsRoot) return;
    const stats = [
      { label: "Operations", value: state.operation_count || 0 },
      { label: "Paths", value: state.path_count || 0 },
      { label: "Schemas", value: state.schema_count || 0 },
      { label: "Sections", value: state.tag_count || 0 },
    ];
    statsRoot.innerHTML = stats.map((item) => `
      <div class="hero-stat docs-stat">
        <span>${escapeHtml(item.label)}</span>
        <strong>${escapeHtml(String(item.value))}</strong>
      </div>
    `).join("");
  }

  function renderSections() {
    if (!tagsRoot) return;
    const sections = Array.isArray(state.sections) ? state.sections : [];
    if (!sections.length) {
      tagsRoot.innerHTML = `<span class="small-pill">No grouped sections available</span>`;
      return;
    }
    tagsRoot.innerHTML = sections.map((section) => `
      <span class="docs-tag-pill">
        <span>${escapeHtml(section.name)}</span>
        <strong>${escapeHtml(String(section.operations))}</strong>
      </span>
    `).join("");
  }

  function updateChromeCopy() {
    const productTitle = state.info?.title || "Scaler Labs";
    const version = state.info?.version ? ` · ${state.info.version}` : "";
    document.title = `API Docs · ${productTitle}`;
    if (panelTitle) {
      panelTitle.textContent = `${productTitle}${version}`;
    }
  }

  function initializeSwagger() {
    if (typeof window.SwaggerUIBundle !== "function") {
      if (loading) {
        loading.textContent = "We couldn't load the interactive reference assets.";
      }
      return;
    }

    window.ui = window.SwaggerUIBundle({
      url: state.openapi_url || "/openapi.json",
      dom_id: "#swagger-ui",
      deepLinking: true,
      docExpansion: "list",
      filter: true,
      defaultModelsExpandDepth: -1,
      displayRequestDuration: true,
      showExtensions: true,
      tryItOutEnabled: true,
      presets: [
        window.SwaggerUIBundle.presets.apis,
      ],
      plugins: [
        window.SwaggerUIBundle.plugins.DownloadUrl,
      ],
      layout: "BaseLayout",
      onComplete: () => {
        if (loading) {
          loading.remove();
        }
      },
    });
  }

  renderStats();
  renderSections();
  updateChromeCopy();
  initializeSwagger();
})();
