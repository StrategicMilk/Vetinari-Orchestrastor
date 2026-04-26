<script>
  /**
   * Output/artifact display view — code output with syntax highlighting,
   * file tree for multi-file outputs, version history, and approval controls.
   *
   * Uses highlight.js for syntax highlighting loaded via CDN script tag.
   */
  import { appState } from '$lib/stores/app.svelte.js';
  import * as api from '$lib/api.js';
  import * as fmt from '$lib/utils/format.js';
  import { showToast } from '$lib/stores/toast.svelte.js';

  // -- State -------------------------------------------------------------------

  let review = $state(null);
  let loading = $state(true);
  /** Set to a message string when the API call itself failed, distinct from "no review yet". */
  let loadError = $state(null);
  let actionPending = $state(false);
  let selectedFile = $state(null);
  let selectedVersion = $state(null);
  let hlReady = $state(false);
  let codeEl = $state(null);

  // -- Derived -----------------------------------------------------------------

  let files = $derived(
    review?.files ?? review?.artifacts ?? (review?.output ? [{ name: 'output', content: review.output, language: 'text' }] : [])
  );

  let versions = $derived(review?.versions ?? []);

  let activeFile = $derived(
    selectedFile
      ? files.find((f) => f.name === selectedFile) ?? files[0]
      : files[0]
  );

  let codeContent = $derived(activeFile?.content ?? '');
  let codeLanguage = $derived(activeFile?.language ?? detectLanguage(activeFile?.name ?? ''));

  let canApprove = $derived(review?.status === 'pending_review' || review?.status === 'needs_approval');

  // -- Data loading ------------------------------------------------------------

  async function loadReview() {
    const pid = appState.currentProjectId;
    if (!pid) {
      loading = false;
      return;
    }
    loading = true;
    loadError = null;
    try {
      const data = await api.getProjectReview(pid);
      review = data;
      if (files.length > 0) {
        selectedFile = files[0].name;
      }
    } catch (err) {
      // Distinguish a fetch failure from a legitimately empty review.
      // Leaving review as null here is intentional: no partial data to show.
      review = null;
      loadError = err?.message ?? 'Failed to load output';
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    loadReview();
  });

  // -- Highlight.js integration ------------------------------------------------

  $effect(() => {
    if (!hlReady || !codeEl || !codeContent) return;
    try {
      const hljs = window.hljs;
      if (!hljs) return;
      codeEl.textContent = codeContent;
      codeEl.className = `language-${codeLanguage}`;
      hljs.highlightElement(codeEl);
    } catch {
      // Graceful degradation if hljs isn't available
    }
  });

  function loadHighlightJs() {
    if (window.hljs) { hlReady = true; return; }
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css';
    document.head.appendChild(link);
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js';
    script.onload = () => { hlReady = true; };
    document.head.appendChild(script);
  }

  $effect(() => { loadHighlightJs(); });

  // -- Actions -----------------------------------------------------------------

  async function handleApprove() {
    if (!appState.currentProjectId) return;
    actionPending = true;
    try {
      await api.approveProject(appState.currentProjectId);
      showToast('Output approved', 'success');
      await loadReview();
    } catch (err) {
      showToast(`Approval failed: ${err.message}`, 'error');
    } finally {
      actionPending = false;
    }
  }

  async function handleReject() {
    if (!appState.currentProjectId) return;
    actionPending = true;
    try {
      await api.cancelProject(appState.currentProjectId);
      showToast('Output rejected', 'info');
      await loadReview();
    } catch (err) {
      showToast(`Rejection failed: ${err.message}`, 'error');
    } finally {
      actionPending = false;
    }
  }

  function selectVersion(v) {
    selectedVersion = v;
    // If version contains file snapshots, update review content
    if (v?.files) {
      review = { ...review, files: v.files, status: v.status };
      if (v.files.length > 0) selectedFile = v.files[0].name;
    }
  }

  async function copyCode() {
    if (!codeContent) return;
    try {
      await navigator.clipboard.writeText(codeContent);
      showToast('Copied to clipboard', 'success');
    } catch {
      showToast('Copy failed', 'error');
    }
  }

  function detectLanguage(filename) {
    const ext = filename.split('.').pop()?.toLowerCase() ?? '';
    const map = {
      py: 'python', js: 'javascript', ts: 'typescript', tsx: 'typescript',
      jsx: 'javascript', html: 'html', css: 'css', json: 'json',
      yaml: 'yaml', yml: 'yaml', sh: 'bash', md: 'markdown',
      rs: 'rust', go: 'go', java: 'java', cpp: 'cpp', c: 'c',
    };
    return map[ext] ?? 'text';
  }

  function fileIcon(name) {
    const ext = name.split('.').pop()?.toLowerCase() ?? '';
    const map = {
      py: 'fab fa-python', js: 'fab fa-js', ts: 'fab fa-js',
      html: 'fab fa-html5', css: 'fab fa-css3-alt',
      json: 'fas fa-code', md: 'fab fa-markdown',
    };
    return map[ext] ?? 'fas fa-file-code';
  }
</script>

<div class="output-view">
  <div class="view-header">
    <h2>
      <i class="fas fa-code"></i>
      Output
      {#if review}
        <span class="status-badge status-{review.status === 'pending_review' ? 'warning' : review.status === 'approved' ? 'success' : 'muted'}">
          {review.status ?? 'unknown'}
        </span>
      {/if}
    </h2>
    <div class="header-actions">
      {#if canApprove}
        <button
          class="btn btn-success"
          onclick={handleApprove}
          disabled={actionPending}
          aria-label="Approve this output"
        >
          <i class="fas fa-check"></i> Approve
        </button>
        <button
          class="btn btn-danger"
          onclick={handleReject}
          disabled={actionPending}
          aria-label="Reject this output"
        >
          <i class="fas fa-times"></i> Reject
        </button>
      {/if}
      <button
        class="btn btn-secondary btn-sm"
        onclick={loadReview}
        disabled={loading}
        aria-label="Refresh output"
      >
        <i class="fas fa-sync-alt" class:fa-spin={loading}></i>
      </button>
    </div>
  </div>

  {#if !appState.currentProjectId}
    <div class="empty-state">
      <i class="fas fa-folder-open"></i>
      <p>No project selected. Open a project to view its output.</p>
    </div>
  {:else if loading}
    <div class="loading-state" role="status" aria-live="polite">
      <i class="fas fa-spinner fa-spin"></i>
      Loading output...
    </div>
  {:else if loadError}
    <div class="error-state" role="alert">
      <i class="fas fa-exclamation-triangle"></i>
      <p>Could not load output: {loadError}</p>
      <button class="btn btn-secondary btn-sm" onclick={loadReview}>
        <i class="fas fa-redo"></i> Retry
      </button>
    </div>
  {:else if !review}
    <div class="empty-state">
      <i class="fas fa-hourglass-half"></i>
      <p>No output available yet. The project may still be in progress.</p>
    </div>
  {:else}
    <div class="output-layout" aria-live="polite">
      <!-- File tree sidebar -->
      {#if files.length > 1}
        <nav class="file-tree" aria-label="Output file tree">
          <h3 class="tree-title">
            <i class="fas fa-folder-tree"></i>
            Files
            <span class="badge">{files.length}</span>
          </h3>
          <ul class="file-list" role="listbox" aria-label="Select a file">
            {#each files as file (file.name)}
              <li>
                <button
                  class="file-btn"
                  class:active={selectedFile === file.name}
                  onclick={() => { selectedFile = file.name; }}
                  role="option"
                  aria-selected={selectedFile === file.name}
                  aria-label="View file {file.name}"
                >
                  <i class="{fileIcon(file.name)}"></i>
                  <span class="file-name">{file.name}</span>
                  {#if file.size != null}
                    <span class="file-size">{fmt.fileSize(file.size)}</span>
                  {/if}
                </button>
              </li>
            {/each}
          </ul>
        </nav>
      {/if}

      <!-- Code area -->
      <div class="code-area" aria-label="Code output">
        {#if activeFile}
          <div class="code-toolbar">
            <div class="code-toolbar-left">
              <i class="{fileIcon(activeFile.name)}"></i>
              <span class="code-filename">{activeFile.name}</span>
              <span class="code-lang">{codeLanguage}</span>
            </div>
            <div class="code-toolbar-right">
              {#if activeFile.tokens != null}
                <span class="code-meta">{fmt.integer(activeFile.tokens)} tokens</span>
              {/if}
              {#if activeFile.size != null}
                <span class="code-meta">{fmt.fileSize(activeFile.size)}</span>
              {/if}
              <button
                class="btn btn-secondary btn-sm"
                onclick={copyCode}
                aria-label="Copy code to clipboard"
                title="Copy"
              >
                <i class="fas fa-copy"></i>
              </button>
            </div>
          </div>
          <div class="code-scroll" role="region" aria-label="Code content for {activeFile.name}">
            <pre class="code-pre"><code bind:this={codeEl} class="language-{codeLanguage}">{codeContent}</code></pre>
          </div>
        {/if}
      </div>

      <!-- Version history sidebar -->
      {#if versions.length > 0}
        <aside class="version-history" aria-label="Version history">
          <h3 class="tree-title">
            <i class="fas fa-history"></i>
            Versions
          </h3>
          <ul class="version-list" role="listbox" aria-label="Output versions">
            {#each versions as v, i ((v.id ?? v.version ?? i))}
              <li>
                <button
                  class="version-btn"
                  class:active={selectedVersion === v}
                  onclick={() => selectVersion(v)}
                  role="option"
                  aria-selected={selectedVersion === v}
                  aria-label="Version {v.version ?? i + 1}: {v.status ?? ''}"
                >
                  <div class="version-row">
                    <span class="version-num">v{v.version ?? i + 1}</span>
                    <span class="status-badge status-{v.status === 'approved' ? 'success' : v.status === 'rejected' ? 'danger' : 'muted'}">
                      {v.status ?? 'draft'}
                    </span>
                  </div>
                  <div class="version-meta">
                    <span>{fmt.relativeTime(v.created_at)}</span>
                    {#if v.author}
                      <span>{v.author}</span>
                    {/if}
                  </div>
                </button>
              </li>
            {/each}
          </ul>
        </aside>
      {/if}
    </div>

    <!-- Review info footer -->
    {#if review.feedback || review.notes}
      <section class="review-notes card" aria-label="Review notes">
        <h3 class="notes-title"><i class="fas fa-comment-alt"></i> Inspector Notes</h3>
        <p class="notes-text">{review.feedback ?? review.notes}</p>
      </section>
    {/if}
  {/if}
</div>

<style>
  .output-view {
    padding: 24px;
    max-width: 1600px;
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .view-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
    gap: 12px;
    flex-shrink: 0;
  }

  .view-header h2 {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .view-header h2 i { color: var(--secondary); }

  .header-actions { display: flex; gap: 8px; }

  .loading-state {
    display: flex;
    align-items: center;
    gap: 10px;
    color: var(--text-muted);
    padding: 48px 24px;
  }

  .error-state {
    text-align: center;
    padding: 64px 24px;
    color: var(--danger);
    font-size: 0.9375rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
  }

  .error-state i { font-size: 2.5rem; opacity: 0.7; }
  .error-state p { margin: 0; }

  .empty-state {
    text-align: center;
    padding: 64px 24px;
    color: var(--text-muted);
    font-size: 0.9375rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
  }

  .empty-state i {
    font-size: 2.5rem;
    opacity: 0.35;
  }

  .empty-state p { margin: 0; }

  /* Layout */
  .output-layout {
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 0;
    flex: 1;
    min-height: 0;
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    overflow: hidden;
    background: var(--surface-bg);
  }

  /* File tree */
  .file-tree {
    width: 220px;
    border-right: 1px solid var(--border-default);
    overflow-y: auto;
    background: var(--surface-elevated);
    padding: 12px 0;
  }

  .tree-title {
    font-size: 0.8125rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 0 0 8px 0;
    padding: 0 12px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .file-list, .version-list { list-style: none; margin: 0; padding: 0; }

  .file-btn {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: none;
    border: none;
    cursor: pointer;
    font-size: 0.8125rem;
    font-family: var(--font-mono);
    color: var(--text-secondary);
    text-align: left;
    transition: background var(--transition-base), color var(--transition-base);
  }

  .file-btn:hover { background: var(--surface-hover); color: var(--text-primary); }
  .file-btn.active { background: var(--primary-muted); color: var(--primary); }

  .file-name {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .file-size {
    font-size: 0.6875rem;
    color: var(--text-muted);
    flex-shrink: 0;
  }

  /* Code area */
  .code-area {
    display: flex;
    flex-direction: column;
    min-width: 0;
    overflow: hidden;
  }

  .code-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 14px;
    border-bottom: 1px solid var(--border-default);
    background: var(--surface-elevated);
    flex-shrink: 0;
  }

  .code-toolbar-left {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.875rem;
    color: var(--text-primary);
  }

  .code-filename { font-weight: 500; font-family: var(--font-mono); }
  .code-lang {
    font-size: 0.75rem;
    color: var(--text-muted);
    background: var(--surface-hover);
    padding: 1px 6px;
    border-radius: var(--radius-sm);
  }

  .code-toolbar-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .code-meta { font-size: 0.75rem; color: var(--text-muted); }

  .code-scroll {
    flex: 1;
    overflow: auto;
  }

  .code-pre {
    margin: 0;
    padding: 16px 20px;
    font-family: var(--font-mono);
    font-size: 0.8125rem;
    line-height: 1.6;
    min-height: 100%;
    white-space: pre;
    tab-size: 2;
    background: transparent;
  }

  .code-pre code {
    font-family: inherit;
    background: none;
    padding: 0;
    border: none;
  }

  /* Version sidebar */
  .version-history {
    width: 200px;
    border-left: 1px solid var(--border-default);
    overflow-y: auto;
    background: var(--surface-elevated);
    padding: 12px 0;
  }

  .version-btn {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding: 8px 12px;
    background: none;
    border: none;
    cursor: pointer;
    font-family: inherit;
    text-align: left;
    transition: background var(--transition-base);
  }

  .version-btn:hover { background: var(--surface-hover); }
  .version-btn.active { background: var(--primary-muted); }

  .version-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
  }

  .version-num { font-weight: 600; font-size: 0.875rem; color: var(--text-primary); }

  .version-meta {
    display: flex;
    flex-direction: column;
    gap: 1px;
    font-size: 0.6875rem;
    color: var(--text-muted);
  }

  /* Review notes */
  .review-notes {
    margin-top: 16px;
    background: var(--surface-elevated);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    padding: 14px 16px;
    flex-shrink: 0;
  }

  .notes-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin: 0 0 8px 0;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .notes-text {
    font-size: 0.875rem;
    color: var(--text-secondary);
    margin: 0;
    line-height: var(--leading-relaxed);
    white-space: pre-wrap;
  }

  /* Status badges */
  .status-badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: var(--radius-full);
    font-size: 0.6875rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    flex-shrink: 0;
  }

  .status-success { background: var(--success-muted); color: var(--success); }
  .status-warning { background: var(--warning-muted); color: var(--warning); }
  .status-danger  { background: var(--danger-muted);  color: var(--danger);  }
  .status-muted   { background: var(--surface-hover); color: var(--text-muted); }

  .badge {
    background: var(--surface-hover);
    color: var(--text-muted);
    border-radius: var(--radius-full);
    padding: 1px 7px;
    font-size: 0.6875rem;
    font-weight: 600;
  }

  /* Buttons */
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    border-radius: var(--radius-md);
    font-size: 0.875rem;
    font-weight: 500;
    font-family: inherit;
    cursor: pointer;
    border: none;
    transition: background var(--transition-base);
  }

  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-sm { padding: 6px 10px; font-size: 0.8125rem; }

  .btn-secondary { background: var(--surface-hover); color: var(--text-primary); border: 1px solid var(--border-default); }
  .btn-secondary:hover:not(:disabled) { background: var(--surface-pressed); }

  .btn-success { background: var(--success-muted); color: var(--success); border: 1px solid rgba(56,211,159,0.3); }
  .btn-success:hover:not(:disabled) { background: rgba(56,211,159,0.2); }

  .btn-danger { background: var(--danger-muted); color: var(--danger); border: 1px solid rgba(240,98,98,0.3); }
  .btn-danger:hover:not(:disabled) { background: rgba(240,98,98,0.2); }

  @media (max-width: 900px) {
    .output-layout { grid-template-columns: 1fr; }
    .file-tree { width: 100%; border-right: none; border-bottom: 1px solid var(--border-default); }
    .version-history { width: 100%; border-left: none; border-top: 1px solid var(--border-default); }
  }
</style>
