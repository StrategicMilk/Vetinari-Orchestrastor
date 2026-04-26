<script>
  import { appState } from '$lib/stores/app.svelte.js';
  import * as api from '$lib/api.js';
  import { relativeTime } from '$lib/utils/format.js';

  /**
   * Navigation items for the primary sidebar menu.
   * icon: Font Awesome class, view: key in VIEW_MAP, label: display text.
   */
  const NAV_ITEMS = [
    { icon: 'fas fa-comments', view: 'prompt', label: 'Chat' },
    { icon: 'fas fa-microchip', view: 'models', label: 'Models' },
    { icon: 'fas fa-graduation-cap', view: 'training', label: 'Training' },
    { icon: 'fas fa-brain', view: 'memory', label: 'Memory' },
    { icon: 'fas fa-cog', view: 'settings', label: 'Settings' },
    { icon: 'fas fa-chart-line', view: 'dashboard', label: 'Dashboard' },
  ];

  /** Advanced tools (collapsible section). */
  const ADVANCED_ITEMS = [
    { icon: 'fas fa-sitemap', view: 'workflow', label: 'Projects' },
    { icon: 'fas fa-robot', view: 'agents', label: 'Agents' },
    { icon: 'fas fa-file-code', view: 'output', label: 'Output Viewer' },
    { icon: 'fas fa-tasks', view: 'tasks', label: 'Task Queue' },
    { icon: 'fas fa-project-diagram', view: 'decomposition', label: 'Plan Builder' },
  ];

  let advancedOpen = $state(false);
  let projects = $state([]);

  function switchView(view) {
    appState.currentView = view;
  }

  function toggleAdvanced() {
    advancedOpen = !advancedOpen;
  }

  async function loadProjects() {
    try {
      const data = await api.listProjects();
      projects = Array.isArray(data) ? data : data?.projects ?? [];
    } catch {
      projects = [];
    }
  }

  $effect(() => {
    loadProjects();
  });
</script>

<aside
  class="sidebar"
  class:collapsed={appState.sidebarCollapsed}
  role="navigation"
  aria-label="Main navigation"
>
  <div class="logo">
    <i class="fas fa-brain"></i>
    {#if !appState.sidebarCollapsed}
      <span>Vetinari</span>
    {/if}
  </div>

  <nav class="nav-menu" aria-label="Main menu">
    <ul class="nav-list" role="list">
      {#each NAV_ITEMS as item}
        <li role="listitem">
          <button
            class="nav-item"
            class:active={appState.currentView === item.view}
            onclick={() => switchView(item.view)}
            title={appState.sidebarCollapsed ? item.label : undefined}
          >
            <i class={item.icon}></i>
            {#if !appState.sidebarCollapsed}
              <span>{item.label}</span>
            {/if}
          </button>
        </li>
      {/each}
    </ul>

    <!-- Advanced tools divider -->
    <div class="nav-section-divider">
      <button
        class="nav-section-toggle"
        onclick={toggleAdvanced}
        aria-expanded={advancedOpen}
        title="Toggle advanced tools"
      >
        <i class="fas" class:fa-chevron-right={!advancedOpen} class:fa-chevron-down={advancedOpen}></i>
        {#if !appState.sidebarCollapsed}
          <span>Advanced Tools</span>
        {/if}
      </button>
    </div>

    {#if advancedOpen}
      <ul class="nav-advanced nav-list" role="list">
        {#each ADVANCED_ITEMS as item}
          <li role="listitem">
            <button
              class="nav-item nav-item-advanced"
              class:active={appState.currentView === item.view}
              onclick={() => switchView(item.view)}
              title={appState.sidebarCollapsed ? item.label : undefined}
            >
              <i class={item.icon}></i>
              {#if !appState.sidebarCollapsed}
                <span>{item.label}</span>
              {/if}
            </button>
          </li>
        {/each}
      </ul>
    {/if}
  </nav>

  <!-- Project list -->
  {#if !appState.sidebarCollapsed && projects.length > 0}
    <div class="projects-panel">
      <h3 class="projects-panel-title">Projects</h3>
      <div class="project-list">
        {#each projects.slice(0, 10) as project}
          <button
            class="project-card"
            class:active={appState.currentProjectId === project.id}
            onclick={() => {
              appState.currentProjectId = project.id;
              appState.currentView = 'prompt';
            }}
          >
            <span class="project-name">{project.name || project.id}</span>
            <span class="project-meta">{relativeTime(project.created_at)}</span>
          </button>
        {/each}
      </div>
    </div>
  {/if}

  <div class="sidebar-footer">
    <div class="status-indicator" class:online={appState.serverConnected}>
      <span class="status-dot"></span>
      {#if !appState.sidebarCollapsed}
        <span>{appState.serverConnected ? 'Stream active' : 'No stream'}</span>
      {/if}
    </div>
  </div>
</aside>
