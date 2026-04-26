<script>
  import { appState } from '$lib/stores/app.svelte.js';
  import Sidebar from '$components/shell/Sidebar.svelte';
  import Header from '$components/shell/Header.svelte';
  import CommandPalette from '$components/shell/CommandPalette.svelte';
  import Toast from '$components/shell/Toast.svelte';
  import Dashboard from '$views/Dashboard.svelte';
  import ModelsView from '$views/ModelsView.svelte';
  import ChatView from '$views/ChatView.svelte';
  import TrainingView from '$views/TrainingView.svelte';
  import MemoryView from '$views/MemoryView.svelte';
  import SettingsView from '$views/SettingsView.svelte';
  import ProjectsView from '$views/ProjectsView.svelte';
  import AgentsView from '$views/AgentsView.svelte';
  import OutputView from '$views/OutputView.svelte';
  import TasksView from '$views/TasksView.svelte';
  import PlanBuilderView from '$views/PlanBuilderView.svelte';

  /** Valid view names for routing. */
  const VALID_VIEWS = new Set([
    'dashboard', 'prompt', 'models', 'training', 'memory', 'settings',
    'workflow', 'agents', 'output', 'tasks', 'decomposition',
  ]);

  /** Handle keyboard shortcuts at the app level. */
  function handleKeydown(e) {
    // Ctrl+B: toggle sidebar
    if (e.ctrlKey && e.key === 'b') {
      e.preventDefault();
      appState.sidebarCollapsed = !appState.sidebarCollapsed;
    }
    // Ctrl+K: toggle command palette
    if (e.ctrlKey && e.key === 'k') {
      e.preventDefault();
      appState.commandPaletteOpen = !appState.commandPaletteOpen;
    }
    // Escape: close overlays
    if (e.key === 'Escape') {
      appState.commandPaletteOpen = false;
    }
  }

  /** Sync hash-based routing on load and popstate. */
  function syncRouteFromHash() {
    const hash = window.location.hash.slice(1);
    if (hash && VALID_VIEWS.has(hash)) {
      appState.currentView = hash;
    }
  }

  $effect(() => {
    syncRouteFromHash();
    const onPopState = () => syncRouteFromHash();
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  });

  // Push hash when view changes
  $effect(() => {
    const view = appState.currentView;
    if (window.location.hash !== `#${view}`) {
      window.history.replaceState(null, '', `#${view}`);
    }
  });

  // Apply theme class to document
  $effect(() => {
    document.documentElement.setAttribute('data-theme', appState.theme);
  });
</script>

<svelte:window onkeydown={handleKeydown} />

<a href="#main-content" class="skip-link skip-to-content">Skip to main content</a>

<div
  class="app-container"
  class:sidebar-collapsed={appState.sidebarCollapsed}
  aria-label="Vetinari - Local LLM Orchestration System"
>
  <Sidebar />

  <main class="main-content" class:sidebar-collapsed={appState.sidebarCollapsed}>
    <Header />

    <div id="main-content" class="view-container">
      {#if appState.currentView === 'dashboard'}
        <Dashboard />
      {:else if appState.currentView === 'prompt'}
        <ChatView />
      {:else if appState.currentView === 'models'}
        <ModelsView />
      {:else if appState.currentView === 'training'}
        <TrainingView />
      {:else if appState.currentView === 'memory'}
        <MemoryView />
      {:else if appState.currentView === 'settings'}
        <SettingsView />
      {:else if appState.currentView === 'workflow'}
        <ProjectsView />
      {:else if appState.currentView === 'agents'}
        <AgentsView />
      {:else if appState.currentView === 'output'}
        <OutputView />
      {:else if appState.currentView === 'tasks'}
        <TasksView />
      {:else if appState.currentView === 'decomposition'}
        <PlanBuilderView />
      {:else}
        <ChatView />
      {/if}
    </div>
  </main>
</div>

{#if appState.commandPaletteOpen}
  <CommandPalette />
{/if}

<Toast />
