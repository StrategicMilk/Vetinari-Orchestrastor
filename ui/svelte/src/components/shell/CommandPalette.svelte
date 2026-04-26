<script>
  import { appState } from '$lib/stores/app.svelte.js';

  /** Command definitions for the palette. */
  const COMMANDS = [
    { id: 'chat', label: 'Go to Chat', icon: 'fas fa-comments', action: () => nav('prompt') },
    { id: 'models', label: 'Go to Models', icon: 'fas fa-microchip', action: () => nav('models') },
    { id: 'training', label: 'Go to Training', icon: 'fas fa-graduation-cap', action: () => nav('training') },
    { id: 'memory', label: 'Go to Memory', icon: 'fas fa-brain', action: () => nav('memory') },
    { id: 'settings', label: 'Go to Settings', icon: 'fas fa-cog', action: () => nav('settings') },
    { id: 'dashboard', label: 'Go to Dashboard', icon: 'fas fa-chart-line', action: () => nav('dashboard') },
    { id: 'projects', label: 'Go to Projects', icon: 'fas fa-sitemap', action: () => nav('workflow') },
    { id: 'agents', label: 'Go to Agents', icon: 'fas fa-robot', action: () => nav('agents') },
    { id: 'tasks', label: 'Go to Task Queue', icon: 'fas fa-tasks', action: () => nav('tasks') },
    { id: 'plan', label: 'Go to Plan Builder', icon: 'fas fa-project-diagram', action: () => nav('decomposition') },
    { id: 'theme', label: 'Toggle Theme', icon: 'fas fa-palette', action: () => {
      appState.theme = appState.theme === 'dark' ? 'light' : 'dark';
      close();
    }},
  ];

  let query = $state('');
  let selectedIndex = $state(-1);
  let inputEl;

  let filtered = $derived(
    query.length === 0
      ? COMMANDS
      : COMMANDS.filter((c) => c.label.toLowerCase().includes(query.toLowerCase()))
  );

  // Reset selection whenever the filtered set changes so a stale index never
  // points into an empty array (which would produce NaN via modulo).
  $effect(() => {
    // Access filtered.length to register the dependency.
    void filtered.length;
    selectedIndex = -1;
  });

  function nav(view) {
    appState.currentView = view;
    close();
  }

  function close() {
    appState.commandPaletteOpen = false;
    query = '';
    selectedIndex = -1;
  }

  function handleKeydown(e) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      // Bail when there are no results — modulo zero produces NaN.
      if (filtered.length === 0) return;
      selectedIndex = selectedIndex < 0 ? 0 : (selectedIndex + 1) % filtered.length;
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (filtered.length === 0) return;
      selectedIndex = selectedIndex <= 0 ? filtered.length - 1 : selectedIndex - 1;
    } else if (e.key === 'Enter' && selectedIndex >= 0 && filtered[selectedIndex]) {
      e.preventDefault();
      filtered[selectedIndex].action();
    } else if (e.key === 'Escape') {
      close();
    }
  }

  $effect(() => {
    // Focus the input when palette opens
    if (inputEl) inputEl.focus();
  });
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="command-palette-overlay" onclick={close} onkeydown={handleKeydown}>
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div class="command-palette" onclick={(e) => e.stopPropagation()}>
    <div class="command-palette-input-wrap">
      <i class="fas fa-search"></i>
      <input
        bind:this={inputEl}
        bind:value={query}
        type="text"
        class="command-palette-input"
        placeholder="Type a command..."
        aria-label="Command palette search"
        onkeydown={handleKeydown}
      />
    </div>

    <div class="command-palette-results" role="listbox">
      {#each filtered as cmd, i}
        <button
          class="command-palette-item"
          class:selected={i === selectedIndex}
          role="option"
          aria-selected={i === selectedIndex}
          onclick={cmd.action}
          onmouseenter={() => { selectedIndex = i; }}
        >
          <i class={cmd.icon}></i>
          <span>{cmd.label}</span>
        </button>
      {/each}

      {#if filtered.length === 0}
        <div class="command-palette-empty">No matching commands</div>
      {/if}
    </div>
  </div>
</div>

<style>
  .command-palette-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding-top: 15vh;
    z-index: 1000;
    backdrop-filter: blur(4px);
  }

  .command-palette {
    background: var(--surface-elevated, #1a202d);
    border: 1px solid var(--border-default);
    border-radius: 12px;
    width: 560px;
    max-width: 90vw;
    overflow: hidden;
    box-shadow: 0 24px 48px rgba(0, 0, 0, 0.3);
  }

  .command-palette-input-wrap {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-default);
  }

  .command-palette-input-wrap i {
    color: var(--text-muted);
    font-size: 14px;
  }

  .command-palette-input {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    color: var(--text-primary);
    font-size: 16px;
    font-family: inherit;
  }

  .command-palette-results {
    max-height: 400px;
    overflow-y: auto;
    padding: 8px;
  }

  .command-palette-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    border: none;
    background: transparent;
    color: var(--text-primary);
    font-size: 14px;
    font-family: inherit;
    cursor: pointer;
    border-radius: 8px;
    width: 100%;
    text-align: left;
    transition: background 100ms;
  }

  .command-palette-item.selected,
  .command-palette-item:hover {
    background: var(--glass-bg, rgba(255, 255, 255, 0.05));
  }

  .command-palette-item i {
    color: var(--text-muted);
    width: 20px;
    text-align: center;
  }

  .command-palette-empty {
    padding: 20px;
    text-align: center;
    color: var(--text-muted);
    font-size: 14px;
  }
</style>
