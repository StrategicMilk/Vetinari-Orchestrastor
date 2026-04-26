<script>
  import { appState } from '$lib/stores/app.svelte.js';
  import { integer } from '$lib/utils/format.js';

  function toggleSidebar() {
    appState.sidebarCollapsed = !appState.sidebarCollapsed;
  }

  function toggleTheme() {
    appState.theme = appState.theme === 'dark' ? 'light' : 'dark';
  }

  function openCommandPalette() {
    appState.commandPaletteOpen = true;
  }

  let searchQuery = $state('');
</script>

<header class="header">
  <div class="header-left">
    <button
      class="btn btn-ghost"
      id="sidebarToggle"
      onclick={toggleSidebar}
      aria-expanded={!appState.sidebarCollapsed}
      aria-label="Toggle sidebar"
      title="Toggle sidebar (Ctrl+B)"
    >
      <i class="fas fa-bars"></i>
    </button>
  </div>

  <div class="header-center">
    <div class="search-container">
      <i class="fas fa-search search-icon"></i>
      <input
        type="text"
        class="input search-input"
        placeholder="Search... (Ctrl+K)"
        bind:value={searchQuery}
        onfocus={openCommandPalette}
        aria-label="Global search"
      />
    </div>
  </div>

  <div class="header-right">
    <span class="token-counter" title="Session tokens used">
      <i class="fas fa-coins"></i>
      {integer(appState.sessionTokens)}
    </span>

    <button
      class="btn btn-ghost"
      onclick={toggleTheme}
      title="Toggle theme"
      aria-label="Toggle light/dark theme"
    >
      <i class="fas" class:fa-moon={appState.theme === 'dark'} class:fa-sun={appState.theme === 'light'}></i>
    </button>

    <button
      class="btn btn-ghost"
      onclick={() => { appState.currentView = 'models'; }}
      title="Discover models"
      aria-label="Discover models"
    >
      <i class="fas fa-compass"></i>
    </button>
  </div>
</header>
