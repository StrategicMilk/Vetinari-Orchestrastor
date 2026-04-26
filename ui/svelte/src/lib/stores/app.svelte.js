/**
 * Central application state using Svelte 5 runes.
 *
 * Single reactive store for global UI state. Persists select keys to
 * localStorage so preferences survive page reloads.
 */

const STORAGE_KEYS = {
  theme: 'theme',
  sidebarCollapsed: 'sidebarCollapsed',
  setupComplete: 'setupComplete',
  focusMode: 'focusModeEnabled',
};

/** Read a boolean or string from localStorage with a default. */
function loadStored(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return fallback;
    if (raw === 'true') return true;
    if (raw === 'false') return false;
    return raw;
  } catch {
    return fallback;
  }
}

/** Persist a value to localStorage. */
function saveStored(key, value) {
  try {
    localStorage.setItem(key, String(value));
  } catch {
    // Storage full or unavailable — degrade silently
  }
}

// -- Reactive state ----------------------------------------------------------

let _currentView = $state(loadStored('currentView', 'prompt'));
let _currentProjectId = $state(null);
let _sidebarCollapsed = $state(loadStored(STORAGE_KEYS.sidebarCollapsed, false));
let _theme = $state(loadStored(STORAGE_KEYS.theme, 'dark'));
let _commandPaletteOpen = $state(false);
let _setupComplete = $state(loadStored(STORAGE_KEYS.setupComplete, false));
let _focusMode = $state(loadStored(STORAGE_KEYS.focusMode, false));
// Starts false — shell shows Disconnected until SSE handshakes and sets this true.
let _serverConnected = $state(false);
let _sessionTokens = $state(0);

/**
 * Reactive application state object.
 *
 * Access and mutate properties directly — Svelte 5 runes handle reactivity.
 * Persisted keys auto-sync to localStorage on write.
 */
export const appState = {
  get currentView() { return _currentView; },
  set currentView(v) {
    _currentView = v;
    saveStored('currentView', v);
  },

  get currentProjectId() { return _currentProjectId; },
  set currentProjectId(v) { _currentProjectId = v; },

  get sidebarCollapsed() { return _sidebarCollapsed; },
  set sidebarCollapsed(v) {
    _sidebarCollapsed = v;
    saveStored(STORAGE_KEYS.sidebarCollapsed, v);
  },

  get theme() { return _theme; },
  set theme(v) {
    _theme = v;
    saveStored(STORAGE_KEYS.theme, v);
  },

  get commandPaletteOpen() { return _commandPaletteOpen; },
  set commandPaletteOpen(v) { _commandPaletteOpen = v; },

  get setupComplete() { return _setupComplete; },
  set setupComplete(v) {
    _setupComplete = v;
    saveStored(STORAGE_KEYS.setupComplete, v);
  },

  get focusMode() { return _focusMode; },
  set focusMode(v) {
    _focusMode = v;
    saveStored(STORAGE_KEYS.focusMode, v);
  },

  get serverConnected() { return _serverConnected; },
  set serverConnected(v) { _serverConnected = v; },

  get sessionTokens() { return _sessionTokens; },
  set sessionTokens(v) { _sessionTokens = v; },
};
