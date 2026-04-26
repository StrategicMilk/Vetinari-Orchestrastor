/**
 * Programmatic access to CSS design tokens.
 *
 * Reads computed CSS custom property values from :root so Svelte
 * components and Chart.js instances can use the same palette.
 */

/** Cache of resolved token values, invalidated on theme change. */
let _cache = null;

/** Read a CSS custom property from :root. */
function getToken(name) {
  return getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
}

/** Invalidate the token cache (call on theme switch). */
export function invalidateTokenCache() {
  _cache = null;
}

/**
 * Get all design tokens as an object.
 *
 * Values are lazily cached until invalidateTokenCache() is called.
 * @returns {Record<string, string>}
 */
export function getTokens() {
  if (_cache) return _cache;

  _cache = {
    // Backgrounds
    baseBg: getToken('--base-bg'),
    surfaceBg: getToken('--surface-bg'),
    surfaceElevated: getToken('--surface-elevated'),

    // Text
    textPrimary: getToken('--text-primary'),
    textSecondary: getToken('--text-secondary'),
    textMuted: getToken('--text-muted'),

    // Borders
    borderDefault: getToken('--border-default'),
    borderSubtle: getToken('--border-subtle'),

    // Brand colors
    primary: getToken('--primary'),
    secondary: getToken('--secondary'),
    success: getToken('--success'),
    warning: getToken('--warning'),
    danger: getToken('--danger'),

    // Chart-specific
    chartTick: getToken('--text-muted') || '#9ca3af',
    chartGridX: getToken('--border-subtle') || 'rgba(255,255,255,0.04)',
    chartGridY: getToken('--border-default') || 'rgba(255,255,255,0.06)',
  };

  return _cache;
}

/**
 * Get Chart.js-compatible default options using design tokens.
 *
 * @param {'line'|'bar'|'doughnut'} [type='line']
 * @returns {object} Chart.js configuration defaults.
 */
export function getChartDefaults(type = 'line') {
  const t = getTokens();
  return {
    type,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: { display: false },
      },
      scales: type === 'doughnut' ? {} : {
        x: {
          ticks: { color: t.chartTick, maxTicksLimit: 8 },
          grid: { color: t.chartGridX },
        },
        y: {
          ticks: { color: t.chartTick },
          grid: { color: t.chartGridY },
        },
      },
    },
  };
}
