/**
 * Locale-aware formatting utilities.
 *
 * Replaces bare .toFixed() and string concatenation with Intl-based
 * formatters that respect the user's locale and configured currency.
 */

/** User-configurable currency symbol. */
let currencySymbol = '$';

/** Set the currency symbol used by format.currency(). */
export function setCurrencySymbol(sym) {
  currencySymbol = sym;
}

/** Locale for Intl formatters. undefined = browser default. */
let locale;

/** Override the locale for all formatters. */
export function setLocale(loc) {
  locale = loc;
}

/**
 * Format a number with fixed decimal places.
 * @param {number|null|undefined} value
 * @param {number} [decimals=1]
 * @returns {string} Formatted number or em-dash for null/NaN.
 */
export function decimal(value, decimals = 1) {
  if (value == null || Number.isNaN(+value)) return '\u2014';
  return new Intl.NumberFormat(locale, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(+value);
}

/**
 * Format a whole number with thousands separator.
 * @param {number|null|undefined} value
 * @returns {string}
 */
export function integer(value) {
  if (value == null || Number.isNaN(+value)) return '\u2014';
  return new Intl.NumberFormat(locale, {
    maximumFractionDigits: 0,
  }).format(+value);
}

/**
 * Format as percentage (0\u2013100 scale).
 * @param {number|null|undefined} value
 * @param {number} [decimals=1]
 * @returns {string}
 */
export function percent(value, decimals = 1) {
  if (value == null || Number.isNaN(+value)) return '\u2014';
  return new Intl.NumberFormat(locale, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(+value) + '%';
}

/**
 * Format as currency with the configured symbol.
 * @param {number|null|undefined} value
 * @param {number} [decimals=2]
 * @returns {string}
 */
export function currency(value, decimals = 2) {
  if (value == null || Number.isNaN(+value)) return '\u2014';
  return currencySymbol + new Intl.NumberFormat(locale, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(+value);
}

/**
 * Format bytes as a human-readable size (KB, MB, GB).
 * @param {number|null|undefined} bytes
 * @returns {string}
 */
export function fileSize(bytes) {
  if (bytes == null || Number.isNaN(+bytes)) return '\u2014';
  const b = +bytes;
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${decimal(b / 1024, 1)} KB`;
  if (b < 1024 * 1024 * 1024) return `${decimal(b / (1024 * 1024), 1)} MB`;
  return `${decimal(b / (1024 * 1024 * 1024), 2)} GB`;
}

/**
 * Format milliseconds as a human-readable duration.
 * @param {number|null|undefined} ms
 * @returns {string}
 */
export function duration(ms) {
  if (ms == null || Number.isNaN(+ms)) return '\u2014';
  const v = +ms;
  if (v < 1000) return `${Math.round(v)} ms`;
  if (v < 60_000) return `${decimal(v / 1000, 1)} s`;
  if (v < 3_600_000) return `${decimal(v / 60_000, 1)} min`;
  return `${decimal(v / 3_600_000, 1)} h`;
}

/**
 * Format milliseconds as "X ms" for latency displays.
 * @param {number|null|undefined} ms
 * @returns {string}
 */
export function latency(ms) {
  if (ms == null || Number.isNaN(+ms)) return '\u2014';
  return `${decimal(+ms, 1)} ms`;
}

/**
 * Format an ISO timestamp or Date as locale time string.
 *
 * Returns "—" for null/undefined/empty values and for strings that parse to
 * an Invalid Date (e.g. "N/A", "bad-value"). toLocaleTimeString() on an
 * Invalid Date returns the string "Invalid Date" — we guard this explicitly.
 *
 * @param {string|Date|null|undefined} value
 * @returns {string}
 */
export function time(value) {
  if (!value) return '\u2014';
  try {
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return '\u2014';
    return d.toLocaleTimeString(locale);
  } catch {
    return '\u2014';
  }
}

/**
 * Format an ISO timestamp or Date as locale date+time string.
 *
 * Returns "—" for null/undefined/empty values and for strings that parse to
 * an Invalid Date (e.g. "N/A", "bad-value").
 *
 * @param {string|Date|null|undefined} value
 * @returns {string}
 */
export function datetime(value) {
  if (!value) return '\u2014';
  try {
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return '\u2014';
    return d.toLocaleString(locale);
  } catch {
    return '\u2014';
  }
}

/**
 * Format a relative time (e.g. "3 minutes ago").
 *
 * Returns "—" for null/undefined/empty and for non-parseable strings (e.g.
 * "N/A", "--") that produce an Invalid Date. This prevents "NaN d ago" from
 * ever reaching the DOM.
 *
 * @param {string|Date|null|undefined} value
 * @returns {string}
 */
export function relativeTime(value) {
  if (!value) return '\u2014';
  const then = new Date(value).getTime();
  // new Date("N/A").getTime() === NaN — treat as missing
  if (Number.isNaN(then)) return '\u2014';
  const now = Date.now();
  const diff = now - then;

  if (diff < 60_000) return 'just now';
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)} min ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)} h ago`;
  return `${Math.floor(diff / 86_400_000)} d ago`;
}
