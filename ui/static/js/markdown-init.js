/**
 * Markdown, syntax highlighting, and mermaid initialization.
 * Loaded after marked.js, highlight.js, and mermaid CDN scripts.
 */

'use strict';

// ── Diff renderer helpers ──────────────────────────────────────────────────

/**
 * Escape HTML special characters to prevent XSS in diff output.
 *
 * @param {string} str - Raw string to escape.
 * @returns {string} HTML-safe string.
 */
function _escapeDiffHtml(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

/**
 * Classify a single diff line into a type token.
 *
 * @param {string} line - Raw line text.
 * @returns {string} One of: 'added', 'removed', 'hunk', 'file-header', 'context'.
 */
function _diffLineType(line) {
    if (line.startsWith('@@')) return 'hunk';
    if (line.startsWith('+++') || line.startsWith('---')) return 'file-header';
    if (line.startsWith('+')) return 'added';
    if (line.startsWith('-')) return 'removed';
    return 'context';
}

/**
 * Render a sequence of collapsed context lines as an expander + hidden block.
 *
 * @param {Array<{num: number, html: string}>} lines - Context lines to collapse.
 * @returns {string} HTML string for the expander widget.
 */
function _renderCollapsed(lines) {
    var count = lines.length;
    var inner = lines.map(function(l) { return l.html; }).join('');
    return (
        '<div class="diff-expander" onclick="this.style.display=\'none\';this.nextElementSibling.style.display=\'block\';">' +
        '... ' + count + ' unchanged line' + (count === 1 ? '' : 's') + ' ...' +
        '</div>' +
        '<div class="diff-collapsed" style="display:none;">' + inner + '</div>'
    );
}

/**
 * Render a unified diff string as a styled HTML diff view.
 *
 * Lines are classified by their leading character, consecutive context runs
 * longer than 5 lines are collapsed into a clickable expander, and line
 * numbers are shown on the left.
 *
 * @param {string} content - Raw diff text (content of the fenced code block).
 * @returns {string} HTML string for the complete diff view.
 */
function _renderDiffBlock(content) {
    var COLLAPSE_THRESHOLD = 5;
    var lines = content.split('\n');
    // Remove a trailing empty line produced by the final newline in the fence.
    if (lines.length > 0 && lines[lines.length - 1] === '') lines.pop();

    // Build classified line objects with rendered HTML.
    var classified = lines.map(function(line, idx) {
        var type = _diffLineType(line);
        var prefix = line.charAt(0) || ' ';
        var text = line.length > 0 ? line.slice(1) : '';
        var lineNum = idx + 1;
        var modifierClass = {
            'added': 'diff-line--added',
            'removed': 'diff-line--removed',
            'hunk': 'diff-line--hunk',
            'file-header': 'diff-line--file-header',
            'context': 'diff-line--context'
        }[type];
        var html = (
            '<div class="diff-line ' + modifierClass + '">' +
            '<span class="diff-linenum">' + lineNum + '</span>' +
            '<span class="diff-prefix">' + _escapeDiffHtml(prefix) + '</span>' +
            '<span class="diff-text">' + _escapeDiffHtml(text) + '</span>' +
            '</div>'
        );
        return { type: type, html: html };
    });

    // Walk the classified lines, collapsing long context runs.
    var output = '';
    var contextBuffer = [];

    function flushContext() {
        if (contextBuffer.length === 0) return;
        if (contextBuffer.length > COLLAPSE_THRESHOLD) {
            output += _renderCollapsed(contextBuffer);
        } else {
            contextBuffer.forEach(function(l) { output += l.html; });
        }
        contextBuffer = [];
    }

    classified.forEach(function(item) {
        if (item.type === 'context') {
            contextBuffer.push(item);
        } else {
            flushContext();
            output += item.html;
        }
    });
    flushContext();

    return '<div class="diff-view">' + output + '</div>';
}

// ── marked configuration ───────────────────────────────────────────────────

// Configure marked with highlight.js and a custom renderer for diff fences.
if (typeof marked !== 'undefined') {
    var _diffRenderer = new marked.Renderer();

    /**
     * Override the code renderer to handle `diff` language fences specially.
     *
     * For all other languages the default highlight.js path is used.
     *
     * @param {string} code - The fenced code block content.
     * @param {string} lang - The language identifier from the fence.
     * @returns {string} Rendered HTML string.
     */
    _diffRenderer.code = function(code, lang) {
        if (lang === 'diff') {
            return _renderDiffBlock(code);
        }
        // Fall back to highlight.js for all other languages.
        var highlighted = code;
        if (typeof hljs !== 'undefined') {
            if (lang && hljs.getLanguage(lang)) {
                try { highlighted = hljs.highlight(code, { language: lang }).value; } catch(e) { /* fallback */ }
            } else {
                highlighted = hljs.highlightAuto(code).value;
            }
        }
        var langClass = lang ? ' class="language-' + lang + '"' : '';
        return '<pre><code' + langClass + '>' + highlighted + '</code></pre>';
    };

    marked.setOptions({
        renderer: _diffRenderer,
        breaks: true,
        gfm: true
    });
}

if (typeof mermaid !== 'undefined') {
    mermaid.initialize({ startOnLoad: false, theme: 'dark' });
}

// Global markdown renderer with copy buttons (using event delegation, no inline onclick)
function renderMarkdown(text) {
    if (typeof marked === 'undefined') return escapeHtml(text).replace(/\n/g, '<br>');
    var html = marked.parse(text || '');
    // Add copy buttons to code blocks — uses class-based delegation instead of onclick
    html = html.replace(/<pre><code([^>]*)>/g, function(match, attrs) {
        return '<div class="code-block-wrapper"><button class="code-copy-btn" title="Copy code"><i class="fas fa-copy"></i></button><pre><code' + attrs + '>';
    });
    html = html.replace(/<\/code><\/pre>/g, '</code></pre></div>');
    return html;
}

// Event delegation for dynamically-created copy buttons
document.addEventListener('click', function(e) {
    var btn = e.target.closest('.code-copy-btn');
    if (!btn) return;
    var code = btn.parentElement.querySelector('code');
    if (code) {
        navigator.clipboard.writeText(code.innerText).then(function() {
            btn.innerHTML = '<i class="fas fa-check"></i>';
            btn.classList.add('copy-success');
            setTimeout(function() {
                btn.innerHTML = '<i class="fas fa-copy"></i>';
                btn.classList.remove('copy-success');
            }, 2000);
        });
    }
});

window.renderMarkdown = renderMarkdown;
