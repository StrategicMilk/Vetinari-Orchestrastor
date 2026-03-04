# Vetinari UX Redesign - Quick Reference Guide

## 🎨 Design System at a Glance

### Color Tokens
```
Primary:       #4e9af9 (Blue)
Secondary:     #21d4fd (Teal)
Success:       #38d39f (Green)
Warning:       #f5a524 (Orange)
Danger:        #f06262 (Red)
```

### Spacing (8px Grid)
```
Small:    4px   (var(--space-1) to --space-3)
Medium:   8px   (var(--space-4))
Large:    16px  (var(--space-6) to --space-8)
XLarge:   24px+ (var(--space-10+))
```

### Typography
```
Font:       Inter (system-ui fallback)
Sizes:      12px, 14px, 16px, 18px, 20px, 24px, 30px, 36px
Weights:    400 (normal), 500 (medium), 600 (semibold), 700 (bold)
```

---

## 🏗️ Component Quick Start

### Button
```html
<!-- Primary (filled) -->
<button class="btn btn-primary">Action</button>

<!-- Secondary (outlined) -->
<button class="btn btn-secondary">Cancel</button>

<!-- Ghost (transparent) -->
<button class="btn btn-ghost">Link-like</button>

<!-- Sizes -->
<button class="btn btn-primary btn-small">Small</button>
<button class="btn btn-primary btn-large">Large</button>
```

### Input & Select
```html
<div class="input-group">
  <label class="input-label">Label</label>
  <input type="text" class="input" placeholder="Text...">
</div>

<div class="input-group">
  <label class="input-label">Dropdown</label>
  <select class="select">
    <option>Option 1</option>
  </select>
</div>

<!-- Checkbox -->
<label class="checkbox-label">
  <input type="checkbox">
  <span>Option</span>
</label>
```

### Card
```html
<div class="card">
  <div class="card-header">
    <h3>Title</h3>
    <button class="btn btn-small btn-secondary">Action</button>
  </div>
  <div class="card-body">
    Content here
  </div>
  <div class="card-footer">
    <button class="btn btn-secondary">Cancel</button>
    <button class="btn btn-primary">Save</button>
  </div>
</div>
```

### Badge
```html
<span class="badge">Default</span>
<span class="badge success">Success</span>
<span class="badge warning">Warning</span>
<span class="badge danger">Danger</span>
```

### Modal
```html
<div class="modal-backdrop">
  <div class="modal">
    <div class="modal-header">
      <h2>Title</h2>
      <button class="modal-close">&times;</button>
    </div>
    <div class="modal-body">Content</div>
    <div class="modal-footer">
      <button class="btn btn-secondary">Close</button>
      <button class="btn btn-primary">Save</button>
    </div>
  </div>
</div>
```

### Loading/Skeleton
```html
<div class="skeleton">
  <div class="skeleton-line"></div>
  <div class="skeleton-line"></div>
</div>

<div class="skeleton skeleton-avatar"></div>

<div class="loading">
  <div class="spinner"></div>
</div>
```

---

## 🎯 Layout Patterns

### Page with Header
```html
<div class="content">
  <div class="page-header">
    <div>
      <h1>Page Title</h1>
      <p>Subtitle or description</p>
    </div>
    <div class="page-actions">
      <button class="btn btn-secondary">Refresh</button>
    </div>
  </div>
  
  <!-- Content grids -->
  <div class="grid grid-2">
    <!-- Cards here -->
  </div>
</div>
```

### Responsive Grids
```html
<!-- Auto-fit grid (2 columns on desktop) -->
<div class="grid grid-2">
  <div class="card">...</div>
</div>

<!-- 3-column grid -->
<div class="grid grid-3">
  <div class="card">...</div>
</div>

<!-- 4-column grid -->
<div class="grid grid-4">
  <div class="card">...</div>
</div>
```

### Sidebar Navigation
```html
<aside class="sidebar">
  <div class="logo">
    <i class="fas fa-brain"></i>
    <span>App Name</span>
  </div>
  
  <nav class="nav-menu" role="menu">
    <a href="#" class="nav-item active" data-view="dashboard">
      <i class="fas fa-home"></i>
      <span>Home</span>
    </a>
    <a href="#" class="nav-item" data-view="settings">
      <i class="fas fa-cog"></i>
      <span>Settings</span>
    </a>
  </nav>
</aside>
```

---

## ♿ Accessibility Checklist

### HTML
- [ ] Use semantic HTML (nav, main, header, aside, section)
- [ ] All buttons & links have descriptive text
- [ ] Images have alt text (if applicable)
- [ ] Form inputs have associated labels
- [ ] Headings follow proper hierarchy (h1 → h2 → h3)

### ARIA
```html
<!-- Navigation -->
<nav role="navigation" aria-label="Main menu">

<!-- Live regions -->
<div role="alert" aria-live="polite">Status updates</div>

<!-- Buttons with icons -->
<button aria-label="Close modal">&times;</button>

<!-- Custom controls -->
<div role="tablist">
  <div role="tab" aria-selected="true">Tab 1</div>
</div>
```

### CSS Accessibility
- Focus states are always visible (outline or background change)
- Color is not the only indicator (use icons, text, underlines)
- Touch targets are ≥44px height/width on mobile
- Text contrast ≥4.5:1 (WCAG AA)
- Animations respect `prefers-reduced-motion`

### Keyboard Navigation
- `Tab` / `Shift+Tab`: Navigate forward/backward
- `Enter`: Activate button
- `Space`: Toggle checkbox
- `Arrow Keys`: Select in dropdowns
- `Escape`: Close modal/dropdown
- `Ctrl/Cmd+B`: Toggle sidebar (custom)

---

## 🎬 Animation & Motion

### Using Transitions
```css
.element {
  transition: all var(--transition-base);  /* 150ms ease-out */
}

.element:hover {
  background: var(--primary);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}
```

### Available Durations
```
--transition-fast:    100ms
--transition-base:    150ms (default)
--transition-slow:    200ms
--transition-slowest: 300ms
```

### Respecting Reduced Motion
```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

---

## 🌓 Theme Support

### Current Themes
1. **Dark** (default): `data-theme="dark"`
2. **Light**: `data-theme="light"`

### Toggling Theme
```javascript
function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme');
  const newTheme = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
}
```

---

## 🔧 Common Customizations

### Change Primary Color
```css
:root {
  --primary: #your-color;
  --primary-hover: #lighter-shade;
  --primary-active: #darker-shade;
  --primary-muted: rgba(your-color, 0.12);
}
```

### Change Spacing
```css
:root {
  --space-4: 1rem;  /* Increase from 8px to 16px */
  --space-6: 2rem;  /* Increase from 12px to 32px */
}
```

### Create New Button Variant
```css
.btn-success {
  background: var(--success);
  color: white;
}

.btn-success:hover {
  background: var(--success-hover);
  /* or derive lighter shade */
}
```

---

## 📱 Responsive Breakpoints

```
Mobile:  <768px   (single column, mobile sidebar)
Tablet:  768-1024px (reduced sidebar, 2 cols)
Desktop: ≥1024px  (full layout)
```

### Media Query Template
```css
@media (max-width: 768px) {
  /* Mobile styles */
  .sidebar {
    position: fixed;
    left: -280px;
  }
}

@media (max-width: 480px) {
  /* Extra small (phones) */
  .header {
    flex-wrap: wrap;
  }
}
```

---

## 🧪 Testing the UX

### Visual Regression
- Use Percy, Chromatic, or similar for screenshot diffing
- Test dark mode rendering
- Test responsive layouts at key breakpoints

### Accessibility
```bash
# Run Axe DevTools in Chrome
# Run Pa11y command-line checker
npm install -g pa11y-ci
pa11y-ci http://localhost:5000
```

### Performance
```bash
# Lighthouse audit
npm install -g lighthouse
lighthouse http://localhost:5000
```

---

## 📚 File Structure
```
ui/
├── static/
│   ├── css/
│   │   └── style.css         (1900+ lines, all-in-one design system)
│   └── js/
│       └── app.js            (3400+ lines, core functionality)
└── templates/
    └── index.html            (900 lines, semantic HTML)
```

---

## 🚀 Next Steps

1. **Deploy & Test**: Push changes to staging; test across devices
2. **Collect Feedback**: Gather user feedback on UX improvements
3. **Monitor Analytics**: Track user engagement with new UI
4. **Iterate**: Refine based on real-world usage patterns
5. **Document**: Keep this guide updated as components evolve

---

## 💡 Tips & Tricks

- **Search**: `Ctrl/Cmd+F` in DevTools to find CSS variables
- **Debug Colors**: Use `color: var(--primary)` to verify token usage
- **Test Contrast**: Use WebAIM Contrast Checker tool
- **Mobile Testing**: Use Chrome DevTools device emulation
- **Performance**: Check CSS is minified in production

---

**Last Updated**: 2026-03-03
**Maintainers**: Design & Frontend Team
