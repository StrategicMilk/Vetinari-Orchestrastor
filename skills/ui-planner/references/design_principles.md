# Design Principles

## Color

### Color Palette
- Primary: Brand color (used for main actions)
- Secondary: Supporting color (used for secondary elements)
- Neutral: Grays (text, backgrounds, borders)
- Semantic: Success, warning, error, info

### Dark Mode
- Background: #121212 or #1a1a1a
- Surface: #1e1e1e or #252525
- Text: #ffffff (primary), #a0a0a0 (secondary)

### Contrast Ratios
- AA (normal text): 4.5:1
- AA (large text): 3:1
- AAA: 7:1

## Typography

### Font Scale
```
--text-xs: 0.75rem (12px)
--text-sm: 0.875rem (14px)
--text-base: 1rem (16px)
--text-lg: 1.125rem (18px)
--text-xl: 1.25rem (20px)
--text-2xl: 1.5rem (24px)
--text-3xl: 1.875rem (30px)
```

### Line Height
- Tight: 1.25 (headings)
- Normal: 1.5 (body)
- Relaxed: 1.75 (longform)

## Spacing

### Spacing Scale
```
--space-1: 4px
--space-2: 8px
--space-3: 12px
--space-4: 16px
--space-6: 24px
--space-8: 32px
--space-12: 48px
--space-16: 64px
```

## Shadows

### Elevation
```css
--shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
--shadow-md: 0 4px 6px rgba(0,0,0,0.1);
--shadow-lg: 0 10px 15px rgba(0,0,0,0.1);
--shadow-xl: 0 20px 25px rgba(0,0,0,0.15);
```

## Border Radius

### Scale
```
--radius-sm: 4px
--radius-md: 8px
--radius-lg: 12px
--radius-xl: 16px
--radius-full: 9999px
```

## Animation

### Duration Scale
- Instant: 0ms
- Fast: 150ms
- Normal: 200ms
- Slow: 300ms
- Slower: 500ms

### Easing
- ease-out: `cubic-bezier(0, 0, 0.2, 1)`
- ease-in: `cubic-bezier(0.4, 0, 1, 1)`
- ease-in-out: `cubic-bezier(0.4, 0, 0.2, 1)`

## Responsive Breakpoints

```css
--breakpoint-sm: 640px;
--breakpoint-md: 768px;
--breakpoint-lg: 1024px;
--breakpoint-xl: 1280px;
--breakpoint-2xl: 1536px;
```

## Accessibility Checklist

- [ ] Color contrast passes WCAG AA
- [ ] Focus states visible
- [ ] Keyboard navigation works
- [ ] ARIA labels where needed
- [ ] Reduced motion respected
- [ ] Screen reader tested
