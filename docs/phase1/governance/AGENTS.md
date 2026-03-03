# Vetinari Agents

## Overview

This document describes the specialized agents in Vetinari and their responsibilities. This file is auto-generated - see `/docs/governance/` for the governance process.

## Agent Types

### Explorer

- **Purpose**: Codebase search and file discovery
- **When to use**: Finding code, exploring project structure, locating functions/classes
- **Keywords**: `find`, `search`, `where`, `locate`, `explore codebase`
- **Skills**: grep, file discovery, pattern matching, symbol lookup, import analysis
- **Thinking Modes**: low (quick grep), medium (comprehensive), high (full mapping)

### Librarian

- **Purpose**: Documentation research and example finding
- **When to use**: Learning libraries, API usage, best practices
- **Keywords**: `how does`, `library`, `docs`, `example`, `documentation`
- **Skills**: docs lookup, GitHub examples, API reference, package info
- **Thinking Modes**: low (quick lookup), medium (detailed), high (comprehensive)

### Oracle

- **Purpose**: Strategic decisions and architecture guidance
- **When to use**: Design decisions, trade-offs, debugging complex issues
- **Keywords**: `should I`, `better`, `architecture`, `design decision`, `trade-off`
- **Skills**: architecture analysis, trade-off evaluation, debugging strategy
- **Thinking Modes**: low (quick), medium (compare), high (deep analysis)

### UI Planner

- **Purpose**: Frontend design and visual polish
- **When to use**: UI/UX improvements, CSS, responsive layouts
- **Keywords**: `design`, `css`, `ui`, `style`, `visual`, `animation`
- **Skills**: CSS design, responsive layout, animation, accessibility
- **Thinking Modes**: low (quick fix), medium (component), high (full design)

### Builder

- **Purpose**: Code implementation and refactoring
- **When to use**: Creating features, implementing logic, writing tests
- **Keywords**: `create`, `implement`, `write`, `build`, `add feature`
- **Skills**: feature implementation, refactoring, test writing, error handling
- **Thinking Modes**: low (quick), medium (full feature), high (production-ready)

### Researcher

- **Purpose**: Comprehensive investigation and fact-finding
- **When to use**: Deep research, source verification, complex analysis
- **Keywords**: `research`, `investigate`, `analyze`, `deep dive`, `compare`
- **Skills**: deep dive, source verification, comparative analysis, fact-finding
- **Thinking Modes**: low (quick facts), medium (detailed), high (comprehensive)

### Evaluator

- **Purpose**: Code review and quality assessment
- **When to use**: Reviews, audits, testing strategies, validation
- **Keywords**: `review`, `check`, `validate`, `test`, `audit`, `assess`
- **Skills**: code review, quality assessment, security audit, test strategy
- **Thinking Modes**: low (quick), medium (detailed), high (full audit)

### Synthesizer

- **Purpose**: Result consolidation and reporting
- **When to use**: Summarizing findings, combining results, final reports
- **Keywords**: `combine`, `summarize`, `merge`, `synthesize`, `final report`
- **Skills**: result combination, summarization, report generation, consolidation
- **Thinking Modes**: low (summary), medium (structured), high (comprehensive)

## Agent Collaboration

Agents work together through the shared memory system:

1. **Explorer** finds relevant code → stores in memory
2. **Librarian** researches context → stores in memory
3. **Oracle** makes decisions → stores in memory with reasoning
4. **Builder** implements → stores results in memory
5. **Evaluator** reviews → stores findings in memory
6. **Synthesizer** combines all → creates final output

## Auto-Update

This file is auto-generated. Manual changes may be overwritten.
To update: Edit `/docs/governance/templates/` and run the governance update script.

---

*Last Updated: 2026-03-02*
*Version: 1.0*
