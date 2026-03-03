# Explorer Search Patterns

## Quick Reference

### Common Search Patterns

| Pattern | Description | Example |
|---------|-------------|---------|
| `function:name` | Find function definitions | `function:handleLogin` |
| `class:name` | Find class declarations | `class:UserModel` |
| `import:"module"` | Find specific imports | `import:"react"` |
| `regex:pattern` | Regex search | `regex:auth.*?` |

### File Extensions by Language

| Language | Extensions |
|----------|------------|
| JavaScript | .js, .jsx, .mjs |
| TypeScript | .ts, .tsx |
| Python | .py |
| Rust | .rs |
| Go | .go |
| Java | .java |
| C/C++ | .c, .cpp, .h, .hpp |
| Ruby | .rb |
| PHP | .php |
| Swift | .swift |
| Kotlin | .kt |

### Ignore Patterns

Always respect these directories:
- `node_modules/`
- `.git/`
- `__pycache__/`
- `venv/`
- `.venv/`
- `dist/`
- `build/`
- `target/`
- `.next/`
- `.nuxt/`

## Advanced Techniques

### 1. Contextual Search

```bash
# Search with 3 lines before and after
grep -C 3 "pattern"

# Search in specific file types
grep --include="*.ts" "pattern"

# Exclude directories
grep --exclude-dir=node_modules "pattern"
```

### 2. Multi-Pass Search Strategy

**Pass 1**: Broad search with common keywords
**Pass 2**: Narrow by file type
**Pass 3**: Deep dive into specific files

### 3. Project Type Detection

Look for these indicators:
- `package.json` → Node.js project
- `requirements.txt` → Python project
- `Cargo.toml` → Rust project
- `go.mod` → Go project
- `pom.xml` → Java/Maven project

### 4. Import Tracing

For file `src/utils/auth.ts`:
1. Find what it imports
2. Find what imports it
3. Trace the dependency chain
4. Identify entry points

## Ranking Results

When results are numerous, rank by:

1. **Exact matches** - Function/class definitions
2. **Usage frequency** - Most referenced
3. **Recency** - Recently modified
4. **Context relevance** - Matches search intent

## Performance Optimization

- Use `ripgrep` (rg) for speed
- Cache results for session
- Limit initial results to top 20
- Paginate for large result sets

## Common Queries

| User Intent | Search Strategy |
|-------------|-----------------|
| "Where is X defined?" | `function:X` or exact match |
| "How is X used?" | References search |
| "What files use X?" | Import/usage search |
| "Show project structure" | Tree/map generation |
| "Find all errors" | `error`, `Exception`, `throw` |
| "Find security issues" | `password`, `secret`, `token`, `api_key` |
