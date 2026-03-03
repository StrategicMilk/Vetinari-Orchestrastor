# Quality Criteria

## Code Quality Metrics

### Complexity
| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| Cyclomatic | < 10 | 10-20 | > 20 |
| Lines per function | < 20 | 20-50 | > 50 |
| Nesting depth | < 3 | 3-5 | > 5 |

### Coupling
| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| Fan-in | High | Low | - |
| Fan-out | Low | Medium | High |
| Afferent coupling | Low | Medium | High |

### Other Metrics
- **Lines of code per file**: < 400
- **Comments**: 10-30% 
- **Test coverage**: > 80%

## Security Checklist

### Input Validation
- [ ] All inputs validated
- [ ] Sanitized for SQL
- [ ] Sanitized for XSS
- [ ] Type checking
- [ ] Range checking

### Authentication
- [ ] Strong passwords enforced
- [ ] Session management secure
- [ ] Tokens expire
- [ ] MFA available

### Authorization
- [ ] Role-based access
- [ ] Least privilege
- [ ] Resource ownership

### Data
- [ ] Encryption at rest
- [ ] Encryption in transit
- [ ] Secrets in env vars

## Performance Criteria

### Response Times
- API: < 200ms (p95)
- Page load: < 3s
- Database: < 100ms (p95)

### Resource Usage
- CPU: < 80% average
- Memory: < 90% limit
- Disk: < 85% capacity

## Test Coverage Goals

### By Type
- Unit tests: 80%+
- Integration: 60%+
- E2E: Key flows covered

### By Priority
- Critical paths: 100%
- Important features: 80%
- Edge cases: 60%

## Code Style

### Naming
- Variables: camelCase
- Constants: UPPER_SNAKE
- Classes: PascalCase
- Files: kebab-case

### Functions
- Do one thing
- Max 3 parameters
- Return early
- Name describes action
