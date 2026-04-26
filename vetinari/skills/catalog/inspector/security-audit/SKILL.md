---
name: Security Audit
description: OWASP Top 10 coverage with CWE mapping, secrets detection, and dependency vulnerability scanning
mode: security_audit
agent: inspector
version: "1.0.0"
capabilities:
  - security_audit
  - vulnerability_detection
tags:
  - quality
  - security
  - owasp
  - cwe
  - audit
---

# Security Audit

## Purpose

Security Audit performs a comprehensive security review of code and configuration, structured around the OWASP Top 10 vulnerability categories with CWE (Common Weakness Enumeration) mapping for each finding. It goes beyond code correctness to evaluate the security posture of the entire change: input validation, authentication boundaries, data exposure risks, dependency vulnerabilities, and secrets management. Every finding is mapped to a standard CWE ID for traceability and compliance, and classified by severity using CVSS-inspired scoring.

## When to Use

- On any code that handles user input, authentication, or authorization
- When adding new API endpoints or modifying existing ones
- When changing data storage, serialization, or transmission logic
- When integrating external services or dependencies
- When modifying security-sensitive configuration (CORS, CSP, auth tokens)
- Before any release that touches security-relevant code paths
- When dependency-analysis identifies packages with known CVEs

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| code            | string          | Yes      | Code or configuration to audit                                     |
| mode            | string          | No       | "security_audit" (default)                                         |
| context         | dict            | No       | Application context: auth mechanism, data classification, deployment|
| focus_areas     | list[string]    | No       | Specific OWASP categories or CWE IDs to focus on                   |
| thinking_mode   | string          | No       | Thinking budget (recommend "high" for security audits)             |

## Process Steps

1. **A01: Broken Access Control (CWE-284)** -- Check for:
   - Missing authentication on protected endpoints
   - Missing authorization checks (user can access another user's data)
   - Insecure Direct Object References (IDOR) -- user-controlled IDs without ownership check
   - Path traversal: user input used in file paths without sanitization
   - CORS misconfiguration: overly permissive origins or credentials
   - Missing CSRF protection on state-changing endpoints

2. **A02: Cryptographic Failures (CWE-310)** -- Check for:
   - Hardcoded secrets, API keys, passwords, or tokens in source code
   - Weak hashing algorithms (MD5, SHA1 for passwords)
   - Missing encryption for sensitive data at rest or in transit
   - Predictable random values used for security tokens
   - Sensitive data in logs, error messages, or debug output

3. **A03: Injection (CWE-74)** -- Check for:
   - SQL injection: string concatenation in SQL queries
   - Command injection: user input passed to os.system, subprocess without sanitization
   - Template injection: user input rendered in templates without escaping
   - LDAP injection, XML injection, expression language injection
   - Deserialization of untrusted data (pickle.loads, yaml.load without SafeLoader)

4. **A04: Insecure Design (CWE-501)** -- Check for:
   - Business logic flaws: missing validation of state transitions
   - Rate limiting: absence of rate limits on authentication endpoints
   - Insufficient anti-automation (no captcha, no progressive delays)
   - Missing trust boundaries between internal and external components
   - Security controls missing from the design (not just the implementation)

5. **A05: Security Misconfiguration (CWE-16)** -- Check for:
   - Debug mode enabled in production
   - Default credentials not changed
   - Unnecessary features enabled (admin endpoints, debug endpoints)
   - Missing security headers (Content-Security-Policy, X-Frame-Options)
   - Overly verbose error messages revealing stack traces to users

6. **A06: Vulnerable Components (CWE-1035)** -- Check for:
   - Dependencies with known CVEs
   - Outdated dependencies past end-of-life
   - Dependencies with no active maintenance
   - Unnecessary dependencies increasing attack surface

7. **A07: Auth Failures (CWE-287)** -- Check for:
   - Weak password policies
   - Missing account lockout after failed attempts
   - Session fixation vulnerabilities
   - Insecure session token generation or storage
   - Missing multi-factor authentication on sensitive operations

8. **A08: Data Integrity Failures (CWE-345)** -- Check for:
   - Missing integrity verification on downloaded code or data
   - Insecure deserialization allowing code execution
   - Missing signature verification on critical data
   - CI/CD pipeline security (untrusted code in build process)

9. **A09: Logging Failures (CWE-778)** -- Check for:
   - Insufficient logging of security events (failed auth, privilege changes)
   - Sensitive data in logs (passwords, tokens, PII)
   - Missing audit trail for administrative actions
   - Log injection vulnerabilities (user input in log messages)

10. **A10: SSRF (CWE-918)** -- Check for:
    - User-controlled URLs fetched server-side without validation
    - Internal service URLs exposed through URL parameters
    - Missing allowlist for outbound request destinations
    - DNS rebinding vulnerabilities

11. **Secrets scanning** -- Beyond OWASP, explicitly scan for:
    - API keys, tokens, passwords in source code
    - Private keys or certificates
    - Connection strings with embedded credentials
    - Environment variable names that suggest secrets without proper handling

12. **Finding compilation** -- For each finding:
    - Map to OWASP category and CWE ID
    - Assign severity: critical, high, medium, low, info
    - Describe the vulnerability and its potential impact
    - Provide a specific remediation recommendation

## Output Format

The skill produces a security audit report:

```json
{
  "passed": false,
  "grade": "C",
  "score": 0.62,
  "issues": [
    {
      "severity": "critical",
      "category": "injection",
      "description": "SQL injection via string formatting in query builder",
      "file": "vetinari/web/projects_api.py",
      "line": 67,
      "cwe": "CWE-89",
      "owasp": "A03:2021-Injection",
      "suggestion": "Use parameterized queries: cursor.execute('SELECT * FROM projects WHERE id = ?', (project_id,))"
    },
    {
      "severity": "high",
      "category": "cryptographic_failures",
      "description": "API key hardcoded in source file",
      "file": "vetinari/config/settings.py",
      "line": 12,
      "cwe": "CWE-798",
      "owasp": "A02:2021-Cryptographic Failures",
      "suggestion": "Move to environment variable: os.environ.get('API_KEY')"
    }
  ],
  "suggestions": [
    "Add Content-Security-Policy header to all responses",
    "Implement rate limiting on authentication endpoints"
  ],
  "metrics": {
    "owasp_categories_checked": 10,
    "findings_by_severity": {"critical": 1, "high": 1, "medium": 2, "low": 3, "info": 1},
    "secrets_scan_clean": false
  }
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-INS-002**: Security audit MUST check OWASP Top 10 and map findings to CWE IDs
- **STD-INS-003**: Security audit MUST scan for hardcoded credentials, secrets, and API keys
- **STD-INS-005**: Every issue MUST have a severity level and actionable description
- **STD-INS-006**: Inspector MUST NOT modify code -- only report findings and suggestions
- **STD-INS-007**: Gate decision MUST be based on objective criteria
- **STD-INS-008**: Gate decisions cannot be overridden by any other agent
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **CON-INS-001**: Inspector is READ-ONLY
- **CON-INS-004**: Security audit must complete within timeout even for large codebases
- **GDL-INS-001**: Run security_audit on any code handling user input, authentication, or external data

## Examples

### Example: Auditing a file upload handler

**Input:**
```
code: |
  @app.route('/upload', methods=['POST'])
  def upload_file():
      f = request.files['file']
      f.save(os.path.join('/uploads', f.filename))
      return jsonify({"status": "uploaded"})
context: {auth: "none", deployment: "internet-facing"}
```

**Output (abbreviated):**
```
passed: false
grade: F
issues:
  - [critical/CWE-22] "Path traversal: f.filename could be '../../../etc/passwd', writing outside /uploads"
    fix: "Use werkzeug.utils.secure_filename(f.filename)"

  - [critical/CWE-434] "Unrestricted file upload: no validation of file type, size, or content"
    fix: "Validate file extension, check MIME type, enforce size limit"

  - [high/CWE-284] "No authentication on upload endpoint -- any anonymous user can upload files"
    fix: "Add @login_required decorator"

  - [high/CWE-400] "No file size limit -- attacker could exhaust disk space"
    fix: "Set MAX_CONTENT_LENGTH in Flask config"

  - [medium/CWE-778] "No logging of upload events -- security incidents would be untraceable"
    fix: "Log filename, user, IP, and timestamp for audit trail"
```
