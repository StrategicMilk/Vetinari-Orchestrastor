"""Runtime journey tests — end-to-end coverage of the Litestar HTTP stack.

These tests exercise the full request pipeline (routing, middleware, CSRF,
serialisation, exception handlers) for the user stories US-004, US-005,
and US-006 from the SESSION-30 PRD.  Every request goes through the real
``TestClient`` — handler functions are NEVER called directly.
"""
