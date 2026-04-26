"""Chaos injection tests for resilience verification.

Injects controlled failures (OOM, lock contention, network timeouts, disk full)
to verify the system degrades gracefully rather than crashing.
"""
