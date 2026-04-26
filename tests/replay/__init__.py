"""Deterministic replay testing infrastructure for multi-agent interactions.

Records LLM calls and tool invocations as structured events, then replays
them with stubs for regression testing without live model calls.
"""
