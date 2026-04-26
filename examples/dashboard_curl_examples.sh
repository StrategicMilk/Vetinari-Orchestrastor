#!/usr/bin/env bash
# =============================================================================
# Vetinari Dashboard — cURL Examples
#
# Start the server first:
#   python -m vetinari serve --port 5000
#
# Then run individual commands below, or execute this whole script:
#   bash examples/dashboard_curl_examples.sh
# =============================================================================

BASE="http://localhost:5000"

sep() { echo; echo "── $* ──────────────────────────────────────────"; }

# ── Health check ─────────────────────────────────────────────────────────────
sep "GET /api/v1/health"
curl -s "${BASE}/api/v1/health" | python -m json.tool

# ── Latest metrics snapshot ───────────────────────────────────────────────────
sep "GET /api/v1/metrics/latest"
curl -s "${BASE}/api/v1/metrics/latest" | python -m json.tool

# ── Latency time-series (all providers) ──────────────────────────────────────
sep "GET /api/v1/metrics/timeseries?metric=latency"
curl -s "${BASE}/api/v1/metrics/timeseries?metric=latency" | python -m json.tool

# ── Success-rate time-series ──────────────────────────────────────────────────
sep "GET /api/v1/metrics/timeseries?metric=success_rate"
curl -s "${BASE}/api/v1/metrics/timeseries?metric=success_rate" | python -m json.tool

# ── Token-usage time-series ───────────────────────────────────────────────────
sep "GET /api/v1/metrics/timeseries?metric=token_usage"
curl -s "${BASE}/api/v1/metrics/timeseries?metric=token_usage" | python -m json.tool

# ── Memory latency time-series ────────────────────────────────────────────────
sep "GET /api/v1/metrics/timeseries?metric=memory_latency"
curl -s "${BASE}/api/v1/metrics/timeseries?metric=memory_latency" | python -m json.tool

# ── Latency filtered to a specific provider ───────────────────────────────────
sep "GET /api/v1/metrics/timeseries?metric=latency&provider=openai%2Fgpt-4"
curl -s "${BASE}/api/v1/metrics/timeseries?metric=latency&provider=openai%2Fgpt-4" \
  | python -m json.tool

# ── Invalid metric name (expect 400) ─────────────────────────────────────────
sep "GET /api/v1/metrics/timeseries?metric=badname  (expect 400)"
curl -s "${BASE}/api/v1/metrics/timeseries?metric=badname" | python -m json.tool

# ── List recent traces (limit 10) ─────────────────────────────────────────────
sep "GET /api/v1/traces?limit=10"
curl -s "${BASE}/api/v1/traces?limit=10" | python -m json.tool

# ── Search for a specific trace ID ────────────────────────────────────────────
sep "GET /api/v1/traces?trace_id=example-trace-001"
curl -s "${BASE}/api/v1/traces?trace_id=example-trace-001" | python -m json.tool

# ── Get trace detail ──────────────────────────────────────────────────────────
sep "GET /api/v1/traces/example-trace-002"
curl -s "${BASE}/api/v1/traces/example-trace-002" | python -m json.tool

# ── Missing trace (expect 404) ────────────────────────────────────────────────
sep "GET /api/v1/traces/nonexistent  (expect 404)"
curl -s "${BASE}/api/v1/traces/nonexistent" | python -m json.tool

echo
echo "Done."
