#!/bin/bash
# Vetinari Startup Script (Linux/macOS)
# Usage: ./start.sh [options]
#   ./start.sh                          -- Start with dashboard
#   ./start.sh --goal "My goal here"    -- Run a specific goal
#   ./start.sh --no-dashboard           -- CLI only
#   ./start.sh serve                    -- Dashboard only

set -e

# Change to script directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Set default environment variables if not already set
export LM_STUDIO_HOST="${LM_STUDIO_HOST:-http://localhost:1234}"

echo "============================================================"
echo " VETINARI AI Orchestration System"
echo " Host: $LM_STUDIO_HOST"
echo "============================================================"

# Run Vetinari
if [ "$1" = "serve" ]; then
    python -m vetinari serve "${@:2}"
elif [ "$1" = "run" ]; then
    python -m vetinari run "${@:2}"
elif [ "$1" = "status" ]; then
    python -m vetinari status
elif [ "$1" = "review" ]; then
    python -m vetinari review
else
    python -m vetinari start "$@"
fi
