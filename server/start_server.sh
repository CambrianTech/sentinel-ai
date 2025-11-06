#!/bin/bash
# Start Sentinel HTTP Server
#
# Usage:
#   ./server/start_server.sh [port]
#
# Default port: 11435
# Set SENTINEL_PORT environment variable to change

cd "$(dirname "$0")/.."

PORT="${1:-${SENTINEL_PORT:-11435}}"

echo "🧬 Starting Sentinel AI Server on port $PORT..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r server/requirements.txt
else
    source venv/bin/activate
fi

# Start server
export SENTINEL_PORT=$PORT
python3 server/sentinel_server.py
