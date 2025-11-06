#!/bin/bash
# Run Sentinel Server Integration Tests
# ======================================
#
# Prerequisites:
# - Server must be running on port 11435
# - Run: ./server/start_server.sh

cd "$(dirname "$0")/.."

# Check if server is running
if ! curl -s http://localhost:11435/api/health > /dev/null 2>&1; then
    echo "❌ Sentinel server not running on port 11435"
    echo "   Start it with: ./server/start_server.sh"
    exit 1
fi

echo "✅ Server detected, running tests..."
echo ""

# Activate venv and run tests
source venv/bin/activate
python3 -m pytest tests/integration/test_server.py -v --tb=short
