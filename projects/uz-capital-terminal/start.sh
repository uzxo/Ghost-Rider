#!/bin/bash
# ---------------------------------------------------------------
# UZ Capital — local launcher for the demo terminal.
# Serves terminal.html on http://localhost:8090 and opens it in
# your default browser. Running locally avoids the GitHub Pages
# CORS-proxy rate limits, so live prices populate reliably.
#
# Usage:  bash start.sh
# Stop:   Ctrl+C
# ---------------------------------------------------------------
set -e
cd "$(dirname "$0")"

# Kill anything already on 8090 (silently)
lsof -ti:8090 | xargs kill -9 2>/dev/null || true

python3 -m http.server 8090 &
SERVER_PID=$!
sleep 1

URL="http://localhost:8090/terminal.html?v=$(date +%s)"

# Open in default browser (macOS / Linux)
if command -v open >/dev/null 2>&1; then
  open "$URL"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL"
else
  echo "Open this URL in your browser: $URL"
fi

echo "UZ Capital running at http://localhost:8090 — press Ctrl+C to stop."
trap "kill $SERVER_PID 2>/dev/null" EXIT
wait $SERVER_PID
