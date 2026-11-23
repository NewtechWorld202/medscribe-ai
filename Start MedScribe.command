#!/bin/bash
# Double-click to open MedScribe AI in Google Chrome.
# (Chrome only allows live microphone transcription on localhost/https, not on files opened directly.)
# If the backend virtualenv exists, the FastAPI backend (Groq speech-to-text + clinic medicine list)
# is started too, on http://127.0.0.1:8770. Without it the page uses the browser's own speech recognition.
cd "$(dirname "$0")" || exit 1
PORT=8765
BACKEND_PORT=8770
URL="http://localhost:$PORT/medscribe-ai.html"
PIDS=()

if ! lsof -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  python3 -m http.server $PORT --bind 127.0.0.1 >/dev/null 2>&1 &
  PIDS+=($!)
fi

if [ -x .venv/bin/uvicorn ]; then
  # Local AI note writing uses the Ollama app; start it if it is installed but not running.
  if ! curl -s -m 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && [ -d /Applications/Ollama.app ]; then
    open -g -a Ollama
    for i in 1 2 3 4 5 6 7 8 9 10; do curl -s -m 1 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break; sleep 1; done
  fi
  if ! lsof -iTCP:$BACKEND_PORT -sTCP:LISTEN >/dev/null 2>&1; then
    PATH="/opt/homebrew/bin:/usr/local/bin:$PATH" .venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port $BACKEND_PORT --log-level warning &
    PIDS+=($!)
  fi
  echo "  Backend: http://127.0.0.1:$BACKEND_PORT (speech-to-text and AI notes run on this Mac; settings in .env)"
else
  echo "  Backend not installed. Browser speech recognition only."
  echo "  To install it: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
fi
sleep 1

open -a "Google Chrome" "$URL" 2>/dev/null || open "$URL"

echo ""
echo "  MedScribe AI is running at $URL"
echo "  Keep this window open while you use the app. Close it to stop."
echo ""
trap 'kill ${PIDS[@]} 2>/dev/null' EXIT
if [ ${#PIDS[@]} -gt 0 ]; then wait; fi
