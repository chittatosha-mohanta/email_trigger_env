#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT="${PORT:-7860}"
HOST="${HOST:-127.0.0.1}"
BASE_URL="${ENV_BASE_URL:-http://${HOST}:${PORT}}"

echo "==> Python syntax check"
PY="${PY:-python3}"
if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
  PY="$ROOT_DIR/.venv/bin/python"
fi

"$PY" -m compileall server inference.py >/dev/null

echo "==> Running tests"
"$PY" -m pytest -q

echo "==> Starting API server"
LOG_DIR="$ROOT_DIR/.tmp"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/server.log"

"$PY" -m uvicorn server.app:app --host "$HOST" --port "$PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"

cleanup() {
  kill "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==> Waiting for /health"
for _ in {1..30}; do
  if curl -fsS "${BASE_URL}/health" >/dev/null; then
    break
  fi
  sleep 0.2
done
curl -fsS "${BASE_URL}/health" >/dev/null

echo "==> Smoke: reset endpoint"
curl -fsS -X POST "${BASE_URL}/reset" -H 'content-type: application/json' -d '{"task_id": 1}' >/dev/null

echo "==> Running inference baseline (fallback if no API key)"
ENV_BASE_URL="$BASE_URL" "$PY" inference.py

echo "==> Docker build"
if ! command -v docker >/dev/null 2>&1; then
  echo "==> Docker not found; skipping Docker checks"
  echo "==> OK: local validation passed (without Docker checks)"
  exit 0
fi

docker build -t email-triage-openenv:local .

echo "==> Docker run smoke"
docker run --rm -d -p 17860:7860 --name email-triage-openenv-local email-triage-openenv:local >/dev/null
sleep 1
curl -fsS "http://127.0.0.1:17860/health" >/dev/null
docker rm -f email-triage-openenv-local >/dev/null

echo "==> OK: local validation passed"

