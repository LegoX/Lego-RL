#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/../logs}"
PORT="${PORT:-8090}"
TUNNEL="${TUNNEL:-true}"

echo "=== Harbor RL Training Dashboard ==="
echo "  Log dir:  $LOG_DIR"
echo "  Port:     $PORT"
echo "  Tunnel:   $TUNNEL"

# Build frontend if dist/ is missing or stale
if [ ! -f "$SCRIPT_DIR/dist/index.html" ]; then
  echo "[build] Building frontend..."
  cd "$SCRIPT_DIR"
  npm run build
fi

# Start the Python API + static server
echo "[server] Starting on :$PORT ..."
python3 "$SCRIPT_DIR/server.py" \
  --port "$PORT" \
  --log-dir "$LOG_DIR" \
  --static-dir "$SCRIPT_DIR/dist" \
  ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
  ${WANDB_PROJECT:+--wandb-project "$WANDB_PROJECT"} &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

sleep 1
if ! kill -0 $SERVER_PID 2>/dev/null; then
  echo "[error] Server failed to start"
  exit 1
fi
echo "[server] Running (PID $SERVER_PID)"

# Open Cloudflare quick tunnel for public access
if [ "$TUNNEL" = "true" ]; then
  echo "[tunnel] Opening Cloudflare tunnel to localhost:$PORT ..."
  cloudflared tunnel --url "http://127.0.0.1:$PORT" 2>&1 &
  TUNNEL_PID=$!
  echo "[tunnel] Running (PID $TUNNEL_PID) — public URL will appear above"
  trap "kill $SERVER_PID $TUNNEL_PID 2>/dev/null; exit" INT TERM EXIT
fi

echo ""
echo "Dashboard ready at http://127.0.0.1:$PORT"
echo "Press Ctrl+C to stop."
wait
