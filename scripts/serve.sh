#!/usr/bin/env bash
# Start, stop or check the local model servers.
#
#   scripts/serve.sh agent        # start one role (agent, judge, embed, agent-small, ...)
#   scripts/serve.sh status       # which servers are up
#   scripts/serve.sh stop agent
#
# Which model serves each role is set in config/models.env (D18), the same
# file the Python client reads by default. This script only starts local
# llama.cpp servers; hosted roles (config/models.openrouter.env, D22) need none. Each server runs in its own detached tmux
# session named after the role and appends its output to ~/logs/<role>.log.
# Models are loaded by path, so nothing is downloaded, and each gets a fixed
# --alias: the model name the client sends and every saved call records.
# --metrics turns on each server's /metrics page, which scripts/monitor.py reads (D31).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO/config/models.env"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
LOGS="$HOME/logs"
CHAT_FLAGS=(-np 1 -ngl 99 --jinja --no-mmproj --no-reasoning-preserve --metrics)

[ -f "$CONFIG" ] || { echo "missing $CONFIG" >&2; exit 1; }
# shellcheck source=/dev/null
source "$CONFIG"

roles() {  # role names from the config, e.g. AGENT_SMALL_NAME -> agent-small
  sed -n 's/^\([A-Z_]*\)_NAME=.*/\1/p' "$CONFIG" | tr 'A-Z_' 'a-z-'
}

setting() {  # setting <role> <FIELD>: the value of e.g. AGENT_SMALL_PORT
  local var
  var="$(printf '%s' "$1" | tr 'a-z-' 'A-Z_')_$2"
  printf '%s' "${!var:-}"
}

usage() {
  echo "usage: scripts/serve.sh <role>|status|stop <role>   (roles: $(roles | tr '\n' ' '))" >&2
  exit 2
}

case "${1:-}" in
  "" | -h | --help) usage ;;
  status)
    for r in $(roles); do
      prov="$(setting "$r" PROVIDER)"
      if [ -n "$prov" ] && [ "$prov" != llamacpp ]; then
        printf '%-12s hosted (%s), nothing to start\n' "$r" "$prov"
        continue
      fi
      p="$(setting "$r" PORT)"
      if tmux has-session -t "$r" 2>/dev/null; then s="tmux session running"; else s="no tmux session"; fi
      if curl -sf "http://localhost:$p/health" >/dev/null 2>&1; then h="healthy"; else h="not answering"; fi
      printf '%-12s port %s: %s, %s\n' "$r" "$p" "$s" "$h"
    done
    exit 0
    ;;
  stop)
    target="${2:-}"
    [ -n "$target" ] || usage
    tmux kill-session -t "$target" && echo "stopped $target"
    exit 0
    ;;
esac

role="$1"
roles | grep -qx "$role" || { echo "unknown role '$role'" >&2; usage; }
PROVIDER="$(setting "$role" PROVIDER)"
if [ -n "$PROVIDER" ] && [ "$PROVIDER" != llamacpp ]; then
  echo "role '$role' is hosted ($PROVIDER); there is no local server to start" >&2
  exit 1
fi
KIND="$(setting "$role" KIND)"
MODEL="$(setting "$role" FILE)"
ALIAS="$(setting "$role" NAME)"
PORT="$(setting "$role" PORT)"
CTX="$(setting "$role" CTX)"
for v in KIND MODEL ALIAS PORT CTX; do
  [ -n "${!v}" ] || { echo "config/models.env is missing a setting for $role ($v)" >&2; exit 1; }
done
case "$KIND" in
  chat) FLAGS=(-c "$CTX" "${CHAT_FLAGS[@]}") ;;
  embed) FLAGS=(--embedding -c "$CTX" -np 1 -ngl 99 --metrics) ;;
  *) echo "unknown KIND '$KIND' for $role (expected chat or embed)" >&2; exit 1 ;;
esac

[ -x "$LLAMA_SERVER" ] || { echo "llama-server not found at $LLAMA_SERVER" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "model file not found: $MODEL" >&2; exit 1; }
if tmux has-session -t "$role" 2>/dev/null; then
  echo "a tmux session named '$role' already exists; stop it first: scripts/serve.sh stop $role" >&2
  exit 1
fi

mkdir -p "$LOGS"
cmd=("$LLAMA_SERVER" -m "$MODEL" --alias "$ALIAS" --port "$PORT" "${FLAGS[@]}")
{
  echo
  echo "=== $(date -Iseconds) starting $role: $("$LLAMA_SERVER" --version 2>&1 | head -1)"
  echo "=== ${cmd[*]}"
} >> "$LOGS/$role.log"

tmux new-session -d -s "$role" "$(printf '%q ' "${cmd[@]}") 2>&1 | tee -a $(printf '%q' "$LOGS/$role.log")"
echo "started $role in tmux session '$role' (port $PORT); log: $LOGS/$role.log"
echo "check with: scripts/serve.sh status   (loading takes a minute or two)"
