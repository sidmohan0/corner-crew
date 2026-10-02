#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
project=/Users/sid/projects/corner-crew/corner-crew
session=corner-crew-local
model=/Users/sid/models/qwen3-vl-8b-instruct/Qwen3VL-8B-Instruct-Q4_K_M.gguf

tmux has-session -t "=$session" 2>/dev/null ||
  tmux new-session -d -s "$session" -n control -c "$project"

start_window() {
  local name="$1" port="$2" command="$3"
  if tmux list-windows -t "=$session" -F '#{window_name}' | grep -Fxq "$name"; then
    return
  fi
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $port is already occupied; leaving its service untouched." >&2
    exit 1
  fi
  tmux new-window -d -t "$session:" -n "$name" -c "$project" "$command"
}

ready() {
  local url="$1"
  for ((attempt=0; attempt<90; attempt++)); do
    if curl --silent --fail --max-time 2 "$url" >/dev/null; then return; fi
    sleep 1
  done
  echo "Readiness timed out: $url. Inspect: tmux attach -t $session" >&2
  exit 1
}

start_window model 8080 "exec /opt/homebrew/bin/llama-server -m '$model' --alias local-model --host 127.0.0.1 --port 8080 -c 4096"
ready http://127.0.0.1:8080/health
minicpm=/Users/sid/models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf
if [[ -f "$minicpm" ]]; then
  start_window minicpm 8081 "exec /opt/homebrew/bin/llama-server -m '$minicpm' --alias MiniCPM5-2B --host 127.0.0.1 --port 8081 -c 4096 --jinja"
  ready http://127.0.0.1:8081/health
else
  echo "MiniCPM is required by the chat pipeline. Download its weights to $minicpm (see README)." >&2
  exit 1
fi
start_window api 8000 "exec /opt/homebrew/bin/uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"
ready http://127.0.0.1:8000/health
start_window web 3000 "export PATH='/Users/sid/.nvm/versions/node/v25.8.2/bin:/opt/homebrew/bin:'$PATH; exec /Users/sid/.nvm/versions/node/v25.8.2/bin/npm --prefix web run dev"
ready http://localhost:3000
echo "Chat: http://localhost:3000 | API docs: http://127.0.0.1:8000/docs"
