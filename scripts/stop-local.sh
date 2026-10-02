#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
session=corner-crew-local
if tmux has-session -t "=$session" 2>/dev/null; then
  for window in web api minicpm model; do
    if tmux list-windows -t "=$session" -F '#{window_name}' | grep -Fxq "$window"; then
      tmux send-keys -t "$session:$window" C-c
      for ((attempt=0; attempt<20; attempt++)); do
        tmux list-windows -t "=$session" -F '#{window_name}' | grep -Fxq "$window" || break
        sleep 0.5
      done
      tmux kill-window -t "$session:$window" 2>/dev/null || true
    fi
  done
fi
echo "Stopped Corner Crew services. Control window and model files preserved."
