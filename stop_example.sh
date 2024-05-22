#!/usr/bin/env bash
WORK_DIR=$(dirname $(readlink -f $0))
echo "${WORK_DIR}"

PID=$1

for session in $(tmux ls | grep ^xinhai | grep "$PID" | awk -F: '{print $1}');
do
  tmux kill-session -t "$session"
done