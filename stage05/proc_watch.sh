#!/bin/bash
# 체인 소멸 사건(2026-11-01 01:45~01:47, /workspace 쿼터 초과 의심) 진단용. 로컬 디스크에 10초마다 남긴다.
while true; do
  w=ok; echo x > /workspace/sr-reid/logs/.q 2>/dev/null || w=FAIL; rm -f /workspace/sr-reid/logs/.q
  echo "$(date +%T) write=$w tmux=$(tmux ls -F '#S' 2>/dev/null | tr '\n' ' ') py=$(pgrep -c python) usage=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes) oom=$(tail -1 /sys/fs/cgroup/memory/memory.oom_control | cut -d' ' -f2)"
  sleep 10
done
