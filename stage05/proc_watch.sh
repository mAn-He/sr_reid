#!/bin/bash
# 체인 소멸 사건(2026-11-01 01:45~01:47, /workspace 쿼터 초과 의심) 진단용 감시. 10초마다 한 줄.
# cgroup v1·v2 자동 판별. v2: memory.current, memory.stat(anon·file·shmem), memory.events(oom·oom_kill·max).
# v1: memory.usage_in_bytes, memory.stat(total_rss·total_cache·total_shmem), memory.oom_control.
# 출력은 표준출력. 로컬 디스크에 남기려면: bash stage05/proc_watch.sh >> /root/srreid_cache/proc_watch.log
G2=/sys/fs/cgroup
G1=/sys/fs/cgroup/memory
if [ -f $G2/memory.current ]; then V=v2; elif [ -f $G1/memory.usage_in_bytes ]; then V=v1; else echo "cgroup 메모리 인터페이스 없음" >&2; exit 1; fi
st() { awk -v k="$2" '$1==k{print $2}' "$1"; }
gib() { awk -v b="${1:-0}" 'BEGIN{printf "%.2f", b/1073741824}'; }
echo "# proc_watch cgroup=$V $(date -Is)"
while true; do
  w=ok; echo x > /workspace/sr-reid/logs/.q 2>/dev/null || w=FAIL; rm -f /workspace/sr-reid/logs/.q
  if [ $V = v2 ]; then
    usage=$(cat $G2/memory.current); anon=$(st $G2/memory.stat anon); file=$(st $G2/memory.stat file); shmem=$(st $G2/memory.stat shmem)
    oom=$(st $G2/memory.events oom); kill=$(st $G2/memory.events oom_kill); hitmax=$(st $G2/memory.events max)
  else
    usage=$(cat $G1/memory.usage_in_bytes); anon=$(st $G1/memory.stat total_rss); file=$(st $G1/memory.stat total_cache); shmem=$(st $G1/memory.stat total_shmem)
    oom=$(st $G1/memory.oom_control under_oom); kill=$(st $G1/memory.oom_control oom_kill); hitmax=$(cat $G1/memory.failcnt 2>/dev/null)
  fi
  echo "$(date +%T) write=$w tmux=$(tmux ls -F '#S' 2>/dev/null | tr '\n' ' ')py=$(pgrep -c python) usage=$(gib $usage)G anon=$(gib $anon)G file=$(gib $file)G shmem=$(gib $shmem)G oom=$oom oom_kill=$kill max_events=$hitmax"
  sleep 10
done
