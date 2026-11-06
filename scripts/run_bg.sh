#!/bin/bash
# 긴 GPU·CPU 작업을 Claude Code와 분리된 tmux 세션(없으면 nohup)으로 띄운다. Claude Code가 끊겨도 계속 돈다.
# 사용: bash scripts/run_bg.sh <세션이름> <명령...>
#   예: bash scripts/run_bg.sh s3e bash stage05/run_after_tpfinal_s2.sh
# 로그: logs/bg/<이름>.log, 종료 코드: logs/bg/<이름>.exit (끝나면 생김). 상태: tmux ls, tail -f logs/bg/<이름>.log
# 재실행 안전성은 각 실행 스크립트의 완료 표시(runs/stage05/done/*.json, 완성 WebP 존재)로 보장한다. 같은 이름이 돌고 있으면 시작하지 않는다.
set -u
ROOT=/workspace/sr-reid
NAME=${1:-}
[ -n "$NAME" ] && [ $# -ge 2 ] || { echo "사용: $0 <세션이름> <명령...>"; exit 2; }
shift
case "$NAME" in *[!A-Za-z0-9_-]*) echo "세션 이름은 영문·숫자·_·-만"; exit 2;; esac
mkdir -p "$ROOT/logs/bg"
LOG="$ROOT/logs/bg/$NAME.log"
EXIT="$ROOT/logs/bg/$NAME.exit"
RUN="$ROOT/logs/bg/$NAME.sh"
if tmux has-session -t "=$NAME" 2>/dev/null || pgrep -f "bash $RUN" >/dev/null; then
  echo "이미 실행 중: $NAME"; exit 1
fi
rm -f "$EXIT"
{
  echo '#!/bin/bash'
  echo "cd $ROOT || exit 1"
  echo 'export LANG=${LANG:-C.UTF-8} LC_ALL=${LC_ALL:-C.UTF-8} TORCH_HOME=${TORCH_HOME:-/workspace/weights/torch} HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}'
  printf 'echo "[$(date -Is)] 시작:" %q >> %q\n' "$*" "$LOG"
  printf '%q ' "$@"; echo ">> $LOG 2>&1"
  echo 'rc=$?'
  echo "echo \$rc > $EXIT"
  echo "echo \"[\$(date -Is)] 종료 rc=\$rc\" >> $LOG"
} > "$RUN"
if command -v tmux >/dev/null; then
  tmux -u new-session -d -s "$NAME" "bash $RUN"
  echo "tmux 세션 '$NAME' 시작. 로그: $LOG"
else
  nohup setsid bash "$RUN" >/dev/null 2>&1 &
  echo "nohup 시작(pid $!). 로그: $LOG"
fi
