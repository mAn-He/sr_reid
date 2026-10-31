#!/usr/bin/env bash
# Stage 0 무인 실행. 재실행하면 끝난 단계와 끝난 측정을 건너뛴다. test 가드 위반이면 전체 중단.
#   tmux new -d -s stage0 -c /workspace/sr-reid "bash scripts/run_stage0.sh 2>&1 | tee -a logs/stage0.log"
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=/workspace/miniforge3/envs/srreid-core/bin/python
KPY=/workspace/miniforge3/envs/srreid-kpr/bin/python
export PYTHONPATH=$ROOT PYTHONNOUSERSITE=1 WANDB_MODE=disabled
unset ALLOW_TEST
DONE=runs/stage0/done
SLOG=logs/stage0
mkdir -p "$DONE" "$SLOG" results
TSV=logs/stages_stage0.tsv

stage() {
  local name=$1; shift
  [ -f "$DONE/$name.done" ] && { echo "[stage0] 건너뜀: $name"; return 0; }
  echo "[stage0] 시작: $name ($(date '+%F %T'))"
  local t0=$SECONDS rc=0
  "$@" >> "$SLOG/$name.log" 2>&1 || rc=$?
  printf '%s\t%s\t%s\t%s\n' "$name" "rc=$rc" "$((SECONDS - t0))" "$(date '+%F %T')" >> "$TSV"
  if [ -f logs/GUARD_VIOLATION ] || [ "$rc" = 3 ]; then echo "[stage0] test 가드 위반, 전체 중단"; exit 3; fi
  if [ "$rc" = 0 ]; then touch "$DONE/$name.done"; echo "[stage0] 성공: $name"; else echo "[stage0] 실패: $name rc=$rc, 로그 $SLOG/$name.log"; fi
  return "$rc"
}

wait_pilot() {
  while tmux has-session -t srreid 2>/dev/null; do sleep 60; done
  echo "파일럿 tmux 종료 확인 $(date '+%F %T')"
}

stage envs bash env/setup_envs_stage0.sh || exit 1
stage fetch "$PY" scripts/stage0_fetch.py
stage extract_kpr "$PY" scripts/stage0_extract_kpr.py || exit 1
stage prepare_data "$PY" stage0/prepare_data.py || exit 1
stage verify_scale "$KPY" stage0/verify_scale.py
stage wait_pilot wait_pilot
stage probe "$PY" stage0/run_probe.py
stage report "$PY" stage0/report.py
echo "[stage0] 종료. STAGE0_REPORT.md 확인."
