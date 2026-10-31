#!/usr/bin/env bash
# 전체 파일럿 무인 실행. 사용자 승인이 필요한 단계가 없다. 재실행하면 done 표식이 있는 단계와 검증된 SR 출력을 건너뛴다.
#   tmux new -d -s srreid "bash scripts/run_all.sh 2>&1 | tee logs/run_all.log"
# 중단 조건은 둘뿐이다: LR 생성 게이트(데이터 확보·검증 포함) 실패, test 가드 위반. 나머지 실패는 기록하고 계속한다.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
CONFIG=${1:-configs/pilot.yaml}
PY=/workspace/miniforge3/envs/srreid-core/bin/python
export PYTHONPATH=$ROOT PYTHONNOUSERSITE=1 HF_HUB_ENABLE_HF_TRANSFER=0
export TORCH_HOME=/workspace/weights/torch XDG_CACHE_HOME=/workspace/cache/xdg
unset ALLOW_TEST
mkdir -p logs

RUN_ID=$(sed -n 's/^run_id: *\([^ #]*\).*/\1/p' "$CONFIG")
RUNS=$(sed -n 's/^  runs: *//p' "$CONFIG")
RESULTS=$(sed -n 's/^  results: *//p' "$CONFIG")
RUN=$RUNS/$RUN_ID
DONE=$RUN/done
SLOG=logs/$RUN_ID
mkdir -p "$DONE" "$SLOG" "$RESULTS"
TSV=logs/stages_$RUN_ID.tsv
FAILS=$RESULTS/failures.csv
rm -f "$RUN/pause_sr"

status() { "$PY" report/status.py "$RUN_ID" "$RESULTS" "$1"; }
abort() {
  echo "[run_all] 전체 중단: $1"
  status "전체 중단: $1"
  kill_tree $$
  exit "${2:-1}"
}
kill_tree() {
  local c
  for c in $(pgrep -P "$1"); do
    kill_tree "$c"
    kill "$c" 2>/dev/null
  done
}
check_guard() { [ -f logs/GUARD_VIOLATION ] && abort "test 가드 위반 (logs/GUARD_VIOLATION)" 3; return 0; }
record_failure() {
  [ -f "$FAILS" ] || echo "time,run_id,stage,condition,image,reason" > "$FAILS"
  echo "$(date '+%F %T'),$RUN_ID,$1,,,rc=$2 로그: $SLOG/$1.log" >> "$FAILS"
}

# stage <이름> <fatal|soft> <명령...>: done 표식이 있으면 건너뛴다.
stage() {
  local name=$1 mode=$2; shift 2
  if [ -f "$DONE/$name.done" ]; then
    echo "[run_all] 건너뜀: $name"
    return 0
  fi
  echo "[run_all] 시작: $name ($(date '+%F %T'))"
  local t0=$SECONDS rc=0
  "$@" >> "$SLOG/$name.log" 2>&1 || rc=$?
  local dt=$((SECONDS - t0))
  check_guard
  [ "$rc" = 3 ] && abort "test 가드 위반 ($name)" 3
  if [ "$rc" = 0 ]; then
    touch "$DONE/$name.done"
    printf '%s\tsuccess\t%s\t%s\n' "$name" "$dt" "$(date '+%F %T')" >> "$TSV"
    echo "[run_all] 성공: $name (${dt}초)"
    return 0
  fi
  printf '%s\tfailed(rc=%s)\t%s\t%s\n' "$name" "$rc" "$dt" "$(date '+%F %T')" >> "$TSV"
  echo "[run_all] 실패: $name rc=$rc, 로그 $SLOG/$name.log"
  [ "$mode" = fatal ] && abort "$name 실패 (rc=$rc)" 2
  record_failure "$name" "$rc"
  return "$rc"
}

# 학습: OOM(77)이면 SR을 멈추게 하고 최대 3회 재시도. last.pt에서 이어간다.
train_with_retry() {
  local mode=$1; shift
  local rc=0 try
  for try in 1 2 3 4; do
    rc=0
    "$@" || rc=$?
    [ "$rc" != 77 ] && break
    echo "[train_$mode] OOM, SR 일시 중지 후 재시도 $try"
    touch "$RUN/pause_sr"
    sleep 120
  done
  [ "$rc" = 0 ] && rm -f "$RUN/pause_sr"
  return "$rc"
}
export -f train_with_retry

train_stage() {
  local mode=$1; shift
  stage "train_$mode" soft train_with_retry "$mode" "$@"
  touch "$DONE/train_$mode.finished"  # 성공 여부와 무관하게 SR 마지막 재시도의 대기를 푼다.
}

# ---------- 준비 ----------
stage setup_envs fatal bash env/setup_envs.sh
stage freeze_config fatal "$PY" scripts/freeze_config.py "$CONFIG"
stage tests fatal "$PY" -m pytest -q tests
stage fetch_weights soft "$PY" scripts/fetch_weights.py --config "$CONFIG"
if grep -q '^  fetch_market: false' "$CONFIG"; then
  echo "[run_all] fetch_market 생략 (config)"
else
  stage fetch_market fatal "$PY" scripts/fetch_market.py
fi

# ---------- 데이터 검증, LR 생성, 게이트. 실패하면 전체 중단 ----------
stage prepare_split fatal "$PY" data/prepare_split.py --config "$CONFIG"
stage make_lr fatal "$PY" data/make_lr.py --config "$CONFIG"
stage lr_gate fatal "$PY" data/make_lr.py --config "$CONFIG" --verify

# ---------- 시간 측정: 처음 50장을 단독 실행. 학습과 나머지 SR을 시작하기 전에 끝낸다 ----------
stage sr_exclusive soft "$PY" sr/run.py --config "$CONFIG" --phase exclusive --require-exclusive
stage sr_determinism soft "$PY" sr/run.py --config "$CONFIG" --phase determinism

# ---------- 병렬: 학습 3개와 나머지 SR ----------
rm -f "$DONE"/train_*.finished
train_stage hr "$PY" reid/train_hr.py --config "$CONFIG" &
P1=$!
train_stage mixed "$PY" reid/train_lr.py --config "$CONFIG" --mixed &
P2=$!
train_stage lr_only "$PY" reid/train_lr.py --config "$CONFIG" &
P3=$!
stage sr_rest soft "$PY" sr/run.py --config "$CONFIG" --phase rest &
P4=$!
while kill -0 $P1 2>/dev/null || kill -0 $P2 2>/dev/null || kill -0 $P3 2>/dev/null || kill -0 $P4 2>/dev/null; do
  check_guard
  sleep 30
done
wait $P1 $P2 $P3 $P4
check_guard

# ---------- 평가: bicubic으로 먼저 end-to-end 점검. 실패해도 SR 출력은 재생성하지 않는다 ----------
stage eval_bicubic soft "$PY" eval/run_reid.py --config "$CONFIG" --only bicubic
stage eval_reid soft "$PY" eval/run_reid.py --config "$CONFIG"
stage eval_fidelity soft "$PY" eval/metrics_fidelity.py --config "$CONFIG"
stage eval_bootstrap soft "$PY" eval/bootstrap.py --config "$CONFIG"
stage eval_correlate soft "$PY" eval/correlate.py --config "$CONFIG"
stage report_summary soft "$PY" report/summary.py --config "$CONFIG"
stage report_gonogo soft "$PY" report/go_nogo.py --config "$CONFIG"

status "전체 단계 실행 완료. 단계별 성공 여부는 아래 표를 확인하세요."
echo "[run_all] 종료. logs/STATUS.md 확인."
