#!/usr/bin/env bash
# 스모크 테스트와 본 실행을 한 작업으로 묶는다. 스모크가 통과해야 본 실행을 시작한다.
#   tmux new -d -s srreid -c /workspace/sr-reid "bash scripts/run_pipeline.sh 2>&1 | tee -a logs/run_all.log"
# 재실행: 통과 표식이 있으면 스모크를 건너뛰고 본 실행을 이어간다. 표식이 없으면 스모크를 처음부터 다시 한다.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=/workspace/miniforge3/envs/srreid-core/bin/python
export PYTHONPATH=$ROOT
PASSED=runs/smoke_passed

if [ -f "$PASSED" ]; then
  echo "[pipeline] 스모크 통과 표식 있음, 건너뜀"
else
  echo "[pipeline] 스모크 시작 ($(date '+%F %T'))"
  rm -rf runs/smoke_001 logs/smoke_001 logs/stages_smoke_001.tsv \
    runs/smoke_data/processed runs/smoke_data/results runs/smoke_data/REPORT.md
  [ -d runs/smoke_data/market ] || "$PY" tests/make_synth_market.py --out runs/smoke_data/market
  bash scripts/run_all.sh configs/smoke.yaml
  rc=$?
  "$PY" scripts/smoke_gate.py || { echo "[pipeline] 본 실행을 시작하지 않음 (run_all rc=$rc). logs/STATUS.md 확인"; exit 1; }
  [ "$rc" = 0 ] || { echo "[pipeline] 스모크 run_all rc=$rc, 본 실행을 시작하지 않음"; exit 1; }
  touch "$PASSED"
fi

echo "[pipeline] 본 실행 시작 ($(date '+%F %T'))"
bash scripts/run_all.sh configs/pilot.yaml
