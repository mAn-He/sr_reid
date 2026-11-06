#!/bin/bash
# 지시 3e. 최종 처리량. 다른 작업 없이 단독으로 돈다.
# 주의(2026-11-06): 서버 이전으로 시작하지 않았다. 새 서버에서는 run_after_tpfinal_s2.sh를 쓴다.
cd /workspace/sr-reid || exit 1
bash scripts/disk_guard.sh 20 || exit 1
ulimit -n 524288
export PYTHONPATH=/workspace/sr-reid PYTHONNOUSERSITE=1
/workspace/miniforge3/envs/srreid-core/bin/python stage05/run_train.py tpfinal 2>&1 | tee -a logs/stage05/run_train.log
echo "run_3e 종료" | tee -a logs/stage05/run_train.log
