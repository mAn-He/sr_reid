#!/bin/bash
# 메모리 진단 체인이 끝나면 이어서 돈다. D.5 보강 검증.
cd /workspace/sr-reid || exit 1
until grep -q "run_after 종료" logs/stage05/run_train.log; do sleep 20; done
ulimit -n 524288
export PYTHONPATH=/workspace/sr-reid PYTHONNOUSERSITE=1
/workspace/miniforge3/envs/srreid-core/bin/python stage05/run_train.py h6 dataeq deteq 2>&1 | tee -a logs/stage05/run_train.log
echo "run_equiv 종료" | tee -a logs/stage05/run_train.log
