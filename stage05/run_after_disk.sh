#!/bin/bash
# 볼륨 확장 후 지시 3a~3d: ×4 checkpointing 등가성, 워커 정규화, KPR copy_read, 메모리 추가 측정. 볼륨이 부족하면 시작하지 않는다.
cd /workspace/sr-reid || exit 1
bash scripts/disk_guard.sh 20 || exit 1
ulimit -n 524288
export PYTHONPATH=/workspace/sr-reid PYTHONNOUSERSITE=1
/workspace/miniforge3/envs/srreid-core/bin/python stage05/run_train.py ckpt4 wnorm kprcopy h8 2>&1 | tee -a logs/stage05/run_train.log
echo "run_3abcd 종료" | tee -a logs/stage05/run_train.log
