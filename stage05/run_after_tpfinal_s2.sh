#!/bin/bash
# 지시 3e(새 서버). 다른 작업 없이 단독으로 돈다. 띄우기: bash scripts/run_bg.sh s3e bash stage05/run_after_tpfinal_s2.sh
# 재실행하면 끝난 단계(runs/stage05/done/*_s2.json)는 건너뛴다. 이전 서버 run_after_tpfinal.sh(tpfinal)는 실행하지 않는다.
# 인자로 다른 단계를 줄 수 있다. 예: bash stage05/run_after_tpfinal_s2.sh threads_s2 (3e 후속 스레드 진단)
cd /workspace/sr-reid || exit 1
bash scripts/disk_guard.sh 20 || exit 1
ulimit -n 524288
export PYTHONPATH=/workspace/sr-reid PYTHONNOUSERSITE=1
export SRREID_CACHE=${SRREID_CACHE:-/root/srreid_cache}
/workspace/miniforge3/envs/srreid-core/bin/python stage05/run_train.py ${1:-tpfinal_s2} 2>&1 | tee -a logs/stage05/run_train.log
echo "run_3e_s2(${1:-tpfinal_s2}) 종료" | tee -a logs/stage05/run_train.log
