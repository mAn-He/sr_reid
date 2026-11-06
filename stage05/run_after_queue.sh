#!/bin/bash
# 앞선 run_train 큐가 끝나면 이어서 돈다. 스크립트 오류로 실패한 표본 프로파일 표시만 지우고 다시 잰다.
cd /workspace/sr-reid || exit 1
L=$1
until [ "$(tail -n +$((L+1)) logs/stage05/run_train.log | grep -c 'run_train 종료')" -ge 1 ]; do sleep 20; done
mkdir -p logs/stage05/script_bug_rows
for n in d1b_sample_kpr d1b_sample_kpr_memmap; do
  mv runs/stage05/done/$n.json logs/stage05/script_bug_rows/ 2>/dev/null
done
ulimit -n 524288
export PYTHONPATH=/workspace/sr-reid PYTHONNOUSERSITE=1
/workspace/miniforge3/envs/srreid-core/bin/python stage05/run_train.py profile h4 ckpt extra 2>&1 | tee -a logs/stage05/run_train.log
echo "run_after 종료" | tee -a logs/stage05/run_train.log
