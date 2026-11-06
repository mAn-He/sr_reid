#!/bin/bash
# Part F 계획용 SR 스모크(새 서버). train 100장만 runs/stage05/sr_smoke에 만든다. 일괄 생성이 아니다(data/에 쓰지 않음).
# 방법별 GPU 시간과 WebP 재측정용 실제 SR 이미지를 얻는다. 재실행하면 끝난 파일은 건너뛴다.
# 띄우기: bash scripts/run_bg.sh srsmoke bash stage05/run_sr_smoke_s2.sh
cd /workspace/sr-reid || exit 1
export PYTHONPATH=/workspace/sr-reid PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export SRREID_RAW=${SRREID_RAW:-/root/srreid_raw}
/workspace/miniforge3/envs/srreid-core/bin/python stage05/sr_generate.py --smoke 100 --splits bounding_box_train --k 4 3
