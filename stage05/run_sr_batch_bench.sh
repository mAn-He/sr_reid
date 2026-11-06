#!/bin/bash
# Stage 1 준비: 학습형 SR 배치 추론 측정. GPU 작업을 하나씩 차례로 돌린다(동시 실행 없음).
# 띄우기: bash scripts/run_bg.sh srbatch bash stage05/run_sr_batch_bench.sh [작업...]
# 끝난 작업(logs/stage1/sr_batch_bench/<tag>.json 존재)은 건너뛴다. 마지막에 results/stage1_sr_batch_bench.csv를 다시 만든다.
# diag_* 작업은 공식 수치가 아닌 원인 진단용이다(파일 이름에 __diag, CSV에서 빠짐).
cd /workspace/sr-reid || exit 1
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 SRREID_RAW=${SRREID_RAW:-/root/srreid_raw}
MF=/workspace/miniforge3/envs
TP=/workspace/sr-reid/third_party
O=logs/stage1/sr_batch_bench
L=logs/stage1
run() {  # run <env> <repo> <tag> <args...>   (진단 변수는 호출 앞에 env로)
  local env=$1 repo=$2 tag=$3; shift 3
  if [ -f "$O/$tag.json" ]; then echo "[건너뜀] $tag"; return 0; fi
  echo "[$(date -Is)] 시작 $tag"
  PYTHONPATH=$TP/$repo $MF/$env/bin/python stage05/sr_batch_bench.py "$@" > "$L/sr_batch_bench__$tag.log" 2>&1
  echo "[$(date -Is)] 종료 $tag rc=$?"
}
ALL="realesrgan swinir3 swinir4 hat3 hat4 mamba3 mamba4 mamba3r mamba4r osd_global osd_replay pisa_global pisa_replay pisa_anchors diag_mamba diag_osd diag_pisa"
JOBS=${@:-$ALL}
for j in $JOBS; do
  case $j in
    realesrgan) run srreid-realesrgan Real-ESRGAN realesrgan__x4 --method realesrgan --k 4;;
    swinir3) run srreid-mambair SwinIR swinir__x3 --method swinir --k 3;;
    swinir4) run srreid-mambair SwinIR swinir__x4 --method swinir --k 4;;
    hat3) run srreid-mambair HAT hat__x3 --method hat --k 3;;
    hat4) run srreid-mambair HAT hat__x4 --method hat --k 4;;
    mamba3) run srreid-mambair MambaIR mambairv2__x3__noise-global --method mambairv2 --k 3 --noise global;;
    mamba4) run srreid-mambair MambaIR mambairv2__x4__noise-global --method mambairv2 --k 4 --noise global;;
    mamba3r) run srreid-mambair MambaIR mambairv2__x3__noise-replay --method mambairv2 --k 3 --noise replay;;
    mamba4r) run srreid-mambair MambaIR mambairv2__x4__noise-replay --method mambairv2 --k 4 --noise replay;;
    osd_global) run srreid-osdhuman OSDHuman osdhuman__x4__noise-global --method osdhuman --k 4 --noise global;;
    osd_replay) run srreid-osdhuman OSDHuman osdhuman__x4__noise-replay --method osdhuman --k 4 --noise replay;;
    pisa_global) run srreid-pisasr PiSA-SR pisa_sr_sem1.0__x4__noise-global --method pisa_sr --lambda-sem 1.0 --k 4 --noise global;;
    pisa_replay) run srreid-pisasr PiSA-SR pisa_sr_sem1.0__x4__noise-replay --method pisa_sr --lambda-sem 1.0 --k 4 --noise replay;;
    pisa_anchors) for s in 0.0 0.5 1.5 2.0; do run srreid-pisasr PiSA-SR pisa_sr_sem${s}__x4__noise-global__anchor --method pisa_sr --lambda-sem $s --k 4 --noise global --anchor-only; done;;
    diag_mamba)  # 공식 대비 ±1 원인: 프로세스 간 재현성(cudnn.benchmark 켬 2회, 끔 1회)
      SRREID_DIAG_TAG=__diag_rep1 run srreid-mambair MambaIR mambairv2__x4__noise-global__anchor__diag_rep1 --method mambairv2 --k 4 --noise global --anchor-only
      SRREID_DIAG_TAG=__diag_rep2 run srreid-mambair MambaIR mambairv2__x4__noise-global__anchor__diag_rep2 --method mambairv2 --k 4 --noise global --anchor-only
      SRREID_DIAG_TAG=__diag_nobench SRREID_DIAG_CUDNN_BENCH=0 run srreid-mambair MambaIR mambairv2__x4__noise-global__anchor__diag_nobench --method mambairv2 --k 4 --noise global --anchor-only;;
    diag_osd)  # mem-efficient SDPA를 끄면(flash) 배치 1끼리, 배치 간(replay) 비트 일치하는지
      SRREID_DIAG_TAG=__diag_nomem SRREID_DIAG_SDPA=nomem run srreid-osdhuman OSDHuman osdhuman__x4__noise-replay__diag_nomem --method osdhuman --k 4 --noise replay --batches 1 2 4 8 16;;
    diag_pisa)
      SRREID_DIAG_TAG=__diag_nomem SRREID_DIAG_SDPA=nomem run srreid-pisasr PiSA-SR pisa_sr_sem1.0__x4__noise-replay__diag_nomem --method pisa_sr --lambda-sem 1.0 --k 4 --noise replay --batches 1 2 4 8 16;;
    *) echo "모르는 작업 $j"; exit 2;;
  esac
done
/workspace/miniforge3/envs/srreid-core/bin/python stage05/sr_batch_bench.py --collect
