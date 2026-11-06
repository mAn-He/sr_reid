#!/bin/bash
# Stage 1 0단계 SR 쪽: (A) 24조합 로드·추론 확인(표본 5장), (B) 재현성(20장, 별도 프로세스 2회: 정방향/역순).
# 띄우기: bash scripts/run_bg.sh s1step0 bash stage1/run_step0.sh [load] [repro]
# GPU 작업은 flock /root/srreid_gpu.lock으로 평가 모델 확인 작업과 겹치지 않게 한다. 끝난 작업(summary.json 존재)은 건너뛴다.
cd /workspace/sr-reid || exit 1
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 SRREID_RAW=${SRREID_RAW:-/root/srreid_raw} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
MF=/workspace/miniforge3/envs
TP=/workspace/sr-reid/third_party
S0=/workspace/sr-reid/runs/stage1_step0
envof() { case $1 in realesrgan) echo srreid-realesrgan;; osdhuman) echo srreid-osdhuman;; pisa_sr) echo srreid-pisasr;; swinir|hat|mambairv2) echo srreid-mambair;; *) echo srreid-core;; esac; }
repoof() { case $1 in realesrgan) echo Real-ESRGAN;; osdhuman) echo OSDHuman;; pisa_sr) echo PiSA-SR;; swinir) echo SwinIR;; hat) echo HAT;; mambairv2) echo MambaIR;; *) echo none;; esac; }
gen() {  # gen <out> <logname> <method> <args...>
  local out=$1 ln=$2 m=$3; shift 3
  if [ -f "$out/_logs/${ln}__summary.json" ]; then echo "[건너뜀] $ln"; return 0; fi
  echo "[$(date -Is)] 시작 $ln"
  mkdir -p logs/stage1/step0
  flock /root/srreid_gpu.lock env PYTHONPATH=$TP/$(repoof $m) $MF/$(envof $m)/bin/python stage1/sr_gen.py --method $m --out $out --log-name $ln "$@" > logs/stage1/step0/$ln.log 2>&1
  local rc=$?
  echo "[$(date -Is)] 종료 $ln rc=$rc"
}
JOBS=${@:-load repro}
for j in $JOBS; do
  case $j in
    load)
      O=$S0/load
      for m in bicubic lanczos; do for k in 3 4; do gen $O ${m}__x$k $m --k $k --limit 5; done; done
      for m in swinir hat mambairv2; do for k in 3 4; do gen $O ${m}__x$k $m --k $k --limit 5; done; done
      gen $O realesrgan__x4 realesrgan --k 4 --limit 5; gen $O realesrgan__x3 realesrgan --k 3 --from-x4 --limit 5
      gen $O osdhuman__x4 osdhuman --k 4 --limit 5; gen $O osdhuman__x3 osdhuman --k 3 --from-x4 --limit 5
      for s in 0.0 0.5 1.0 1.5 2.0; do gen $O pisa_sr_sem${s}__x4 pisa_sr --lambda-sem $s --k 4 --limit 5; gen $O pisa_sr_sem${s}__x3 pisa_sr --lambda-sem $s --k 3 --from-x4 --limit 5; done
      ;;
    repro)
      O=$S0/repro
      for cfg in off det; do
        D=""; [ $cfg = det ] && D="--det"
        for r in fwd rev; do
          R=""; [ $r = rev ] && R="--reverse"
          for k in 3 4; do gen $O mambairv2__x${k}__${cfg}__$r mambairv2 --k $k --limit 20 --no-write $D $R; done
          gen $O hat__x4__${cfg}__$r hat --k 4 --limit 20 --no-write $D $R
          gen $O osdhuman__x4__${cfg}__$r osdhuman --k 4 --limit 20 --no-write $D $R
          for s in 1.0 2.0; do gen $O pisa_sr_sem${s}__x4__${cfg}__$r pisa_sr --lambda-sem $s --k 4 --limit 20 --no-write $D $R; done
        done
      done
      ;;
    repro2)  # pisa_sr 첫 추론 차이 확인: --warmup으로 det 정·역순, det-strict 정·역순, osdhuman·hat도 warmup det로 재확인
      O=$S0/repro
      for r in fwd rev; do
        R=""; [ $r = rev ] && R="--reverse"
        for s in 1.0 2.0; do gen $O pisa_sr_sem${s}__x4__detw__$r pisa_sr --lambda-sem $s --k 4 --limit 20 --no-write --det --warmup $R; done
        CUBLAS_WORKSPACE_CONFIG=:4096:8 gen $O pisa_sr_sem1.0__x4__strictw__$r pisa_sr --lambda-sem 1.0 --k 4 --limit 20 --no-write --det-strict --warmup $R
        gen $O osdhuman__x4__detw__$r osdhuman --k 4 --limit 20 --no-write --det --warmup $R
        gen $O pisa_sr_sem1.0__x4__offw__$r pisa_sr --lambda-sem 1.0 --k 4 --limit 20 --no-write --warmup $R
      done
      ;;
    x3cmp)  # 확산 2종 ×3 방식 비교, 표본 앞 100장. STEP0_DET=1이면 결정적 설정(--det)
      O=$S0/x3cmp; D=""; [ "${STEP0_DET:-0}" = 1 ] && D="--det --warmup"
      gen $O osdhuman__x4 osdhuman --k 4 --limit 100 $D
      gen $O osdhuman__x3 osdhuman --k 3 --from-x4 --limit 100
      gen $O osdhuman__x3_direct osdhuman --k 3 --upscale-direct --tag direct --limit 100 $D
      for s in 0.0 0.5 1.0 1.5 2.0; do
        gen $O pisa_sr_sem${s}__x4 pisa_sr --lambda-sem $s --k 4 --limit 100 $D
        gen $O pisa_sr_sem${s}__x3 pisa_sr --lambda-sem $s --k 3 --from-x4 --limit 100
        gen $O pisa_sr_sem${s}__x3_direct pisa_sr --lambda-sem $s --k 3 --upscale-direct --tag direct --limit 100 $D
      done
      for m in bicubic; do gen $O ${m}__x3 $m --k 3 --limit 100; done
      ;;
    *) echo "모르는 작업 $j"; exit 2;;
  esac
done
