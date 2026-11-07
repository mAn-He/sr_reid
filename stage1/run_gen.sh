#!/bin/bash
# Stage 1 1단계: 표본 2,250장 × 12종 × {×3, ×4} SR 생성 → runs/stage1_sample/<method>/x<k>/<원본파일명>.webp
# 띄우기: bash scripts/run_bg.sh s1gen bash stage1/run_gen.sh [main] [official20] [det20]
# 설정(2026-11-07 사용자 결정):
#   - osdhuman, pisa_sr(λ 5단계), mambairv2, hat: 결정적 설정(--det: math SDPA, cudnn deterministic) + 예열 1회(--warmup) + 파일명 seed
#   - swinir, realesrgan: 공식 설정(파일명 seed는 걸지만 난수를 쓰지 않음). realesrgan b4, 나머지 b1
#   - realesrgan·osdhuman·pisa_sr ×3: ×4 출력을 PIL BICUBIC으로 384×192 축소(--from-x4, 세 방법 모두 같은 커널)
# 재시작 안전: sr_gen.py가 완성 WebP를 건너뛰고, gen()은 summary.json이 있는 작업을 건너뛴다.
# GPU 작업은 flock /root/srreid_gpu.lock. CPU 작업(bicubic·lanczos·from_x4)은 락 없이 뒤에서 병렬로 돈다.
cd /workspace/sr-reid || exit 1
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 SRREID_RAW=${SRREID_RAW:-/root/srreid_raw} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
MF=/workspace/miniforge3/envs
TP=/workspace/sr-reid/third_party
OUT=/workspace/sr-reid/runs/stage1_sample
S0=/workspace/sr-reid/runs/stage1_step0
N=$(($(wc -l < configs/stage1_sample.csv) - 1))
mkdir -p logs/stage1/gen
envof() { case $1 in realesrgan) echo srreid-realesrgan;; osdhuman) echo srreid-osdhuman;; pisa_sr) echo srreid-pisasr;; swinir|hat|mambairv2) echo srreid-mambair;; *) echo srreid-core;; esac; }
repoof() { case $1 in realesrgan) echo Real-ESRGAN;; osdhuman) echo OSDHuman;; pisa_sr) echo PiSA-SR;; swinir) echo SwinIR;; hat) echo HAT;; mambairv2) echo MambaIR;; *) echo none;; esac; }
run() {  # run <lock:1|0> <out> <logname> <method> <args...>
  local lk=$1 out=$2 ln=$3 m=$4; shift 4
  if [ -f "$out/_logs/${ln}__summary.json" ]; then echo "[건너뜀] $ln"; return 0; fi
  echo "[$(date -Is)] 시작 $ln"
  local pre=""; [ "$lk" = 1 ] && pre="flock /root/srreid_gpu.lock"
  $pre env PYTHONPATH=$TP/$(repoof $m) $MF/$(envof $m)/bin/python stage1/sr_gen.py --method $m --out $out --log-name $ln "$@" > logs/stage1/gen/$ln.log 2>&1
  local rc=$?
  echo "[$(date -Is)] 종료 $ln rc=$rc"
  return $rc
}
nfiles() { ls "$1" 2>/dev/null | grep -c '\.webp$'; }
DET="--det --warmup"
JOBS=${@:-main official20 det20}
for j in $JOBS; do
  case $j in
    main)
      for m in bicubic lanczos; do for k in 3 4; do run 0 $OUT ${m}__x$k $m --k $k & done; done
      if run 1 $OUT realesrgan__x4 realesrgan --k 4 && [ "$(nfiles $OUT/realesrgan/x4)" -ge $N ]; then run 0 $OUT realesrgan__x3 realesrgan --k 3 --from-x4 & fi
      for m in hat mambairv2; do for k in 3 4; do run 1 $OUT ${m}__x$k $m --k $k $DET; done; done
      for k in 3 4; do run 1 $OUT swinir__x$k swinir --k $k; done
      if run 1 $OUT osdhuman__x4 osdhuman --k 4 $DET && [ "$(nfiles $OUT/osdhuman/x4)" -ge $N ]; then run 0 $OUT osdhuman__x3 osdhuman --k 3 --from-x4 & fi
      for s in 0.0 0.5 1.0 1.5 2.0; do
        if run 1 $OUT pisa_sr_sem${s}__x4 pisa_sr --lambda-sem $s --k 4 $DET && [ "$(nfiles $OUT/pisa_sr_sem$s/x4)" -ge $N ]; then
          run 0 $OUT pisa_sr_sem${s}__x3 pisa_sr --lambda-sem $s --k 3 --from-x4 &
        fi
      done
      wait
      ;;
    official20)  # 공식 기본 설정(--det·--warmup 없음, 파일명 seed) 20장을 두 번(A, B): 결정적 설정과의 차이 + 공식 설정의 실행 간 차이
      for r in A B; do
        O=$S0/official20/$r
        for m in hat mambairv2; do for k in 3 4; do run 1 $O ${m}__x$k $m --k $k --limit 20; done; done
        run 1 $O osdhuman__x4 osdhuman --k 4 --limit 20 && run 0 $O osdhuman__x3 osdhuman --k 3 --from-x4 --limit 20
        for s in 0.0 0.5 1.0 1.5 2.0; do run 1 $O pisa_sr_sem${s}__x4 pisa_sr --lambda-sem $s --k 4 --limit 20 && run 0 $O pisa_sr_sem${s}__x3 pisa_sr --lambda-sem $s --k 3 --from-x4 --limit 20; done
      done
      ;;
    det20)  # 본 생성과 같은 설정으로 20장을 별도 프로세스에서 다시 생성 → 본 출력과 비트 일치 확인(swinir·realesrgan 포함)
      O=$S0/det20
      for m in hat mambairv2; do for k in 3 4; do run 1 $O ${m}__x$k $m --k $k --limit 20 $DET; done; done
      for k in 3 4; do run 1 $O swinir__x$k swinir --k $k --limit 20; done
      run 1 $O realesrgan__x4 realesrgan --k 4 --limit 20
      run 1 $O osdhuman__x4 osdhuman --k 4 --limit 20 $DET
      for s in 0.0 0.5 1.0 1.5 2.0; do run 1 $O pisa_sr_sem${s}__x4 pisa_sr --lambda-sem $s --k 4 --limit 20 $DET; done
      ;;
    *) echo "모르는 작업 $j"; exit 2;;
  esac
done
echo "[$(date -Is)] 전체 끝"
