#!/bin/bash
# Stage 1 A100 서버 본 실행(2026-11-08): 1단계 SR 생성 + 방법별 평가 모델 지표를 GPU 2장에 나눠 동시에 돌린다.
# 띄우기: P=<GPU당 동시 프로세스 수> bash scripts/run_bg.sh s1a100 bash stage1/run_a100.sh [main] [check20] [post]
#
# 설정(2026-11-07 결정 그대로, 2026-11-08 가속 방침 반영):
#   - osdhuman, pisa_sr(λ 5단계), mambairv2, hat: --det --warmup + 파일명 seed. swinir, realesrgan: 공식 설정.
#   - 배치 정책 (가): realesrgan b4, 나머지 b1(배치를 키우지 않는다). 대신 GPU당 P개 프로세스를 동시에 띄운다.
#   - realesrgan·osdhuman·pisa_sr ×3 = ×4 출력을 PIL BICUBIC으로 384×192 축소(--from-x4).
#   - 작업 단위 = 방법 하나(pisa_sr는 λ별). 한 방법의 ×3·×4 생성과 그 방법의 지표(임베딩·A-FINE)는 같은 GPU에서 끝까지.
#     GPU별 작업 목록은 아래 Q0·Q1(고정 배정, GPU 번호는 summary.json의 cuda_visible_devices와 logs/stage1/a100/assign.csv에 기록).
#   - 스레드: 프로세스마다 OMP/MKL 4. 동시 프로세스 최대 2P개 × 4 ≤ cgroup CPU 54.4코어.
# 재시작 안전: 생성은 summary.json이 있으면 건너뛰고(sr_gen.py도 완성 WebP를 건너뜀), 지표는 있는 파일을 건너뛴다.
cd /workspace/sr-reid || exit 1
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 SRREID_RAW=${SRREID_RAW:-/root/srreid_raw} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export SRREID_SERVER="A100 서버(A100 SXM4 80GB ×2, EPYC 7713, cgroup v2)"
P=${P:-1}
export SRREID_CONC_NOTE="병렬 실행 중 측정: GPU 2장 × GPU당 ${P}개 프로세스 동시 실행"
MF=/workspace/miniforge3/envs
TP=/workspace/sr-reid/third_party
OUT=/workspace/sr-reid/runs/stage1_sample
S0=/workspace/sr-reid/runs/stage1_step0
PQ=$MF/srreid-quality/bin/python
N=$(($(wc -l < configs/stage1_sample.csv) - 1))
L=logs/stage1/a100
mkdir -p $L/gen $L/met $L/q
envof() { case $1 in realesrgan) echo srreid-realesrgan;; osdhuman) echo srreid-osdhuman;; pisa_sr) echo srreid-pisasr;; swinir|hat|mambairv2) echo srreid-mambair;; *) echo srreid-core;; esac; }
repoof() { case $1 in realesrgan) echo Real-ESRGAN;; osdhuman) echo OSDHuman;; pisa_sr) echo PiSA-SR;; swinir) echo SwinIR;; hat) echo HAT;; mambairv2) echo MambaIR;; *) echo none;; esac; }
log() { echo "[$(date -Is)] $*"; }
gen() {  # gen <gpu> <out> <logname> <method> <args...>
  local g=$1 out=$2 ln=$3 m=$4; shift 4
  if [ -f "$out/_logs/${ln}__summary.json" ]; then log "[건너뜀] $ln"; return 0; fi
  log "시작 $ln gpu$g"
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$TP/$(repoof $m) $MF/$(envof $m)/bin/python stage1/sr_gen.py --method $m --out $out --log-name $ln "$@" > $L/gen/$ln.log 2>&1
  local rc=$?; log "종료 $ln rc=$rc"; return $rc
}
nfiles() { ls "$1" 2>/dev/null | grep -c '\.webp$'; }
met() {  # met <gpu> <logname> <metrics_embed 인자...>
  local g=$1 ln=$2; shift 2
  log "시작 met_$ln gpu$g"
  CUDA_VISIBLE_DEVICES=$g $PQ stage1/metrics_embed.py "$@" > $L/met/$ln.log 2>&1
  local rc=$?; log "종료 met_$ln rc=$rc"; return $rc
}
DET="--det --warmup"
job() {  # job <gpu> <작업 이름>: 방법 하나의 ×4·×3 생성 + 그 방법의 지표
  local g=$1 j=$2 m s
  echo "$j,$g" >> $L/assign.csv
  case $j in
    base) met $g base --methods none ;;
    classic)
      for m in bicubic lanczos; do for k in 3 4; do gen $g $OUT ${m}__x$k $m --k $k || return 1; done; done
      met $g classic --methods bicubic lanczos --no-base ;;
    realesrgan|osdhuman)
      local d=""; [ $j = osdhuman ] && d="$DET"
      gen $g $OUT ${j}__x4 $j --k 4 $d || return 1
      [ "$(nfiles $OUT/$j/x4)" -ge $N ] || { log "$j x4 미완"; return 1; }
      gen $g $OUT ${j}__x3 $j --k 3 --from-x4 || return 1
      met $g $j --methods $j --no-base ;;
    pisa_sr_sem*)
      s=${j#pisa_sr_sem}
      gen $g $OUT ${j}__x4 pisa_sr --lambda-sem $s --k 4 $DET || return 1
      [ "$(nfiles $OUT/$j/x4)" -ge $N ] || { log "$j x4 미완"; return 1; }
      gen $g $OUT ${j}__x3 pisa_sr --lambda-sem $s --k 3 --from-x4 || return 1
      met $g $j --methods $j --no-base ;;
    hat|mambairv2|swinir)
      local d="$DET"; [ $j = swinir ] && d=""
      for k in 3 4; do gen $g $OUT ${j}__x$k $j --k $k $d || return 1; done
      met $g $j --methods $j --no-base ;;
    *) log "모르는 작업 $j"; return 2 ;;
  esac
}
worker() {  # worker <gpu> <큐 파일>: 큐에서 작업을 하나씩 꺼내 실행(같은 GPU의 P개 worker가 큐를 나눠 씀)
  local g=$1 q=$2 j rc=0
  while true; do
    j=$(flock $q.lock bash -c "head -n1 $q; sed -i 1d $q")
    [ -z "$j" ] && break
    job $g $j || { log "실패 $j gpu$g"; rc=1; }
  done
  return $rc
}
# GPU별 고정 배정(A100 단독 s/장 기준 GPU당 합이 비슷하게: pisa λ 하나 ≈ 10분, osdhuman ≈ 9분, mambairv2 ≈ 12분, hat·swinir ≈ 6분)
# 무거운 작업을 앞에 둔다.
Q0="pisa_sr_sem0.0 pisa_sr_sem1.0 pisa_sr_sem2.0 hat swinir realesrgan classic"
Q1="pisa_sr_sem0.5 pisa_sr_sem1.5 osdhuman mambairv2 base"
fail=0
for J in ${@:-main check20 post}; do
  case $J in
    main)
      bash scripts/disk_guard.sh 20 || exit 1
      cp results/stage1_4090/baseline_othercam_pairs.csv results/stage1/ 2>/dev/null  # 표본·데이터만으로 정해지는 결정적 목록(하드웨어 무관)
      echo "job,gpu" > $L/assign.csv
      echo $Q0 | tr ' ' '\n' > $L/q/q0; echo $Q1 | tr ' ' '\n' > $L/q/q1
      PIDS=""
      for p in $(seq 1 $P); do worker 0 $L/q/q0 & PIDS="$PIDS $!"; worker 1 $L/q/q1 & PIDS="$PIDS $!"; done
      for pid in $PIDS; do wait $pid || fail=1; done
      log "main 끝 fail=$fail"
      [ $fail = 0 ] || exit 1
      ;;
    check20)  # 본 생성 뒤 GPU당 한 프로세스씩(단독): GPU1 = det20(본 설정 20장 재생성 → 동시 실행 본 출력과 비트 비교), GPU0 = official20 A→B(공식 설정)
      export SRREID_CONC_NOTE="단독(GPU당 1개 프로세스, 다른 GPU에서 다른 작업 동시 실행)"
      (
        O=$S0/det20
        for m in hat mambairv2; do for k in 3 4; do gen 1 $O ${m}__x$k $m --k $k --limit 20 $DET; done; done
        for k in 3 4; do gen 1 $O swinir__x$k swinir --k $k --limit 20; done
        gen 1 $O realesrgan__x4 realesrgan --k 4 --limit 20
        gen 1 $O osdhuman__x4 osdhuman --k 4 --limit 20 $DET
        for s in 0.0 0.5 1.0 1.5 2.0; do gen 1 $O pisa_sr_sem${s}__x4 pisa_sr --lambda-sem $s --k 4 --limit 20 $DET; done
      ) &
      D=$!
      for r in A B; do
        O=$S0/official20/$r
        for m in hat mambairv2; do for k in 3 4; do gen 0 $O ${m}__x$k $m --k $k --limit 20; done; done
        gen 0 $O osdhuman__x4 osdhuman --k 4 --limit 20 && gen 0 $O osdhuman__x3 osdhuman --k 3 --from-x4 --limit 20
        for s in 0.0 0.5 1.0 1.5 2.0; do gen 0 $O pisa_sr_sem${s}__x4 pisa_sr --lambda-sem $s --k 4 --limit 20 && gen 0 $O pisa_sr_sem${s}__x3 pisa_sr --lambda-sem $s --k 3 --from-x4 --limit 20; done
      done
      wait $D
      ;;
    post)
      export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
      PC=$MF/srreid-core/bin/python
      set -e
      log "시작 manifest"; $PC stage1/manifest.py > $L/manifest.log 2>&1; log "종료 manifest"
      log "시작 embed 잔여"; CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 $PQ stage1/metrics_embed.py > $L/met/rest.log 2>&1; log "종료 embed 잔여"
      log "시작 fft"; $PQ stage1/metrics_fft.py --procs 48 > $L/fft.log 2>&1; log "종료 fft"
      log "시작 detoff"; CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 $PQ stage1/det_vs_official.py > $L/detoff.log 2>&1; log "종료 detoff"
      export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
      for s in aggregate viz_plots viz_rows report; do log "시작 $s"; CUDA_VISIBLE_DEVICES=0 $PQ stage1/$s.py > $L/$s.log 2>&1; log "종료 $s"; done
      ;;
    *) log "모르는 단계 $J"; exit 2 ;;
  esac
done
log "전체 끝"
