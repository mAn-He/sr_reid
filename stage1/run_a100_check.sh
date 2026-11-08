#!/bin/bash
# A100 서버 Stage 1 사전 확인(2026-11-08).
#  repro: hat·mambairv2(×3·×4), osdhuman ×4, pisa_sr λ 5단계 ×4를 결정적 설정 + 예열 + 파일명 seed로 20장, 별도 프로세스 2회
#         (정방향 = GPU0, 역순 = GPU1, 동시에 각 GPU에서 한 프로세스씩) → runs/stage1_step0/repro/
#  bench: 50장, GPU당 동시 프로세스 N = 1·2·3·4 (두 GPU 동시, 프로세스 = pisa_sr λ 하나, 배치 1, 결정적 설정). 모델 적재·예열 뒤 같은 시각에 시작(--start-at).
#         출력은 /root/srreid_bench(로컬 NVMe, 재생성 가능)에 WebP로 실제 저장해 본 생성과 같은 CPU 부하를 준다. 기록 runs/stage1_step0/bench/
# 띄우기: bash scripts/run_bg.sh a100chk bash stage1/run_a100_check.sh [repro] [bench]
cd /workspace/sr-reid || exit 1
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 SRREID_RAW=${SRREID_RAW:-/root/srreid_raw} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
MF=/workspace/miniforge3/envs; TP=/workspace/sr-reid/third_party; S0=/workspace/sr-reid/runs/stage1_step0
envof() { case $1 in realesrgan) echo srreid-realesrgan;; osdhuman) echo srreid-osdhuman;; pisa_sr) echo srreid-pisasr;; swinir|hat|mambairv2) echo srreid-mambair;; *) echo srreid-core;; esac; }
repoof() { case $1 in realesrgan) echo Real-ESRGAN;; osdhuman) echo OSDHuman;; pisa_sr) echo PiSA-SR;; swinir) echo SwinIR;; hat) echo HAT;; mambairv2) echo MambaIR;; esac; }
gen() {  # gen <gpu> <out> <logname> <method> <args...>
  local g=$1 out=$2 ln=$3 m=$4; shift 4
  if [ -f "$out/_logs/${ln}__summary.json" ]; then echo "[건너뜀] $ln"; return 0; fi
  mkdir -p logs/stage1/check; echo "[$(date -Is)] 시작 $ln gpu$g"
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$TP/$(repoof $m) $MF/$(envof $m)/bin/python stage1/sr_gen.py --method $m --out $out --log-name $ln "$@" > logs/stage1/check/$ln.log 2>&1
  local rc=$?; echo "[$(date -Is)] 종료 $ln rc=$rc"; return $rc
}
lane() {  # lane <gpu> <fwd|rev>
  local g=$1 r=$2 R=""; [ $r = rev ] && R="--reverse"; local O=$S0/repro
  for m in hat mambairv2; do for k in 3 4; do gen $g $O ${m}__x${k}__detw__$r $m --k $k --limit 20 --no-write --det --warmup $R; done; done
  gen $g $O osdhuman__x4__detw__$r osdhuman --k 4 --limit 20 --no-write --det --warmup $R
  for s in 0.0 0.5 1.0 1.5 2.0; do gen $g $O pisa_sr_sem${s}__x4__detw__$r pisa_sr --lambda-sem $s --k 4 --limit 20 --no-write --det --warmup $R; done
}
LAMS=(0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0)
for j in ${@:-repro bench}; do
  case $j in
    repro) lane 0 fwd & lane 1 rev & wait ;;
    bench)
      for N in 1 2 3 4; do
        O=$S0/bench/n$N; [ -f $O/done ] && { echo "[건너뜀] bench n$N"; continue; }
        rm -rf /root/srreid_bench/n$N; T=$(( $(date +%s) + 240 ))   # 적재(최대 8개 동시) 여유 240초
        mkdir -p $O; nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 2 > $O/nvsmi.csv & SMI=$!
        top -b -d 2 -n 200 | grep --line-buffered '^%Cpu' > $O/cpu_top.txt & TOP=$!
        i=0; PIDS=""
        for g in 0 1; do for p in $(seq 1 $N); do
          s=${LAMS[$i]}; i=$((i+1))
          SRREID_CONC_NOTE="bench N=$N per GPU" gen $g /root/srreid_bench/n$N/g${g}p$p bench_n${N}_g${g}_p$p pisa_sr --lambda-sem $s --k 4 --limit 50 --det --warmup --start-at $T &
          PIDS="$PIDS $!"
        done; done
        wait $PIDS
        kill $SMI $TOP; cp /root/srreid_bench/n$N/g*/_logs/* $O/; touch $O/done
      done ;;
    *) echo "모르는 작업 $j"; exit 2 ;;
  esac
done
echo "[$(date -Is)] 끝"
