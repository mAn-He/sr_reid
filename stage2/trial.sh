#!/bin/bash
# Stage 2 학습 동시 실행 시험(200 iteration). 결과 runs/stage2/trial/<round>/<job>/trial.json, 자원 기록 nvsmi.csv·cpu.csv.
# 단독(solo)도 다른 GPU의 시험과 동시에 돌았으므로 CPU는 공유 상태다(병렬 실행 중 측정).
cd /workspace/sr-reid || exit 1
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 SRREID_CACHE=/dev/shm/srreid_cache WANDB_MODE=disabled
MF=/workspace/miniforge3/envs; T=runs/stage2/trial
py() { case $1 in bot) echo $MF/srreid-core/bin/python;; clipreid) echo $MF/srreid-clipreid/bin/python;; *) echo $MF/srreid-kpr/bin/python;; esac; }
job() {  # job <round> <gpu> <model> <k> <iters> <ckpt> <delay>
  local r=$1 g=$2 m=$3 k=$4 n=$5 c=$6 d=$7 o=$T/$1/g$2_$3_x$4_c$6
  [ -f $o/trial.json ] && return 0
  sleep $d; mkdir -p $o
  case $m in
    bot) a="stage2/train_bot.py --cond bicubic --k $k" ;;
    clipreid) a="stage2/train_clipreid.py --cond bicubic --k $k" ;;
    *) a="stage2/train_kpr.py --model $m --cond bicubic --k $k" ;;
  esac
  CUDA_VISIBLE_DEVICES=$g setsid $(py $m) $a --out $o --max-iters $n --ckpt $c > $o/log.txt 2>&1
  echo "[$(date -Is)] $r g$g $m x$k c$c rc=$?"
}
mon() { nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 5 > $T/$1/nvsmi.csv & echo $! > $T/$1/.smi
  ( while true; do echo "$(date +%s) $(grep usage_usec /sys/fs/cgroup/cpu.stat | awk '{print $2}')"; sleep 5; done ) > $T/$1/cpu.csv & echo $! > $T/$1/.cpu; }
unmon() { kill $(cat $T/$1/.smi) $(cat $T/$1/.cpu) 2>/dev/null; }
round() { local r=$1; shift; mkdir -p $T/$r; [ -f $T/$r/done ] && return; "$@"; local jp=$(jobs -p); mon $r; wait $jp; unmon $r; touch $T/$r/done; }
r1() { job r1 0 kpr_solider 4 200 0 0 & job r1 1 kpr_solider 4 200 1 0 & }
r2() { job r2 0 kpr 3 200 0 0 & ( job r2 1 bot 4 300 0 0; job r2 1 bot 3 300 0 0; job r2 1 clipreid 4 200 0 0; job r2 1 clipreid 3 200 0 0 ) & }
r3() { job r3 0 kpr_solider 4 200 1 0 & job r3 0 bot 4 600 0 170 & job r3 0 clipreid 4 400 0 170 &
       job r3 1 kpr 3 200 0 0 & job r3 1 bot 3 700 0 170 & job r3 1 clipreid 3 450 0 170 & }
r4() { job r4 0 kpr_solider 4 200 1 0 & job r4 0 bot 4 600 0 170 &
       job r4 1 bot 4 600 0 0 & job r4 1 clipreid 4 400 0 0 & job r4 1 clipreid 3 450 0 0 & }
for r in ${@:-r1 r2 r3 r4}; do round $r $r; echo "[$(date -Is)] $r 끝"; done
