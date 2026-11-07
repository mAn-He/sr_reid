#!/bin/bash
# Stage 1 2·3단계 일괄: manifest → 평가 모델 지표 → FFT·ΔE → 결정적 대 공식 차이 → 집계 → 그림 → 비포·애프터 → 보고서.
# 띄우기: bash scripts/run_bg.sh s1met bash stage1/run_metrics.sh [단계...]
# 각 스크립트가 끝난 집합(runs/stage1_metrics/ 의 파일, 생성된 그림)을 건너뛰므로 재시작해도 된다. 집계·보고서는 매번 다시 만든다.
cd /workspace/sr-reid || exit 1
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PQ=/workspace/miniforge3/envs/srreid-quality/bin/python
PC=/workspace/miniforge3/envs/srreid-core/bin/python
STEPS=${@:-manifest embed fft detoff aggregate plots rows report}
set -e
for s in $STEPS; do
  echo "[$(date -Is)] 시작 $s"
  case $s in
    manifest) $PC stage1/manifest.py ;;
    embed) flock /root/srreid_gpu.lock $PQ stage1/metrics_embed.py ;;
    fft) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PQ stage1/metrics_fft.py --procs 12 ;;
    detoff) flock /root/srreid_gpu.lock $PQ stage1/det_vs_official.py ;;
    aggregate) $PQ stage1/aggregate.py ;;
    plots) $PQ stage1/viz_plots.py ;;
    rows) $PQ stage1/viz_rows.py ;;
    report) $PQ stage1/report.py ;;
    *) echo "모르는 단계 $s"; exit 2 ;;
  esac
  echo "[$(date -Is)] 종료 $s"
done
