#!/bin/bash
# 체인·큰 작업 시작 전 볼륨 남은 용량 검사. 2026-11-01 쿼터 초과로 프로세스가 죽은 사건 이후 추가. 남은 용량이 기준 미만이면 exit 1.
# df는 쿼터가 아니라 클러스터 전체를 보여 주므로 du로 잰다. 잠긴 폴더(data/locked)는 du가 못 읽어 값을 더한다.
# 볼륨 한도: configs/volume_limit_gib 한 줄 (기본 186 = 200GB). 볼륨을 키우면 이 파일을 고친다.
ROOT=/workspace/sr-reid
MIN_GIB=${1:-20}
LOCKED_GIB=6
LIMIT=$(cat "$ROOT/configs/volume_limit_gib" 2>/dev/null || echo 186)
USED=$(du -s --block-size=1G /workspace 2>/dev/null | cut -f1)
FREE=$((LIMIT - USED - LOCKED_GIB))
MSG="$(date '+%F %T') 볼륨 한도 ${LIMIT}GiB, 사용 $((USED + LOCKED_GIB))GiB, 남음 ${FREE}GiB, 기준 ${MIN_GIB}GiB"
if [ "$FREE" -lt "$MIN_GIB" ]; then
  echo "$MSG → 시작 안 함" | tee -a /root/disk_guard.log
  exit 1
fi
echo "$MSG → 통과" | tee -a /root/disk_guard.log
