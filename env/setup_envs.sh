#!/usr/bin/env bash
# 프로젝트 전용 conda env 4개를 만든다. 여러 번 실행해도 안전하다. 기존 env(geo, sr, osediff)는 건드리지 않는다.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MF=/workspace/miniforge3
CONDA=$MF/bin/conda
export CONDA_PKGS_DIRS=/workspace/cache/conda PIP_CACHE_DIR=/workspace/cache/pip
export HF_HUB_ENABLE_HF_TRANSFER=0

for n in srreid-core srreid-realesrgan srreid-pisasr srreid-osdhuman; do
  if [ ! -x "$MF/envs/$n/bin/python" ]; then
    $CONDA env create -y -n "$n" -f "$ROOT/env/$n.yml"
  fi
done

for n in srreid-core srreid-realesrgan srreid-pisasr srreid-osdhuman; do
  $MF/envs/$n/bin/python -m pip install -q "setuptools<81"
done

# basicsr 1.4.2는 setup.py가 torch를 import하므로 --no-build-isolation이 필요하다.
PY=$MF/envs/srreid-realesrgan/bin/python
$PY -m pip install --no-build-isolation basicsr==1.4.2 facexlib==0.3.0 gfpgan==1.3.8 numpy==1.26.3 Pillow==9.5.0 opencv-python==4.8.0.74
$PY -m pip install --no-build-isolation --no-deps -e "$ROOT/third_party/Real-ESRGAN"
$MF/envs/srreid-pisasr/bin/python -m pip install --no-build-isolation basicsr==1.4.2 numpy==1.23.5 Pillow==9.5.0 opencv-python==4.8.0.74

for n in srreid-core srreid-realesrgan srreid-pisasr srreid-osdhuman; do
  { $CONDA list -n "$n" --explicit; echo "# pip freeze"; $MF/envs/$n/bin/python -m pip freeze; } > "$ROOT/env/$n.lock.txt"
done
echo "setup_envs done"
