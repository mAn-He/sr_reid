#!/usr/bin/env bash
# Stage 0 전용 conda env 2개. 여러 번 실행해도 안전하다. 기존 env는 건드리지 않는다.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MF=/workspace/miniforge3
CONDA=$MF/bin/conda
export CONDA_PKGS_DIRS=/workspace/cache/conda PIP_CACHE_DIR=/workspace/cache/pip
for n in srreid-kpr srreid-clipreid; do
  [ -x "$MF/envs/$n/bin/python" ] || $CONDA env create -y -n "$n" -f "$ROOT/env/$n.yml"
  { $CONDA list -n "$n" --explicit; echo "# pip freeze"; $MF/envs/$n/bin/python -m pip freeze; } > "$ROOT/env/$n.lock.txt"
done
echo "setup_envs_stage0 done"
