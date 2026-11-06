#!/usr/bin/env bash
# Stage 0.5 전용 venv 2개. 볼륨이 쿼터에 가까워(2026-11-01 초과 사건) torch를 다시 깔지 않고 기존 env를 상속한다. 기존 env는 건드리지 않는다.
# srreid-mambair: SwinIR, HAT, MambaIRv2 추론. srreid-realesrgan(torch 2.1.2+cu118, basicsr 1.4.2) 상속. MambaIR README가 시험한 조합은 Python 3.9·torch 2.0.1+cu117로 다르다.
# srreid-quality: Part G 지표. srreid-core(torch 2.1.2+cu118) 상속.
set -euo pipefail
E=/workspace/miniforge3/envs
export PIP_CACHE_DIR=/workspace/cache/pip PYTHONNOUSERSITE=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ ! -x "$E/srreid-mambair/bin/python" ]; then
  "$E/srreid-realesrgan/bin/python" -m venv --system-site-packages "$E/srreid-mambair"
  "$E/srreid-mambair/bin/pip" install --no-build-isolation --no-deps causal-conv1d==1.0.0 mamba-ssm==1.0.1 einops==0.7.0
  "$E/srreid-mambair/bin/pip" install --no-deps transformers==4.37.1 tokenizers==0.15.2 safetensors regex packaging filelock huggingface_hub==0.20.3 fsspec timm==0.9.12
fi
"$E/srreid-mambair/bin/pip" freeze --local > "$ROOT/env/srreid-mambair.lock.txt"
if [ ! -x "$E/srreid-quality/bin/python" ]; then
  "$E/srreid-core/bin/python" -m venv --system-site-packages "$E/srreid-quality"
  "$E/srreid-quality/bin/pip" install "numpy<2" dreamsim "transformers==4.37.1" "tokenizers==0.15.2" "huggingface_hub==0.20.3" "accelerate==0.25.0"
  "$E/srreid-quality/bin/pip" install --no-deps "peft==0.11.1"
fi
"$E/srreid-quality/bin/pip" freeze --local > "$ROOT/env/srreid-quality.lock.txt"
echo "setup_envs_stage05 done"
