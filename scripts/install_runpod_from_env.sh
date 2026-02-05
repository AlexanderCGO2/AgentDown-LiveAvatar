#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "${ROOT_DIR}/.env" ]]; then
  echo "ERROR: ${ROOT_DIR}/.env not found. Create it first."
  exit 1
fi

set -a
source "${ROOT_DIR}/.env"
set +a

missing=0
for key in ELEVENLABS_API_KEY OPENAI_API_KEY REPLICATE_API_TOKEN HF_TOKEN; do
  if [[ -z "${!key:-}" ]]; then
    echo "ERROR: ${key} missing in .env"
    missing=1
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  echo "Fix missing keys in .env and re-run."
  exit 1
fi

echo "✅ Keys loaded from .env (not printing secrets)"

apt-get update && apt-get install -y ffmpeg git-lfs unzip

export TMPDIR=/workspace/tmp
mkdir -p "${TMPDIR}"

python -m pip install --upgrade pip
pip install \
  python-dotenv huggingface_hub einops protobuf tiktoken albumentations \
  pydub librosa soundfile imageio imageio-ffmpeg opencv-python \
  elevenlabs openai pydantic aiohttp dashscope peft hjson ninja py-cpuinfo

mkdir -p "${ROOT_DIR}/models"

python - <<'PY'
from huggingface_hub import snapshot_download
import os

print("Downloading Wan2.2-S2V-14B (~28GB)...")
snapshot_download(
    "Wan-AI/Wan2.2-S2V-14B",
    local_dir="models/Wan2.2-S2V-14B",
    token=os.environ.get("HF_TOKEN"),
)

print("Downloading LiveAvatar LoRA (~1GB)...")
snapshot_download(
    "Quark-Vision/Live-Avatar",
    local_dir="models/LiveAvatar",
    token=os.environ.get("HF_TOKEN"),
)

print("✅ Models downloaded")
PY

python - <<'PY'
import requests

url = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.4cxx11abiFALSE-"
    "cp311-cp311-linux_x86_64.whl"
)

print("Downloading flash-attn wheel...")
r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
r.raise_for_status()
with open("/workspace/flash_attn.whl", "wb") as f:
    f.write(r.content)
print("✅ flash-attn wheel downloaded")
PY

mv /workspace/flash_attn.whl "/workspace/flash_attn-2.5.9.post1+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
pip install "/workspace/flash_attn-2.5.9.post1+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${GPU_COUNT}" -gt 0 ]]; then
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPU_COUNT - 1)))"
fi

export PYTHONPATH="${ROOT_DIR}/liveavatar_lib:${PYTHONPATH:-}"

echo "✅ Install complete"
echo "Run: python run.py segment --type HUMAN_WATCH --topic \"Quick test\" --duration 15"
