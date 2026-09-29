#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1
# Kit must see the GPU connected to the display; do not isolate it away.
export WEATHERSTRIP_ALL_GPUS=1
export EFORREST_CUDA_GRAPH="${WEATHERSTRIP_CUDA_GRAPH:-${EFORREST_CUDA_GRAPH:-1}}"
physics_device="${WEATHERSTRIP_DEVICE:-cuda:0}"
render_gpu="${WEATHERSTRIP_RENDER_GPU:-0}"
exec "$root/scripts/python.sh" "$root/scripts/isaaclab_demo.py" --viz kit --device "$physics_device" \
    --kit_args "--/renderer/activeGpu=$render_gpu --/renderer/multiGpu/enabled=false --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false" "$@"
