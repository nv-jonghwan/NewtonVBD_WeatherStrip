#!/usr/bin/env bash
# Explicit interpreter > local venv > optional workstation binding > active Python.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${WEATHERSTRIP_PYTHON:-}" ]]; then
    runtime_python="$WEATHERSTRIP_PYTHON"
elif [[ -x "$root/.venv/bin/python" ]]; then
    runtime_python="$root/.venv/bin/python"
elif [[ -f "$root/.workspace/toolchain/bin/activate" ]]; then
    source "$root/.workspace/toolchain/bin/activate"
    runtime_python="$VIRTUAL_ENV/bin/python"
else
    runtime_python="$(command -v python3)"
fi
# AppLauncher can run from a pinned Isaac Lab source checkout without installing
# the unrelated training dependency set (which pins a different Warp version).
if [[ -n "${WEATHERSTRIP_ISAACLAB:-}" ]]; then
    if [[ ! -d "$WEATHERSTRIP_ISAACLAB/source/isaaclab/isaaclab/app" ]]; then
        echo "WEATHERSTRIP_ISAACLAB must point to an Isaac Lab source checkout" >&2
        exit 2
    fi
    for extension in "$WEATHERSTRIP_ISAACLAB"/source/*; do
        [[ -d "$extension" ]] && export PYTHONPATH="$extension${PYTHONPATH:+:$PYTHONPATH}"
    done
fi
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
# Respect the caller's device visibility unless explicitly selecting a physical GPU.
# EFORREST_* aliases keep existing workstation services compatible.
if [[ "${WEATHERSTRIP_ALL_GPUS:-${EFORREST_ALL_GPUS:-0}}" == 1 ]]; then
    unset CUDA_VISIBLE_DEVICES
elif [[ -n "${WEATHERSTRIP_GPU:-${EFORREST_GPU:-}}" ]]; then
    export CUDA_VISIBLE_DEVICES="${WEATHERSTRIP_GPU:-$EFORREST_GPU}"
fi
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$root/.cache/warp}"
mkdir -p "$root/results"
cd "$root"
exec "$runtime_python" "$@"
