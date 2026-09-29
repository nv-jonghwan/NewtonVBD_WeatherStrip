#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1
export EFORREST_CUDA_GRAPH="${WEATHERSTRIP_CUDA_GRAPH:-${EFORREST_CUDA_GRAPH:-1}}"
exec "$root/scripts/python.sh" -m newton_weatherstrip --viewer usd --test "$@"
