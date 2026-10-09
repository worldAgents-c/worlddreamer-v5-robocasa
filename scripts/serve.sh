#!/usr/bin/env bash
# Copyright (C) 2026 World Agents Corporation.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_PATH:?Set MODEL_PATH to the converted RoboCasa model directory}"
: "${TEXT_CACHE:?Set TEXT_CACHE to the text feature queue directory}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
exec "${MODEL_PYTHON:-python3.10}" -m robocasa_runtime.server \
  --model-path "$MODEL_PATH" --text-cache "$TEXT_CACHE" "$@"
