#!/usr/bin/env bash
# Copyright (C) 2026 World Agents Corporation.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${SIM_PYTHON:-python3.10}" -m robocasa_runtime.rollout "$@"
