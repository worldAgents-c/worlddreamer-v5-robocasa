#!/usr/bin/env bash
# Copyright (C) 2026 World Agents Corporation.
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv --python 3.10 .venv-model
uv pip install --python .venv-model/bin/python -e '.[model]'
uv pip check --python .venv-model/bin/python
