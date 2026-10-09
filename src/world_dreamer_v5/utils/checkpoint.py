# Copyright (C) 2026 World Agents Corporation.
import json
from pathlib import Path


def read_video_config(path):
    config = json.loads(Path(path).read_text())
    return {key: value for key, value in config.items() if not key.startswith('_')}
