# Copyright (C) 2026 World Agents Corporation.
import hashlib
import json
from pathlib import Path
import uuid


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def text_key(instruction):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('A nonempty English task instruction is required')
    return hashlib.sha256(instruction.encode('utf-8')).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)
