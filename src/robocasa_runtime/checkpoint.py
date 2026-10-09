# Copyright (C) 2026 World Agents Corporation.
import json
from pathlib import Path
import numpy as np
from .common import sha256
from world_dreamer_v5.models.control_profiles import control_metadata, validate_control_metadata, DIMENSION_KEYS


def contained(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f'Asset escapes model directory: {name}')
    return path


def inspect_package(directory):
    root = Path(directory).resolve()
    metadata = json.loads((root / 'checkpoint.json').read_text())
    if metadata.get('format_version') != 2:
        raise ValueError('Expected a native model package')
    expected = dict(**control_metadata(14, 18, 4, 16),
                    normalization_dataset='RoboCasa', observation_interval=2,
                    action_normalization='identity_latent', state_normalization='none',
                    action_decoder='feedback_to_chunk_anchor', control_mode='separate_scalar')
    contract = metadata['robot_action_contract']
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f'Unsupported RoboCasa contract: {key}')
    for name, digest in metadata['asset_sha256'].items():
        if sha256(contained(root, name)) != digest:
            raise ValueError(f'Asset checksum mismatch: {name}')
    if sha256(root / 'model.safetensors') != metadata['model_sha256']:
        raise ValueError('Model checksum mismatch')
    if sha256(root / 'normalize.json') != contract['normalize_sha256']:
        raise ValueError('Normalization checksum mismatch')
    stats = json.loads((root / 'normalize.json').read_text())['RoboCasa']
    mean, std = np.asarray(stats['mean']), np.asarray(stats['std'])
    if mean.shape != (16, 18) or std.shape != (16, 18) or np.any(mean != 0) or np.any(std != 1):
        raise ValueError('This policy requires identity action normalization')
    config = json.loads((root / 'model_config.json').read_text())['model']
    for key in ('vlm_config_path', 'video_config_path'):
        config[key] = str(contained(root, config[key]))
    validate_control_metadata(contract, expected={key: config.get(key) for key in DIMENSION_KEYS})
    return config, metadata


def load_model(directory, device='cuda:0'):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from world_dreamer_v5.models.worldDreamerv5 import WorldDreamerv5
    from world_dreamer_v5.models.variants import normalize_model_config
    config, metadata = inspect_package(directory)
    with init_empty_weights():
        model = WorldDreamerv5(**normalize_model_config(config))
    tensors = load_file(str(Path(directory) / 'model.safetensors'))
    model.load_state_dict(tensors, strict=True, assign=True)
    if model.state_shape != (4, 14) or model.action_shape != (16, 18):
        raise ValueError('Unexpected model input/output shapes')
    model = model.to(device).eval().requires_grad_(False)
    if any(p.is_meta for p in model.parameters()):
        raise ValueError('Uninitialized model parameters')
    return model, metadata
