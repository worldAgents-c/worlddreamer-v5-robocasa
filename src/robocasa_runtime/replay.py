# Copyright (C) 2026 World Agents Corporation.
import argparse
import os
from pathlib import Path
import numpy as np
from .common import atomic_json
from .rollout import make_request


def main():
    parser = argparse.ArgumentParser(description='Run one recorded simulator observation through the policy without a simulator.')
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--text-cache', type=Path, required=True)
    parser.add_argument('--observation', type=Path, required=True,
                        help='NPZ: rgb, video, state, K, E, base, instruction')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=50007)
    args = parser.parse_args()
    os.environ.setdefault('WMD_KERNEL_BACKEND', 'torch')
    os.environ.setdefault('WMD_SDPA_BACKEND', 'math')
    from .policy import RoboCasaPolicy
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.observation, allow_pickle=False) as archive:
        observation = {k: archive[k] for k in archive.files}
    policy = RoboCasaPolicy(args.model_path, args.text_cache)
    instruction = str(observation['instruction'])
    policy.reset(instruction, args.seed)
    response = policy.infer(make_request(instruction, [observation], 0, 0))
    np.savez_compressed(args.output / 'actions.npz', action_latents=response.pop('action_latents'), control_mode=response.pop('control_mode'))
    atomic_json(args.output / 'report.json', dict(scope='recorded observation only', **response))
    print(args.output / 'report.json')


if __name__ == '__main__':
    main()
