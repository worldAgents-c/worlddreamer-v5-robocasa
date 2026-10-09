# Copyright (C) 2026 World Agents Corporation.
import argparse
import json
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
from .common import atomic_json
from .observations import capture
from .action_adapter import SimulationActionAdapter
from .transport import connect, rpc


def make_request(instruction, observations, start, step):
    return dict(command='infer', instruction=instruction, video_frame_start=start, observation_frame=step,
                video_rgb=np.stack([v['video'] for v in observations], axis=1),
                vlm_rgb=np.stack([v['rgb'] for v in observations], axis=1),
                state16_sequence=np.stack([v['state'] for v in observations]),
                video_intrinsics=np.stack([v['K'] for v in observations], axis=1),
                video_extrinsics=np.stack([v['E'] for v in observations], axis=1),
                base_to_world=observations[-1]['base'])


def run_episode(args, task, index, horizon):
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
    from robocasa.utils.env_utils import convert_action
    import cv2
    seed = args.seed + index
    folder = args.output / task / f'{index:04d}'
    folder.mkdir(parents=True, exist_ok=False)
    result = dict(task=task, episode=index, seed=seed, split=args.split, horizon=horizon,
                  execute_steps=args.execute_steps, environment_seed=args.environment_seed, status='running', success=False, steps=0)
    atomic_json(folder / 'result.json', result)
    random.seed(seed); np.random.seed(seed)
    env = connection = writer = None
    actions, applied, physical_actions, modes = [], [], [], []
    started = time.perf_counter()
    try:
        env = RoboCasaGymEnv(env_name=task, split=args.split, seed=args.environment_seed,
                            camera_widths=256, camera_heights=256, enable_render=True)
        obs, _ = env.reset(seed=seed)
        instruction = obs['annotation.human.task_description'].strip()
        result['instruction'] = instruction
        atomic_json(folder / 'scene.json', env.env.get_ep_meta())
        adapter = SimulationActionAdapter(env)
        current = capture(env, obs)
        np.savez_compressed(folder / 'initial_observation.npz', **current, instruction=instruction)
        if args.video:
            writer = cv2.VideoWriter(str(folder / 'rollout.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (768, 256))
            if not writer.isOpened():
                raise RuntimeError('Video recorder did not open')
        connection = connect(args.host, args.port)
        identity = rpc(connection, command='reset', instruction=instruction, seed=seed)
        if int(identity['action_length']) != 16 or int(identity['action_dimensions']) != 18:
            raise ValueError('Policy uses a different action contract')
        result['model_sha256'] = str(identity['model_sha256'])
        pending, sent = [current], 0
        while result['steps'] < horizon:
            step = result['steps']
            physics_time = env.env.sim.data.time
            before = time.perf_counter()
            response = rpc(connection, **make_request(instruction, pending, sent, step))
            elapsed = time.perf_counter() - before
            if env.env.sim.data.time != physics_time:
                raise RuntimeError('Simulation advanced while awaiting inference')
            if (str(response['model_sha256']) != result['model_sha256'] or
                    int(response['observation_frame']) != step or int(response['video_lag_frames']) != step % 4):
                raise ValueError('Policy response does not match this observation')
            controls, mode_values = response['action_latents'], response['control_mode']
            if controls.shape != (16, 18) or mode_values.shape != (16,) or not np.isfinite(controls).all() or not np.isfinite(mode_values).all():
                raise ValueError('Invalid controls')
            sent += len(pending); pending = []
            with (folder / 'inference.jsonl').open('a') as log:
                log.write(json.dumps(dict(step=step, rpc_seconds=elapsed,
                    **{k: float(response[k]) for k in ('video_encode_seconds', 'observation_pack_seconds', 'model_seconds')})) + '\n')
            adapter.begin_chunk(current)
            for offset, control in enumerate(controls[:min(args.execute_steps, horizon - step)]):
                physical, executable = adapter.materialize(control, mode_values[offset], current)
                action = convert_action(executable.copy())
                physical_actions.append(physical.copy()); modes.append(mode_values[offset])
                obs, reward, terminated, truncated, info = env.step(action)
                result['steps'] += 1
                result['success'] = bool(info.get('success', False)) or bool(env.env._check_success())
                actions.append(control.copy())
                applied.append(np.concatenate([np.asarray(action['action.' + key]).reshape(-1)
                    for key in ('end_effector_position', 'end_effector_rotation', 'gripper_close',
                                'base_motion', 'control_mode')]))
                current = capture(env, obs); pending.append(current)
                if writer is not None:
                    writer.write(cv2.cvtColor(np.concatenate(list(current['rgb']), axis=1), cv2.COLOR_RGB2BGR))
                if result['success'] or terminated or truncated:
                    break
            atomic_json(folder / 'result.json', result)
            print(f'{task} episode={index} steps={result["steps"]}/{horizon} success={result["success"]} rpc={elapsed:.2f}s', flush=True)
            if result['success'] or terminated or truncated:
                break
        result['status'] = 'complete'
    except BaseException as error:
        result.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'error',
                      error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter() - started
        # Save evidence before cleanup, which may itself fail.
        atomic_json(folder / 'result.json', result)
        np.savez_compressed(folder / 'actions.npz', model_latents=np.asarray(actions), action18=np.asarray(physical_actions), control_mode=np.asarray(modes), applied=np.asarray(applied))
        if connection is not None: connection.close()
        if writer is not None: writer.release()
        if env is not None: env.env.close()
    return result


def main():
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
    parser = argparse.ArgumentParser(description='Sequential seeded simulation evaluation, independent of the model environment.')
    parser.add_argument('--tasks', nargs='+', required=True, help='Task names, or all')
    parser.add_argument('--task-config', type=Path, default=Path('configs/task_sets.yaml'))
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=50007)
    parser.add_argument('--environment-seed', type=int, default=7,
                        help='Construction seed, separate from the episode reset seed')
    parser.add_argument('--split', choices=('pretrain', 'target'), default='pretrain')
    parser.add_argument('--horizon', type=int, help='Explicit override; recorded in every result')
    parser.add_argument('--execute-steps', type=int, choices=(8, 16), default=16)
    parser.add_argument('--host', default='127.0.0.1'); parser.add_argument('--port', type=int, default=18120)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(args.task_config.read_text())
    tasks = list(config['task_horizons']) if args.tasks == ['all'] else args.tasks
    if len(set(tasks)) != len(tasks) or args.episodes < 1 or (args.horizon is not None and args.horizon < 1):
        raise ValueError('Invalid evaluation scope')
    for task in tasks:
        if task not in config['task_horizons']:
            raise ValueError(f'Unknown task: {task}')
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / 'plan.json', dict(tasks=tasks, episodes=args.episodes, seed=args.seed,
        split=args.split, environment_seed=args.environment_seed,
        execute_steps=args.execute_steps, horizon_override=args.horizon))
    results = []
    try:
        for index in range(args.episodes):
            for task in tasks:
                results.append(run_episode(args, task, index, args.horizon or config['task_horizons'][task]))
    finally:
        # Errors stay visible, and a partial run never becomes a full benchmark score.
        rows = [json.loads(p.read_text()) for p in args.output.glob('*/*/result.json')]
        complete = [r for r in rows if r['status'] == 'complete']
        finished = len(complete) == len(tasks) * args.episodes
        atomic_json(args.output / 'summary.json', dict(expected=len(tasks) * args.episodes,
            complete=len(complete), errors=sum(r['status'] == 'error' for r in rows),
            successes=sum(r['success'] for r in complete), scope_complete=finished,
            success_rate=sum(r['success'] for r in complete) / len(complete) if finished else None))


if __name__ == '__main__':
    main()
