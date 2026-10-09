# Copyright (C) 2026 World Agents Corporation.
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from .common import atomic_json
from .transport import receive, send


def main():
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError('Use Python 3.10 for the model service')
    parser = argparse.ArgumentParser(description='Serve a fixed RoboCasa checkpoint over a local NumPy socket.')
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--text-cache', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18120)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--text-timeout', type=float, default=600)
    parser.add_argument('--log-dir', type=Path, default=Path('outputs/policy'))
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('WMD_KERNEL_BACKEND', 'torch')
    os.environ.setdefault('WMD_SDPA_BACKEND', 'math')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    from .policy import RoboCasaPolicy
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    def health(status, **extra):
        atomic_json(args.log_dir / 'health.json', dict(status=status, time=time.time(), pid=os.getpid(), **extra))
    health('loading')
    policy = RoboCasaPolicy(args.model_path, args.text_cache, args.device, args.text_timeout)
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port)); listener.listen(4)
        health('ready', **policy.identity())
        print('POLICY_READY ' + json.dumps(policy.identity()), flush=True)
        try:
            while True:
                connection, address = listener.accept()
                policy.instruction = None
                with connection:
                    connection.settimeout(1200)
                    while True:
                        try:
                            request = receive(connection)
                            command = str(request['command'])
                            if command == 'health':
                                response = policy.identity()
                            elif command == 'reset':
                                health('resetting')
                                response = policy.reset(str(request['instruction']), int(request['seed']))
                            elif command == 'infer':
                                health('inference', **policy.identity())
                                response = policy.infer(request)
                                record = {k: v for k, v in response.items() if k not in ('action_latents', 'control_mode')}
                                with (args.log_dir / 'timings.jsonl').open('a') as log:
                                    log.write(json.dumps(record) + '\n')
                                print(json.dumps(record), flush=True)
                            else:
                                raise ValueError(f'Unknown command: {command}')
                            send(connection, dict(status='ok', **response))
                            health('ready', **policy.identity())
                        except (EOFError, ConnectionError, socket.timeout):
                            break
                        except Exception as error:
                            health('error', error=str(error), traceback=traceback.format_exc())
                            traceback.print_exc()
                            try:
                                send(connection, dict(status='error', error=str(error)))
                            except OSError:
                                pass
                            # A partially consumed observation must never be retried.
                            break
                policy.instruction = None
        finally:
            health('stopped', **policy.identity())


if __name__ == '__main__':
    main()
