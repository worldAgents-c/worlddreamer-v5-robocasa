# Copyright (C) 2026 World Agents Corporation.
import argparse
import json
import time
from pathlib import Path
from .common import atomic_json, sha256, text_key

REVISION = '6748a1f5861a859aca13b30a0c65dd6f9942ca35'


def read_features(directory, instruction, timeout=0):
    import torch
    from safetensors.torch import load_file
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    key = text_key(instruction)
    path, receipt = directory / (key + '.safetensors'), directory / (key + '.json')
    atomic_json(directory / (key + '.request.json'), dict(instruction=instruction))
    deadline = time.monotonic() + timeout
    while not receipt.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Text features unavailable for {key}; start the text-cache worker with --queue {directory}')
        time.sleep(.2)
    record = json.loads(receipt.read_text())
    if record.get('status') != 'passed':
        raise RuntimeError(f'Text encoding failed: {record.get("error", record.get("traceback"))}')
    if record.get('instruction') != instruction or record.get('revision') != REVISION or sha256(path) != record['sha256']:
        raise ValueError('Text cache identity or checksum mismatch')
    features = load_file(str(path))
    embedding, mask = features['embeddings'], features['mask']
    if (embedding.ndim != 2 or embedding.shape[1] != 2560 or not 0 < len(embedding) < 512 or
            mask.shape != (len(embedding),) or not mask.bool().all() or not torch.isfinite(embedding).all()):
        raise ValueError('Invalid text feature dimensions or values')
    return features


def main():
    parser = argparse.ArgumentParser(description='Text features are stored outside the model package, keyed by exact instruction.')
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--integrity', type=Path, required=True)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--drain', action='store_true')
    args = parser.parse_args()
    import os
    import fcntl
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    import torch
    from transformers import GemmaTokenizer, T5Gemma2Encoder
    from safetensors.torch import save_file
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    args.queue.mkdir(parents=True, exist_ok=True)
    lock = (args.queue / '.worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    integrity = json.loads(args.integrity.read_text())
    if integrity['revision'] != REVISION:
        raise ValueError('Text encoder revision differs from training')
    for name, record in integrity['files'].items():
        if sha256(args.assets / name) != record['sha256']:
            raise ValueError(f'Text encoder asset checksum mismatch: {name}')
    tokenizer = GemmaTokenizer.from_pretrained(args.assets / 'tokenizer', local_files_only=True)
    tokenizer.padding_side = 'right'
    encoder, info = T5Gemma2Encoder.from_pretrained(args.assets / 'text_encoder', local_files_only=True,
        dtype=torch.float32, attn_implementation='sdpa', output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys')):
        raise ValueError('Text encoder did not load strictly')
    encoder = encoder.eval().to('cpu')
    print('Text cache worker ready', flush=True)
    while True:
        for request in sorted(args.queue.glob('*.request.json')):
            key = request.name.split('.')[0]
            receipt = args.queue / (key + '.json')
            if receipt.exists():
                continue
            started = time.monotonic()
            try:
                instruction = json.loads(request.read_text())['instruction']
                if text_key(instruction) != key:
                    raise ValueError('Request key mismatch')
                with torch.inference_mode():
                    inputs = tokenizer([instruction], padding='max_length', max_length=512, truncation=True,
                        add_special_tokens=True, return_attention_mask=True, return_tensors='pt')
                    keep = inputs['attention_mask'][0].bool()
                    if not 0 < int(keep.sum()) < 512:
                        raise ValueError('Empty or truncated task instruction')
                    encoded = encoder(**inputs)[0].to(torch.bfloat16)
                if encoded.shape != (1, 512, 2560) or not torch.isfinite(encoded).all():
                    raise ValueError('Invalid text encoder output')
                path = args.queue / (key + '.safetensors'); temporary = path.with_suffix('.partial')
                save_file(dict(embeddings=encoded[0, keep].contiguous(), mask=keep[keep].contiguous()), str(temporary))
                temporary.replace(path)
                atomic_json(receipt, dict(status='passed', instruction=instruction, revision=REVISION,
                            sha256=sha256(path), seconds=time.monotonic() - started))
                print(f'Encoded {key} in {time.monotonic() - started:.2f}s', flush=True)
            except Exception as error:
                atomic_json(receipt, dict(status='error', error=str(error)))
        if args.drain:
            break
        time.sleep(.5)


if __name__ == '__main__':
    main()
