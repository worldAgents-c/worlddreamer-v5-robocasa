# Copyright (C) 2026 World Agents Corporation.
from collections import deque
from pathlib import Path
import time
import numpy as np
from .checkpoint import load_model
from .observations import ObservationEncoder, history_indices
from .text_cache import read_features
from .video import VideoSession


def sample_action_noise(action):
    import torch
    if action.shape != (1, 4, 16, 18):
        raise ValueError('Expected four robot slots with 16 actions of 18 dimensions')
    noise = torch.zeros_like(action, dtype=torch.bfloat16)
    noise[:, 0] = torch.randn_like(noise[:, 0])
    noise[:, :, :, :7] = 0
    return noise


class RoboCasaPolicy:
    def __init__(self, model_path, text_cache, device='cuda:0', text_timeout=600):
        import torch
        from .video_encoder import VideoEncoder
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        self.model, self.metadata = load_model(model_path, self.device)
        root = Path(model_path)
        vae = VideoEncoder.load(root / 'assets/video/vae', self.device)
        self.video = VideoSession(vae, self.device)
        self.encoder = ObservationEncoder(root / 'assets/vlm')
        self.text_cache, self.text_timeout = text_cache, text_timeout
        self.instruction = None

    def identity(self):
        return dict(model_sha256=self.metadata['model_sha256'],
                    action_length=16,
                    action_dimensions=18, state_dimensions=14)

    def reset(self, instruction, seed):
        self.instruction = None
        text = read_features(self.text_cache, instruction, self.text_timeout)
        self.seed = int(seed)
        if not 0 <= self.seed < 2**31 - 1:
            raise ValueError('Invalid episode seed')
        self.text = text
        self.video.reset(self.seed)
        self.rgb_history, self.state_history = deque(maxlen=7), deque(maxlen=7)
        self.replans = 0
        self.instruction = instruction
        return self.identity()

    def infer(self, request):
        import torch
        if self.instruction is None or str(request['instruction']) != self.instruction:
            raise ValueError('Episode reset is required before inference')
        rgb, states = request['vlm_rgb'], request['state16_sequence']
        count = request['video_rgb'].shape[1]
        start = int(request['video_frame_start'])
        if (rgb.shape != (3, count, 256, 256, 3) or rgb.dtype != np.uint8
                or states.shape != (count, 16) or not np.isfinite(states).all()
                or int(request['observation_frame']) != start + count - 1):
            raise ValueError('Misaligned image/state history')
        timings = {}
        def stamp(name, before):
            torch.cuda.synchronize(self.device)
            now = time.perf_counter(); timings[name] = now - before
            return now
        torch.cuda.synchronize(self.device); started = time.perf_counter()
        with torch.inference_mode():
            self.video.ingest(request['video_rgb'], request['video_intrinsics'],
                              request['video_extrinsics'], start)
            before = stamp('video_encode_seconds', started)
            self.rgb_history.extend(rgb.transpose(1, 0, 2, 3, 4))
            self.state_history.extend(states)
            indices = history_indices(len(self.state_history) - 1)
            batch = self.encoder(self.instruction, np.asarray(self.rgb_history)[indices].transpose(1, 0, 2, 3, 4),
                                 np.asarray(self.state_history)[indices])
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            batch.update(self.video.inputs(self.text, request['base_to_world']))
            before = stamp('observation_pack_seconds', before)
            seed = (self.seed + self.replans) % (2**31 - 1)
            with torch.random.fork_rng(devices=[self.device.index]):
                torch.manual_seed(seed)
                batch['inference_action_noise'] = sample_action_noise(batch['action'])
                mode_noise = torch.zeros(1, 4, 16, 1, device=self.device, dtype=torch.bfloat16)
                mode_noise[:, 0] = torch.randn_like(mode_noise[:, 0])
                batch['control_mode_noise'] = mode_noise
                result = self.model.generate(batch)
            stamp('model_seconds', before)
            action_tensor, mode_tensor = result['action_latents'], result['control_mode']
            if action_tensor.shape != (1, 4, 16, 18) or not torch.isfinite(action_tensor).all() or mode_tensor.shape != (1, 4, 16, 1) or not torch.isfinite(mode_tensor).all():
                raise ValueError('Model produced invalid controls')
            action = action_tensor[0, 0].float().cpu().numpy()
            modes = mode_tensor[0, 0, :, 0].float().cpu().numpy()
        self.replans += 1
        return dict(action_latents=action, control_mode=modes, policy_seed=seed, replans=self.replans,
                    total_seconds=time.perf_counter() - started,
                    **timings, **self.identity(), **self.video.timing())
