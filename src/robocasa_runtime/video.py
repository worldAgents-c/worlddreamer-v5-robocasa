# Copyright (C) 2026 World Agents Corporation.
from collections import deque
import numpy as np


class VideoSession:
    def __init__(self, vae, device):
        import torch
        self.vae, self.device = vae, device
        self.mean = torch.tensor(vae.config.latents_mean, device=device, dtype=vae.dtype).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(vae.config.latents_std, device=device, dtype=vae.dtype).view(1, -1, 1, 1, 1)
        if vae.config.patch_size is not None:
            raise ValueError('Expected an unpatchified video VAE')
        self.reset(0)

    def reset(self, seed):
        import torch
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.cache = [None] * int(self.vae._cached_conv_counts['encoder'])
        self.pending = deque()
        self.history = deque(maxlen=3)
        self.frames = self.latent_count = 0

    def ingest(self, rgb, intrinsic, extrinsic, start):
        import torch
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        count = rgb.shape[1]
        if (start != self.frames or not 1 <= count <= 16 or rgb.shape != (3, count, 240, 320, 3)
                or rgb.dtype != np.uint8 or intrinsic.shape != (3, count, 3, 3)
                or extrinsic.shape != (3, count, 3, 4)
                or not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all()):
            raise ValueError('Invalid or noncontiguous video observations')
        self.pending.extend((rgb[:, i], intrinsic[:, i], extrinsic[:, i]) for i in range(count))
        self.frames += count
        while len(self.pending) >= (1 if self.latent_count == 0 else 4):
            chunk = [self.pending.popleft() for _ in range(1 if self.latent_count == 0 else 4)]
            value = torch.from_numpy(np.stack([v[0] for v in chunk], axis=1)).permute(0, 4, 1, 2, 3)
            value = value.to(self.device, dtype=self.vae.dtype) / 127.5 - 1
            moments = self.vae.quant_conv(self.vae.encoder(value, feat_cache=self.cache, feat_idx=[0]))
            latent = DiagonalGaussianDistribution(moments).sample(generator=self.generator)
            latent = (latent - self.mean) / self.std
            if latent.shape != (3, 16, 1, 30, 40) or not torch.isfinite(latent).all():
                raise ValueError('Invalid video latent')
            self.history.append((latent[:, :, 0], chunk[-1][1], chunk[-1][2]))
            self.latent_count += 1

    def timing(self):
        if not self.latent_count:
            raise ValueError('No video observations available')
        anchor = 4 * (self.latent_count - 1)
        return dict(observation_frame=self.frames - 1, video_anchor_frame=anchor,
                    video_lag_frames=self.frames - 1 - anchor)

    def inputs(self, text, base):
        import torch
        self.timing()
        if base.shape != (4, 4) or not np.isfinite(base).all():
            raise ValueError('Invalid current base transform')
        raw = [self.latent_count - 3, self.latent_count - 2, self.latent_count - 1]
        start = max(0, self.latent_count - 3)
        samples = [self.history[max(0, index) - start] for index in raw]
        tensor = lambda x: torch.as_tensor(x, device=self.device)
        latent = torch.stack([s[0] for s in samples], dim=2)[None]
        intrinsic = tensor(np.stack([s[1] for s in samples], axis=1))[None]
        extrinsic = tensor(np.stack([s[2] for s in samples], axis=1))[None] @ tensor(base)
        depth = torch.zeros(1, 3, 3, 15, 20, device=self.device, dtype=torch.float16)
        return dict(video_latents=latent, video_text=text['embeddings'].to(self.device)[None],
                    video_text_mask=text['mask'].to(self.device).bool()[None],
                    video_view_mask=tensor([[True, True, True]]),
                    video_latent_valid=tensor([[index >= 0 for index in raw]]),
                    camera_intrinsics=intrinsic.float(), camera_extrinsics=extrinsic.float(),
                    camera_depth=depth, camera_depth_valid=torch.zeros_like(depth, dtype=torch.bool),
                    camera_pixel_valid=torch.ones_like(depth, dtype=torch.bool),
                    camera_roles=tensor([[0, 0, 1]]), agent_camera_indices=tensor([[2, -1, -1, -1]]))
