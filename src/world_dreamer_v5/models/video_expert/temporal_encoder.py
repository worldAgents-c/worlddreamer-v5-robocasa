# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations

import torch
import torch.nn as nn

from world_dreamer_v5.models.trifield import prefix_asymmetric_mask
from .execution import encode_streams, latent_grid


class VideoExpert(nn.Module):
    def __init__(self, config_path=None, backbone=None, observed_latents=3, *, config=None):
        super().__init__()
        if backbone is None:
            if config_path is None and config is None:
                raise ValueError("config or config_path is required")
            from .backbone import VideoTransformer
            from world_dreamer_v5.utils.checkpoint import read_video_config
            model_config = config if config is not None else read_video_config(config_path)
            backbone = VideoTransformer.from_config(model_config).to(torch.bfloat16)
        self.backbone = backbone
        self.observed_latents = int(observed_latents)

    def forward(self, latents, text, text_mask, view_mask, latent_valid=None,
                flow_timestep=None):
        if latents.ndim != 6:
            raise ValueError(f"expected [B,V,C,T,H,W] latents, got {tuple(latents.shape)}")
        batch_size, view_count, channels, latent_frames, height, width = latents.shape
        if channels != 16 or latent_frames != self.observed_latents:
            raise ValueError("Expected the configured observed latent history")
        if view_mask.shape != (batch_size, view_count):
            raise ValueError("view_mask shape does not match latent views")
        if latent_valid is None:
            latent_valid = torch.ones(batch_size, latent_frames, device=latents.device, dtype=torch.bool)
        if latent_valid.shape != (batch_size, latent_frames) or latent_valid.dtype != torch.bool:
            raise ValueError("latent_valid must be boolean [batch, latent_frames]")
        if not latent_valid.any(dim=1).all():
            raise ValueError("each scene requires a valid observed latent")
        active_views = view_mask.bool()
        view_indices = active_views.nonzero(as_tuple=False)
        if not len(view_indices):
            raise ValueError("each batch must contain at least one valid camera view")
        selected = latents[active_views]
        selected_valid = latent_valid[view_indices[:, 0]]
        selected = selected.masked_fill(~selected_valid[:, None, :, None, None], 0)
        selected_text = text[view_indices[:, 0]]
        selected_text_mask = text_mask[view_indices[:, 0]]
        if flow_timestep is None:
            flow_timestep = torch.rand(batch_size, device=latents.device, dtype=torch.float32)
        if flow_timestep.shape != (batch_size,):
            raise ValueError(f"expected one flow timestep per scene, got {tuple(flow_timestep.shape)}")
        timestep = flow_timestep[view_indices[:, 0]]
        condition_mask = selected.new_ones(len(selected), 1, latent_frames, height, width)
        condition_mask = condition_mask.masked_fill(~selected_valid[:, None, :, None, None], 0)
        hidden_states = torch.cat([selected, selected, condition_mask], dim=1)
        backbone = self.backbone
        grid = latent_grid(backbone, hidden_states)
        if grid[0] != latent_frames:
            raise ValueError("history validity requires one temporal token per latent")
        rotary = backbone.rope(hidden_states)
        conditioning = backbone.time_text_embed(timestep * 1000.0)
        video = backbone.x_embedder(hidden_states)
        context = backbone.context_embedder(selected_text)
        mask = prefix_asymmetric_mask(grid[0] * grid[1] * grid[2], 0, selected_text_mask)
        token_valid = selected_valid.repeat_interleave(grid[1] * grid[2], dim=1)
        mask[:, :, :, :token_valid.shape[1]] &= token_valid[:, None, None, :]
        encoded, _ = encode_streams(backbone, video, context, conditioning, mask, rotary)
        observed = encoded.view(len(selected), grid[0], grid[1] * grid[2], -1)
        observed = observed.masked_fill(~selected_valid[:, :, None, None], 0)
        restored = observed.new_zeros(*active_views.shape, *observed.shape[1:])
        restored[active_views] = observed
        return {"scene_tokens": restored, "flow_timestep": flow_timestep,
                "patch_grid": grid[1:]}
