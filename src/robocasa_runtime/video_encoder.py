# Copyright (C) 2026 World Agents Corporation.
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from diffusers.models.autoencoders.autoencoder_kl_wan import WanEncoder3d, WanCausalConv3d


class VideoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = SimpleNamespace(**config)
        if self.config.patch_size is not None:
            raise ValueError("Expected an unpatchified video encoder")
        self.encoder = WanEncoder3d(
            in_channels=config['in_channels'], dim=config['base_dim'],
            z_dim=config['z_dim'] * 2, dim_mult=config['dim_mult'],
            num_res_blocks=config['num_res_blocks'], attn_scales=config['attn_scales'],
            temperal_downsample=config['temperal_downsample'],
            dropout=config['dropout'], is_residual=config['is_residual'],
        )
        self.quant_conv = WanCausalConv3d(config['z_dim'] * 2, config['z_dim'] * 2, 1)
        self._cached_conv_counts = {
            'encoder': sum(isinstance(m, WanCausalConv3d) for m in self.encoder.modules())
        }

    @property
    def dtype(self):
        return self.quant_conv.weight.dtype

    @classmethod
    def load(cls, directory, device):
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        directory = Path(directory)
        config = json.loads((directory / 'config.json').read_text())
        if config.get('_class_name') != cls.__name__:
            raise ValueError("Unexpected video encoder configuration")
        with init_empty_weights():
            model = cls(config)
        model.load_state_dict(load_file(str(directory / 'diffusion_pytorch_model.safetensors')),
                              strict=True, assign=True)
        model = model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        if any(p.is_meta for p in model.parameters()):
            raise ValueError("Uninitialized video encoder parameters")
        return model
