# Copyright (C) 2026 World Agents Corporation.

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
)
from diffusers.models.modeling_utils import ModelMixin


from .layers import ExchangeBlock, FusionBlock
from .embeddings import (
    VideoPatchEmbedding,
    FlowConditionEmbedding,
    VideoRotaryEmbedding,
    ImageConditionProjection,
)
from .execution import encode_streams


class VideoTransformer(
    ModelMixin,
    ConfigMixin,
    FromOriginalModelMixin,
    CacheMixin,
):

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
        "x_embedder",
        "context_embedder",
    ]
    _repeated_blocks = ["FusionBlock", "ExchangeBlock"]
    _no_split_modules = [
        "ExchangeBlock",
        "FusionBlock",
        "VideoPatchEmbedding",
    ]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 33,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        norm_type: str = "layer_norm",
        text_embed_dim: int = 4096,
        image_embed_dim: int | None = None,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int, ...] = (16, 56, 56),
    ) -> None:
        super().__init__()

        model_width = num_attention_heads * attention_head_dim
        self._build_tokenizers(
            model_width, (patch_size_t, patch_size, patch_size),
            in_channels, text_embed_dim, image_embed_dim,
        )
        self._build_conditioning(
            model_width, patch_size, patch_size_t, rope_axes_dim, rope_theta,
        )
        self._build_stream_stack(
            num_layers, num_single_layers,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            mlp_ratio=mlp_ratio, qk_norm=qk_norm, norm_type=norm_type,
        )
        self._validate_cross_attention()
        self.gradient_checkpointing = False

    def _build_tokenizers(self, width, patch_shape, channels, text_width, image_width):
        self.x_embedder = VideoPatchEmbedding(patch_shape, channels, width)
        self.context_embedder = PixArtAlphaTextProjection(
            in_features=text_width, hidden_size=width
        )
        self.image_embed_dim = image_width
        if image_width is not None:
            self.image_embedder = ImageConditionProjection(image_width, width)

    def _build_conditioning(self, width, spatial_patch, temporal_patch, axes, theta):
        self.time_text_embed = FlowConditionEmbedding(width)
        self.rope = VideoRotaryEmbedding(spatial_patch, temporal_patch, axes, theta)

    def _build_stream_stack(self, exchange_count, fusion_count, **block_config):
        self.exchange_blocks = nn.ModuleList()
        for _ in range(exchange_count):
            self.exchange_blocks.append(ExchangeBlock(
                **block_config,
                enable_text_cross_attention=False,
            ))
        self.fusion_blocks = nn.ModuleList()
        for _ in range(fusion_count):
            self.fusion_blocks.append(FusionBlock(
                **block_config, enable_text_cross_attention=True,
            ))

    def _validate_cross_attention(self):
        groups = (
            ("exchange_blocks", self.exchange_blocks,
             [False] * len(self.exchange_blocks)),
            ("fusion_blocks", self.fusion_blocks,
             [True] * len(self.fusion_blocks)),
        )
        for group, blocks, expected_flags in groups:
            for index, (block, expected) in enumerate(zip(blocks, expected_flags)):
                actual = block.enable_text_cross_attention
                if actual != expected:
                    raise ValueError(
                        f"{group}[{index}].enable_text_cross_attention={actual}, "
                        f"expected {expected}."
                    )

    def _prepare_context(self, text, images, mask, video_count):
        context = self.context_embedder(text)
        image_count = 0
        if images is not None:
            if self.image_embed_dim is None:
                raise ValueError("image conditioning requires image_embed_dim")
            image_tokens = self.image_embedder(images)
            image_count = image_tokens.shape[1]
            context = torch.cat((image_tokens, context), dim=1)
            if mask is not None:
                mask = F.pad(mask, (image_count, 0), value=1)
        if mask is not None:
            mask = F.pad(mask.bool(), (video_count, 0), value=True)[:, None, None]
        return context, mask, image_count

    def forward(self, latents, timestep, text, text_mask=None, images=None):
        rotary = self.rope(latents)
        conditioning = self.time_text_embed(timestep)
        video = self.x_embedder(latents)
        context, mask, image_count = self._prepare_context(
            text, images, text_mask, video.shape[1]
        )
        encoded, _ = encode_streams(
            self, video, context, conditioning, mask, rotary, image_count
        )
        return encoded
