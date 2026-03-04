"""
LTX-2 Video-Only Transformer

A video-only variant of LTX2VideoTransformer3DModel that does not instantiate
any audio components (no audio stream blocks, audio projections, audio cross-attn,
or audio RoPE).  Weights for kept video modules use the same attribute names as
the full AV model, so the video parameters load cleanly from the AV checkpoint
with strict=False.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from safetensors.torch import save_file

from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_ltx2 import (
    LTX2AdaLayerNormSingle,
    LTX2AudioVideoRotaryPosEmbed,
    LTX2Attention,
)


# ---------------------------------------------------------------------------
# Audio key detection helpers
# ---------------------------------------------------------------------------

_AUDIO_KEY_PREFIXES: Tuple[str, ...] = (
    "audio_",          # audio_proj_in, audio_time_embed, audio_rope, etc.
    "av_cross_attn_",  # global AV cross-attention modulation
    "cross_attn_rope", # cross-modal RoPE (shared video/audio)
)

_AUDIO_KEY_SUBSTRINGS: Tuple[str, ...] = (
    ".audio_",                  # per-block: audio_norm1, audio_attn1, ...
    ".video_a2v_cross_attn_",   # per-block: video a2v scale-shift table
    ".audio_a2v_cross_attn_",   # per-block: audio a2v scale-shift table
    ".audio_to_video_",         # per-block: a2v cross-attention
    ".video_to_audio_",         # per-block: v2a cross-attention
)


def is_audio_key(key: str) -> bool:
    """Return True if *key* is an audio-related state-dict key."""
    for prefix in _AUDIO_KEY_PREFIXES:
        if key.startswith(prefix):
            return True
    for substr in _AUDIO_KEY_SUBSTRINGS:
        if substr in key:
            return True
    return False


# ---------------------------------------------------------------------------
# Video-only transformer block
# ---------------------------------------------------------------------------

class LTX2VideoOnlyTransformerBlock(nn.Module):
    """Video-only transformer block stripped of all audio modules.

    Attribute names for the kept video modules are identical to those in
    ``LTX2VideoTransformerBlock`` so that weights load cleanly from an AV
    checkpoint using ``strict=False``.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        rope_type: str = "interleaved",
    ) -> None:
        super().__init__()

        # 1. Self-attention (video only)
        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = LTX2Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        # 2. Video–text cross-attention
        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = LTX2Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        # 3. Feed-forward (video only)
        self.norm3 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.ff = FeedForward(dim, activation_fn=activation_fn)

        # 4. Per-layer video AdaLN modulation parameters
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        video_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)

        # --- 1. Video self-attention with AdaLN ---
        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None].to(temb.device) + temb.reshape(
            batch_size, temb.size(1), num_ada_params, -1
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            query_rotary_emb=video_rotary_emb,
        )
        hidden_states = hidden_states + attn_output * gate_msa

        # --- 2. Video–text cross-attention ---
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            query_rotary_emb=None,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + attn_output

        # --- 3. Feed-forward ---
        norm_hidden_states = self.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


# ---------------------------------------------------------------------------
# Video-only top-level transformer model
# ---------------------------------------------------------------------------

class LTX2VideoOnlyTransformer3DModel(nn.Module):
    """Video-only variant of LTX2VideoTransformer3DModel.

    Does **not** instantiate audio VAE, vocoder, or any audio stream modules.
    Loads cleanly from an AV checkpoint by filtering out audio keys (strict=False).
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        cross_attention_dim: int = 4096,
        vae_scale_factors: Tuple[int, ...] = (8, 32, 32),
        pos_embed_max_pos: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        num_layers: int = 48,
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 3840,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        rope_theta: float = 10000.0,
        rope_double_precision: bool = True,
        causal_offset: int = 1,
        timestep_scale_multiplier: int = 1000,
        rope_type: str = "interleaved",
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        # Persist config for save / load
        self._config: Dict[str, Any] = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "patch_size": patch_size,
            "patch_size_t": patch_size_t,
            "num_attention_heads": num_attention_heads,
            "attention_head_dim": attention_head_dim,
            "cross_attention_dim": cross_attention_dim,
            "vae_scale_factors": list(vae_scale_factors),
            "pos_embed_max_pos": pos_embed_max_pos,
            "base_height": base_height,
            "base_width": base_width,
            "num_layers": num_layers,
            "activation_fn": activation_fn,
            "qk_norm": qk_norm,
            "norm_elementwise_affine": norm_elementwise_affine,
            "norm_eps": norm_eps,
            "caption_channels": caption_channels,
            "attention_bias": attention_bias,
            "attention_out_bias": attention_out_bias,
            "rope_theta": rope_theta,
            "rope_double_precision": rope_double_precision,
            "causal_offset": causal_offset,
            "timestep_scale_multiplier": timestep_scale_multiplier,
            "rope_type": rope_type,
        }

        # 1. Input projection (video)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 2. Text / caption projection (video)
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels, hidden_size=inner_dim
        )

        # 3. Timestep embedding and global modulation parameters (video)
        self.time_embed = LTX2AdaLayerNormSingle(
            inner_dim, num_mod_params=6, use_additional_conditions=False
        )
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)

        # 4. RoPE positional embeddings (video only – no audio_rope)
        self.rope = LTX2AudioVideoRotaryPosEmbed(
            dim=inner_dim,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            base_num_frames=pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            scale_factors=vae_scale_factors,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
        )

        # 5. Transformer blocks (video-only)
        self.transformer_blocks = nn.ModuleList(
            [
                LTX2VideoOnlyTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                    rope_type=rope_type,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Output layers (video only)
        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels)

        self.gradient_checkpointing = False

    # ------------------------------------------------------------------
    # Class / factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LTX2VideoOnlyTransformer3DModel":
        """Instantiate from a config dict, ignoring audio-specific keys."""
        _video_keys = {
            "in_channels", "out_channels", "patch_size", "patch_size_t",
            "num_attention_heads", "attention_head_dim", "cross_attention_dim",
            "vae_scale_factors", "pos_embed_max_pos", "base_height", "base_width",
            "num_layers", "activation_fn", "qk_norm", "norm_elementwise_affine",
            "norm_eps", "caption_channels", "attention_bias", "attention_out_bias",
            "rope_theta", "rope_double_precision", "causal_offset",
            "timestep_scale_multiplier", "rope_type",
        }
        kwargs = {k: v for k, v in config.items() if k in _video_keys}
        return cls(**kwargs)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        subfolder: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "LTX2VideoOnlyTransformer3DModel":
        """Load a saved ``LTX2VideoOnlyTransformer3DModel`` from *path*."""
        if subfolder is not None:
            path = os.path.join(path, subfolder)

        config_path = os.path.join(path, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        config.pop("_class_name", None)

        model = cls.from_config(config)

        weights_path = os.path.join(path, "model.safetensors")
        if os.path.exists(weights_path):
            from safetensors.torch import load_file as _load_file
            state_dict = _load_file(weights_path)
        else:
            weights_path = os.path.join(path, "pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location="cpu")

        model.load_state_dict(state_dict, strict=True)
        if torch_dtype is not None:
            model = model.to(torch_dtype)
        return model

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self):
        """Config-like object whose attributes mirror the constructor kwargs.

        The object is cached so repeated access is cheap.
        """
        if not hasattr(self, "_config_obj"):
            class _Cfg:
                pass
            cfg = _Cfg()
            for k, v in self._config.items():
                setattr(cfg, k, v)
            object.__setattr__(self, "_config_obj", cfg)
        return self._config_obj

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    # ------------------------------------------------------------------
    # Forward pass (video only)
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: float = 24.0,
        video_coords: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        # Accept and silently ignore any audio-related keyword args for
        # signature compatibility with call-sites that pass audio arguments.
        # Ignored kwargs include: audio_hidden_states, audio_encoder_hidden_states,
        # audio_timestep, audio_coords, audio_num_frames, audio_encoder_attention_mask,
        # attention_kwargs.
        **kwargs,
    ):
        batch_size = hidden_states.size(0)

        # Normalise encoder attention mask to additive bias
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Prepare RoPE positional embeddings
        if video_coords is None:
            video_coords = self.rope.prepare_video_coords(
                batch_size, num_frames, height, width, hidden_states.device, fps=fps
            )
        video_rotary_emb = self.rope(video_coords, device=hidden_states.device)

        # 2. Input projection
        hidden_states = self.proj_in(hidden_states)

        # 3. Timestep embedding and modulation parameters
        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        # 4. Caption projection
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

        # 5. Transformer blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                import torch.utils.checkpoint as _ckpt

                def _create_custom_fwd(blk):
                    def _fwd(*args):
                        return blk(*args)
                    return _fwd

                hidden_states = _ckpt.checkpoint(
                    _create_custom_fwd(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    video_rotary_emb,
                    encoder_attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    video_rotary_emb=video_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                )

        # 6. Output layer
        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output, None)  # None = no audio output
        return output

    # ------------------------------------------------------------------
    # Save / load helpers
    # ------------------------------------------------------------------

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
    ) -> None:
        """Save model weights and config to *save_directory*."""
        os.makedirs(save_directory, exist_ok=True)

        # Config
        config_path = os.path.join(save_directory, "config.json")
        config_to_save = {"_class_name": "LTX2VideoOnlyTransformer3DModel", **self._config}
        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)

        # Weights
        if safe_serialization:
            save_file(self.state_dict(), os.path.join(save_directory, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
