"""
Smoke tests for LTX-2 video-only mode.

These tests do NOT require a GPU or actual model weights – they validate the
structure and configuration of the video-only implementation using tiny random
tensors.

Run with:
    python testing/test_ltx2_video_only.py
"""

import os
import sys
import tempfile
import unittest

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


class TestLTX2VideoOnlyTransformer(unittest.TestCase):
    """Tests for LTX2VideoOnlyTransformer3DModel."""

    def _make_tiny_config(self):
        """Return a minimal config for quick tests (2 layers, small dims)."""
        return {
            "in_channels": 4,
            "out_channels": 4,
            "patch_size": 1,
            "patch_size_t": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "cross_attention_dim": 16,
            "vae_scale_factors": [8, 32, 32],
            "pos_embed_max_pos": 20,
            "base_height": 2048,
            "base_width": 2048,
            "num_layers": 2,
            "activation_fn": "gelu-approximate",
            "qk_norm": "rms_norm_across_heads",
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
            "caption_channels": 16,
            "attention_bias": True,
            "attention_out_bias": True,
            "rope_theta": 10000.0,
            "rope_double_precision": True,
            "causal_offset": 1,
            "timestep_scale_multiplier": 1000,
            "rope_type": "split",
        }

    def test_imports(self):
        """The video-only module must be importable."""
        from extensions_built_in.diffusion_models.ltx2.ltx2_video_only_transformer import (
            LTX2VideoOnlyTransformer3DModel,
            LTX2VideoOnlyTransformerBlock,
            is_audio_key,
        )
        self.assertTrue(callable(is_audio_key))

    def test_is_audio_key_helper(self):
        """is_audio_key() must correctly identify audio-related state-dict keys."""
        from extensions_built_in.diffusion_models.ltx2.ltx2_video_only_transformer import is_audio_key

        # Keys that MUST be identified as audio
        audio_keys = [
            "audio_proj_in.weight",
            "audio_time_embed.linear.weight",
            "audio_scale_shift_table",
            "audio_rope.something",
            "av_cross_attn_video_scale_shift.linear.weight",
            "av_cross_attn_audio_v2a_gate.linear.bias",
            "cross_attn_rope.theta",
            "transformer_blocks.0.audio_norm1.weight",
            "transformer_blocks.0.audio_attn1.to_q.weight",
            "transformer_blocks.3.video_a2v_cross_attn_scale_shift_table",
            "transformer_blocks.3.audio_a2v_cross_attn_scale_shift_table",
            "transformer_blocks.0.audio_to_video_attn.to_q.weight",
            "transformer_blocks.0.video_to_audio_norm.weight",
        ]
        for key in audio_keys:
            self.assertTrue(is_audio_key(key), f"Expected audio key, got False for: {key}")

        # Keys that must NOT be identified as audio
        video_keys = [
            "proj_in.weight",
            "time_embed.linear.weight",
            "scale_shift_table",
            "rope.theta",
            "caption_projection.linear_1.weight",
            "norm_out.weight",
            "proj_out.weight",
            "transformer_blocks.0.norm1.weight",
            "transformer_blocks.0.attn1.to_q.weight",
            "transformer_blocks.0.scale_shift_table",
            "transformer_blocks.0.norm2.weight",
            "transformer_blocks.0.attn2.to_k.weight",
            "transformer_blocks.0.ff.net.0.proj.weight",
        ]
        for key in video_keys:
            self.assertFalse(is_audio_key(key), f"Expected video key, got True for: {key}")

    def test_from_config(self):
        """LTX2VideoOnlyTransformer3DModel can be instantiated from a config dict."""
        from extensions_built_in.diffusion_models.ltx2.ltx2_video_only_transformer import (
            LTX2VideoOnlyTransformer3DModel,
        )
        cfg = self._make_tiny_config()
        # Also pass audio keys that should be silently ignored
        cfg_with_audio = {
            **cfg,
            "audio_in_channels": 128,
            "audio_out_channels": 128,
            "audio_patch_size": 1,
            "audio_patch_size_t": 1,
            "audio_num_attention_heads": 32,
            "audio_attention_head_dim": 64,
            "audio_cross_attention_dim": 2048,
            "audio_scale_factor": 4,
            "audio_pos_embed_max_pos": 20,
            "audio_sampling_rate": 16000,
            "audio_hop_length": 160,
            "cross_attn_timestep_scale_multiplier": 1000,
        }
        model = LTX2VideoOnlyTransformer3DModel.from_config(cfg_with_audio)
        self.assertIsInstance(model, LTX2VideoOnlyTransformer3DModel)

    def test_no_audio_parameters(self):
        """The model state dict must not contain any 'audio_' parameter keys."""
        from extensions_built_in.diffusion_models.ltx2.ltx2_video_only_transformer import (
            LTX2VideoOnlyTransformer3DModel,
            is_audio_key,
        )
        model = LTX2VideoOnlyTransformer3DModel.from_config(self._make_tiny_config())
        state_dict = model.state_dict()
        audio_params = [k for k in state_dict if is_audio_key(k)]
        self.assertEqual(
            audio_params,
            [],
            f"Found unexpected audio parameters: {audio_params}",
        )

    def test_load_from_av_state_dict_strict_false(self):
        """Loading from a mixed AV state dict (with audio keys) must succeed using strict=False."""
        from extensions_built_in.diffusion_models.ltx2.ltx2_video_only_transformer import (
            LTX2VideoOnlyTransformer3DModel,
        )
        cfg = self._make_tiny_config()
        model = LTX2VideoOnlyTransformer3DModel.from_config(cfg)

        # Build a state dict that includes extra "audio" keys
        av_state_dict = dict(model.state_dict())  # real video keys
        # Add fake audio keys that should be ignored
        inner_dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        av_state_dict["audio_proj_in.weight"] = torch.randn(inner_dim, cfg["in_channels"])
        av_state_dict["audio_proj_in.bias"] = torch.randn(inner_dim)
        av_state_dict["audio_scale_shift_table"] = torch.randn(2, inner_dim)
        for i in range(cfg["num_layers"]):
            av_state_dict[f"transformer_blocks.{i}.audio_scale_shift_table"] = torch.randn(6, inner_dim)

        # Should load without raising even though audio keys are present
        missing, unexpected = model.load_state_dict(av_state_dict, strict=False)
        # All audio keys end up in "unexpected" (they don't match model params)
        self.assertTrue(
            all("audio" in k for k in unexpected),
            f"Unexpected non-audio keys: {[k for k in unexpected if 'audio' not in k]}",
        )
        # No video keys should be missing
        self.assertEqual(missing, [], f"Missing video keys: {missing}")

    def test_config_property(self):
        """The config property must expose the constructor kwargs."""
        from extensions_built_in.diffusion_models.ltx2.ltx2_video_only_transformer import (
            LTX2VideoOnlyTransformer3DModel,
        )
        cfg = self._make_tiny_config()
        model = LTX2VideoOnlyTransformer3DModel.from_config(cfg)
        self.assertEqual(model.config.num_layers, cfg["num_layers"])
        self.assertEqual(model.config.in_channels, cfg["in_channels"])

    def test_save_and_load_pretrained(self):
        """save_pretrained / from_pretrained round-trip must preserve weights."""
        from extensions_built_in.diffusion_models.ltx2.ltx2_video_only_transformer import (
            LTX2VideoOnlyTransformer3DModel,
        )
        cfg = self._make_tiny_config()
        model = LTX2VideoOnlyTransformer3DModel.from_config(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            # Config file must exist
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "config.json")))
            # Weights file must exist
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "model.safetensors")))

            loaded = LTX2VideoOnlyTransformer3DModel.from_pretrained(tmpdir)
            # Parameters must match
            for (k1, v1), (k2, v2) in zip(
                model.state_dict().items(), loaded.state_dict().items()
            ):
                self.assertEqual(k1, k2)
                self.assertTrue(torch.allclose(v1, v2), f"Mismatch at key {k1}")

    def test_forward_shape(self):
        """Forward pass must return (video_pred, None) with the correct shape."""
        from extensions_built_in.diffusion_models.ltx2.ltx2_video_only_transformer import (
            LTX2VideoOnlyTransformer3DModel,
        )
        cfg = self._make_tiny_config()
        model = LTX2VideoOnlyTransformer3DModel.from_config(cfg)
        model.eval()

        B = 1
        num_video_tokens = 4  # H=2, W=2, F=1 => 1*2*2 = 4 tokens after patch_size=1
        in_channels = cfg["in_channels"]
        caption_channels = cfg["caption_channels"]
        inner_dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]

        hidden_states = torch.randn(B, num_video_tokens, in_channels)
        encoder_hidden_states = torch.randn(B, 3, caption_channels)
        timestep = torch.full((B,), 500.0)

        # Prepare minimal video_coords: shape (B, 3, num_tokens, 2)
        video_coords = torch.zeros(B, 3, num_video_tokens, 2)

        with torch.no_grad():
            output, audio_out = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_coords=video_coords,
                return_dict=False,
            )

        self.assertIsNone(audio_out, "Audio output should be None in video-only mode")
        self.assertEqual(output.shape, (B, num_video_tokens, in_channels))

    def test_forward_ignores_audio_kwargs(self):
        """Forward pass must accept and silently ignore audio-related kwargs."""
        from extensions_built_in.diffusion_models.ltx2.ltx2_video_only_transformer import (
            LTX2VideoOnlyTransformer3DModel,
        )
        cfg = self._make_tiny_config()
        model = LTX2VideoOnlyTransformer3DModel.from_config(cfg)
        model.eval()

        B = 1
        num_video_tokens = 4
        in_channels = cfg["in_channels"]
        caption_channels = cfg["caption_channels"]

        hidden_states = torch.randn(B, num_video_tokens, in_channels)
        encoder_hidden_states = torch.randn(B, 3, caption_channels)
        timestep = torch.full((B,), 500.0)
        video_coords = torch.zeros(B, 3, num_video_tokens, 2)
        dummy_audio = torch.randn(B, 10, 8)  # would be audio hidden states

        # This must NOT raise even though audio kwargs are passed
        with torch.no_grad():
            output, audio_out = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_coords=video_coords,
                return_dict=False,
                # These should be silently ignored:
                audio_hidden_states=dummy_audio,
                audio_encoder_hidden_states=encoder_hidden_states,
                audio_timestep=timestep,
                audio_coords=torch.zeros(B, 1, 10, 2),
                audio_num_frames=10,
            )
        self.assertIsNone(audio_out)
        self.assertEqual(output.shape, (B, num_video_tokens, in_channels))


class TestModelConfigLtx2Mode(unittest.TestCase):
    """Tests for the ltx2_mode ModelConfig flag."""

    def test_default_is_av(self):
        """Default ltx2_mode must be 'av' for backward compatibility."""
        from toolkit.config_modules import ModelConfig
        cfg = ModelConfig(name_or_path="dummy/path")
        self.assertEqual(cfg.ltx2_mode, "av")

    def test_video_mode(self):
        """ltx2_mode='video' must be accepted and stored."""
        from toolkit.config_modules import ModelConfig
        cfg = ModelConfig(name_or_path="dummy/path", ltx2_mode="video")
        self.assertEqual(cfg.ltx2_mode, "video")

    def test_av_mode_explicit(self):
        """ltx2_mode='av' must be accepted and stored."""
        from toolkit.config_modules import ModelConfig
        cfg = ModelConfig(name_or_path="dummy/path", ltx2_mode="av")
        self.assertEqual(cfg.ltx2_mode, "av")


class TestConvertLtx2TransformerVideoOnly(unittest.TestCase):
    """Tests for convert_ltx2_transformer(..., video_only=True)."""

    def test_video_only_flag_accepted(self):
        """convert_ltx2_transformer must accept video_only=True without error."""
        try:
            from extensions_built_in.diffusion_models.ltx2.convert_ltx2_to_diffusers import (
                convert_ltx2_transformer,
            )
        except ImportError:
            self.skipTest("diffusers not installed")

        # Build a minimal state dict that mimics a tiny AV checkpoint
        # (just the video-relevant keys the transformer expects)
        # We skip actual model conversion here – just verify the interface.
        import inspect
        sig = inspect.signature(convert_ltx2_transformer)
        self.assertIn("video_only", sig.parameters)


if __name__ == "__main__":
    unittest.main()
