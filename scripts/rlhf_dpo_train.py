#!/usr/bin/env python3
"""
RLHF-DPO Training Script for Z-Image models.

Uses Diffusion-DPO loss with a single model copy and LoRA multiplier toggling
to fit within 16 GB VRAM. Z-Image uses a transformer-based (DiT) architecture
with Qwen3 text encoder and flow matching — NOT a UNet/CLIP/DDPM model.

CLI Usage:
    python scripts/rlhf_dpo_train.py \
        --preference_data /path/to/preferences.json \
        --model_path Tongyi-MAI/Z-Image \
        --output_dir /path/to/output \
        --status_file /path/to/status.json \
        --beta 5000 --learning_rate 1e-5 --max_train_steps 2000 \
        --lora_rank 16 --gradient_checkpointing

Input preferences.json:
    [
        {"prompt": "a cat", "winner_path": "/path/win.png", "loser_path": "/path/lose.png"},
        ...
    ]

Output status.json (written every step):
    {"step": 150, "total_steps": 2000, "loss": 0.693, "status": "running"}
"""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file as safetensors_save_file
from torchvision import transforms


# ---------------------------------------------------------------------------
# Z-Image constants
# ---------------------------------------------------------------------------
ZIMAGE_HF_ID = "Tongyi-MAI/Z-Image"
ZIMAGE_VAE_SCALING_FACTOR = 0.3611
ZIMAGE_VAE_SHIFT_FACTOR = 0.1159
ZIMAGE_VAE_SCALE_FACTOR = 8  # spatial downsampling
ZIMAGE_NUM_TRAIN_TIMESTEPS = 1000
ZIMAGE_SCHEDULER_SHIFT = 3.0
ZIMAGE_MAX_SEQ_LEN = 512
ZIMAGE_LORA_TARGET_MODULES = ["ZImageTransformerBlock"]
ZIMAGE_LORA_EXCLUDE_PATTERNS = [r".*(_modulation|_refiner).*"]
SEQ_MULTI_OF = 32


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RLHF-DPO training for Z-Image")
    parser.add_argument("--preference_data", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True,
                        help="Local path or HuggingFace repo ID (e.g. Tongyi-MAI/Z-Image)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="",
                        help="Directory for latent/embedding caches (persists across runs)")
    parser.add_argument("--status_file", type=str, required=True)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--beta", type=float, default=5000.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--blocks_to_swap", type=int, default=16)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--status_every", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--quantize", type=str, default="none",
                        choices=["none", "qfloat8"],
                        help="Quantize base model weights (qfloat8 = FP8, auto-converts to torchao float8 when block swapping)")
    parser.add_argument("--control_file", type=str, default="",
                        help="Path to control.json for pause/resume/stop signalling")

    # Sample preview generation
    parser.add_argument("--sample_every", type=int, default=0,
                        help="Generate sample images every N steps (0 = disabled)")
    parser.add_argument("--sample_steps", type=int, default=25,
                        help="Number of inference steps for sample generation")
    parser.add_argument("--sample_guidance_scale", type=float, default=3.0,
                        help="CFG guidance scale for sample generation")
    parser.add_argument("--sample_width", type=int, default=1024)
    parser.add_argument("--sample_height", type=int, default=1024)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--sample_prompts_file", type=str, default="",
                        help="Path to JSON file containing array of prompt strings for sampling")
    parser.add_argument("--skip_first_sample", action="store_true",
                        help="Skip generating baseline samples before training starts")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Status file helpers
# ---------------------------------------------------------------------------

def write_status(path: str, step: int, total_steps: int, loss: float, status: str = "running", speed_string: str = ""):
    try:
        data = {"step": step, "total_steps": total_steps, "loss": loss, "status": status}
        if speed_string:
            data["speed_string"] = speed_string
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[rlhf] Warning: could not write status file: {e}", file=sys.stderr)


def append_loss_log(output_dir: str, step: int, loss: float):
    """Append a loss point to loss_log.jsonl for the loss graph UI."""
    try:
        log_path = os.path.join(output_dir, "loss_log.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps({"step": step, "loss": loss}) + "\n")
    except Exception as e:
        print(f"[rlhf] Warning: could not write loss log: {e}", file=sys.stderr)


def save_lora_weights(lora_network, ckpt_dir: str, dtype):
    """Save LoRA weights as safetensors with architecture-aware key names.

    Keys use the original model layer paths (e.g.
    'transformer.layers.0.attention.to_q.lora_down.weight') so that ComfyUI
    and other inference tools can map them back to the correct layers.
    """
    state_dict = {}
    for i, mod in enumerate(lora_network.lora_modules):
        layer_name = lora_network.module_layer_names[i]
        state_dict[f"transformer.{layer_name}.lora_down.weight"] = mod.lora_down.weight.data.to(dtype)
        state_dict[f"transformer.{layer_name}.lora_up.weight"] = mod.lora_up.weight.data.to(dtype)
        state_dict[f"transformer.{layer_name}.alpha"] = torch.tensor(mod.scale * mod.lora_down.in_features)
    save_path = os.path.join(ckpt_dir, "lora_weights.safetensors")
    safetensors_save_file(state_dict, save_path)


def check_control(control_file: str) -> str:
    """Read control.json and return the action: 'run', 'pause', 'resume', or 'stop'.

    Returns 'run' if file doesn't exist or can't be read.
    """
    if not control_file:
        return "run"
    try:
        if not os.path.exists(control_file):
            return "run"
        with open(control_file, "r") as f:
            data = json.load(f)
        return data.get("action", "run")
    except Exception:
        return "run"


# ---------------------------------------------------------------------------
# Model resolution / download helpers
# ---------------------------------------------------------------------------

def resolve_model_path(model_path: str) -> str:
    """Resolve a model path: if it's a local directory, use it; otherwise download from HuggingFace."""
    if os.path.isdir(model_path):
        print(f"[rlhf] Using local model: {model_path}")
        return model_path

    # Check if it looks like a HuggingFace repo ID (org/model)
    if "/" in model_path and not os.path.exists(model_path):
        from huggingface_hub import snapshot_download

        # Try local HF cache first (no network call)
        try:
            local_path = snapshot_download(model_path, local_files_only=True)
            print(f"[rlhf] Using cached model: {local_path}")
            return local_path
        except Exception:
            pass

        # Not in cache, download from HuggingFace
        print(f"[rlhf] Model path '{model_path}' not found locally, downloading from HuggingFace...")
        try:
            local_path = snapshot_download(model_path)
            print(f"[rlhf] Downloaded to: {local_path}")
            return local_path
        except Exception as e:
            print(f"[rlhf] ERROR: Failed to download model from HuggingFace: {e}", file=sys.stderr)
            raise

    return model_path


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess_image(image_path: str, resolution: int) -> torch.Tensor:
    """Load an image and return a tensor in [-1, 1] range, shape (1, C, H, W).

    Resolution is snapped to a multiple of VAE_SCALE_FACTOR * 2 (= 16 for Z-Image)
    to match the model's bucket divisibility requirement.
    """
    bucket_div = ZIMAGE_VAE_SCALE_FACTOR * 2  # 16
    resolution = (resolution // bucket_div) * bucket_div

    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return transform(img).unsqueeze(0)  # (1, 3, H, W)


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------

class LoRALinear(torch.nn.Module):
    def __init__(self, orig: torch.nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.orig = orig
        orig_dtype = orig.weight.dtype
        self.lora_down = torch.nn.Linear(orig.in_features, rank, bias=False, dtype=orig_dtype)
        self.lora_up = torch.nn.Linear(rank, orig.out_features, bias=False, dtype=orig_dtype)
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)
        self.scale = alpha / rank
        self.multiplier = 1.0
        # Freeze original weights
        self.orig.requires_grad_(False)

    def forward(self, x):
        orig_out = self.orig(x)
        if self.multiplier == 0.0:
            return orig_out
        lora_out = self.lora_up(self.lora_down(x)) * self.scale * self.multiplier
        return orig_out + lora_out


class SimpleLoRANetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_modules = torch.nn.ModuleList()
        self.module_layer_names = []  # original model layer path for each module
        self.multiplier = 1.0

    def set_multiplier(self, m):
        self.multiplier = m
        for mod in self.lora_modules:
            mod.multiplier = m

    def trainable_params(self):
        params = []
        for mod in self.lora_modules:
            params.extend(mod.lora_down.parameters())
            params.extend(mod.lora_up.parameters())
        return params


def create_lora_network(transformer, lora_rank: int, lora_alpha: float = None):
    """Create a LoRA network targeting ZImageTransformerBlock layers.

    Injects LoRA adapters into all Linear layers within ZImageTransformerBlock
    modules (excluding modulation/refiner layers). Returns the network with
    layer name tracking for correct weight serialization.
    """
    if lora_alpha is None:
        lora_alpha = float(lora_rank)

    network = SimpleLoRANetwork()

    # Collect all Linear layers that live inside target block types,
    # recursing through the full module tree.
    target_block_prefixes = []
    for name, module in transformer.named_modules():
        is_target = any(t in type(module).__name__ for t in ZIMAGE_LORA_TARGET_MODULES)
        if not is_target:
            continue
        is_excluded = any(re.match(p, name) for p in ZIMAGE_LORA_EXCLUDE_PATTERNS)
        if is_excluded:
            continue
        target_block_prefixes.append(name + ".")

    for full_name, module in transformer.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        # Check this Linear lives inside a target block
        if not any(full_name.startswith(prefix) for prefix in target_block_prefixes):
            continue
        # Check exclusion patterns
        if any(re.match(p, full_name) for p in ZIMAGE_LORA_EXCLUDE_PATTERNS):
            continue
        # Navigate to the parent module and replace
        parts = full_name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, child_attr = parts
            parent = dict(transformer.named_modules())[parent_name]
        else:
            parent = transformer
            child_attr = full_name
        lora_linear = LoRALinear(module, lora_rank, lora_alpha)
        setattr(parent, child_attr, lora_linear)
        network.lora_modules.append(lora_linear)
        network.module_layer_names.append(full_name)

    print(f"[rlhf] Created LoRA network (rank={lora_rank}, alpha={lora_alpha}, {len(network.lora_modules)} modules)")
    return network


# ---------------------------------------------------------------------------
# Cache pre-check helpers
# ---------------------------------------------------------------------------

def all_latents_cached(image_paths, cache_dir):
    """Check if all image latents are already cached on disk."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return False
    for img_path in set(image_paths):
        path_hash = hashlib.sha256(img_path.encode()).hexdigest()[:12]
        if not (cache_dir / f"lat_{path_hash}.pt").exists():
            return False
    return True


def all_embeddings_cached(prompts, cache_dir):
    """Check if all prompt embeddings are already cached on disk."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return False
    for prompt in set(prompts):
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        if not (cache_dir / f"emb_{prompt_hash}.pt").exists():
            return False
    return True


def load_latents_from_cache(image_paths, cache_dir):
    """Load all latents from disk cache without needing the VAE."""
    cache_dir = Path(cache_dir)
    latents = {}
    unique_paths = list(dict.fromkeys(image_paths))
    for i, img_path in enumerate(unique_paths, 1):
        path_hash = hashlib.sha256(img_path.encode()).hexdigest()[:12]
        cache_path = cache_dir / f"lat_{path_hash}.pt"
        latents[img_path] = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"[rlhf]   latent {i}/{len(unique_paths)} (cached) {os.path.basename(img_path)}")
    return latents


def load_embeddings_from_cache(prompts, cache_dir):
    """Load all embeddings from disk cache without needing the text encoder."""
    cache_dir = Path(cache_dir)
    embeddings = {}
    unique_prompts = list(dict.fromkeys(prompts))
    for i, prompt in enumerate(unique_prompts, 1):
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        cache_path = cache_dir / f"emb_{prompt_hash}.pt"
        embeddings[prompt] = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"[rlhf]   embedding {i}/{len(unique_prompts)} (cached) {prompt[:60]}")
    return embeddings


# ---------------------------------------------------------------------------
# VAE latent caching — Z-Image uses different scaling
# ---------------------------------------------------------------------------

def encode_images_to_latents(vae, image_paths, resolution, device, cache_dir):
    """Encode images to latents using Z-Image VAE and cache to disk.

    Z-Image latents are scaled: model_latents = (vae_latents - shift) * scale
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    latents = {}
    unique_paths = list(dict.fromkeys(image_paths))  # deduplicate, preserve order
    total = len(unique_paths)

    for i, img_path in enumerate(unique_paths, 1):
        # Use full path hash for cache key to handle duplicate filenames
        path_hash = hashlib.sha256(img_path.encode()).hexdigest()[:12]
        cache_path = cache_dir / f"lat_{path_hash}.pt"
        if cache_path.exists():
            latents[img_path] = torch.load(cache_path, map_location="cpu", weights_only=True)
            print(f"[rlhf]   latent {i}/{total} (cached) {os.path.basename(img_path)}")
            continue

        print(f"[rlhf]   latent {i}/{total} encoding {os.path.basename(img_path)}")
        img_tensor = load_and_preprocess_image(img_path, resolution).to(device, dtype=torch.float32)
        with torch.no_grad():
            # Z-Image VAE expects float32 input
            posterior = vae.encode(img_tensor)
            if hasattr(posterior, 'latent_dist'):
                latent = posterior.latent_dist.sample()
            else:
                latent = posterior.sample()
            # Apply Z-Image scaling: (latent - shift) * scale
            latent = (latent - ZIMAGE_VAE_SHIFT_FACTOR) * ZIMAGE_VAE_SCALING_FACTOR

        latents[img_path] = latent.cpu()
        torch.save(latents[img_path], cache_path)

    return latents


# ---------------------------------------------------------------------------
# Text embedding caching — Z-Image uses Qwen3
# ---------------------------------------------------------------------------

def encode_prompts(tokenizer, text_encoder, prompts, device, cache_dir):
    """Encode prompts using Qwen3 text encoder (Z-Image's text encoder).

    Returns dict mapping prompt -> (embed, mask) where:
      embed: (1, seq_len, 2560)
      mask: (1, seq_len) bool
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings = {}
    unique_prompts = list(dict.fromkeys(prompts))  # deduplicate, preserve order
    total = len(unique_prompts)

    for i, prompt in enumerate(unique_prompts, 1):
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        cache_path = cache_dir / f"emb_{prompt_hash}.pt"
        if cache_path.exists():
            embeddings[prompt] = torch.load(cache_path, map_location="cpu", weights_only=True)
            print(f"[rlhf]   embedding {i}/{total} (cached) {prompt[:60]}")
            continue

        print(f"[rlhf]   embedding {i}/{total} encoding {prompt[:60]}")
        # Apply chat template (required for Qwen3)
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )

        inputs = tokenizer(
            [formatted],
            padding="max_length",
            max_length=ZIMAGE_MAX_SEQ_LEN,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device).bool()

        with torch.no_grad():
            # Use second-to-last hidden state (matches Z-Image convention)
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            embed = outputs.hidden_states[-2]

        data = {"embed": embed.cpu(), "mask": attention_mask.cpu()}
        embeddings[prompt] = data
        torch.save(data, cache_path)

    return embeddings


# ---------------------------------------------------------------------------
# Flow matching noise utilities for Z-Image
# ---------------------------------------------------------------------------

def flow_match_add_noise(latents, noise, sigma):
    """Add noise using flow matching interpolation: noisy = (1 - sigma) * latents + sigma * noise."""
    return (1 - sigma) * latents + sigma * noise


def sample_timestep_sigma(batch_size: int = 1):
    """Sample a random timestep in [1, 1000] and compute the corresponding sigma with shift."""
    t = torch.randint(1, ZIMAGE_NUM_TRAIN_TIMESTEPS + 1, (batch_size,), dtype=torch.float32)
    sigma = t / ZIMAGE_NUM_TRAIN_TIMESTEPS
    # Apply shift (same schedule used by Z-Image)
    sigma = ZIMAGE_SCHEDULER_SHIFT * sigma / (1 + (ZIMAGE_SCHEDULER_SHIFT - 1) * sigma)
    return t, sigma


# ---------------------------------------------------------------------------
# DPO loss computation — adapted for Z-Image flow matching
# ---------------------------------------------------------------------------

def compute_dpo_loss(
    transformer,
    lora_network,
    winner_latent: torch.Tensor,
    loser_latent: torch.Tensor,
    prompt_embed: torch.Tensor,
    prompt_mask,
    sigma: torch.Tensor,
    timestep: torch.Tensor,
    noise: torch.Tensor,
    beta: float,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """
    Computes Diffusion-DPO loss for Z-Image (flow matching variant):
        L = -log sigma_fn(beta * (ref_w_err - policy_w_err - ref_l_err + policy_l_err))
    where _err is MSE(model_prediction, target).

    For flow matching, target = latents - noise (velocity prediction).

    Uses LoRA multiplier toggling: multiplier=0 for reference, multiplier=1 for policy.
    """
    winner_latent = winner_latent.to(device, dtype=dtype)
    loser_latent = loser_latent.to(device, dtype=dtype)
    prompt_embed = prompt_embed.to(device, dtype=dtype)
    noise = noise.to(device, dtype=dtype)

    # Noise both latents with the same noise and timestep
    sigma_d = sigma.to(device, dtype=dtype)
    noisy_winner = flow_match_add_noise(winner_latent, noise, sigma_d)
    noisy_loser = flow_match_add_noise(loser_latent, noise, sigma_d)

    # Flow matching target: latents - noise (velocity)
    target_winner = winner_latent - noise
    target_loser = loser_latent - noise

    # Add frame dimension for Z-Image transformer: [B, C, H, W] -> [C, 1, H, W] per sample
    # Diffusers ZImageTransformer2DModel expects x as List[Tensor] with shape (C, F, H, W)
    noisy_winner_list = [noisy_winner[i].unsqueeze(1) for i in range(noisy_winner.shape[0])]
    noisy_loser_list = [noisy_loser[i].unsqueeze(1) for i in range(noisy_loser.shape[0])]

    # Timestep for Z-Image: (1000 - t) / 1000, model internally multiplies by t_scale=1000
    t_input = ((1000.0 - timestep.to(device, dtype=dtype)) / 1000.0)

    # cap_feats as List[Tensor] with shape (seq_len, dim) per sample
    cap_feats_list = [prompt_embed[i] for i in range(prompt_embed.shape[0])]

    def _call_transformer(x_list, t_val, cap_list):
        """Call the diffusers ZImageTransformer2DModel and extract output tensor."""
        out = transformer(x=x_list, t=t_val, cap_feats=cap_list, return_dict=False)
        # out is a tuple; out[0] is a list of tensors (C, F, H, W) per sample
        preds = out[0]
        # Stack into batch and remove frame dim: [B, C, H, W]
        return torch.stack([p.squeeze(1) for p in preds], dim=0)

    # --- Reference predictions (LoRA multiplier = 0, no gradient) ---
    lora_network.set_multiplier(0.0)
    with torch.no_grad():
        ref_winner_pred = _call_transformer(noisy_winner_list, t_input, cap_feats_list)
        ref_loser_pred = _call_transformer(noisy_loser_list, t_input, cap_feats_list)

    ref_w_err = F.mse_loss(ref_winner_pred.float(), target_winner.float(), reduction="mean")
    ref_l_err = F.mse_loss(ref_loser_pred.float(), target_loser.float(), reduction="mean")

    # --- Policy predictions (LoRA multiplier = 1, with gradient) ---
    lora_network.set_multiplier(1.0)
    policy_winner_pred = _call_transformer(noisy_winner_list, t_input, cap_feats_list)
    policy_loser_pred = _call_transformer(noisy_loser_list, t_input, cap_feats_list)

    policy_w_err = F.mse_loss(policy_winner_pred.float(), target_winner.float(), reduction="mean")
    policy_l_err = F.mse_loss(policy_loser_pred.float(), target_loser.float(), reduction="mean")

    # DPO reward margin and loss
    margin = beta * (ref_w_err - policy_w_err - ref_l_err + policy_l_err)
    loss = -F.logsigmoid(margin)
    return loss


# ---------------------------------------------------------------------------
# Sample preview generation
# ---------------------------------------------------------------------------

def generate_samples(
    transformer,
    lora_network,
    sample_embed_cache: dict,
    sample_prompts: list,
    step: int,
    output_dir: str,
    vae_path: str,
    num_steps: int = 25,
    guidance_scale: float = 3.0,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    dtype=torch.bfloat16,
    device: str = "cuda",
    status_file: str = "",
    total_steps: int = 0,
    running_loss: float = 0.0,
):
    """Generate sample preview images using the current LoRA weights.

    Uses Euler discrete sampling with the shifted flow-matching schedule.
    VAE is loaded temporarily for decoding, then unloaded to free VRAM.
    """
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Write sampling status
    if status_file:
        write_status(status_file, step, total_steps, running_loss, "sampling")

    # Set LoRA to policy mode and switch to eval
    lora_network.set_multiplier(1.0)
    transformer.eval()

    latent_h = height // ZIMAGE_VAE_SCALE_FACTOR
    latent_w = width // ZIMAGE_VAE_SCALE_FACTOR
    latent_c = 16  # Z-Image latent channels

    # Build sigma schedule: linearly space num_steps+1 values from 1→0
    # Then apply shift: shifted = shift * s / (1 + (shift-1) * s)
    sigmas_unshifted = torch.linspace(1.0, 0.0, num_steps + 1)
    sigmas_shifted = ZIMAGE_SCHEDULER_SHIFT * sigmas_unshifted / (1 + (ZIMAGE_SCHEDULER_SHIFT - 1) * sigmas_unshifted)

    # Get negative embedding for CFG if needed
    neg_embed = None
    if guidance_scale > 1.0 and "" in sample_embed_cache:
        neg_embed = sample_embed_cache[""]

    denoised_latents = []

    print(f"\n[rlhf] ==== GENERATING SAMPLES at step {step} ====")
    print(f"[rlhf]   {len(sample_prompts)} prompts, {num_steps} steps, cfg={guidance_scale}, "
          f"{width}x{height}, seed={seed}")

    # Diagnostic: check LoRA weights for NaN/Inf
    lora_nan = False
    for name, param in lora_network.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"[rlhf]   WARNING: NaN/Inf detected in LoRA param {name}!")
            lora_nan = True
    if lora_nan:
        print(f"[rlhf]   ABORTING sample generation due to NaN/Inf in LoRA weights")
        transformer.train()
        if status_file:
            write_status(status_file, step, total_steps, running_loss, "running")
        return
    else:
        # Log LoRA weight magnitude for training diagnostics
        lora_norms = []
        for name, param in lora_network.named_parameters():
            if param.requires_grad:
                lora_norms.append(param.data.float().norm().item())
        if lora_norms:
            avg_norm = sum(lora_norms) / len(lora_norms)
            max_norm = max(lora_norms)
            print(f"[rlhf]   LoRA weights OK — avg_norm={avg_norm:.6f}, max_norm={max_norm:.6f}")

    with torch.no_grad():
        for prompt_idx, prompt in enumerate(sample_prompts):
            emb_data = sample_embed_cache.get(prompt)
            if emb_data is None:
                print(f"[rlhf]   Skipping prompt {prompt_idx} (no embedding): {prompt[:60]}")
                continue

            embed = emb_data["embed"].to(device, dtype=dtype)
            mask = emb_data["mask"]
            # Trim to actual length
            actual_len = int(mask.sum(dim=1).item())
            embed = embed[:, :actual_len, :]

            # Start from random noise with deterministic seed
            generator = torch.Generator(device=device).manual_seed(seed + prompt_idx)
            latents = torch.randn(1, latent_c, latent_h, latent_w, generator=generator,
                                  device=device, dtype=dtype)

            # Euler discrete sampling loop
            for i in range(num_steps):
                sigma = sigmas_shifted[i]
                sigma_next = sigmas_shifted[i + 1]

                # Timestep input: (1 - sigma_shifted) — the pipeline computes
                # timestep = (1000 - t) / 1000 where t = sigma_shifted * 1000
                t_input = (1.0 - sigma).to(device, dtype=dtype).unsqueeze(0)

                # Prepare transformer input: add frame dimension
                latent_input = [latents[0].unsqueeze(1)]  # [(C, 1, H, W)]
                cap_feats = [embed[0]]  # [(seq_len, dim)]

                # Forward pass — raw output predicts (data − noise).
                # Pipeline negates before scheduler step (pipeline_z_image.py:558).
                out = transformer(x=latent_input, t=t_input, cap_feats=cap_feats, return_dict=False)
                pred = out[0][0].squeeze(1).unsqueeze(0)  # (1, C, H, W)

                # CFG: pipeline uses cond + scale * (cond - uncond)
                if guidance_scale > 1.0 and neg_embed is not None:
                    neg_e = neg_embed["embed"].to(device, dtype=dtype)
                    neg_m = neg_embed["mask"]
                    neg_len = int(neg_m.sum(dim=1).item())
                    neg_e = neg_e[:, :neg_len, :]
                    neg_input = [latents[0].unsqueeze(1)]
                    neg_cap = [neg_e[0]]
                    out_neg = transformer(x=neg_input, t=t_input, cap_feats=neg_cap, return_dict=False)
                    pred_neg = out_neg[0][0].squeeze(1).unsqueeze(0)
                    pred = pred + guidance_scale * (pred - pred_neg)

                # Negate (matches pipeline_z_image.py:558) then Euler step
                velocity = -pred
                dt = (sigma_next - sigma).to(device, dtype=dtype)
                latents = latents + dt * velocity

                # Diagnostic: log stats at first, middle, and last step
                if i == 0 or i == num_steps // 2 or i == num_steps - 1:
                    lat_min, lat_max = latents.min().item(), latents.max().item()
                    lat_nan = torch.isnan(latents).any().item()
                    pred_nan = torch.isnan(pred).any().item()
                    print(f"[rlhf]     step {i}/{num_steps}: sigma={sigma:.4f} "
                          f"latent=[{lat_min:.3f}, {lat_max:.3f}] "
                          f"nan_lat={lat_nan} nan_pred={pred_nan}")

            denoised_latents.append(latents.cpu())
            print(f"[rlhf]   Prompt {prompt_idx}: denoised ({prompt[:50]})")

    # Load VAE temporarily for decoding
    if denoised_latents:
        # Free cached GPU memory before loading VAE
        gc.collect()
        torch.cuda.empty_cache()
        log_vram("before VAE load for decode")

        print(f"[rlhf]   Loading VAE for decode...")
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae", torch_dtype=torch.float32)
        vae = vae.to(device)
        vae.eval()

        with torch.no_grad():
            for prompt_idx, latent in enumerate(denoised_latents):
                latent = latent.to(device, dtype=torch.float32)
                # Reverse Z-Image scaling: latent / scale + shift
                latent = latent / ZIMAGE_VAE_SCALING_FACTOR + ZIMAGE_VAE_SHIFT_FACTOR
                decoded = vae.decode(latent, return_dict=False)[0]
                # Clamp to [0, 1] and save
                img = decoded.squeeze(0).clamp(-1, 1).permute(1, 2, 0).cpu().float()
                img = ((img + 1) / 2 * 255).clamp(0, 255).byte().numpy()
                img_pil = Image.fromarray(img)
                img_path = os.path.join(samples_dir, f"{step:06d}_{prompt_idx:02d}.jpg")
                img_pil.save(img_path, quality=90)
                print(f"[rlhf]   Saved {os.path.basename(img_path)}")

        del vae
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[rlhf]   VAE unloaded")

    # Restore training state
    transformer.train()
    if status_file:
        write_status(status_file, step, total_steps, running_loss, "running")
    print(f"[rlhf] ==== SAMPLES COMPLETE ====\n")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def format_bytes(n: int) -> str:
    """Format byte count to human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format seconds to human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"


def log_vram(label: str = ""):
    """Log current GPU VRAM usage."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    suffix = f" [{label}]" if label else ""
    print(f"[rlhf]   VRAM: {format_bytes(allocated)} allocated, "
          f"{format_bytes(reserved)} reserved, "
          f"{format_bytes(total)} total{suffix}")


def log_gpu_info():
    """Log GPU hardware details."""
    if not torch.cuda.is_available():
        print("[rlhf] CUDA not available — running on CPU")
        return
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"[rlhf]   GPU {i}: {props.name}  "
              f"{format_bytes(props.total_memory)} VRAM  "
              f"compute={props.major}.{props.minor}  "
              f"SMs={props.multi_processor_count}")


def main():
    train_start_time = time.time()
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    write_status(args.status_file, 0, args.max_train_steps, 0.0, "starting")

    # -------------------------------------------------------------------
    # Configuration dump
    # -------------------------------------------------------------------
    print("=" * 70)
    print("  RLHF-DPO Training for Z-Image")
    print("=" * 70)
    print(f"[rlhf] Training configuration:")
    print(f"[rlhf]   model_path        = {args.model_path}")
    print(f"[rlhf]   output_dir        = {args.output_dir}")
    print(f"[rlhf]   preference_data   = {args.preference_data}")
    print(f"[rlhf]   run_id            = {args.run_id or '(none)'}")
    print(f"[rlhf]   beta              = {args.beta}")
    print(f"[rlhf]   learning_rate     = {args.learning_rate:.2e}")
    print(f"[rlhf]   max_train_steps   = {args.max_train_steps}")
    print(f"[rlhf]   lora_rank         = {args.lora_rank}")
    print(f"[rlhf]   blocks_to_swap    = {args.blocks_to_swap}")
    print(f"[rlhf]   save_every        = {args.save_every}")
    print(f"[rlhf]   status_every      = {args.status_every}")
    print(f"[rlhf]   resolution        = {args.resolution}")
    print(f"[rlhf]   mixed_precision   = {args.mixed_precision}")
    print(f"[rlhf]   grad_checkpointing= {args.gradient_checkpointing}")
    print(f"[rlhf]   quantize          = {args.quantize}")
    print(f"[rlhf]   control_file      = {args.control_file or '(none)'}")
    print(f"[rlhf]   sample_every      = {args.sample_every}")
    if args.sample_every > 0:
        print(f"[rlhf]   sample_steps      = {args.sample_steps}")
        print(f"[rlhf]   sample_cfg        = {args.sample_guidance_scale}")
        print(f"[rlhf]   sample_size       = {args.sample_width}x{args.sample_height}")
        print(f"[rlhf]   sample_seed       = {args.sample_seed}")
        print(f"[rlhf]   sample_prompts    = {args.sample_prompts_file or '(none)'}")

    # -------------------------------------------------------------------
    # Hardware info
    # -------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"[rlhf] Hardware:")
    print(f"[rlhf]   device            = {device}")
    print(f"[rlhf]   dtype             = {dtype}")
    print(f"[rlhf]   torch version     = {torch.__version__}")
    print(f"[rlhf]   CUDA available    = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[rlhf]   CUDA version      = {torch.version.cuda}")
        print(f"[rlhf]   cuDNN version     = {torch.backends.cudnn.version()}")
        print(f"[rlhf]   GPU count         = {torch.cuda.device_count()}")
        log_gpu_info()
    log_vram("startup")
    print("-" * 70)

    # Load preference data
    with open(args.preference_data) as f:
        preferences = json.load(f)

    if not preferences:
        print("[rlhf] ERROR: No preference data found", file=sys.stderr)
        write_status(args.status_file, 0, args.max_train_steps, 0.0, "error")
        sys.exit(1)

    unique_prompts = len(set(p["prompt"] for p in preferences))
    print(f"[rlhf] Loaded {len(preferences)} preference pairs ({unique_prompts} unique prompts)")

    # -----------------------------------------------------------------------
    # Determine what needs loading vs what's fully cached
    # -----------------------------------------------------------------------
    all_image_paths = []
    for p in preferences:
        all_image_paths.append(p["winner_path"])
        all_image_paths.append(p["loser_path"])
    prompts = [p["prompt"] for p in preferences]
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(args.output_dir, "cache")

    # Load sample prompts if configured
    sample_prompts_list = []
    if args.sample_every > 0 and args.sample_prompts_file and os.path.exists(args.sample_prompts_file):
        with open(args.sample_prompts_file) as f:
            sample_prompts_list = json.load(f)
        print(f"[rlhf] Loaded {len(sample_prompts_list)} sample prompts for preview generation")

    # Merge sample prompts (+ empty string for CFG) into the set to encode
    all_prompts_to_encode = list(prompts)
    if sample_prompts_list:
        all_prompts_to_encode.extend(sample_prompts_list)
        if args.sample_guidance_scale > 1.0:
            all_prompts_to_encode.append("")  # negative/unconditional embedding for CFG

    latents_fully_cached = all_latents_cached(all_image_paths, cache_dir)
    embeddings_fully_cached = all_embeddings_cached(all_prompts_to_encode, cache_dir)

    if latents_fully_cached:
        print(f"[rlhf] All {len(set(all_image_paths))} latents found in cache — skipping VAE load")
    if embeddings_fully_cached:
        print(f"[rlhf] All {len(set(all_prompts_to_encode))} embeddings found in cache — skipping text encoder load")

    # -----------------------------------------------------------------------
    # Resolve model path (needed for all loading)
    # -----------------------------------------------------------------------
    model_path = resolve_model_path(args.model_path)
    print(f"[rlhf] Model path: {model_path}")

    try:
        from diffusers.models.transformers import ZImageTransformer2DModel
        if not latents_fully_cached:
            from diffusers import AutoencoderKL
        if not embeddings_fully_cached:
            from transformers import AutoTokenizer, Qwen3ForCausalLM
    except ImportError as e:
        print(f"[rlhf] ERROR: Missing dependencies. Install with: pip install diffusers transformers", file=sys.stderr)
        print(f"[rlhf]   Import error: {e}", file=sys.stderr)
        write_status(args.status_file, 0, args.max_train_steps, 0.0, "error")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Phase 1: Cache latents and embeddings BEFORE loading transformer.
    # This way VAE and text encoder are fully unloaded (CPU + GPU) before
    # the large transformer ever touches memory.
    # -----------------------------------------------------------------------

    # --- Latents ---
    t0 = time.time()
    unique_images = len(set(all_image_paths))
    write_status(args.status_file, 0, args.max_train_steps, 0.0, "caching_latents")

    if latents_fully_cached:
        print(f"[rlhf] Loading {unique_images} latents from cache...")
        latents_cache = load_latents_from_cache(all_image_paths, cache_dir)
    else:
        print(f"[rlhf] Encoding {len(all_image_paths)} images to latents ({unique_images} unique)...")
        try:
            vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
            print(f"[rlhf]   VAE loaded (float32)")
            vae = vae.to(device)
            log_vram("VAE on GPU")
            latents_cache = encode_images_to_latents(vae, all_image_paths, args.resolution, device, cache_dir)
        finally:
            del vae
            gc.collect()
            torch.cuda.empty_cache()
            log_vram("after VAE unload")

    sample_latent = next(iter(latents_cache.values()))
    print(f"[rlhf]   Latent shape: {list(sample_latent.shape)}, dtype={sample_latent.dtype}")
    print(f"[rlhf]   {len(latents_cache)} latents ready in {time.time() - t0:.1f}s")

    # --- Embeddings ---
    t0 = time.time()
    write_status(args.status_file, 0, args.max_train_steps, 0.0, "caching_embeddings")

    if embeddings_fully_cached:
        print(f"[rlhf] Loading {len(set(all_prompts_to_encode))} embeddings from cache...")
        all_embed_cache = load_embeddings_from_cache(all_prompts_to_encode, cache_dir)
    else:
        print(f"[rlhf] Encoding {len(set(all_prompts_to_encode))} unique prompts with Qwen3...")
        try:
            text_encoder = Qwen3ForCausalLM.from_pretrained(
                model_path, subfolder="text_encoder", torch_dtype=dtype
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
            num_te_params = sum(p.numel() for p in text_encoder.parameters())
            print(f"[rlhf]   Text encoder loaded ({num_te_params:,} params, vocab_size={tokenizer.vocab_size})")
            text_encoder = text_encoder.to(device)
            text_encoder.eval()
            log_vram("text encoder on GPU")
            all_embed_cache = encode_prompts(tokenizer, text_encoder, all_prompts_to_encode, device, cache_dir)
        finally:
            del text_encoder, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            log_vram("after text encoder unload")

    # Split into training embed_cache and sample_embed_cache
    embed_cache = {p: all_embed_cache[p] for p in set(prompts) if p in all_embed_cache}
    sample_embed_cache = {}
    if sample_prompts_list:
        for p in sample_prompts_list:
            if p in all_embed_cache:
                sample_embed_cache[p] = all_embed_cache[p]
        if "" in all_embed_cache:
            sample_embed_cache[""] = all_embed_cache[""]
        print(f"[rlhf]   Sample embed cache: {len(sample_embed_cache)} entries")

    sample_emb = next(iter(embed_cache.values()))
    print(f"[rlhf]   Embedding shape: {list(sample_emb['embed'].shape)}, dtype={sample_emb['embed'].dtype}")
    print(f"[rlhf]   {len(embed_cache)} training embeddings ready in {time.time() - t0:.1f}s")
    print("-" * 70)

    # -----------------------------------------------------------------------
    # Phase 2: Load transformer into a clean memory state.
    # VAE and text encoder are completely gone at this point.
    # -----------------------------------------------------------------------
    write_status(args.status_file, 0, args.max_train_steps, 0.0, "loading_model")

    try:
        t0 = time.time()
        print(f"[rlhf] Loading transformer...")
        transformer_path = os.path.join(model_path, "transformer")
        if os.path.isdir(transformer_path):
            print(f"[rlhf]   from_pretrained: path={transformer_path}, dtype={dtype}")
            transformer = ZImageTransformer2DModel.from_pretrained(
                transformer_path, torch_dtype=dtype
            )
        else:
            print(f"[rlhf]   from_pretrained: path={model_path}, subfolder=transformer, dtype={dtype}")
            transformer = ZImageTransformer2DModel.from_pretrained(
                model_path, subfolder="transformer", torch_dtype=dtype
            )
        num_transformer_params = sum(p.numel() for p in transformer.parameters())
        print(f"[rlhf]   Transformer loaded in {time.time() - t0:.1f}s "
              f"({num_transformer_params:,} params, "
              f"{format_bytes(sum(p.numel() * p.element_size() for p in transformer.parameters()))})")
        log_vram("after transformer load")
    except Exception as e:
        print(f"[rlhf] ERROR loading model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        write_status(args.status_file, 0, args.max_train_steps, 0.0, "error")
        sys.exit(1)

    print("-" * 70)

    # -----------------------------------------------------------------------
    # Quantize transformer (FP8) — before LoRA so LoRA wraps quantized layers
    # -----------------------------------------------------------------------
    if args.quantize != "none":
        t0 = time.time()
        qtype_name = args.quantize  # "qfloat8"

        # Auto-convert qfloat8 → float8 (torchao) when block swapping is active.
        # quanto QTensors don't transfer cleanly between CPU/GPU.
        # (Same pattern as config_modules.py)
        if args.blocks_to_swap > 0 and qtype_name == "qfloat8":
            qtype_name = "float8"
            print(f"[rlhf] Quantization: auto-converted qfloat8 → float8 (torchao) for block swapping compatibility")

        # Add toolkit root to sys.path so we can import toolkit.util.quantize
        toolkit_root = os.path.join(os.path.dirname(__file__), "..")
        if toolkit_root not in sys.path:
            sys.path.insert(0, toolkit_root)

        from toolkit.util.quantize import quantize as qt_quantize, get_qtype
        from optimum.quanto import freeze

        quantization_type = get_qtype(qtype_name)
        print(f"[rlhf] Quantizing transformer to {qtype_name}...")

        # Quantize block-by-block: move each block to GPU → quantize → freeze → move back to CPU
        # This avoids OOM from quantizing the entire model on GPU at once.
        block_attrs = ["layers", "noise_refiner", "context_refiner"]
        all_blocks = []
        for attr in block_attrs:
            bl = getattr(transformer, attr, None)
            if bl is not None:
                all_blocks.extend(list(bl))

        print(f"[rlhf]   Quantizing {len(all_blocks)} transformer blocks...")
        for i, block in enumerate(all_blocks):
            block.to(device, dtype=dtype)
            qt_quantize(block, weights=quantization_type)
            freeze(block)
            block.to("cpu")

        # Quantize remaining non-block submodules (embedders, norms, etc.)
        # These are small and stay on GPU permanently (block-swapping only
        # offloads blocks). We leave them on GPU after quantizing because
        # torchao AffineQuantizedTensor doesn't move with .to(device).
        print(f"[rlhf]   Quantizing remaining layers (embedders, norms)...")
        block_child_names = set()
        for attr in block_attrs:
            bl = getattr(transformer, attr, None)
            if bl is not None:
                block_child_names.add(attr)
        for name, child in transformer.named_children():
            if name not in block_child_names:
                child.to(device, dtype=dtype)
                qt_quantize(child, weights=quantization_type)
                freeze(child)

        num_params_after = sum(p.numel() for p in transformer.parameters())
        model_bytes = sum(p.numel() * p.element_size() for p in transformer.parameters())
        print(f"[rlhf]   Quantization complete in {time.time() - t0:.1f}s "
              f"({num_params_after:,} params, {format_bytes(model_bytes)})")
        log_vram("after quantization")
        print("-" * 70)

    # -----------------------------------------------------------------------
    # Create LoRA network on transformer
    # -----------------------------------------------------------------------
    t0 = time.time()
    print(f"[rlhf] Creating LoRA adapter (rank={args.lora_rank}, alpha={args.lora_rank})")
    lora_network = create_lora_network(transformer, args.lora_rank)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        print(f"[rlhf]   Gradient checkpointing enabled")
    print(f"[rlhf]   LoRA created in {time.time() - t0:.1f}s")
    log_vram("after LoRA creation")
    print("-" * 70)

    # -----------------------------------------------------------------------
    # Move transformer to device (with optional block swapping for low VRAM)
    # -----------------------------------------------------------------------
    t0 = time.time()
    blocks_to_swap = args.blocks_to_swap
    if blocks_to_swap > 0:
        # Block swapping: keep N blocks on CPU, move to GPU on-the-fly during
        # forward/backward.  The non-block parameters (patch embed, final norm,
        # etc.) always stay on GPU since they are small.
        print(f"[rlhf] Block swapping enabled: {blocks_to_swap} blocks on CPU")

        # Identify the block lists.
        # ZImageTransformer2DModel uses: layers, noise_refiner, context_refiner
        block_attrs = ["layers", "noise_refiner", "context_refiner"]
        all_blocks = []
        for attr in block_attrs:
            bl = getattr(transformer, attr, None)
            if bl is not None:
                all_blocks.extend(list(bl))

        n_swap = min(blocks_to_swap, len(all_blocks))
        gpu_blocks = all_blocks[: len(all_blocks) - n_swap]
        cpu_blocks = all_blocks[len(all_blocks) - n_swap :]
        print(f"[rlhf]   Total blocks: {len(all_blocks)}, on GPU: {len(gpu_blocks)}, on CPU: {len(cpu_blocks)}")

        if args.quantize != "none":
            # When quantized, non-block child modules are already on GPU from
            # the quantization step and torchao tensors don't survive .to() moves.
            # Move blocks to their designated devices, and ensure any top-level
            # params/buffers (e.g. pad_token) not inside child modules are on GPU.
            block_prefixes = tuple(a + "." for a in block_attrs)
            for name, param in transformer.named_parameters():
                if not name.startswith(block_prefixes):
                    param.data = param.data.to(device)
            for name, buf in transformer.named_buffers():
                if not name.startswith(block_prefixes):
                    buf.data = buf.data.to(device)
            for block in cpu_blocks:
                block.to("cpu")
            for block in gpu_blocks:
                block.to(device)
        else:
            # Start with everything on CPU, then selectively move to GPU
            transformer = transformer.to("cpu")

            # Move non-block params (embedders, norms, etc.) to GPU
            block_prefixes = tuple(a + "." for a in block_attrs)
            for name, param in transformer.named_parameters():
                if not name.startswith(block_prefixes):
                    param.data = param.data.to(device)
            for name, buf in transformer.named_buffers():
                if not name.startswith(block_prefixes):
                    buf.data = buf.data.to(device)

            # Move GPU-resident blocks
            for block in gpu_blocks:
                block.to(device)

        # Register forward hooks to swap CPU blocks to/from GPU on demand
        def _make_pre_hook(block_ref):
            def hook(module, args):
                block_ref.to(device)
            return hook
        def _make_post_hook(block_ref):
            def hook(module, args, output):
                block_ref.to("cpu")
            return hook
        for block in cpu_blocks:
            block.register_forward_pre_hook(_make_pre_hook(block))
            block.register_forward_hook(_make_post_hook(block))

        # Don't call lora_network.to(device) here — LoRA weights live inside
        # the transformer blocks and are already on the correct device.
        # GPU blocks have their LoRA on GPU, CPU blocks have theirs on CPU.
        # The forward hooks will swap CPU blocks (and their LoRA) on demand.
    else:
        print(f"[rlhf] Moving transformer to {device} (no block swapping)...")
        transformer = transformer.to(device)
        lora_network = lora_network.to(device)

    print(f"[rlhf]   Transformer placement done in {time.time() - t0:.1f}s")
    log_vram("transformer on device")

    transformer.requires_grad_(False)  # Freeze base weights
    lora_network.train()

    # Collect trainable LoRA parameters
    trainable_params = lora_network.trainable_params()

    # Re-enable gradients on LoRA params (they may have been frozen by
    # transformer.requires_grad_(False) since LoRA modules live inside the transformer)
    for p in trainable_params:
        p.requires_grad_(True)

    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in transformer.parameters())
    trainable_bytes = sum(p.numel() * p.element_size() for p in trainable_params)
    print(f"[rlhf] Parameter summary:")
    print(f"[rlhf]   Trainable (LoRA): {num_trainable:,} ({format_bytes(trainable_bytes)})")
    print(f"[rlhf]   Total (transformer): {num_total:,}")
    print(f"[rlhf]   Trainable ratio: {100*num_trainable/num_total:.2f}%")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_train_steps)
    print(f"[rlhf] Optimizer: AdamW (lr={args.learning_rate:.2e}, weight_decay=1e-4)")
    print(f"[rlhf] LR scheduler: CosineAnnealingLR (T_max={args.max_train_steps})")

    setup_time = time.time() - train_start_time
    print("-" * 70)
    print(f"[rlhf] Setup complete in {format_duration(setup_time)}")
    log_vram("before training loop")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    # Clear stale samples from previous training runs in this session
    samples_dir = os.path.join(args.output_dir, "samples")
    if os.path.isdir(samples_dir):
        import glob as glob_mod
        old_samples = glob_mod.glob(os.path.join(samples_dir, "*.jpg"))
        if old_samples:
            for f in old_samples:
                os.remove(f)
            print(f"[rlhf] Cleared {len(old_samples)} stale sample images from previous run")

    print(f"[rlhf] Starting training for {args.max_train_steps} steps")
    print(f"[rlhf]   Save checkpoints every {args.save_every} steps")
    print(f"[rlhf]   Verbose log every step")
    write_status(args.status_file, 0, args.max_train_steps, 0.0, "running")
    running_loss = 0.0
    control_file = args.control_file

    # Generate baseline samples before training (step 0)
    if args.sample_every > 0 and sample_prompts_list and not args.skip_first_sample:
        print(f"[rlhf] Generating baseline samples before training")
        try:
            generate_samples(
                transformer=transformer,
                lora_network=lora_network,
                sample_embed_cache=sample_embed_cache,
                sample_prompts=sample_prompts_list,
                step=0,
                output_dir=args.output_dir,
                vae_path=model_path,
                num_steps=args.sample_steps,
                guidance_scale=args.sample_guidance_scale,
                width=args.sample_width,
                height=args.sample_height,
                seed=args.sample_seed,
                dtype=dtype,
                device=device,
                status_file=args.status_file,
                total_steps=args.max_train_steps,
                running_loss=0.0,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[rlhf] WARNING: OOM during initial sample generation, skipping", file=sys.stderr)
                torch.cuda.empty_cache()
            else:
                print(f"[rlhf] WARNING: Initial sample generation failed: {e}", file=sys.stderr)
    elif args.skip_first_sample and args.sample_every > 0 and sample_prompts_list:
        print(f"[rlhf] Skipping first sample (skip_first_sample=True)")

    # Timing accumulators
    step_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    loop_start_time = time.time()
    consecutive_oom = 0

    for step in range(1, args.max_train_steps + 1):
        step_start = time.time()

        # Check control file for pause/stop signals
        action = check_control(control_file)
        if action == "stop":
            print(f"\n[rlhf] **** Stop signal received at step {step} ****")
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            save_lora_weights(lora_network, ckpt_dir, dtype)
            print(f"[rlhf] Saved checkpoint at step {step} before stopping")
            write_status(args.status_file, step, args.max_train_steps, running_loss, "stopped")
            total_time = time.time() - train_start_time
            print(f"[rlhf] Stopped after {format_duration(total_time)} total ({step - 1} steps completed)")
            return
        if action == "pause":
            print(f"\n[rlhf] **** Pause signal received at step {step} ****")
            write_status(args.status_file, step, args.max_train_steps, running_loss, "paused")
            pause_start = time.time()
            while True:
                time.sleep(1)
                action = check_control(control_file)
                if action == "resume" or action == "run":
                    pause_dur = time.time() - pause_start
                    print(f"[rlhf] Resuming training at step {step} (paused for {format_duration(pause_dur)})")
                    write_status(args.status_file, step, args.max_train_steps, running_loss, "running")
                    break
                if action == "stop":
                    print(f"[rlhf] **** Stop signal received while paused at step {step} ****")
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    save_lora_weights(lora_network, ckpt_dir, dtype)
                    print(f"[rlhf] Saved checkpoint at step {step} before stopping")
                    write_status(args.status_file, step, args.max_train_steps, running_loss, "stopped")
                    total_time = time.time() - train_start_time
                    print(f"[rlhf] Stopped after {format_duration(total_time)} total ({step - 1} steps completed)")
                    return

        pair = random.choice(preferences)

        w_lat = latents_cache[pair["winner_path"]]
        l_lat = latents_cache[pair["loser_path"]]

        # Ensure batch dimension
        winner_latent = w_lat.unsqueeze(0) if w_lat.dim() == 3 else w_lat
        loser_latent = l_lat.unsqueeze(0) if l_lat.dim() == 3 else l_lat

        # Get cached prompt embedding
        emb_data = embed_cache[pair["prompt"]]
        prompt_embed = emb_data["embed"]
        prompt_mask = emb_data["mask"]

        # Trim text embeddings to actual length (batch_size=1 optimization)
        actual_len = int(prompt_mask.sum(dim=1).item())
        prompt_embed = prompt_embed[:, :actual_len, :]
        prompt_mask = None  # No mask needed after trimming for batch_size=1

        # Sample timestep and noise
        t, sigma = sample_timestep_sigma(batch_size=1)
        noise = torch.randn_like(winner_latent)

        try:
            optimizer.zero_grad()

            # Forward pass (reference + policy)
            t_fwd = time.time()
            loss = compute_dpo_loss(
                transformer, lora_network,
                winner_latent, loser_latent,
                prompt_embed, prompt_mask,
                sigma, t, noise,
                args.beta, dtype, device,
            )
            fwd_elapsed = time.time() - t_fwd
            forward_times.append(fwd_elapsed)

            # Backward pass
            t_bwd = time.time()
            loss.backward()
            bwd_elapsed = time.time() - t_bwd
            backward_times.append(bwd_elapsed)

            # Gradient norm (before clipping)
            grad_norm_raw = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            # NaN detection: skip optimizer step if loss or gradients are NaN
            loss_val_check = loss.item()
            if math.isnan(loss_val_check) or math.isinf(loss_val_check):
                print(f"[rlhf] WARNING: NaN/Inf loss at step {step} (loss={loss_val_check}), skipping optimizer step")
                optimizer.zero_grad()
                continue

            # Optimizer step
            t_opt = time.time()
            optimizer.step()
            lr_scheduler.step()
            opt_elapsed = time.time() - t_opt
            optimizer_times.append(opt_elapsed)

            consecutive_oom = 0

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                consecutive_oom += 1
                print(f"\n[rlhf] *** CUDA OOM at step {step} (consecutive: {consecutive_oom}/3) ***")
                log_vram("OOM")
                if consecutive_oom >= 3:
                    print("[rlhf] ERROR: 3 consecutive OOMs, aborting training", file=sys.stderr)
                    write_status(args.status_file, step, args.max_train_steps, running_loss, "error")
                    sys.exit(1)
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                print(f"[rlhf]   Skipping step {step}, cleared GPU memory")
                continue
            else:
                raise

        loss_val = loss.item()
        running_loss = 0.9 * running_loss + 0.1 * loss_val if step > 1 else loss_val

        step_elapsed = time.time() - step_start
        step_times.append(step_elapsed)

        # Speed calculation (updated every step)
        avg_step_time = sum(step_times[-50:]) / len(step_times[-50:])
        if avg_step_time < 1.0:
            speed_str = f"{1.0 / avg_step_time:.2f} iter/s"
        else:
            speed_str = f"{avg_step_time:.2f} s/iter"

        # Write status every step so the UI stays responsive
        write_status(args.status_file, step, args.max_train_steps, running_loss, "running", speed_str)

        lr_current = optimizer.param_groups[0]['lr']

        # ETA
        remaining_steps = args.max_train_steps - step
        eta_seconds = remaining_steps * avg_step_time
        eta_str = format_duration(eta_seconds)
        elapsed_str = format_duration(time.time() - loop_start_time)

        # Timing breakdown (averages over last 50 steps)
        avg_fwd = sum(forward_times[-50:]) / max(len(forward_times[-50:]), 1)
        avg_bwd = sum(backward_times[-50:]) / max(len(backward_times[-50:]), 1)
        avg_opt = sum(optimizer_times[-50:]) / max(len(optimizer_times[-50:]), 1)

        print(f"[rlhf] ---------- Step {step}/{args.max_train_steps} ----------")
        print(f"[rlhf]   loss={running_loss:.4e}  loss_raw={loss_val:.4e}  "
              f"lr={lr_current:.2e}  grad_norm={grad_norm_raw:.4f}")
        print(f"[rlhf]   {speed_str}  elapsed={elapsed_str}  ETA={eta_str}")
        print(f"[rlhf]   timing: fwd={avg_fwd:.3f}s  bwd={avg_bwd:.3f}s  opt={avg_opt:.3f}s  "
              f"step={avg_step_time:.3f}s")

        append_loss_log(args.output_dir, step, running_loss)

        # Sample preview generation
        if args.sample_every > 0 and sample_prompts_list and step % args.sample_every == 0:
            try:
                # Free gradient tensors and GPU cache before sampling
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                generate_samples(
                    transformer=transformer,
                    lora_network=lora_network,
                    sample_embed_cache=sample_embed_cache,
                    sample_prompts=sample_prompts_list,
                    step=step,
                    output_dir=args.output_dir,
                    vae_path=model_path,
                    num_steps=args.sample_steps,
                    guidance_scale=args.sample_guidance_scale,
                    width=args.sample_width,
                    height=args.sample_height,
                    seed=args.sample_seed,
                    dtype=dtype,
                    device=device,
                    status_file=args.status_file,
                    total_steps=args.max_train_steps,
                    running_loss=running_loss,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[rlhf] WARNING: OOM during sample generation, skipping", file=sys.stderr)
                    torch.cuda.empty_cache()
                else:
                    print(f"[rlhf] WARNING: Sample generation failed: {e}", file=sys.stderr)

        # VRAM logging every 100 steps
        if step % 100 == 0:
            log_vram(f"step {step}")

        if step % args.save_every == 0 or step == args.max_train_steps:
            print(f"\n[rlhf] ==== SAVING CHECKPOINT at step {step} ====")
            t0 = time.time()
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            save_lora_weights(lora_network, ckpt_dir, dtype)
            save_time = time.time() - t0
            # Report checkpoint file sizes
            total_ckpt_size = sum(
                f.stat().st_size for f in Path(ckpt_dir).rglob("*") if f.is_file()
            )
            print(f"[rlhf]   Saved to {ckpt_dir} ({format_bytes(total_ckpt_size)}) in {save_time:.1f}s")

    # -----------------------------------------------------------------------
    # Training complete — save final and print summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Training Complete")
    print("=" * 70)

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    save_lora_weights(lora_network, final_dir, dtype)
    print(f"[rlhf] Final LoRA saved to {final_dir}")
    write_status(args.status_file, args.max_train_steps, args.max_train_steps, running_loss, "completed")

    total_time = time.time() - train_start_time
    loop_time = time.time() - loop_start_time
    avg_step_time = sum(step_times) / max(len(step_times), 1)
    avg_fwd = sum(forward_times) / max(len(forward_times), 1)
    avg_bwd = sum(backward_times) / max(len(backward_times), 1)
    avg_opt = sum(optimizer_times) / max(len(optimizer_times), 1)

    print(f"[rlhf] Summary:")
    print(f"[rlhf]   Total time:       {format_duration(total_time)} (setup: {format_duration(setup_time)}, training: {format_duration(loop_time)})")
    print(f"[rlhf]   Steps completed:  {args.max_train_steps}")
    print(f"[rlhf]   Final loss:       {running_loss:.4e}")
    print(f"[rlhf]   Avg step time:    {avg_step_time:.3f}s")
    print(f"[rlhf]   Avg forward:      {avg_fwd:.3f}s")
    print(f"[rlhf]   Avg backward:     {avg_bwd:.3f}s")
    print(f"[rlhf]   Avg optimizer:    {avg_opt:.3f}s")
    if avg_step_time < 1.0:
        print(f"[rlhf]   Throughput:       {1.0 / avg_step_time:.2f} iter/s")
    else:
        print(f"[rlhf]   Throughput:       {avg_step_time:.2f} s/iter")
    log_vram("final")
    print("=" * 70)


if __name__ == "__main__":
    main()
