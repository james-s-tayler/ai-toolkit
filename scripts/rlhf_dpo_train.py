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

Output status.json (written every 10 steps):
    {"step": 150, "total_steps": 2000, "loss": 0.693, "status": "running"}
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Z-Image constants (from musubi-tuner zimage_config)
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
    parser.add_argument("--status_file", type=str, required=True)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--beta", type=float, default=5000.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--status_every", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=1024)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Status file helpers
# ---------------------------------------------------------------------------

def write_status(path: str, step: int, total_steps: int, loss: float, status: str = "running"):
    try:
        with open(path, "w") as f:
            json.dump({"step": step, "total_steps": total_steps, "loss": loss, "status": status}, f)
    except Exception as e:
        print(f"[rlhf] Warning: could not write status file: {e}", file=sys.stderr)


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
        print(f"[rlhf] Model path '{model_path}' not found locally, downloading from HuggingFace...")
        try:
            from huggingface_hub import snapshot_download
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
# LoRA helpers — uses musubi-tuner style LoRA network
# ---------------------------------------------------------------------------

def create_lora_network(transformer, lora_rank: int, lora_alpha: float = None):
    """Create a LoRA network targeting ZImageTransformerBlock layers.

    Returns the network and applies it to the transformer.
    Uses the musubi-tuner LoRA implementation if available, otherwise falls
    back to a simple manual approach.
    """
    if lora_alpha is None:
        lora_alpha = float(lora_rank)

    try:
        # Try musubi-tuner LoRA (preferred — matches existing training infra)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "musubi-tuner", "src"))
        from musubi_tuner.networks import lora as mt_lora

        network = mt_lora.create_network(
            ZIMAGE_LORA_TARGET_MODULES,
            "lora_unet",
            1.0,  # multiplier
            lora_rank,
            lora_alpha,
            None,   # vae
            [],     # text_encoders
            transformer,
            exclude_patterns=ZIMAGE_LORA_EXCLUDE_PATTERNS,
        )
        network.apply_to([], transformer, apply_text_encoder=False, apply_unet=True)
        print(f"[rlhf] Created musubi-tuner LoRA network (rank={lora_rank}, alpha={lora_alpha})")
        print(f"[rlhf]   LoRA modules: {len(network.unet_loras)}")
        return network

    except ImportError:
        print("[rlhf] musubi-tuner not found, falling back to manual LoRA creation")

    # Fallback: manually inject LoRA into Linear layers of target modules
    from collections import OrderedDict

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

    print(f"[rlhf] Created fallback LoRA network (rank={lora_rank}, {len(network.lora_modules)} modules)")
    return network


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

    for img_path in image_paths:
        # Use full path hash for cache key to handle duplicate filenames
        path_hash = hashlib.sha256(img_path.encode()).hexdigest()[:12]
        cache_path = cache_dir / f"lat_{path_hash}.pt"
        if cache_path.exists():
            latents[img_path] = torch.load(cache_path, map_location="cpu", weights_only=True)
            continue

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

    for prompt in set(prompts):
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        cache_path = cache_dir / f"emb_{prompt_hash}.pt"
        if cache_path.exists():
            embeddings[prompt] = torch.load(cache_path, map_location="cpu", weights_only=True)
            continue

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
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    write_status(args.status_file, 0, args.max_train_steps, 0.0, "starting")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"[rlhf] Using device={device}, dtype={dtype}")

    # Load preference data
    with open(args.preference_data) as f:
        preferences = json.load(f)

    if not preferences:
        print("[rlhf] ERROR: No preference data found", file=sys.stderr)
        write_status(args.status_file, 0, args.max_train_steps, 0.0, "error")
        sys.exit(1)

    print(f"[rlhf] Loaded {len(preferences)} preference pairs")

    # -----------------------------------------------------------------------
    # Load Z-Image model components
    # -----------------------------------------------------------------------
    model_path = resolve_model_path(args.model_path)

    print(f"[rlhf] Loading Z-Image transformer from {model_path}")
    write_status(args.status_file, 0, args.max_train_steps, 0.0, "loading_model")

    try:
        from diffusers import AutoencoderKL
        from diffusers.models.transformers import ZImageTransformer2DModel
        from transformers import AutoTokenizer, Qwen3ForCausalLM
    except ImportError as e:
        print(f"[rlhf] ERROR: Missing dependencies. Install with: pip install diffusers transformers", file=sys.stderr)
        print(f"[rlhf]   Import error: {e}", file=sys.stderr)
        write_status(args.status_file, 0, args.max_train_steps, 0.0, "error")
        sys.exit(1)

    try:
        # Load transformer
        transformer_path = os.path.join(model_path, "transformer")
        if os.path.isdir(transformer_path):
            transformer = ZImageTransformer2DModel.from_pretrained(
                transformer_path, torch_dtype=dtype
            )
        else:
            transformer = ZImageTransformer2DModel.from_pretrained(
                model_path, subfolder="transformer", torch_dtype=dtype
            )
        print(f"[rlhf] Transformer loaded")

        # Load VAE (always float32 for encoding accuracy)
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
        print(f"[rlhf] VAE loaded")

        # Load Qwen3 text encoder + tokenizer
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=dtype
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        print(f"[rlhf] Text encoder + tokenizer loaded")

    except Exception as e:
        print(f"[rlhf] ERROR loading model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        write_status(args.status_file, 0, args.max_train_steps, 0.0, "error")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Create LoRA network on transformer
    # -----------------------------------------------------------------------
    print(f"[rlhf] Creating LoRA adapter (rank={args.lora_rank})")
    lora_network = create_lora_network(transformer, args.lora_rank)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # -----------------------------------------------------------------------
    # Cache image latents, then unload VAE
    # -----------------------------------------------------------------------
    print("[rlhf] Encoding images to latents...")
    write_status(args.status_file, 0, args.max_train_steps, 0.0, "caching_latents")
    vae = vae.to(device)
    cache_dir = os.path.join(args.output_dir, "cache")

    all_image_paths = []
    for p in preferences:
        all_image_paths.append(p["winner_path"])
        all_image_paths.append(p["loser_path"])

    latents_cache = encode_images_to_latents(vae, all_image_paths, args.resolution, device, cache_dir)
    del vae
    torch.cuda.empty_cache()
    print(f"[rlhf] VAE unloaded, {len(latents_cache)} latents cached")

    # -----------------------------------------------------------------------
    # Cache text embeddings, then unload text encoder
    # -----------------------------------------------------------------------
    print("[rlhf] Encoding prompts with Qwen3...")
    write_status(args.status_file, 0, args.max_train_steps, 0.0, "caching_embeddings")
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    prompts = [p["prompt"] for p in preferences]
    embed_cache = encode_prompts(tokenizer, text_encoder, prompts, device, cache_dir)
    del text_encoder, tokenizer
    torch.cuda.empty_cache()
    print(f"[rlhf] Text encoder unloaded, {len(embed_cache)} embeddings cached")

    # -----------------------------------------------------------------------
    # Move transformer to device and set up optimizer
    # -----------------------------------------------------------------------
    transformer = transformer.to(device)
    lora_network = lora_network.to(device)
    transformer.requires_grad_(False)  # Freeze base weights
    lora_network.train()

    # Collect trainable LoRA parameters
    if hasattr(lora_network, 'trainable_params'):
        trainable_params = lora_network.trainable_params()
    else:
        # musubi-tuner LoRA network — parameters are on the network module
        trainable_params = list(lora_network.parameters())

    # Re-enable gradients on LoRA params (they may have been frozen by
    # transformer.requires_grad_(False) since LoRA modules live inside the transformer)
    for p in trainable_params:
        p.requires_grad_(True)

    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in transformer.parameters())
    print(f"[rlhf] Trainable params: {num_trainable:,} / {num_total:,} ({100*num_trainable/num_total:.2f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_train_steps)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print(f"[rlhf] Starting training for {args.max_train_steps} steps")
    write_status(args.status_file, 0, args.max_train_steps, 0.0, "running")
    running_loss = 0.0

    for step in range(1, args.max_train_steps + 1):
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

        optimizer.zero_grad()
        loss = compute_dpo_loss(
            transformer, lora_network,
            winner_latent, loser_latent,
            prompt_embed, prompt_mask,
            sigma, t, noise,
            args.beta, dtype, device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        loss_val = loss.item()
        running_loss = 0.9 * running_loss + 0.1 * loss_val if step > 1 else loss_val

        if step % args.status_every == 0:
            print(f"[rlhf] Step {step}/{args.max_train_steps} | loss={running_loss:.4f}")
            write_status(args.status_file, step, args.max_train_steps, running_loss, "running")

        if step % args.save_every == 0 or step == args.max_train_steps:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            if hasattr(lora_network, 'save_weights'):
                lora_network.save_weights(ckpt_dir, dtype=dtype)
            else:
                torch.save(lora_network.state_dict(), os.path.join(ckpt_dir, "lora_weights.safetensors"))
            print(f"[rlhf] Saved checkpoint to {ckpt_dir}")

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    if hasattr(lora_network, 'save_weights'):
        lora_network.save_weights(final_dir, dtype=dtype)
    else:
        torch.save(lora_network.state_dict(), os.path.join(final_dir, "lora_weights.safetensors"))
    print(f"[rlhf] Training complete. Final LoRA saved to {final_dir}")
    write_status(args.status_file, args.max_train_steps, args.max_train_steps, running_loss, "completed")


if __name__ == "__main__":
    main()
