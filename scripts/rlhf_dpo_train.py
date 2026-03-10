#!/usr/bin/env python3
"""
RLHF-DPO Training Script for Z-Image / Lumina2 models.

Uses Diffusion-DPO loss with a single model copy and LoRA multiplier toggling
to fit within 16 GB VRAM.

CLI Usage:
    python scripts/rlhf_dpo_train.py \
        --preference_data /path/to/preferences.json \
        --model_path /path/to/model \
        --output_dir /path/to/output \
        --status_file /path/to/status.json \
        --beta 5000 --learning_rate 1e-5 --max_train_steps 2000 \
        --lora_rank 16 --batch_size 1 --blocks_to_swap 0 \
        --mixed_precision bf16 --gradient_checkpointing

Input preferences.json:
    [
        {"prompt": "a cat", "winner_path": "/path/win.png", "loser_path": "/path/lose.png"},
        ...
    ]

Output status.json (written every 10 steps):
    {"step": 150, "total_steps": 2000, "loss": 0.693, "status": "running"}
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RLHF-DPO training")
    parser.add_argument("--preference_data", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
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
    parser.add_argument("--resolution", type=int, default=512)
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
# Image preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess_image(image_path: str, resolution: int) -> torch.Tensor:
    """Load an image and return a tensor in [-1, 1] range, shape (1, C, H, W)."""
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return transform(img).unsqueeze(0)  # (1, 3, H, W)


# ---------------------------------------------------------------------------
# LoRA helpers (PEFT-based)
# ---------------------------------------------------------------------------

def set_lora_scale(model, scale: float):
    """Set LoRA scale for all LoRA layers in a PEFT-wrapped model."""
    try:
        from peft.tuners.lora import LoraLayer
    except ImportError as exc:
        raise RuntimeError(
            "peft is required for LoRA scale toggling. Install with: pip install peft"
        ) from exc
    for module in model.modules():
        if isinstance(module, LoraLayer):
            for key in module.scaling:
                module.scaling[key] = scale


# ---------------------------------------------------------------------------
# Latent caching
# ---------------------------------------------------------------------------

def encode_images_to_latents(vae, image_paths, resolution, dtype, device, cache_dir):
    """Encode images to latents using VAE and cache to disk."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    latents = {}

    for path in image_paths:
        cache_path = cache_dir / (Path(path).stem + "_latent.pt")
        if cache_path.exists():
            latents[path] = torch.load(cache_path, map_location="cpu", weights_only=True)
            continue

        img_tensor = load_and_preprocess_image(path, resolution).to(device, dtype=dtype)
        with torch.no_grad():
            latent = vae.encode(img_tensor).latent_dist.sample()
            latent = latent * vae.config.scaling_factor
        latents[path] = latent.cpu()
        torch.save(latents[path], cache_path)

    return latents


def encode_prompts_to_embeddings(text_encoder, tokenizer, prompts, device, cache_dir):
    """Encode prompts with text encoder and cache to disk."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings = {}

    for prompt in set(prompts):
        cache_path = cache_dir / f"emb_{hash(prompt) & 0xFFFFFF:06x}.pt"
        if cache_path.exists():
            embeddings[prompt] = torch.load(cache_path, map_location="cpu", weights_only=True)
            continue

        inputs = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(device)
        with torch.no_grad():
            emb = text_encoder(**inputs).last_hidden_state
        embeddings[prompt] = emb.cpu()
        torch.save(embeddings[prompt], cache_path)

    return embeddings


# ---------------------------------------------------------------------------
# DPO loss computation
# ---------------------------------------------------------------------------

def compute_dpo_loss(
    unet,
    scheduler,
    winner_latent: torch.Tensor,
    loser_latent: torch.Tensor,
    prompt_embed: torch.Tensor,
    timestep: torch.Tensor,
    noise: torch.Tensor,
    beta: float,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """
    Computes Diffusion-DPO loss:
        L = -log σ(β · (ref_w_err - policy_w_err - ref_l_err + policy_l_err))
    where _err is MSE(predicted_noise, actual_noise).

    Uses LoRA multiplier toggling: scale=0 for reference, scale=1 for policy.
    """
    winner_latent = winner_latent.to(device, dtype=dtype)
    loser_latent = loser_latent.to(device, dtype=dtype)
    prompt_embed = prompt_embed.to(device, dtype=dtype)
    noise = noise.to(device, dtype=dtype)
    timestep = timestep.to(device)

    noisy_winner = scheduler.add_noise(winner_latent, noise, timestep)
    noisy_loser = scheduler.add_noise(loser_latent, noise, timestep)

    # Reference predictions (LoRA scale = 0, no gradient)
    set_lora_scale(unet, 0.0)
    with torch.no_grad():
        ref_winner_pred = unet(noisy_winner, timestep, encoder_hidden_states=prompt_embed).sample
        ref_loser_pred = unet(noisy_loser, timestep, encoder_hidden_states=prompt_embed).sample

    ref_w_err = F.mse_loss(ref_winner_pred, noise, reduction="mean")
    ref_l_err = F.mse_loss(ref_loser_pred, noise, reduction="mean")

    # Policy predictions (LoRA scale = 1, with gradient)
    set_lora_scale(unet, 1.0)
    policy_winner_pred = unet(noisy_winner, timestep, encoder_hidden_states=prompt_embed).sample
    policy_loser_pred = unet(noisy_loser, timestep, encoder_hidden_states=prompt_embed).sample

    policy_w_err = F.mse_loss(policy_winner_pred, noise, reduction="mean")
    policy_l_err = F.mse_loss(policy_loser_pred, noise, reduction="mean")

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
    write_status(args.status_file, 0, args.max_train_steps, 0.0, "running")

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

    # Load models
    print(f"[rlhf] Loading model from {args.model_path}")
    try:
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer
        from peft import LoraConfig, get_peft_model

        model_path = args.model_path
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype)
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
        text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    except Exception as e:
        print(f"[rlhf] ERROR loading model: {e}", file=sys.stderr)
        write_status(args.status_file, 0, args.max_train_steps, 0.0, "error")
        sys.exit(1)

    # Add LoRA adapter to UNet
    print(f"[rlhf] Adding LoRA adapter (rank={args.lora_rank})")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    unet = unet.to(device)

    # Cache image latents then unload VAE
    print("[rlhf] Encoding images to latents...")
    vae = vae.to(device)
    cache_dir = os.path.join(args.output_dir, "cache")
    all_image_paths = []
    for p in preferences:
        all_image_paths.append(p["winner_path"])
        all_image_paths.append(p["loser_path"])

    latents_cache = encode_images_to_latents(vae, all_image_paths, args.resolution, dtype, device, cache_dir)
    del vae
    torch.cuda.empty_cache()
    print("[rlhf] VAE unloaded, latents cached")

    # Cache text embeddings then unload text encoder
    print("[rlhf] Encoding prompts...")
    text_encoder = text_encoder.to(device)
    prompts = [p["prompt"] for p in preferences]
    embed_cache = encode_prompts_to_embeddings(text_encoder, tokenizer, prompts, device, cache_dir)
    del text_encoder
    torch.cuda.empty_cache()
    print("[rlhf] Text encoder unloaded, embeddings cached")

    # Optimizer over LoRA parameters only
    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_train_steps
    )

    # Training loop
    print(f"[rlhf] Starting training for {args.max_train_steps} steps")
    unet.train()
    running_loss = 0.0
    num_timesteps = scheduler.config.num_train_timesteps

    for step in range(1, args.max_train_steps + 1):
        pair = random.choice(preferences)

        w_lat = latents_cache[pair["winner_path"]]
        l_lat = latents_cache[pair["loser_path"]]
        # Validate and ensure batch dimension is present
        for name, lat in (("winner", w_lat), ("loser", l_lat)):
            if lat.dim() not in (3, 4):
                raise ValueError(
                    f"Cached {name} latent has unexpected shape {tuple(lat.shape)}; "
                    "expected 3-D (C,H,W) or 4-D (1,C,H,W)"
                )
        winner_latent = w_lat.unsqueeze(0) if w_lat.dim() == 3 else w_lat
        loser_latent = l_lat.unsqueeze(0) if l_lat.dim() == 3 else l_lat
        prompt_embed = embed_cache[pair["prompt"]]

        t = torch.randint(0, num_timesteps, (1,), dtype=torch.long)
        noise = torch.randn_like(winner_latent)

        optimizer.zero_grad()
        loss = compute_dpo_loss(
            unet, scheduler,
            winner_latent, loser_latent,
            prompt_embed, t, noise,
            args.beta, dtype, device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        running_loss = 0.9 * running_loss + 0.1 * loss.item()

        if step % args.status_every == 0:
            print(f"[rlhf] Step {step}/{args.max_train_steps} | loss={running_loss:.4f}")
            write_status(args.status_file, step, args.max_train_steps, running_loss, "running")

        if step % args.save_every == 0 or step == args.max_train_steps:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            unet.save_pretrained(ckpt_dir)
            print(f"[rlhf] Saved checkpoint to {ckpt_dir}")

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    unet.save_pretrained(final_dir)
    print(f"[rlhf] Training complete. Final model saved to {final_dir}")
    write_status(args.status_file, args.max_train_steps, args.max_train_steps, running_loss, "completed")


if __name__ == "__main__":
    main()
