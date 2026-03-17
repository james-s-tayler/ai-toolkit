# DPO Training Batching Implementation Plan

## Overview

Currently each training step processes one preference pair (1 winner + 1 loser). This plan adds proper batch_size > 1 support, processing multiple pairs per optimizer step for better gradient estimates and faster training.

## Why it matters

- **Gradient quality**: Averaging gradients over N pairs per step reduces variance, producing more stable training
- **Speed**: The transformer forward pass has fixed overhead per call (block swapping, kernel launches). Batching amortizes this — batch_size=2 won't be 2x faster, but should be noticeably faster than 2 separate steps
- **VRAM cost**: Each additional pair in the batch adds latent + noise memory. With 4 transformer passes per step (ref_win, ref_lose, policy_win, policy_lose), VRAM scales roughly linearly with batch_size

## Files to modify

| File | Change |
|------|--------|
| `scripts/rlhf_dpo_train.py` | Re-add `--batch_size` arg, batch the training loop, update `compute_dpo_loss` |
| `ui/src/components/rlhf/TrainingConfig.tsx` | Re-add batch_size to interface/defaults/form |
| `ui/src/app/api/rlhf/[sessionId]/train/route.ts` | Re-add batch_size to destructuring/config/args |

## Implementation details

### 1. Training loop — sample multiple pairs per step

Current code (single pair):
```python
pair = random.choice(preferences)
w_lat = latents_cache[pair["winner_path"]]
# ... single pair processing
```

New code (batched):
```python
batch_pairs = random.choices(preferences, k=args.batch_size)

# Stack latents into batch tensors: (B, C, H, W)
winner_latents = torch.stack([
    latents_cache[p["winner_path"]].squeeze(0) for p in batch_pairs
])
loser_latents = torch.stack([
    latents_cache[p["loser_path"]].squeeze(0) for p in batch_pairs
])

# Sample one shared timestep + noise per batch (same noise/timestep for all pairs in batch)
t, sigma = sample_timestep_sigma(batch_size=1)
noise = torch.randn_like(winner_latents[0:1]).expand_as(winner_latents)
# Or: per-pair noise for more variance:
# noise = torch.randn_like(winner_latents)
```

### 2. Handle variable-length prompt embeddings

Different prompts have different token lengths after trimming. Two approaches:

**Option A — Pad to max length in batch (recommended)**:
```python
embeds = []
for p in batch_pairs:
    emb_data = embed_cache[p["prompt"]]
    embeds.append(emb_data["embed"])  # (1, seq_len, dim) — varies per prompt

# Pad to max length
max_len = max(e.shape[1] for e in embeds)
padded = torch.zeros(len(embeds), max_len, embeds[0].shape[2])
for i, e in enumerate(embeds):
    padded[i, :e.shape[1], :] = e.squeeze(0)
prompt_embed = padded  # (B, max_seq_len, dim)
```

**Option B — Process one pair at a time, accumulate gradients**:
```python
optimizer.zero_grad()
total_loss = 0
for pair in batch_pairs:
    loss = compute_dpo_loss(...)  # single pair
    (loss / len(batch_pairs)).backward()  # scale gradient
    total_loss += loss.item()
optimizer.step()
```

Option B is simpler and avoids padding issues, but doesn't batch the transformer calls so it's slower. Option A is preferred for actual speedup.

### 3. Update `compute_dpo_loss` for batched inputs

The function already handles batch dims in most places. Key changes:

```python
# Current: builds list of single items
noisy_winner_list = [noisy_winner[i].unsqueeze(1) for i in range(noisy_winner.shape[0])]
cap_feats_list = [prompt_embed[i] for i in range(prompt_embed.shape[0])]
```

This already works for batch_size > 1 — it creates a list of B tensors, which is what the ZImageTransformer2DModel expects for its `x` and `cap_feats` arguments (list of per-sample tensors).

The `_call_transformer` helper stacks the outputs back into a batch tensor. So the loss computation (MSE with reduction="mean") will average over the batch.

**One subtlety**: The DPO loss should average over the batch. Currently:
```python
margin = beta * (ref_w_err - policy_w_err - ref_l_err + policy_l_err)
loss = -F.logsigmoid(margin)
```

With batched MSE (reduction="mean"), the errors are already averaged over spatial dims AND batch. This means the margin is computed on the batch-averaged errors. For proper DPO, you might want **per-pair margins** instead:

```python
# Per-pair MSE: (B,) shaped
ref_w_err = ((ref_winner_pred - target_winner) ** 2).mean(dim=(1, 2, 3))
ref_l_err = ((ref_loser_pred - target_loser) ** 2).mean(dim=(1, 2, 3))
policy_w_err = ((policy_winner_pred - target_winner) ** 2).mean(dim=(1, 2, 3))
policy_l_err = ((policy_loser_pred - target_loser) ** 2).mean(dim=(1, 2, 3))

# Per-pair margins, then average the loss
margins = beta * (ref_w_err - policy_w_err - ref_l_err + policy_l_err)
loss = -F.logsigmoid(margins).mean()
```

### 4. Shared vs per-pair timesteps

DPO requires the same timestep and noise for both winner and loser within a pair (so the comparison is fair). But across pairs in a batch, you can either:

- **Shared timestep** (simpler): One `t` for all pairs. Currently how it works.
- **Per-pair timesteps** (better variance): Sample B different timesteps. Requires the transformer to handle different timesteps per list item — check if `t` can be a (B,) tensor when `x` is a list of B items.

### 5. VRAM budget

With block swapping at 16 blocks on a 16GB GPU:
- batch_size=1: ~14GB peak (current)
- batch_size=2: ~15-16GB peak (tight but possible)
- batch_size=4: likely OOM

Consider adding a gradient accumulation option as an alternative:
```python
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
```
This gives the gradient quality benefit of larger batches without the VRAM cost — process N single pairs, accumulate gradients, then do one optimizer step.

### 6. Gradient accumulation (easier alternative)

If VRAM is tight, gradient accumulation is simpler to implement and equally effective for gradient quality:

```python
optimizer.zero_grad()
accum_loss = 0.0

for micro_step in range(args.gradient_accumulation_steps):
    pair = random.choice(preferences)
    # ... prepare single pair ...
    loss = compute_dpo_loss(...) / args.gradient_accumulation_steps
    loss.backward()
    accum_loss += loss.item() * args.gradient_accumulation_steps

optimizer.step()
```

This requires no changes to `compute_dpo_loss` at all — just a loop around the existing single-pair code with scaled loss.

## Recommended implementation order

1. **Start with gradient accumulation** — minimal code change, no VRAM increase, immediate benefit
2. **Then add true batching** — for speed improvement on high-VRAM GPUs
3. **Both can coexist**: `effective_batch = batch_size * gradient_accumulation_steps`
