# RLHF/DPO Training Feature for ai-toolkit

## Context

Z-Image-Turbo is an RLHF finetune of Z-Image. Since Z-Image (base) is open source, we can build our own RLHF training pipeline using Diffusion-DPO. The goal is a new "RLHF / DPO" submenu in ai-toolkit that lets the user:

1. Select a model and ComfyUI workflow
2. Queue batch generation of image pairs overnight
3. Evaluate pairs side-by-side the next day (pick winner/loser)
4. Run DPO training on the preference data to produce a LoRA

We use **custom Diffusion-DPO loss** (~300 lines) rather than an existing framework, since Z-Image uses non-standard Lumina2 architecture.

### Prerequisite: Gallery Branch
The `copilot/add-gallery-menu-item` branch adds a gallery feature (GalleryFolder model, image serving routes, gallery UI pages) that should be merged into the RLHF-DPO branch first. The RLHF feature reuses several gallery patterns:
- `TopBar` + `MainContent` layout components
- `apiClient` for API calls
- `DatasetImageViewer` for image enlargement
- `Modal` component from `@/components/Modal`
- `/api/gallery/img/` image serving pattern (validates paths against registered folders)
- IntersectionObserver-based lazy loading for image grids

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│  ai-toolkit UI (Next.js)                             │
│  ┌──────────┬──────────┬──────────┬────────────────┐ │
│  │ Session  │ Batch    │ Evaluate │ DPO Training   │ │
│  │ Setup    │ Generate │ Pairs    │ Monitor        │ │
│  └──────────┴──────────┴──────────┴────────────────┘ │
│        │           │                      │          │
│        ▼           ▼                      ▼          │
│   Prisma/SQLite  ComfyUI API        Python script    │
│                  (port 9188)        (rlhf_dpo_train) │
└──────────────────────────────────────────────────────┘
```

---

## 1. Database Schema Changes

**File:** `ui/prisma/schema.prisma`

Add 3 new models:

```prisma
model RlhfSession {
  id                String   @id @default(uuid())
  name              String   @unique
  status            String   @default("setup")
  // setup | generating | generated | evaluating | training | completed | error
  model_path        String
  comfyui_url       String   @default("http://127.0.0.1:9188")
  workflow_json     String   @default("")
  config_json       String   @default("{}")
  output_dir        String   @default("")
  gpu_ids           String   @default("0")
  created_at        DateTime @default(now())
  updated_at        DateTime @updatedAt
  pairs             RlhfPair[]
  training_runs     RlhfTrainingRun[]
  @@index([status])
}

model RlhfPair {
  id              String      @id @default(uuid())
  session_id      String
  session         RlhfSession @relation(fields: [session_id], references: [id], onDelete: Cascade)
  prompt          String
  seed_a          Int
  seed_b          Int
  image_a_path    String   @default("")
  image_b_path    String   @default("")
  comfyui_id_a    String   @default("")
  comfyui_id_b    String   @default("")
  gen_status      String   @default("pending")  // pending | queued | completed | error
  preference      String   @default("none")     // none | a | b | tie | skip
  created_at      DateTime @default(now())
  updated_at      DateTime @updatedAt
  @@index([session_id])
  @@index([gen_status])
  @@index([preference])
}

model RlhfTrainingRun {
  id              String      @id @default(uuid())
  session_id      String
  session         RlhfSession @relation(fields: [session_id], references: [id], onDelete: Cascade)
  status          String   @default("pending")  // pending | running | completed | error
  config_json     String   @default("{}")
  output_path     String   @default("")
  step            Int      @default(0)
  total_steps     Int      @default(0)
  loss            Float    @default(0)
  log_path        String   @default("")
  pid             Int?
  created_at      DateTime @default(now())
  updated_at      DateTime @updatedAt
  @@index([session_id])
  @@index([status])
}
```

---

## 2. New Files

```
ui/prisma/schema.prisma                              # MODIFY: Add 3 models above

ui/src/app/rlhf/
  page.tsx                                            # Sessions list
  new/page.tsx                                        # Create session form
  [sessionId]/page.tsx                                # Session detail (tabbed)

ui/src/app/api/rlhf/
  route.ts                                            # GET list / POST create sessions
  [sessionId]/route.ts                                # GET/PUT/DELETE session
  [sessionId]/generate/route.ts                       # POST start generation, GET status
  [sessionId]/pairs/route.ts                          # GET list / POST bulk-create pairs
  [sessionId]/pairs/[pairId]/route.ts                 # PUT update preference
  [sessionId]/evaluate/route.ts                       # GET next unevaluated pair + stats
  [sessionId]/train/route.ts                          # POST start training, GET runs list
  [sessionId]/train/[runId]/route.ts                  # GET training run status

ui/src/components/rlhf/
  SessionCard.tsx                                     # Card for session list
  SessionSetupForm.tsx                                # Session creation form
  GenerationMonitor.tsx                               # Generation progress
  EvaluationUI.tsx                                    # Side-by-side comparison (core UX)
  TrainingConfig.tsx                                  # DPO hyperparameter form
  TrainingMonitor.tsx                                 # Loss graph + step counter

ui/cron/actions/processRlhfGeneration.ts              # Poll ComfyUI, download images

scripts/rlhf_dpo_train.py                             # Standalone DPO training (~300 lines)
```

**Files to modify:**
- `ui/src/components/Sidebar.tsx` — Add "RLHF / DPO" nav item (use `Zap` icon from lucide-react)
- `ui/cron/worker.ts` — Add `processRlhfGeneration()` to the loop

---

## 3. Sidebar Navigation

**File:** `ui/src/components/Sidebar.tsx`

Add between "Training Queue" and "Datasets":
```typescript
import { ..., Zap } from 'lucide-react';

const navigation = [
  ...
  { name: 'Training Queue', href: '/jobs', icon: BrainCircuit },
  { name: 'RLHF / DPO', href: '/rlhf', icon: Zap },        // NEW
  { name: 'Datasets', href: '/datasets', icon: Images },
  ...
];
```

---

## 4. ComfyUI Integration

### Workflow Template
The session stores a ComfyUI workflow JSON with placeholders:
- `{{PROMPT}}` — replaced with pair's prompt text
- `{{SEED}}` — replaced with specific seed

### Batch Generation Strategy
1. The `/generate` API endpoint creates pairs with random seed pairs and sets session status to `"generating"`
2. **Cron worker** (`processRlhfGeneration.ts`) handles the actual ComfyUI interaction:
   - Checks ComfyUI queue depth via `GET /queue`
   - Submits pending pairs in batches of 10-20 (staggered to avoid flooding)
   - For each pair: submit 2 workflows (same prompt, different seeds) via `POST /prompt`
   - Polls `GET /history/{prompt_id}` for completion
   - Downloads completed images via `GET /view?filename=X&subfolder=Y&type=Z`
   - Saves to `{DATA_ROOT}/rlhf/{session_name}/{pair_id}/image_a.png`
   - Updates pair `gen_status` and `image_a_path`/`image_b_path`
3. When all pairs complete, session status → `"generated"`

### Worker Integration
**File:** `ui/cron/worker.ts`
```typescript
async loop() {
  await processQueue();
  await processDownloads();
  await processRlhfGeneration();  // NEW
}
```

---

## 5. Evaluation UI (Core UX)

**Component:** `ui/src/components/rlhf/EvaluationUI.tsx`

### Layout
- Two images side by side (50/50 split), maximized
- Prompt text displayed below images
- Progress bar: "42/100 evaluated | 58 remaining"

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `A` or Left Click | Prefer left image |
| `D` or Right Click | Prefer right image |
| `T` | Tie |
| `S` | Skip |
| `Z` | Undo last |

### Anti-bias
- Randomize which image (A/B) appears on left vs right
- Track mapping internally, store preference correctly

### Performance
- Prefetch next 5 pairs for instant transitions
- Send preference updates via PUT to `/api/rlhf/.../pairs/[pairId]`
- Brief flash animation on selection, auto-advance to next pair

### Image Serving
Reuse the gallery branch's `/api/gallery/img/[...imagePath]` pattern — it validates that image paths belong to registered folders. For RLHF, the session's output directory (where generated images are saved) acts as the allowed path. Alternatively, use the existing `/api/img/[...imagePath]` route which validates against `dataRoot`.

### Reusable Components (from gallery branch)
- `DatasetImageViewer` — full-screen image viewer with arrow key navigation (already handles prev/next)
- `GalleryImageCard` pattern — lazy-loaded image cards with IntersectionObserver
- `TopBar` + `MainContent` — standard page layout

---

## 6. DPO Training Script

**File:** `scripts/rlhf_dpo_train.py`

### CLI Interface
```bash
python scripts/rlhf_dpo_train.py \
  --preference_data /path/to/preferences.json \
  --model_path /path/to/model \
  --output_dir /path/to/output \
  --status_file /path/to/status.json \
  --beta 5000 --learning_rate 1e-5 --max_train_steps 2000 \
  --lora_rank 16 --batch_size 1 --blocks_to_swap 0 \
  --mixed_precision bf16 --gradient_checkpointing
```

### Input: `preferences.json`
```json
[
  { "prompt": "a cat on a windowsill", "winner_path": "/path/winner.png", "loser_path": "/path/loser.png" }
]
```

### Output: `status.json` (written every 10 steps)
```json
{ "step": 150, "total_steps": 2000, "loss": 0.693, "status": "running" }
```

### DPO Loss Implementation
```
L = -log σ(β · (ref_w_err - policy_w_err - ref_l_err + policy_l_err))
```

Key technique: **LoRA multiplier toggling** — single model copy, set LoRA multiplier to 0.0 for reference model, 1.0 for policy model. Avoids loading two full models in 16GB VRAM.

### Training Loop Pseudocode
1. Load base model + create LoRA adapter (policy, trainable)
2. Pre-encode all images with VAE → cache latents to disk → unload VAE
3. Pre-encode all prompts with text encoder → cache → unload text encoder
4. For each step:
   - Sample random pair, timestep `t`, noise `ε`
   - Noise both winner/loser latents with same `t` and `ε`
   - LoRA multiplier → 0.0: compute reference predictions (no_grad)
   - LoRA multiplier → 1.0: compute policy predictions (with grad)
   - Compute DPO loss, backprop through LoRA weights only
5. Save LoRA checkpoint periodically

### VRAM Strategy (16GB RTX 5060 Ti)
1. Single model + LoRA multiplier toggle (no dual model)
2. Sequential forward passes (reference first, then policy)
3. Gradient checkpointing
4. Block swapping to CPU (reuse musubi-tuner infrastructure)
5. bf16 precision
6. Batch size 1
7. Latent/text caching to disk (unload VAE + text encoder before training)

### Process Launching
Follow `startJob.ts` pattern: `spawn()` with `detached: true`, `stdio: 'ignore'`, `subprocess.unref()`. Store PID in `RlhfTrainingRun.pid`.

---

## 7. Session Detail Page (Tabbed)

**File:** `ui/src/app/rlhf/[sessionId]/page.tsx`

4 tabs:

| Tab | Content |
|-----|---------|
| **Generation** | Progress bar, pair grid with status indicators, Start/Pause buttons |
| **Evaluation** | EvaluationUI component (side-by-side), progress stats |
| **Training** | Config form, Start Training button, loss graph (Recharts), step counter |
| **Pairs** | Paginated grid of all pairs, filter by preference, click to inspect |

Auto-refresh: poll every 5 seconds during generation and training.

---

## 8. Implementation Order

### Phase 0: Prerequisites
0. Merge `copilot/add-gallery-menu-item` branch into `RLHF-DPO` branch (provides GalleryFolder model, image serving, UI components)

### Phase 1: Database & Infrastructure
1. Add Prisma models, run migration
2. Add Sidebar nav item
3. Create session CRUD API routes
4. Add `processRlhfGeneration` to cron worker (stub)

### Phase 2: Session Setup UI
5. Sessions list page (`/rlhf`)
6. New session form (`/rlhf/new`)
7. Session detail page shell with tabs

### Phase 3: ComfyUI Generation
8. Pair creation API (bulk create with prompts + random seeds)
9. Generation API (submit to ComfyUI)
10. Implement `processRlhfGeneration.ts` (poll + download)
11. GenerationMonitor component

### Phase 4: Evaluation UI
12. Evaluation API (next pair, stats)
13. EvaluationUI component with keyboard shortcuts
14. Preference update API
15. PairGrid browser

### Phase 5: DPO Training
16. Write `rlhf_dpo_train.py` (core ~300 lines)
17. Training launch API (spawn script, read status.json)
18. TrainingConfig + TrainingMonitor components

### Phase 6: Polish
19. Error handling, retry failed generations
20. End-to-end test

---

## 9. Key Reference Files

| Purpose | File |
|---------|------|
| Prisma schema | `ui/prisma/schema.prisma` |
| Sidebar nav | `ui/src/components/Sidebar.tsx` |
| Worker loop | `ui/cron/worker.ts` |
| Job spawn pattern | `ui/cron/actions/startJob.ts` |
| Image serving | `ui/src/app/api/img/[...imagePath]/route.ts` |
| Gallery image card pattern | `ui/src/app/gallery/[folderId]/page.tsx` (on `copilot/add-gallery-menu-item` branch) |
| Gallery image route | `ui/src/app/api/gallery/img/[...imagePath]/route.ts` (on gallery branch) |
| Z-Image model loading | `musubi-tuner/src/musubi_tuner/zimage_train_network.py` |
| Z-Image LoRA | `musubi-tuner/src/musubi_tuner/networks/lora_zimage.py` |
| Z-Image flow matching | `musubi-tuner/src/musubi_tuner/zimage_train_network.py:329` (`call_dit`) |

---

## 10. Verification

1. **Schema**: Run `npx prisma db push` — tables created without errors
2. **Sidebar**: New "RLHF / DPO" item visible, links to `/rlhf`
3. **Session CRUD**: Create/list/delete sessions via API
4. **Generation**: Create pairs → start generation → images appear in filesystem → gen_status updates
5. **Evaluation**: Navigate pairs with keyboard → preferences saved correctly
6. **Training**: Launch DPO script → status.json updates → loss decreases over steps → LoRA saved
7. **End-to-end**: Full flow from session creation through to LoRA output
