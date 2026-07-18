# ai-toolkit (fork)

Personal fork of `ostris/ai-toolkit`. Notes for agents working in this repo.

## How to run / verify

The app must be run **only** via `d-run-safe-ai-toolkit.sh` (on PATH at `/home/me/source/scripts/d-run-safe-ai-toolkit.sh`) — never launch it any other way.

## Repo layout

- `ui/` — Next.js (App Router) UI. All web code, including API routes.
- `extensions_built_in/` — training-side Python code (diffusion model adapters, samplers, etc.).
- `config/` — example training configs.

### UI conventions

- Bulk video features live under `ui/src/app/api/video/<action>/route.ts` paired with `ui/src/components/Bulk<Action>Modal.tsx`, wired up from `ui/src/app/datasets/[datasetName]/page.tsx` (visible only when `allSelectedAreVideos`).
- Video API routes validate paths against `getDatasetsRoot()` / `getTrainingFolder()` from `@/server/settings`, reject `..`, check the file extension, and replace files in place via temp-write + atomic rename. See `ui/src/app/api/video/trim/route.ts` for the canonical pattern.
- Media metadata (width, height, duration, fps) is cached next to each file as `<name>.meta.json`, keyed by `mtimeMs` — `ui/src/utils/mediaMetadata.ts`. Any in-place re-encode invalidates the cache automatically.

## Fork-specific guidance

- When syncing with upstream, merge additions from both sides for `.gitignore`, `requirements.txt`, `package.json` rather than picking one.
- `gh pr create` from this fork always needs `-R james-s-tayler/ai-toolkit`, otherwise it targets `ostris/ai-toolkit` (the upstream parent).
- **Captioning:** this fork has its own captioning solution (`BulkCaptionModal`, `CaptionModal`, `caption_presets`, and the `/api/caption/get` route). Upstream has a different captioning UI/pipeline. When syncing upstream, **discard upstream's captioning entirely** and keep this fork's as the only one. Upstream captioning files to delete if they reappear (whether as merge conflicts or silent new additions): `ui/src/components/AutoCaptionButton.tsx`, `CaptionDatasetModal.tsx`, `CaptionSimpleJob.tsx`, `CaptionMonitor.tsx`, `ui/src/hooks/useCaptionBatch.tsx`, `ui/src/helpers/captionJobConfig.ts`, `ui/src/helpers/captionOptions.ts`, `ui/src/app/api/caption/getBatch/route.ts`. Also strip: the `caption` job-type branch (and `openCaptionDatasetModal`) from `ui/src/components/JobActionBar.tsx` (keep only the `job_type === 'train'` edit branch), and the `CaptionDatasetModal` import + `<CaptionDatasetModal />` mount from `ui/src/app/layout.tsx`. NOTE: `IdeogramCaptionSidebar.tsx` and `BoundingBoxOverlay.tsx` are upstream's Ideogram *prompt* editors (used by the kept `PromptBoxEditorModal`/`UpsamplePromptsModal`) — those are NOT the dataset-captioning system; keep them.
- **`DatasetImageViewer` name collision:** both this fork and upstream created a `ui/src/components/DatasetImageViewer.tsx`, but they're different components. This fork's is a lightbox (click-to-enlarge + arrow nav + caption display), used by both the datasets page and the gallery page (the gallery has no upstream equivalent). Upstream's is a full in-modal caption/bounding-box editor tied to its captioning system. **Keep this fork's lightbox version** and discard upstream's on sync (`git checkout --ours` on that file). Correspondingly, keep this fork's `datasets/[datasetName]/page.tsx` and `DatasetImageCard.tsx` (both heavily customized) rather than upstream's virtualized/caption-editor versions.

### Upstream sync workflow & gotchas

Steps that have worked for these large syncs (last done 2026-07, `d144cb5..cfdc903`, ~215 commits):

1. `git fetch upstream`. The merge base is usually the last-synced upstream commit; new upstream commits = `git log <base>..upstream/main`.
2. Work on a `sync-upstream-YYYY-MM` branch, `git merge upstream/main`, resolve, then PR to `main` (remember `-R james-s-tayler/ai-toolkit`).
3. **Big heavily-customized fork files reliably conflict** — `ui/src/app/datasets/[datasetName]/page.tsx`, `ui/src/components/DatasetImageCard.tsx`, `DatasetImageViewer.tsx`. Upstream's changes to these are almost all captioning/virtualization/its-own-viewer. Resolve by **taking the fork's whole version** (`git checkout --ours <file>`) — they're self-consistent and the fork's page passes exactly the props the fork's card expects. Don't hand-merge hunk-by-hunk; you'll create dangling refs to upstream-only state (e.g. `isRefreshingRef`, `VirtuosoGrid`, `captionExt`).
4. **Python conflicts** (`toolkit/unloader.py`, `toolkit/memory_management/manager.py`) are the fork's verbose-logging (`print_verbose`) vs upstream refactors. Combine both: keep the `print_verbose` calls AND take upstream's behavioral change (e.g. `_detach_and_cpu`, `detach()`, OstrisLinear handling). Keep the fork's counter vars (`linear_count`, etc.).
5. **Config files** — merge both dep sets in `ui/package.json` (this fork adds `recharts`/`sharp`; upstream adds `react-virtuoso`/`react-zoom-pan-pinch` — all four are needed, upstream's sample viewer/images import them). Then **regenerate** `ui/package-lock.json`: `git checkout --theirs ui/package-lock.json` then `npm install`.
6. **Find upstream files added silently** (added on upstream, never existed in the fork, so they merge in with no conflict) with: `comm -23 <(git ls-tree -r --name-only upstream/main -- ui/src | sort) <(git ls-tree -r --name-only main -- ui/src | sort)`. Grep that list for captioning to catch new discards (this is how `CaptionMonitor`/`useCaptionBatch`/`caption/getBatch` were found in 2026-07).
7. **Post-merge regeneration (required before typecheck/run):**
   - `cd ui && npx prisma generate` — upstream adds `Job` columns (e.g. `total_steps`, `save_now`, `sample_now`, `job_type`, `job_ref`); the schema auto-merges but the generated client is stale, so `tsc` shows `TS2353`/`TS2339` on `Job` fields until you regen. (`npm run update_db` does generate + `db push`.)
   - `npm install` — installs newly-merged deps into `node_modules`.
8. **Verify:** `cd ui && npx tsc -p tsconfig.worker.json` (worker, should be clean) and `npx next build` (should exit 0); `python -m py_compile` the touched `.py` files. Then **run the app (via the run script — see *How to run*) and use the `prove-it-with-playwright` skill to confirm UI functionality still works after the merge** — it drives the browser through login (Bearer token from `AI_TOOLKIT_AUTH`, stored in `localStorage`) and the key pages, asserting real data renders. This is the required functional check after any sync, not just the static build/typecheck above.
9. **Pre-existing errors to IGNORE, not chase:** `npx tsc --noEmit` reports ~40 `TS2344` errors in `.next/types/**` — the fork's route handlers/pages use Next.js-14-style sync `params` instead of Next 15's `Promise` params. These are pre-existing across *all* dynamic routes and unrelated to the merge; `next build` ignores them via `typescript.ignoreBuildErrors: true` in `ui/next.config.ts`. A clean merge = **0 source-file (non-`.next/`) type errors**.
10. **Watch for a concurrent partial sync landing on `main` while you work.** This fork sometimes receives an independent, *partial* upstream merge via GitHub's "Merge pull request #NN from ostris/main" (a sync PR that merges only an older upstream commit, not the tip). If one lands after you branched, `gh pr merge` fails with *"the merge commit cannot be cleanly created"* even though your local branch is clean. Fix: `git fetch origin`, then from your sync branch `git merge origin/main` (it auto-resolves cleanly because your full-tip merge already contains that older commit as an ancestor — a superset), re-run the build to confirm, `git push`, then merge the PR. Right before merging any sync PR, re-check `git rev-parse origin/main` hasn't moved since you branched. Merge sync PRs with `--merge` (a real merge commit), **never `--squash`** — squashing severs `main`'s ancestry link to `upstream/main` and breaks `git merge-base main upstream/main` for the next sync.
