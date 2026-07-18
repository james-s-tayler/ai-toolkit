# ai-toolkit (fork)

Personal fork of `ostris/ai-toolkit`. Notes for agents working in this repo.

## How to run / verify

The app must be run **only** via `d-run-safe-ai-toolkit.sh` (on PATH at `/home/me/source/scripts/d-run-safe-ai-toolkit.sh`) — never launch it any other way.

Automated UI regression coverage lives in `ui/tests/e2e/` (Playwright). Run it against the running instance after any merge — it's the primary gate for merge regressions. See sync-workflow step 8 below and `.claude/rules/e2e-tests.md` for conventions (notably: it must leave the shared instance's manual-testing data untouched).

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
7. **Regeneration + deps are handled by the run script.** Its `build_and_start` runs `npm install` (newly-merged deps) then `update_db` = `prisma generate` + `db push`, which applies upstream's new `Job` columns (`total_steps`, `save_now`, `sample_now`, `job_type`, `job_ref`) to the DB. No manual `npm`/`prisma` steps.
8. **Verify: run it with the run script → run the committed E2E suite → `prove-it-with-playwright` for the rest.** Launch the app via the run script (see *How to run*) — it installs, migrates, builds, and starts it; never build or launch it any other way. Then run the committed Playwright regression suite against the running instance — this is the automated gate that catches exactly the API-shape breakage upstream syncs cause:
   ```bash
   cd ui
   npx playwright install chromium   # once per machine
   AI_TOOLKIT_AUTH=<same token the app was started with> npm run test:e2e
   ```
   The suite (`ui/tests/e2e/`, see its `README.md` and `.claude/rules/e2e-tests.md`) is data-agnostic and non-destructive to the shared instance (mutation tests use a throwaway dataset). All specs must pass (or `skip` when data is absent) before merging. For anything the suite doesn't cover — new or changed features especially — also use the **`prove-it-with-playwright` skill** to drive the browser through login (Bearer token from `AI_TOOLKIT_AUTH`, stored in `localStorage`) and the affected pages, asserting real data renders. (The UI run doesn't exercise training-side Python, so also `python -m py_compile` any touched `.py`.)
9. **Pre-existing type errors (context, not a task):** the fork's route handlers/pages use Next.js-14-style sync `params` instead of Next 15's `Promise` params, so type-checking surfaces ~40 `TS2344` errors under `.next/types/**`. They're pre-existing across *all* dynamic routes, unrelated to the merge, and the build ignores them via `typescript.ignoreBuildErrors: true` in `ui/next.config.ts`.
10. **Watch for a concurrent partial sync landing on `main` while you work.** This fork sometimes receives an independent, *partial* upstream merge via GitHub's "Merge pull request #NN from ostris/main" (a sync PR that merges only an older upstream commit, not the tip). If one lands after you branched, `gh pr merge` fails with *"the merge commit cannot be cleanly created"* even though your local branch is clean. Fix: `git fetch origin`, then from your sync branch `git merge origin/main` (it auto-resolves cleanly because your full-tip merge already contains that older commit as an ancestor — a superset), re-run the build to confirm, `git push`, then merge the PR. Right before merging any sync PR, re-check `git rev-parse origin/main` hasn't moved since you branched. Merge sync PRs with `--merge` (a real merge commit), **never `--squash`** — squashing severs `main`'s ancestry link to `upstream/main` and breaks `git merge-base main upstream/main` for the next sync.
