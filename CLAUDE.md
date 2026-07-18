# ai-toolkit (fork)

Personal fork of `ostris/ai-toolkit`. Notes for agents working in this repo.

## How to run / verify

Start the app via `d-run-safe-ai-toolkit.sh` (on PATH at `/home/me/source/scripts/d-run-safe-ai-toolkit.sh`). It:

- Refuses to run if certain external drives are mounted (safety).
- Sets up `~/.cache/huggingface` as a symlink to `/media/me/big_monster/.cache/huggingface`.
- Activates the project venv, `cd`s into `ui/`, and execs `npm run build_and_start`.
- Runs the whole thing under a systemd user scope named `ai-toolkit.scope` with memory limits.

The UI listens on **port 8675** in this production-style start (`next start --port 8675`). The cron worker runs alongside under `concurrently`.

To stop after testing: `systemctl --user stop ai-toolkit.scope`.

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
