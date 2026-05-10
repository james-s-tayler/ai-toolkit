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
- **Captioning:** this fork has its own captioning solution (`BulkCaptionModal`, `CaptionModal`, `caption_presets`). Upstream later added a different captioning UI (`AutoCaptionButton`, `CaptionDatasetModal`, `CaptionSimpleJob`, `helpers/captionJobConfig`, `helpers/captionOptions`, plus a `caption` job type in `JobActionBar` and a `<CaptionDatasetModal />` mount in `layout.tsx`). When syncing upstream, **discard upstream's captioning entirely** — delete those files if upstream re-adds them, and strip their imports/JSX from `ui/src/app/layout.tsx` and `ui/src/components/JobActionBar.tsx`. Keep this fork's captioning as the only one.
