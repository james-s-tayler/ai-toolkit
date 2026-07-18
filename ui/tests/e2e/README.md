# E2E regression suite

Playwright tests that guard the fork's UI against the kind of breakage upstream
syncs introduce — most concretely, an API response shape changing out from under
a fork-customized page (see the `img_path` regression these tests were born from).

## What it covers

| Spec | Covers |
| --- | --- |
| `smoke.spec.ts` | Every sidebar page renders its shell with no forbidden console errors |
| `datasets.spec.ts` | Dataset viewer thumbnails render, lightbox opens, Compare modal → Compare page (the exact `listImages` regression) |
| `gallery.spec.ts` | Gallery folder list + folder image rendering (fork-only feature) |
| `fork-features.spec.ts` | Bulk caption modal + presets, per-image caption modal, bulk-video action bar + Re-encode FPS modal |
| `dataset-mutations.spec.ts` | Caption edit persistence, rotate, trash — on a **throwaway** dataset created/torn down by the test |

Specs are **data-agnostic**: they discover a suitable dataset/gallery folder at
runtime (`helpers.ts`) and `test.skip()` when the environment has nothing to
exercise, so the suite is portable rather than tied to one machine's data.

## Running it

The app is only ever launched via the run script (see the repo `CLAUDE.md`), so
this suite does **not** start a server — it runs against an already-running
instance.

1. Start the app the normal way (the run script), so it's serving on `:8675`.
2. Install the browser once: `npx playwright install chromium`.
3. Run:

   ```bash
   cd ui
   AI_TOOLKIT_AUTH=<the token the app was started with> npm run test:e2e
   ```

   - `AI_TOOLKIT_AUTH` must match the token the running app was started with; the
     tests seed it into `localStorage` and send it on API calls. Omit it only if
     the app is running with auth disabled.
   - `E2E_BASE_URL` overrides the target (default `http://localhost:8675`).
   - `npm run test:e2e:ui` opens the Playwright UI runner; add `--trace on` to
     `test:e2e` to always capture traces.

## Not covered (intentionally)

- Training-side Python (samplers, adapters) — this suite is UI-only.
- Long-running / destructive operations (actually executing bulk re-encode,
  captioning inference, scoring). We assert their modals/controls open, not that
  the jobs run, to keep the suite fast and side-effect-free.
