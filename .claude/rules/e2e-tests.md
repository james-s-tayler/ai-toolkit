# E2E test rules

Playwright end-to-end tests for the UI live in `ui/tests/e2e/` (config:
`ui/playwright.config.ts`, shared helpers: `ui/tests/e2e/helpers.ts`). They exist
to catch merge regressions — especially an API response shape changing out from
under a fork-customized page (the `img_path`/`listImages` bug that started this
suite). Follow these rules when adding or changing e2e tests.

## 1. Never leave shared data mutated (most important)

The suite runs against the **same running instance the user keeps real
datasets/gallery folders in for manual testing**. A test run MUST leave that data
byte-for-byte as it found it.

- **Read-only by default.** Prefer tests that navigate and assert without changing
  anything (open a page, open a modal, assert controls render). Do NOT submit
  destructive/long-running operations (bulk re-encode, captioning inference,
  scoring) — assert the modal/controls open, not that the job runs.
- **Mutation tests operate only on a throwaway dataset** created in `beforeAll`
  and deleted in `afterAll` via `createSeededDataset()` / `deleteDataset()` in
  `helpers.ts`. Never rotate/caption/trash/move a real dataset's images.
- Teardown must be unconditional (`afterAll`), so a mid-test failure still cleans
  up. `beforeAll` also deletes any leftover throwaway from an interrupted run
  before re-seeding.
- If you must touch existing data to assert something, snapshot and restore it in
  the same test — but strongly prefer a throwaway fixture instead.

## 2. Keep tests data-agnostic

Don't hard-code dataset names, gallery folders, or counts — the instance's data
changes. Discover at runtime with the `helpers.ts` finders
(`findDatasetWithImages`, `findComparablePair`, `findAllVideoDataset`,
`listGalleryFolders`) and `test.skip(...)` when the environment has nothing to
exercise. Tests must pass on a fresh/empty instance (by skipping), not just on
this machine.

## 3. Launch the app only via the run script

Per the repo `CLAUDE.md`, the app is started **only** through
`d-run-safe-ai-toolkit.sh`. The Playwright config therefore has **no `webServer`**
— tests run against an already-running instance. Never add a `webServer` that
builds or launches the app another way. After changing app source that a test
depends on, rebuild/restart via the run script before re-running the suite.

## 4. Auth + running

Tests seed the bearer token into `localStorage` and send it on API calls. Run:

```bash
cd ui && AI_TOOLKIT_AUTH=<token the app was started with> npm run test:e2e
```

Omit the token only if the app runs with auth disabled. `E2E_BASE_URL` overrides
the target (default `http://localhost:8675`). Install the browser once with
`npx playwright install chromium`.

## 5. Assertions

- Assert **real rendered data**, not just that an element exists — e.g. an image
  has actually loaded (`el.complete && el.naturalWidth > 0`), not merely that an
  `<img>` tag is present.
- Every page/flow test collects console + page errors (the `consoleErrors`
  fixture) and calls `assertNoForbiddenErrors(consoleErrors)`. Add new failure
  signatures to `FORBIDDEN_ERROR_PATTERNS` in `helpers.ts` when a new class of
  runtime error is worth guarding against.
- Prefer role/label selectors (`getByRole`, `getByLabel`) over CSS/text. When
  text appears in multiple roles (heading + button), scope with `getByRole` to
  avoid strict-mode violations.

## 6. Keep it green and out of the way

`test-results/`, `playwright-report/`, and the browser cache are git-ignored.
`@playwright/test` is a `devDependency`; the browser download is intentionally
NOT wired into `build_and_start`, so app startup is unaffected. Don't add e2e runs
to the run script.
