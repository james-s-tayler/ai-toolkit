import { defineConfig, devices } from '@playwright/test';

/**
 * E2E regression suite for the ai-toolkit UI.
 *
 * These tests exist to catch the class of breakage that upstream syncs introduce
 * (e.g. an API response shape changing out from under a fork-customized page).
 * They run against an ALREADY-RUNNING instance — this repo's rule is that the app
 * is only ever launched via the run script (see CLAUDE.md), so this config does
 * NOT start a web server. Start the app the normal way, then run the suite.
 *
 * Config via env:
 *   E2E_BASE_URL      base URL of the running app (default http://localhost:8675)
 *   AI_TOOLKIT_AUTH   bearer token the app was started with (blank if auth disabled)
 */
export default defineConfig({
  testDir: './tests/e2e',
  // The suite mutates a throwaway dataset in some specs; keep it serial so runs
  // are deterministic and never race on shared server state.
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  timeout: 60_000,
  expect: { timeout: 15_000 },
  reporter: process.env.CI ? [['github'], ['html', { open: 'never' }]] : [['list']],
  use: {
    baseURL: process.env.E2E_BASE_URL || 'http://localhost:8675',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    headless: true,
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
});
