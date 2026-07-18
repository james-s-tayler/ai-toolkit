import { test, expect, assertNoForbiddenErrors } from './helpers';

/**
 * Every top-level page must render its shell and log no forbidden console errors.
 * This is the cheapest, broadest guard against merge breakage — a page that
 * throws on render (bad import, changed API shape, renamed prop) fails here.
 */
const PAGES: { path: string; heading: RegExp }[] = [
  { path: '/dashboard', heading: /dashboard/i },
  { path: '/jobs', heading: /job|queue/i },
  { path: '/jobs/new', heading: /job|training|config|name/i },
  { path: '/datasets', heading: /dataset/i },
  { path: '/gallery', heading: /galler/i },
  { path: '/downloads', heading: /download/i },
  { path: '/caption-presets', heading: /caption|preset/i },
  { path: '/trash', heading: /trash/i },
  { path: '/settings', heading: /setting/i },
];

for (const { path, heading } of PAGES) {
  test(`page loads: ${path}`, async ({ page, consoleErrors }) => {
    await page.goto(path, { waitUntil: 'networkidle' });

    // The sidebar (rendered by the app shell) proves auth passed and layout mounted.
    await expect(page.getByRole('link', { name: 'Datasets' })).toBeVisible();

    // Page-specific content surfaced (not a blank/error shell).
    await expect(page.getByText(heading).first()).toBeVisible();

    // No auth redirect back to a login/blank state.
    expect(page.url()).toContain(path === '/dashboard' ? '/dashboard' : path);

    assertNoForbiddenErrors(consoleErrors);
  });
}

test('root redirects to dashboard', async ({ page }) => {
  await page.goto('/');
  await page.waitForURL('**/dashboard');
  await expect(page.getByRole('link', { name: 'Datasets' })).toBeVisible();
});
