import { test, expect, assertNoForbiddenErrors, createSeededDataset, deleteDataset } from './helpers';

/**
 * Mutating card actions (caption / rotate / trash), driven through the UI.
 *
 * These operate ONLY on a throwaway dataset this file creates and tears down —
 * never on real data. The dataset is seeded with tiny PNGs via the upload API,
 * then every assertion goes through the browser like a human would.
 */

const RAW_NAME = 'e2e_regression_tmp';
let datasetName: string;

test.beforeAll(async ({ request }) => {
  // Clean any leftover from a previous interrupted run, then seed fresh.
  await deleteDataset(request, 'e2e_regression_tmp');
  datasetName = await createSeededDataset(request, RAW_NAME, 2);
});

test.afterAll(async ({ request }) => {
  if (datasetName) await deleteDataset(request, datasetName);
});

test('caption edit persists across reload', async ({ page, consoleErrors }) => {
  await page.goto(`/datasets/${datasetName}`);
  await page.locator('img[src*="/api/img/"]').first().waitFor();

  const caption = `e2e caption ${Date.now()}`;
  const textarea = page.locator('textarea').first();
  await textarea.click();
  await textarea.fill(caption);

  // The card saves on blur (POST /api/img/caption); wait for that to land.
  const saved = page.waitForResponse(
    r => r.url().includes('/api/img/caption') && r.request().method() === 'POST' && r.ok(),
  );
  await page.getByRole('heading', { name: new RegExp(datasetName) }).click();
  await saved;

  await page.reload();
  await page.locator('img[src*="/api/img/"]').first().waitFor();
  await expect(page.locator('textarea').first()).toHaveValue(caption);

  assertNoForbiddenErrors(consoleErrors);
});

test('rotate updates the thumbnail', async ({ page, consoleErrors }) => {
  await page.goto(`/datasets/${datasetName}`);
  const thumb = page.locator('img[src*="/api/img/"]').first();
  await thumb.waitFor();
  const beforeSrc = await thumb.getAttribute('src');

  await page.getByRole('button', { name: 'Rotate image right' }).first().click();

  // The card busts its cache by bumping the ?v= key after a successful rotate.
  await expect.poll(() => page.locator('img[src*="/api/img/"]').first().getAttribute('src')).not.toBe(beforeSrc);

  assertNoForbiddenErrors(consoleErrors);
});

test('trash removes an image from the grid', async ({ page, consoleErrors }) => {
  await page.goto(`/datasets/${datasetName}`);
  await page.locator('img[src*="/api/img/"]').first().waitFor();

  const before = await page.locator('img[src*="/api/img/"]').count();
  expect(before).toBeGreaterThan(0);

  await page.getByRole('button', { name: 'Move image to trash' }).first().click();

  await expect.poll(() => page.locator('img[src*="/api/img/"]').count()).toBe(before - 1);
  assertNoForbiddenErrors(consoleErrors);
});
