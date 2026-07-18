import { test, expect, assertNoForbiddenErrors, findDatasetWithImages, findAllVideoDataset } from './helpers';

/**
 * Fork-specific features that have no upstream equivalent (or that upstream
 * implements differently and we discard on sync). These modals wire together
 * fork-only components + API routes, so an upstream sync can silently break
 * their imports/props. We assert each opens and shows its controls — we do NOT
 * execute the (slow, mutating) operations.
 */

test('bulk caption modal opens and loads presets', async ({ page, request, consoleErrors }) => {
  const dataset = await findDatasetWithImages(request);
  test.skip(!dataset, 'no dataset with images available');

  await page.goto(`/datasets/${dataset}`);
  await page.locator('img[src*="/api/img/"]').first().waitFor();

  await page.getByRole('button', { name: 'Caption Images' }).click();

  // The fork's BulkCaptionModal.
  await expect(page.getByText('Caption All Images')).toBeVisible();
  // Presets are fetched from /api/caption-presets and populate this select.
  await expect(page.getByLabel('Caption preset')).toBeVisible();

  assertNoForbiddenErrors(consoleErrors);
});

test('per-image caption modal opens', async ({ page, request, consoleErrors }) => {
  const dataset = await findDatasetWithImages(request);
  test.skip(!dataset, 'no dataset with images available');

  await page.goto(`/datasets/${dataset}`);
  await page.locator('img[src*="/api/img/"]').first().waitFor();

  await page.getByRole('button', { name: 'Generate AI caption' }).first().click();
  await expect(page.getByRole('heading', { name: 'Generate Caption' })).toBeVisible();

  assertNoForbiddenErrors(consoleErrors);
});

test('bulk-video action bar appears and Re-encode FPS modal opens', async ({ page, request, consoleErrors }) => {
  const dataset = await findAllVideoDataset(request);
  test.skip(!dataset, 'no all-video dataset available for bulk-video features');

  await page.goto(`/datasets/${dataset}`);
  // Wait for the video cards to mount.
  await expect(page.locator('video').first()).toBeVisible();

  await page.getByRole('button', { name: 'Multi-Select' }).click();
  // Ctrl+A selects all items (all videos here) → the video-only action bar shows.
  await page.keyboard.press('Control+a');

  // Fork-only bulk-video actions, gated on allSelectedAreVideos.
  await expect(page.getByRole('button', { name: 'Re-encode FPS' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Split Videos' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Extract Frames' })).toBeVisible();

  await page.getByRole('button', { name: 'Re-encode FPS' }).click();
  await expect(page.getByRole('heading', { name: /Re-encode .* FPS/i })).toBeVisible();
  await expect(page.getByLabel('Target framerate')).toBeVisible();

  assertNoForbiddenErrors(consoleErrors);
});
