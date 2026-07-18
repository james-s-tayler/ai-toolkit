import { test, expect, assertNoForbiddenErrors, findDatasetWithImages, findComparablePair } from './helpers';

/**
 * Regression coverage for the upstream-sync bug where /api/datasets/listImages
 * switched to a compact { root, images: string[] } payload and the fork's
 * consumers (which expect { img_path }) crashed on sort. If that ever regresses,
 * "thumbnails render" and the forbidden-error assertion both fail here.
 */

test('dataset viewer renders image thumbnails', async ({ page, request, consoleErrors }) => {
  const dataset = await findDatasetWithImages(request);
  test.skip(!dataset, 'no dataset with images available');

  await page.goto('/datasets');
  await page.getByRole('link', { name: dataset!, exact: true }).click();
  await page.waitForURL(`**/datasets/${dataset}`);

  const thumb = page.locator('img[src*="/api/img/"]').first();
  await expect(thumb).toBeVisible();
  await expect.poll(() => thumb.evaluate((el: HTMLImageElement) => el.complete && el.naturalWidth)).toBeGreaterThan(0);

  await expect(page.getByText(/Error fetching images/i)).toHaveCount(0);
  assertNoForbiddenErrors(consoleErrors);
});

test('lightbox opens from a thumbnail', async ({ page, request, consoleErrors }) => {
  const dataset = await findDatasetWithImages(request);
  test.skip(!dataset, 'no dataset with images available');

  await page.goto(`/datasets/${dataset}`);
  await page.locator('img[src*="/api/img/"]').first().waitFor();

  // The fork's DatasetImageViewer lightbox opens via the per-card "Enlarge" button.
  await page.getByRole('button', { name: 'Enlarge image' }).first().click();

  // The enlarged view shows a larger image also served from /api/img/.
  const enlarged = page.locator('img[src*="/api/img/"]');
  await expect(enlarged.first()).toBeVisible();
  assertNoForbiddenErrors(consoleErrors);
});

test('compare modal validates and navigates to the compare page', async ({ page, request, consoleErrors }) => {
  const pair = await findComparablePair(request);
  test.skip(!pair, 'no comparable dataset pair (matching basenames) available');
  const [left, right] = pair!;

  await page.goto('/datasets');
  await page.getByRole('button', { name: 'Compare Datasets' }).click();

  const selects = page.locator('select');
  await expect(selects.first()).toBeVisible();
  await selects.first().selectOption(left);
  await selects.last().selectOption(right);

  await page.getByRole('button', { name: 'Compare', exact: true }).click();
  await page.waitForURL('**/datasets/compare**');

  // Compare page (the other fixed consumer) must render paired thumbnails.
  const cmpThumb = page.locator('img[src*="/api/img/"]').first();
  await expect(cmpThumb).toBeVisible();
  await expect
    .poll(() => cmpThumb.evaluate((el: HTMLImageElement) => el.complete && el.naturalWidth))
    .toBeGreaterThan(0);

  assertNoForbiddenErrors(consoleErrors);
});
