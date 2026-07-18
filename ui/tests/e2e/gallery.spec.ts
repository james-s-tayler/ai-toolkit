import { test, expect, assertNoForbiddenErrors, listGalleryFolders, BASE_URL, authHeaders } from './helpers';

/**
 * Gallery is a fork-only feature (no upstream equivalent). It uses the same
 * DatasetImageViewer lightbox as the datasets page, so it shares the img_path
 * failure mode — worth its own regression coverage.
 */

test('gallery list renders registered folders', async ({ page, request, consoleErrors }) => {
  const folders = await listGalleryFolders(request);
  test.skip(folders.length === 0, 'no gallery folders registered');

  await page.goto('/gallery');
  await expect(page.getByRole('link', { name: new RegExp(escapeRegex(folders[0].path)) })).toBeVisible();
  assertNoForbiddenErrors(consoleErrors);
});

test('opening a gallery folder renders its images', async ({ page, request, consoleErrors }) => {
  const folders = await listGalleryFolders(request);
  test.skip(folders.length === 0, 'no gallery folders registered');

  // Find a folder that actually has images so the render assertion is meaningful.
  let folderId: number | null = null;
  for (const f of folders) {
    const res = await request.get(`${BASE_URL}/api/gallery/images?folderPath=${encodeURIComponent(f.path)}`, {
      headers: authHeaders,
    });
    if (res.ok()) {
      const body = (await res.json()) as { images?: unknown[] };
      if (Array.isArray(body.images) && body.images.length > 0) {
        folderId = f.id;
        break;
      }
    }
  }
  test.skip(folderId === null, 'no gallery folder with images');

  await page.goto(`/gallery/${folderId}`);
  const thumb = page.locator('img[src*="/api/gallery/img"], img[src*="/api/img/"]').first();
  await expect(thumb).toBeVisible();
  await expect.poll(() => thumb.evaluate((el: HTMLImageElement) => el.complete && el.naturalWidth)).toBeGreaterThan(0);
  assertNoForbiddenErrors(consoleErrors);
});

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
