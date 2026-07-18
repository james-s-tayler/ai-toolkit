import { test as base, expect, type APIRequestContext } from '@playwright/test';

export const BASE_URL = process.env.E2E_BASE_URL || 'http://localhost:8675';
const TOKEN = process.env.AI_TOOLKIT_AUTH || '';

/** Headers for direct API calls (setup/teardown/discovery). Empty when auth is off. */
export const authHeaders: Record<string, string> = TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {};

const VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv', '.webm'];
const AUDIO_EXTS = ['.mp3', '.wav', '.flac', '.ogg'];

export function isVideoPath(p: string): boolean {
  return VIDEO_EXTS.some(ext => p.toLowerCase().endsWith(ext));
}
export function isAudioPath(p: string): boolean {
  return AUDIO_EXTS.some(ext => p.toLowerCase().endsWith(ext));
}

/**
 * Console-error patterns that should never appear on a healthy page. The
 * img_path/localeCompare entries are the exact signature of the upstream-sync
 * regression this suite was created to guard against.
 */
const FORBIDDEN_ERROR_PATTERNS = [
  /localeCompare/i,
  /img_path is undefined/i,
  /Error fetching images/i,
  /is not a function/i,
  /Cannot read propert/i,
  /can't access property/i,
  /is not defined/i,
  /Minified React error/i,
];

/**
 * Extended test that (1) seeds the auth token into localStorage before any app
 * code runs and (2) collects console/page errors into `consoleErrors`.
 */
export const test = base.extend<{ consoleErrors: string[] }>({
  context: async ({ context }, use) => {
    await context.addInitScript(token => {
      if (token) window.localStorage.setItem('AI_TOOLKIT_AUTH', token);
    }, TOKEN);
    await use(context);
  },
  consoleErrors: async ({ page }, use) => {
    const errors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    page.on('pageerror', err => errors.push(String(err?.message ?? err)));
    await use(errors);
  },
});

export { expect };

/** Fail if any collected console error matches a forbidden pattern. */
export function assertNoForbiddenErrors(errors: string[]) {
  const hits = errors.filter(line => FORBIDDEN_ERROR_PATTERNS.some(p => p.test(line)));
  expect(hits, `Forbidden console errors:\n${hits.join('\n')}`).toEqual([]);
}

// ---------------------------------------------------------------------------
// Data discovery helpers — keep specs portable (no hard-coded dataset names).
// ---------------------------------------------------------------------------

export async function listDatasets(request: APIRequestContext): Promise<string[]> {
  const res = await request.get(`${BASE_URL}/api/datasets/list`, { headers: authHeaders });
  if (!res.ok()) return [];
  return (await res.json()) as string[];
}

/** Full img_paths for a dataset, rebuilt from the compact {root, images} payload. */
export async function listDatasetImages(request: APIRequestContext, name: string): Promise<string[]> {
  const res = await request.post(`${BASE_URL}/api/datasets/listImages`, {
    headers: { ...authHeaders, 'Content-Type': 'application/json' },
    data: { datasetName: name },
  });
  if (!res.ok()) return [];
  const body = (await res.json()) as { root?: string; images?: unknown };
  const root = body.root ?? '';
  const images = Array.isArray(body.images) ? body.images : [];
  return images.map((e: any) => (typeof e === 'string' ? root + e : e.img_path));
}

/** First dataset that contains at least one image (non-video, non-audio). */
export async function findDatasetWithImages(request: APIRequestContext): Promise<string | null> {
  for (const ds of await listDatasets(request)) {
    const imgs = await listDatasetImages(request, ds);
    if (imgs.some(p => !isVideoPath(p) && !isAudioPath(p))) return ds;
  }
  return null;
}

/** Two distinct datasets whose image basenames fully overlap (valid Compare pair). */
export async function findComparablePair(request: APIRequestContext): Promise<[string, string] | null> {
  const datasets = await listDatasets(request);
  const byNames: { ds: string; names: Set<string> }[] = [];
  for (const ds of datasets) {
    const names = new Set((await listDatasetImages(request, ds)).map(p => p.replace(/^.*[\\/]/, '')));
    if (names.size > 0) byNames.push({ ds, names });
  }
  for (let i = 0; i < byNames.length; i++) {
    for (let j = i + 1; j < byNames.length; j++) {
      const a = byNames[i];
      const b = byNames[j];
      const aSubsetB = [...a.names].every(n => b.names.has(n));
      const bSubsetA = [...b.names].every(n => a.names.has(n));
      if (aSubsetB || bSubsetA) return [a.ds, b.ds];
    }
  }
  return null;
}

/** First dataset whose files are ALL videos (so the bulk-video action bar shows). */
export async function findAllVideoDataset(request: APIRequestContext): Promise<string | null> {
  for (const ds of await listDatasets(request)) {
    const files = await listDatasetImages(request, ds);
    if (files.length > 0 && files.every(p => isVideoPath(p))) return ds;
  }
  return null;
}

export async function listGalleryFolders(request: APIRequestContext): Promise<{ id: number; path: string }[]> {
  const res = await request.get(`${BASE_URL}/api/gallery/list`, { headers: authHeaders });
  if (!res.ok()) return [];
  return (await res.json()) as { id: number; path: string }[];
}

// ---------------------------------------------------------------------------
// Throwaway-dataset lifecycle — mutation specs operate here, never on real data.
// ---------------------------------------------------------------------------

// 1x1 PNG — smallest valid image the UI will render and rotate.
const TINY_PNG_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

/** Create a dataset (returns the server-cleaned name) and upload `count` tiny PNGs. */
export async function createSeededDataset(request: APIRequestContext, rawName: string, count = 2): Promise<string> {
  const createRes = await request.post(`${BASE_URL}/api/datasets/create`, {
    headers: { ...authHeaders, 'Content-Type': 'application/json' },
    data: { name: rawName },
  });
  const { name } = (await createRes.json()) as { name: string };
  const png = Buffer.from(TINY_PNG_BASE64, 'base64');
  for (let i = 0; i < count; i++) {
    const res = await request.post(`${BASE_URL}/api/datasets/upload`, {
      headers: authHeaders,
      multipart: {
        datasetName: name,
        files: { name: `e2e_${i}.png`, mimeType: 'image/png', buffer: png },
      },
    });
    expect(res.ok(), `upload of e2e_${i}.png failed`).toBeTruthy();
  }
  return name;
}

export async function deleteDataset(request: APIRequestContext, name: string): Promise<void> {
  await request.post(`${BASE_URL}/api/datasets/delete`, {
    headers: { ...authHeaders, 'Content-Type': 'application/json' },
    data: { name },
  });
}
