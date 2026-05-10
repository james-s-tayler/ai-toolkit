import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { videoExtensions, imgExtensions } from '@/utils/basic';

const execFileAsync = promisify(execFile);

const CACHE_SCHEMA_VERSION = 1;
const CACHE_SUFFIX = '.meta.json';

export interface MediaMetadata {
  width?: number;
  height?: number;
  duration?: number;
  fps?: number;
}

interface CachedEntry extends MediaMetadata {
  schemaVersion: number;
  sourceMtimeMs: number;
}

export function getCachePath(mediaPath: string): string {
  const parsed = path.parse(mediaPath);
  return path.join(parsed.dir, `${parsed.name}${CACHE_SUFFIX}`);
}

export function isCacheFile(fileName: string): boolean {
  return fileName.endsWith(CACHE_SUFFIX);
}

function parseFrameRate(raw: string | undefined): number | undefined {
  if (!raw) return undefined;
  const trimmed = raw.trim();
  if (!trimmed || trimmed === '0/0') return undefined;
  if (trimmed.includes('/')) {
    const [numStr, denStr] = trimmed.split('/');
    const num = parseFloat(numStr);
    const den = parseFloat(denStr);
    if (!isFinite(num) || !isFinite(den) || den === 0) return undefined;
    return num / den;
  }
  const v = parseFloat(trimmed);
  return isFinite(v) ? v : undefined;
}

async function probeVideo(videoPath: string): Promise<MediaMetadata> {
  try {
    const { stdout } = await execFileAsync('ffprobe', [
      '-v', 'error',
      '-select_streams', 'v:0',
      '-show_entries', 'stream=width,height,r_frame_rate',
      '-show_entries', 'format=duration',
      '-of', 'json',
      videoPath,
    ]);
    const parsed = JSON.parse(stdout) as {
      streams?: { width?: number; height?: number; r_frame_rate?: string }[];
      format?: { duration?: string };
    };
    const stream = parsed.streams?.[0];
    const result: MediaMetadata = {};
    if (stream?.width && stream.width > 0) result.width = stream.width;
    if (stream?.height && stream.height > 0) result.height = stream.height;
    const fps = parseFrameRate(stream?.r_frame_rate);
    if (fps !== undefined && fps > 0) result.fps = fps;
    const duration = parseFloat(parsed.format?.duration ?? '');
    if (isFinite(duration)) result.duration = duration;
    return result;
  } catch {
    return {};
  }
}

async function probeImage(imgPath: string): Promise<MediaMetadata> {
  try {
    const meta = await sharp(imgPath).metadata();
    const result: MediaMetadata = {};
    if (meta.width) result.width = meta.width;
    if (meta.height) result.height = meta.height;
    return result;
  } catch {
    return {};
  }
}

function readCache(mediaPath: string, sourceMtimeMs: number): MediaMetadata | null {
  const cachePath = getCachePath(mediaPath);
  try {
    const raw = fs.readFileSync(cachePath, 'utf-8');
    const parsed = JSON.parse(raw) as CachedEntry;
    if (parsed.schemaVersion !== CACHE_SCHEMA_VERSION) return null;
    if (parsed.sourceMtimeMs !== sourceMtimeMs) return null;
    return {
      width: parsed.width,
      height: parsed.height,
      duration: parsed.duration,
      fps: parsed.fps,
    };
  } catch {
    return null;
  }
}

function writeCache(mediaPath: string, sourceMtimeMs: number, meta: MediaMetadata): void {
  const cachePath = getCachePath(mediaPath);
  const entry: CachedEntry = {
    schemaVersion: CACHE_SCHEMA_VERSION,
    sourceMtimeMs,
    ...meta,
  };
  try {
    fs.writeFileSync(cachePath, JSON.stringify(entry));
  } catch {
    // best-effort cache; ignore write failures (e.g. read-only directory)
  }
}

export async function getMediaMetadata(mediaPath: string): Promise<MediaMetadata> {
  const ext = path.extname(mediaPath).toLowerCase();
  const isVideo = videoExtensions.includes(ext);
  const isImage = imgExtensions.includes(ext);
  if (!isVideo && !isImage) return {};

  let stat: fs.Stats;
  try {
    stat = fs.statSync(mediaPath);
  } catch {
    return {};
  }

  const cached = readCache(mediaPath, stat.mtimeMs);
  if (cached) return cached;

  const probed = isVideo ? await probeVideo(mediaPath) : await probeImage(mediaPath);
  if (probed.width !== undefined || probed.height !== undefined || probed.duration !== undefined || probed.fps !== undefined) {
    writeCache(mediaPath, stat.mtimeMs, probed);
  }
  return probed;
}
