import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';
import { videoExtensions } from '@/utils/basic';
import { getMediaMetadata, isCacheFile } from '@/utils/mediaMetadata';

interface ImageStats {
  totalCount: number;
  imageCount: number;
  videoCount: number;
  totalVideoDuration: number;
  resolutionBreakdown: { [resolution: string]: number };
}

export async function GET(request: Request) {
  const datasetsPath = await getDatasetsRoot();
  const { searchParams } = new URL(request.url);
  const datasetName = searchParams.get('datasetName');

  // Validate datasetName
  if (!datasetName || typeof datasetName !== 'string' || datasetName.trim() === '') {
    return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
  }

  // Prevent path traversal attacks
  if (datasetName.includes('..') || datasetName.includes('/') || datasetName.includes('\\')) {
    return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
  }

  const datasetFolder = path.join(datasetsPath, datasetName);

  // Verify the resolved path is within datasetsPath
  if (!datasetFolder.startsWith(datasetsPath)) {
    return NextResponse.json({ error: 'Invalid dataset path' }, { status: 400 });
  }

  // Check if folder exists
  try {
    await fs.promises.access(datasetFolder);
  } catch {
    return NextResponse.json({ error: `Folder '${datasetName}' not found` }, { status: 404 });
  }

  // Initialize stats with defaults
  let totalCount = 0;
  let imageCount = 0;
  let videoCount = 0;
  let totalVideoDuration = 0;
  const resolutionBreakdown: { [resolution: string]: number } = {};

  try {
    // Find all images recursively (async to avoid blocking the event loop)
    const imageFiles = await findImagesRecursively(datasetFolder);
    totalCount = imageFiles.length;

    // Separate video files from image files in a single pass
    const videoFiles: string[] = [];
    const nonVideoFiles: string[] = [];
    for (const f of imageFiles) {
      if (videoExtensions.includes(path.extname(f).toLowerCase())) {
        videoFiles.push(f);
      } else {
        nonVideoFiles.push(f);
      }
    }
    imageCount = nonVideoFiles.length;
    videoCount = videoFiles.length;

    const CONCURRENCY_LIMIT = 10;

    // Get video durations concurrently (cached via shared helper)
    for (let i = 0; i < videoFiles.length; i += CONCURRENCY_LIMIT) {
      const batch = videoFiles.slice(i, i + CONCURRENCY_LIMIT);
      const metas = await Promise.all(batch.map(vp => getMediaMetadata(vp)));
      totalVideoDuration += metas.reduce((sum, m) => sum + (m.duration ?? 0), 0);
    }

    // Get resolution for each image with concurrent processing
    for (let i = 0; i < nonVideoFiles.length; i += CONCURRENCY_LIMIT) {
      const batch = nonVideoFiles.slice(i, i + CONCURRENCY_LIMIT);
      await Promise.allSettled(
        batch.map(async imgPath => {
          const meta = await getMediaMetadata(imgPath);
          if (meta.width && meta.height) {
            const resolution = `${meta.width}x${meta.height}`;
            resolutionBreakdown[resolution] = (resolutionBreakdown[resolution] || 0) + 1;
          } else {
            const unknownKey = 'unknown resolution';
            resolutionBreakdown[unknownKey] = (resolutionBreakdown[unknownKey] || 0) + 1;
          }
        })
      );
    }
  } catch (error) {
    console.error('Error calculating image stats:', error);
  }

  // Always return stats with what we have, even if there were errors
  const stats: ImageStats = {
    totalCount,
    imageCount,
    videoCount,
    totalVideoDuration,
    resolutionBreakdown,
  };

  return NextResponse.json(stats);
}

/**
 * Recursively finds all image files in a directory and its subdirectories.
 * Uses an iterative BFS with async I/O to avoid blocking the event loop,
 * unbounded concurrency, and symlink loops.
 * @param dir Directory to search
 * @returns Array of absolute paths to image files
 */
async function findImagesRecursively(dir: string): Promise<string[]> {
  const imageExtensions = ['.png', '.jpg', '.jpeg', '.webp', '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv'];
  const results: string[] = [];
  const queue: string[] = [dir];

  while (queue.length > 0) {
    const current = queue.shift()!;
    let dirents: fs.Dirent[];
    try {
      dirents = await fs.promises.readdir(current, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const dirent of dirents) {
      // Skip symlinks to avoid loops and unintended traversal
      if (dirent.isSymbolicLink()) continue;
      const itemPath = path.join(current, dirent.name);
      if (dirent.isDirectory() && dirent.name !== '_controls' && !dirent.name.startsWith('.')) {
        queue.push(itemPath);
      } else if (dirent.isFile()) {
        if (isCacheFile(dirent.name)) continue;
        const ext = path.extname(dirent.name).toLowerCase();
        if (imageExtensions.includes(ext) && !dirent.name.startsWith('trash_')) {
          results.push(itemPath);
        }
      }
    }
  }

  return results;
}
