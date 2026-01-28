import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import { getDatasetsRoot } from '@/server/settings';

interface ImageStats {
  totalCount: number;
  resolutionBreakdown: { [resolution: string]: number };
}

export async function POST(request: Request) {
  const datasetsPath = await getDatasetsRoot();
  const body = await request.json();
  const { datasetName } = body;
  const datasetFolder = path.join(datasetsPath, datasetName);

  try {
    // Check if folder exists
    if (!fs.existsSync(datasetFolder)) {
      return NextResponse.json({ error: `Folder '${datasetName}' not found` }, { status: 404 });
    }

    // Find all images recursively
    const imageFiles = findImagesRecursively(datasetFolder);
    const totalCount = imageFiles.length;
    const resolutionBreakdown: { [resolution: string]: number } = {};

    // Get resolution for each image
    for (const imgPath of imageFiles) {
      try {
        const ext = path.extname(imgPath).toLowerCase();
        // Skip video files for now as getting their resolution requires different approach
        if (['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv'].includes(ext)) {
          const videoKey = 'video (resolution unavailable)';
          resolutionBreakdown[videoKey] = (resolutionBreakdown[videoKey] || 0) + 1;
          continue;
        }

        const metadata = await sharp(imgPath).metadata();
        const width = metadata.width || 0;
        const height = metadata.height || 0;
        const resolution = `${width}x${height}`;
        resolutionBreakdown[resolution] = (resolutionBreakdown[resolution] || 0) + 1;
      } catch (error) {
        console.error(`Error reading image metadata for ${imgPath}:`, error);
        // If we can't read the image, count it as unknown
        const unknownKey = 'unknown resolution';
        resolutionBreakdown[unknownKey] = (resolutionBreakdown[unknownKey] || 0) + 1;
      }
    }

    const stats: ImageStats = {
      totalCount,
      resolutionBreakdown,
    };

    return NextResponse.json(stats);
  } catch (error) {
    console.error('Error calculating image stats:', error);
    return NextResponse.json({ error: 'Failed to process request' }, { status: 500 });
  }
}

/**
 * Recursively finds all image files in a directory and its subdirectories
 * @param dir Directory to search
 * @returns Array of absolute paths to image files
 */
function findImagesRecursively(dir: string): string[] {
  const imageExtensions = ['.png', '.jpg', '.jpeg', '.webp', '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv'];
  let results: string[] = [];

  const items = fs.readdirSync(dir);

  for (const item of items) {
    const itemPath = path.join(dir, item);
    const stat = fs.statSync(itemPath);

    if (stat.isDirectory() && item !== '_controls' && !item.startsWith('.')) {
      // If it's a directory, recursively search it
      results = results.concat(findImagesRecursively(itemPath));
    } else {
      // If it's a file, check if it's an image
      const ext = path.extname(itemPath).toLowerCase();
      if (imageExtensions.includes(ext)) {
        results.push(itemPath);
      }
    }
  }

  return results;
}
