import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

const execFileAsync = promisify(execFile);

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { videoPath, intervalSeconds, destinationDataset } = body;
    const datasetsPath = await getDatasetsRoot();
    const trainingPath = await getTrainingFolder();

    if (!videoPath || (!videoPath.startsWith(datasetsPath) && !videoPath.startsWith(trainingPath))) {
      return NextResponse.json({ error: 'Invalid video path' }, { status: 400 });
    }

    if (videoPath.includes('..')) {
      return NextResponse.json({ error: 'Invalid video path' }, { status: 400 });
    }

    if (!/\.(mp4|avi|mov|mkv|wmv|m4v|flv)$/i.test(videoPath)) {
      return NextResponse.json({ error: 'Not a video file' }, { status: 400 });
    }

    const interval = parseFloat(intervalSeconds);
    if (isNaN(interval) || interval <= 0) {
      return NextResponse.json({ error: 'Invalid interval' }, { status: 400 });
    }

    if (!fs.existsSync(videoPath)) {
      return NextResponse.json({ error: 'Video not found' }, { status: 404 });
    }

    if (!destinationDataset || typeof destinationDataset !== 'string') {
      return NextResponse.json({ error: 'Destination dataset is required' }, { status: 400 });
    }
    const cleanName = destinationDataset.toLowerCase().replace(/[^a-z0-9]+/g, '_');
    if (!cleanName) {
      return NextResponse.json({ error: 'Invalid destination dataset name' }, { status: 400 });
    }

    const destinationDir = path.join(datasetsPath, cleanName);
    if (!fs.existsSync(destinationDir)) {
      fs.mkdirSync(destinationDir, { recursive: true });
    }

    const ext = path.extname(videoPath);
    const base = path.basename(videoPath, ext);
    const outputPattern = path.join(destinationDir, `${base}_frame_%05d.jpg`);

    await execFileAsync('ffmpeg', [
      '-i', videoPath,
      '-vf', `fps=1/${interval}`,
      '-q:v', '2',
      outputPattern,
    ]);

    return NextResponse.json({ success: true, destinationDataset: cleanName });
  } catch (error: any) {
    console.error('Error extracting frames:', error);
    if (error.code === 'ENOENT') {
      return NextResponse.json({ error: 'ffmpeg is not installed or not found in PATH' }, { status: 500 });
    }
    const message = error.stderr ? `Failed to extract frames: ${error.stderr}` : 'Failed to extract frames';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
