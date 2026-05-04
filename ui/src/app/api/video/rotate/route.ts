import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

const execFileAsync = promisify(execFile);

export async function POST(request: Request) {
  let tempOutput: string | null = null;
  try {
    const body = await request.json();
    const { videoPath, direction } = body;
    let datasetsPath = await getDatasetsRoot();
    const trainingPath = await getTrainingFolder();

    if (!videoPath.startsWith(datasetsPath) && !videoPath.startsWith(trainingPath)) {
      return NextResponse.json({ error: 'Invalid video path' }, { status: 400 });
    }

    if (videoPath.includes('..')) {
      return NextResponse.json({ error: 'Invalid video path' }, { status: 400 });
    }

    if (!/\.(mp4|avi|mov|mkv|wmv|m4v|flv)$/i.test(videoPath)) {
      return NextResponse.json({ error: 'Not a video file' }, { status: 400 });
    }

    if (direction !== 'left' && direction !== 'right') {
      return NextResponse.json({ error: 'Invalid direction' }, { status: 400 });
    }

    if (!fs.existsSync(videoPath)) {
      return NextResponse.json({ error: 'Video not found' }, { status: 404 });
    }

    // transpose=1 -> 90° clockwise (right), transpose=2 -> 90° counterclockwise (left)
    const transpose = direction === 'right' ? '1' : '2';

    const dir = path.dirname(videoPath);
    const ext = path.extname(videoPath);
    const base = path.basename(videoPath, ext);
    tempOutput = path.join(dir, `${base}_rotated_temp${ext}`);

    await execFileAsync('ffmpeg', [
      '-y',
      '-i', videoPath,
      '-vf', `transpose=${transpose}`,
      '-metadata:s:v', 'rotate=0',
      '-c:v', 'libx264',
      '-c:a', 'copy',
      tempOutput,
    ]);

    fs.unlinkSync(videoPath);
    fs.renameSync(tempOutput, videoPath);
    tempOutput = null;

    return NextResponse.json({ success: true });
  } catch (error: any) {
    console.error('Error rotating video:', error);
    if (error.code === 'ENOENT') {
      return NextResponse.json({ error: 'ffmpeg is not installed or not found in PATH' }, { status: 500 });
    }
    const message = error.stderr ? `Failed to rotate video: ${error.stderr}` : 'Failed to rotate video';
    return NextResponse.json({ error: message }, { status: 500 });
  } finally {
    if (tempOutput && fs.existsSync(tempOutput)) {
      try {
        fs.unlinkSync(tempOutput);
      } catch (_) {
        // ignore cleanup errors
      }
    }
  }
}
