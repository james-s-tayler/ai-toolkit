import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';
import { getMediaMetadata } from '@/utils/mediaMetadata';

const ALLOWED_FPS = [8, 12, 16, 24, 25, 30, 60];

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body) {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }
  const { videoPath, targetFps } = body as { videoPath?: string; targetFps?: number };

  if (typeof videoPath !== 'string' || typeof targetFps !== 'number') {
    return NextResponse.json({ error: 'videoPath and targetFps are required' }, { status: 400 });
  }

  if (!ALLOWED_FPS.includes(targetFps)) {
    return NextResponse.json({ error: `targetFps must be one of ${ALLOWED_FPS.join(', ')}` }, { status: 400 });
  }

  const datasetsPath = await getDatasetsRoot();
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
  if (!fs.existsSync(videoPath)) {
    return NextResponse.json({ error: 'Video not found' }, { status: 404 });
  }

  const dir = path.dirname(videoPath);
  const ext = path.extname(videoPath);
  const base = path.basename(videoPath, ext);
  const tempOutput = path.join(dir, `${base}_reencode_temp${ext}`);

  // Duration is the denominator for percent. Missing => emit percent: null (indeterminate).
  const meta = await getMediaMetadata(videoPath);
  const totalDurationSec = meta.duration && meta.duration > 0 ? meta.duration : null;

  const encoder = new TextEncoder();

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const emit = (obj: unknown) => {
        try {
          controller.enqueue(encoder.encode(JSON.stringify(obj) + '\n'));
        } catch {
          // controller may already be closed
        }
      };

      const cleanupTemp = () => {
        if (fs.existsSync(tempOutput)) {
          try { fs.unlinkSync(tempOutput); } catch { /* ignore */ }
        }
      };

      let child;
      try {
        child = spawn('ffmpeg', [
          '-nostats',
          '-progress', 'pipe:1',
          '-i', videoPath,
          '-r', String(targetFps),
          '-c:v', 'libx264',
          '-c:a', 'aac',
          '-y',
          tempOutput,
        ]);
      } catch (err: any) {
        emit({ type: 'error', message: err?.message || 'Failed to spawn ffmpeg' });
        controller.close();
        return;
      }

      let stdoutBuf = '';
      let stderrBuf = '';
      let lastEmittedPercent = -1;

      const handleStdoutChunk = (chunk: Buffer) => {
        stdoutBuf += chunk.toString('utf-8');
        let nlIdx;
        while ((nlIdx = stdoutBuf.indexOf('\n')) !== -1) {
          const line = stdoutBuf.slice(0, nlIdx).trim();
          stdoutBuf = stdoutBuf.slice(nlIdx + 1);
          if (!line) continue;
          const eq = line.indexOf('=');
          if (eq < 0) continue;
          const key = line.slice(0, eq);
          const value = line.slice(eq + 1);

          if (key === 'out_time_us' || key === 'out_time_ms') {
            // Both are microseconds in modern ffmpeg, despite the _ms name.
            const us = parseInt(value, 10);
            if (!isFinite(us) || us < 0) continue;
            const outTimeSec = us / 1_000_000;
            if (totalDurationSec) {
              const percent = Math.min(100, Math.max(0, (outTimeSec / totalDurationSec) * 100));
              if (Math.abs(percent - lastEmittedPercent) >= 0.5 || percent >= 100) {
                lastEmittedPercent = percent;
                emit({ type: 'progress', percent, outTimeSec });
              }
            } else {
              emit({ type: 'progress', percent: null, outTimeSec });
            }
          }
        }
      };

      child.stdout.on('data', handleStdoutChunk);
      child.stderr.on('data', (chunk: Buffer) => {
        // Cap stderr buffer to keep memory bounded on huge transcodes.
        if (stderrBuf.length < 16_384) stderrBuf += chunk.toString('utf-8');
      });

      const onAbort = () => {
        try { child.kill('SIGTERM'); } catch { /* ignore */ }
      };
      request.signal.addEventListener('abort', onAbort);

      child.on('error', (err: NodeJS.ErrnoException) => {
        request.signal.removeEventListener('abort', onAbort);
        cleanupTemp();
        if (err.code === 'ENOENT') {
          emit({ type: 'error', message: 'ffmpeg is not installed or not found in PATH' });
        } else {
          emit({ type: 'error', message: err.message || 'ffmpeg failed' });
        }
        controller.close();
      });

      child.on('close', (code: number) => {
        request.signal.removeEventListener('abort', onAbort);
        if (request.signal.aborted) {
          cleanupTemp();
          controller.close();
          return;
        }
        if (code !== 0) {
          cleanupTemp();
          const msg = stderrBuf.trim().split('\n').slice(-3).join(' ').slice(0, 1000) || `ffmpeg exited with code ${code}`;
          emit({ type: 'error', message: `Failed to re-encode: ${msg}` });
          controller.close();
          return;
        }
        try {
          fs.unlinkSync(videoPath);
          fs.renameSync(tempOutput, videoPath);
        } catch (err: any) {
          cleanupTemp();
          emit({ type: 'error', message: `Failed to replace original: ${err?.message || 'unknown error'}` });
          controller.close();
          return;
        }
        emit({ type: 'done' });
        controller.close();
      });
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'application/x-ndjson',
      'Cache-Control': 'no-store',
      'X-Accel-Buffering': 'no',
    },
  });
}
