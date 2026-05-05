import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { spawn, ChildProcess } from 'child_process';
import { getDatasetsRoot } from '@/server/settings';
import { TOOLKIT_ROOT } from '@/paths';

const SCRIPT_PATH = path.join(TOOLKIT_ROOT, 'scripts', 'caption_image.py');

const ALLOWED_MODELS = new Set([
  'Qwen/Qwen3-VL-4B-Instruct',
  'Qwen/Qwen3-VL-8B-Instruct',
  'prithivMLmods/Qwen3-VL-4B-Instruct-abliterated-v1',
  'prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it',
]);

interface CaptionState {
  status: 'running' | 'completed' | 'cancelled' | 'error';
  downloading?: boolean;
  caption?: string;
  error?: string;
  process?: ChildProcess;
}

const captionStates = new Map<string, CaptionState>();

function getPythonPath(): string {
  const venvDirs = ['.venv', 'venv'];
  const isWindows = process.platform === 'win32';
  for (const venvDir of venvDirs) {
    const candidate = isWindows
      ? path.join(TOOLKIT_ROOT, venvDir, 'Scripts', 'python.exe')
      : path.join(TOOLKIT_ROOT, venvDir, 'bin', 'python');
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return process.env.PYTHON_EXECUTABLE || 'python3';
}

function validateImgPath(imgPath: unknown, datasetsPath: string): string | null {
  if (!imgPath || typeof imgPath !== 'string') return 'imgPath is required';
  if (imgPath.includes('..')) return 'Invalid image path';
  if (!imgPath.startsWith(datasetsPath)) return 'Invalid image path';
  if (!fs.existsSync(imgPath)) return 'File does not exist';
  return null;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const imgPath = searchParams.get('imgPath');

  if (!imgPath) {
    return NextResponse.json({ error: 'imgPath is required' }, { status: 400 });
  }

  const state = captionStates.get(imgPath);
  if (!state) {
    return NextResponse.json({ status: 'idle' });
  }

  return NextResponse.json({
    status: state.status,
    downloading: state.downloading,
    caption: state.caption,
    error: state.error,
  });
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath, triggerWord, systemPrompt, modelId, useQuorum, videoFps, videoMaxFrames } = body;

    const datasetsPath = await getDatasetsRoot();
    const pathError = validateImgPath(imgPath, datasetsPath);
    if (pathError) {
      const status = pathError === 'File does not exist' ? 404 : 400;
      return NextResponse.json({ error: pathError }, { status });
    }

    const resolvedModelId = modelId || 'prithivMLmods/Qwen3-VL-4B-Instruct-abliterated-v1';
    if (!ALLOWED_MODELS.has(resolvedModelId)) {
      return NextResponse.json({ error: 'Invalid model ID' }, { status: 400 });
    }

    const resolvedVideoFps = typeof videoFps === 'number' && Number.isFinite(videoFps) ? videoFps : 2.0;
    if (resolvedVideoFps < 0.1 || resolvedVideoFps > 16) {
      return NextResponse.json({ error: 'videoFps must be between 0.1 and 16' }, { status: 400 });
    }
    const resolvedVideoMaxFrames = typeof videoMaxFrames === 'number' && Number.isInteger(videoMaxFrames)
      ? videoMaxFrames
      : 8;
    if (resolvedVideoMaxFrames < 1 || resolvedVideoMaxFrames > 64) {
      return NextResponse.json({ error: 'videoMaxFrames must be between 1 and 64' }, { status: 400 });
    }

    if (!fs.existsSync(SCRIPT_PATH)) {
      return NextResponse.json({ error: 'Captioning script not found' }, { status: 500 });
    }

    const existing = captionStates.get(imgPath as string);
    if (existing && existing.status === 'running') {
      return NextResponse.json({ error: 'Captioning already in progress for this image' }, { status: 409 });
    }

    const args = [
      SCRIPT_PATH,
      '--img_path', imgPath as string,
      '--trigger_word', (triggerWord || '').toString(),
      '--system_prompt', (systemPrompt || '').toString(),
      '--model_id', resolvedModelId,
      '--video_fps', resolvedVideoFps.toString(),
      '--video_max_frames', resolvedVideoMaxFrames.toString(),
      ...(useQuorum ? ['--quorum'] : []),
    ];

    const state: CaptionState = { status: 'running' };
    captionStates.set(imgPath as string, state);

    const proc = spawn(getPythonPath(), args, { stdio: ['ignore', 'pipe', 'pipe'] });
    state.process = proc;

    let stdoutBuffer = '';
    proc.stdout.on('data', (data: Buffer) => {
      stdoutBuffer += data.toString();
      const lines = stdoutBuffer.split('\n');
      stdoutBuffer = lines.pop() ?? '';
      for (const raw of lines) {
        const line = raw.trim();
        if (!line) continue;
        if (line === 'STATUS:downloading') {
          state.downloading = true;
        } else if (line.startsWith('{')) {
          // Final JSON result line from the script.
          try {
            const parsed = JSON.parse(line) as { caption?: string; error?: string };
            if (parsed.error) {
              state.error = parsed.error;
            } else if (parsed.caption !== undefined) {
              state.caption = parsed.caption;
              state.downloading = false;
            }
          } catch {
            // Not valid JSON; ignore.
          }
        }
      }
    });

    let stderrBuffer = '';
    proc.stderr.on('data', (data: Buffer) => {
      stderrBuffer += data.toString();
      // Keep stderr bounded — we only need the tail for error reporting.
      if (stderrBuffer.length > 8192) {
        stderrBuffer = stderrBuffer.slice(-8192);
      }
    });

    proc.on('close', (code: number | null) => {
      const current = captionStates.get(imgPath as string);
      if (!current || current.status !== 'running') return;
      // Drain any remaining buffered line.
      const tail = stdoutBuffer.trim();
      if (tail.startsWith('{')) {
        try {
          const parsed = JSON.parse(tail) as { caption?: string; error?: string };
          if (parsed.caption !== undefined && !current.caption) current.caption = parsed.caption;
          if (parsed.error && !current.error) current.error = parsed.error;
        } catch {
          // ignore
        }
      }
      if (code === 0 && current.caption !== undefined) {
        current.status = 'completed';
      } else {
        current.status = 'error';
        if (!current.error) {
          const stderrTail = stderrBuffer.trim().split('\n').slice(-3).join('\n');
          current.error = stderrTail || `Process exited with code ${code}`;
        }
      }
      current.downloading = false;
      current.process = undefined;
    });

    proc.on('error', (err: Error) => {
      const current = captionStates.get(imgPath as string);
      if (current) {
        current.status = 'error';
        current.error = err.message;
        current.process = undefined;
      }
    });

    return NextResponse.json({ status: 'running' });
  } catch (error: any) {
    console.error('Error starting AI captioning:', error);
    return NextResponse.json({ error: 'Failed to start captioning' }, { status: 500 });
  }
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const imgPath = searchParams.get('imgPath');

  if (!imgPath) {
    return NextResponse.json({ error: 'imgPath is required' }, { status: 400 });
  }

  const state = captionStates.get(imgPath);
  if (!state || state.status !== 'running') {
    return NextResponse.json({ error: 'No active captioning for this image' }, { status: 404 });
  }

  if (state.process) {
    state.process.kill('SIGTERM');
    state.process = undefined;
  }
  state.status = 'cancelled';

  return NextResponse.json({ status: 'cancelled' });
}
