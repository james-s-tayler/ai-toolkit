import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';
import { TOOLKIT_ROOT } from '@/paths';

const SCRIPT_PATH = path.join(TOOLKIT_ROOT, 'scripts', 'extract_faces.py');

const ALLOWED_TARGETS = new Set([512, 768, 1024, 1280]);

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

interface FaceResult {
  path: string;
  size: number;
}

function runScript(args: string[]): Promise<{ faces: FaceResult[] }> {
  return new Promise((resolve, reject) => {
    const proc = spawn(getPythonPath(), [SCRIPT_PATH, ...args], { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdoutBuf = '';
    let stderrBuf = '';
    proc.stdout.on('data', (d: Buffer) => {
      stdoutBuf += d.toString();
    });
    proc.stderr.on('data', (d: Buffer) => {
      stderrBuf += d.toString();
      if (stderrBuf.length > 8192) stderrBuf = stderrBuf.slice(-8192);
    });
    proc.on('error', err => reject(err));
    proc.on('close', code => {
      if (code !== 0) {
        const trimmed = stderrBuf.trim();
        const lastJsonLine = trimmed.split('\n').reverse().find(l => l.trim().startsWith('{'));
        if (lastJsonLine) {
          try {
            const parsed = JSON.parse(lastJsonLine) as { error?: string };
            if (parsed.error) return reject(new Error(parsed.error));
          } catch {
            // fall through
          }
        }
        return reject(new Error(trimmed || `Process exited with code ${code}`));
      }
      const lastLine = stdoutBuf.trim().split('\n').filter(l => l.trim().startsWith('{')).pop() ?? '';
      try {
        const parsed = JSON.parse(lastLine) as { faces?: FaceResult[]; error?: string };
        if (parsed.error) return reject(new Error(parsed.error));
        resolve({ faces: parsed.faces ?? [] });
      } catch (e) {
        reject(new Error(`Failed to parse script output: ${(e as Error).message}`));
      }
    });
  });
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath, destinationDataset, padding, targets, threshold } = body;

    if (!imgPath || typeof imgPath !== 'string') {
      return NextResponse.json({ error: 'imgPath is required' }, { status: 400 });
    }
    if (imgPath.includes('..')) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    const datasetsPath = await getDatasetsRoot();
    const trainingPath = await getTrainingFolder();
    if (!imgPath.startsWith(datasetsPath) && !imgPath.startsWith(trainingPath)) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }
    if (!/\.(png|jpe?g|webp|bmp)$/i.test(imgPath)) {
      return NextResponse.json({ error: 'Not a supported image file' }, { status: 400 });
    }
    if (!fs.existsSync(imgPath)) {
      return NextResponse.json({ error: 'Image not found' }, { status: 404 });
    }

    if (!destinationDataset || typeof destinationDataset !== 'string') {
      return NextResponse.json({ error: 'Destination dataset is required' }, { status: 400 });
    }
    const cleanName = destinationDataset.toLowerCase().replace(/[^a-z0-9]+/g, '_');
    if (!cleanName) {
      return NextResponse.json({ error: 'Invalid destination dataset name' }, { status: 400 });
    }

    const paddingNum = typeof padding === 'number' && Number.isFinite(padding) ? padding : 1.5;
    if (paddingNum < 1.0 || paddingNum > 4.0) {
      return NextResponse.json({ error: 'padding must be between 1.0 and 4.0' }, { status: 400 });
    }

    const thresholdNum = typeof threshold === 'number' && Number.isFinite(threshold) ? threshold : 0.5;
    if (thresholdNum < 0 || thresholdNum > 1) {
      return NextResponse.json({ error: 'threshold must be between 0.0 and 1.0' }, { status: 400 });
    }

    let resolvedTargets: number[] = [512, 768, 1024, 1280];
    if (Array.isArray(targets) && targets.length > 0) {
      const filtered = targets.map((t: unknown) => Number(t)).filter((t: number) => ALLOWED_TARGETS.has(t));
      if (filtered.length === 0) {
        return NextResponse.json({ error: 'targets must include at least one of 512, 768, 1024, 1280' }, { status: 400 });
      }
      resolvedTargets = filtered;
    }

    if (!fs.existsSync(SCRIPT_PATH)) {
      return NextResponse.json({ error: 'Face extraction script not found' }, { status: 500 });
    }

    const destinationDir = path.join(datasetsPath, cleanName);
    if (!fs.existsSync(destinationDir)) {
      fs.mkdirSync(destinationDir, { recursive: true });
    }

    const args = [
      '--img_path', imgPath,
      '--output_dir', destinationDir,
      '--padding', paddingNum.toString(),
      '--threshold', thresholdNum.toString(),
      '--targets', resolvedTargets.join(','),
    ];

    const result = await runScript(args);
    return NextResponse.json({
      success: true,
      destinationDataset: cleanName,
      faceCount: result.faces.length,
      faces: result.faces,
    });
  } catch (error: any) {
    console.error('Error extracting faces:', error);
    if (error.code === 'ENOENT') {
      return NextResponse.json({ error: 'Python is not installed or not found' }, { status: 500 });
    }
    return NextResponse.json({ error: error.message || 'Failed to extract faces' }, { status: 500 });
  }
}
