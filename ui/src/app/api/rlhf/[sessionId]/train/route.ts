import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';
import { TOOLKIT_ROOT } from '@/paths';

const prisma = new PrismaClient();
const isWindows = process.platform === 'win32';

function getPythonPath(): string {
  if (fs.existsSync(path.join(TOOLKIT_ROOT, '.venv'))) {
    return isWindows
      ? path.join(TOOLKIT_ROOT, '.venv', 'Scripts', 'python.exe')
      : path.join(TOOLKIT_ROOT, '.venv', 'bin', 'python');
  }
  if (fs.existsSync(path.join(TOOLKIT_ROOT, 'venv'))) {
    return isWindows
      ? path.join(TOOLKIT_ROOT, 'venv', 'Scripts', 'python.exe')
      : path.join(TOOLKIT_ROOT, 'venv', 'bin', 'python');
  }
  return 'python';
}

export async function GET(request: NextRequest, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  try {
    const runs = await prisma.rlhfTrainingRun.findMany({
      where: { session_id: sessionId },
      orderBy: { created_at: 'desc' },
    });
    return NextResponse.json({ runs });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch training runs' }, { status: 500 });
  }
}

export async function POST(request: NextRequest, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  try {
    const body = await request.json();
    const {
      beta = 5000,
      learning_rate = 1e-5,
      max_train_steps = 2000,
      lora_rank = 16,
      batch_size = 1,
      blocks_to_swap = 0,
      mixed_precision = 'bf16',
      gradient_checkpointing = true,
    } = body;

    const session = await prisma.rlhfSession.findUnique({ where: { id: sessionId } });
    if (!session) return NextResponse.json({ error: 'Session not found' }, { status: 404 });

    // Get all winning/losing pairs
    const pairs = await prisma.rlhfPair.findMany({
      where: {
        session_id: sessionId,
        gen_status: 'completed',
        preference: { in: ['a', 'b'] },
      },
    });

    if (pairs.length === 0) {
      return NextResponse.json({ error: 'No evaluated pairs available for training' }, { status: 400 });
    }

    const preferences = pairs.map(p => ({
      prompt: p.prompt,
      winner_path: p.preference === 'a' ? p.image_a_path : p.image_b_path,
      loser_path: p.preference === 'a' ? p.image_b_path : p.image_a_path,
    }));

    const outputDir = path.join(
      session.output_dir || path.join(TOOLKIT_ROOT, 'data', 'rlhf', session.name),
      'training'
    );
    if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

    const prefDataPath = path.join(outputDir, 'preferences.json');
    fs.writeFileSync(prefDataPath, JSON.stringify(preferences, null, 2));

    const statusFilePath = path.join(outputDir, 'status.json');
    const logPath = path.join(outputDir, 'train.log');

    const config = {
      beta,
      learning_rate,
      max_train_steps,
      lora_rank,
      batch_size,
      blocks_to_swap,
      mixed_precision,
      gradient_checkpointing,
    };

    const run = await prisma.rlhfTrainingRun.create({
      data: {
        session_id: sessionId,
        status: 'running',
        config_json: JSON.stringify(config),
        output_path: outputDir,
        total_steps: max_train_steps,
        log_path: logPath,
      },
    });

    const pythonPath = getPythonPath();
    const scriptPath = path.join(TOOLKIT_ROOT, 'scripts', 'rlhf_dpo_train.py');

    const args = [
      scriptPath,
      '--preference_data', prefDataPath,
      '--model_path', session.model_path,
      '--output_dir', outputDir,
      '--status_file', statusFilePath,
      '--run_id', run.id,
      '--beta', String(beta),
      '--learning_rate', String(learning_rate),
      '--max_train_steps', String(max_train_steps),
      '--lora_rank', String(lora_rank),
      '--batch_size', String(batch_size),
      '--blocks_to_swap', String(blocks_to_swap),
      '--mixed_precision', mixed_precision,
    ];

    if (gradient_checkpointing) args.push('--gradient_checkpointing');

    const additionalEnv: Record<string, string> = {
      CUDA_DEVICE_ORDER: 'PCI_BUS_ID',
      CUDA_VISIBLE_DEVICES: session.gpu_ids,
    };

    const logFd = fs.openSync(logPath, 'a');
    const subprocess = spawn(pythonPath, args, {
      detached: true,
      stdio: ['ignore', logFd, logFd],
      env: { ...process.env, ...additionalEnv },
      cwd: TOOLKIT_ROOT,
    });

    subprocess.on('close', () => { fs.closeSync(logFd); });
    if (subprocess.unref) subprocess.unref();

    const pid = subprocess.pid;
    await prisma.rlhfTrainingRun.update({ where: { id: run.id }, data: { pid: pid ?? null } });
    await prisma.rlhfSession.update({ where: { id: sessionId }, data: { status: 'training' } });

    return NextResponse.json(run);
  } catch (error: any) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to start training', details: error?.message || String(error) }, { status: 500 });
  }
}
