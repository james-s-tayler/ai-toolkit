import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import fs from 'fs';
import path from 'path';

const prisma = new PrismaClient();

/**
 * Kill a detached training process and its entire process group.
 * The process is spawned with `detached: true`, making it a process group
 * leader. Using negative PID targets the whole group.
 */
function killTrainingProcess(pid: number | null | undefined): boolean {
  if (!pid) return false;
  try {
    // Kill the entire process group (negative PID)
    process.kill(-pid, 'SIGTERM');
    return true;
  } catch {
    // Process group kill failed — try killing just the PID directly
    try {
      process.kill(pid, 'SIGTERM');
      return true;
    } catch {
      // Process already gone
      return false;
    }
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ sessionId: string; runId: string }> }
) {
  const { runId } = await params;
  try {
    const run = await prisma.rlhfTrainingRun.findUnique({ where: { id: runId } });
    if (!run) return NextResponse.json({ error: 'Not found' }, { status: 404 });

    // Try to read status.json written by the training script
    // Don't overwrite status if it was manually set to stopped/paused/completed by the UI
    const dbStatusIsManual = ['stopped', 'paused', 'completed'].includes(run.status);
    const statusFilePath = run.output_path ? `${run.output_path}/status.json` : null;
    if (statusFilePath && fs.existsSync(statusFilePath)) {
      try {
        const statusData = JSON.parse(fs.readFileSync(statusFilePath, 'utf-8'));
        const { step, total_steps, loss, status, speed_string } = statusData;
        // Only persist real run statuses to DB; intermediate phases (loading_model, etc.) are informational only
        const validDbStatuses = ['running', 'paused', 'stopped', 'completed', 'error'];
        const shouldUpdateStatus = !dbStatusIsManual && status && validDbStatuses.includes(status);
        const updated = await prisma.rlhfTrainingRun.update({
          where: { id: runId },
          data: {
            step: step ?? run.step,
            total_steps: total_steps ?? run.total_steps,
            loss: loss ?? run.loss,
            ...(shouldUpdateStatus ? { status } : {}),
          },
        });
        return NextResponse.json({ ...updated, speed_string: speed_string ?? '' });
      } catch (e) {
        // fall through to return db record
      }
    }

    return NextResponse.json(run);
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch training run' }, { status: 500 });
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ sessionId: string; runId: string }> }
) {
  const { sessionId, runId } = await params;
  try {
    const body = await request.json();
    const { action } = body;

    if (!action || !['pause', 'resume', 'stop', 'mark_stopped'].includes(action)) {
      return NextResponse.json({ error: 'Invalid action. Must be pause, resume, stop, or mark_stopped.' }, { status: 400 });
    }
    const run = await prisma.rlhfTrainingRun.findUnique({ where: { id: runId } });
    if (!run) return NextResponse.json({ error: 'Not found' }, { status: 404 });

    // mark_stopped: kill the process and force the DB status
    if (action === 'mark_stopped') {
      killTrainingProcess(run.pid);
      await prisma.rlhfTrainingRun.update({
        where: { id: runId },
        data: { status: 'stopped' },
      });
      // Also update status.json so it won't revert on next read
      if (run.output_path) {
        const sf = path.join(run.output_path, 'status.json');
        if (fs.existsSync(sf)) {
          try {
            const sd = JSON.parse(fs.readFileSync(sf, 'utf-8'));
            sd.status = 'stopped';
            fs.writeFileSync(sf, JSON.stringify(sd));
          } catch (e) { /* non-critical */ }
        }
      }
      // Also reset session status if it's stuck on 'training'
      const session = await prisma.rlhfSession.findUnique({ where: { id: sessionId } });
      if (session && session.status === 'training') {
        await prisma.rlhfSession.update({
          where: { id: sessionId },
          data: { status: 'generated' },
        });
      }
      return NextResponse.json({ success: true, action });
    }

    if (!run.output_path) return NextResponse.json({ error: 'No output path for this run' }, { status: 400 });

    // Write control.json to the run's output directory
    const controlPath = path.join(run.output_path, 'control.json');
    fs.writeFileSync(controlPath, JSON.stringify({ action }, null, 2));

    // Also update status.json so it agrees with the DB and won't revert on next read
    const statusFilePath = path.join(run.output_path, 'status.json');
    const newDbStatus = action === 'pause' ? 'paused' : action === 'stop' ? 'stopped' : action === 'resume' ? 'running' : null;
    if (newDbStatus && fs.existsSync(statusFilePath)) {
      try {
        const statusData = JSON.parse(fs.readFileSync(statusFilePath, 'utf-8'));
        statusData.status = newDbStatus;
        fs.writeFileSync(statusFilePath, JSON.stringify(statusData));
      } catch (e) {
        // non-critical
      }
    }

    if (action === 'pause') {
      killTrainingProcess(run.pid);
      await prisma.rlhfTrainingRun.update({
        where: { id: runId },
        data: { status: 'paused' },
      });
    }

    if (action === 'stop') {
      killTrainingProcess(run.pid);
      await prisma.rlhfTrainingRun.update({
        where: { id: runId },
        data: { status: 'stopped' },
      });
    }

    if (action === 'resume') {
      await prisma.rlhfTrainingRun.update({
        where: { id: runId },
        data: { status: 'running' },
      });
    }

    return NextResponse.json({ success: true, action });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to update training run' }, { status: 500 });
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ sessionId: string; runId: string }> }
) {
  const { sessionId, runId } = await params;
  try {
    const run = await prisma.rlhfTrainingRun.findUnique({ where: { id: runId } });
    if (!run) return NextResponse.json({ error: 'Not found' }, { status: 404 });

    // Kill the process if it has a PID (regardless of DB status — it may be stale)
    killTrainingProcess(run.pid);

    // Clean up output directory (checkpoints, safetensors, logs, etc.)
    if (run.output_path) {
      try {
        fs.rmSync(run.output_path, { recursive: true, force: true });
      } catch (e) {
        console.error('Error cleaning up output directory:', e);
      }
    }

    await prisma.rlhfTrainingRun.delete({ where: { id: runId } });

    // Reset session status if it was 'training'
    const session = await prisma.rlhfSession.findUnique({ where: { id: sessionId } });
    if (session && session.status === 'training') {
      await prisma.rlhfSession.update({
        where: { id: sessionId },
        data: { status: 'generated' },
      });
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to delete training run' }, { status: 500 });
  }
}
