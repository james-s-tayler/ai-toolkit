import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import fs from 'fs';

const prisma = new PrismaClient();

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ sessionId: string; runId: string }> }
) {
  const { runId } = await params;
  try {
    const run = await prisma.rlhfTrainingRun.findUnique({ where: { id: runId } });
    if (!run) return NextResponse.json({ error: 'Not found' }, { status: 404 });

    // Try to read status.json written by the training script
    const statusFilePath = run.output_path ? `${run.output_path}/status.json` : null;
    if (statusFilePath && fs.existsSync(statusFilePath)) {
      try {
        const statusData = JSON.parse(fs.readFileSync(statusFilePath, 'utf-8'));
        const { step, total_steps, loss, status } = statusData;
        const updated = await prisma.rlhfTrainingRun.update({
          where: { id: runId },
          data: {
            step: step ?? run.step,
            total_steps: total_steps ?? run.total_steps,
            loss: loss ?? run.loss,
            status: status ?? run.status,
          },
        });
        return NextResponse.json(updated);
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
