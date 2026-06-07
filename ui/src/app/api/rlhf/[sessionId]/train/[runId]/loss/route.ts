import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import fs from 'fs';
import path from 'path';

const prisma = new PrismaClient();

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ sessionId: string; runId: string }> }
) {
  const { runId } = await params;
  try {
    const run = await prisma.rlhfTrainingRun.findUnique({ where: { id: runId } });
    if (!run) return NextResponse.json({ error: 'Not found' }, { status: 404 });
    if (!run.output_path) return NextResponse.json({ points: [] });

    const lossLogPath = path.join(run.output_path, 'loss_log.jsonl');
    if (!fs.existsSync(lossLogPath)) {
      return NextResponse.json({ points: [] });
    }

    const sinceStep = parseInt(request.nextUrl.searchParams.get('since_step') ?? '0', 10);

    const content = fs.readFileSync(lossLogPath, 'utf-8');
    const points: { step: number; value: number }[] = [];

    for (const line of content.split('\n')) {
      if (!line.trim()) continue;
      try {
        const entry = JSON.parse(line);
        if (entry.step > sinceStep) {
          points.push({ step: entry.step, value: entry.loss });
        }
      } catch {
        // skip malformed lines
      }
    }

    return NextResponse.json({ points });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to read loss log' }, { status: 500 });
  }
}
