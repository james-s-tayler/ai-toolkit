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

    if (!run.log_path || !fs.existsSync(run.log_path)) {
      return NextResponse.json({ log: '' });
    }

    let log = '';
    try {
      log = fs.readFileSync(run.log_path, 'utf-8');
    } catch (error) {
      console.error('Error reading log file:', error);
      log = 'Error reading log file';
    }

    return NextResponse.json({ log });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch training log' }, { status: 500 });
  }
}
