import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { TOOLKIT_ROOT } from '@/paths';

const prisma = new PrismaClient();
const DATA_ROOT = path.join(TOOLKIT_ROOT, 'data');

export async function POST(request: NextRequest, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  try {
    const session = await prisma.rlhfSession.findUnique({ where: { id: sessionId } });
    if (!session) return NextResponse.json({ error: 'Session not found' }, { status: 404 });

    const outputDir = path.join(DATA_ROOT, 'rlhf', session.name);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    await prisma.rlhfSession.update({
      where: { id: sessionId },
      data: { status: 'generating', output_dir: outputDir },
    });

    // Reset error pairs to pending so they get retried
    await prisma.rlhfPair.updateMany({
      where: { session_id: sessionId, gen_status: 'error' },
      data: { gen_status: 'pending' },
    });

    return NextResponse.json({ success: true, output_dir: outputDir });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to start generation' }, { status: 500 });
  }
}

export async function GET(request: NextRequest, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  try {
    const session = await prisma.rlhfSession.findUnique({ where: { id: sessionId } });
    if (!session) return NextResponse.json({ error: 'Session not found' }, { status: 404 });

    const [total, pending, queued, completed, error] = await Promise.all([
      prisma.rlhfPair.count({ where: { session_id: sessionId } }),
      prisma.rlhfPair.count({ where: { session_id: sessionId, gen_status: 'pending' } }),
      prisma.rlhfPair.count({ where: { session_id: sessionId, gen_status: 'queued' } }),
      prisma.rlhfPair.count({ where: { session_id: sessionId, gen_status: 'completed' } }),
      prisma.rlhfPair.count({ where: { session_id: sessionId, gen_status: 'error' } }),
    ]);

    return NextResponse.json({
      session_status: session.status,
      total,
      pending,
      queued,
      completed,
      error,
    });
  } catch (err) {
    console.error(err);
    return NextResponse.json({ error: 'Failed to get generation status' }, { status: 500 });
  }
}
