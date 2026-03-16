import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';

const prisma = new PrismaClient();

// Matches "checkpoint-123" directories
const CHECKPOINT_DIR_RE = /^checkpoint-(\d+)$/;

export async function GET(request: NextRequest, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;

  try {
    const session = await prisma.rlhfSession.findUnique({ where: { id: sessionId } });
    if (!session) return NextResponse.json({ files: [] });

    const run = await prisma.rlhfTrainingRun.findFirst({
      where: { session_id: sessionId },
      orderBy: { created_at: 'desc' },
    });

    if (!run || !run.output_path) {
      return NextResponse.json({ files: [] });
    }

    const outputDir = run.output_path;

    if (!fs.existsSync(outputDir)) {
      return NextResponse.json({ files: [] });
    }

    const sessionName = session.name;

    // Only look in checkpoint-{step}/ and final/ subdirectories for .safetensors files
    const fileObjects: { path: string; size: number; name: string }[] = [];

    for (const entry of fs.readdirSync(outputDir, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;

      const dirName = entry.name;
      const ckptMatch = CHECKPOINT_DIR_RE.exec(dirName);
      const isFinal = dirName === 'final';

      if (!ckptMatch && !isFinal) continue;

      const subdir = path.join(outputDir, dirName);
      for (const file of fs.readdirSync(subdir)) {
        if (!file.endsWith('.safetensors')) continue;

        const fullPath = path.join(subdir, file);
        const stats = fs.statSync(fullPath);
        const displayName = isFinal
          ? `${sessionName}_final`
          : `${sessionName}_step-${ckptMatch![1]}`;

        fileObjects.push({
          path: fullPath,
          size: stats.size,
          name: displayName,
        });
      }
    }

    // Sort by step number (final last)
    fileObjects.sort((a, b) => {
      const stepA = a.name.includes('_final') ? Infinity : parseInt(a.name.split('step-')[1] ?? '0');
      const stepB = b.name.includes('_final') ? Infinity : parseInt(b.name.split('step-')[1] ?? '0');
      return stepA - stepB;
    });

    return NextResponse.json({ files: fileObjects });
  } catch (error) {
    console.error('Error listing RLHF checkpoint files:', error);
    return NextResponse.json({ error: 'Failed to list files' }, { status: 500 });
  }
}
