import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;

  try {
    // Find the latest training run for this session
    const run = await prisma.rlhfTrainingRun.findFirst({
      where: { session_id: sessionId },
      orderBy: { created_at: 'desc' },
    });

    if (!run || !run.output_path) {
      return NextResponse.json({ samples: [], prompts: [] });
    }

    const samplesFolder = path.join(run.output_path, 'samples');
    if (!fs.existsSync(samplesFolder)) {
      return NextResponse.json({ samples: [], prompts: [] });
    }

    const samples = fs
      .readdirSync(samplesFolder)
      .filter(file => {
        return file.endsWith('.png') || file.endsWith('.jpg') || file.endsWith('.jpeg') || file.endsWith('.webp');
      })
      .map(file => path.join(samplesFolder, file))
      .sort();

    // Load sample prompts from the training config or prompts file
    let prompts: string[] = [];
    const promptsFile = path.join(run.output_path, 'sample_prompts.json');
    if (fs.existsSync(promptsFile)) {
      try {
        prompts = JSON.parse(fs.readFileSync(promptsFile, 'utf-8'));
      } catch {
        // ignore parse errors
      }
    }

    return NextResponse.json({ samples, prompts });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch samples' }, { status: 500 });
  }
}
