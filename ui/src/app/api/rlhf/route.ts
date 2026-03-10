import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET() {
  try {
    const sessions = await prisma.rlhfSession.findMany({
      orderBy: { created_at: 'desc' },
      include: {
        _count: { select: { pairs: true } },
      },
    });
    return NextResponse.json({ sessions });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch sessions' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { name, model_path, comfyui_url, workflow_json, gpu_ids } = body;

    if (!name?.trim()) {
      return NextResponse.json({ error: 'Name is required' }, { status: 400 });
    }
    if (!model_path?.trim()) {
      return NextResponse.json({ error: 'Model path is required' }, { status: 400 });
    }

    const session = await prisma.rlhfSession.create({
      data: {
        name: name.trim(),
        model_path: model_path.trim(),
        comfyui_url: comfyui_url?.trim() || 'http://127.0.0.1:9188',
        workflow_json: workflow_json || '',
        gpu_ids: gpu_ids || '0',
      },
    });
    return NextResponse.json(session);
  } catch (error: any) {
    if (error.code === 'P2002') {
      return NextResponse.json({ error: 'Session name already exists' }, { status: 409 });
    }
    console.error(error);
    return NextResponse.json({ error: 'Failed to create session' }, { status: 500 });
  }
}
