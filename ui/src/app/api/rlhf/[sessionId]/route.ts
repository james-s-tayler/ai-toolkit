import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  try {
    const session = await prisma.rlhfSession.findUnique({
      where: { id: sessionId },
      include: {
        _count: { select: { pairs: true, training_runs: true } },
      },
    });
    if (!session) return NextResponse.json({ error: 'Not found' }, { status: 404 });
    return NextResponse.json(session);
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch session' }, { status: 500 });
  }
}

export async function PUT(request: NextRequest, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  try {
    const body = await request.json();
    const { name, model_path, comfyui_url, workflow_json, gpu_ids, status, output_dir, config_json } = body;
    const session = await prisma.rlhfSession.update({
      where: { id: sessionId },
      data: {
        ...(name !== undefined && { name }),
        ...(model_path !== undefined && { model_path }),
        ...(comfyui_url !== undefined && { comfyui_url }),
        ...(workflow_json !== undefined && { workflow_json }),
        ...(gpu_ids !== undefined && { gpu_ids }),
        ...(status !== undefined && { status }),
        ...(output_dir !== undefined && { output_dir }),
        ...(config_json !== undefined && { config_json }),
      },
    });
    return NextResponse.json(session);
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to update session' }, { status: 500 });
  }
}

export async function DELETE(request: NextRequest, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  try {
    await prisma.rlhfSession.delete({ where: { id: sessionId } });
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to delete session' }, { status: 500 });
  }
}
