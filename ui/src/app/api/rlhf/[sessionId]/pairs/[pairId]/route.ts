import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: Promise<{ sessionId: string; pairId: string }> }) {
  const { pairId } = await params;
  try {
    const pair = await prisma.rlhfPair.findUnique({ where: { id: pairId } });
    if (!pair) return NextResponse.json({ error: 'Not found' }, { status: 404 });
    return NextResponse.json(pair);
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch pair' }, { status: 500 });
  }
}

export async function PUT(request: NextRequest, { params }: { params: Promise<{ sessionId: string; pairId: string }> }) {
  const { pairId } = await params;
  try {
    const body = await request.json();
    const { preference, gen_status } = body;

    const validPreferences = ['none', 'a', 'b', 'tie', 'skip'];
    const validGenStatuses = ['pending', 'queued', 'completed', 'error'];

    const data: any = {};
    if (preference !== undefined) {
      if (!validPreferences.includes(preference)) {
        return NextResponse.json({ error: 'Invalid preference value' }, { status: 400 });
      }
      data.preference = preference;
    }
    if (gen_status !== undefined) {
      if (!validGenStatuses.includes(gen_status)) {
        return NextResponse.json({ error: 'Invalid gen_status value' }, { status: 400 });
      }
      data.gen_status = gen_status;
    }

    const pair = await prisma.rlhfPair.update({ where: { id: pairId }, data });
    return NextResponse.json(pair);
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to update pair' }, { status: 500 });
  }
}
