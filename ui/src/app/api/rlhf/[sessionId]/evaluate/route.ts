import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  const { searchParams } = new URL(request.url);
  const afterId = searchParams.get('after'); // for prefetching next N

  try {
    const where: any = {
      session_id: sessionId,
      gen_status: 'completed',
      preference: 'none',
    };

    if (afterId) {
      // Return up to 5 unevaluated pairs after the specified id for prefetching
      const all = await prisma.rlhfPair.findMany({
        where: { session_id: sessionId, gen_status: 'completed', preference: 'none' },
        orderBy: { created_at: 'asc' },
      });
      const idx = all.findIndex(p => p.id === afterId);
      const next = idx >= 0 ? all.slice(idx + 1, idx + 6) : all.slice(0, 5);
      return NextResponse.json({ pairs: next });
    }

    const nextPair = await prisma.rlhfPair.findFirst({
      where,
      orderBy: { created_at: 'asc' },
    });

    const [total, evaluated, skipped] = await Promise.all([
      prisma.rlhfPair.count({ where: { session_id: sessionId, gen_status: 'completed' } }),
      prisma.rlhfPair.count({
        where: { session_id: sessionId, gen_status: 'completed', preference: { not: 'none' } },
      }),
      prisma.rlhfPair.count({ where: { session_id: sessionId, preference: 'skip' } }),
    ]);

    return NextResponse.json({
      pair: nextPair,
      stats: { total, evaluated, skipped, remaining: total - evaluated },
    });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch evaluation data' }, { status: 500 });
  }
}
