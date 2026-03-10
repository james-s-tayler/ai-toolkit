import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { randomInt } from 'crypto';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  const { searchParams } = new URL(request.url);
  const gen_status = searchParams.get('gen_status');
  const preference = searchParams.get('preference');
  const page = parseInt(searchParams.get('page') || '1');
  const perPage = parseInt(searchParams.get('perPage') || '50');

  try {
    const where: any = { session_id: sessionId };
    if (gen_status) where.gen_status = gen_status;
    if (preference) where.preference = preference;

    const [pairs, total] = await Promise.all([
      prisma.rlhfPair.findMany({
        where,
        orderBy: { created_at: 'asc' },
        skip: (page - 1) * perPage,
        take: perPage,
      }),
      prisma.rlhfPair.count({ where }),
    ]);

    return NextResponse.json({ pairs, total, page, perPage });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch pairs' }, { status: 500 });
  }
}

export async function POST(request: NextRequest, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  try {
    const body = await request.json();
    const { prompts } = body; // array of prompt strings

    if (!Array.isArray(prompts) || prompts.length === 0) {
      return NextResponse.json({ error: 'prompts array is required' }, { status: 400 });
    }

    const pairs = await Promise.all(
      prompts.map((prompt: string) =>
        prisma.rlhfPair.create({
          data: {
            session_id: sessionId,
            prompt,
            seed_a: randomInt(2147483647),
            seed_b: randomInt(2147483647),
          },
        })
      )
    );

    return NextResponse.json({ pairs, count: pairs.length });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to create pairs' }, { status: 500 });
  }
}
