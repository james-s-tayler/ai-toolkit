import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { getDatasetsRoot } from '@/server/settings';

const prisma = new PrismaClient();

const IMAGE_EXTS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']);

function getImageMap(dirPath: string): Map<string, string> {
  const map = new Map<string, string>();
  if (!fs.existsSync(dirPath)) return map;
  for (const entry of fs.readdirSync(dirPath, { withFileTypes: true })) {
    if (!entry.isFile()) continue;
    const ext = path.extname(entry.name).toLowerCase();
    if (!IMAGE_EXTS.has(ext)) continue;
    const stem = path.basename(entry.name, ext);
    map.set(stem, path.join(dirPath, entry.name));
  }
  return map;
}

function readCaption(imagePath: string): string {
  const dir = path.dirname(imagePath);
  const stem = path.basename(imagePath, path.extname(imagePath));
  const txtPath = path.join(dir, stem + '.txt');
  if (fs.existsSync(txtPath)) {
    return fs.readFileSync(txtPath, 'utf-8').trim();
  }
  return '';
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ sessionId: string }> }
) {
  const { sessionId } = await params;
  try {
    const body = await request.json();
    const { accepted_dataset, rejected_dataset } = body;

    if (!accepted_dataset || !rejected_dataset) {
      return NextResponse.json(
        { error: 'Both accepted_dataset and rejected_dataset are required' },
        { status: 400 }
      );
    }
    if (accepted_dataset === rejected_dataset) {
      return NextResponse.json(
        { error: 'Accepted and rejected datasets must be different' },
        { status: 400 }
      );
    }

    const session = await prisma.rlhfSession.findUnique({ where: { id: sessionId } });
    if (!session) {
      return NextResponse.json({ error: 'Session not found' }, { status: 404 });
    }

    const datasetsRoot = await getDatasetsRoot();
    const acceptedDir = path.join(datasetsRoot, accepted_dataset);
    const rejectedDir = path.join(datasetsRoot, rejected_dataset);

    if (!fs.existsSync(acceptedDir)) {
      return NextResponse.json(
        { error: `Accepted dataset folder not found: ${accepted_dataset}` },
        { status: 400 }
      );
    }
    if (!fs.existsSync(rejectedDir)) {
      return NextResponse.json(
        { error: `Rejected dataset folder not found: ${rejected_dataset}` },
        { status: 400 }
      );
    }

    const acceptedMap = getImageMap(acceptedDir);
    const rejectedMap = getImageMap(rejectedDir);

    // Match by stem
    const matched: { stem: string; accepted: string; rejected: string }[] = [];
    const unmatchedAccepted: string[] = [];
    const unmatchedRejected: string[] = [];
    const captionWarnings: string[] = [];

    for (const [stem, acceptedPath] of acceptedMap) {
      const rejectedPath = rejectedMap.get(stem);
      if (rejectedPath) {
        matched.push({ stem, accepted: acceptedPath, rejected: rejectedPath });
      } else {
        unmatchedAccepted.push(stem);
      }
    }
    for (const stem of rejectedMap.keys()) {
      if (!acceptedMap.has(stem)) {
        unmatchedRejected.push(stem);
      }
    }

    if (matched.length === 0) {
      return NextResponse.json(
        {
          error: 'No matching filenames found between datasets',
          unmatched_accepted: unmatchedAccepted.length,
          unmatched_rejected: unmatchedRejected.length,
        },
        { status: 400 }
      );
    }

    // Create pairs in batch
    const pairData = matched.map(({ stem, accepted, rejected }) => {
      const acceptedCaption = readCaption(accepted);
      const rejectedCaption = readCaption(rejected);
      if (acceptedCaption && rejectedCaption && acceptedCaption !== rejectedCaption) {
        captionWarnings.push(
          `${stem}: captions differ, using accepted caption`
        );
      }
      const prompt = acceptedCaption || rejectedCaption || stem;
      return {
        session_id: sessionId,
        prompt,
        seed_a: 0,
        seed_b: 0,
        image_a_path: accepted,
        image_b_path: rejected,
        gen_status: 'completed',
        preference: 'a',
      };
    });

    await prisma.rlhfPair.createMany({ data: pairData });

    // Update session
    const configJson = JSON.parse(session.config_json || '{}');
    configJson.accepted_dataset = accepted_dataset;
    configJson.rejected_dataset = rejected_dataset;
    configJson.import_stats = {
      matched: matched.length,
      unmatched_accepted: unmatchedAccepted.length,
      unmatched_rejected: unmatchedRejected.length,
    };

    await prisma.rlhfSession.update({
      where: { id: sessionId },
      data: {
        status: 'imported',
        config_json: JSON.stringify(configJson),
      },
    });

    return NextResponse.json({
      matched: matched.length,
      unmatched_accepted: unmatchedAccepted.length,
      unmatched_rejected: unmatchedRejected.length,
      caption_warnings: captionWarnings,
    });
  } catch (error: any) {
    console.error(error);
    return NextResponse.json(
      { error: 'Failed to import datasets', details: error?.message || String(error) },
      { status: 500 }
    );
  }
}
