import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

import sqlite3 from 'sqlite3';

export const runtime = 'nodejs';

const prisma = new PrismaClient();

function openDb(filename: string) {
  const db = new sqlite3.Database(filename);
  db.configure('busyTimeout', 30_000);
  return db;
}

function all<T = any>(db: sqlite3.Database, sql: string, params: any[] = []) {
  return new Promise<T[]>((resolve, reject) => {
    db.all(sql, params, (err, rows) => {
      if (err) reject(err);
      else resolve(rows as T[]);
    });
  });
}

function closeDb(db: sqlite3.Database) {
  return new Promise<void>((resolve, reject) => {
    db.close((err) => (err ? reject(err) : resolve()));
  });
}

function readJsonlLossLog(
  jsonlPath: string,
  key: string,
  sinceStep: number | null,
  stride: number,
  limit: number
) {
  const content = fs.readFileSync(jsonlPath, 'utf-8');
  const points: { step: number; wall_time: number; value: number }[] = [];

  for (const line of content.split('\n')) {
    if (!line.trim()) continue;
    try {
      const entry = JSON.parse(line);
      const step: number = entry.step;
      const value: number | undefined = entry[key] ?? entry.loss;
      if (value === undefined || !Number.isFinite(value)) continue;
      if (sinceStep != null && step <= sinceStep) continue;
      if (step % stride !== 0) continue;
      points.push({ step, wall_time: entry.wall_time ?? 0, value });
      if (points.length >= limit) break;
    } catch {
      // skip malformed lines
    }
  }

  return points;
}

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  // this must be awaited to avoid TS error
  const { jobID } = await params;

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const dbPath = path.join(jobFolder, 'loss_log.db');
  const jsonlPath = path.join(jobFolder, 'loss_log.jsonl');

  const url = new URL(request.url);
  const key = url.searchParams.get('key') ?? 'loss';
  const limit = Math.min(Number(url.searchParams.get('limit') ?? 2000), 20000);
  const sinceStepParam = url.searchParams.get('since_step');
  const sinceStep = sinceStepParam != null ? Number(sinceStepParam) : null;
  const stride = Math.max(1, Number(url.searchParams.get('stride') ?? 1));

  // Prefer SQLite database, fall back to JSONL (used by DPO/RLHF training)
  if (fs.existsSync(dbPath)) {
    const db = openDb(dbPath);

    try {
      const keysRows = await all<{ key: string }>(db, `SELECT key FROM metric_keys ORDER BY key ASC`);
      const keys = keysRows.map((r) => r.key);

      const points = await all<{
        step: number;
        wall_time: number;
        value: number | null;
        value_text: string | null;
      }>(
        db,
        `
        SELECT
          m.step AS step,
          s.wall_time AS wall_time,
          m.value_real AS value,
          m.value_text AS value_text
        FROM metrics m
        JOIN steps s ON s.step = m.step
        WHERE m.key = ?
          AND (? IS NULL OR m.step > ?)
          AND (m.step % ?) = 0
        ORDER BY m.step ASC
        LIMIT ?
        `,
        [key, sinceStep, sinceStep, stride, limit]
      );

      return NextResponse.json({
        key,
        keys,
        points: points.map((p) => ({
          step: p.step,
          wall_time: p.wall_time,
          value: p.value ?? (p.value_text ? Number(p.value_text) : null),
        })),
      });
    } finally {
      await closeDb(db);
    }
  }

  if (fs.existsSync(jsonlPath)) {
    const points = readJsonlLossLog(jsonlPath, key, sinceStep, stride, limit);
    return NextResponse.json({
      key,
      keys: ['loss'],
      points,
    });
  }

  return NextResponse.json({ keys: [], key, points: [] });
}
