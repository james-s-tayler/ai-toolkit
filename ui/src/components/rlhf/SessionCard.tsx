'use client';

import Link from 'next/link';
import { RlhfSession } from '@prisma/client';

const statusColors: Record<string, string> = {
  setup: 'text-gray-400',
  generating: 'text-blue-400',
  generated: 'text-cyan-400',
  evaluating: 'text-yellow-400',
  training: 'text-purple-400',
  completed: 'text-green-400',
  error: 'text-red-400',
};

interface SessionWithCount extends RlhfSession {
  _count?: { pairs: number };
}

export default function SessionCard({ session, onDelete }: { session: SessionWithCount; onDelete: (id: string) => void }) {
  return (
    <div className="bg-gray-900 rounded-lg p-4 flex items-start justify-between gap-4">
      <div className="flex-1 min-w-0">
        <Link href={`/rlhf/${session.id}`} className="text-gray-100 font-medium hover:text-white truncate block">
          {session.name}
        </Link>
        <div className="flex gap-4 mt-1 text-sm">
          <span className={statusColors[session.status] ?? 'text-gray-400'}>{session.status}</span>
          <span className="text-gray-500">{session._count?.pairs ?? 0} pairs</span>
          <span className="text-gray-600 text-xs truncate">{session.model_path}</span>
        </div>
      </div>
      <div className="flex gap-2 items-center flex-shrink-0">
        <Link href={`/rlhf/${session.id}`} className="text-gray-300 bg-gray-700 px-3 py-1 rounded-md text-sm hover:bg-gray-600">
          Open
        </Link>
        <button
          onClick={() => onDelete(session.id)}
          className="text-gray-400 hover:text-red-400 px-2 py-1 rounded-md text-sm"
        >
          Delete
        </button>
      </div>
    </div>
  );
}
