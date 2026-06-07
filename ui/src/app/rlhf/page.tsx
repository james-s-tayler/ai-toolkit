'use client';

import { useState, useEffect, useCallback } from 'react';
import { TopBar, MainContent } from '@/components/layout';
import Link from 'next/link';
import { apiClient } from '@/utils/api';
import { openConfirm } from '@/components/ConfirmModal';
import SessionCard from '@/components/rlhf/SessionCard';
import { RlhfSession } from '@prisma/client';

interface SessionWithCount extends RlhfSession {
  _count?: { pairs: number };
}

export default function RlhfPage() {
  const [sessions, setSessions] = useState<SessionWithCount[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const fetchSessions = useCallback(async () => {
    try {
      const res = await apiClient.get('/api/rlhf');
      setSessions(res.data.sessions);
    } catch (e) {
      console.error('Error fetching sessions:', e);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => { fetchSessions(); }, [fetchSessions]);

  const handleDelete = (id: string) => {
    openConfirm({
      title: 'Delete Session',
      message: 'Delete this RLHF session and all its pairs? This cannot be undone.',
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: () => {
        apiClient.delete(`/api/rlhf/${id}`)
          .then(() => fetchSessions())
          .catch(e => console.error('Error deleting session:', e));
      },
    });
  };

  return (
    <>
      <TopBar>
        <div><h1 className="text-lg">RLHF / DPO</h1></div>
        <div className="flex-1" />
        <div>
          <Link href="/rlhf/new" className="text-gray-200 bg-slate-600 px-3 py-1 rounded-md text-sm hover:bg-slate-500">
            New Session
          </Link>
        </div>
      </TopBar>
      <MainContent>
        {isLoading && <p className="text-gray-400">Loading...</p>}
        {!isLoading && sessions.length === 0 && (
          <div className="text-center py-16 text-gray-500">
            <p className="text-lg mb-2">No RLHF sessions yet.</p>
            <Link href="/rlhf/new" className="text-blue-400 hover:text-blue-300">Create your first session →</Link>
          </div>
        )}
        <div className="space-y-3">
          {sessions.map(s => (
            <SessionCard key={s.id} session={s} onDelete={handleDelete} />
          ))}
        </div>
      </MainContent>
    </>
  );
}
