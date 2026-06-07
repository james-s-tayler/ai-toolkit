'use client';

import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '@/utils/api';

interface GenerationStats {
  session_status: string;
  total: number;
  pending: number;
  queued: number;
  completed: number;
  error: number;
}

interface Props {
  sessionId: string;
  sessionStatus: string;
  onStatusChange: () => void;
}

export default function GenerationMonitor({ sessionId, sessionStatus, onStatusChange }: Props) {
  const [stats, setStats] = useState<GenerationStats | null>(null);
  const [prompts, setPrompts] = useState('');
  const [isStarting, setIsStarting] = useState(false);
  const [isAddingPairs, setIsAddingPairs] = useState(false);
  const [error, setError] = useState('');

  const fetchStats = useCallback(async () => {
    try {
      const res = await apiClient.get(`/api/rlhf/${sessionId}/generate`);
      setStats(res.data);
    } catch (e) {
      console.error('Error fetching generation stats:', e);
    }
  }, [sessionId]);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  // Poll while generating
  useEffect(() => {
    if (sessionStatus !== 'generating') return;
    const timer = setInterval(() => { fetchStats(); onStatusChange(); }, 5000);
    return () => clearInterval(timer);
  }, [sessionStatus, fetchStats, onStatusChange]);

  const handleAddPairs = async () => {
    const lines = prompts.split('\n').map(l => l.trim()).filter(Boolean);
    if (lines.length === 0) { setError('Enter at least one prompt (one per line)'); return; }
    setIsAddingPairs(true);
    setError('');
    try {
      await apiClient.post(`/api/rlhf/${sessionId}/pairs`, { prompts: lines });
      setPrompts('');
      await fetchStats();
    } catch (e: any) {
      setError(e?.response?.data?.error || 'Failed to add prompts');
    } finally {
      setIsAddingPairs(false);
    }
  };

  const handleStartGeneration = async () => {
    setIsStarting(true);
    setError('');
    try {
      await apiClient.post(`/api/rlhf/${sessionId}/generate`);
      onStatusChange();
      await fetchStats();
    } catch (e: any) {
      setError(e?.response?.data?.error || 'Failed to start generation');
    } finally {
      setIsStarting(false);
    }
  };

  const pct = stats && stats.total > 0 ? Math.round((stats.completed / stats.total) * 100) : 0;

  return (
    <div className="space-y-6">
      {/* Stats */}
      {stats && stats.total > 0 && (
        <div className="bg-gray-900 rounded-lg p-4 space-y-3">
          <div className="flex justify-between text-sm text-gray-400">
            <span>{stats.completed} / {stats.total} completed</span>
            <span>{pct}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div className="bg-blue-500 h-2 rounded-full transition-all" style={{ width: `${pct}%` }} />
          </div>
          <div className="flex gap-4 text-xs text-gray-500">
            <span className="text-yellow-400">{stats.queued} queued</span>
            <span className="text-gray-400">{stats.pending} pending</span>
            {stats.error > 0 && <span className="text-red-400">{stats.error} errors</span>}
          </div>
        </div>
      )}

      {/* Add prompts */}
      {(sessionStatus === 'setup' || sessionStatus === 'generated' || sessionStatus === 'error') && (
        <div className="bg-gray-900 rounded-lg p-4 space-y-3">
          <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">Add Prompts</h3>
          <textarea
            value={prompts}
            onChange={e => setPrompts(e.target.value)}
            rows={6}
            placeholder="Enter one prompt per line..."
            className="w-full text-sm px-3 py-2 bg-gray-800 border border-gray-700 rounded-sm focus:ring-2 focus:ring-gray-600 font-mono"
          />
          <button
            onClick={handleAddPairs}
            disabled={isAddingPairs || !prompts.trim()}
            className="text-gray-200 bg-slate-600 px-4 py-1.5 rounded-md hover:bg-slate-500 disabled:opacity-50 text-sm"
          >
            {isAddingPairs ? 'Adding...' : 'Add Pairs'}
          </button>
        </div>
      )}

      {error && <p className="text-red-400 text-sm">{error}</p>}

      {/* Actions */}
      <div className="flex gap-3">
        {(sessionStatus === 'setup' || sessionStatus === 'generated' || sessionStatus === 'error') && stats && stats.total > 0 && (
          <button
            onClick={handleStartGeneration}
            disabled={isStarting}
            className="text-gray-200 bg-blue-700 px-4 py-2 rounded-md hover:bg-blue-600 disabled:opacity-50"
          >
            {isStarting ? 'Starting...' : sessionStatus === 'error' ? 'Retry Generation' : 'Start Generation'}
          </button>
        )}
        {sessionStatus === 'generating' && (
          <div className="flex items-center gap-2 text-blue-400 text-sm">
            <div className="animate-spin w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full" />
            Generation in progress...
          </div>
        )}
      </div>
    </div>
  );
}
