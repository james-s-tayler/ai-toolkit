'use client';

import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '@/utils/api';
import { RlhfTrainingRun } from '@prisma/client';

interface Props {
  sessionId: string;
  onStatusChange: () => void;
}

export default function TrainingMonitor({ sessionId, onStatusChange }: Props) {
  const [runs, setRuns] = useState<RlhfTrainingRun[]>([]);
  const [selectedRun, setSelectedRun] = useState<RlhfTrainingRun | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const fetchRuns = useCallback(async () => {
    try {
      const res = await apiClient.get(`/api/rlhf/${sessionId}/train`);
      setRuns(res.data.runs);
      // Only set the initial selected run; use functional update to avoid dependency on selectedRun
      setSelectedRun(prev => (prev === null && res.data.runs.length > 0 ? res.data.runs[0] : prev));
    } catch (e) {
      console.error('Error fetching runs:', e);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  const fetchRunStatus = useCallback(async (runId: string) => {
    try {
      const res = await apiClient.get(`/api/rlhf/${sessionId}/train/${runId}`);
      setSelectedRun(res.data);
      if (res.data.status === 'completed') onStatusChange();
      return res.data;
    } catch (e) {
      console.error('Error fetching run status:', e);
    }
  }, [sessionId, onStatusChange]);

  useEffect(() => { fetchRuns(); }, [fetchRuns]);

  // Poll active run
  useEffect(() => {
    if (!selectedRun || (selectedRun.status !== 'running' && selectedRun.status !== 'pending')) return;
    const timer = setInterval(() => fetchRunStatus(selectedRun.id), 5000);
    return () => clearInterval(timer);
  }, [selectedRun, fetchRunStatus]);

  if (isLoading) return <div className="text-gray-400 text-sm">Loading...</div>;
  if (runs.length === 0) return <div className="text-gray-500 text-sm">No training runs yet. Configure and start a run below.</div>;

  const run = selectedRun;
  if (!run) return null;

  const pct = run.total_steps > 0 ? Math.round((run.step / run.total_steps) * 100) : 0;

  const statusColors: Record<string, string> = {
    pending: 'text-gray-400',
    running: 'text-blue-400',
    completed: 'text-green-400',
    error: 'text-red-400',
  };

  return (
    <div className="space-y-4">
      {/* Run selector */}
      {runs.length > 1 && (
        <div className="flex gap-2 overflow-x-auto">
          {runs.map((r, i) => (
            <button
              key={r.id}
              onClick={() => setSelectedRun(r)}
              className={`text-sm px-3 py-1 rounded-md whitespace-nowrap ${r.id === run.id ? 'bg-gray-700 text-white' : 'bg-gray-900 text-gray-400 hover:bg-gray-800'}`}
            >
              Run {runs.length - i}
            </button>
          ))}
        </div>
      )}

      {/* Status card */}
      <div className="bg-gray-900 rounded-lg p-4 space-y-3">
        <div className="flex justify-between">
          <span className={`text-sm font-medium ${statusColors[run.status] ?? 'text-gray-400'}`}>
            {run.status.charAt(0).toUpperCase() + run.status.slice(1)}
          </span>
          <span className="text-gray-500 text-xs">{new Date(run.created_at).toLocaleString()}</span>
        </div>

        <div className="flex justify-between text-sm text-gray-400">
          <span>Step {run.step} / {run.total_steps}</span>
          <span>Loss: {run.loss.toFixed(4)}</span>
          <span>{pct}%</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all ${run.status === 'completed' ? 'bg-green-500' : 'bg-purple-500'}`}
            style={{ width: `${pct}%` }}
          />
        </div>

        {run.output_path && (
          <p className="text-xs text-gray-600 truncate">Output: {run.output_path}</p>
        )}
        {run.log_path && (
          <p className="text-xs text-gray-600 truncate">Log: {run.log_path}</p>
        )}
      </div>
    </div>
  );
}
