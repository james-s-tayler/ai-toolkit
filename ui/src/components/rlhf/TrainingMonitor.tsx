'use client';

import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '@/utils/api';
import { RlhfTrainingRun } from '@prisma/client';

interface TrainingRunWithSpeed extends RlhfTrainingRun {
  speed_string?: string;
}

interface Props {
  sessionId: string;
  onStatusChange: () => void;
}

export default function TrainingMonitor({ sessionId, onStatusChange }: Props) {
  const [runs, setRuns] = useState<TrainingRunWithSpeed[]>([]);
  const [selectedRun, setSelectedRun] = useState<TrainingRunWithSpeed | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const fetchRuns = useCallback(async () => {
    try {
      const res = await apiClient.get(`/api/rlhf/${sessionId}/train`);
      setRuns(res.data.runs);
      // Always select the latest run (index 0, sorted desc by created_at)
      if (res.data.runs.length > 0) {
        setSelectedRun(prev => prev === null ? res.data.runs[0] : prev);
      }
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
    if (!selectedRun || (selectedRun.status !== 'running' && selectedRun.status !== 'pending' && selectedRun.status !== 'paused')) return;
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
    paused: 'text-amber-400',
    completed: 'text-green-400',
    stopped: 'text-gray-400',
    error: 'text-red-400',
  };

  return (
    <div className="space-y-4">
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
          <span>{run.speed_string || ''}</span>
          <span>Loss: {run.loss.toFixed(4)}</span>
          <span>{pct}%</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all ${run.status === 'completed' ? 'bg-green-500' : run.status === 'paused' ? 'bg-amber-500' : 'bg-purple-500'}`}
            style={{ width: `${pct}%` }}
          />
        </div>

        {run.output_path && (
          <p className="text-xs text-gray-600 truncate">Output: {run.output_path}</p>
        )}
      </div>
    </div>
  );
}
