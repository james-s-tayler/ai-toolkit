'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { apiClient } from '@/utils/api';
import { RlhfPair } from '@prisma/client';

interface EvalStats {
  total: number;
  evaluated: number;
  skipped: number;
  remaining: number;
}

interface Props {
  sessionId: string;
  sessionStatus: string;
  initialPairId?: string | null;
}

export default function EvaluationUI({ sessionId, sessionStatus, initialPairId }: Props) {
  const [currentPair, setCurrentPair] = useState<RlhfPair | null>(null);
  const [prefetchedPairs, setPrefetchedPairs] = useState<RlhfPair[]>([]);
  const prefetchedRef = useRef<RlhfPair[]>([]);
  const [stats, setStats] = useState<EvalStats>({ total: 0, evaluated: 0, skipped: 0, remaining: 0 });
  const [isLoading, setIsLoading] = useState(true);
  const [flash, setFlash] = useState<'left' | 'right' | null>(null);
  const [history, setHistory] = useState<{ pair: RlhfPair; preference: string }[]>([]);
  // Randomize which image (A or B) appears on the left to reduce order bias
  const [swapped, setSwapped] = useState(false);
  const isSendingRef = useRef(false);

  // Keep ref in sync with state to avoid stale closures
  useEffect(() => { prefetchedRef.current = prefetchedPairs; }, [prefetchedPairs]);

  const fetchStats = useCallback(async () => {
    try {
      const res = await apiClient.get(`/api/rlhf/${sessionId}/evaluate`);
      setStats(res.data.stats);
    } catch (e) {
      // ignore
    }
  }, [sessionId]);

  const fetchPairById = useCallback(async (pairId: string) => {
    setIsLoading(true);
    try {
      const res = await apiClient.get(`/api/rlhf/${sessionId}/pairs/${pairId}`);
      setCurrentPair(res.data);
      setPrefetchedPairs([]);
      setSwapped(false);
      await fetchStats();
    } catch (e) {
      console.error('Error fetching pair:', e);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, fetchStats]);

  const fetchNext = useCallback(async (prefetched?: RlhfPair[]) => {
    if (prefetched && prefetched.length > 0) {
      const [next, ...rest] = prefetched;
      setCurrentPair(next);
      setPrefetchedPairs(rest);
      setSwapped(Math.random() < 0.5);
      return;
    }
    setIsLoading(true);
    try {
      const res = await apiClient.get(`/api/rlhf/${sessionId}/evaluate`);
      setCurrentPair(res.data.pair);
      setStats(res.data.stats);
      setSwapped(Math.random() < 0.5);
      // Prefetch next 5
      if (res.data.pair) {
        const prefetchRes = await apiClient.get(`/api/rlhf/${sessionId}/evaluate?after=${res.data.pair.id}`);
        setPrefetchedPairs(prefetchRes.data.pairs || []);
      }
    } catch (e) {
      console.error('Error fetching pair:', e);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    if (initialPairId) {
      fetchPairById(initialPairId);
    } else {
      fetchNext();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const sendPreference = useCallback(async (visualSide: 'left' | 'right' | 'tie' | 'skip') => {
    if (!currentPair || isSendingRef.current) return;
    isSendingRef.current = true;

    // Map visual side to actual pair A/B preference
    let preference: string;
    if (visualSide === 'tie') preference = 'tie';
    else if (visualSide === 'skip') preference = 'skip';
    else if (visualSide === 'left') preference = swapped ? 'b' : 'a';
    else preference = swapped ? 'a' : 'b';

    setFlash(visualSide === 'tie' || visualSide === 'skip' ? null : visualSide);
    setHistory(prev => [...prev.slice(-20), { pair: currentPair, preference }]);

    try {
      await apiClient.put(`/api/rlhf/${sessionId}/pairs/${currentPair.id}`, { preference });
      setStats(prev => ({ ...prev, evaluated: prev.evaluated + 1, remaining: Math.max(0, prev.remaining - 1) }));
    } catch (e) {
      console.error('Error saving preference:', e);
    }

    setTimeout(() => {
      setFlash(null);
      fetchNext(prefetchedRef.current);
      isSendingRef.current = false;
    }, 200);
  }, [currentPair, swapped, fetchNext, sessionId]);

  const handleUndo = useCallback(async () => {
    if (history.length === 0) return;
    const last = history[history.length - 1];
    setHistory(prev => prev.slice(0, -1));
    try {
      await apiClient.put(`/api/rlhf/${sessionId}/pairs/${last.pair.id}`, { preference: 'none' });
      setCurrentPair(last.pair);
      setSwapped(Math.random() < 0.5);
      setStats(prev => ({
        ...prev,
        evaluated: Math.max(0, prev.evaluated - 1),
        remaining: prev.remaining + 1,
      }));
    } catch (e) {
      console.error('Error undoing:', e);
    }
  }, [history, sessionId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement) return;
      switch (e.key.toLowerCase()) {
        case 'a': case 'arrowleft': sendPreference('left'); break;
        case 'd': case 'arrowright': sendPreference('right'); break;
        case 't': sendPreference('tie'); break;
        case 's': sendPreference('skip'); break;
        case 'z': if (e.ctrlKey || e.metaKey) handleUndo(); break;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [sendPreference, handleUndo]);

  const imageUrl = (p: string) => `/api/img/${encodeURIComponent(p)}`;

  const leftPath = currentPair ? (swapped ? currentPair.image_b_path : currentPair.image_a_path) : '';
  const rightPath = currentPair ? (swapped ? currentPair.image_a_path : currentPair.image_b_path) : '';

  if (isLoading) return <div className="flex items-center justify-center h-64 text-gray-400">Loading...</div>;

  if (!currentPair) {
    return (
      <div className="text-center py-16 text-gray-400 space-y-2">
        <div className="text-2xl">🎉</div>
        <div>All pairs evaluated!</div>
        <div className="text-sm">{stats.evaluated} evaluated, {stats.skipped} skipped</div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Progress */}
      <div className="flex items-center gap-4">
        <div
          className="flex-1 bg-gray-700 rounded-full h-2"
          role="progressbar"
          aria-valuenow={stats.total > 0 ? Math.round((stats.evaluated / stats.total) * 100) : 0}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Evaluation progress"
        >
          <div
            className="bg-green-500 h-2 rounded-full transition-all"
            style={{ width: stats.total > 0 ? `${(stats.evaluated / stats.total) * 100}%` : '0%' }}
          />
        </div>
        <span className="text-gray-400 text-sm whitespace-nowrap">
          {stats.evaluated} / {stats.total} evaluated · {stats.remaining} remaining
        </span>
      </div>

      {/* Image pair */}
      <div className="flex gap-4 h-[60vh]">
        {/* Left image */}
        <div
          className={`flex-1 relative cursor-pointer rounded-lg overflow-hidden border-2 transition-all ${
            flash === 'left' ? 'border-green-400 scale-[0.99]' : 'border-gray-700 hover:border-gray-500'
          }`}
          onClick={() => sendPreference('left')}
        >
          {leftPath ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={imageUrl(leftPath)} alt={`Image A: ${currentPair.prompt}`} className="w-full h-full object-contain bg-gray-950" />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-gray-600">No image</div>
          )}
          <div className="absolute bottom-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded">Left · Press A</div>
        </div>

        {/* Right image */}
        <div
          className={`flex-1 relative cursor-pointer rounded-lg overflow-hidden border-2 transition-all ${
            flash === 'right' ? 'border-green-400 scale-[0.99]' : 'border-gray-700 hover:border-gray-500'
          }`}
          onClick={() => sendPreference('right')}
        >
          {rightPath ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={imageUrl(rightPath)} alt={`Image B: ${currentPair.prompt}`} className="w-full h-full object-contain bg-gray-950" />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-gray-600">No image</div>
          )}
          <div className="absolute bottom-2 right-2 bg-black/60 text-white text-xs px-2 py-1 rounded">Right · Press D</div>
        </div>
      </div>

      {/* Prompt */}
      <div className="text-center text-gray-300 text-sm bg-gray-900 rounded-lg px-4 py-2">
        {currentPair.prompt}
      </div>

      {/* Keyboard hints & action buttons */}
      <div className="flex justify-center gap-3 text-sm">
        <button onClick={() => sendPreference('left')} className="text-gray-300 bg-gray-700 px-4 py-1.5 rounded-md hover:bg-gray-600">
          ← Prefer Left (A)
        </button>
        <button onClick={() => sendPreference('tie')} className="text-gray-400 bg-gray-800 px-4 py-1.5 rounded-md hover:bg-gray-700">
          Tie (T)
        </button>
        <button onClick={() => sendPreference('skip')} className="text-gray-400 bg-gray-800 px-4 py-1.5 rounded-md hover:bg-gray-700">
          Skip (S)
        </button>
        <button onClick={handleUndo} disabled={history.length === 0} className="text-gray-400 bg-gray-800 px-4 py-1.5 rounded-md hover:bg-gray-700 disabled:opacity-40">
          Undo (Ctrl+Z)
        </button>
        <button onClick={() => sendPreference('right')} className="text-gray-300 bg-gray-700 px-4 py-1.5 rounded-md hover:bg-gray-600">
          Prefer Right (D) →
        </button>
      </div>
    </div>
  );
}
