'use client';

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { apiClient } from '@/utils/api';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

interface Props {
  sessionId: string;
  runId: string | null;
  isTraining: boolean;
}

interface LossPoint {
  step: number;
  value: number;
}

function formatNum(v: number) {
  if (!Number.isFinite(v)) return '';
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(3);
  if (Math.abs(v) >= 1) return v.toFixed(4);
  return v.toPrecision(4);
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

function emaSmooth(points: LossPoint[], alpha: number): LossPoint[] {
  if (points.length === 0) return [];
  const a = clamp01(alpha);
  const out: LossPoint[] = new Array(points.length);
  let prev = points[0].value;
  out[0] = { step: points[0].step, value: prev };
  for (let i = 1; i < points.length; i++) {
    prev = a * points[i].value + (1 - a) * prev;
    out[i] = { step: points[i].step, value: prev };
  }
  return out;
}

function ToggleButton({ checked, onClick, label }: { checked: boolean; onClick: () => void; label: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        'px-3 py-1 rounded-md text-xs border transition-colors',
        checked
          ? 'bg-blue-500/10 text-blue-300 border-blue-500/30 hover:bg-blue-500/15'
          : 'bg-gray-900 text-gray-300 border-gray-800 hover:bg-gray-800/60',
      ].join(' ')}
      aria-pressed={checked}
    >
      {label}
    </button>
  );
}

export default function RlhfLossGraph({ sessionId, runId, isTraining }: Props) {
  const [points, setPoints] = useState<LossPoint[]>([]);
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const lastStepRef = useRef(0);

  const [useLogScale, setUseLogScale] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const [showSmoothed, setShowSmoothed] = useState(true);
  const [smoothing, setSmoothing] = useState(90);
  const [clipOutliers, setClipOutliers] = useState(false);

  const fetchLoss = useCallback(async (incremental = false) => {
    if (!runId) return;
    try {
      const sinceStep = incremental ? lastStepRef.current : 0;
      const res = await apiClient.get(
        `/api/rlhf/${sessionId}/train/${runId}/loss?since_step=${sinceStep}`
      );
      const newPoints: LossPoint[] = res.data.points ?? [];
      if (!incremental) {
        setPoints(newPoints);
        lastStepRef.current = newPoints.length > 0 ? newPoints[newPoints.length - 1].step : 0;
      } else if (newPoints.length > 0) {
        setPoints(prev => [...prev, ...newPoints]);
        lastStepRef.current = newPoints[newPoints.length - 1].step;
      }
      setStatus('success');
    } catch {
      setStatus('error');
    }
  }, [sessionId, runId]);

  // Initial load
  useEffect(() => {
    if (!runId) {
      setStatus('success');
      return;
    }
    lastStepRef.current = 0;
    setPoints([]);
    fetchLoss(false);
  }, [runId, fetchLoss]);

  // Poll while training
  useEffect(() => {
    if (!runId || !isTraining) return;
    const timer = setInterval(() => fetchLoss(true), 2000);
    return () => clearInterval(timer);
  }, [runId, isTraining, fetchLoss]);

  const rawPoints = useMemo(() => {
    return points.filter(p => Number.isFinite(p.value) && (!useLogScale || p.value > 0));
  }, [points, useLogScale]);

  const smoothedPoints = useMemo(() => {
    const t = clamp01(smoothing / 100);
    const alpha = 1.0 - t * 0.98;
    return emaSmooth(rawPoints, alpha);
  }, [rawPoints, smoothing]);

  const chartData = useMemo(() => {
    return rawPoints.map((p, i) => ({
      step: p.step,
      ...(showRaw ? { raw: p.value } : {}),
      ...(showSmoothed && smoothedPoints[i] ? { smoothed: smoothedPoints[i].value } : {}),
    }));
  }, [rawPoints, smoothedPoints, showRaw, showSmoothed]);

  const yDomain = useMemo((): [number | 'auto', number | 'auto'] => {
    if (!clipOutliers || chartData.length < 10) return ['auto', 'auto'];
    const vals = chartData
      .map(d => (d as any).smoothed ?? (d as any).raw)
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
      .sort((a, b) => a - b);
    if (vals.length < 10) return ['auto', 'auto'];
    const lo = vals[Math.floor(vals.length * 0.02)];
    const hi = vals[Math.ceil(vals.length * 0.98) - 1];
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) return ['auto', 'auto'];
    return [lo, hi];
  }, [clipOutliers, chartData]);

  const hasData = chartData.length > 1;
  const latest = rawPoints.length > 0 ? rawPoints[rawPoints.length - 1] : null;

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col">
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-purple-400" />
          <h2 className="text-gray-100 text-sm font-medium">Loss Graph</h2>
          <span className="text-xs text-gray-400">
            {status === 'loading' && 'Loading...'}
            {status === 'error' && 'Error'}
            {status === 'success' && hasData && `${chartData.length.toLocaleString()} points`}
            {status === 'success' && !hasData && 'No data yet'}
          </span>
        </div>
        {latest && (
          <span className="text-xs text-gray-300">
            Step {latest.step} | Loss {formatNum(latest.value)}
          </span>
        )}
      </div>

      <div className="px-4 pt-4 pb-4">
        <div className="bg-gray-950 rounded-lg border border-gray-800 h-96 relative">
          {!hasData ? (
            <div className="h-full w-full flex items-center justify-center text-sm text-gray-400">
              {status === 'error' ? 'Failed to load loss logs.' : 'Waiting for loss points...'}
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 16, bottom: 10, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  dataKey="step"
                  tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 12 }}
                  tickLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  minTickGap={40}
                />
                <YAxis
                  scale={useLogScale ? 'log' : 'linear'}
                  tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 12 }}
                  tickLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  width={72}
                  tickFormatter={formatNum}
                  domain={yDomain}
                  allowDataOverflow={clipOutliers}
                />
                <Tooltip
                  cursor={{ stroke: 'rgba(147,51,234,0.25)', strokeWidth: 1 }}
                  contentStyle={{
                    background: 'rgba(17,24,39,0.96)',
                    border: '1px solid rgba(31,41,55,1)',
                    borderRadius: 10,
                    color: 'rgba(255,255,255,0.9)',
                    fontSize: 12,
                  }}
                  labelStyle={{ color: 'rgba(255,255,255,0.75)' }}
                  labelFormatter={(label: any) => `step ${label}`}
                  formatter={(value: any, name: any) => [formatNum(Number(value)), name]}
                />
                {showRaw && (
                  <Line
                    type="monotone"
                    dataKey="raw"
                    name="loss (raw)"
                    stroke="rgba(167,139,250,0.40)"
                    strokeWidth={1.25}
                    dot={false}
                    isAnimationActive={false}
                  />
                )}
                {showSmoothed && (
                  <Line
                    type="monotone"
                    dataKey="smoothed"
                    name="loss"
                    stroke="rgba(167,139,250,1)"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      <div className="px-4 pb-3">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <label className="block text-xs text-gray-400 mb-2">Display</label>
            <div className="flex flex-wrap gap-2">
              <ToggleButton checked={showSmoothed} onClick={() => setShowSmoothed(v => !v)} label="Smoothed" />
              <ToggleButton checked={showRaw} onClick={() => setShowRaw(v => !v)} label="Raw" />
              <ToggleButton checked={useLogScale} onClick={() => setUseLogScale(v => !v)} label="Log Y" />
              <ToggleButton checked={clipOutliers} onClick={() => setClipOutliers(v => !v)} label="Clip outliers" />
            </div>
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <label className="block text-xs text-gray-400">Smoothing</label>
              <span className="text-xs text-gray-300">{smoothing}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={smoothing}
              onChange={e => setSmoothing(Number(e.target.value))}
              className="w-full accent-purple-500"
              disabled={!showSmoothed}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
