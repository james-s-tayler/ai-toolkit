import React, { useState, useEffect } from 'react';
import { Modal } from './Modal';

interface BulkReencodeFpsModalProps {
  isOpen: boolean;
  onClose: () => void;
  videoPaths: string[];
  onComplete: (paths: string[]) => void;
}

const FPS_PRESETS = [8, 12, 16, 24, 25, 30, 60];
const DEFAULT_FPS = 24;

const basename = (p: string) => p.split(/[\\/]/).pop() ?? p;

const BulkReencodeFpsModal: React.FC<BulkReencodeFpsModalProps> = ({ isOpen, onClose, videoPaths, onComplete }) => {
  const [targetFps, setTargetFps] = useState<number>(DEFAULT_FPS);
  const [isLoading, setIsLoading] = useState(false);
  const [completedCount, setCompletedCount] = useState(0);
  const [currentFile, setCurrentFile] = useState<string | null>(null);
  const [currentPercent, setCurrentPercent] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setTargetFps(DEFAULT_FPS);
      setIsLoading(false);
      setCompletedCount(0);
      setCurrentFile(null);
      setCurrentPercent(null);
      setError(null);
    }
  }, [isOpen]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setCompletedCount(0);

    const failed: string[] = [];

    for (const videoPath of videoPaths) {
      setCurrentFile(videoPath);
      setCurrentPercent(0);

      let videoFailed = false;
      let lastErrorMessage: string | null = null;

      try {
        const token = typeof window !== 'undefined' ? localStorage.getItem('AI_TOOLKIT_AUTH') : null;
        const headers: Record<string, string> = { 'Content-Type': 'application/json' };
        if (token) headers['Authorization'] = `Bearer ${token}`;
        const res = await fetch('/api/video/reencodeFps', {
          method: 'POST',
          headers,
          body: JSON.stringify({ videoPath, targetFps }),
        });

        if (!res.ok || !res.body) {
          let message = `HTTP ${res.status}`;
          try {
            const errBody = await res.json();
            if (errBody?.error) message = errBody.error;
          } catch { /* ignore */ }
          videoFailed = true;
          lastErrorMessage = message;
        } else {
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buf = '';
          let sawDone = false;

          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            let nl;
            while ((nl = buf.indexOf('\n')) !== -1) {
              const line = buf.slice(0, nl).trim();
              buf = buf.slice(nl + 1);
              if (!line) continue;
              try {
                const evt = JSON.parse(line);
                if (evt.type === 'progress') {
                  setCurrentPercent(typeof evt.percent === 'number' ? evt.percent : null);
                } else if (evt.type === 'done') {
                  sawDone = true;
                } else if (evt.type === 'error') {
                  videoFailed = true;
                  lastErrorMessage = evt.message ?? 'Unknown error';
                }
              } catch {
                // ignore malformed lines
              }
            }
          }

          if (!sawDone && !videoFailed) {
            videoFailed = true;
            lastErrorMessage = 'Stream ended unexpectedly';
          }
        }
      } catch (err: any) {
        videoFailed = true;
        lastErrorMessage = err?.message ?? 'Network error';
      }

      if (videoFailed) {
        console.error('Failed to re-encode:', videoPath, lastErrorMessage);
        failed.push(`${basename(videoPath)} (${lastErrorMessage ?? 'failed'})`);
      }

      setCompletedCount(prev => prev + 1);
    }

    setIsLoading(false);
    setCurrentFile(null);
    setCurrentPercent(null);

    if (failed.length > 0) {
      setError(`Failed: ${failed.join(', ')}`);
    } else {
      onComplete(videoPaths);
      onClose();
    }
  };

  const total = videoPaths.length;
  const overallPct = total > 0 ? (completedCount / total) * 100 : 0;
  const showPerVideoBar = isLoading && currentFile !== null;

  return (
    <Modal
      isOpen={isOpen}
      onClose={isLoading ? () => {} : onClose}
      title={`Re-encode ${total} Video${total !== 1 ? 's' : ''} FPS`}
      size="sm"
    >
      <form onSubmit={handleSubmit} className="space-y-4 text-gray-200">
        <p className="text-sm text-gray-400">
          Re-encode the selected {total === 1 ? 'video' : `${total} videos`} at the target framerate. Each original
          file will be replaced. Frames are duplicated or dropped to match — playback duration stays the same.
        </p>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Target framerate</label>
          <select
            value={targetFps}
            onChange={e => setTargetFps(parseInt(e.target.value, 10))}
            className="w-full rounded-md bg-gray-700 border border-gray-600 text-gray-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            aria-label="Target framerate"
            disabled={isLoading}
          >
            {FPS_PRESETS.map(fps => (
              <option key={fps} value={fps}>{fps} fps</option>
            ))}
          </select>
        </div>

        {showPerVideoBar && (
          <div>
            <div className="flex justify-between text-sm text-gray-400 mb-1">
              <span className="truncate pr-2" title={currentFile ?? undefined}>
                Encoding {basename(currentFile!)}
              </span>
              <span>{currentPercent === null ? '…' : `${currentPercent.toFixed(0)}%`}</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
              {currentPercent === null ? (
                <div className="bg-blue-500 h-2 rounded-full animate-pulse" style={{ width: '100%' }} />
              ) : (
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-200"
                  style={{ width: `${currentPercent}%` }}
                />
              )}
            </div>
          </div>
        )}

        {isLoading && (
          <div>
            <div className="flex justify-between text-sm text-gray-400 mb-1">
              <span>Overall</span>
              <span>{completedCount} / {total}</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-700 h-2 rounded-full transition-all duration-300"
                style={{ width: `${overallPct}%` }}
              />
            </div>
          </div>
        )}

        {error && <p className="text-red-400 text-sm">{error}</p>}

        <div className="flex justify-end gap-3 pt-2">
          <button
            type="button"
            className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 focus:outline-none disabled:opacity-50"
            onClick={onClose}
            disabled={isLoading}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 focus:outline-none disabled:opacity-50"
            disabled={isLoading || total === 0}
          >
            {isLoading ? 'Re-encoding…' : 'Re-encode'}
          </button>
        </div>
      </form>
    </Modal>
  );
};

export default BulkReencodeFpsModal;
