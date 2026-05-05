import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Modal } from './Modal';
import { apiClient } from '@/utils/api';

interface FrameExtractModalProps {
  isOpen: boolean;
  onClose: () => void;
  videoPaths: string[];
  currentDataset: string;
  onComplete: (destinationDataset: string) => void;
}

const FrameExtractModal: React.FC<FrameExtractModalProps> = ({
  isOpen,
  onClose,
  videoPaths,
  currentDataset,
  onComplete,
}) => {
  const [intervalSeconds, setIntervalSeconds] = useState('1');
  const [destinationMode, setDestinationMode] = useState<'existing' | 'new'>('existing');
  const [datasets, setDatasets] = useState<string[]>([]);
  const [existingTarget, setExistingTarget] = useState<string>(currentDataset);
  const [newName, setNewName] = useState<string>(`${currentDataset}-frames`);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState<{ completed: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    if (!isOpen) return;
    setIntervalSeconds('1');
    setDestinationMode('existing');
    setExistingTarget(currentDataset);
    setNewName(`${currentDataset}-frames`);
    setError(null);
    setProgress(null);
    apiClient
      .get('/api/datasets/list')
      .then(res => {
        const data = (res.data as string[]) ?? [];
        setDatasets(data);
        if (!data.includes(currentDataset) && data.length > 0) {
          setExistingTarget(data[0]);
        }
      })
      .catch(() => setDatasets([]));
  }, [isOpen, currentDataset]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const interval = parseFloat(intervalSeconds);
    if (isNaN(interval) || interval <= 0) {
      setError('Please enter a valid interval (greater than 0).');
      return;
    }
    const destinationDataset = destinationMode === 'existing' ? existingTarget : newName;
    if (!destinationDataset.trim()) {
      setError('Please choose a destination dataset.');
      return;
    }
    setIsLoading(true);
    setError(null);
    setProgress({ completed: 0, total: videoPaths.length });

    const failed: string[] = [];
    let resolvedDestination = destinationDataset;
    for (const videoPath of videoPaths) {
      try {
        const res = await apiClient.post('/api/video/extractFrames', {
          videoPath,
          intervalSeconds: interval,
          destinationDataset,
        });
        if (res.data?.destinationDataset) resolvedDestination = res.data.destinationDataset;
      } catch (err) {
        console.error('Failed to extract frames:', videoPath, err);
        failed.push(videoPath.split(/[\\/]/).pop() ?? videoPath);
      }
      setProgress(prev => (prev ? { ...prev, completed: prev.completed + 1 } : null));
    }

    setIsLoading(false);
    if (failed.length > 0) {
      setError(`Failed for: ${failed.join(', ')}`);
      setProgress(null);
    } else {
      onComplete(resolvedDestination);
      onClose();
    }
  };

  const count = videoPaths.length;
  const title = count === 1 ? 'Extract Frames' : `Extract Frames from ${count} Videos`;

  if (!mounted) return null;

  return createPortal(
    <Modal isOpen={isOpen} onClose={isLoading ? () => {} : onClose} title={title} size="sm">
      <form onSubmit={handleSubmit} className="space-y-4 text-gray-200">
        <p className="text-sm text-gray-400">
          Extract one frame every N seconds from {count === 1 ? 'this video' : `these ${count} videos`}.
          The source {count === 1 ? 'video stays' : 'videos stay'} in place; frames are written without captions.
        </p>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Interval (seconds)</label>
          <input
            type="number"
            min={0.1}
            step={0.1}
            value={intervalSeconds}
            onChange={e => setIntervalSeconds(e.target.value)}
            placeholder="e.g. 1"
            className="w-full rounded-md bg-gray-700 border border-gray-600 text-gray-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            aria-label="Interval in seconds"
            autoFocus
            disabled={isLoading}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Destination</label>
          <div className="flex flex-col gap-2">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name="destinationMode"
                value="existing"
                checked={destinationMode === 'existing'}
                onChange={() => setDestinationMode('existing')}
                className="accent-blue-500"
                disabled={isLoading}
              />
              <span>Existing dataset</span>
            </label>
            {destinationMode === 'existing' && (
              <select
                className="w-full rounded-md bg-gray-700 border border-gray-600 text-gray-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                value={existingTarget}
                onChange={e => setExistingTarget(e.target.value)}
                disabled={isLoading || datasets.length === 0}
              >
                {datasets.map(d => (
                  <option key={d} value={d}>
                    {d}
                    {d === currentDataset ? ' (current)' : ''}
                  </option>
                ))}
              </select>
            )}
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name="destinationMode"
                value="new"
                checked={destinationMode === 'new'}
                onChange={() => setDestinationMode('new')}
                className="accent-blue-500"
                disabled={isLoading}
              />
              <span>New dataset</span>
            </label>
            {destinationMode === 'new' && (
              <input
                type="text"
                value={newName}
                onChange={e => setNewName(e.target.value)}
                placeholder="new dataset name"
                className="w-full rounded-md bg-gray-700 border border-gray-600 text-gray-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                aria-label="New dataset name"
                disabled={isLoading}
              />
            )}
          </div>
        </div>
        {progress && (
          <div>
            <div className="flex justify-between text-sm text-gray-400 mb-1">
              <span>Extracting frames…</span>
              <span>
                {progress.completed} / {progress.total}
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{
                  width: progress.total > 0 ? `${(progress.completed / progress.total) * 100}%` : '0%',
                }}
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
            disabled={isLoading || !intervalSeconds}
          >
            {isLoading ? 'Extracting…' : 'Extract Frames'}
          </button>
        </div>
      </form>
    </Modal>,
    document.body,
  );
};

export default FrameExtractModal;
