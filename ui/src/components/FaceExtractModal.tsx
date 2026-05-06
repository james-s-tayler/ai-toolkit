import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Modal } from './Modal';
import { apiClient } from '@/utils/api';

interface FaceExtractModalProps {
  isOpen: boolean;
  onClose: () => void;
  imagePaths: string[];
  currentDataset: string;
  onComplete: (destinationDataset: string, totalFaces: number) => void;
}

const ALL_TARGETS = [512, 768, 1024, 1280] as const;

const FaceExtractModal: React.FC<FaceExtractModalProps> = ({
  isOpen,
  onClose,
  imagePaths,
  currentDataset,
  onComplete,
}) => {
  const [padding, setPadding] = useState('1.5');
  const [threshold, setThreshold] = useState('0.5');
  const [selectedTargets, setSelectedTargets] = useState<Set<number>>(new Set(ALL_TARGETS));
  const [destinationMode, setDestinationMode] = useState<'existing' | 'new'>('existing');
  const [datasets, setDatasets] = useState<string[]>([]);
  const [existingTarget, setExistingTarget] = useState<string>(currentDataset);
  const [newName, setNewName] = useState<string>(`${currentDataset}-faces`);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState<{ completed: number; total: number; faces: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    if (!isOpen) return;
    setPadding('1.5');
    setThreshold('0.5');
    setSelectedTargets(new Set(ALL_TARGETS));
    setDestinationMode('existing');
    setExistingTarget(currentDataset);
    setNewName(`${currentDataset}-faces`);
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

  const toggleTarget = (t: number) => {
    setSelectedTargets(prev => {
      const next = new Set(prev);
      if (next.has(t)) {
        if (next.size > 1) next.delete(t);
      } else {
        next.add(t);
      }
      return next;
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const paddingNum = parseFloat(padding);
    if (isNaN(paddingNum) || paddingNum < 1.0 || paddingNum > 4.0) {
      setError('Padding must be between 1.0 and 4.0.');
      return;
    }
    const thresholdNum = parseFloat(threshold);
    if (isNaN(thresholdNum) || thresholdNum < 0 || thresholdNum > 1) {
      setError('Threshold must be between 0.0 and 1.0.');
      return;
    }
    if (selectedTargets.size === 0) {
      setError('Select at least one target resolution.');
      return;
    }
    const destinationDataset = destinationMode === 'existing' ? existingTarget : newName;
    if (!destinationDataset.trim()) {
      setError('Please choose a destination dataset.');
      return;
    }
    setIsLoading(true);
    setError(null);
    setProgress({ completed: 0, total: imagePaths.length, faces: 0 });

    const failed: string[] = [];
    let resolvedDestination = destinationDataset;
    let totalFaces = 0;
    const targets = Array.from(selectedTargets).sort((a, b) => a - b);

    for (const imgPath of imagePaths) {
      try {
        const res = await apiClient.post('/api/img/extractFaces', {
          imgPath,
          destinationDataset,
          padding: paddingNum,
          threshold: thresholdNum,
          targets,
        });
        if (res.data?.destinationDataset) resolvedDestination = res.data.destinationDataset;
        const count = typeof res.data?.faceCount === 'number' ? res.data.faceCount : 0;
        totalFaces += count;
      } catch (err) {
        console.error('Failed to extract faces:', imgPath, err);
        failed.push(imgPath.split(/[\\/]/).pop() ?? imgPath);
      }
      setProgress(prev =>
        prev ? { ...prev, completed: prev.completed + 1, faces: totalFaces } : null,
      );
    }

    setIsLoading(false);
    if (failed.length > 0) {
      setError(`Failed for: ${failed.join(', ')}`);
      setProgress(null);
    } else {
      onComplete(resolvedDestination, totalFaces);
      onClose();
    }
  };

  const count = imagePaths.length;
  const title = count === 1 ? 'Extract Faces' : `Extract Faces from ${count} Images`;

  if (!mounted) return null;

  return createPortal(
    <Modal isOpen={isOpen} onClose={isLoading ? () => {} : onClose} title={title} size="sm">
      <form onSubmit={handleSubmit} className="space-y-4 text-gray-200">
        <p className="text-sm text-gray-400">
          Detect faces in {count === 1 ? 'this image' : `these ${count} images`} and save a square crop of each
          face, resized to the nearest selected resolution. The source {count === 1 ? 'image stays' : 'images stay'} in place;
          crops are written without captions.
        </p>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Target resolutions</label>
          <div className="flex flex-wrap gap-2">
            {ALL_TARGETS.map(t => {
              const checked = selectedTargets.has(t);
              return (
                <label
                  key={t}
                  className={`px-3 py-1 rounded-md cursor-pointer text-sm border ${
                    checked
                      ? 'bg-blue-600 border-blue-500 text-white'
                      : 'bg-gray-700 border-gray-600 text-gray-300'
                  } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <input
                    type="checkbox"
                    className="sr-only"
                    checked={checked}
                    disabled={isLoading}
                    onChange={() => toggleTarget(t)}
                  />
                  {t}
                </label>
              );
            })}
          </div>
          <p className="text-xs text-gray-500 mt-1">Each face crop is resized to the closest match among the selected sizes.</p>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Padding ratio</label>
          <input
            type="number"
            min={1.0}
            max={4.0}
            step={0.1}
            value={padding}
            onChange={e => setPadding(e.target.value)}
            className="w-full rounded-md bg-gray-700 border border-gray-600 text-gray-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            aria-label="Padding ratio"
            disabled={isLoading}
          />
          <p className="text-xs text-gray-500 mt-1">
            Multiplier applied to the detected face bounding box for context (1.0 = tight, 1.5 = default, 2.0+ = lots of context).
          </p>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Detection confidence threshold</label>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={threshold}
            onChange={e => setThreshold(e.target.value)}
            className="w-full rounded-md bg-gray-700 border border-gray-600 text-gray-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            aria-label="Detection confidence threshold"
            disabled={isLoading}
          />
          <p className="text-xs text-gray-500 mt-1">
            Reject detections below this score (0–1). Raise to 0.6–0.8 if you see false positives; lower to catch harder faces.
          </p>
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
              <span>Extracting faces…</span>
              <span>
                {progress.completed} / {progress.total} ({progress.faces} face{progress.faces === 1 ? '' : 's'})
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
            disabled={isLoading || selectedTargets.size === 0}
          >
            {isLoading ? 'Extracting…' : 'Extract Faces'}
          </button>
        </div>
      </form>
    </Modal>,
    document.body,
  );
};

export default FaceExtractModal;
