'use client';

import { useState, useEffect } from 'react';

export interface TrainingParams {
  beta: number;
  learning_rate: number;
  max_train_steps: number;
  lora_rank: number;
  batch_size: number;
  blocks_to_swap: number;
  save_every: number;
  mixed_precision: string;
  gradient_checkpointing: boolean;
}

export const DEFAULT_TRAINING_PARAMS: TrainingParams = {
  beta: 5000,
  learning_rate: 1e-5,
  max_train_steps: 2000,
  lora_rank: 16,
  batch_size: 1,
  blocks_to_swap: 16,
  save_every: 250,
  mixed_precision: 'bf16',
  gradient_checkpointing: true,
};

interface Props {
  onStartTraining: (params: TrainingParams) => void;
  initialConfig?: TrainingParams | null;
  evaluatedCount: number;
  isStarting?: boolean;
  disabled?: boolean;
}

export default function TrainingConfig({ onStartTraining, initialConfig, evaluatedCount, isStarting, disabled }: Props) {
  const [params, setParams] = useState<TrainingParams>(initialConfig ?? DEFAULT_TRAINING_PARAMS);

  useEffect(() => {
    if (initialConfig) setParams(initialConfig);
  }, [initialConfig]);

  const setParam = (key: keyof TrainingParams) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const value = e.target.type === 'checkbox'
      ? (e.target as HTMLInputElement).checked
      : e.target.type === 'number' ? Number(e.target.value) : e.target.value;
    setParams(prev => ({ ...prev, [key]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onStartTraining(params);
  };

  const fieldClass = `w-full text-sm px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-sm focus:ring-2 focus:ring-gray-600 ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`;
  const labelClass = "block text-xs mb-1 mt-3 text-gray-300";

  return (
    <form onSubmit={handleSubmit} className="space-y-2 max-w-lg">
      <div className="bg-gray-900 rounded-lg p-4 mb-4">
        <p className="text-sm text-gray-400">
          {evaluatedCount} pairs available for training.
        </p>
        {evaluatedCount === 0 && (
          <p className="text-yellow-400 text-xs mt-1">Evaluate at least some pairs before starting training.</p>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className={labelClass}>Beta (DPO scale)</label>
          <input type="number" value={params.beta} onChange={setParam('beta')} disabled={disabled} className={fieldClass} />
        </div>
        <div>
          <label className={labelClass}>Learning Rate</label>
          <input type="number" step="1e-6" value={params.learning_rate} onChange={setParam('learning_rate')} disabled={disabled} className={fieldClass} />
        </div>
        <div>
          <label className={labelClass}>Max Train Steps</label>
          <input type="number" value={params.max_train_steps} onChange={setParam('max_train_steps')} disabled={disabled} className={fieldClass} />
        </div>
        <div>
          <label className={labelClass}>LoRA Rank</label>
          <input type="number" value={params.lora_rank} onChange={setParam('lora_rank')} disabled={disabled} className={fieldClass} />
        </div>
        <div>
          <label className={labelClass}>Batch Size</label>
          <input type="number" value={params.batch_size} onChange={setParam('batch_size')} min={1} disabled={disabled} className={fieldClass} />
        </div>
        <div>
          <label className={labelClass}>Blocks to Swap (CPU)</label>
          <input type="number" value={params.blocks_to_swap} onChange={setParam('blocks_to_swap')} min={0} disabled={disabled} className={fieldClass} />
        </div>
        <div>
          <label className={labelClass}>Save Checkpoint Every N Steps</label>
          <input type="number" value={params.save_every} onChange={setParam('save_every')} min={1} disabled={disabled} className={fieldClass} />
        </div>
        <div>
          <label className={labelClass}>Mixed Precision</label>
          <select value={params.mixed_precision} onChange={setParam('mixed_precision')} disabled={disabled} className={fieldClass}>
            <option value="bf16">bf16</option>
            <option value="fp16">fp16</option>
            <option value="no">no (fp32)</option>
          </select>
        </div>
        <div className="flex items-end pb-1.5">
          <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
            <input type="checkbox" checked={params.gradient_checkpointing} onChange={setParam('gradient_checkpointing')} disabled={disabled} className="w-4 h-4" />
            Gradient Checkpointing
          </label>
        </div>
      </div>

      <div className="pt-2">
        <button
          type="submit"
          disabled={isStarting || disabled || evaluatedCount === 0}
          className="text-gray-200 bg-purple-700 px-4 py-2 rounded-md hover:bg-purple-600 disabled:opacity-50"
        >
          {isStarting ? 'Starting...' : 'Start Training'}
        </button>
      </div>
    </form>
  );
}
