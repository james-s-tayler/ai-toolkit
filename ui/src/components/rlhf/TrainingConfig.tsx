'use client';

import { useState, useEffect } from 'react';

export interface TrainingParams {
  beta: number;
  learning_rate: number;
  max_train_steps: number;
  lora_rank: number;
  save_every: number;
  mixed_precision: string;
  gradient_checkpointing: boolean;
  quantize: string;
  sample_every: number;
  sample_steps: number;
  sample_guidance_scale: number;
  sample_width: number;
  sample_height: number;
  sample_seed: number;
  sample_prompts: string[];
  skip_first_sample: boolean;
}

export const DEFAULT_TRAINING_PARAMS: TrainingParams = {
  beta: 5000,
  learning_rate: 1e-5,
  max_train_steps: 2000,
  lora_rank: 16,
  save_every: 250,
  mixed_precision: 'bf16',
  gradient_checkpointing: true,
  quantize: 'none',
  sample_every: 0,
  sample_steps: 25,
  sample_guidance_scale: 3.0,
  sample_width: 1024,
  sample_height: 1024,
  sample_seed: 42,
  sample_prompts: [],
  skip_first_sample: false,
};

interface Props {
  onStartTraining: (params: TrainingParams) => void;
  initialConfig?: TrainingParams | null;
  evaluatedCount: number;
  isStarting?: boolean;
  disabled?: boolean;
}

export default function TrainingConfig({ onStartTraining, initialConfig, evaluatedCount, isStarting, disabled }: Props) {
  const [params, setParams] = useState<TrainingParams>({ ...DEFAULT_TRAINING_PARAMS, ...initialConfig });

  useEffect(() => {
    if (initialConfig) setParams(prev => ({ ...DEFAULT_TRAINING_PARAMS, ...initialConfig }));
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
        <div>
          <label className={labelClass}>Quantize Base Model</label>
          <select value={params.quantize} onChange={setParam('quantize')} disabled={disabled} className={fieldClass}>
            <option value="none">None</option>
            <option value="qfloat8">FP8</option>
          </select>
        </div>
        <div className="flex items-end pb-1.5">
          <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
            <input type="checkbox" checked={params.gradient_checkpointing} onChange={setParam('gradient_checkpointing')} disabled={disabled} className="w-4 h-4" />
            Gradient Checkpointing
          </label>
        </div>
      </div>

      {/* Sample Preview Section */}
      <div className="border-t border-gray-700 pt-4 mt-4">
        <h3 className="text-sm text-gray-300 mb-2">Sample Preview</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className={labelClass}>Sample Every N Steps (0 = off)</label>
            <input type="number" value={params.sample_every} onChange={setParam('sample_every')} min={0} disabled={disabled} className={fieldClass} />
          </div>
          <div>
            <label className={labelClass}>Sample Steps</label>
            <input type="number" value={params.sample_steps} onChange={setParam('sample_steps')} min={1} disabled={disabled} className={fieldClass} />
          </div>
          <div>
            <label className={labelClass}>Guidance Scale</label>
            <input type="number" step="0.1" value={params.sample_guidance_scale} onChange={setParam('sample_guidance_scale')} min={1} disabled={disabled} className={fieldClass} />
          </div>
          <div>
            <label className={labelClass}>Seed</label>
            <input type="number" value={params.sample_seed} onChange={setParam('sample_seed')} disabled={disabled} className={fieldClass} />
          </div>
          <div>
            <label className={labelClass}>Width</label>
            <input type="number" value={params.sample_width} onChange={setParam('sample_width')} min={256} step={16} disabled={disabled} className={fieldClass} />
          </div>
          <div>
            <label className={labelClass}>Height</label>
            <input type="number" value={params.sample_height} onChange={setParam('sample_height')} min={256} step={16} disabled={disabled} className={fieldClass} />
          </div>
        </div>

        {/* Sample Prompts */}
        <div className="mt-3">
          <label className={labelClass}>Sample Prompts ({params.sample_prompts.length})</label>
          {params.sample_prompts.map((prompt, i) => (
            <div key={i} className="flex gap-2 mb-2">
              <input
                type="text"
                value={prompt}
                onChange={e => {
                  const updated = [...params.sample_prompts];
                  updated[i] = e.target.value;
                  setParams(prev => ({ ...prev, sample_prompts: updated }));
                }}
                placeholder="Enter prompt"
                disabled={disabled}
                className={fieldClass}
              />
              <button
                type="button"
                onClick={() => setParams(prev => ({
                  ...prev,
                  sample_prompts: prev.sample_prompts.filter((_, idx) => idx !== i),
                }))}
                disabled={disabled}
                className="text-gray-500 hover:text-red-400 px-2 text-sm"
              >
                X
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={() => setParams(prev => ({
              ...prev,
              sample_prompts: [...prev.sample_prompts, ''],
            }))}
            disabled={disabled}
            className="w-full px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded-sm text-sm text-gray-300 transition-colors"
          >
            Add Prompt
          </button>
        </div>

        <div className="mt-3">
          <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
            <input type="checkbox" checked={params.skip_first_sample} onChange={setParam('skip_first_sample')} disabled={disabled} className="w-4 h-4" />
            Skip First Sample
          </label>
          <p className="text-xs text-gray-500 mt-1 ml-6">By default, baseline samples are generated before training starts. Check this to skip them.</p>
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
