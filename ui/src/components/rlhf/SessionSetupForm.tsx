'use client';

import { useState, useEffect } from 'react';
import { TextInput } from '@/components/formInputs';
import { apiClient } from '@/utils/api';

interface SessionFormData {
  name: string;
  model_path: string;
  comfyui_url: string;
  workflow_json: string;
  gpu_ids: string;
}

interface Props {
  onSubmit: (data: SessionFormData) => Promise<void>;
  isSubmitting: boolean;
  error?: string;
}

interface WorkflowPreset {
  name: string;
  content: string;
}

const MODEL_OPTIONS = [
  { value: 'Tongyi-MAI/Z-Image', label: 'Z-Image' },
];

export default function SessionSetupForm({ onSubmit, isSubmitting, error }: Props) {
  const [form, setForm] = useState<SessionFormData>({
    name: '',
    model_path: MODEL_OPTIONS[0].value,
    comfyui_url: 'http://127.0.0.1:9188',
    workflow_json: '',
    gpu_ids: '0',
  });
  const [presets, setPresets] = useState<WorkflowPreset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState('');

  const set = (key: keyof SessionFormData) => (value: string) => setForm(prev => ({ ...prev, [key]: value }));

  // Fetch workflow presets on mount
  useEffect(() => {
    apiClient.get('/api/rlhf-workflow-presets').then((res: any) => {
      const fetched: WorkflowPreset[] = res.data?.presets ?? [];
      setPresets(fetched);
      // Auto-select Z-Image-Default if textarea is empty
      const defaultPreset = fetched.find(p => p.name === 'Z-Image-Default');
      if (defaultPreset) {
        setSelectedPreset(defaultPreset.name);
        setForm(prev => prev.workflow_json === '' ? { ...prev, workflow_json: defaultPreset.content } : prev);
      }
    }).catch(() => {});
  }, []);

  const handlePresetChange = (presetName: string) => {
    setSelectedPreset(presetName);
    const preset = presets.find(p => p.name === presetName);
    if (preset) {
      set('workflow_json')(preset.content);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onSubmit(form);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4 max-w-2xl">
      <TextInput label="Session Name" value={form.name} onChange={set('name')} placeholder="my-rlhf-session" />

      <div>
        <label className="block text-xs mb-1 mt-2 text-gray-300">Model Type</label>
        <select
          value={form.model_path}
          onChange={e => set('model_path')(e.target.value)}
          className="w-full text-sm px-3 py-1 bg-gray-800 border border-gray-700 rounded-sm focus:ring-2 focus:ring-gray-600 focus:border-transparent"
        >
          {MODEL_OPTIONS.map(opt => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
        <p className="text-xs text-gray-500 mt-1">Model auto-downloads on first use.</p>
      </div>

      <TextInput label="ComfyUI URL" value={form.comfyui_url} onChange={set('comfyui_url')} placeholder="http://127.0.0.1:9188" />
      <TextInput label="GPU IDs" value={form.gpu_ids} onChange={set('gpu_ids')} placeholder="0" />

      <div>
        <label className="block text-xs mb-1 mt-2 text-gray-300">ComfyUI Workflow JSON</label>
        {presets.length > 0 && (
          <select
            value={selectedPreset}
            onChange={e => handlePresetChange(e.target.value)}
            className="w-full text-sm px-3 py-1 bg-gray-800 border border-gray-700 rounded-sm focus:ring-2 focus:ring-gray-600 focus:border-transparent mb-2"
          >
            <option value="">— select a preset —</option>
            {presets.map(p => (
              <option key={p.name} value={p.name}>{p.name}</option>
            ))}
          </select>
        )}
        <textarea
          value={form.workflow_json}
          onChange={e => set('workflow_json')(e.target.value)}
          rows={8}
          placeholder={'{\n  "prompt": "{{PROMPT}}",\n  "seed": {{SEED}}\n}'}
          className="w-full text-sm px-3 py-2 bg-gray-800 border border-gray-700 rounded-sm focus:ring-2 focus:ring-gray-600 focus:border-transparent font-mono"
        />
        <p className="text-xs text-gray-500 mt-1">
          Use <code className="text-gray-300">{'{{PROMPT}}'}</code> and <code className="text-gray-300">{'{{SEED}}'}</code> as placeholders in your workflow JSON.
        </p>
      </div>
      {error && <p className="text-red-400 text-sm">{error}</p>}
      <button
        type="submit"
        disabled={isSubmitting}
        className="text-gray-200 bg-green-800 px-4 py-2 rounded-md hover:bg-green-700 disabled:opacity-50"
      >
        {isSubmitting ? 'Creating...' : 'Create Session'}
      </button>
    </form>
  );
}
