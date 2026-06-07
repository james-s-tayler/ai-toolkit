'use client';

import { useState, useEffect } from 'react';
import { TextInput } from '@/components/formInputs';
import { apiClient } from '@/utils/api';
import useDatasetList from '@/hooks/useDatasetList';

interface SessionFormData {
  name: string;
  model_path: string;
  comfyui_url: string;
  workflow_json: string;
  gpu_ids: string;
  dataset_mode: 'comfyui' | 'import';
  accepted_dataset: string;
  rejected_dataset: string;
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
    dataset_mode: 'comfyui',
    accepted_dataset: '',
    rejected_dataset: '',
  });
  const [presets, setPresets] = useState<WorkflowPreset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState('');
  const { datasets } = useDatasetList();

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
    if (form.dataset_mode === 'import') {
      if (!form.accepted_dataset) return;
      if (!form.rejected_dataset) return;
      if (form.accepted_dataset === form.rejected_dataset) return;
    }
    await onSubmit(form);
  };

  const isImport = form.dataset_mode === 'import';

  const importValid =
    isImport &&
    form.accepted_dataset &&
    form.rejected_dataset &&
    form.accepted_dataset !== form.rejected_dataset;

  const canSubmit = form.name.trim() && form.model_path.trim() && (!isImport || importValid);

  return (
    <form onSubmit={handleSubmit} className="space-y-4 max-w-2xl">
      {/* Mode selector */}
      <div>
        <label className="block text-xs mb-2 text-gray-300">Session Mode</label>
        <div className="flex gap-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="dataset_mode"
              value="comfyui"
              checked={!isImport}
              onChange={() => setForm(prev => ({ ...prev, dataset_mode: 'comfyui' }))}
              className="accent-green-600"
            />
            <span className="text-sm text-gray-200">Generate via ComfyUI</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="dataset_mode"
              value="import"
              checked={isImport}
              onChange={() => setForm(prev => ({ ...prev, dataset_mode: 'import' }))}
              className="accent-green-600"
            />
            <span className="text-sm text-gray-200">Import Existing Datasets</span>
          </label>
        </div>
      </div>

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

      {isImport ? (
        <>
          {/* Import mode: dataset selectors */}
          <div>
            <label className="block text-xs mb-1 mt-2 text-gray-300">Accepted (Winner) Dataset</label>
            <select
              value={form.accepted_dataset}
              onChange={e => set('accepted_dataset')(e.target.value)}
              className="w-full text-sm px-3 py-1 bg-gray-800 border border-gray-700 rounded-sm focus:ring-2 focus:ring-gray-600 focus:border-transparent"
            >
              <option value="">— select dataset —</option>
              {datasets.map(d => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs mb-1 mt-2 text-gray-300">Rejected (Loser) Dataset</label>
            <select
              value={form.rejected_dataset}
              onChange={e => set('rejected_dataset')(e.target.value)}
              className="w-full text-sm px-3 py-1 bg-gray-800 border border-gray-700 rounded-sm focus:ring-2 focus:ring-gray-600 focus:border-transparent"
            >
              <option value="">— select dataset —</option>
              {datasets.map(d => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>
          {form.accepted_dataset && form.rejected_dataset && form.accepted_dataset === form.rejected_dataset && (
            <p className="text-red-400 text-sm">Accepted and rejected datasets must be different.</p>
          )}
          <TextInput label="GPU IDs" value={form.gpu_ids} onChange={set('gpu_ids')} placeholder="0" />
        </>
      ) : (
        <>
          {/* ComfyUI mode: existing fields */}
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
        </>
      )}
      {error && <p className="text-red-400 text-sm">{error}</p>}
      <button
        type="submit"
        disabled={isSubmitting || !canSubmit}
        className="text-gray-200 bg-green-800 px-4 py-2 rounded-md hover:bg-green-700 disabled:opacity-50"
      >
        {isSubmitting ? (isImport ? 'Importing...' : 'Creating...') : 'Create Session'}
      </button>
    </form>
  );
}
