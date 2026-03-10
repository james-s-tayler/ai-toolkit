'use client';

import { useState } from 'react';
import { TextInput } from '@/components/formInputs';

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

export default function SessionSetupForm({ onSubmit, isSubmitting, error }: Props) {
  const [form, setForm] = useState<SessionFormData>({
    name: '',
    model_path: '',
    comfyui_url: 'http://127.0.0.1:9188',
    workflow_json: '',
    gpu_ids: '0',
  });

  const set = (key: keyof SessionFormData) => (value: string) => setForm(prev => ({ ...prev, [key]: value }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onSubmit(form);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4 max-w-2xl">
      <TextInput label="Session Name" value={form.name} onChange={set('name')} placeholder="my-rlhf-session" />
      <TextInput label="Model Path" value={form.model_path} onChange={set('model_path')} placeholder="/path/to/model.safetensors" />
      <TextInput label="ComfyUI URL" value={form.comfyui_url} onChange={set('comfyui_url')} placeholder="http://127.0.0.1:9188" />
      <TextInput label="GPU IDs" value={form.gpu_ids} onChange={set('gpu_ids')} placeholder="0" />
      <div>
        <label className="block text-xs mb-1 mt-2 text-gray-300">ComfyUI Workflow JSON</label>
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
