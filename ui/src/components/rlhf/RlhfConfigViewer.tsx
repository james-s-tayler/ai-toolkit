'use client';

import { useMemo } from 'react';
import YAML from 'yaml';

interface Props {
  config: Record<string, any>;
}

const yamlConfig: YAML.DocumentOptions &
  YAML.SchemaOptions &
  YAML.ParseOptions &
  YAML.CreateNodeOptions &
  YAML.ToStringOptions = {
  indent: 2,
  lineWidth: 999999999999,
  defaultStringType: 'QUOTE_DOUBLE',
  defaultKeyType: 'PLAIN',
  directives: true,
};

export default function RlhfConfigViewer({ config }: Props) {
  const yamlContent = useMemo(() => {
    if (!config || Object.keys(config).length === 0) return '# No config saved yet';
    return YAML.stringify(config, yamlConfig);
  }, [config]);

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800">
      <div className="bg-gray-800 px-4 py-2">
        <h3 className="text-sm text-gray-300">Config Preview (YAML)</h3>
      </div>
      <pre className="text-xs text-gray-300 p-4 overflow-auto max-h-96 font-mono whitespace-pre">
        {yamlContent}
      </pre>
    </div>
  );
}
