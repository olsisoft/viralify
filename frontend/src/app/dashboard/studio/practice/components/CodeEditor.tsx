'use client';

import React, { useState, useCallback } from 'react';
import { Play, RotateCcw, Copy, Check } from 'lucide-react';

interface CodeEditorProps {
  code: string;
  onChange: (code: string) => void;
  onSubmit: () => void;
  onReset: () => void;
  language?: string;
  isExecuting?: boolean;
  disabled?: boolean;
  starterCode?: string;
}

export function CodeEditor({
  code,
  onChange,
  onSubmit,
  onReset,
  language = 'yaml',
  isExecuting = false,
  disabled = false,
  starterCode = '',
}: CodeEditorProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [code]);

  const handleReset = useCallback(() => {
    onChange(starterCode);
    onReset();
  }, [starterCode, onChange, onReset]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      // Ctrl/Cmd + Enter to submit
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        onSubmit();
        return;
      }

      // Tab for indentation
      if (e.key === 'Tab') {
        e.preventDefault();
        const textarea = e.currentTarget;
        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;

        const newCode =
          code.substring(0, start) + '  ' + code.substring(end);
        onChange(newCode);

        // Set cursor position after the tab
        setTimeout(() => {
          textarea.selectionStart = textarea.selectionEnd = start + 2;
        }, 0);
      }
    },
    [code, onChange, onSubmit]
  );

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-400">{language.toUpperCase()}</span>
          <span className="text-xs text-gray-500">Ctrl+Enter pour exécuter</span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
            title="Copier"
          >
            {copied ? (
              <Check className="w-4 h-4 text-green-500" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </button>

          <button
            onClick={handleReset}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
            title="Réinitialiser"
            disabled={disabled || isExecuting}
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          <button
            onClick={onSubmit}
            disabled={disabled || isExecuting || !code.trim()}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white text-sm font-medium rounded transition-colors"
          >
            {isExecuting ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Exécution...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Exécuter
              </>
            )}
          </button>
        </div>
      </div>

      {/* Editor */}
      <div className="flex-1 relative">
        {/* Line numbers */}
        <div className="absolute left-0 top-0 bottom-0 w-12 bg-gray-800/50 border-r border-gray-700 overflow-hidden">
          <div className="pt-4 text-right pr-3">
            {code.split('\n').map((_, i) => (
              <div key={i} className="text-xs text-gray-500 leading-6">
                {i + 1}
              </div>
            ))}
          </div>
        </div>

        {/* Code textarea */}
        <textarea
          value={code}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled || isExecuting}
          className="w-full h-full pl-14 pr-4 py-4 bg-transparent text-gray-100 font-mono text-sm leading-6 resize-none focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
          placeholder="# Écrivez votre code ici..."
          spellCheck={false}
          autoCapitalize="off"
          autoCorrect="off"
        />
      </div>
    </div>
  );
}
