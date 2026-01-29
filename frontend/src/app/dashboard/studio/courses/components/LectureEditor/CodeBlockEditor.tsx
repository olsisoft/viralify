'use client';

import React, { useState, useCallback, useEffect, useRef } from 'react';
import type { CodeBlockComponent } from '../../lib/lecture-editor-types';

// Common programming languages for code blocks
const LANGUAGES = [
  { value: 'javascript', label: 'JavaScript' },
  { value: 'typescript', label: 'TypeScript' },
  { value: 'python', label: 'Python' },
  { value: 'java', label: 'Java' },
  { value: 'csharp', label: 'C#' },
  { value: 'cpp', label: 'C++' },
  { value: 'c', label: 'C' },
  { value: 'go', label: 'Go' },
  { value: 'rust', label: 'Rust' },
  { value: 'ruby', label: 'Ruby' },
  { value: 'php', label: 'PHP' },
  { value: 'swift', label: 'Swift' },
  { value: 'kotlin', label: 'Kotlin' },
  { value: 'sql', label: 'SQL' },
  { value: 'html', label: 'HTML' },
  { value: 'css', label: 'CSS' },
  { value: 'scss', label: 'SCSS' },
  { value: 'json', label: 'JSON' },
  { value: 'yaml', label: 'YAML' },
  { value: 'xml', label: 'XML' },
  { value: 'markdown', label: 'Markdown' },
  { value: 'bash', label: 'Bash' },
  { value: 'shell', label: 'Shell' },
  { value: 'powershell', label: 'PowerShell' },
  { value: 'dockerfile', label: 'Dockerfile' },
  { value: 'plaintext', label: 'Plain Text' },
];

interface CodeBlockEditorProps {
  codeBlock: CodeBlockComponent;
  onChange: (updated: CodeBlockComponent) => void;
  onClose: () => void;
  onSave: () => void;
}

export function CodeBlockEditor({
  codeBlock,
  onChange,
  onClose,
  onSave,
}: CodeBlockEditorProps) {
  const [code, setCode] = useState(codeBlock.code);
  const [language, setLanguage] = useState(codeBlock.language);
  const [filename, setFilename] = useState(codeBlock.filename || '');
  const [showLineNumbers, setShowLineNumbers] = useState(codeBlock.showLineNumbers);
  const [highlightLines, setHighlightLines] = useState<string>(
    codeBlock.highlightLines.join(', ')
  );
  const [hasChanges, setHasChanges] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const lineNumbersRef = useRef<HTMLDivElement>(null);

  // Track changes
  useEffect(() => {
    const changed =
      code !== codeBlock.code ||
      language !== codeBlock.language ||
      filename !== (codeBlock.filename || '') ||
      showLineNumbers !== codeBlock.showLineNumbers ||
      highlightLines !== codeBlock.highlightLines.join(', ');
    setHasChanges(changed);
  }, [code, language, filename, showLineNumbers, highlightLines, codeBlock]);

  // Parse highlight lines
  const parseHighlightLines = (input: string): number[] => {
    const lines: number[] = [];
    const parts = input.split(',').map(p => p.trim()).filter(Boolean);
    for (const part of parts) {
      if (part.includes('-')) {
        const [start, end] = part.split('-').map(Number);
        if (!isNaN(start) && !isNaN(end)) {
          for (let i = start; i <= end; i++) {
            lines.push(i);
          }
        }
      } else {
        const num = parseInt(part);
        if (!isNaN(num)) {
          lines.push(num);
        }
      }
    }
    return [...new Set(lines)].sort((a, b) => a - b);
  };

  // Handle save
  const handleSave = useCallback(() => {
    onChange({
      ...codeBlock,
      code,
      language,
      filename: filename || undefined,
      showLineNumbers,
      highlightLines: parseHighlightLines(highlightLines),
    });
    onSave();
  }, [codeBlock, code, language, filename, showLineNumbers, highlightLines, onChange, onSave]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Save with Ctrl+S / Cmd+S
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        handleSave();
      }
      // Close with Escape
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleSave, onClose]);

  // Sync scroll between textarea and line numbers
  const handleScroll = useCallback(() => {
    if (lineNumbersRef.current && textareaRef.current) {
      lineNumbersRef.current.scrollTop = textareaRef.current.scrollTop;
    }
  }, []);

  // Handle tab key in textarea
  const handleKeyDownTextarea = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const start = e.currentTarget.selectionStart;
      const end = e.currentTarget.selectionEnd;
      const newCode = code.substring(0, start) + '  ' + code.substring(end);
      setCode(newCode);
      // Set cursor position after the tab
      setTimeout(() => {
        if (textareaRef.current) {
          textareaRef.current.selectionStart = textareaRef.current.selectionEnd = start + 2;
        }
      }, 0);
    }
  }, [code]);

  // Calculate line count
  const lineCount = code.split('\n').length;
  const parsedHighlightLines = parseHighlightLines(highlightLines);

  return (
    <div className="fixed inset-0 bg-black/80 z-[60] flex items-center justify-center p-4">
      <div className="bg-gray-900 rounded-xl border border-gray-700 shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
          <h3 className="text-white font-semibold">Éditeur de code</h3>
          <div className="flex items-center gap-2">
            {hasChanges && (
              <span className="text-yellow-400 text-xs flex items-center gap-1">
                <span className="w-1.5 h-1.5 bg-yellow-400 rounded-full" />
                Non sauvegardé
              </span>
            )}
            <button
              onClick={onClose}
              className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Settings bar */}
        <div className="flex items-center gap-4 px-4 py-2 bg-gray-800/50 border-b border-gray-800">
          {/* Language selector */}
          <div className="flex items-center gap-2">
            <label className="text-gray-400 text-xs">Langage:</label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="bg-gray-800 text-white text-sm rounded px-2 py-1 border border-gray-700 focus:border-purple-500 focus:outline-none"
            >
              {LANGUAGES.map((lang) => (
                <option key={lang.value} value={lang.value}>
                  {lang.label}
                </option>
              ))}
            </select>
          </div>

          {/* Filename */}
          <div className="flex items-center gap-2">
            <label className="text-gray-400 text-xs">Fichier:</label>
            <input
              type="text"
              value={filename}
              onChange={(e) => setFilename(e.target.value)}
              placeholder="ex: main.py"
              className="bg-gray-800 text-white text-sm rounded px-2 py-1 border border-gray-700 focus:border-purple-500 focus:outline-none w-32"
            />
          </div>

          {/* Highlight lines */}
          <div className="flex items-center gap-2">
            <label className="text-gray-400 text-xs">Lignes en évidence:</label>
            <input
              type="text"
              value={highlightLines}
              onChange={(e) => setHighlightLines(e.target.value)}
              placeholder="1, 3-5, 8"
              className="bg-gray-800 text-white text-sm rounded px-2 py-1 border border-gray-700 focus:border-purple-500 focus:outline-none w-28"
            />
          </div>

          {/* Line numbers toggle */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showLineNumbers}
              onChange={(e) => setShowLineNumbers(e.target.checked)}
              className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-purple-600 focus:ring-purple-500"
            />
            <span className="text-gray-400 text-xs">Numéros de ligne</span>
          </label>
        </div>

        {/* Code editor area */}
        <div className="flex-1 overflow-hidden flex">
          {/* Line numbers */}
          {showLineNumbers && (
            <div
              ref={lineNumbersRef}
              className="flex-shrink-0 bg-gray-950 text-gray-500 text-xs font-mono py-3 overflow-hidden select-none border-r border-gray-800"
              style={{ minWidth: '3rem' }}
            >
              {Array.from({ length: lineCount }, (_, i) => {
                const lineNum = i + 1;
                const isHighlighted = parsedHighlightLines.includes(lineNum);
                return (
                  <div
                    key={i}
                    className={`px-2 text-right leading-5 ${isHighlighted ? 'bg-yellow-500/20 text-yellow-400' : ''}`}
                  >
                    {lineNum}
                  </div>
                );
              })}
            </div>
          )}

          {/* Code textarea */}
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={code}
              onChange={(e) => setCode(e.target.value)}
              onScroll={handleScroll}
              onKeyDown={handleKeyDownTextarea}
              className="absolute inset-0 w-full h-full bg-gray-950 text-green-400 font-mono text-sm p-3 resize-none focus:outline-none leading-5"
              spellCheck={false}
              placeholder="// Écrivez votre code ici..."
            />
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-3 bg-gray-800/50 border-t border-gray-800">
          <div className="text-gray-500 text-xs">
            {lineCount} ligne{lineCount > 1 ? 's' : ''} • {code.length} caractères
            {parsedHighlightLines.length > 0 && (
              <span className="ml-2">
                • {parsedHighlightLines.length} ligne{parsedHighlightLines.length > 1 ? 's' : ''} en évidence
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-gray-500 text-xs mr-2">
              <kbd className="px-1.5 py-0.5 bg-gray-700 rounded text-xs">Ctrl+S</kbd> pour sauvegarder
            </span>
            <button
              onClick={onClose}
              className="px-4 py-1.5 text-sm bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors"
            >
              Annuler
            </button>
            <button
              onClick={handleSave}
              disabled={!hasChanges}
              className="px-4 py-1.5 text-sm bg-purple-600 text-white rounded-lg hover:bg-purple-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Sauvegarder
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CodeBlockEditor;
