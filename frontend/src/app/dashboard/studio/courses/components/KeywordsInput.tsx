'use client';

import { useState, useCallback, KeyboardEvent } from 'react';

const MAX_KEYWORDS = 5;

interface KeywordsInputProps {
  keywords: string[];
  onChange: (keywords: string[]) => void;
  placeholder?: string;
  suggestions?: string[];
}

export function KeywordsInput({
  keywords,
  onChange,
  placeholder = 'Ajouter un mot-clé...',
  suggestions = [],
}: KeywordsInputProps) {
  // Ensure keywords is always an array (defensive against undefined)
  const safeKeywords = keywords ?? [];

  const [inputValue, setInputValue] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);

  const addKeyword = useCallback((keyword: string) => {
    const trimmed = keyword.trim().toLowerCase();
    if (
      trimmed &&
      safeKeywords.length < MAX_KEYWORDS &&
      !safeKeywords.includes(trimmed)
    ) {
      onChange([...safeKeywords, trimmed]);
      setInputValue('');
    }
  }, [safeKeywords, onChange]);

  const removeKeyword = useCallback((index: number) => {
    onChange(safeKeywords.filter((_, i) => i !== index));
  }, [safeKeywords, onChange]);

  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      addKeyword(inputValue);
    } else if (e.key === 'Backspace' && !inputValue && safeKeywords.length > 0) {
      removeKeyword(safeKeywords.length - 1);
    }
  }, [inputValue, safeKeywords.length, addKeyword, removeKeyword]);

  const filteredSuggestions = suggestions.filter(
    s =>
      s.toLowerCase().includes(inputValue.toLowerCase()) &&
      !safeKeywords.includes(s.toLowerCase())
  ).slice(0, 5);

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2 p-3 bg-gray-800 border border-gray-700 rounded-lg min-h-[48px]">
        {safeKeywords.map((keyword, index) => (
          <span
            key={keyword}
            className="inline-flex items-center gap-1 px-3 py-1 bg-purple-600/20 border border-purple-500/30 text-purple-300 rounded-full text-sm"
          >
            {keyword}
            <button
              type="button"
              onClick={() => removeKeyword(index)}
              className="ml-1 hover:text-purple-100 transition-colors"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </span>
        ))}
        {safeKeywords.length < MAX_KEYWORDS && (
          <div className="relative flex-1 min-w-[120px]">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => {
                setInputValue(e.target.value);
                setShowSuggestions(true);
              }}
              onKeyDown={handleKeyDown}
              onFocus={() => setShowSuggestions(true)}
              onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
              placeholder={safeKeywords.length === 0 ? placeholder : ''}
              className="w-full bg-transparent text-white text-sm outline-none placeholder-gray-500"
            />
            {/* Suggestions dropdown */}
            {showSuggestions && filteredSuggestions.length > 0 && inputValue && (
              <div className="absolute z-10 top-full left-0 mt-1 w-48 bg-gray-800 border border-gray-700 rounded-lg shadow-lg overflow-hidden">
                {filteredSuggestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    onClick={() => addKeyword(suggestion)}
                    className="w-full px-3 py-2 text-left text-sm text-gray-300 hover:bg-gray-700 transition-colors"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="flex justify-between items-center text-xs text-gray-500">
        <span>Appuyez sur Entrée ou virgule pour ajouter</span>
        <span className={safeKeywords.length >= MAX_KEYWORDS ? 'text-yellow-500' : ''}>
          {safeKeywords.length}/{MAX_KEYWORDS}
        </span>
      </div>

      {/* Quick suggestions */}
      {suggestions.length > 0 && safeKeywords.length < MAX_KEYWORDS && (
        <div className="flex flex-wrap gap-1.5 pt-1">
          <span className="text-xs text-gray-500 mr-1">Suggestions:</span>
          {suggestions
            .filter(s => !safeKeywords.includes(s.toLowerCase()))
            .slice(0, 6)
            .map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => addKeyword(suggestion)}
                className="px-2 py-0.5 text-xs bg-gray-700/50 text-gray-400 rounded hover:bg-gray-700 hover:text-gray-300 transition-colors"
              >
                + {suggestion}
              </button>
            ))}
        </div>
      )}
    </div>
  );
}
