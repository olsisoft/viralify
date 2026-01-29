'use client';

import React, { useState, useRef } from 'react';
import type { MediaType } from '../../lib/lecture-editor-types';

interface EditorToolbarProps {
  canUndo: boolean;
  canRedo: boolean;
  historyLength: number;
  futureLength: number;
  onUndo: () => void;
  onRedo: () => void;
  onSave: () => void;
  onInsertMedia: (type: MediaType) => void;
  onRecompose: () => void;
  isSaving: boolean;
  isRegenerating: boolean;
  hasUnsavedChanges: boolean;
}

export function EditorToolbar({
  canUndo,
  canRedo,
  historyLength,
  futureLength,
  onUndo,
  onRedo,
  onSave,
  onInsertMedia,
  onRecompose,
  isSaving,
  isRegenerating,
  hasUnsavedChanges,
}: EditorToolbarProps) {
  const [showInsertMenu, setShowInsertMenu] = useState(false);
  const insertMenuRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (insertMenuRef.current && !insertMenuRef.current.contains(event.target as Node)) {
        setShowInsertMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="flex items-center gap-2">
      {/* Undo/Redo group */}
      <div className="flex items-center bg-gray-800 rounded-lg">
        {/* Undo button */}
        <button
          onClick={onUndo}
          disabled={!canUndo || isSaving || isRegenerating}
          className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-l-lg disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          title={`Annuler (Ctrl+Z)${historyLength > 0 ? ` - ${historyLength} action${historyLength > 1 ? 's' : ''}` : ''}`}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
          </svg>
        </button>

        {/* History count */}
        {(historyLength > 0 || futureLength > 0) && (
          <div className="px-2 py-1 text-xs text-gray-500 border-l border-r border-gray-700 min-w-[40px] text-center">
            {historyLength}/{historyLength + futureLength}
          </div>
        )}

        {/* Redo button */}
        <button
          onClick={onRedo}
          disabled={!canRedo || isSaving || isRegenerating}
          className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-r-lg disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          title={`Rétablir (Ctrl+Y)${futureLength > 0 ? ` - ${futureLength} action${futureLength > 1 ? 's' : ''}` : ''}`}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 10h-10a8 8 0 00-8 8v2M21 10l-6 6m6-6l-6-6" />
          </svg>
        </button>
      </div>

      {/* Divider */}
      <div className="w-px h-6 bg-gray-700" />

      {/* Insert media dropdown */}
      <div className="relative" ref={insertMenuRef}>
        <button
          onClick={() => setShowInsertMenu(!showInsertMenu)}
          disabled={isSaving || isRegenerating}
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-gray-800 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          Insérer
          <svg className={`w-3 h-3 transition-transform ${showInsertMenu ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {/* Dropdown menu */}
        {showInsertMenu && (
          <div className="absolute top-full left-0 mt-1 bg-gray-900 border border-gray-700 rounded-lg shadow-xl z-20 min-w-[160px] py-1">
            <button
              onClick={() => {
                onInsertMedia('image');
                setShowInsertMenu(false);
              }}
              className="w-full px-4 py-2 text-left text-sm text-white hover:bg-gray-800 flex items-center gap-3"
            >
              <span className="text-lg">🖼️</span>
              Image
            </button>
            <button
              onClick={() => {
                onInsertMedia('video');
                setShowInsertMenu(false);
              }}
              className="w-full px-4 py-2 text-left text-sm text-white hover:bg-gray-800 flex items-center gap-3"
            >
              <span className="text-lg">🎬</span>
              Vidéo
            </button>
            <button
              onClick={() => {
                onInsertMedia('audio');
                setShowInsertMenu(false);
              }}
              className="w-full px-4 py-2 text-left text-sm text-white hover:bg-gray-800 flex items-center gap-3"
            >
              <span className="text-lg">🎵</span>
              Audio
            </button>
          </div>
        )}
      </div>

      {/* Divider */}
      <div className="w-px h-6 bg-gray-700" />

      {/* Save indicator / Recompose button */}
      <button
        onClick={onRecompose}
        disabled={isRegenerating || !hasUnsavedChanges}
        className={`
          flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg transition-colors
          ${hasUnsavedChanges
            ? 'bg-purple-600 text-white hover:bg-purple-500'
            : 'bg-gray-800 text-gray-400'
          }
          disabled:opacity-50 disabled:cursor-not-allowed
        `}
        title="Ctrl+S"
      >
        {isSaving || isRegenerating ? (
          <>
            <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            <span>{isSaving ? 'Sauvegarde...' : 'Recomposition...'}</span>
          </>
        ) : (
          <>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            <span>Recomposer</span>
            {hasUnsavedChanges && (
              <span className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
            )}
          </>
        )}
      </button>
    </div>
  );
}

export default EditorToolbar;
