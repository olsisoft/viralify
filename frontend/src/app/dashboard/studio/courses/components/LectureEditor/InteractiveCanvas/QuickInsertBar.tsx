'use client';

import React, { memo, useRef } from 'react';

interface QuickInsertBarProps {
  onInsertImage: (file: File) => void;
  onInsertText: () => void;
  onInsertShape: (shape: 'rectangle' | 'circle' | 'rounded_rect') => void;
  disabled?: boolean;
}

export const QuickInsertBar = memo(function QuickInsertBar({
  onInsertImage,
  onInsertText,
  onInsertShape,
  disabled = false,
}: QuickInsertBarProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      onInsertImage(file);
    }
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="flex items-center gap-1 bg-gray-800/90 backdrop-blur rounded-lg p-1 shadow-lg border border-gray-700">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/gif,image/webp"
        onChange={handleFileSelect}
        className="hidden"
      />

      {/* Insert Image */}
      <button
        onClick={() => fileInputRef.current?.click()}
        disabled={disabled}
        className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        title="Ajouter une image"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
        <span>Image</span>
      </button>

      <div className="w-px h-5 bg-gray-700" />

      {/* Insert Text */}
      <button
        onClick={onInsertText}
        disabled={disabled}
        className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        title="Ajouter un texte"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
        </svg>
        <span>Texte</span>
      </button>

      <div className="w-px h-5 bg-gray-700" />

      {/* Insert Shape - dropdown */}
      <div className="relative group">
        <button
          disabled={disabled}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          title="Ajouter une forme"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
          </svg>
          <span>Forme</span>
          <svg className="w-3 h-3 ml-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {/* Shape dropdown */}
        <div className="absolute bottom-full left-0 mb-1 hidden group-hover:block">
          <div className="bg-gray-800 border border-gray-700 rounded-lg shadow-xl p-1 min-w-[120px]">
            <button
              onClick={() => onInsertShape('rectangle')}
              disabled={disabled}
              className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors"
            >
              <div className="w-4 h-3 bg-purple-500 rounded-sm" />
              <span>Rectangle</span>
            </button>
            <button
              onClick={() => onInsertShape('rounded_rect')}
              disabled={disabled}
              className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors"
            >
              <div className="w-4 h-3 bg-purple-500 rounded" />
              <span>Arrondi</span>
            </button>
            <button
              onClick={() => onInsertShape('circle')}
              disabled={disabled}
              className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors"
            >
              <div className="w-4 h-4 bg-purple-500 rounded-full" />
              <span>Cercle</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
});

export default QuickInsertBar;
