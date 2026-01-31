'use client';

import React, { memo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface ZoomControlsProps {
  scale: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onResetZoom: () => void;
  onSetZoom: (scale: number) => void;
  zoomPresets: number[];
  disabled?: boolean;
}

export const ZoomControls = memo(function ZoomControls({
  scale,
  onZoomIn,
  onZoomOut,
  onResetZoom,
  onSetZoom,
  zoomPresets,
  disabled = false,
}: ZoomControlsProps) {
  const [showPresets, setShowPresets] = useState(false);

  const zoomPercent = Math.round(scale * 100);

  return (
    <div className="flex items-center gap-1 bg-gray-800/95 backdrop-blur rounded-lg p-1 shadow-lg border border-gray-700">
      {/* Zoom out button */}
      <button
        onClick={onZoomOut}
        disabled={disabled || scale <= 0.25}
        className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        title="Zoom arrière (Ctrl + molette)"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
        </svg>
      </button>

      {/* Zoom level display/selector */}
      <div className="relative">
        <button
          onClick={() => setShowPresets(!showPresets)}
          disabled={disabled}
          className="px-2 py-1 min-w-[60px] text-xs font-medium text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors disabled:opacity-50"
          title="Cliquez pour choisir un niveau de zoom"
        >
          {zoomPercent}%
        </button>

        {/* Zoom presets dropdown */}
        <AnimatePresence>
          {showPresets && (
            <motion.div
              initial={{ opacity: 0, y: 5, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 5, scale: 0.95 }}
              transition={{ duration: 0.15 }}
              className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 bg-gray-800 border border-gray-700 rounded-lg shadow-xl overflow-hidden z-50"
            >
              <div className="py-1">
                {zoomPresets.map((preset) => (
                  <button
                    key={preset}
                    onClick={() => {
                      onSetZoom(preset);
                      setShowPresets(false);
                    }}
                    className={`
                      w-full px-4 py-1.5 text-xs text-left transition-colors
                      ${Math.abs(scale - preset) < 0.01
                        ? 'bg-purple-600 text-white'
                        : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                      }
                    `}
                  >
                    {Math.round(preset * 100)}%
                  </button>
                ))}
                <div className="border-t border-gray-700 mt-1 pt-1">
                  <button
                    onClick={() => {
                      onResetZoom();
                      setShowPresets(false);
                    }}
                    className="w-full px-4 py-1.5 text-xs text-left text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                  >
                    Réinitialiser
                  </button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Click outside to close */}
        {showPresets && (
          <div
            className="fixed inset-0 z-40"
            onClick={() => setShowPresets(false)}
          />
        )}
      </div>

      {/* Zoom in button */}
      <button
        onClick={onZoomIn}
        disabled={disabled || scale >= 4}
        className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        title="Zoom avant (Ctrl + molette)"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
        </svg>
      </button>

      {/* Separator */}
      <div className="w-px h-5 bg-gray-700 mx-0.5" />

      {/* Fit to screen button */}
      <button
        onClick={onResetZoom}
        disabled={disabled}
        className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        title="Ajuster à l'écran"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
        </svg>
      </button>
    </div>
  );
});

export default ZoomControls;
