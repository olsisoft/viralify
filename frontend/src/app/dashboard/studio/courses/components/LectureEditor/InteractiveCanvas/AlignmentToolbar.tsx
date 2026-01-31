'use client';

import React, { memo, useCallback } from 'react';
import { motion } from 'framer-motion';
import type { SlideElement, UpdateElementRequest } from '../../../lib/lecture-editor-types';

interface AlignmentToolbarProps {
  selectedElements: SlideElement[];
  onUpdateElements: (updates: Array<{ elementId: string; updates: UpdateElementRequest }>) => Promise<void>;
  disabled?: boolean;
}

type AlignmentType = 'left' | 'center' | 'right' | 'top' | 'middle' | 'bottom';
type DistributeType = 'horizontal' | 'vertical';

const ALIGNMENT_BUTTONS: Array<{ type: AlignmentType; icon: JSX.Element; label: string }> = [
  {
    type: 'left',
    label: 'Aligner à gauche',
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M4 4v16M8 8h12M8 16h8" />
      </svg>
    ),
  },
  {
    type: 'center',
    label: 'Centrer horizontalement',
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 4v16M6 8h12M8 16h8" />
      </svg>
    ),
  },
  {
    type: 'right',
    label: 'Aligner à droite',
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M20 4v16M4 8h12M8 16h8" />
      </svg>
    ),
  },
  {
    type: 'top',
    label: 'Aligner en haut',
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M4 4h16M8 8v12M16 8v8" />
      </svg>
    ),
  },
  {
    type: 'middle',
    label: 'Centrer verticalement',
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M4 12h16M8 6v12M16 8v8" />
      </svg>
    ),
  },
  {
    type: 'bottom',
    label: 'Aligner en bas',
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M4 20h16M8 4v12M16 8v8" />
      </svg>
    ),
  },
];

const DISTRIBUTE_BUTTONS: Array<{ type: DistributeType; icon: JSX.Element; label: string }> = [
  {
    type: 'horizontal',
    label: 'Distribuer horizontalement',
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="4" y="8" width="4" height="8" rx="1" />
        <rect x="10" y="8" width="4" height="8" rx="1" />
        <rect x="16" y="8" width="4" height="8" rx="1" />
      </svg>
    ),
  },
  {
    type: 'vertical',
    label: 'Distribuer verticalement',
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="8" y="4" width="8" height="4" rx="1" />
        <rect x="8" y="10" width="8" height="4" rx="1" />
        <rect x="8" y="16" width="8" height="4" rx="1" />
      </svg>
    ),
  },
];

export const AlignmentToolbar = memo(function AlignmentToolbar({
  selectedElements,
  onUpdateElements,
  disabled = false,
}: AlignmentToolbarProps) {
  const hasMultiple = selectedElements.length > 1;
  const hasSelection = selectedElements.length > 0;

  // Align elements
  const handleAlign = useCallback(async (type: AlignmentType) => {
    if (selectedElements.length === 0) return;

    let updates: Array<{ elementId: string; updates: UpdateElementRequest }> = [];

    if (selectedElements.length === 1) {
      // Single element: align to canvas
      const element = selectedElements[0];
      let newX = element.x;
      let newY = element.y;

      switch (type) {
        case 'left':
          newX = 0;
          break;
        case 'center':
          newX = 50 - element.width / 2;
          break;
        case 'right':
          newX = 100 - element.width;
          break;
        case 'top':
          newY = 0;
          break;
        case 'middle':
          newY = 50 - element.height / 2;
          break;
        case 'bottom':
          newY = 100 - element.height;
          break;
      }

      updates = [{ elementId: element.id, updates: { x: newX, y: newY } }];
    } else {
      // Multiple elements: align to each other
      // Find bounds of selection
      const bounds = {
        minX: Math.min(...selectedElements.map((e) => e.x)),
        maxX: Math.max(...selectedElements.map((e) => e.x + e.width)),
        minY: Math.min(...selectedElements.map((e) => e.y)),
        maxY: Math.max(...selectedElements.map((e) => e.y + e.height)),
      };

      const centerX = (bounds.minX + bounds.maxX) / 2;
      const centerY = (bounds.minY + bounds.maxY) / 2;

      updates = selectedElements.map((element) => {
        let newX = element.x;
        let newY = element.y;

        switch (type) {
          case 'left':
            newX = bounds.minX;
            break;
          case 'center':
            newX = centerX - element.width / 2;
            break;
          case 'right':
            newX = bounds.maxX - element.width;
            break;
          case 'top':
            newY = bounds.minY;
            break;
          case 'middle':
            newY = centerY - element.height / 2;
            break;
          case 'bottom':
            newY = bounds.maxY - element.height;
            break;
        }

        return { elementId: element.id, updates: { x: newX, y: newY } };
      });
    }

    await onUpdateElements(updates);
  }, [selectedElements, onUpdateElements]);

  // Distribute elements evenly
  const handleDistribute = useCallback(async (type: DistributeType) => {
    if (selectedElements.length < 3) return;

    // Sort elements by position
    const sorted = [...selectedElements].sort((a, b) =>
      type === 'horizontal' ? a.x - b.x : a.y - b.y
    );

    // Calculate total space and element sizes
    const first = sorted[0];
    const last = sorted[sorted.length - 1];

    let totalSpace: number;
    let totalElementSize: number;

    if (type === 'horizontal') {
      totalSpace = (last.x + last.width) - first.x;
      totalElementSize = sorted.reduce((sum, e) => sum + e.width, 0);
    } else {
      totalSpace = (last.y + last.height) - first.y;
      totalElementSize = sorted.reduce((sum, e) => sum + e.height, 0);
    }

    // Calculate gap between elements
    const gap = (totalSpace - totalElementSize) / (sorted.length - 1);

    // Calculate new positions
    let currentPos = type === 'horizontal' ? first.x : first.y;

    const updates = sorted.map((element, index) => {
      if (index === 0) {
        // First element stays in place
        currentPos += type === 'horizontal' ? element.width + gap : element.height + gap;
        return { elementId: element.id, updates: {} };
      }

      const newPos = currentPos;
      currentPos += (type === 'horizontal' ? element.width : element.height) + gap;

      return {
        elementId: element.id,
        updates: type === 'horizontal' ? { x: newPos } : { y: newPos },
      };
    });

    await onUpdateElements(updates.filter((u) => Object.keys(u.updates).length > 0));
  }, [selectedElements, onUpdateElements]);

  if (!hasSelection) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -5 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -5 }}
      className="flex items-center gap-1 bg-gray-800/95 backdrop-blur rounded-lg p-1 shadow-xl border border-gray-700"
    >
      {/* Alignment buttons */}
      <div className="flex items-center gap-0.5">
        {ALIGNMENT_BUTTONS.map((btn) => (
          <button
            key={btn.type}
            onClick={() => handleAlign(btn.type)}
            disabled={disabled}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title={btn.label}
          >
            {btn.icon}
          </button>
        ))}
      </div>

      {/* Separator */}
      {hasMultiple && (
        <>
          <div className="w-px h-5 bg-gray-700 mx-0.5" />

          {/* Distribute buttons (only for multiple selection) */}
          <div className="flex items-center gap-0.5">
            {DISTRIBUTE_BUTTONS.map((btn) => (
              <button
                key={btn.type}
                onClick={() => handleDistribute(btn.type)}
                disabled={disabled || selectedElements.length < 3}
                className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                title={`${btn.label} (3+ éléments)`}
              >
                {btn.icon}
              </button>
            ))}
          </div>
        </>
      )}
    </motion.div>
  );
});

export default AlignmentToolbar;
