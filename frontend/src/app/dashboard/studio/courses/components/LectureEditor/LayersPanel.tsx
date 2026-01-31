'use client';

import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence, Reorder } from 'framer-motion';
import type { SlideElement } from '../../lib/lecture-editor-types';

interface LayersPanelProps {
  elements: SlideElement[];
  selectedElementId: string | null;
  onSelectElement: (elementId: string | null) => void;
  onReorderElements: (newOrder: SlideElement[]) => void;
  onToggleVisibility: (elementId: string) => void;
  onToggleLock: (elementId: string) => void;
  onDeleteElement: (elementId: string) => void;
  onDuplicateElement?: (elementId: string) => void;
  onBringToFront?: (elementId: string) => void;
  onSendToBack?: (elementId: string) => void;
  disabled?: boolean;
}

// Get element type icon
function getElementIcon(element: SlideElement): string {
  switch (element.type) {
    case 'image':
      return '🖼️';
    case 'text_block':
      return '📝';
    case 'shape':
      if (element.shapeContent?.shape === 'circle') return '⚪';
      if (element.shapeContent?.shape === 'line') return '➖';
      return '⬛';
    default:
      return '📦';
  }
}

// Get element label
function getElementLabel(element: SlideElement): string {
  switch (element.type) {
    case 'image':
      return element.imageContent?.originalFilename || 'Image';
    case 'text_block':
      const text = element.textContent?.text || '';
      return text.length > 20 ? text.substring(0, 20) + '...' : text || 'Texte';
    case 'shape':
      const shapeLabels: Record<string, string> = {
        rectangle: 'Rectangle',
        circle: 'Cercle',
        rounded_rect: 'Rect arrondi',
        line: 'Ligne',
        arrow: 'Flèche',
      };
      return shapeLabels[element.shapeContent?.shape || ''] || 'Forme';
    default:
      return 'Élément';
  }
}

// Layer item component
function LayerItem({
  element,
  isSelected,
  onSelect,
  onToggleVisibility,
  onToggleLock,
  onDelete,
  disabled,
}: {
  element: SlideElement;
  isSelected: boolean;
  onSelect: () => void;
  onToggleVisibility: () => void;
  onToggleLock: () => void;
  onDelete: () => void;
  disabled?: boolean;
}) {
  const [showActions, setShowActions] = useState(false);

  return (
    <Reorder.Item
      value={element}
      id={element.id}
      className={`relative group ${disabled ? 'opacity-50 pointer-events-none' : ''}`}
      whileDrag={{ scale: 1.02, boxShadow: '0 4px 12px rgba(0,0,0,0.3)' }}
    >
      <div
        onClick={onSelect}
        onMouseEnter={() => setShowActions(true)}
        onMouseLeave={() => setShowActions(false)}
        className={`
          flex items-center gap-2 px-2 py-1.5 rounded-lg cursor-pointer transition-colors
          ${isSelected ? 'bg-purple-600/20 border border-purple-500' : 'hover:bg-gray-800 border border-transparent'}
          ${!element.visible ? 'opacity-50' : ''}
        `}
      >
        {/* Drag handle */}
        <div className="text-gray-600 hover:text-gray-400 cursor-grab active:cursor-grabbing flex-shrink-0">
          <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
            <path d="M8 6a2 2 0 1 1-4 0 2 2 0 0 1 4 0zm0 6a2 2 0 1 1-4 0 2 2 0 0 1 4 0zm0 6a2 2 0 1 1-4 0 2 2 0 0 1 4 0zm8-12a2 2 0 1 1-4 0 2 2 0 0 1 4 0zm0 6a2 2 0 1 1-4 0 2 2 0 0 1 4 0zm0 6a2 2 0 1 1-4 0 2 2 0 0 1 4 0z" />
          </svg>
        </div>

        {/* Icon */}
        <span className="text-sm flex-shrink-0">{getElementIcon(element)}</span>

        {/* Label */}
        <span className="text-xs text-gray-300 truncate flex-1">
          {getElementLabel(element)}
        </span>

        {/* Status icons */}
        <div className="flex items-center gap-1 flex-shrink-0">
          {element.locked && (
            <span className="text-gray-500" title="Verrouillé">
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                <path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3.1-9H8.9V6c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2z" />
              </svg>
            </span>
          )}
          {!element.visible && (
            <span className="text-gray-500" title="Masqué">
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 7c2.76 0 5 2.24 5 5 0 .65-.13 1.26-.36 1.83l2.92 2.92c1.51-1.26 2.7-2.89 3.43-4.75-1.73-4.39-6-7.5-11-7.5-1.4 0-2.74.25-3.98.7l2.16 2.16C10.74 7.13 11.35 7 12 7zM2 4.27l2.28 2.28.46.46C3.08 8.3 1.78 10.02 1 12c1.73 4.39 6 7.5 11 7.5 1.55 0 3.03-.3 4.38-.84l.42.42L19.73 22 21 20.73 3.27 3 2 4.27zM7.53 9.8l1.55 1.55c-.05.21-.08.43-.08.65 0 1.66 1.34 3 3 3 .22 0 .44-.03.65-.08l1.55 1.55c-.67.33-1.41.53-2.2.53-2.76 0-5-2.24-5-5 0-.79.2-1.53.53-2.2zm4.31-.78l3.15 3.15.02-.16c0-1.66-1.34-3-3-3l-.17.01z" />
              </svg>
            </span>
          )}
        </div>

        {/* Action buttons on hover */}
        <AnimatePresence>
          {showActions && (
            <motion.div
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              className="flex items-center gap-0.5 flex-shrink-0"
            >
              {/* Visibility toggle */}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleVisibility();
                }}
                className={`p-1 rounded transition-colors ${
                  element.visible ? 'text-gray-400 hover:text-white' : 'text-gray-600 hover:text-gray-400'
                }`}
                title={element.visible ? 'Masquer' : 'Afficher'}
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  {element.visible ? (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  ) : (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                  )}
                </svg>
              </button>

              {/* Lock toggle */}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleLock();
                }}
                className={`p-1 rounded transition-colors ${
                  element.locked ? 'text-yellow-500 hover:text-yellow-400' : 'text-gray-400 hover:text-white'
                }`}
                title={element.locked ? 'Déverrouiller' : 'Verrouiller'}
              >
                <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                  {element.locked ? (
                    <path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3.1-9H8.9V6c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2z" />
                  ) : (
                    <path d="M12 17c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm6-9h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6h1.9c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm0 12H6V10h12v10z" />
                  )}
                </svg>
              </button>

              {/* Delete */}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete();
                }}
                className="p-1 text-gray-400 hover:text-red-500 rounded transition-colors"
                title="Supprimer"
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </Reorder.Item>
  );
}

export function LayersPanel({
  elements,
  selectedElementId,
  onSelectElement,
  onReorderElements,
  onToggleVisibility,
  onToggleLock,
  onDeleteElement,
  onDuplicateElement,
  onBringToFront,
  onSendToBack,
  disabled = false,
}: LayersPanelProps) {
  // Sort elements by z-index (highest first for layer panel display)
  const sortedElements = [...elements].sort((a, b) => b.zIndex - a.zIndex);

  // Handle reorder from drag
  const handleReorder = useCallback((newOrder: SlideElement[]) => {
    // Reassign z-indices based on new order (reversed because UI shows highest first)
    const updatedElements = newOrder.map((el, index) => ({
      ...el,
      zIndex: newOrder.length - index,
    }));
    onReorderElements(updatedElements);
  }, [onReorderElements]);

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="px-3 py-2 border-b border-gray-800 flex items-center justify-between">
        <h3 className="text-xs font-medium text-gray-300 uppercase tracking-wide">Calques</h3>
        <span className="text-xs text-gray-500">{elements.length}</span>
      </div>

      {/* Quick actions for selected element */}
      {selectedElementId && (
        <div className="px-3 py-2 border-b border-gray-800 flex items-center gap-1">
          {onBringToFront && (
            <button
              onClick={() => onBringToFront(selectedElementId)}
              className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
              title="Premier plan"
              disabled={disabled}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 11l7-7 7 7M5 19l7-7 7 7" />
              </svg>
            </button>
          )}
          {onSendToBack && (
            <button
              onClick={() => onSendToBack(selectedElementId)}
              className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
              title="Arrière-plan"
              disabled={disabled}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 13l-7 7-7-7m14-8l-7 7-7-7" />
              </svg>
            </button>
          )}
          <div className="flex-1" />
          {onDuplicateElement && (
            <button
              onClick={() => onDuplicateElement(selectedElementId)}
              className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
              title="Dupliquer"
              disabled={disabled}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            </button>
          )}
        </div>
      )}

      {/* Layers list */}
      <div className="flex-1 overflow-y-auto p-2">
        {elements.length === 0 ? (
          <div className="text-center py-8">
            <svg
              className="w-10 h-10 mx-auto mb-2 text-gray-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            <p className="text-sm text-gray-500">Aucun calque</p>
            <p className="text-xs text-gray-600 mt-1">
              Ajoutez des éléments au canvas
            </p>
          </div>
        ) : (
          <Reorder.Group
            axis="y"
            values={sortedElements}
            onReorder={handleReorder}
            className="space-y-1"
          >
            {sortedElements.map((element) => (
              <LayerItem
                key={element.id}
                element={element}
                isSelected={selectedElementId === element.id}
                onSelect={() => onSelectElement(element.id)}
                onToggleVisibility={() => onToggleVisibility(element.id)}
                onToggleLock={() => onToggleLock(element.id)}
                onDelete={() => onDeleteElement(element.id)}
                disabled={disabled}
              />
            ))}
          </Reorder.Group>
        )}
      </div>

      {/* Footer info */}
      <div className="px-3 py-2 border-t border-gray-800">
        <p className="text-xs text-gray-600">
          Glissez pour réordonner • Double-clic pour renommer
        </p>
      </div>
    </div>
  );
}

export default LayersPanel;
