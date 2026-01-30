'use client';

import React, { useRef, useCallback, useState, memo, useMemo } from 'react';
import type { SlideComponent, SlideElement, AddElementRequest, UpdateElementRequest } from '../../../lib/lecture-editor-types';
import { SelectableElement } from './SelectableElement';
import { AlignmentGuides } from './AlignmentGuides';
import { QuickInsertBar } from './QuickInsertBar';
import { ElementPropertiesPanel } from './ElementPropertiesPanel';
import { ElementContextMenu } from './ElementContextMenu';
import { useCanvasInteraction } from './useCanvasInteraction';

interface InteractiveCanvasProps {
  slide: SlideComponent;
  // Element callbacks
  onAddElement: (request: AddElementRequest) => Promise<SlideElement | null>;
  onUpdateElement: (elementId: string, updates: UpdateElementRequest) => Promise<SlideElement | null>;
  onDeleteElement: (elementId: string) => Promise<boolean>;
  onUploadImage: (file: File, position?: { x: number; y: number }) => Promise<SlideElement | null>;
  onDuplicateElement?: (element: SlideElement) => Promise<SlideElement | null>;
  onBringToFront?: (elementId: string) => Promise<boolean>;
  onSendToBack?: (elementId: string) => Promise<boolean>;
  // State
  isLoading?: boolean;
  disabled?: boolean;
}

export const InteractiveCanvas = memo(function InteractiveCanvas({
  slide,
  onAddElement,
  onUpdateElement,
  onDeleteElement,
  onUploadImage,
  onDuplicateElement,
  onBringToFront,
  onSendToBack,
  isLoading = false,
  disabled = false,
}: InteractiveCanvasProps) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const [selectedElementId, setSelectedElementId] = useState<string | null>(null);
  const [pendingUpdates, setPendingUpdates] = useState<Map<string, UpdateElementRequest>>(new Map());
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number } | null>(null);

  // Debounced update to avoid too many API calls during drag
  const handleUpdateElement = useCallback(async (elementId: string, updates: UpdateElementRequest) => {
    // Optimistic local update (already done in useCanvasInteraction)
    // Batch API call with debounce
    setPendingUpdates((prev) => {
      const newMap = new Map(prev);
      const existing = newMap.get(elementId) || {};
      newMap.set(elementId, { ...existing, ...updates });
      return newMap;
    });

    // Simple debounce - flush after interaction ends (handled by mouseup)
  }, []);

  // Flush pending updates
  const flushUpdates = useCallback(async () => {
    for (const [elementId, updates] of pendingUpdates.entries()) {
      await onUpdateElement(elementId, updates);
    }
    setPendingUpdates(new Map());
  }, [pendingUpdates, onUpdateElement]);

  // Handle delete
  const handleDeleteElement = useCallback(async (elementId: string) => {
    const success = await onDeleteElement(elementId);
    if (success && selectedElementId === elementId) {
      setSelectedElementId(null);
    }
  }, [onDeleteElement, selectedElementId]);

  // Wrapper for duplicate that handles the async result
  const handleDuplicateElement = useCallback(async (element: SlideElement) => {
    if (!onDuplicateElement) return;
    const newElement = await onDuplicateElement(element);
    if (newElement) {
      setSelectedElementId(newElement.id);
    }
  }, [onDuplicateElement]);

  // Wrappers for depth control
  const handleBringToFront = useCallback(async (elementId: string) => {
    if (!onBringToFront) return;
    await onBringToFront(elementId);
  }, [onBringToFront]);

  const handleSendToBack = useCallback(async (elementId: string) => {
    if (!onSendToBack) return;
    await onSendToBack(elementId);
  }, [onSendToBack]);

  // Toggle element lock
  const handleToggleLock = useCallback(async (elementId: string) => {
    const element = slide.elements.find((e) => e.id === elementId);
    if (!element) return;
    await onUpdateElement(elementId, { locked: !element.locked });
  }, [slide.elements, onUpdateElement]);

  // Open context menu
  const handleContextMenu = useCallback((elementId: string, e: React.MouseEvent) => {
    e.preventDefault();
    setSelectedElementId(elementId);
    setContextMenu({ x: e.clientX, y: e.clientY });
  }, []);

  // Close context menu
  const closeContextMenu = useCallback(() => {
    setContextMenu(null);
  }, []);

  // Delete element (for context menu)
  const handleDeleteFromContextMenu = useCallback(async () => {
    if (!selectedElementId) return;
    await handleDeleteElement(selectedElementId);
  }, [selectedElementId, handleDeleteElement]);

  // Canvas interaction hook
  const {
    startDrag,
    startResize,
    startRotate,
    handleCanvasClick,
    alignmentGuides,
    isDragging,
    isResizing,
    isRotating,
    hasClipboard,
    copyElement,
    pasteElement,
    duplicateElement: duplicateElementAction,
  } = useCanvasInteraction({
    elements: slide.elements,
    selectedElementId,
    canvasRef,
    onSelectElement: setSelectedElementId,
    onUpdateElement: handleUpdateElement,
    onDeleteElement: handleDeleteElement,
    onDuplicateElement: handleDuplicateElement,
    onBringToFront: handleBringToFront,
    onSendToBack: handleSendToBack,
    disabled: disabled || isLoading,
  });

  // Flush updates when interaction ends
  React.useEffect(() => {
    if (!isDragging && !isResizing && !isRotating && pendingUpdates.size > 0) {
      flushUpdates();
    }
  }, [isDragging, isResizing, isRotating, pendingUpdates.size, flushUpdates]);

  // Insert handlers
  const handleInsertImage = useCallback(async (file: File) => {
    const element = await onUploadImage(file);
    if (element) {
      setSelectedElementId(element.id);
    }
  }, [onUploadImage]);

  const handleInsertText = useCallback(async () => {
    const element = await onAddElement({
      type: 'text_block',
      x: 30,
      y: 40,
      width: 40,
      height: 20,
      textContent: {
        text: 'Nouveau texte',
        fontSize: 24,
        fontWeight: 'normal',
        fontFamily: 'Inter',
        color: '#FFFFFF',
        textAlign: 'center',
        lineHeight: 1.4,
        padding: 12,
      },
    });
    if (element) {
      setSelectedElementId(element.id);
    }
  }, [onAddElement]);

  const handleInsertShape = useCallback(async (shape: 'rectangle' | 'circle' | 'rounded_rect') => {
    const element = await onAddElement({
      type: 'shape',
      x: 35,
      y: 35,
      width: shape === 'circle' ? 20 : 30,
      height: 20,
      shapeContent: {
        shape,
        fillColor: '#6366F1',
        opacity: 0.9,
        strokeWidth: 0,
        borderRadius: shape === 'rounded_rect' ? 12 : 0,
      },
    });
    if (element) {
      setSelectedElementId(element.id);
    }
  }, [onAddElement]);

  // Handle drop for image files
  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/') && canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 100 - 15; // Center the 30% wide image
      const y = ((e.clientY - rect.top) / rect.height) * 100 - 15;
      const element = await onUploadImage(file, { x: Math.max(0, x), y: Math.max(0, y) });
      if (element) {
        setSelectedElementId(element.id);
      }
    }
  }, [onUploadImage]);

  // Sort elements by z-index for rendering
  const sortedElements = [...slide.elements].sort((a, b) => a.zIndex - b.zIndex);

  // Get selected element
  const selectedElement = useMemo(() => {
    if (!selectedElementId) return null;
    return slide.elements.find((e) => e.id === selectedElementId) || null;
  }, [selectedElementId, slide.elements]);

  // State for properties panel visibility
  const [showProperties, setShowProperties] = useState(true);

  // Handle text change from inline editing
  const handleTextChange = useCallback(async (elementId: string, text: string) => {
    const element = slide.elements.find((e) => e.id === elementId);
    if (!element || element.type !== 'text_block' || !element.textContent) return;

    await onUpdateElement(elementId, {
      textContent: {
        ...element.textContent,
        text,
      },
    });
  }, [slide.elements, onUpdateElement]);

  // Handle property changes from panel (direct API call, no debounce)
  const handlePropertyChange = useCallback(async (updates: UpdateElementRequest) => {
    if (!selectedElementId) return;
    await onUpdateElement(selectedElementId, updates);
  }, [selectedElementId, onUpdateElement]);

  return (
    <div className="relative w-full h-full flex">
      {/* Main canvas area */}
      <div className="relative flex-1 flex flex-col">
        {/* Canvas */}
        <div
          ref={canvasRef}
          className={`
            relative flex-1 bg-gray-900 rounded-lg overflow-hidden
            ${isDragging || isResizing ? 'cursor-grabbing' : ''}
            ${isRotating ? 'cursor-grabbing' : ''}
            ${disabled ? 'opacity-75 pointer-events-none' : ''}
          `}
          onClick={(e) => {
            handleCanvasClick(e);
            closeContextMenu();
          }}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          {/* Slide background image */}
          {slide.imageUrl && (
            <img
              src={slide.imageUrl}
              alt=""
              className="absolute inset-0 w-full h-full object-contain pointer-events-none"
              draggable={false}
            />
          )}

          {/* Elements layer */}
          <div className="absolute inset-0">
            {sortedElements.map((element) => (
              <SelectableElement
                key={element.id}
                element={element}
                isSelected={selectedElementId === element.id}
                onSelect={(e) => {
                  e.stopPropagation();
                  setSelectedElementId(element.id);
                  closeContextMenu();
                }}
                onStartDrag={(e) => startDrag(element.id, e)}
                onStartResize={(handle, e) => startResize(element.id, handle, e)}
                onStartRotate={(e) => startRotate(element.id, e)}
                onTextChange={(text) => handleTextChange(element.id, text)}
                onToggleLock={() => handleToggleLock(element.id)}
                onContextMenu={(e) => handleContextMenu(element.id, e)}
                disabled={disabled || isLoading}
              />
            ))}
          </div>

          {/* Alignment guides */}
          <AlignmentGuides guides={alignmentGuides} />

          {/* Loading overlay */}
          {isLoading && (
            <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-50">
              <div className="flex items-center gap-2 text-white">
                <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <span className="text-sm">Chargement...</span>
              </div>
            </div>
          )}

          {/* Empty state hint */}
          {slide.elements.length === 0 && !slide.imageUrl && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="text-center text-gray-500">
                <svg className="w-12 h-12 mx-auto mb-2 opacity-30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <p className="text-sm">Glissez une image ou utilisez la barre ci-dessous</p>
              </div>
            </div>
          )}

          {/* Quick insert bar - always visible at bottom */}
          <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-40">
            <QuickInsertBar
              onInsertImage={handleInsertImage}
              onInsertText={handleInsertText}
              onInsertShape={handleInsertShape}
              disabled={disabled || isLoading}
            />
          </div>

          {/* Toggle properties panel button */}
          <button
            onClick={() => setShowProperties(!showProperties)}
            className={`absolute top-2 right-2 p-1.5 rounded-lg transition-colors z-40 ${
              showProperties
                ? 'bg-purple-600 text-white'
                : 'bg-gray-800/90 text-gray-400 hover:text-white'
            }`}
            title={showProperties ? 'Masquer les propriétés' : 'Afficher les propriétés'}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
            </svg>
          </button>

          {/* Element action bar */}
          {selectedElementId && (
            <div className="absolute top-2 left-2 flex items-center gap-1 z-40">
              {/* Copy */}
              <button
                onClick={copyElement}
                className="p-1.5 bg-gray-800/90 hover:bg-gray-700 text-gray-400 hover:text-white rounded transition-colors"
                title="Copier (Ctrl+C)"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </button>

              {/* Paste */}
              <button
                onClick={pasteElement}
                disabled={!hasClipboard}
                className="p-1.5 bg-gray-800/90 hover:bg-gray-700 text-gray-400 hover:text-white rounded transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                title="Coller (Ctrl+V)"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </button>

              {/* Duplicate */}
              <button
                onClick={duplicateElementAction}
                className="p-1.5 bg-gray-800/90 hover:bg-gray-700 text-gray-400 hover:text-white rounded transition-colors"
                title="Dupliquer (Ctrl+D)"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
              </button>

              <div className="w-px h-4 bg-gray-700 mx-1" />

              {/* Bring to front */}
              <button
                onClick={() => selectedElementId && handleBringToFront(selectedElementId)}
                className="p-1.5 bg-gray-800/90 hover:bg-gray-700 text-gray-400 hover:text-white rounded transition-colors"
                title="Premier plan (Ctrl+])"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 11l7-7 7 7M5 19l7-7 7 7" />
                </svg>
              </button>

              {/* Send to back */}
              <button
                onClick={() => selectedElementId && handleSendToBack(selectedElementId)}
                className="p-1.5 bg-gray-800/90 hover:bg-gray-700 text-gray-400 hover:text-white rounded transition-colors"
                title="Arrière-plan (Ctrl+[)"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 13l-7 7-7-7m14-8l-7 7-7-7" />
                </svg>
              </button>
            </div>
          )}

          {/* Keyboard hints */}
          {selectedElementId && (
            <div className="absolute bottom-16 left-1/2 transform -translate-x-1/2 bg-gray-800/80 backdrop-blur rounded-lg px-3 py-1.5 text-xs text-gray-400 z-30 flex items-center gap-3">
              <span><kbd className="px-1 py-0.5 bg-gray-700 rounded text-gray-300">Suppr</kbd> effacer</span>
              <span><kbd className="px-1 py-0.5 bg-gray-700 rounded text-gray-300">↑↓←→</kbd> déplacer</span>
              {selectedElement?.type === 'text_block' && (
                <span><kbd className="px-1 py-0.5 bg-gray-700 rounded text-gray-300">Double-clic</kbd> éditer</span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Properties panel */}
      {showProperties && (
        <div className="w-56 bg-gray-900/95 border-l border-gray-800 overflow-y-auto flex-shrink-0">
          <div className="sticky top-0 bg-gray-900 border-b border-gray-800 px-3 py-2 flex items-center justify-between">
            <h3 className="text-xs font-medium text-gray-300 uppercase tracking-wide">Propriétés</h3>
            <button
              onClick={() => setShowProperties(false)}
              className="p-1 text-gray-500 hover:text-white transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <ElementPropertiesPanel
            element={selectedElement}
            onUpdate={handlePropertyChange}
            disabled={disabled || isLoading}
          />
        </div>
      )}

      {/* Context menu */}
      {contextMenu && selectedElement && (
        <ElementContextMenu
          position={contextMenu}
          onClose={closeContextMenu}
          onCopy={copyElement}
          onPaste={pasteElement}
          onDuplicate={duplicateElementAction}
          onDelete={handleDeleteFromContextMenu}
          onBringToFront={() => selectedElementId && handleBringToFront(selectedElementId)}
          onSendToBack={() => selectedElementId && handleSendToBack(selectedElementId)}
          onToggleLock={() => selectedElementId && handleToggleLock(selectedElementId)}
          isLocked={selectedElement.locked}
          hasClipboard={hasClipboard}
        />
      )}
    </div>
  );
});

export default InteractiveCanvas;
