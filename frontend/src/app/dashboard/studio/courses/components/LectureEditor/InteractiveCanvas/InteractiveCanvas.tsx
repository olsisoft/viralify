'use client';

import React, { useRef, useCallback, useState, memo, useMemo, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { SlideComponent, SlideElement, AddElementRequest, UpdateElementRequest } from '../../../lib/lecture-editor-types';
import { SelectableElement } from './SelectableElement';
import { AlignmentGuides } from './AlignmentGuides';
import { QuickInsertBar } from './QuickInsertBar';
import { ElementPropertiesPanel } from './ElementPropertiesPanel';
import { ElementContextMenu } from './ElementContextMenu';
import { AlignmentToolbar } from './AlignmentToolbar';
import { ZoomControls } from './ZoomControls';
import { useCanvasInteraction } from './useCanvasInteraction';
import { useCanvasZoom } from './useCanvasZoom';
import { useHistory } from './useHistory';

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
  onRestoreElements?: (elements: SlideElement[]) => Promise<void>;
  // State
  isLoading?: boolean;
  disabled?: boolean;
}

// Drop zone feedback states
type DropZoneState = 'idle' | 'active' | 'valid' | 'invalid';

// Marquee selection state
interface MarqueeState {
  isActive: boolean;
  startX: number;
  startY: number;
  currentX: number;
  currentY: number;
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
  onRestoreElements,
  isLoading = false,
  disabled = false,
}: InteractiveCanvasProps) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const canvasContainerRef = useRef<HTMLDivElement>(null);

  // Multi-selection state
  const [selectedElementIds, setSelectedElementIds] = useState<Set<string>>(new Set());
  const [pendingUpdates, setPendingUpdates] = useState<Map<string, UpdateElementRequest>>(new Map());
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number } | null>(null);
  const [dropZoneState, setDropZoneState] = useState<DropZoneState>('idle');
  const [dropPosition, setDropPosition] = useState<{ x: number; y: number } | null>(null);
  const [showProperties, setShowProperties] = useState(true);

  // Marquee selection
  const [marquee, setMarquee] = useState<MarqueeState>({
    isActive: false,
    startX: 0,
    startY: 0,
    currentX: 0,
    currentY: 0,
  });

  // Zoom & Pan
  const {
    scale,
    panX,
    panY,
    isPanning,
    isSpacePressed,
    zoomIn,
    zoomOut,
    resetZoom,
    setZoom,
    transformStyle,
    zoomPresets,
  } = useCanvasZoom({
    canvasRef: canvasContainerRef,
    disabled: disabled || isLoading,
  });

  // History (Undo/Redo)
  const {
    canUndo,
    canRedo,
    notification: historyNotification,
    saveState,
    undo,
    redo,
  } = useHistory({
    elements: slide.elements,
    onRestore: onRestoreElements || (async () => {}),
  });

  // Save state before making changes
  const saveHistoryState = useCallback((description: string) => {
    saveState(description);
  }, [saveState]);

  // Helper for single selection compatibility
  const selectedElementId = useMemo(() => {
    const ids = Array.from(selectedElementIds);
    return ids.length === 1 ? ids[0] : null;
  }, [selectedElementIds]);

  // Get selected elements
  const selectedElements = useMemo(() => {
    return slide.elements.filter((e) => selectedElementIds.has(e.id));
  }, [slide.elements, selectedElementIds]);

  // Select single element
  const selectElement = useCallback((elementId: string | null, addToSelection = false) => {
    if (elementId === null) {
      setSelectedElementIds(new Set());
    } else if (addToSelection) {
      setSelectedElementIds((prev) => {
        const next = new Set(prev);
        if (next.has(elementId)) {
          next.delete(elementId);
        } else {
          next.add(elementId);
        }
        return next;
      });
    } else {
      setSelectedElementIds(new Set([elementId]));
    }
  }, []);

  // Debounced update to avoid too many API calls during drag
  const handleUpdateElement = useCallback(async (elementId: string, updates: UpdateElementRequest) => {
    setPendingUpdates((prev) => {
      const newMap = new Map(prev);
      const existing = newMap.get(elementId) || {};
      newMap.set(elementId, { ...existing, ...updates });
      return newMap;
    });
  }, []);

  // Flush pending updates
  const flushUpdates = useCallback(async () => {
    const entries = Array.from(pendingUpdates.entries());
    for (const [elementId, updates] of entries) {
      await onUpdateElement(elementId, updates);
    }
    setPendingUpdates(new Map());
  }, [pendingUpdates, onUpdateElement]);

  // Handle delete
  const handleDeleteElement = useCallback(async (elementId: string) => {
    saveHistoryState('Supprimer élément');
    const success = await onDeleteElement(elementId);
    if (success) {
      setSelectedElementIds((prev) => {
        const next = new Set(prev);
        next.delete(elementId);
        return next;
      });
    }
  }, [onDeleteElement, saveHistoryState]);

  // Delete all selected elements
  const handleDeleteSelected = useCallback(async () => {
    if (selectedElementIds.size === 0) return;
    saveHistoryState('Supprimer éléments');
    for (const id of selectedElementIds) {
      await onDeleteElement(id);
    }
    setSelectedElementIds(new Set());
  }, [selectedElementIds, onDeleteElement, saveHistoryState]);

  // Wrapper for duplicate that handles the async result
  const handleDuplicateElement = useCallback(async (element: SlideElement) => {
    if (!onDuplicateElement) return;
    saveHistoryState('Dupliquer élément');
    const newElement = await onDuplicateElement(element);
    if (newElement) {
      setSelectedElementIds(new Set([newElement.id]));
    }
  }, [onDuplicateElement, saveHistoryState]);

  // Wrappers for depth control
  const handleBringToFront = useCallback(async (elementId: string) => {
    if (!onBringToFront) return;
    saveHistoryState('Premier plan');
    await onBringToFront(elementId);
  }, [onBringToFront, saveHistoryState]);

  const handleSendToBack = useCallback(async (elementId: string) => {
    if (!onSendToBack) return;
    saveHistoryState('Arrière-plan');
    await onSendToBack(elementId);
  }, [onSendToBack, saveHistoryState]);

  // Toggle element lock
  const handleToggleLock = useCallback(async (elementId: string) => {
    const element = slide.elements.find((e) => e.id === elementId);
    if (!element) return;
    await onUpdateElement(elementId, { locked: !element.locked });
  }, [slide.elements, onUpdateElement]);

  // Open context menu
  const handleContextMenu = useCallback((elementId: string, e: React.MouseEvent) => {
    e.preventDefault();
    selectElement(elementId);
    setContextMenu({ x: e.clientX, y: e.clientY });
  }, [selectElement]);

  // Close context menu
  const closeContextMenu = useCallback(() => {
    setContextMenu(null);
  }, []);

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
    isInteracting,
    hasClipboard,
    copyElement,
    pasteElement,
    duplicateElement: duplicateElementAction,
  } = useCanvasInteraction({
    elements: slide.elements,
    selectedElementId,
    canvasRef,
    onSelectElement: (id) => selectElement(id),
    onUpdateElement: handleUpdateElement,
    onDeleteElement: handleDeleteElement,
    onDuplicateElement: handleDuplicateElement,
    onBringToFront: handleBringToFront,
    onSendToBack: handleSendToBack,
    disabled: disabled || isLoading || isPanning,
  });

  // Flush updates when interaction ends
  useEffect(() => {
    if (!isDragging && !isResizing && !isRotating && pendingUpdates.size > 0) {
      flushUpdates();
    }
  }, [isDragging, isResizing, isRotating, pendingUpdates.size, flushUpdates]);

  // Marquee selection handlers
  const handleMarqueeStart = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0 || isSpacePressed || disabled) return;

    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = (e.clientX - rect.left) / scale;
    const y = (e.clientY - rect.top) / scale;

    setMarquee({
      isActive: true,
      startX: x,
      startY: y,
      currentX: x,
      currentY: y,
    });
  }, [isSpacePressed, disabled, scale]);

  const handleMarqueeMove = useCallback((e: MouseEvent) => {
    if (!marquee.isActive) return;

    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = (e.clientX - rect.left) / scale;
    const y = (e.clientY - rect.top) / scale;

    setMarquee((prev) => ({
      ...prev,
      currentX: x,
      currentY: y,
    }));
  }, [marquee.isActive, scale]);

  const handleMarqueeEnd = useCallback(() => {
    if (!marquee.isActive) return;

    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    // Calculate marquee bounds in percentage
    const left = Math.min(marquee.startX, marquee.currentX) / rect.width * 100;
    const right = Math.max(marquee.startX, marquee.currentX) / rect.width * 100;
    const top = Math.min(marquee.startY, marquee.currentY) / rect.height * 100;
    const bottom = Math.max(marquee.startY, marquee.currentY) / rect.height * 100;

    // Find elements within marquee
    const selected = slide.elements.filter((element) => {
      const elemLeft = element.x;
      const elemRight = element.x + element.width;
      const elemTop = element.y;
      const elemBottom = element.y + element.height;

      return (
        elemLeft < right &&
        elemRight > left &&
        elemTop < bottom &&
        elemBottom > top
      );
    });

    if (selected.length > 0) {
      setSelectedElementIds(new Set(selected.map((e) => e.id)));
    }

    setMarquee({
      isActive: false,
      startX: 0,
      startY: 0,
      currentX: 0,
      currentY: 0,
    });
  }, [marquee, slide.elements]);

  // Marquee mouse event listeners
  useEffect(() => {
    if (!marquee.isActive) return;

    window.addEventListener('mousemove', handleMarqueeMove);
    window.addEventListener('mouseup', handleMarqueeEnd);

    return () => {
      window.removeEventListener('mousemove', handleMarqueeMove);
      window.removeEventListener('mouseup', handleMarqueeEnd);
    };
  }, [marquee.isActive, handleMarqueeMove, handleMarqueeEnd]);

  // Update multiple elements (for alignment)
  const handleUpdateMultipleElements = useCallback(async (
    updates: Array<{ elementId: string; updates: UpdateElementRequest }>
  ) => {
    saveHistoryState('Aligner éléments');
    for (const { elementId, updates: u } of updates) {
      await onUpdateElement(elementId, u);
    }
  }, [onUpdateElement, saveHistoryState]);

  // Insert handlers
  const handleInsertImage = useCallback(async (file: File) => {
    saveHistoryState('Ajouter image');
    const element = await onUploadImage(file);
    if (element) {
      setSelectedElementIds(new Set([element.id]));
    }
  }, [onUploadImage, saveHistoryState]);

  const handleInsertText = useCallback(async () => {
    saveHistoryState('Ajouter texte');
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
      setSelectedElementIds(new Set([element.id]));
    }
  }, [onAddElement, saveHistoryState]);

  const handleInsertShape = useCallback(async (shape: 'rectangle' | 'circle' | 'rounded_rect') => {
    saveHistoryState('Ajouter forme');
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
      setSelectedElementIds(new Set([element.id]));
    }
  }, [onAddElement, saveHistoryState]);

  // Calculate drop position from event
  const getDropPosition = useCallback((e: React.DragEvent): { x: number; y: number } => {
    if (!canvasRef.current) return { x: 35, y: 35 };
    const rect = canvasRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100 - 15;
    const y = ((e.clientY - rect.top) / rect.height) * 100 - 15;
    return {
      x: Math.max(0, Math.min(70, x)),
      y: Math.max(0, Math.min(70, y)),
    };
  }, []);

  // Handle drag enter from asset library
  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const hasAssetData = e.dataTransfer.types.includes('application/json');
    const hasFiles = e.dataTransfer.types.includes('Files');

    if (hasAssetData || hasFiles) {
      setDropZoneState('active');
      setDropPosition(getDropPosition(e));
    }
  }, [getDropPosition]);

  // Handle drag over
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    setDropPosition(getDropPosition(e));

    const hasAssetData = e.dataTransfer.types.includes('application/json');
    const hasFiles = e.dataTransfer.types.includes('Files');

    if (hasAssetData || hasFiles) {
      e.dataTransfer.dropEffect = 'copy';
      setDropZoneState('valid');
    } else {
      e.dataTransfer.dropEffect = 'none';
      setDropZoneState('invalid');
    }
  }, [getDropPosition]);

  // Handle drag leave
  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const rect = canvasRef.current?.getBoundingClientRect();
    if (rect) {
      const { clientX, clientY } = e;
      if (
        clientX < rect.left ||
        clientX > rect.right ||
        clientY < rect.top ||
        clientY > rect.bottom
      ) {
        setDropZoneState('idle');
        setDropPosition(null);
      }
    }
  }, []);

  // Handle drop for image files or assets from library
  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    setDropZoneState('idle');
    setDropPosition(null);

    const position = getDropPosition(e);

    // Check for asset data from library
    const assetDataStr = e.dataTransfer.getData('application/json');
    if (assetDataStr) {
      try {
        const assetData = JSON.parse(assetDataStr);
        if (assetData.type === 'asset' && assetData.assetType === 'image') {
          saveHistoryState('Ajouter image');
          const element = await onAddElement({
            type: 'image',
            x: position.x,
            y: position.y,
            width: 30,
            height: 30,
            imageContent: {
              url: assetData.url,
              fit: 'cover',
              opacity: 1,
              borderRadius: 0,
            },
          });
          if (element) {
            setSelectedElementIds(new Set([element.id]));
          }
          return;
        }
      } catch (err) {
        console.warn('Failed to parse asset data:', err);
      }
    }

    // Handle file drop
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      saveHistoryState('Ajouter image');
      const element = await onUploadImage(file, position);
      if (element) {
        setSelectedElementIds(new Set([element.id]));
      }
    }
  }, [getDropPosition, onAddElement, onUploadImage, saveHistoryState]);

  // Sort elements by z-index for rendering
  const sortedElements = [...slide.elements].sort((a, b) => a.zIndex - b.zIndex);

  // Get first selected element for properties panel
  const selectedElement = useMemo(() => {
    if (selectedElementIds.size === 0) return null;
    const firstId = Array.from(selectedElementIds)[0];
    return slide.elements.find((e) => e.id === firstId) || null;
  }, [selectedElementIds, slide.elements]);

  // Handle text change from inline editing
  const handleTextChange = useCallback(async (elementId: string, text: string) => {
    const element = slide.elements.find((e) => e.id === elementId);
    if (!element || element.type !== 'text_block' || !element.textContent) return;

    saveHistoryState('Modifier texte');
    await onUpdateElement(elementId, {
      textContent: {
        ...element.textContent,
        text,
      },
    });
  }, [slide.elements, onUpdateElement, saveHistoryState]);

  // Handle property changes from panel
  const handlePropertyChange = useCallback(async (updates: UpdateElementRequest) => {
    if (!selectedElementId) return;
    saveHistoryState('Modifier propriétés');
    await onUpdateElement(selectedElementId, updates);
  }, [selectedElementId, onUpdateElement, saveHistoryState]);

  // Marquee rect calculation
  const marqueeRect = useMemo(() => {
    if (!marquee.isActive) return null;
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return null;

    return {
      left: Math.min(marquee.startX, marquee.currentX),
      top: Math.min(marquee.startY, marquee.currentY),
      width: Math.abs(marquee.currentX - marquee.startX),
      height: Math.abs(marquee.currentY - marquee.startY),
    };
  }, [marquee]);

  return (
    <div className="relative w-full h-full flex">
      {/* Main canvas area */}
      <div className="relative flex-1 flex flex-col overflow-hidden">
        {/* Top toolbar */}
        <div className="absolute top-2 left-2 right-2 z-50 flex items-center justify-between pointer-events-none">
          {/* Left: Undo/Redo and element actions */}
          <div className="flex items-center gap-2 pointer-events-auto">
            {/* Undo/Redo buttons */}
            <div className="flex items-center gap-1 bg-gray-800/95 backdrop-blur rounded-lg p-1 border border-gray-700">
              <button
                onClick={undo}
                disabled={!canUndo || disabled}
                className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                title="Annuler (Ctrl+Z)"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
                </svg>
              </button>
              <button
                onClick={redo}
                disabled={!canRedo || disabled}
                className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                title="Rétablir (Ctrl+Y)"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 10h-10a8 8 0 00-8 8v2M21 10l-6 6m6-6l-6-6" />
                </svg>
              </button>
            </div>

            {/* Element actions when selected */}
            <AnimatePresence>
              {selectedElementIds.size > 0 && (
                <motion.div
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  className="flex items-center gap-2"
                >
                  {/* Alignment toolbar */}
                  <AlignmentToolbar
                    selectedElements={selectedElements}
                    onUpdateElements={handleUpdateMultipleElements}
                    disabled={disabled}
                  />

                  {/* Copy/Paste/Duplicate */}
                  <div className="flex items-center gap-1 bg-gray-800/95 backdrop-blur rounded-lg p-1 border border-gray-700">
                    <button
                      onClick={copyElement}
                      className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
                      title="Copier (Ctrl+C)"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                    </button>
                    <button
                      onClick={pasteElement}
                      disabled={!hasClipboard}
                      className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors disabled:opacity-30"
                      title="Coller (Ctrl+V)"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                      </svg>
                    </button>
                    <button
                      onClick={duplicateElementAction}
                      className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
                      title="Dupliquer (Ctrl+D)"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                    </button>

                    <div className="w-px h-5 bg-gray-700 mx-0.5" />

                    {/* Z-index controls */}
                    <button
                      onClick={() => selectedElementId && handleBringToFront(selectedElementId)}
                      className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
                      title="Premier plan"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 11l7-7 7 7M5 19l7-7 7 7" />
                      </svg>
                    </button>
                    <button
                      onClick={() => selectedElementId && handleSendToBack(selectedElementId)}
                      className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
                      title="Arrière-plan"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 13l-7 7-7-7m14-8l-7 7-7-7" />
                      </svg>
                    </button>

                    <div className="w-px h-5 bg-gray-700 mx-0.5" />

                    {/* Delete */}
                    <button
                      onClick={handleDeleteSelected}
                      className="p-1.5 text-red-400 hover:text-red-300 hover:bg-red-500/20 rounded transition-colors"
                      title="Supprimer (Suppr)"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Right: Zoom controls and properties toggle */}
          <div className="flex items-center gap-2 pointer-events-auto">
            <ZoomControls
              scale={scale}
              onZoomIn={zoomIn}
              onZoomOut={zoomOut}
              onResetZoom={resetZoom}
              onSetZoom={setZoom}
              zoomPresets={zoomPresets}
              disabled={disabled}
            />

            <button
              onClick={() => setShowProperties(!showProperties)}
              className={`p-1.5 rounded-lg transition-colors ${
                showProperties
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-800/95 border border-gray-700 text-gray-400 hover:text-white'
              }`}
              title={showProperties ? 'Masquer propriétés' : 'Afficher propriétés'}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
              </svg>
            </button>
          </div>
        </div>

        {/* Canvas container with zoom/pan */}
        <div
          ref={canvasContainerRef}
          className={`
            relative flex-1 overflow-hidden bg-gray-950
            ${isSpacePressed ? 'cursor-grab' : ''}
            ${isPanning ? 'cursor-grabbing' : ''}
          `}
        >
          {/* Zoomable/Pannable canvas */}
          <div
            ref={canvasRef}
            className={`
              relative w-full h-full bg-gray-900 rounded-lg overflow-hidden transition-shadow
              ${isDragging || isResizing ? 'cursor-grabbing' : ''}
              ${isRotating ? 'cursor-grabbing' : ''}
              ${disabled ? 'opacity-75 pointer-events-none' : ''}
              ${dropZoneState === 'valid' ? 'ring-2 ring-purple-500' : ''}
              ${dropZoneState === 'active' ? 'ring-2 ring-purple-400/50' : ''}
            `}
            style={transformStyle}
            onClick={(e) => {
              if (e.target === canvasRef.current) {
                selectElement(null);
                closeContextMenu();
              }
            }}
            onMouseDown={handleMarqueeStart}
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
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
                  isSelected={selectedElementIds.has(element.id)}
                  onSelect={(e) => {
                    e.stopPropagation();
                    selectElement(element.id, e.shiftKey);
                    closeContextMenu();
                  }}
                  onStartDrag={(e) => startDrag(element.id, e)}
                  onStartResize={(handle, e) => startResize(element.id, handle, e)}
                  onStartRotate={(e) => startRotate(element.id, e)}
                  onTextChange={(text) => handleTextChange(element.id, text)}
                  onToggleLock={() => handleToggleLock(element.id)}
                  onContextMenu={(e) => handleContextMenu(element.id, e)}
                  disabled={disabled || isLoading || isPanning}
                />
              ))}
            </div>

            {/* Marquee selection rectangle */}
            {marqueeRect && (
              <div
                className="absolute border-2 border-purple-500 bg-purple-500/10 pointer-events-none z-50"
                style={{
                  left: marqueeRect.left,
                  top: marqueeRect.top,
                  width: marqueeRect.width,
                  height: marqueeRect.height,
                }}
              />
            )}

            {/* Alignment guides */}
            <AlignmentGuides guides={alignmentGuides} />

            {/* Drop zone overlay */}
            <AnimatePresence>
              {dropZoneState !== 'idle' && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="absolute inset-0 pointer-events-none z-40"
                >
                  <div
                    className={`absolute inset-0 ${
                      dropZoneState === 'valid'
                        ? 'bg-purple-500/10'
                        : dropZoneState === 'invalid'
                          ? 'bg-red-500/10'
                          : 'bg-gray-500/10'
                    }`}
                  />

                  {dropPosition && dropZoneState === 'valid' && (
                    <motion.div
                      initial={{ scale: 0.8, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      className="absolute w-[30%] h-[30%] border-2 border-dashed border-purple-500 rounded-lg bg-purple-500/20"
                      style={{
                        left: `${dropPosition.x}%`,
                        top: `${dropPosition.y}%`,
                      }}
                    >
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="bg-purple-600 px-3 py-1.5 rounded-lg text-white text-sm font-medium shadow-lg">
                          Déposer ici
                        </div>
                      </div>
                    </motion.div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>

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

            {/* Empty state */}
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
          </div>
        </div>

        {/* Quick insert bar */}
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-40">
          <QuickInsertBar
            onInsertImage={handleInsertImage}
            onInsertText={handleInsertText}
            onInsertShape={handleInsertShape}
            disabled={disabled || isLoading}
          />
        </div>

        {/* Keyboard hints */}
        {selectedElementIds.size > 0 && !isInteracting && (
          <div className="absolute bottom-16 left-1/2 transform -translate-x-1/2 bg-gray-800/80 backdrop-blur rounded-lg px-3 py-1.5 text-xs text-gray-400 z-30 flex items-center gap-3">
            <span><kbd className="px-1 py-0.5 bg-gray-700 rounded text-gray-300">Suppr</kbd> effacer</span>
            <span><kbd className="px-1 py-0.5 bg-gray-700 rounded text-gray-300">Shift+clic</kbd> multi-sélection</span>
            <span><kbd className="px-1 py-0.5 bg-gray-700 rounded text-gray-300">Ctrl+molette</kbd> zoom</span>
          </div>
        )}

        {/* History notification */}
        <AnimatePresence>
          {historyNotification && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              className={`absolute bottom-20 left-1/2 transform -translate-x-1/2 px-4 py-2 rounded-lg text-sm font-medium shadow-lg z-50 ${
                historyNotification.type === 'undo'
                  ? 'bg-amber-600 text-white'
                  : 'bg-green-600 text-white'
              }`}
            >
              {historyNotification.message}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Properties panel */}
      <AnimatePresence>
        {showProperties && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 224, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="bg-gray-900/95 border-l border-gray-800 overflow-y-auto overflow-x-hidden flex-shrink-0"
          >
            <div className="w-56">
              <div className="sticky top-0 bg-gray-900 border-b border-gray-800 px-3 py-2 flex items-center justify-between">
                <h3 className="text-xs font-medium text-gray-300 uppercase tracking-wide">
                  Propriétés
                  {selectedElementIds.size > 1 && (
                    <span className="ml-1 text-purple-400">({selectedElementIds.size})</span>
                  )}
                </h3>
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
          </motion.div>
        )}
      </AnimatePresence>

      {/* Context menu */}
      {contextMenu && selectedElement && (
        <ElementContextMenu
          position={contextMenu}
          onClose={closeContextMenu}
          onCopy={copyElement}
          onPaste={pasteElement}
          onDuplicate={duplicateElementAction}
          onDelete={handleDeleteSelected}
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
