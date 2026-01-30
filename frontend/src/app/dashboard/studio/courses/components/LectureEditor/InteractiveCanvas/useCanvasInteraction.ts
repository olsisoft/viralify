'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import type { SlideElement, UpdateElementRequest } from '../../../lib/lecture-editor-types';

interface Position {
  x: number;
  y: number;
}

interface Size {
  width: number;
  height: number;
}

type ResizeHandle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w';

interface DragState {
  isDragging: boolean;
  startPosition: Position;
  startElementPosition: Position;
}

interface ResizeState {
  isResizing: boolean;
  handle: ResizeHandle | null;
  startPosition: Position;
  startElementBounds: { x: number; y: number; width: number; height: number };
}

interface RotateState {
  isRotating: boolean;
  startAngle: number;
  startRotation: number;
  elementCenter: Position;
}

interface UseCanvasInteractionOptions {
  elements: SlideElement[];
  selectedElementId: string | null;
  canvasRef: React.RefObject<HTMLDivElement>;
  onSelectElement: (elementId: string | null) => void;
  onUpdateElement: (elementId: string, updates: UpdateElementRequest) => void;
  onDeleteElement: (elementId: string) => void;
  onDuplicateElement?: (element: SlideElement) => void;
  onBringToFront?: (elementId: string) => void;
  onSendToBack?: (elementId: string) => void;
  disabled?: boolean;
}

interface AlignmentGuide {
  type: 'vertical' | 'horizontal';
  position: number; // percentage
}

// Snap threshold in percentage
const SNAP_THRESHOLD = 1.5;

export function useCanvasInteraction({
  elements,
  selectedElementId,
  canvasRef,
  onSelectElement,
  onUpdateElement,
  onDeleteElement,
  onDuplicateElement,
  onBringToFront,
  onSendToBack,
  disabled = false,
}: UseCanvasInteractionOptions) {
  // Clipboard for copy/paste
  const [clipboard, setClipboard] = useState<SlideElement | null>(null);

  const [dragState, setDragState] = useState<DragState>({
    isDragging: false,
    startPosition: { x: 0, y: 0 },
    startElementPosition: { x: 0, y: 0 },
  });

  const [resizeState, setResizeState] = useState<ResizeState>({
    isResizing: false,
    handle: null,
    startPosition: { x: 0, y: 0 },
    startElementBounds: { x: 0, y: 0, width: 0, height: 0 },
  });

  const [rotateState, setRotateState] = useState<RotateState>({
    isRotating: false,
    startAngle: 0,
    startRotation: 0,
    elementCenter: { x: 0, y: 0 },
  });

  const [alignmentGuides, setAlignmentGuides] = useState<AlignmentGuide[]>([]);

  // Get canvas dimensions
  const getCanvasDimensions = useCallback(() => {
    if (!canvasRef.current) return { width: 1, height: 1 };
    const rect = canvasRef.current.getBoundingClientRect();
    return { width: rect.width, height: rect.height };
  }, [canvasRef]);

  // Convert pixel position to percentage
  const pixelToPercent = useCallback((pixelX: number, pixelY: number): Position => {
    const { width, height } = getCanvasDimensions();
    return {
      x: (pixelX / width) * 100,
      y: (pixelY / height) * 100,
    };
  }, [getCanvasDimensions]);

  // Get mouse position relative to canvas
  const getRelativePosition = useCallback((e: MouseEvent | React.MouseEvent): Position => {
    if (!canvasRef.current) return { x: 0, y: 0 };
    const rect = canvasRef.current.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  }, [canvasRef]);

  // Calculate snap guides based on other elements
  const calculateSnapGuides = useCallback((
    movingElement: SlideElement,
    newX: number,
    newY: number,
    newWidth?: number,
    newHeight?: number
  ): { guides: AlignmentGuide[]; snappedX: number; snappedY: number } => {
    const guides: AlignmentGuide[] = [];
    let snappedX = newX;
    let snappedY = newY;
    const width = newWidth ?? movingElement.width;
    const height = newHeight ?? movingElement.height;

    // Canvas center lines
    const centerX = 50;
    const centerY = 50;

    // Element center and edges
    const elemCenterX = newX + width / 2;
    const elemCenterY = newY + height / 2;
    const elemRight = newX + width;
    const elemBottom = newY + height;

    // Snap to canvas center
    if (Math.abs(elemCenterX - centerX) < SNAP_THRESHOLD) {
      snappedX = centerX - width / 2;
      guides.push({ type: 'vertical', position: centerX });
    }
    if (Math.abs(elemCenterY - centerY) < SNAP_THRESHOLD) {
      snappedY = centerY - height / 2;
      guides.push({ type: 'horizontal', position: centerY });
    }

    // Snap to other elements
    elements.forEach((other) => {
      if (other.id === movingElement.id) return;

      const otherCenterX = other.x + other.width / 2;
      const otherCenterY = other.y + other.height / 2;
      const otherRight = other.x + other.width;
      const otherBottom = other.y + other.height;

      // Vertical alignments (left edge, center, right edge)
      if (Math.abs(newX - other.x) < SNAP_THRESHOLD) {
        snappedX = other.x;
        guides.push({ type: 'vertical', position: other.x });
      }
      if (Math.abs(elemCenterX - otherCenterX) < SNAP_THRESHOLD) {
        snappedX = otherCenterX - width / 2;
        guides.push({ type: 'vertical', position: otherCenterX });
      }
      if (Math.abs(elemRight - otherRight) < SNAP_THRESHOLD) {
        snappedX = otherRight - width;
        guides.push({ type: 'vertical', position: otherRight });
      }
      if (Math.abs(newX - otherRight) < SNAP_THRESHOLD) {
        snappedX = otherRight;
        guides.push({ type: 'vertical', position: otherRight });
      }
      if (Math.abs(elemRight - other.x) < SNAP_THRESHOLD) {
        snappedX = other.x - width;
        guides.push({ type: 'vertical', position: other.x });
      }

      // Horizontal alignments (top edge, center, bottom edge)
      if (Math.abs(newY - other.y) < SNAP_THRESHOLD) {
        snappedY = other.y;
        guides.push({ type: 'horizontal', position: other.y });
      }
      if (Math.abs(elemCenterY - otherCenterY) < SNAP_THRESHOLD) {
        snappedY = otherCenterY - height / 2;
        guides.push({ type: 'horizontal', position: otherCenterY });
      }
      if (Math.abs(elemBottom - otherBottom) < SNAP_THRESHOLD) {
        snappedY = otherBottom - height;
        guides.push({ type: 'horizontal', position: otherBottom });
      }
      if (Math.abs(newY - otherBottom) < SNAP_THRESHOLD) {
        snappedY = otherBottom;
        guides.push({ type: 'horizontal', position: otherBottom });
      }
      if (Math.abs(elemBottom - other.y) < SNAP_THRESHOLD) {
        snappedY = other.y - height;
        guides.push({ type: 'horizontal', position: other.y });
      }
    });

    return { guides, snappedX, snappedY };
  }, [elements]);

  // Start dragging an element
  const startDrag = useCallback((elementId: string, e: React.MouseEvent) => {
    if (disabled) return;
    e.preventDefault();
    e.stopPropagation();

    const element = elements.find((el) => el.id === elementId);
    if (!element || element.locked) return;

    onSelectElement(elementId);

    const pos = getRelativePosition(e);
    setDragState({
      isDragging: true,
      startPosition: pos,
      startElementPosition: { x: element.x, y: element.y },
    });
  }, [disabled, elements, onSelectElement, getRelativePosition]);

  // Start resizing an element
  const startResize = useCallback((elementId: string, handle: ResizeHandle, e: React.MouseEvent) => {
    if (disabled) return;
    e.preventDefault();
    e.stopPropagation();

    const element = elements.find((el) => el.id === elementId);
    if (!element || element.locked) return;

    const pos = getRelativePosition(e);
    setResizeState({
      isResizing: true,
      handle,
      startPosition: pos,
      startElementBounds: {
        x: element.x,
        y: element.y,
        width: element.width,
        height: element.height,
      },
    });
  }, [disabled, elements, getRelativePosition]);

  // Start rotating an element
  const startRotate = useCallback((elementId: string, e: React.MouseEvent) => {
    if (disabled) return;
    e.preventDefault();
    e.stopPropagation();

    const element = elements.find((el) => el.id === elementId);
    if (!element || element.locked) return;

    const { width: canvasWidth, height: canvasHeight } = getCanvasDimensions();

    // Calculate element center in pixels
    const centerX = ((element.x + element.width / 2) / 100) * canvasWidth;
    const centerY = ((element.y + element.height / 2) / 100) * canvasHeight;

    const pos = getRelativePosition(e);

    // Calculate initial angle from center to mouse
    const startAngle = Math.atan2(pos.y - centerY, pos.x - centerX) * (180 / Math.PI);

    setRotateState({
      isRotating: true,
      startAngle,
      startRotation: element.rotation || 0,
      elementCenter: { x: centerX, y: centerY },
    });
  }, [disabled, elements, getRelativePosition, getCanvasDimensions]);

  // Handle mouse move for drag/resize/rotate
  useEffect(() => {
    if (!dragState.isDragging && !resizeState.isResizing && !rotateState.isRotating) return;

    const handleMouseMove = (e: MouseEvent) => {
      const currentPos = getRelativePosition(e);
      const { width: canvasWidth, height: canvasHeight } = getCanvasDimensions();

      if (dragState.isDragging && selectedElementId) {
        const element = elements.find((el) => el.id === selectedElementId);
        if (!element) return;

        // Calculate delta in pixels, then convert to percentage
        const deltaX = ((currentPos.x - dragState.startPosition.x) / canvasWidth) * 100;
        const deltaY = ((currentPos.y - dragState.startPosition.y) / canvasHeight) * 100;

        let newX = dragState.startElementPosition.x + deltaX;
        let newY = dragState.startElementPosition.y + deltaY;

        // Clamp to canvas bounds
        newX = Math.max(0, Math.min(100 - element.width, newX));
        newY = Math.max(0, Math.min(100 - element.height, newY));

        // Calculate snap guides
        const { guides, snappedX, snappedY } = calculateSnapGuides(element, newX, newY);
        setAlignmentGuides(guides);

        onUpdateElement(selectedElementId, { x: snappedX, y: snappedY });
      }

      if (resizeState.isResizing && selectedElementId && resizeState.handle) {
        const element = elements.find((el) => el.id === selectedElementId);
        if (!element) return;

        const deltaX = ((currentPos.x - resizeState.startPosition.x) / canvasWidth) * 100;
        const deltaY = ((currentPos.y - resizeState.startPosition.y) / canvasHeight) * 100;

        const { x: startX, y: startY, width: startW, height: startH } = resizeState.startElementBounds;

        let newX = startX;
        let newY = startY;
        let newWidth = startW;
        let newHeight = startH;

        // Calculate new bounds based on handle
        switch (resizeState.handle) {
          case 'e':
            newWidth = Math.max(5, startW + deltaX);
            break;
          case 'w':
            newWidth = Math.max(5, startW - deltaX);
            newX = startX + startW - newWidth;
            break;
          case 's':
            newHeight = Math.max(5, startH + deltaY);
            break;
          case 'n':
            newHeight = Math.max(5, startH - deltaY);
            newY = startY + startH - newHeight;
            break;
          case 'se':
            newWidth = Math.max(5, startW + deltaX);
            newHeight = Math.max(5, startH + deltaY);
            break;
          case 'sw':
            newWidth = Math.max(5, startW - deltaX);
            newX = startX + startW - newWidth;
            newHeight = Math.max(5, startH + deltaY);
            break;
          case 'ne':
            newWidth = Math.max(5, startW + deltaX);
            newHeight = Math.max(5, startH - deltaY);
            newY = startY + startH - newHeight;
            break;
          case 'nw':
            newWidth = Math.max(5, startW - deltaX);
            newX = startX + startW - newWidth;
            newHeight = Math.max(5, startH - deltaY);
            newY = startY + startH - newHeight;
            break;
        }

        // Clamp to canvas bounds
        newX = Math.max(0, newX);
        newY = Math.max(0, newY);
        newWidth = Math.min(100 - newX, newWidth);
        newHeight = Math.min(100 - newY, newHeight);

        // Calculate snap guides
        const { guides, snappedX, snappedY } = calculateSnapGuides(element, newX, newY, newWidth, newHeight);
        setAlignmentGuides(guides);

        onUpdateElement(selectedElementId, {
          x: snappedX,
          y: snappedY,
          width: newWidth,
          height: newHeight,
        });
      }

      // Handle rotation
      if (rotateState.isRotating && selectedElementId) {
        // Calculate current angle from element center to mouse
        const currentAngle = Math.atan2(
          currentPos.y - rotateState.elementCenter.y,
          currentPos.x - rotateState.elementCenter.x
        ) * (180 / Math.PI);

        // Calculate rotation delta and apply to start rotation
        let newRotation = rotateState.startRotation + (currentAngle - rotateState.startAngle);

        // Normalize to 0-360
        newRotation = ((newRotation % 360) + 360) % 360;

        // Snap to 0, 45, 90, 135, 180, 225, 270, 315 degrees when close
        const snapAngles = [0, 45, 90, 135, 180, 225, 270, 315, 360];
        const snapThreshold = 5;
        for (const snapAngle of snapAngles) {
          if (Math.abs(newRotation - snapAngle) < snapThreshold) {
            newRotation = snapAngle === 360 ? 0 : snapAngle;
            break;
          }
        }

        onUpdateElement(selectedElementId, { rotation: newRotation });
      }
    };

    const handleMouseUp = () => {
      setDragState({
        isDragging: false,
        startPosition: { x: 0, y: 0 },
        startElementPosition: { x: 0, y: 0 },
      });
      setResizeState({
        isResizing: false,
        handle: null,
        startPosition: { x: 0, y: 0 },
        startElementBounds: { x: 0, y: 0, width: 0, height: 0 },
      });
      setRotateState({
        isRotating: false,
        startAngle: 0,
        startRotation: 0,
        elementCenter: { x: 0, y: 0 },
      });
      setAlignmentGuides([]);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [
    dragState,
    resizeState,
    rotateState,
    selectedElementId,
    elements,
    getRelativePosition,
    getCanvasDimensions,
    calculateSnapGuides,
    onUpdateElement,
  ]);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle if typing in an input
      const isInInput = document.activeElement?.tagName === 'INPUT' ||
                        document.activeElement?.tagName === 'TEXTAREA';

      const hasCtrlOrMeta = e.ctrlKey || e.metaKey;

      // Copy (Ctrl+C) - works even without selection to copy selected
      if (hasCtrlOrMeta && e.key.toLowerCase() === 'c' && !isInInput && selectedElementId) {
        e.preventDefault();
        const element = elements.find((el) => el.id === selectedElementId);
        if (element) {
          setClipboard(element);
        }
        return;
      }

      // Paste (Ctrl+V) - paste copied element with offset
      if (hasCtrlOrMeta && e.key.toLowerCase() === 'v' && !isInInput && clipboard && onDuplicateElement) {
        e.preventDefault();
        // Create a copy with offset position
        const pastedElement = {
          ...clipboard,
          x: Math.min(90, clipboard.x + 5),
          y: Math.min(90, clipboard.y + 5),
        };
        onDuplicateElement(pastedElement);
        return;
      }

      // Duplicate (Ctrl+D) - duplicate selected element in place
      if (hasCtrlOrMeta && e.key.toLowerCase() === 'd' && !isInInput && selectedElementId && onDuplicateElement) {
        e.preventDefault();
        const element = elements.find((el) => el.id === selectedElementId);
        if (element) {
          const duplicatedElement = {
            ...element,
            x: Math.min(90, element.x + 3),
            y: Math.min(90, element.y + 3),
          };
          onDuplicateElement(duplicatedElement);
        }
        return;
      }

      // Bring to front (Ctrl+]) or (Ctrl+Shift+Up)
      if (hasCtrlOrMeta && (e.key === ']' || (e.shiftKey && e.key === 'ArrowUp')) && !isInInput && selectedElementId && onBringToFront) {
        e.preventDefault();
        onBringToFront(selectedElementId);
        return;
      }

      // Send to back (Ctrl+[) or (Ctrl+Shift+Down)
      if (hasCtrlOrMeta && (e.key === '[' || (e.shiftKey && e.key === 'ArrowDown')) && !isInInput && selectedElementId && onSendToBack) {
        e.preventDefault();
        onSendToBack(selectedElementId);
        return;
      }

      if (disabled || !selectedElementId) return;

      // Delete element
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (isInInput) return;
        e.preventDefault();
        onDeleteElement(selectedElementId);
        onSelectElement(null);
      }

      // Escape to deselect
      if (e.key === 'Escape') {
        onSelectElement(null);
      }

      // Arrow keys to nudge (without Ctrl)
      if (hasCtrlOrMeta) return; // Don't nudge with Ctrl held

      const element = elements.find((el) => el.id === selectedElementId);
      if (!element || element.locked) return;

      const nudgeAmount = e.shiftKey ? 5 : 1; // Shift for larger nudge
      let newX = element.x;
      let newY = element.y;

      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault();
          newX = Math.max(0, element.x - nudgeAmount);
          break;
        case 'ArrowRight':
          e.preventDefault();
          newX = Math.min(100 - element.width, element.x + nudgeAmount);
          break;
        case 'ArrowUp':
          e.preventDefault();
          newY = Math.max(0, element.y - nudgeAmount);
          break;
        case 'ArrowDown':
          e.preventDefault();
          newY = Math.min(100 - element.height, element.y + nudgeAmount);
          break;
      }

      if (newX !== element.x || newY !== element.y) {
        onUpdateElement(selectedElementId, { x: newX, y: newY });
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [disabled, selectedElementId, elements, clipboard, onSelectElement, onDeleteElement, onUpdateElement, onDuplicateElement, onBringToFront, onSendToBack]);

  // Click on canvas background to deselect
  const handleCanvasClick = useCallback((e: React.MouseEvent) => {
    if (e.target === canvasRef.current) {
      onSelectElement(null);
    }
  }, [canvasRef, onSelectElement]);

  // Copy element to clipboard
  const copyElement = useCallback(() => {
    if (!selectedElementId) return;
    const element = elements.find((el) => el.id === selectedElementId);
    if (element) {
      setClipboard(element);
    }
  }, [selectedElementId, elements]);

  // Paste element from clipboard
  const pasteElement = useCallback(() => {
    if (!clipboard || !onDuplicateElement) return;
    const pastedElement = {
      ...clipboard,
      x: Math.min(90, clipboard.x + 5),
      y: Math.min(90, clipboard.y + 5),
    };
    onDuplicateElement(pastedElement);
  }, [clipboard, onDuplicateElement]);

  // Duplicate selected element
  const duplicateElement = useCallback(() => {
    if (!selectedElementId || !onDuplicateElement) return;
    const element = elements.find((el) => el.id === selectedElementId);
    if (element) {
      const duplicatedElement = {
        ...element,
        x: Math.min(90, element.x + 3),
        y: Math.min(90, element.y + 3),
      };
      onDuplicateElement(duplicatedElement);
    }
  }, [selectedElementId, elements, onDuplicateElement]);

  return {
    startDrag,
    startResize,
    startRotate,
    handleCanvasClick,
    alignmentGuides,
    isDragging: dragState.isDragging,
    isResizing: resizeState.isResizing,
    isRotating: rotateState.isRotating,
    // Clipboard actions
    hasClipboard: clipboard !== null,
    copyElement,
    pasteElement,
    duplicateElement,
  };
}
