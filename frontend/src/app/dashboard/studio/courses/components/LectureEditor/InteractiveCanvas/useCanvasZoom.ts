'use client';

import { useState, useCallback, useEffect, useRef } from 'react';

interface ZoomState {
  scale: number;
  panX: number;
  panY: number;
}

interface UseCanvasZoomOptions {
  canvasRef: React.RefObject<HTMLDivElement>;
  minZoom?: number;
  maxZoom?: number;
  zoomStep?: number;
  disabled?: boolean;
}

const DEFAULT_MIN_ZOOM = 0.25;
const DEFAULT_MAX_ZOOM = 4;
const DEFAULT_ZOOM_STEP = 0.1;

export function useCanvasZoom({
  canvasRef,
  minZoom = DEFAULT_MIN_ZOOM,
  maxZoom = DEFAULT_MAX_ZOOM,
  zoomStep = DEFAULT_ZOOM_STEP,
  disabled = false,
}: UseCanvasZoomOptions) {
  const [zoomState, setZoomState] = useState<ZoomState>({
    scale: 1,
    panX: 0,
    panY: 0,
  });

  const [isPanning, setIsPanning] = useState(false);
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  const panStartRef = useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  // Zoom in
  const zoomIn = useCallback(() => {
    setZoomState((prev) => ({
      ...prev,
      scale: Math.min(maxZoom, prev.scale + zoomStep),
    }));
  }, [maxZoom, zoomStep]);

  // Zoom out
  const zoomOut = useCallback(() => {
    setZoomState((prev) => ({
      ...prev,
      scale: Math.max(minZoom, prev.scale - zoomStep),
    }));
  }, [minZoom, zoomStep]);

  // Reset zoom and pan
  const resetZoom = useCallback(() => {
    setZoomState({ scale: 1, panX: 0, panY: 0 });
  }, []);

  // Fit to screen
  const fitToScreen = useCallback(() => {
    setZoomState({ scale: 1, panX: 0, panY: 0 });
  }, []);

  // Set specific zoom level
  const setZoom = useCallback((scale: number) => {
    setZoomState((prev) => ({
      ...prev,
      scale: Math.max(minZoom, Math.min(maxZoom, scale)),
    }));
  }, [minZoom, maxZoom]);

  // Handle wheel zoom
  useEffect(() => {
    if (disabled || !canvasRef.current) return;

    const canvas = canvasRef.current;

    const handleWheel = (e: WheelEvent) => {
      // Only zoom if Ctrl/Meta is pressed, otherwise let it scroll
      if (!e.ctrlKey && !e.metaKey) return;

      e.preventDefault();

      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      // Calculate zoom
      const delta = e.deltaY > 0 ? -zoomStep : zoomStep;
      const newScale = Math.max(minZoom, Math.min(maxZoom, zoomState.scale + delta));

      if (newScale === zoomState.scale) return;

      // Zoom towards mouse position
      const scaleRatio = newScale / zoomState.scale;
      const newPanX = mouseX - (mouseX - zoomState.panX) * scaleRatio;
      const newPanY = mouseY - (mouseY - zoomState.panY) * scaleRatio;

      setZoomState({
        scale: newScale,
        panX: newPanX,
        panY: newPanY,
      });
    };

    canvas.addEventListener('wheel', handleWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', handleWheel);
  }, [disabled, canvasRef, zoomState, minZoom, maxZoom, zoomStep]);

  // Handle space key for pan mode
  useEffect(() => {
    if (disabled) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !isSpacePressed) {
        // Don't activate if typing
        const isInInput = document.activeElement?.tagName === 'INPUT' ||
                          document.activeElement?.tagName === 'TEXTAREA';
        if (isInInput) return;

        e.preventDefault();
        setIsSpacePressed(true);
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        setIsSpacePressed(false);
        setIsPanning(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [disabled, isSpacePressed]);

  // Handle pan with space + drag
  useEffect(() => {
    if (disabled || !canvasRef.current) return;

    const canvas = canvasRef.current;

    const handleMouseDown = (e: MouseEvent) => {
      if (!isSpacePressed) return;
      e.preventDefault();
      setIsPanning(true);
      panStartRef.current = {
        x: e.clientX,
        y: e.clientY,
        panX: zoomState.panX,
        panY: zoomState.panY,
      };
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!isPanning) return;
      const deltaX = e.clientX - panStartRef.current.x;
      const deltaY = e.clientY - panStartRef.current.y;
      setZoomState((prev) => ({
        ...prev,
        panX: panStartRef.current.panX + deltaX,
        panY: panStartRef.current.panY + deltaY,
      }));
    };

    const handleMouseUp = () => {
      setIsPanning(false);
    };

    canvas.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      canvas.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [disabled, canvasRef, isSpacePressed, isPanning, zoomState.panX, zoomState.panY]);

  // Generate transform style
  const transformStyle = {
    transform: `translate(${zoomState.panX}px, ${zoomState.panY}px) scale(${zoomState.scale})`,
    transformOrigin: '0 0',
  };

  return {
    // State
    scale: zoomState.scale,
    panX: zoomState.panX,
    panY: zoomState.panY,
    isPanning,
    isSpacePressed,

    // Actions
    zoomIn,
    zoomOut,
    resetZoom,
    fitToScreen,
    setZoom,

    // Style
    transformStyle,

    // Zoom presets
    zoomPresets: [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4],
  };
}
