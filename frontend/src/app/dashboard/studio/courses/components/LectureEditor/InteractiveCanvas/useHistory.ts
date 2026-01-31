'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import type { SlideElement } from '../../../lib/lecture-editor-types';

interface HistoryState {
  elements: SlideElement[];
  timestamp: number;
  description: string;
}

interface UseHistoryOptions {
  elements: SlideElement[];
  maxHistory?: number;
  onRestore: (elements: SlideElement[]) => Promise<void>;
}

const DEFAULT_MAX_HISTORY = 50;

export function useHistory({
  elements,
  maxHistory = DEFAULT_MAX_HISTORY,
  onRestore,
}: UseHistoryOptions) {
  const [past, setPast] = useState<HistoryState[]>([]);
  const [future, setFuture] = useState<HistoryState[]>([]);
  const [notification, setNotification] = useState<{ type: 'undo' | 'redo'; message: string } | null>(null);

  // Ref to track the last saved state to avoid duplicates
  const lastSavedRef = useRef<string>('');
  const isRestoringRef = useRef(false);

  // Clear notification after delay
  useEffect(() => {
    if (notification) {
      const timeout = setTimeout(() => setNotification(null), 1500);
      return () => clearTimeout(timeout);
    }
  }, [notification]);

  // Save current state to history
  const saveState = useCallback((description: string = 'Change') => {
    if (isRestoringRef.current) return;

    const stateKey = JSON.stringify(elements.map(e => ({
      id: e.id,
      x: e.x,
      y: e.y,
      width: e.width,
      height: e.height,
      rotation: e.rotation,
      zIndex: e.zIndex,
      visible: e.visible,
      locked: e.locked,
      imageContent: e.imageContent,
      textContent: e.textContent,
      shapeContent: e.shapeContent,
    })));

    // Don't save if nothing changed
    if (stateKey === lastSavedRef.current) return;

    lastSavedRef.current = stateKey;

    setPast((prev) => {
      const newPast = [
        ...prev,
        {
          elements: JSON.parse(JSON.stringify(elements)),
          timestamp: Date.now(),
          description,
        },
      ];
      // Limit history size
      if (newPast.length > maxHistory) {
        return newPast.slice(-maxHistory);
      }
      return newPast;
    });

    // Clear future when new action is taken
    setFuture([]);
  }, [elements, maxHistory]);

  // Undo last action
  const undo = useCallback(async () => {
    if (past.length === 0) return;

    isRestoringRef.current = true;

    const lastState = past[past.length - 1];
    const newPast = past.slice(0, -1);

    // Save current state to future
    setFuture((prev) => [
      ...prev,
      {
        elements: JSON.parse(JSON.stringify(elements)),
        timestamp: Date.now(),
        description: 'Redo point',
      },
    ]);

    setPast(newPast);

    // Restore the previous state
    await onRestore(lastState.elements);

    lastSavedRef.current = JSON.stringify(lastState.elements.map(e => ({
      id: e.id,
      x: e.x,
      y: e.y,
      width: e.width,
      height: e.height,
      rotation: e.rotation,
      zIndex: e.zIndex,
      visible: e.visible,
      locked: e.locked,
      imageContent: e.imageContent,
      textContent: e.textContent,
      shapeContent: e.shapeContent,
    })));

    setNotification({ type: 'undo', message: `Annulé: ${lastState.description}` });

    isRestoringRef.current = false;
  }, [past, elements, onRestore]);

  // Redo last undone action
  const redo = useCallback(async () => {
    if (future.length === 0) return;

    isRestoringRef.current = true;

    const nextState = future[future.length - 1];
    const newFuture = future.slice(0, -1);

    // Save current state to past
    setPast((prev) => [
      ...prev,
      {
        elements: JSON.parse(JSON.stringify(elements)),
        timestamp: Date.now(),
        description: 'Undo point',
      },
    ]);

    setFuture(newFuture);

    // Restore the future state
    await onRestore(nextState.elements);

    lastSavedRef.current = JSON.stringify(nextState.elements.map(e => ({
      id: e.id,
      x: e.x,
      y: e.y,
      width: e.width,
      height: e.height,
      rotation: e.rotation,
      zIndex: e.zIndex,
      visible: e.visible,
      locked: e.locked,
      imageContent: e.imageContent,
      textContent: e.textContent,
      shapeContent: e.shapeContent,
    })));

    setNotification({ type: 'redo', message: 'Rétabli' });

    isRestoringRef.current = false;
  }, [future, elements, onRestore]);

  // Keyboard shortcuts for undo/redo
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const isInInput = document.activeElement?.tagName === 'INPUT' ||
                        document.activeElement?.tagName === 'TEXTAREA';
      if (isInInput) return;

      const hasCtrlOrMeta = e.ctrlKey || e.metaKey;

      // Undo (Ctrl+Z)
      if (hasCtrlOrMeta && e.key.toLowerCase() === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
        return;
      }

      // Redo (Ctrl+Y or Ctrl+Shift+Z)
      if (hasCtrlOrMeta && (e.key.toLowerCase() === 'y' || (e.shiftKey && e.key.toLowerCase() === 'z'))) {
        e.preventDefault();
        redo();
        return;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [undo, redo]);

  // Clear history
  const clearHistory = useCallback(() => {
    setPast([]);
    setFuture([]);
    lastSavedRef.current = '';
  }, []);

  return {
    // State
    canUndo: past.length > 0,
    canRedo: future.length > 0,
    historyLength: past.length,
    futureLength: future.length,
    notification,

    // Actions
    saveState,
    undo,
    redo,
    clearHistory,
  };
}
