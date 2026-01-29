'use client';

import { useState, useCallback, useRef } from 'react';
import type {
  LectureComponents,
  SlideComponent,
  EditorAction,
  EditorActionType
} from '../lib/lecture-editor-types';

/**
 * Deep clone utility for creating state snapshots
 */
function deepClone<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj));
}

/**
 * Compare two objects for equality (shallow for primitives, deep for objects)
 */
function isEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

export interface HistoryEntry {
  action: EditorAction;
  snapshot: LectureComponents;
}

export interface EditorHistoryState {
  past: HistoryEntry[];
  present: LectureComponents | null;
  future: HistoryEntry[];
}

export interface UseEditorHistoryOptions {
  maxHistory?: number;
}

export interface UseEditorHistoryReturn {
  // State
  canUndo: boolean;
  canRedo: boolean;
  historyLength: number;
  futureLength: number;

  // Actions
  initialize: (components: LectureComponents) => void;
  pushAction: (
    type: EditorActionType,
    slideId: string | undefined,
    previousState: unknown,
    newState: unknown,
    newComponents: LectureComponents
  ) => void;
  undo: () => LectureComponents | null;
  redo: () => LectureComponents | null;
  clear: () => void;

  // Current state
  currentState: LectureComponents | null;
}

/**
 * Hook for managing editor history with undo/redo functionality
 * Stores up to maxHistory (default 50) actions with full state snapshots
 */
export function useEditorHistory(
  options: UseEditorHistoryOptions = {}
): UseEditorHistoryReturn {
  const { maxHistory = 50 } = options;

  const [history, setHistory] = useState<EditorHistoryState>({
    past: [],
    present: null,
    future: [],
  });

  // Use ref for quick access to current state without re-renders
  const currentStateRef = useRef<LectureComponents | null>(null);

  // Initialize with initial state
  const initialize = useCallback((components: LectureComponents) => {
    const cloned = deepClone(components);
    currentStateRef.current = cloned;
    setHistory({
      past: [],
      present: cloned,
      future: [],
    });
  }, []);

  // Push a new action to history
  const pushAction = useCallback((
    type: EditorActionType,
    slideId: string | undefined,
    previousState: unknown,
    newState: unknown,
    newComponents: LectureComponents
  ) => {
    setHistory((prev) => {
      if (!prev.present) return prev;

      // Skip if no actual change
      if (isEqual(previousState, newState)) {
        return prev;
      }

      const action: EditorAction = {
        type,
        timestamp: Date.now(),
        previousState: deepClone(previousState),
        newState: deepClone(newState),
        slideId,
      };

      const entry: HistoryEntry = {
        action,
        snapshot: deepClone(prev.present),
      };

      // Add to past, trim if exceeds maxHistory
      let newPast = [...prev.past, entry];
      if (newPast.length > maxHistory) {
        newPast = newPast.slice(-maxHistory);
      }

      const clonedNew = deepClone(newComponents);
      currentStateRef.current = clonedNew;

      return {
        past: newPast,
        present: clonedNew,
        future: [], // Clear future on new action
      };
    });
  }, [maxHistory]);

  // Undo last action
  const undo = useCallback((): LectureComponents | null => {
    let result: LectureComponents | null = null;

    setHistory((prev) => {
      if (prev.past.length === 0 || !prev.present) {
        return prev;
      }

      const newPast = [...prev.past];
      const lastEntry = newPast.pop()!;

      // Create future entry with current state
      const futureEntry: HistoryEntry = {
        action: {
          ...lastEntry.action,
          // Swap previous and new state for redo
          previousState: lastEntry.action.newState,
          newState: lastEntry.action.previousState,
        },
        snapshot: deepClone(prev.present),
      };

      result = deepClone(lastEntry.snapshot);
      currentStateRef.current = result;

      return {
        past: newPast,
        present: result,
        future: [futureEntry, ...prev.future],
      };
    });

    return result;
  }, []);

  // Redo last undone action
  const redo = useCallback((): LectureComponents | null => {
    let result: LectureComponents | null = null;

    setHistory((prev) => {
      if (prev.future.length === 0 || !prev.present) {
        return prev;
      }

      const newFuture = [...prev.future];
      const nextEntry = newFuture.shift()!;

      // Create past entry with current state
      const pastEntry: HistoryEntry = {
        action: {
          ...nextEntry.action,
          // Swap back for undo
          previousState: nextEntry.action.newState,
          newState: nextEntry.action.previousState,
        },
        snapshot: deepClone(prev.present),
      };

      result = deepClone(nextEntry.snapshot);
      currentStateRef.current = result;

      return {
        past: [...prev.past, pastEntry],
        present: result,
        future: newFuture,
      };
    });

    return result;
  }, []);

  // Clear all history
  const clear = useCallback(() => {
    currentStateRef.current = null;
    setHistory({
      past: [],
      present: null,
      future: [],
    });
  }, []);

  return {
    // State
    canUndo: history.past.length > 0,
    canRedo: history.future.length > 0,
    historyLength: history.past.length,
    futureLength: history.future.length,

    // Actions
    initialize,
    pushAction,
    undo,
    redo,
    clear,

    // Current state
    currentState: history.present,
  };
}

export default useEditorHistory;
