'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import {
  PracticeSession,
  Exercise,
  LearnerProgress,
  CreateSessionRequest,
  CreateSessionResponse,
  InteractionResponse,
  SubmitCodeResponse,
  HintResponse,
  SessionSummary,
  Badge,
  LeaderboardEntry,
  DifficultyLevel,
  ExerciseCategory,
  ChatMessage,
  WSMessage,
  WSResponse,
} from '../lib/practice-types';

const API_BASE = process.env.NEXT_PUBLIC_PRACTICE_API_URL || 'http://localhost:8008';
const WS_BASE = process.env.NEXT_PUBLIC_PRACTICE_WS_URL || 'ws://localhost:8008';

interface UsePracticeState {
  session: PracticeSession | null;
  currentExercise: Exercise | null;
  progress: LearnerProgress | null;
  messages: ChatMessage[];
  isLoading: boolean;
  isExecuting: boolean;
  error: string | null;
  wsConnected: boolean;
}

export function usePractice(userId: string) {
  const [state, setState] = useState<UsePracticeState>({
    session: null,
    currentExercise: null,
    progress: null,
    messages: [],
    isLoading: false,
    isExecuting: false,
    error: null,
    wsConnected: false,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Helper to update state
  const updateState = useCallback((updates: Partial<UsePracticeState>) => {
    setState((prev) => ({ ...prev, ...updates }));
  }, []);

  // Add message to chat
  const addMessage = useCallback((role: 'user' | 'assistant' | 'system', content: string) => {
    const message: ChatMessage = {
      role,
      content,
      timestamp: new Date().toISOString(),
    };
    setState((prev) => ({
      ...prev,
      messages: [...prev.messages, message],
    }));
  }, []);

  // API call helper
  const apiCall = useCallback(async <T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> => {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }, []);

  // ==================== SESSION MANAGEMENT ====================

  const createSession = useCallback(async (
    courseId?: string,
    difficulty: DifficultyLevel = DifficultyLevel.BEGINNER,
    categories?: ExerciseCategory[],
    pairProgramming: boolean = false
  ): Promise<CreateSessionResponse | null> => {
    updateState({ isLoading: true, error: null });

    try {
      const request: CreateSessionRequest = {
        user_id: userId,
        course_id: courseId,
        difficulty_preference: difficulty,
        categories_focus: categories,
        pair_programming_enabled: pairProgramming,
      };

      const response = await apiCall<CreateSessionResponse>(
        '/api/v1/practice/sessions',
        {
          method: 'POST',
          body: JSON.stringify(request),
        }
      );

      updateState({
        session: response.session,
        currentExercise: response.suggested_exercise || null,
        messages: [],
        isLoading: false,
      });

      // Add welcome message
      addMessage('assistant', response.welcome_message);

      return response;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to create session';
      updateState({ error: message, isLoading: false });
      return null;
    }
  }, [userId, apiCall, updateState, addMessage]);

  const loadSession = useCallback(async (sessionId: string): Promise<PracticeSession | null> => {
    updateState({ isLoading: true, error: null });

    try {
      const session = await apiCall<PracticeSession>(
        `/api/v1/practice/sessions/${sessionId}`
      );

      updateState({
        session,
        currentExercise: session.current_exercise || null,
        messages: session.messages || [],
        isLoading: false,
      });

      return session;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load session';
      updateState({ error: message, isLoading: false });
      return null;
    }
  }, [apiCall, updateState]);

  const endSession = useCallback(async (): Promise<SessionSummary | null> => {
    if (!state.session) return null;

    updateState({ isLoading: true, error: null });

    try {
      const response = await apiCall<{ status: string; summary: SessionSummary }>(
        `/api/v1/practice/sessions/${state.session.id}`,
        { method: 'DELETE' }
      );

      // Disconnect WebSocket
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }

      updateState({
        session: null,
        currentExercise: null,
        messages: [],
        isLoading: false,
        wsConnected: false,
      });

      return response.summary;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to end session';
      updateState({ error: message, isLoading: false });
      return null;
    }
  }, [state.session, apiCall, updateState]);

  // ==================== EXERCISE MANAGEMENT ====================

  const listExercises = useCallback(async (
    category?: ExerciseCategory,
    difficulty?: DifficultyLevel,
    courseId?: string
  ): Promise<Exercise[]> => {
    const params = new URLSearchParams();
    if (category) params.set('category', category);
    if (difficulty) params.set('difficulty', difficulty);
    if (courseId) params.set('course_id', courseId);

    const response = await apiCall<{ exercises: Exercise[] }>(
      `/api/v1/practice/exercises?${params}`
    );
    return response.exercises;
  }, [apiCall]);

  const selectExercise = useCallback(async (exerciseId: string): Promise<Exercise | null> => {
    updateState({ isLoading: true, error: null });

    try {
      const exercise = await apiCall<Exercise>(
        `/api/v1/practice/exercises/${exerciseId}`
      );

      updateState({
        currentExercise: exercise,
        isLoading: false,
      });

      return exercise;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load exercise';
      updateState({ error: message, isLoading: false });
      return null;
    }
  }, [apiCall, updateState]);

  // ==================== PRACTICE INTERACTION ====================

  const sendMessage = useCallback(async (message: string): Promise<InteractionResponse | null> => {
    if (!state.session) return null;

    addMessage('user', message);
    updateState({ isLoading: true, error: null });

    try {
      const response = await apiCall<InteractionResponse>(
        '/api/v1/practice/interact',
        {
          method: 'POST',
          body: JSON.stringify({
            session_id: state.session.id,
            message,
          }),
        }
      );

      addMessage('assistant', response.response);

      if (response.exercise) {
        updateState({ currentExercise: response.exercise });
      }

      updateState({ isLoading: false });
      return response;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to send message';
      updateState({ error: message, isLoading: false });
      return null;
    }
  }, [state.session, apiCall, updateState, addMessage]);

  const submitCode = useCallback(async (code: string): Promise<SubmitCodeResponse | null> => {
    if (!state.session) return null;

    updateState({ isExecuting: true, error: null });

    try {
      const response = await apiCall<SubmitCodeResponse>(
        `/api/v1/practice/sessions/${state.session.id}/submit`,
        {
          method: 'POST',
          body: JSON.stringify({ code }),
        }
      );

      addMessage('assistant', response.feedback);

      if (response.next_exercise) {
        updateState({ currentExercise: response.next_exercise });
      }

      updateState({ isExecuting: false });
      return response;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to submit code';
      updateState({ error: message, isExecuting: false });
      return null;
    }
  }, [state.session, apiCall, updateState, addMessage]);

  const requestHint = useCallback(async (hintLevel: number = 1): Promise<HintResponse | null> => {
    if (!state.session) return null;

    updateState({ isLoading: true, error: null });

    try {
      const response = await apiCall<HintResponse>(
        `/api/v1/practice/sessions/${state.session.id}/hint`,
        {
          method: 'POST',
          body: JSON.stringify({ hint_level: hintLevel }),
        }
      );

      addMessage('assistant', `💡 Indice ${response.hint_number}: ${response.hint}`);

      updateState({ isLoading: false });
      return response;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to get hint';
      updateState({ error: message, isLoading: false });
      return null;
    }
  }, [state.session, apiCall, updateState, addMessage]);

  // ==================== PROGRESS TRACKING ====================

  const loadProgress = useCallback(async (): Promise<LearnerProgress | null> => {
    try {
      const progress = await apiCall<LearnerProgress>(
        `/api/v1/practice/progress/${userId}`
      );
      updateState({ progress });
      return progress;
    } catch (error) {
      return null;
    }
  }, [userId, apiCall, updateState]);

  const getLeaderboard = useCallback(async (limit: number = 10): Promise<LeaderboardEntry[]> => {
    const response = await apiCall<{ leaderboard: LeaderboardEntry[] }>(
      `/api/v1/practice/leaderboard?limit=${limit}`
    );
    return response.leaderboard;
  }, [apiCall]);

  const getBadges = useCallback(async (): Promise<Badge[]> => {
    const response = await apiCall<{ badges: Badge[] }>(
      '/api/v1/practice/badges'
    );
    return response.badges;
  }, [apiCall]);

  // ==================== WEBSOCKET FOR REAL-TIME ====================

  const connectWebSocket = useCallback(() => {
    if (!state.session || wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(`${WS_BASE}/ws/practice/${state.session.id}`);

    ws.onopen = () => {
      updateState({ wsConnected: true });
    };

    ws.onclose = () => {
      updateState({ wsConnected: false });
      // Attempt reconnection after 3 seconds
      reconnectTimeoutRef.current = setTimeout(() => {
        if (state.session) {
          connectWebSocket();
        }
      }, 3000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onmessage = (event) => {
      try {
        const response: WSResponse = JSON.parse(event.data);

        switch (response.type) {
          case 'response':
            if (response.content) {
              addMessage('assistant', response.content);
            }
            if (response.exercise) {
              updateState({ currentExercise: response.exercise });
            }
            break;

          case 'evaluation':
            if (response.content) {
              addMessage('assistant', response.content);
            }
            updateState({ isExecuting: false });
            break;

          case 'hint':
            if (response.content) {
              addMessage('assistant', `💡 Indice ${response.hint_number}: ${response.content}`);
            }
            break;

          case 'error':
            updateState({ error: response.content || 'Unknown error' });
            break;
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    wsRef.current = ws;
  }, [state.session, updateState, addMessage]);

  const disconnectWebSocket = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    updateState({ wsConnected: false });
  }, [updateState]);

  const sendWSMessage = useCallback((message: WSMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));

      if (message.type === 'message' && message.content) {
        addMessage('user', message.content);
      }
    }
  }, [addMessage]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnectWebSocket();
    };
  }, [disconnectWebSocket]);

  // Auto-connect WebSocket when session is created
  useEffect(() => {
    if (state.session && !wsRef.current) {
      connectWebSocket();
    }
  }, [state.session, connectWebSocket]);

  return {
    // State
    ...state,

    // Session actions
    createSession,
    loadSession,
    endSession,

    // Exercise actions
    listExercises,
    selectExercise,

    // Practice actions
    sendMessage,
    submitCode,
    requestHint,

    // Progress actions
    loadProgress,
    getLeaderboard,
    getBadges,

    // WebSocket actions
    connectWebSocket,
    disconnectWebSocket,
    sendWSMessage,

    // Utils
    clearError: () => updateState({ error: null }),
  };
}
