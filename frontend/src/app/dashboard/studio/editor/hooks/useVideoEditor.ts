'use client';

/**
 * Video Editor Hook
 * Manages video editor state and API interactions
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import {
  VideoProject,
  VideoSegment,
  TextOverlay,
  ImageOverlay,
  CreateProjectRequest,
  AddSegmentRequest,
  UpdateSegmentRequest,
  RenderProjectRequest,
  RenderJobStatus,
  SupportedFormats,
  EditorState,
} from '../lib/editor-types';

// Use API gateway URL - all requests go through the gateway which routes to media-generator
const MEDIA_API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

interface UseVideoEditorOptions {
  projectId?: string;
  userId?: string;
  onError?: (error: string) => void;
}

interface UseVideoEditorReturn {
  // State
  project: VideoProject | null;
  isLoading: boolean;
  isSaving: boolean;
  isRendering: boolean;
  renderJob: RenderJobStatus | null;
  error: string | null;
  supportedFormats: SupportedFormats | null;

  // Project actions
  createProject: (request: CreateProjectRequest) => Promise<string | null>;
  loadProject: (projectId: string) => Promise<void>;
  updateProjectSettings: (settings: Partial<VideoProject>) => Promise<void>;
  deleteProject: () => Promise<boolean>;

  // Segment actions
  addSegment: (request: AddSegmentRequest) => Promise<VideoSegment | null>;
  uploadSegment: (file: File, insertAfterSegmentId?: string) => Promise<VideoSegment | null>;
  updateSegment: (segmentId: string, request: UpdateSegmentRequest) => Promise<VideoSegment | null>;
  removeSegment: (segmentId: string) => Promise<boolean>;
  reorderSegments: (segmentIds: string[]) => Promise<boolean>;
  splitSegment: (segmentId: string, splitTime: number) => Promise<boolean>;

  // Overlay actions
  addTextOverlay: (overlay: Omit<TextOverlay, 'id'>) => Promise<boolean>;
  addImageOverlay: (overlay: Omit<ImageOverlay, 'id'>) => Promise<boolean>;

  // Render actions
  startRender: (request?: RenderProjectRequest) => Promise<string | null>;
  checkRenderStatus: (jobId: string) => Promise<RenderJobStatus | null>;
  createPreview: (startTime: number, duration?: number) => Promise<string | null>;

  // Utility
  refreshProject: () => Promise<void>;
  clearError: () => void;
}

export function useVideoEditor(options: UseVideoEditorOptions = {}): UseVideoEditorReturn {
  const { userId = 'demo-user', onError } = options;

  const [project, setProject] = useState<VideoProject | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isRendering, setIsRendering] = useState(false);
  const [renderJob, setRenderJob] = useState<RenderJobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [supportedFormats, setSupportedFormats] = useState<SupportedFormats | null>(null);

  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;

  const handleError = useCallback((message: string) => {
    setError(message);
    onErrorRef.current?.(message);
  }, []);

  // Fetch supported formats on mount
  useEffect(() => {
    async function fetchFormats() {
      try {
        const response = await fetch(`${MEDIA_API_URL}/api/v1/editor/supported-formats`);
        if (response.ok) {
          const data = await response.json();
          setSupportedFormats(data);
        }
      } catch (e) {
        console.error('Failed to fetch supported formats:', e);
      }
    }
    fetchFormats();
  }, []);

  // ========================================
  // Project Actions
  // ========================================

  const createProject = useCallback(async (request: CreateProjectRequest): Promise<string | null> => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${MEDIA_API_URL}/api/v1/editor/projects`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to create project');
      }

      const data = await response.json();
      return data.project_id;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to create project');
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [handleError]);

  const loadProject = useCallback(async (projectId: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/editor/projects/${projectId}?user_id=${userId}`
      );

      if (!response.ok) {
        throw new Error('Project not found');
      }

      const data = await response.json();
      setProject(data);
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to load project');
    } finally {
      setIsLoading(false);
    }
  }, [userId, handleError]);

  const updateProjectSettings = useCallback(async (settings: Partial<VideoProject>) => {
    if (!project) return;

    setIsSaving(true);
    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/editor/projects/${project.id}/settings?user_id=${userId}`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(settings),
        }
      );

      if (!response.ok) {
        throw new Error('Failed to update settings');
      }

      const data = await response.json();
      setProject(data);
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to update settings');
    } finally {
      setIsSaving(false);
    }
  }, [project, userId, handleError]);

  const deleteProject = useCallback(async (): Promise<boolean> => {
    if (!project) return false;

    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/editor/projects/${project.id}?user_id=${userId}`,
        { method: 'DELETE' }
      );

      if (!response.ok) {
        throw new Error('Failed to delete project');
      }

      setProject(null);
      return true;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to delete project');
      return false;
    }
  }, [project, userId, handleError]);

  const refreshProject = useCallback(async () => {
    if (project?.id) {
      await loadProject(project.id);
    }
  }, [project?.id, loadProject]);

  // ========================================
  // Segment Actions
  // ========================================

  const addSegment = useCallback(async (request: AddSegmentRequest): Promise<VideoSegment | null> => {
    if (!project) return null;

    setIsSaving(true);
    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/editor/projects/${project.id}/segments?user_id=${userId}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(request),
        }
      );

      if (!response.ok) {
        throw new Error('Failed to add segment');
      }

      const segment = await response.json();
      await refreshProject();
      return segment;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to add segment');
      return null;
    } finally {
      setIsSaving(false);
    }
  }, [project, userId, handleError, refreshProject]);

  const uploadSegment = useCallback(async (
    file: File,
    insertAfterSegmentId?: string
  ): Promise<VideoSegment | null> => {
    if (!project) return null;

    setIsSaving(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('user_id', userId);
      if (insertAfterSegmentId) {
        formData.append('insert_after_segment_id', insertAfterSegmentId);
      }
      formData.append('title', file.name);

      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/editor/projects/${project.id}/segments/upload`,
        {
          method: 'POST',
          body: formData,
        }
      );

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to upload');
      }

      const data = await response.json();
      await refreshProject();

      // Find and return the new segment
      const updatedProject = await fetch(
        `${MEDIA_API_URL}/api/v1/editor/projects/${project.id}?user_id=${userId}`
      ).then(r => r.json());

      const newSegment = updatedProject.segments.find(
        (s: VideoSegment) => s.id === data.segment_id
      );

      return newSegment || null;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to upload segment');
      return null;
    } finally {
      setIsSaving(false);
    }
  }, [project, userId, handleError, refreshProject]);

  const updateSegment = useCallback(async (
    segmentId: string,
    request: UpdateSegmentRequest
  ): Promise<VideoSegment | null> => {
    if (!project) return null;

    setIsSaving(true);
    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/editor/projects/${project.id}/segments/${segmentId}?user_id=${userId}`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(request),
        }
      );

      if (!response.ok) {
        throw new Error('Failed to update segment');
      }

      const segment = await response.json();
      await refreshProject();
      return segment;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to update segment');
      return null;
    } finally {
      setIsSaving(false);
    }
  }, [project, userId, handleError, refreshProject]);

  const removeSegment = useCallback(async (segmentId: string): Promise<boolean> => {
    if (!project) return false;

    setIsSaving(true);
    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/editor/projects/${project.id}/segments/${segmentId}?user_id=${userId}`,
        { method: 'DELETE' }
      );

      if (!response.ok) {
        throw new Error('Failed to remove segment');
      }

      await refreshProject();
      return true;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to remove segment');
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [project, userId, handleError, refreshProject]);

  const reorderSegments = useCallback(async (segmentIds: string[]): Promise<boolean> => {
    if (!project) return false;

    setIsSaving(true);
    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/editor/projects/${project.id}/segments/reorder?user_id=${userId}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ segment_ids: segmentIds }),
        }
      );

      if (!response.ok) {
        throw new Error('Failed to reorder segments');
      }

      const data = await response.json();
      setProject(data);
      return true;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to reorder segments');
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [project, userId, handleError]);

  const splitSegment = useCallback(async (
    segmentId: string,
    splitTime: number
  ): Promise<boolean> => {
    if (!project) return false;

    setIsSaving(true);
    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/editor/projects/${project.id}/segments/${segmentId}/split?split_time=${splitTime}&user_id=${userId}`,
        { method: 'POST' }
      );

      if (!response.ok) {
        throw new Error('Failed to split segment');
      }

      await refreshProject();
      return true;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to split segment');
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [project, userId, handleError, refreshProject]);

  // ========================================
  // Overlay Actions
  // ========================================

  const addTextOverlay = useCallback(async (overlay: Omit<TextOverlay, 'id'>): Promise<boolean> => {
    if (!project) return false;

    setIsSaving(true);
    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/editor/projects/${project.id}/overlays/text?user_id=${userId}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(overlay),
        }
      );

      if (!response.ok) {
        throw new Error('Failed to add text overlay');
      }

      await refreshProject();
      return true;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to add text overlay');
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [project, userId, handleError, refreshProject]);

  const addImageOverlay = useCallback(async (overlay: Omit<ImageOverlay, 'id'>): Promise<boolean> => {
    if (!project) return false;

    setIsSaving(true);
    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/editor/projects/${project.id}/overlays/image?user_id=${userId}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(overlay),
        }
      );

      if (!response.ok) {
        throw new Error('Failed to add image overlay');
      }

      await refreshProject();
      return true;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to add image overlay');
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [project, userId, handleError, refreshProject]);

  // ========================================
  // Render Actions
  // ========================================

  const startRender = useCallback(async (request?: RenderProjectRequest): Promise<string | null> => {
    if (!project) return null;

    setIsRendering(true);
    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/editor/projects/${project.id}/render?user_id=${userId}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(request || {}),
        }
      );

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to start render');
      }

      const data = await response.json();
      setRenderJob({
        job_id: data.job_id,
        project_id: project.id,
        user_id: userId,
        status: 'pending',
        progress: 0,
        message: 'Render started',
        created_at: new Date().toISOString(),
      });

      return data.job_id;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to start render');
      setIsRendering(false);
      return null;
    }
  }, [project, userId, handleError]);

  const checkRenderStatus = useCallback(async (jobId: string): Promise<RenderJobStatus | null> => {
    try {
      const response = await fetch(`${MEDIA_API_URL}/api/v1/editor/render-jobs/${jobId}`);

      if (!response.ok) {
        throw new Error('Failed to check render status');
      }

      const data = await response.json();
      setRenderJob(data);

      if (data.status === 'completed' || data.status === 'failed') {
        setIsRendering(false);
        if (data.status === 'completed') {
          await refreshProject();
        }
      }

      return data;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to check render status');
      return null;
    }
  }, [handleError, refreshProject]);

  const createPreview = useCallback(async (
    startTime: number,
    duration: number = 10
  ): Promise<string | null> => {
    if (!project) return null;

    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/editor/projects/${project.id}/preview?start_time=${startTime}&duration=${duration}&user_id=${userId}`,
        { method: 'POST' }
      );

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to create preview');
      }

      const data = await response.json();
      return data.preview_url;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to create preview');
      return null;
    }
  }, [project, userId, handleError]);

  // ========================================
  // Utility
  // ========================================

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    // State
    project,
    isLoading,
    isSaving,
    isRendering,
    renderJob,
    error,
    supportedFormats,

    // Project actions
    createProject,
    loadProject,
    updateProjectSettings,
    deleteProject,

    // Segment actions
    addSegment,
    uploadSegment,
    updateSegment,
    removeSegment,
    reorderSegments,
    splitSegment,

    // Overlay actions
    addTextOverlay,
    addImageOverlay,

    // Render actions
    startRender,
    checkRenderStatus,
    createPreview,

    // Utility
    refreshProject,
    clearError,
  };
}
