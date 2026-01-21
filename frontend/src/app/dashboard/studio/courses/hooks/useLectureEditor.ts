'use client';

import { useState, useCallback } from 'react';
import type {
  LectureComponents,
  LectureComponentsResponse,
  SlideComponent,
  UpdateSlideRequest,
  RegenerateSlideRequest,
  RegenerateLectureRequest,
  RegenerateVoiceoverRequest,
  RecomposeVideoRequest,
  RegenerateResponse,
  VoiceoverComponent,
} from '../lib/lecture-editor-types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

// Transform snake_case API response to camelCase
function transformSlideComponent(data: any): SlideComponent {
  return {
    id: data.id,
    index: data.index,
    type: data.type,
    status: data.status,
    title: data.title,
    subtitle: data.subtitle,
    content: data.content,
    bulletPoints: data.bullet_points || [],
    codeBlocks: (data.code_blocks || []).map((cb: any) => ({
      id: cb.id,
      language: cb.language,
      code: cb.code,
      filename: cb.filename,
      highlightLines: cb.highlight_lines || [],
      executionOrder: cb.execution_order || 0,
      expectedOutput: cb.expected_output,
      actualOutput: cb.actual_output,
      showLineNumbers: cb.show_line_numbers ?? true,
    })),
    voiceoverText: data.voiceover_text || '',
    duration: data.duration || 10,
    transition: data.transition || 'fade',
    diagramType: data.diagram_type,
    diagramData: data.diagram_data,
    imageUrl: data.image_url,
    animationUrl: data.animation_url,
    isEdited: data.is_edited || false,
    editedAt: data.edited_at,
    editedFields: data.edited_fields || [],
    error: data.error,
  };
}

function transformVoiceoverComponent(data: any): VoiceoverComponent | undefined {
  if (!data) return undefined;
  return {
    id: data.id,
    status: data.status,
    audioUrl: data.audio_url,
    durationSeconds: data.duration_seconds || 0,
    voiceId: data.voice_id || 'alloy',
    voiceSettings: data.voice_settings || {},
    fullText: data.full_text || '',
    isCustomAudio: data.is_custom_audio || false,
    originalFilename: data.original_filename,
    isEdited: data.is_edited || false,
    editedAt: data.edited_at,
    error: data.error,
  };
}

function transformLectureComponents(data: any): LectureComponents {
  return {
    id: data.id,
    lectureId: data.lecture_id,
    jobId: data.job_id,
    slides: (data.slides || []).map(transformSlideComponent),
    voiceover: transformVoiceoverComponent(data.voiceover),
    totalDuration: data.total_duration || 0,
    generationParams: data.generation_params || {},
    presentationJobId: data.presentation_job_id,
    videoUrl: data.video_url,
    status: data.status,
    isEdited: data.is_edited || false,
    createdAt: data.created_at,
    updatedAt: data.updated_at,
    error: data.error,
  };
}

// Transform camelCase to snake_case for API requests
function transformUpdateSlideRequest(data: UpdateSlideRequest): any {
  const result: any = {};
  if (data.title !== undefined) result.title = data.title;
  if (data.subtitle !== undefined) result.subtitle = data.subtitle;
  if (data.content !== undefined) result.content = data.content;
  if (data.bulletPoints !== undefined) result.bullet_points = data.bulletPoints;
  if (data.voiceoverText !== undefined) result.voiceover_text = data.voiceoverText;
  if (data.duration !== undefined) result.duration = data.duration;
  if (data.diagramType !== undefined) result.diagram_type = data.diagramType;
  if (data.diagramData !== undefined) result.diagram_data = data.diagramData;
  if (data.codeBlocks !== undefined) {
    result.code_blocks = data.codeBlocks.map((cb) => ({
      id: cb.id,
      language: cb.language,
      code: cb.code,
      filename: cb.filename,
      highlight_lines: cb.highlightLines,
      execution_order: cb.executionOrder,
      expected_output: cb.expectedOutput,
      actual_output: cb.actualOutput,
      show_line_numbers: cb.showLineNumbers,
    }));
  }
  return result;
}

interface UseLectureEditorOptions {
  onError?: (error: string) => void;
  onSuccess?: (message: string) => void;
}

export function useLectureEditor(options: UseLectureEditorOptions = {}) {
  const [components, setComponents] = useState<LectureComponents | null>(null);
  const [selectedSlide, setSelectedSlide] = useState<SlideComponent | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isRegenerating, setIsRegenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load lecture components
  const loadComponents = useCallback(async (jobId: string, lectureId: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/courses/jobs/${jobId}/lectures/${lectureId}/components`
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to load components: ${response.status}`);
      }

      const data = await response.json();
      const transformed = transformLectureComponents(data);
      setComponents(transformed);

      // Select first slide by default
      if (transformed.slides.length > 0) {
        setSelectedSlide(transformed.slides[0]);
      }

      return transformed;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load components';
      setError(message);
      options.onError?.(message);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [options]);

  // Update a slide
  const updateSlide = useCallback(async (
    jobId: string,
    lectureId: string,
    slideId: string,
    updates: UpdateSlideRequest
  ) => {
    setIsSaving(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/courses/jobs/${jobId}/lectures/${lectureId}/slides/${slideId}`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(transformUpdateSlideRequest(updates)),
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to update slide: ${response.status}`);
      }

      const data = await response.json();
      const updatedSlide = transformSlideComponent(data.slide);

      // Update local state
      setComponents((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          isEdited: true,
          slides: prev.slides.map((s) =>
            s.id === slideId ? updatedSlide : s
          ),
        };
      });

      setSelectedSlide(updatedSlide);
      options.onSuccess?.('Slide mis \u00e0 jour');

      return updatedSlide;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to update slide';
      setError(message);
      options.onError?.(message);
      return null;
    } finally {
      setIsSaving(false);
    }
  }, [options]);

  // Regenerate a slide
  const regenerateSlide = useCallback(async (
    jobId: string,
    lectureId: string,
    slideId: string,
    request: RegenerateSlideRequest
  ) => {
    setIsRegenerating(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/courses/jobs/${jobId}/lectures/${lectureId}/slides/${slideId}/regenerate`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            regenerate_image: request.regenerateImage,
            regenerate_animation: request.regenerateAnimation,
            use_edited_content: request.useEditedContent,
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to regenerate slide: ${response.status}`);
      }

      const data = await response.json();
      const updatedSlide = transformSlideComponent(data.slide);

      // Update local state
      setComponents((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          slides: prev.slides.map((s) =>
            s.id === slideId ? updatedSlide : s
          ),
        };
      });

      setSelectedSlide(updatedSlide);
      options.onSuccess?.('Slide r\u00e9g\u00e9n\u00e9r\u00e9');

      return updatedSlide;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to regenerate slide';
      setError(message);
      options.onError?.(message);
      return null;
    } finally {
      setIsRegenerating(false);
    }
  }, [options]);

  // Regenerate voiceover
  const regenerateVoiceover = useCallback(async (
    jobId: string,
    lectureId: string,
    request: RegenerateVoiceoverRequest
  ) => {
    setIsRegenerating(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/courses/jobs/${jobId}/lectures/${lectureId}/regenerate-voiceover`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            voice_id: request.voiceId,
            voice_settings: request.voiceSettings,
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to regenerate voiceover: ${response.status}`);
      }

      const data = await response.json();

      // Update local state
      if (data.result) {
        setComponents((prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            voiceover: prev.voiceover ? {
              ...prev.voiceover,
              audioUrl: data.result.audio_url,
              durationSeconds: data.result.duration_seconds,
              isEdited: true,
            } : undefined,
          };
        });
      }

      options.onSuccess?.('Voiceover r\u00e9g\u00e9n\u00e9r\u00e9');

      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to regenerate voiceover';
      setError(message);
      options.onError?.(message);
      return null;
    } finally {
      setIsRegenerating(false);
    }
  }, [options]);

  // Upload custom audio
  const uploadCustomAudio = useCallback(async (
    jobId: string,
    lectureId: string,
    file: File
  ) => {
    setIsRegenerating(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(
        `${API_BASE_URL}/api/v1/courses/jobs/${jobId}/lectures/${lectureId}/upload-audio`,
        {
          method: 'POST',
          body: formData,
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to upload audio: ${response.status}`);
      }

      const data = await response.json();

      // Update local state
      setComponents((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          voiceover: prev.voiceover ? {
            ...prev.voiceover,
            audioUrl: data.audio_url,
            durationSeconds: data.duration_seconds,
            isCustomAudio: true,
            originalFilename: file.name,
            isEdited: true,
          } : undefined,
        };
      });

      options.onSuccess?.('Audio personnalis\u00e9 t\u00e9l\u00e9charg\u00e9');

      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to upload audio';
      setError(message);
      options.onError?.(message);
      return null;
    } finally {
      setIsRegenerating(false);
    }
  }, [options]);

  // Regenerate entire lecture
  const regenerateLecture = useCallback(async (
    jobId: string,
    lectureId: string,
    request: RegenerateLectureRequest
  ) => {
    setIsRegenerating(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/courses/jobs/${jobId}/lectures/${lectureId}/regenerate`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            use_edited_components: request.useEditedComponents,
            regenerate_voiceover: request.regenerateVoiceover,
            voice_id: request.voiceId,
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to regenerate lecture: ${response.status}`);
      }

      const data: RegenerateResponse = await response.json();

      if (data.success) {
        // Reload components to get updated data
        await loadComponents(jobId, lectureId);
        options.onSuccess?.('Le\u00e7on r\u00e9g\u00e9n\u00e9r\u00e9e');
      }

      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to regenerate lecture';
      setError(message);
      options.onError?.(message);
      return null;
    } finally {
      setIsRegenerating(false);
    }
  }, [loadComponents, options]);

  // Recompose video from current components
  const recomposeVideo = useCallback(async (
    jobId: string,
    lectureId: string,
    request: RecomposeVideoRequest
  ) => {
    setIsRegenerating(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/courses/jobs/${jobId}/lectures/${lectureId}/recompose`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            quality: request.quality,
            include_transitions: request.includeTransitions,
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to recompose video: ${response.status}`);
      }

      const data: RegenerateResponse = await response.json();

      if (data.success && data.result) {
        setComponents((prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            videoUrl: data.result?.video_url,
          };
        });
        options.onSuccess?.('Vid\u00e9o recompos\u00e9e');
      }

      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to recompose video';
      setError(message);
      options.onError?.(message);
      return null;
    } finally {
      setIsRegenerating(false);
    }
  }, [options]);

  // Retry all failed lectures
  const retryFailedLectures = useCallback(async (jobId: string) => {
    setIsRegenerating(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/courses/jobs/${jobId}/retry-failed`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to retry failed lectures: ${response.status}`);
      }

      const data: RegenerateResponse = await response.json();
      options.onSuccess?.(data.message);

      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to retry failed lectures';
      setError(message);
      options.onError?.(message);
      return null;
    } finally {
      setIsRegenerating(false);
    }
  }, [options]);

  // Select a slide
  const selectSlide = useCallback((slide: SlideComponent) => {
    setSelectedSlide(slide);
  }, []);

  // Clear state
  const clear = useCallback(() => {
    setComponents(null);
    setSelectedSlide(null);
    setError(null);
  }, []);

  // Clear error
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    // State
    components,
    selectedSlide,
    isLoading,
    isSaving,
    isRegenerating,
    error,

    // Actions
    loadComponents,
    updateSlide,
    regenerateSlide,
    regenerateVoiceover,
    uploadCustomAudio,
    regenerateLecture,
    recomposeVideo,
    retryFailedLectures,
    selectSlide,
    clear,
    clearError,
  };
}

export default useLectureEditor;
