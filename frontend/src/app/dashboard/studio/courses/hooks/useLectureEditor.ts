'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import type {
  LectureComponents,
  LectureComponentsResponse,
  SlideComponent,
  SlideElement,
  UpdateSlideRequest,
  RegenerateSlideRequest,
  RegenerateLectureRequest,
  RegenerateVoiceoverRequest,
  RecomposeVideoRequest,
  RegenerateResponse,
  VoiceoverComponent,
  MediaType,
  EditorActionType,
  AddElementRequest,
  UpdateElementRequest,
} from '../lib/lecture-editor-types';
import { useEditorHistory } from './useEditorHistory';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

// Transform element from snake_case to camelCase
function transformSlideElement(data: any): SlideElement {
  return {
    id: data.id,
    type: data.type,
    x: data.x,
    y: data.y,
    width: data.width,
    height: data.height,
    rotation: data.rotation || 0,
    zIndex: data.z_index || 0,
    locked: data.locked || false,
    visible: data.visible ?? true,
    imageContent: data.image_content ? {
      url: data.image_content.url,
      originalFilename: data.image_content.original_filename,
      fit: data.image_content.fit || 'cover',
      opacity: data.image_content.opacity ?? 1,
      borderRadius: data.image_content.border_radius || 0,
      crop: data.image_content.crop,
    } : undefined,
    textContent: data.text_content ? {
      text: data.text_content.text,
      fontSize: data.text_content.font_size || 16,
      fontWeight: data.text_content.font_weight || 'normal',
      fontFamily: data.text_content.font_family || 'Inter',
      color: data.text_content.color || '#FFFFFF',
      backgroundColor: data.text_content.background_color,
      textAlign: data.text_content.text_align || 'left',
      lineHeight: data.text_content.line_height || 1.5,
      padding: data.text_content.padding || 8,
    } : undefined,
    shapeContent: data.shape_content ? {
      shape: data.shape_content.shape,
      fillColor: data.shape_content.fill_color || '#6366F1',
      strokeColor: data.shape_content.stroke_color,
      strokeWidth: data.shape_content.stroke_width || 0,
      opacity: data.shape_content.opacity ?? 1,
      borderRadius: data.shape_content.border_radius || 0,
    } : undefined,
    createdAt: data.created_at,
    updatedAt: data.updated_at,
  };
}

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
    mediaType: data.media_type,
    mediaUrl: data.media_url,
    mediaThumbnailUrl: data.media_thumbnail_url,
    mediaOriginalFilename: data.media_original_filename,
    elements: (data.elements || []).map(transformSlideElement),
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

  // History management for undo/redo
  const {
    canUndo,
    canRedo,
    historyLength,
    futureLength,
    initialize: initializeHistory,
    pushAction,
    undo: undoHistory,
    redo: redoHistory,
    clear: clearHistory,
  } = useEditorHistory({ maxHistory: 50 });

  // Use refs for callbacks to avoid infinite loops in useCallback dependencies
  const onErrorRef = useRef(options.onError);
  const onSuccessRef = useRef(options.onSuccess);
  const componentsRef = useRef(components);

  // Keep refs updated
  useEffect(() => {
    onErrorRef.current = options.onError;
    onSuccessRef.current = options.onSuccess;
  }, [options.onError, options.onSuccess]);

  // Keep components ref updated
  useEffect(() => {
    componentsRef.current = components;
  }, [components]);

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
      initializeHistory(transformed);

      // Select first slide by default
      if (transformed.slides.length > 0) {
        setSelectedSlide(transformed.slides[0]);
      }

      return transformed;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load components';
      setError(message);
      onErrorRef.current?.(message);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []); // No dependencies - uses refs for callbacks

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

      // Capture previous state for history
      const previousSlide = componentsRef.current?.slides.find(s => s.id === slideId);

      // Update local state
      setComponents((prev) => {
        if (!prev) return prev;
        const newComponents = {
          ...prev,
          isEdited: true,
          slides: prev.slides.map((s) =>
            s.id === slideId ? updatedSlide : s
          ),
        };

        // Push to history
        if (previousSlide) {
          pushAction('update_slide', slideId, previousSlide, updatedSlide, newComponents);
        }

        return newComponents;
      });

      setSelectedSlide(updatedSlide);
      onSuccessRef.current?.('Slide mis \u00e0 jour');

      return updatedSlide;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to update slide';
      setError(message);
      onErrorRef.current?.(message);
      return null;
    } finally {
      setIsSaving(false);
    }
  }, [pushAction]); // Add pushAction dependency

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
      onSuccessRef.current?.('Slide r\u00e9g\u00e9n\u00e9r\u00e9');

      return updatedSlide;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to regenerate slide';
      setError(message);
      onErrorRef.current?.(message);
      return null;
    } finally {
      setIsRegenerating(false);
    }
  }, []); // No dependencies - uses refs for callbacks

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

      onSuccessRef.current?.('Voiceover r\u00e9g\u00e9n\u00e9r\u00e9');

      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to regenerate voiceover';
      setError(message);
      onErrorRef.current?.(message);
      return null;
    } finally {
      setIsRegenerating(false);
    }
  }, []); // No dependencies - uses refs for callbacks

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

      onSuccessRef.current?.('Audio personnalis\u00e9 t\u00e9l\u00e9charg\u00e9');

      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to upload audio';
      setError(message);
      onErrorRef.current?.(message);
      return null;
    } finally {
      setIsRegenerating(false);
    }
  }, []); // No dependencies - uses refs for callbacks

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
        onSuccessRef.current?.('Le\u00e7on r\u00e9g\u00e9n\u00e9r\u00e9e');
      }

      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to regenerate lecture';
      setError(message);
      onErrorRef.current?.(message);
      return null;
    } finally {
      setIsRegenerating(false);
    }
  }, [loadComponents]); // Only loadComponents dependency

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
            videoUrl: data.result?.video_url as string | undefined,
          };
        });
        onSuccessRef.current?.('Vidéo recomposée');
      }

      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to recompose video';
      setError(message);
      onErrorRef.current?.(message);
      return null;
    } finally {
      setIsRegenerating(false);
    }
  }, []); // No dependencies - uses refs for callbacks

  // Reorder a slide
  const reorderSlide = useCallback(async (
    jobId: string,
    lectureId: string,
    slideId: string,
    newIndex: number
  ) => {
    setIsSaving(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/courses/jobs/${jobId}/lectures/${lectureId}/slides/${slideId}/reorder`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ new_index: newIndex }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to reorder slide: ${response.status}`);
      }

      const data = await response.json();

      // Reload components to get updated order
      await loadComponents(jobId, lectureId);
      onSuccessRef.current?.('Slide réordonné');

      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to reorder slide';
      setError(message);
      onErrorRef.current?.(message);
      return null;
    } finally {
      setIsSaving(false);
    }
  }, [loadComponents]);

  // Delete a slide
  const deleteSlide = useCallback(async (
    jobId: string,
    lectureId: string,
    slideId: string
  ) => {
    setIsSaving(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/courses/jobs/${jobId}/lectures/${lectureId}/slides/${slideId}`,
        {
          method: 'DELETE',
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to delete slide: ${response.status}`);
      }

      const data = await response.json();

      // Capture deleted slide for history
      const deletedSlide = componentsRef.current?.slides.find(s => s.id === slideId);
      const previousSlides = componentsRef.current?.slides ? [...componentsRef.current.slides] : [];

      // Update local state - remove the deleted slide
      setComponents((prev) => {
        if (!prev) return prev;
        const newSlides = prev.slides.filter((s) => s.id !== slideId);
        // Update indices
        newSlides.forEach((s, i) => { s.index = i; });
        const newComponents = {
          ...prev,
          isEdited: true,
          slides: newSlides,
        };

        // Push to history
        if (deletedSlide) {
          pushAction('delete_slide', slideId, previousSlides, newSlides, newComponents);
        }

        return newComponents;
      });

      // If deleted slide was selected, select another
      if (selectedSlide?.id === slideId) {
        setSelectedSlide((prev) => {
          if (!componentsRef.current) return null;
          const remaining = componentsRef.current.slides.filter((s) => s.id !== slideId);
          return remaining.length > 0 ? remaining[0] : null;
        });
      }

      onSuccessRef.current?.('Slide supprimé');

      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete slide';
      setError(message);
      onErrorRef.current?.(message);
      return null;
    } finally {
      setIsSaving(false);
    }
  }, [selectedSlide, pushAction]);

  // Insert a media slide
  const insertMediaSlide = useCallback(async (
    jobId: string,
    lectureId: string,
    mediaType: MediaType,
    file: File,
    options?: {
      insertAfterSlideId?: string;
      title?: string;
      voiceoverText?: string;
      duration?: number;
    }
  ) => {
    setIsSaving(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('media_type', mediaType);
      if (options?.insertAfterSlideId) {
        formData.append('insert_after_slide_id', options.insertAfterSlideId);
      }
      if (options?.title) {
        formData.append('title', options.title);
      }
      if (options?.voiceoverText) {
        formData.append('voiceover_text', options.voiceoverText);
      }
      if (options?.duration !== undefined) {
        formData.append('duration', options.duration.toString());
      }

      const response = await fetch(
        `${API_BASE_URL}/api/v1/courses/jobs/${jobId}/lectures/${lectureId}/slides/insert-media`,
        {
          method: 'POST',
          body: formData,
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to insert media slide: ${response.status}`);
      }

      const data = await response.json();

      // Reload components to get updated slides
      await loadComponents(jobId, lectureId);
      onSuccessRef.current?.('Slide média ajouté');

      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to insert media slide';
      setError(message);
      onErrorRef.current?.(message);
      return null;
    } finally {
      setIsSaving(false);
    }
  }, [loadComponents]);

  // Upload media to existing slide
  const uploadMediaToSlide = useCallback(async (
    jobId: string,
    lectureId: string,
    slideId: string,
    mediaType: MediaType,
    file: File
  ) => {
    setIsSaving(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('media_type', mediaType);

      const response = await fetch(
        `${API_BASE_URL}/api/v1/courses/jobs/${jobId}/lectures/${lectureId}/slides/${slideId}/upload-media`,
        {
          method: 'POST',
          body: formData,
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to upload media: ${response.status}`);
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
      onSuccessRef.current?.('Média uploadé');

      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to upload media';
      setError(message);
      onErrorRef.current?.(message);
      return null;
    } finally {
      setIsSaving(false);
    }
  }, []);

  // =========================================================================
  // Element Management (for positionable images, text, shapes)
  // =========================================================================

  // Add element to slide
  const addElement = useCallback(async (
    jobId: string,
    lectureId: string,
    slideId: string,
    request: AddElementRequest
  ) => {
    setIsSaving(true);
    setError(null);

    try {
      // Transform camelCase to snake_case for API
      const apiRequest: any = {
        type: request.type,
        x: request.x,
        y: request.y,
        width: request.width,
        height: request.height,
      };

      if (request.imageContent) {
        apiRequest.image_content = {
          url: request.imageContent.url,
          original_filename: request.imageContent.originalFilename,
          fit: request.imageContent.fit,
          opacity: request.imageContent.opacity,
          border_radius: request.imageContent.borderRadius,
          crop: request.imageContent.crop,
        };
      }
      if (request.textContent) {
        apiRequest.text_content = {
          text: request.textContent.text,
          font_size: request.textContent.fontSize,
          font_weight: request.textContent.fontWeight,
          font_family: request.textContent.fontFamily,
          color: request.textContent.color,
          background_color: request.textContent.backgroundColor,
          text_align: request.textContent.textAlign,
          line_height: request.textContent.lineHeight,
          padding: request.textContent.padding,
        };
      }
      if (request.shapeContent) {
        apiRequest.shape_content = {
          shape: request.shapeContent.shape,
          fill_color: request.shapeContent.fillColor,
          stroke_color: request.shapeContent.strokeColor,
          stroke_width: request.shapeContent.strokeWidth,
          opacity: request.shapeContent.opacity,
          border_radius: request.shapeContent.borderRadius,
        };
      }

      const response = await fetch(
        `${API_BASE_URL}/api/v1/courses/jobs/${jobId}/lectures/${lectureId}/slides/${slideId}/elements`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(apiRequest),
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to add element: ${response.status}`);
      }

      const data = await response.json();
      const newElement = transformSlideElement(data.element);

      // Update local state
      setComponents((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          isEdited: true,
          slides: prev.slides.map((s) =>
            s.id === slideId
              ? { ...s, elements: [...s.elements, newElement], isEdited: true }
              : s
          ),
        };
      });

      // Update selected slide if it's the one we modified
      if (selectedSlide?.id === slideId) {
        setSelectedSlide((prev) => prev ? {
          ...prev,
          elements: [...prev.elements, newElement],
          isEdited: true,
        } : prev);
      }

      onSuccessRef.current?.('Élément ajouté');
      return newElement;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to add element';
      setError(message);
      onErrorRef.current?.(message);
      return null;
    } finally {
      setIsSaving(false);
    }
  }, [selectedSlide]);

  // Update element position, size, or content
  const updateElement = useCallback(async (
    jobId: string,
    lectureId: string,
    slideId: string,
    elementId: string,
    updates: UpdateElementRequest
  ) => {
    setIsSaving(true);
    setError(null);

    try {
      // Transform camelCase to snake_case
      const apiRequest: any = {};
      if (updates.x !== undefined) apiRequest.x = updates.x;
      if (updates.y !== undefined) apiRequest.y = updates.y;
      if (updates.width !== undefined) apiRequest.width = updates.width;
      if (updates.height !== undefined) apiRequest.height = updates.height;
      if (updates.rotation !== undefined) apiRequest.rotation = updates.rotation;
      if (updates.locked !== undefined) apiRequest.locked = updates.locked;
      if (updates.visible !== undefined) apiRequest.visible = updates.visible;

      if (updates.imageContent) {
        apiRequest.image_content = {
          url: updates.imageContent.url,
          original_filename: updates.imageContent.originalFilename,
          fit: updates.imageContent.fit,
          opacity: updates.imageContent.opacity,
          border_radius: updates.imageContent.borderRadius,
          crop: updates.imageContent.crop,
        };
      }
      if (updates.textContent) {
        apiRequest.text_content = {
          text: updates.textContent.text,
          font_size: updates.textContent.fontSize,
          font_weight: updates.textContent.fontWeight,
          font_family: updates.textContent.fontFamily,
          color: updates.textContent.color,
          background_color: updates.textContent.backgroundColor,
          text_align: updates.textContent.textAlign,
          line_height: updates.textContent.lineHeight,
          padding: updates.textContent.padding,
        };
      }
      if (updates.shapeContent) {
        apiRequest.shape_content = {
          shape: updates.shapeContent.shape,
          fill_color: updates.shapeContent.fillColor,
          stroke_color: updates.shapeContent.strokeColor,
          stroke_width: updates.shapeContent.strokeWidth,
          opacity: updates.shapeContent.opacity,
          border_radius: updates.shapeContent.borderRadius,
        };
      }

      const response = await fetch(
        `${API_BASE_URL}/api/v1/courses/jobs/${jobId}/lectures/${lectureId}/slides/${slideId}/elements/${elementId}`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(apiRequest),
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to update element: ${response.status}`);
      }

      const data = await response.json();
      const updatedElement = transformSlideElement(data.element);

      // Update local state
      setComponents((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          isEdited: true,
          slides: prev.slides.map((s) =>
            s.id === slideId
              ? {
                  ...s,
                  elements: s.elements.map((e) => e.id === elementId ? updatedElement : e),
                  isEdited: true,
                }
              : s
          ),
        };
      });

      // Update selected slide
      if (selectedSlide?.id === slideId) {
        setSelectedSlide((prev) => prev ? {
          ...prev,
          elements: prev.elements.map((e) => e.id === elementId ? updatedElement : e),
          isEdited: true,
        } : prev);
      }

      return updatedElement;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to update element';
      setError(message);
      onErrorRef.current?.(message);
      return null;
    } finally {
      setIsSaving(false);
    }
  }, [selectedSlide]);

  // Delete element from slide
  const deleteElement = useCallback(async (
    jobId: string,
    lectureId: string,
    slideId: string,
    elementId: string
  ) => {
    setIsSaving(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/courses/jobs/${jobId}/lectures/${lectureId}/slides/${slideId}/elements/${elementId}`,
        { method: 'DELETE' }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to delete element: ${response.status}`);
      }

      // Update local state
      setComponents((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          isEdited: true,
          slides: prev.slides.map((s) =>
            s.id === slideId
              ? {
                  ...s,
                  elements: s.elements.filter((e) => e.id !== elementId),
                  isEdited: true,
                }
              : s
          ),
        };
      });

      // Update selected slide
      if (selectedSlide?.id === slideId) {
        setSelectedSlide((prev) => prev ? {
          ...prev,
          elements: prev.elements.filter((e) => e.id !== elementId),
          isEdited: true,
        } : prev);
      }

      onSuccessRef.current?.('Élément supprimé');
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete element';
      setError(message);
      onErrorRef.current?.(message);
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [selectedSlide]);

  // Upload image and add as element
  const addImageElement = useCallback(async (
    jobId: string,
    lectureId: string,
    slideId: string,
    file: File,
    position?: { x: number; y: number }
  ) => {
    setIsSaving(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      if (position) {
        formData.append('x', position.x.toString());
        formData.append('y', position.y.toString());
      }

      const response = await fetch(
        `${API_BASE_URL}/api/v1/courses/jobs/${jobId}/lectures/${lectureId}/slides/${slideId}/elements/upload-image`,
        {
          method: 'POST',
          body: formData,
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to upload image: ${response.status}`);
      }

      const data = await response.json();
      const newElement = transformSlideElement(data.element);

      // Update local state
      setComponents((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          isEdited: true,
          slides: prev.slides.map((s) =>
            s.id === slideId
              ? { ...s, elements: [...s.elements, newElement], isEdited: true }
              : s
          ),
        };
      });

      if (selectedSlide?.id === slideId) {
        setSelectedSlide((prev) => prev ? {
          ...prev,
          elements: [...prev.elements, newElement],
          isEdited: true,
        } : prev);
      }

      onSuccessRef.current?.('Image ajoutée');
      return newElement;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to add image';
      setError(message);
      onErrorRef.current?.(message);
      return null;
    } finally {
      setIsSaving(false);
    }
  }, [selectedSlide]);

  // Duplicate element (create a copy with offset)
  const duplicateElement = useCallback(async (
    jobId: string,
    lectureId: string,
    slideId: string,
    element: SlideElement
  ) => {
    // Create a new element based on the original
    const request: AddElementRequest = {
      type: element.type,
      x: element.x,
      y: element.y,
      width: element.width,
      height: element.height,
    };

    // Copy content based on type
    if (element.type === 'image' && element.imageContent) {
      request.imageContent = { ...element.imageContent };
    } else if (element.type === 'text_block' && element.textContent) {
      request.textContent = { ...element.textContent };
    } else if (element.type === 'shape' && element.shapeContent) {
      request.shapeContent = { ...element.shapeContent };
    }

    return addElement(jobId, lectureId, slideId, request);
  }, [addElement]);

  // Bring element to front (highest z-index)
  const bringElementToFront = useCallback(async (
    jobId: string,
    lectureId: string,
    slideId: string,
    elementId: string
  ) => {
    setIsSaving(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/courses/jobs/${jobId}/lectures/${lectureId}/slides/${slideId}/elements/${elementId}/bring-to-front`,
        { method: 'POST' }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to bring element to front: ${response.status}`);
      }

      const data = await response.json();

      // Update local state - reload elements order
      setComponents((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          isEdited: true,
          slides: prev.slides.map((s) =>
            s.id === slideId
              ? {
                  ...s,
                  elements: s.elements.map((e) =>
                    e.id === elementId
                      ? { ...e, zIndex: Math.max(...s.elements.map((el) => el.zIndex)) + 1 }
                      : e
                  ),
                  isEdited: true,
                }
              : s
          ),
        };
      });

      if (selectedSlide?.id === slideId) {
        setSelectedSlide((prev) => prev ? {
          ...prev,
          elements: prev.elements.map((e) =>
            e.id === elementId
              ? { ...e, zIndex: Math.max(...prev.elements.map((el) => el.zIndex)) + 1 }
              : e
          ),
          isEdited: true,
        } : prev);
      }

      onSuccessRef.current?.('Élément mis au premier plan');
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to bring element to front';
      setError(message);
      onErrorRef.current?.(message);
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [selectedSlide]);

  // Send element to back (lowest z-index)
  const sendElementToBack = useCallback(async (
    jobId: string,
    lectureId: string,
    slideId: string,
    elementId: string
  ) => {
    setIsSaving(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/courses/jobs/${jobId}/lectures/${lectureId}/slides/${slideId}/elements/${elementId}/send-to-back`,
        { method: 'POST' }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to send element to back: ${response.status}`);
      }

      const data = await response.json();

      // Update local state - set z-index to minimum
      setComponents((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          isEdited: true,
          slides: prev.slides.map((s) =>
            s.id === slideId
              ? {
                  ...s,
                  elements: s.elements.map((e) =>
                    e.id === elementId
                      ? { ...e, zIndex: Math.min(...s.elements.map((el) => el.zIndex)) - 1 }
                      : e
                  ),
                  isEdited: true,
                }
              : s
          ),
        };
      });

      if (selectedSlide?.id === slideId) {
        setSelectedSlide((prev) => prev ? {
          ...prev,
          elements: prev.elements.map((e) =>
            e.id === elementId
              ? { ...e, zIndex: Math.min(...prev.elements.map((el) => el.zIndex)) - 1 }
              : e
          ),
          isEdited: true,
        } : prev);
      }

      onSuccessRef.current?.('Élément mis en arrière-plan');
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to send element to back';
      setError(message);
      onErrorRef.current?.(message);
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [selectedSlide]);

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
      onSuccessRef.current?.(data.message);

      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to retry failed lectures';
      setError(message);
      onErrorRef.current?.(message);
      return null;
    } finally {
      setIsRegenerating(false);
    }
  }, []); // No dependencies - uses refs for callbacks

  // Select a slide
  const selectSlide = useCallback((slide: SlideComponent) => {
    setSelectedSlide(slide);
  }, []);

  // Clear state
  const clear = useCallback(() => {
    setComponents(null);
    setSelectedSlide(null);
    setError(null);
    clearHistory();
  }, [clearHistory]);

  // Undo last action
  const undo = useCallback(() => {
    const previousState = undoHistory();
    if (previousState) {
      setComponents(previousState);
      // Update selected slide if it still exists
      if (selectedSlide) {
        const stillExists = previousState.slides.find(s => s.id === selectedSlide.id);
        if (stillExists) {
          setSelectedSlide(stillExists);
        } else if (previousState.slides.length > 0) {
          setSelectedSlide(previousState.slides[0]);
        } else {
          setSelectedSlide(null);
        }
      }
      onSuccessRef.current?.('Action annulée');
    }
  }, [undoHistory, selectedSlide]);

  // Redo last undone action
  const redo = useCallback(() => {
    const nextState = redoHistory();
    if (nextState) {
      setComponents(nextState);
      // Update selected slide if it still exists
      if (selectedSlide) {
        const stillExists = nextState.slides.find(s => s.id === selectedSlide.id);
        if (stillExists) {
          setSelectedSlide(stillExists);
        } else if (nextState.slides.length > 0) {
          setSelectedSlide(nextState.slides[0]);
        } else {
          setSelectedSlide(null);
        }
      }
      onSuccessRef.current?.('Action rétablie');
    }
  }, [redoHistory, selectedSlide]);

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

    // History state
    canUndo,
    canRedo,
    historyLength,
    futureLength,

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
    // Slide management
    reorderSlide,
    deleteSlide,
    insertMediaSlide,
    uploadMediaToSlide,
    // Element management (for canvas)
    addElement,
    updateElement,
    deleteElement,
    addImageElement,
    duplicateElement,
    bringElementToFront,
    sendElementToBack,
    // History actions
    undo,
    redo,
  };
}

export default useLectureEditor;
