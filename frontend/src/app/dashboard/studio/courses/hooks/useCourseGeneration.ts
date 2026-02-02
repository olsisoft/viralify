'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '@/lib/api';
import type { CourseJob, CourseOutline, ErrorQueueResponse, LessonsResponse, LessonError, SceneVideo } from '../lib/course-types';

// Transform snake_case API response to camelCase CourseJob
function transformJobResponse(data: any): CourseJob {
  return {
    jobId: data.job_id,
    status: data.status,
    currentStage: data.current_stage,
    progress: data.progress,
    message: data.message || '',
    outline: data.outline ? transformOutline(data.outline) : undefined,
    lecturesTotal: data.lectures_total || 0,
    lecturesCompleted: data.lectures_completed || 0,
    lecturesInProgress: data.lectures_in_progress || 0,
    lecturesFailed: data.lectures_failed || 0,
    lecturesCancelled: data.lectures_cancelled || 0,
    currentLectureTitle: data.current_lecture_title,
    currentLectures: data.current_lectures || [],
    outputUrls: data.output_urls || [],
    zipUrl: data.zip_url,
    createdAt: data.created_at,
    updatedAt: data.updated_at,
    completedAt: data.completed_at,
    error: data.error,
    // Failed lectures info
    failedLectureIds: data.failed_lecture_ids || [],
    failedLectureErrors: data.failed_lecture_errors || {},
    isPartialSuccess: data.is_partial_success || false,
    canDownloadPartial: data.can_download_partial || false,
    // Cancellation info
    cancelRequested: data.cancel_requested || false,
    cancelledAt: data.cancelled_at,
    // Progressive download
    sceneVideos: (data.scene_videos || []).map((sv: any) => ({
      sceneIndex: sv.scene_index,
      videoUrl: sv.video_url,
      status: sv.status,
      duration: sv.duration,
      title: sv.title,
      readyAt: sv.ready_at,
    })),
  };
}

// Transform error queue response
// Backend returns: failed_count, errors[].error_message, can_retry
// Frontend expects: totalErrors, errors[].error, canRetryAll
function transformErrorQueue(data: any): ErrorQueueResponse {
  return {
    jobId: data.job_id,
    // Backend sends "failed_count", not "total_errors"
    totalErrors: data.failed_count ?? data.total_errors ?? 0,
    errors: (data.errors || []).map((e: any) => ({
      sceneIndex: e.scene_index,
      title: e.title || '',
      // Backend sends "error_message", not "error"
      error: e.error_message ?? e.error ?? '',
      errorType: e.error_type || 'unknown',
      voiceoverText: e.voiceover_text,
      slideData: e.slide_data,
      retryCount: e.retry_count || 0,
      timestamp: e.timestamp || '',
    })),
    // Backend sends "can_retry", not "can_retry_all"
    canRetryAll: data.can_retry ?? data.can_retry_all ?? false,
    hasPartialResults: data.has_partial_results || false,
  };
}

// Transform lessons response for progressive download
// Backend returns: completed, total_lessons, lessons, final_video_url, status
// Frontend expects: readyLessons, totalLessons, lessons, allReady, finalVideoReady, finalVideoUrl
function transformLessonsResponse(data: any): LessonsResponse {
  const totalLessons = data.total_lessons || 0;
  // Backend sends "completed", not "ready_lessons"
  const readyLessons = data.completed ?? data.ready_lessons ?? 0;
  const finalVideoUrl = data.final_video_url || data.output_url;
  const status = data.status || 'unknown';

  return {
    jobId: data.job_id,
    totalLessons,
    readyLessons,
    lessons: (data.lessons || []).map((l: any) => ({
      sceneIndex: l.scene_index,
      videoUrl: l.video_url,
      status: l.status,
      duration: l.duration,
      title: l.title,
      readyAt: l.ready_at,
    })),
    // Compute allReady: all lessons are ready when completed equals total
    allReady: data.all_ready ?? (totalLessons > 0 && readyLessons >= totalLessons),
    // Compute finalVideoReady: final video is ready when URL exists and job is completed
    finalVideoReady: data.final_video_ready ?? (!!finalVideoUrl && status === 'completed'),
    finalVideoUrl,
  };
}

function transformOutline(data: any): CourseOutline {
  return {
    title: data.title,
    description: data.description,
    targetAudience: data.target_audience,
    language: data.language,
    difficultyStart: data.difficulty_start,
    difficultyEnd: data.difficulty_end,
    totalDurationMinutes: data.total_duration_minutes,
    sections: (data.sections || []).map((s: any) => ({
      id: s.id,
      title: s.title,
      description: s.description,
      order: s.order,
      lectures: (s.lectures || []).map((l: any) => ({
        id: l.id,
        title: l.title,
        description: l.description,
        objectives: l.objectives || [],
        difficulty: l.difficulty,
        durationSeconds: l.duration_seconds,
        order: l.order,
        status: l.status,
        presentationJobId: l.presentation_job_id,
        videoUrl: l.video_url,
        error: l.error,
        // Progress tracking fields
        progressPercent: l.progress_percent || 0,
        currentStage: l.current_stage,
        retryCount: l.retry_count || 0,
        // Editing support fields
        componentsId: l.components_id,
        hasComponents: l.has_components || false,
        isEdited: l.is_edited || false,
        canRegenerate: l.can_regenerate ?? true,
      })),
    })),
  };
}

// Transform frontend CourseOutline (camelCase) to backend format (snake_case)
function transformOutlineToApi(outline: CourseOutline): any {
  return {
    title: outline.title,
    description: outline.description,
    target_audience: outline.targetAudience,
    language: outline.language,
    difficulty_start: outline.difficultyStart,
    difficulty_end: outline.difficultyEnd,
    total_duration_minutes: outline.totalDurationMinutes,
    sections: (outline.sections || []).map((s) => ({
      id: s.id,
      title: s.title,
      description: s.description,
      order: s.order,
      lectures: (s.lectures || []).map((l) => ({
        id: l.id,
        title: l.title,
        description: l.description,
        objectives: l.objectives || [],
        difficulty: l.difficulty,
        duration_seconds: l.durationSeconds,
        order: l.order,
        status: l.status,
        presentation_job_id: l.presentationJobId,
        video_url: l.videoUrl,
        error: l.error,
      })),
    })),
  };
}

interface UseCourseGenerationOptions {
  pollInterval?: number;
  onComplete?: (job: CourseJob) => void;
  onError?: (error: string) => void;
}

export function useCourseGeneration(options: UseCourseGenerationOptions = {}) {
  const { pollInterval = 3000, onComplete, onError } = options;

  const [currentJob, setCurrentJob] = useState<CourseJob | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [previewOutline, setPreviewOutline] = useState<CourseOutline | null>(null);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobHistory, setJobHistory] = useState<CourseJob[]>([]);
  // OPTIMIZED: Store RAG context from preview to avoid double-fetching
  const [ragContext, setRagContext] = useState<string | null>(null);

  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  // Track current job ID to prevent stale 404 responses from clearing new job state
  const currentJobIdRef = useRef<string | null>(null);

  // Helper to update both state AND ref atomically (prevents race conditions)
  const updateCurrentJob = useCallback((job: CourseJob | null) => {
    // Update ref FIRST (synchronously) so 404 checks work immediately
    currentJobIdRef.current = job?.jobId || null;
    // Then update state (async, triggers re-render)
    setCurrentJob(job);
  }, []);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  // Poll for job status
  const pollJobStatus = useCallback(async (jobId: string) => {
    try {
      const rawJob = await api.courses.getJobStatus(jobId);
      const job = transformJobResponse(rawJob);
      updateCurrentJob(job);

      if (job.status === 'completed' || job.status === 'partial_success') {
        setIsGenerating(false);
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
        onComplete?.(job);
      } else if (job.status === 'failed') {
        setIsGenerating(false);
        setError(job.error || 'Course generation failed');
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
        onError?.(job.error || 'Course generation failed');
      }
    } catch (err: any) {
      // SIMPLIFIED: On 404, just stop polling but don't clear state
      // State will be cleared when user starts a new job
      if (err.message?.includes('404') || err.message?.includes('not found')) {
        console.log('[pollJobStatus] Job not found (404), stopping polling');
        setIsGenerating(false);
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
      } else {
        console.error('[pollJobStatus] Error:', err.message);
      }
    }
  }, [onComplete, onError]);

  // Preview outline
  const generatePreview = useCallback(async (data: any) => {
    setIsLoadingPreview(true);
    setError(null);
    setRagContext(null); // Reset RAG context

    try {
      // Extract document IDs from all uploaded documents (backend will filter by status)
      // Note: Frontend doesn't poll for status updates, so we include all non-failed docs
      const documentIds = (data.documents || [])
        .filter((doc: any) => doc.id && doc.status !== 'failed' && doc.status !== 'scan_failed' && doc.status !== 'parse_failed')
        .map((doc: any) => doc.id);

      // Combine with source library IDs
      const allSourceIds = [
        ...documentIds,
        ...(data.sourceIds || []),
      ];

      // Build context from detectedCategory or existing context
      const category = data.detectedCategory?.category || data.context?.category || 'education';
      const context = data.context || (data.detectedCategory ? {
        category: data.detectedCategory.category,
        profile_niche: data.detectedCategory.domain || data.topic,
        profile_tone: 'educational',
        profile_audience_level: data.difficultyStart || 'beginner',
        profile_language_level: 'standard',
        profile_primary_goal: 'learn',
        profile_audience_description: '',
        context_answers: data.contextAnswers || {},
        specific_tools: data.detectedCategory.tools?.join(', ') || '',
      } : null);

      // Combine keywords from detectedCategory and customKeywords
      const allKeywords = [
        ...(data.detectedCategory?.keywords || []),
        ...(data.customKeywords || []),
      ].slice(0, 10); // Max 10 keywords

      console.log('[PREVIEW] Sending context:', { category, hasContext: !!context });
      console.log('[PREVIEW] Keywords:', allKeywords);
      console.log('[PREVIEW] Tools:', data.detectedCategory?.tools);

      const rawResponse = await api.courses.previewOutline({
        profile_id: data.profileId,
        topic: data.topic,
        description: data.description,
        difficulty_start: data.difficultyStart,
        difficulty_end: data.difficultyEnd,
        structure: {
          total_duration_minutes: data.structure.totalDurationMinutes,
          number_of_sections: data.structure.numberOfSections,
          lectures_per_section: data.structure.lecturesPerSection,
          random_structure: data.structure.randomStructure,
        },
        language: data.language,
        // RAG document IDs (includes both legacy and source library IDs)
        document_ids: allSourceIds.length > 0 ? allSourceIds : undefined,
        // ✅ NEW: Context with category for adaptive elements
        context: context ? {
          category: context.category,
          profile_niche: context.profile_niche || context.profileNiche,
          profile_tone: context.profile_tone || context.profileTone || 'educational',
          profile_audience_level: context.profile_audience_level || context.profileAudienceLevel || data.difficultyStart,
          profile_language_level: context.profile_language_level || context.profileLanguageLevel || 'standard',
          profile_primary_goal: context.profile_primary_goal || context.profilePrimaryGoal || 'learn',
          profile_audience_description: context.profile_audience_description || context.profileAudienceDescription || '',
          context_answers: context.context_answers || context.contextAnswers || {},
          specific_tools: context.specific_tools || context.specificTools || '',
        } : undefined,
        // ✅ NEW: Keywords (detected + custom)
        keywords: allKeywords.length > 0 ? allKeywords : undefined,
      });

      // OPTIMIZED: New response format includes outline + rag_context
      const outline = transformOutline(rawResponse.outline || rawResponse);
      setPreviewOutline(outline);

      // Store RAG context for reuse in generate (avoids double-fetching)
      if (rawResponse.rag_context) {
        setRagContext(rawResponse.rag_context);
        console.log('[PREVIEW] RAG context cached:', rawResponse.rag_context.length, 'chars');
      }

      return outline;
    } catch (err: any) {
      const message = err.message || 'Failed to generate preview';
      setError(message);
      onError?.(message);
      throw err;
    } finally {
      setIsLoadingPreview(false);
    }
  }, [onError]);

  // Start generation
  const startGeneration = useCallback(async (data: any, approvedOutline?: CourseOutline) => {
    setIsGenerating(true);
    setError(null);

    try {
      // Extract document IDs from all uploaded documents (backend will filter by status)
      // Note: Frontend doesn't poll for status updates, so we include all non-failed docs
      const documentIds = (data.documents || [])
        .filter((doc: any) => doc.id && doc.status !== 'failed' && doc.status !== 'scan_failed' && doc.status !== 'parse_failed')
        .map((doc: any) => doc.id);

      // Combine with source library IDs
      const allSourceIds = [
        ...documentIds,
        ...(data.sourceIds || []),
      ];

      // Build context from detectedCategory or existing context
      const category = data.detectedCategory?.category || data.context?.category || 'education';
      const context = data.context || (data.detectedCategory ? {
        category: data.detectedCategory.category,
        profile_niche: data.detectedCategory.domain || data.topic,
        profile_tone: 'educational',
        profile_audience_level: data.difficultyStart || 'beginner',
        profile_language_level: 'standard',
        profile_primary_goal: 'learn',
        profile_audience_description: '',
        context_answers: data.contextAnswers || {},
        specific_tools: data.detectedCategory.tools?.join(', ') || '',
      } : null);

      // Combine keywords from detectedCategory and customKeywords
      const allKeywords = [
        ...(data.detectedCategory?.keywords || []),
        ...(data.customKeywords || []),
      ].slice(0, 10); // Max 10 keywords

      // Debug: Log all parameters being sent to backend
      console.log('[GENERATE] data.documents:', data.documents?.length || 0, 'data.sourceIds:', data.sourceIds?.length || 0);
      console.log('[GENERATE] allSourceIds being sent:', allSourceIds);
      console.log('[GENERATE] Context:', { category, hasContext: !!context });
      console.log('[GENERATE] Keywords:', allKeywords);
      console.log('[GENERATE] Quiz config:', data.quizConfig);
      console.log('[GENERATE] Adaptive elements:', data.adaptiveElements);

      const rawResponse = await api.courses.generate({
        profile_id: data.profileId,
        topic: data.topic,
        description: data.description,
        difficulty_start: data.difficultyStart,
        difficulty_end: data.difficultyEnd,
        structure: {
          total_duration_minutes: data.structure.totalDurationMinutes,
          number_of_sections: data.structure.numberOfSections,
          lectures_per_section: data.structure.lecturesPerSection,
          random_structure: data.structure.randomStructure,
        },
        lesson_elements: {
          concept_intro: data.lessonElements.conceptIntro,
          diagram_schema: data.lessonElements.diagramSchema,
          code_typing: data.lessonElements.codeTyping,
          code_execution: data.lessonElements.codeExecution,
          voiceover_explanation: data.lessonElements.voiceoverExplanation,
          curriculum_slide: data.lessonElements.curriculumSlide,
        },
        language: data.language,
        voice_id: data.voiceId,
        style: data.style,
        typing_speed: data.typingSpeed,
        title_style: data.titleStyle,  // ✅ Added: title style for slides
        include_avatar: data.includeAvatar,
        avatar_id: data.avatarId || undefined,
        approved_outline: approvedOutline ? transformOutlineToApi(approvedOutline) : undefined,
        // RAG document IDs (includes both legacy and source library IDs)
        document_ids: allSourceIds.length > 0 ? allSourceIds : undefined,
        // OPTIMIZED: Pass pre-fetched RAG context to avoid double-fetching
        rag_context: ragContext || undefined,
        // ✅ NEW: Context with category for adaptive elements
        context: context ? {
          category: context.category,
          profile_niche: context.profile_niche || context.profileNiche,
          profile_tone: context.profile_tone || context.profileTone || 'educational',
          profile_audience_level: context.profile_audience_level || context.profileAudienceLevel || data.difficultyStart,
          profile_language_level: context.profile_language_level || context.profileLanguageLevel || 'standard',
          profile_primary_goal: context.profile_primary_goal || context.profilePrimaryGoal || 'learn',
          profile_audience_description: context.profile_audience_description || context.profileAudienceDescription || '',
          context_answers: context.context_answers || context.contextAnswers || {},
          specific_tools: context.specific_tools || context.specificTools || '',
        } : undefined,
        // ✅ NEW: Keywords (detected + custom)
        keywords: allKeywords.length > 0 ? allKeywords : undefined,
        // ✅ NEW: Quiz configuration
        quiz_config: data.quizConfig ? {
          enabled: data.quizConfig.enabled,
          frequency: data.quizConfig.frequency,
          custom_frequency: data.quizConfig.customFrequency,
          questions_per_quiz: data.quizConfig.questionsPerQuiz,
          question_types: data.quizConfig.questionTypes,
          passing_score: data.quizConfig.passingScore,
          show_explanations: data.quizConfig.showExplanations,
          allow_retry: data.quizConfig.allowRetry,
        } : undefined,
        // ✅ NEW: Adaptive elements configuration
        adaptive_elements: data.adaptiveElements ? {
          common_elements: data.adaptiveElements.commonElements,
          category_elements: data.adaptiveElements.categoryElements,
          use_ai_suggestions: data.adaptiveElements.useAiSuggestions,
        } : undefined,
      });

      const response = transformJobResponse(rawResponse);
      updateCurrentJob(response);

      // Start polling
      pollIntervalRef.current = setInterval(() => {
        pollJobStatus(response.jobId);
      }, pollInterval);

      // Initial poll
      await pollJobStatus(response.jobId);

      return response;
    } catch (err: any) {
      setIsGenerating(false);
      const message = err.message || 'Failed to start generation';
      setError(message);
      onError?.(message);
      throw err;
    }
  }, [pollInterval, pollJobStatus, onError, ragContext, updateCurrentJob]);

  // Reorder outline
  const reorderOutline = useCallback(async (jobId: string, sections: any[]) => {
    try {
      const result = await api.courses.reorderOutline(jobId, sections);
      if (currentJob && currentJob.jobId === jobId) {
        updateCurrentJob({ ...currentJob, outline: (result as any).outline });
      }
      return result;
    } catch (err: any) {
      setError(err.message || 'Failed to reorder outline');
      throw err;
    }
  }, [currentJob, updateCurrentJob]);

  // Fetch job history
  const fetchHistory = useCallback(async (limit: number = 20) => {
    try {
      const rawJobs = await api.courses.listJobs(limit) as any[];
      const jobs = rawJobs.map(transformJobResponse);
      setJobHistory(jobs);
      return jobs;
    } catch (err: any) {
      console.error('Error fetching history:', err);
      return [];
    }
  }, []);

  // Cancel polling
  const cancelPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
    setIsGenerating(false);
  }, []);

  // Clear preview (also clears cached RAG context)
  const clearPreview = useCallback(() => {
    setPreviewOutline(null);
    setRagContext(null);
  }, []);

  // Clear error
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Refresh job status (useful after editing lectures)
  const refreshJob = useCallback(async (jobId: string) => {
    try {
      const rawJob = await api.courses.getJobStatus(jobId);
      const job = transformJobResponse(rawJob);
      updateCurrentJob(job);
      return job;
    } catch (err: any) {
      // SIMPLIFIED: Just log and return null
      console.log('[refreshJob] Error:', err.message);
      return null;
    }
  }, [updateCurrentJob]);

  // ==========================================
  // Job Management Methods
  // ==========================================

  // Get error queue for a job
  const getErrors = useCallback(async (jobId: string): Promise<ErrorQueueResponse | null> => {
    try {
      const rawData = await api.courses.getErrors(jobId);
      return transformErrorQueue(rawData);
    } catch (err) {
      console.error('Error fetching error queue:', err);
      return null;
    }
  }, []);

  // Update lesson content before retry
  const updateLessonContent = useCallback(async (
    jobId: string,
    sceneIndex: number,
    content: { voiceoverText?: string; title?: string; slideData?: any }
  ) => {
    try {
      const apiContent = {
        voiceover_text: content.voiceoverText,
        title: content.title,
        slide_data: content.slideData,
      };
      const result = await api.courses.updateLessonContent(jobId, sceneIndex, apiContent);
      // Refresh job after update
      await refreshJob(jobId);
      return result;
    } catch (err: any) {
      setError(err.message || 'Failed to update lesson content');
      throw err;
    }
  }, [refreshJob]);

  // Retry a single lesson
  const retryLesson = useCallback(async (jobId: string, sceneIndex: number, rebuildFinal: boolean = true) => {
    try {
      setIsGenerating(true);
      const result = await api.courses.retryLesson(jobId, sceneIndex, { rebuild_final: rebuildFinal });

      // Start polling again
      pollIntervalRef.current = setInterval(() => {
        pollJobStatus(jobId);
      }, pollInterval);

      return result;
    } catch (err: any) {
      setIsGenerating(false);
      setError(err.message || 'Failed to retry lesson');
      throw err;
    }
  }, [pollInterval, pollJobStatus]);

  // Retry all failed lessons
  const retryAllFailed = useCallback(async (jobId: string) => {
    try {
      setIsGenerating(true);
      const result = await api.courses.retryAllFailed(jobId);

      // Start polling again
      pollIntervalRef.current = setInterval(() => {
        pollJobStatus(jobId);
      }, pollInterval);

      return result;
    } catch (err: any) {
      setIsGenerating(false);
      setError(err.message || 'Failed to retry failed lessons');
      throw err;
    }
  }, [pollInterval, pollJobStatus]);

  // Cancel job gracefully
  const cancelJob = useCallback(async (jobId: string, keepCompleted: boolean = true) => {
    try {
      const result = await api.courses.cancelJob(jobId, { keep_completed: keepCompleted });
      // Stop polling
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
      setIsGenerating(false);
      // Refresh job to get updated status
      await refreshJob(jobId);
      return result;
    } catch (err: any) {
      setError(err.message || 'Failed to cancel job');
      throw err;
    }
  }, [refreshJob]);

  // Rebuild final video from completed lessons
  const rebuildVideo = useCallback(async (jobId: string) => {
    try {
      setIsGenerating(true);
      const result = await api.courses.rebuildVideo(jobId);

      // Start polling again
      pollIntervalRef.current = setInterval(() => {
        pollJobStatus(jobId);
      }, pollInterval);

      return result;
    } catch (err: any) {
      setIsGenerating(false);
      setError(err.message || 'Failed to rebuild video');
      throw err;
    }
  }, [pollInterval, pollJobStatus]);

  // Get lessons for progressive download
  // SIMPLIFIED: Just return null on error, let GenerationProgress handle polling stop
  const getLessons = useCallback(async (jobId: string): Promise<LessonsResponse | null> => {
    try {
      const rawData = await api.courses.getLessons(jobId);
      return transformLessonsResponse(rawData);
    } catch (err: any) {
      // Just log the error and return null
      // GenerationProgress will stop polling when it receives null
      if (err.message?.includes('404')) {
        console.log('[getLessons] Job not found (404), returning null');
      } else {
        console.error('[getLessons] Error:', err.message);
      }
      return null;
    }
  }, []);

  return {
    // State
    currentJob,
    isGenerating,
    previewOutline,
    isLoadingPreview,
    error,
    jobHistory,

    // Actions
    generatePreview,
    startGeneration,
    reorderOutline,
    fetchHistory,
    cancelPolling,
    clearPreview,
    clearError,
    refreshJob,

    // Job Management Actions
    getErrors,
    updateLessonContent,
    retryLesson,
    retryAllFailed,
    cancelJob,
    rebuildVideo,
    getLessons,
  };
}
