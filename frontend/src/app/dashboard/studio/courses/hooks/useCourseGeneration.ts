'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '@/lib/api';
import type { CourseJob, CourseOutline } from '../lib/course-types';

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
      setCurrentJob(job);

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
    } catch (err) {
      console.error('Error polling job status:', err);
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
        include_avatar: data.includeAvatar,
        avatar_id: data.avatarId || undefined,
        approved_outline: approvedOutline ? transformOutlineToApi(approvedOutline) : undefined,
        // RAG document IDs (includes both legacy and source library IDs)
        document_ids: allSourceIds.length > 0 ? allSourceIds : undefined,
        // OPTIMIZED: Pass pre-fetched RAG context to avoid double-fetching
        rag_context: ragContext || undefined,
      });

      const response = transformJobResponse(rawResponse);
      setCurrentJob(response);

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
  }, [pollInterval, pollJobStatus, onError, ragContext]);

  // Reorder outline
  const reorderOutline = useCallback(async (jobId: string, sections: any[]) => {
    try {
      const result = await api.courses.reorderOutline(jobId, sections);
      if (currentJob && currentJob.jobId === jobId) {
        setCurrentJob({ ...currentJob, outline: (result as any).outline });
      }
      return result;
    } catch (err: any) {
      setError(err.message || 'Failed to reorder outline');
      throw err;
    }
  }, [currentJob]);

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
      setCurrentJob(job);
      return job;
    } catch (err) {
      console.error('Error refreshing job:', err);
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
  };
}
