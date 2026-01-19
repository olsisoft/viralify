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
    currentLectureTitle: data.current_lecture_title,
    outputUrls: data.output_urls || [],
    zipUrl: data.zip_url,
    createdAt: data.created_at,
    updatedAt: data.updated_at,
    completedAt: data.completed_at,
    error: data.error,
  };
}

function transformOutline(data: any): CourseOutline {
  return {
    title: data.title,
    description: data.description,
    targetAudience: data.target_audience,
    category: data.category,
    contextSummary: data.context_summary,
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

      if (job.status === 'completed') {
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

    try {
      // Build context for API if available
      const contextPayload = data.context
        ? {
            category: data.context.category,
            profile_niche: data.context.profileNiche,
            profile_tone: data.context.profileTone,
            profile_audience_level: data.context.profileAudienceLevel,
            profile_language_level: data.context.profileLanguageLevel,
            profile_primary_goal: data.context.profilePrimaryGoal,
            profile_audience_description: data.context.profileAudienceDescription,
            context_answers: data.contextAnswers || {},
          }
        : undefined;

      const rawOutline = await api.courses.previewOutline({
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
        context: contextPayload,
      });

      const outline = transformOutline(rawOutline);
      setPreviewOutline(outline);
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
      // Build context for API if available
      const contextPayload = data.context
        ? {
            category: data.context.category,
            profile_niche: data.context.profileNiche,
            profile_tone: data.context.profileTone,
            profile_audience_level: data.context.profileAudienceLevel,
            profile_language_level: data.context.profileLanguageLevel,
            profile_primary_goal: data.context.profilePrimaryGoal,
            profile_audience_description: data.context.profileAudienceDescription,
            context_answers: data.contextAnswers || {},
            specific_tools: data.context.specificTools,
            practical_focus: data.context.practicalFocus,
            expected_outcome: data.context.expectedOutcome,
          }
        : undefined;

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
        context: contextPayload,
        voice_id: data.voiceId,
        style: data.style,
        typing_speed: data.typingSpeed,
        include_avatar: data.includeAvatar,
        avatar_id: data.avatarId || undefined,
        approved_outline: approvedOutline,
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
  }, [pollInterval, pollJobStatus, onError]);

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

  // Clear preview
  const clearPreview = useCallback(() => {
    setPreviewOutline(null);
  }, []);

  // Clear error
  const clearError = useCallback(() => {
    setError(null);
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
  };
}
