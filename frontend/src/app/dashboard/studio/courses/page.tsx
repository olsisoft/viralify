'use client';

import { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { GraduationCap, Sparkles } from 'lucide-react';
import { CourseForm } from './components/CourseForm';
import { OutlineTree } from './components/OutlineTree';
import { GenerationProgress } from './components/GenerationProgress';
import { CourseHistory } from './components/CourseHistory';
import { LectureEditor } from './components/LectureEditor';
import { useCourseGeneration } from './hooks/useCourseGeneration';
import type {
  CourseFormState,
  CourseOutline,
  defaultCourseFormState,
  Section,
  Lecture,
} from './lib/course-types';

const initialFormState: CourseFormState = {
  profileId: '',
  topic: '',
  description: '',
  difficultyStart: 'beginner',
  difficultyEnd: 'intermediate',
  structure: {
    totalDurationMinutes: 60,
    numberOfSections: 5,
    lecturesPerSection: 3,
    randomStructure: false,
  },
  lessonElements: {
    conceptIntro: true,
    diagramSchema: true,
    codeTyping: true,
    codeExecution: false,
    voiceoverExplanation: true,
    curriculumSlide: true,
  },
  adaptiveElements: {
    commonElements: {
      concept_intro: true,
      voiceover: true,
      curriculum_slide: true,
      conclusion: true,
      quiz: true,
    },
    categoryElements: {},
    useAiSuggestions: true,
  },
  quizConfig: {
    enabled: true,
    frequency: 'per_section',
    questionsPerQuiz: 5,
    questionTypes: ['multiple_choice', 'true_false'],
    passingScore: 70,
    showExplanations: true,
    allowRetry: true,
  },
  context: null,
  contextAnswers: {},
  language: 'fr',
  voiceId: 'alloy',
  style: 'dark',
  typingSpeed: 'natural',
  titleStyle: 'engaging',
  includeAvatar: false,
  avatarId: '',
  documents: [],
  sourceIds: [],
  detectedCategory: null,
  customKeywords: [],
};

export default function CoursesPage() {
  const router = useRouter();
  const [formState, setFormState] = useState<CourseFormState>(initialFormState);
  const [historyRefresh, setHistoryRefresh] = useState(0);
  const [editingLecture, setEditingLecture] = useState<Lecture | null>(null);

  const [isCancelling, setIsCancelling] = useState(false);

  const {
    currentJob,
    isGenerating,
    previewOutline,
    isLoadingPreview,
    error,
    generatePreview,
    startGeneration,
    reorderOutline,
    clearPreview,
    clearError,
    refreshJob,
    // Job management methods
    getErrors,
    updateLessonContent,
    retryLesson,
    retryAllFailed,
    cancelJob,
    rebuildVideo,
    getLessons,
  } = useCourseGeneration({
    onComplete: () => {
      setHistoryRefresh(prev => prev + 1);
    },
  });

  const handlePreview = useCallback(async () => {
    clearError();
    try {
      await generatePreview(formState);
    } catch (err) {
      console.error('Preview failed:', err);
    }
  }, [formState, generatePreview, clearError]);

  const handleGenerate = useCallback(async () => {
    clearError();
    try {
      await startGeneration(formState, previewOutline || undefined);
    } catch (err) {
      console.error('Generation failed:', err);
    }
  }, [formState, previewOutline, startGeneration, clearError]);

  const handleReorder = useCallback(async (sections: Section[]) => {
    if (currentJob?.jobId && previewOutline) {
      // Update local preview
      const updatedOutline: CourseOutline = {
        ...previewOutline,
        sections,
      };
      // This would need to be implemented properly
      // For now, just update the form state
    }
  }, [currentJob, previewOutline]);

  const handleDownload = useCallback(() => {
    if (currentJob?.jobId) {
      // Use the API download endpoint which returns FileResponse
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || '';
      window.open(`${apiUrl}/api/v1/courses/${currentJob.jobId}/download`, '_blank');
    }
  }, [currentJob]);

  const handlePractice = useCallback(() => {
    if (currentJob?.jobId) {
      router.push(`/dashboard/studio/practice?courseId=${currentJob.jobId}`);
    }
  }, [currentJob, router]);

  const handleEditLecture = useCallback((lecture: Lecture) => {
    setEditingLecture(lecture);
  }, []);

  const handleCloseLectureEditor = useCallback(() => {
    setEditingLecture(null);
    // Refresh the job to get updated lecture status
    if (currentJob?.jobId) {
      refreshJob?.(currentJob.jobId);
    }
  }, [currentJob, refreshJob]);

  const handleLectureUpdated = useCallback((updatedLecture: Lecture) => {
    // Refresh the job to reflect the changes
    if (currentJob?.jobId) {
      refreshJob?.(currentJob.jobId);
    }
  }, [currentJob, refreshJob]);

  const handleRetryFailed = useCallback(async () => {
    if (currentJob?.jobId) {
      try {
        await retryAllFailed(currentJob.jobId);
      } catch (err) {
        console.error('Failed to retry lectures:', err);
      }
    }
  }, [currentJob, retryAllFailed]);

  const handleCancelJob = useCallback(async () => {
    if (currentJob?.jobId) {
      setIsCancelling(true);
      try {
        await cancelJob(currentJob.jobId, true); // keep_completed = true
      } catch (err) {
        console.error('Failed to cancel job:', err);
      } finally {
        setIsCancelling(false);
      }
    }
  }, [currentJob, cancelJob]);

  const handleRetryLesson = useCallback(async (sceneIndex: number) => {
    if (currentJob?.jobId) {
      try {
        await retryLesson(currentJob.jobId, sceneIndex, true); // rebuild_final = true
      } catch (err) {
        console.error('Failed to retry lesson:', err);
      }
    }
  }, [currentJob, retryLesson]);

  const handleUpdateLessonContent = useCallback(async (
    sceneIndex: number,
    content: { voiceoverText?: string; title?: string }
  ) => {
    if (currentJob?.jobId) {
      try {
        await updateLessonContent(currentJob.jobId, sceneIndex, content);
      } catch (err) {
        console.error('Failed to update lesson content:', err);
      }
    }
  }, [currentJob, updateLessonContent]);

  const handleRebuildVideo = useCallback(async () => {
    if (currentJob?.jobId) {
      try {
        await rebuildVideo(currentJob.jobId);
      } catch (err) {
        console.error('Failed to rebuild video:', err);
      }
    }
  }, [currentJob, rebuildVideo]);

  const handleGetErrors = useCallback(async () => {
    if (currentJob?.jobId) {
      const result = await getErrors(currentJob.jobId);
      return result?.errors || null;
    }
    return null;
  }, [currentJob, getErrors]);

  const handleGetLessons = useCallback(async () => {
    if (currentJob?.jobId) {
      const result = await getLessons(currentJob.jobId);
      return result?.lessons || null;
    }
    return null;
  }, [currentJob, getLessons]);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-8">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 bg-purple-600/20 rounded-lg">
            <GraduationCap className="w-8 h-8 text-purple-400" />
          </div>
          <div>
            <h1 className="text-3xl font-bold">Course Creator</h1>
            <p className="text-gray-400">
              Generate complete courses with multiple video lectures
            </p>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left column - Form */}
        <div className="space-y-6">
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-purple-400" />
              Course Configuration
            </h2>
            <CourseForm
              formState={formState}
              onFormChange={setFormState}
              onPreview={handlePreview}
              onGenerate={handleGenerate}
              isPreviewLoading={isLoadingPreview}
              isGenerating={isGenerating}
              hasPreview={!!previewOutline}
            />
          </div>
        </div>

        {/* Right column - Preview & Progress */}
        <div className="space-y-6">
          {/* Error display */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4">
              <p className="text-red-400">{error}</p>
              <button
                onClick={clearError}
                className="text-sm text-red-300 hover:text-red-200 mt-2"
              >
                Dismiss
              </button>
            </div>
          )}

          {/* Generation Progress */}
          {currentJob && (
            <GenerationProgress
              job={currentJob}
              onDownload={handleDownload}
              onPractice={handlePractice}
              onEditLecture={handleEditLecture}
              onRetryFailed={handleRetryFailed}
              onCancelJob={handleCancelJob}
              onRetryLesson={handleRetryLesson}
              onUpdateLessonContent={handleUpdateLessonContent}
              onRebuildVideo={handleRebuildVideo}
              onGetErrors={handleGetErrors}
              onGetLessons={handleGetLessons}
              isCancelling={isCancelling}
            />
          )}

          {/* Preview Outline */}
          {previewOutline && !currentJob && (
            <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Course Outline Preview</h3>
                <button
                  onClick={clearPreview}
                  className="text-sm text-gray-400 hover:text-white transition-colors"
                >
                  Clear
                </button>
              </div>
              <div className="mb-4">
                <h4 className="text-xl font-bold text-white">{previewOutline.title}</h4>
                <p className="text-gray-400 text-sm mt-1">{previewOutline.description}</p>
              </div>
              <OutlineTree
                outline={previewOutline}
                onReorder={handleReorder}
              />
              <p className="text-xs text-gray-500 mt-4 text-center">
                Drag and drop to reorder sections and lectures
              </p>
            </div>
          )}

          {/* Placeholder when no preview */}
          {!previewOutline && !currentJob && (
            <div className="bg-gray-800/50 border border-gray-700 border-dashed rounded-xl p-12 text-center">
              <GraduationCap className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-400 mb-2">
                Course Outline Preview
              </h3>
              <p className="text-gray-500 text-sm">
                Enter a topic and click "Preview Outline" to see the generated course structure
              </p>
            </div>
          )}

          {/* Course History */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
            <CourseHistory
              refreshTrigger={historyRefresh}
              onSelectJob={(job) => {
                // Could implement job viewing
                console.log('Selected job:', job);
              }}
            />
          </div>
        </div>
      </div>

      {/* Lecture Editor Modal */}
      {editingLecture && currentJob && (
        <LectureEditor
          jobId={currentJob.jobId}
          lecture={editingLecture}
          onClose={handleCloseLectureEditor}
          onLectureUpdated={handleLectureUpdated}
        />
      )}
    </div>
  );
}
