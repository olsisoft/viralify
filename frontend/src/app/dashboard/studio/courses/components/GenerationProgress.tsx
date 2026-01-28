'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  Loader2,
  CheckCircle2,
  XCircle,
  FileVideo,
  Clock,
  Download,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Circle,
  PlayCircle,
  GraduationCap,
  Edit3,
  ExternalLink,
  Timer,
  FileText,
  StopCircle,
  AlertTriangle,
  Save,
  X,
  Play,
} from 'lucide-react';
import { TraceabilityPanel } from './TraceabilityPanel';
import type { CourseJob, CourseStage, Lecture, LectureStatus, LessonError, SceneVideo } from '../lib/course-types';

/**
 * Calculate estimated remaining time based on elapsed time and progress
 */
function useEstimatedTime(startTime: string, progress: number, isProcessing: boolean) {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    if (!isProcessing) return;

    const start = new Date(startTime).getTime();

    const updateElapsed = () => {
      const now = Date.now();
      setElapsedSeconds(Math.floor((now - start) / 1000));
    };

    updateElapsed();
    const interval = setInterval(updateElapsed, 1000);

    return () => clearInterval(interval);
  }, [startTime, isProcessing]);

  // Calculate estimated remaining time
  const estimatedRemainingSeconds = progress > 5 && elapsedSeconds > 0
    ? Math.max(0, Math.round((elapsedSeconds / progress) * (100 - progress)))
    : null;

  return { elapsedSeconds, estimatedRemainingSeconds };
}

/**
 * Format seconds to human-readable string
 */
function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${seconds}s`;
  } else if (seconds < 3600) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
  }
}

// Error Queue Panel Component
interface ErrorQueuePanelProps {
  errors: LessonError[];
  isLoading: boolean;
  onEditError: (error: LessonError) => void;
  onRetryError: (sceneIndex: number) => void;
  onRefresh: () => void;
}

function ErrorQueuePanel({ errors, isLoading, onEditError, onRetryError, onRefresh }: ErrorQueuePanelProps) {
  if (isLoading) {
    return (
      <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
        <div className="flex items-center justify-center gap-2 text-gray-400">
          <Loader2 className="w-5 h-5 animate-spin" />
          Chargement des erreurs...
        </div>
      </div>
    );
  }

  if (errors.length === 0) {
    return (
      <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
        <p className="text-gray-400 text-center">Aucune erreur à afficher</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800/50 border border-red-500/30 rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-red-400 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4" />
          File d'erreurs ({errors.length})
        </h4>
        <button
          onClick={onRefresh}
          className="text-xs text-gray-400 hover:text-white flex items-center gap-1"
        >
          <RefreshCw className="w-3 h-3" />
          Actualiser
        </button>
      </div>

      <div className="space-y-2 max-h-60 overflow-y-auto">
        {errors.map((error) => (
          <div
            key={error.sceneIndex}
            className="bg-red-500/10 border border-red-500/20 rounded-lg p-3"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-white truncate">
                  Lecture {error.sceneIndex + 1}: {error.title}
                </p>
                <p className="text-xs text-red-400 mt-1">{error.error}</p>
                <p className="text-xs text-gray-500 mt-1">
                  Type: {error.errorType} | Tentatives: {error.retryCount}/3
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => onEditError(error)}
                  className="text-xs px-2 py-1 rounded bg-yellow-600 text-white hover:bg-yellow-500 flex items-center gap-1"
                >
                  <Edit3 className="w-3 h-3" />
                  Éditer
                </button>
                <button
                  onClick={() => onRetryError(error.sceneIndex)}
                  className="text-xs px-2 py-1 rounded bg-green-600 text-white hover:bg-green-500 flex items-center gap-1"
                >
                  <RefreshCw className="w-3 h-3" />
                  Relancer
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Edit Error Modal Component
interface EditErrorModalProps {
  error: LessonError;
  editedContent: { voiceoverText: string; title: string };
  onContentChange: (content: { voiceoverText: string; title: string }) => void;
  onSave: () => void;
  onCancel: () => void;
  isSaving: boolean;
}

function EditErrorModal({ error, editedContent, onContentChange, onSave, onCancel, isSaving }: EditErrorModalProps) {
  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-800 border border-gray-700 rounded-lg max-w-2xl w-full max-h-[80vh] overflow-hidden">
        <div className="p-4 border-b border-gray-700 flex items-center justify-between">
          <h3 className="text-lg font-medium text-white">
            Éditer la lecture {error.sceneIndex + 1}
          </h3>
          <button
            onClick={onCancel}
            className="text-gray-400 hover:text-white"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-4 space-y-4 overflow-y-auto max-h-[60vh]">
          {/* Error info */}
          <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3">
            <p className="text-sm text-red-400">{error.error}</p>
          </div>

          {/* Title field */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">
              Titre de la lecture
            </label>
            <input
              type="text"
              value={editedContent.title}
              onChange={(e) => onContentChange({ ...editedContent, title: e.target.value })}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-white focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
            />
          </div>

          {/* Voiceover text field */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">
              Texte de la voix-off
            </label>
            <textarea
              value={editedContent.voiceoverText}
              onChange={(e) => onContentChange({ ...editedContent, voiceoverText: e.target.value })}
              rows={8}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-white focus:border-purple-500 focus:ring-1 focus:ring-purple-500 resize-none"
              placeholder="Entrez le texte de la voix-off..."
            />
          </div>
        </div>

        <div className="p-4 border-t border-gray-700 flex items-center justify-end gap-3">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors"
          >
            Annuler
          </button>
          <button
            onClick={onSave}
            disabled={isSaving}
            className="flex items-center gap-2 px-4 py-2 text-sm bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50"
          >
            {isSaving ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Save className="w-4 h-4" />
            )}
            Sauvegarder
          </button>
        </div>
      </div>
    </div>
  );
}

// Progressive Download Panel Component
interface ProgressiveDownloadPanelProps {
  lessons: SceneVideo[];
  isLoading: boolean;
  onDownload: (videoUrl: string, title: string) => void;
  onRefresh: () => void;
}

function ProgressiveDownloadPanel({ lessons, isLoading, onDownload, onRefresh }: ProgressiveDownloadPanelProps) {
  const readyLessons = lessons.filter(l => l.status === 'ready' && l.videoUrl);

  if (isLoading && lessons.length === 0) {
    return (
      <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
        <div className="flex items-center justify-center gap-2 text-gray-400">
          <Loader2 className="w-5 h-5 animate-spin" />
          Chargement des lectures...
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800/50 border border-blue-500/30 rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-blue-400 flex items-center gap-2">
          <Download className="w-4 h-4" />
          Téléchargement progressif ({readyLessons.length}/{lessons.length})
        </h4>
        <button
          onClick={onRefresh}
          className="text-xs text-gray-400 hover:text-white flex items-center gap-1"
        >
          <RefreshCw className="w-3 h-3" />
          Actualiser
        </button>
      </div>

      <div className="space-y-2 max-h-60 overflow-y-auto">
        {lessons.map((lesson) => (
          <div
            key={lesson.sceneIndex}
            className={`flex items-center justify-between gap-2 py-2 px-3 rounded-lg ${
              lesson.status === 'ready' ? 'bg-green-500/10 border border-green-500/20' :
              lesson.status === 'failed' ? 'bg-red-500/10 border border-red-500/20' :
              'bg-gray-800/30 border border-gray-700'
            }`}
          >
            <div className="flex items-center gap-2 flex-1 min-w-0">
              {lesson.status === 'ready' ? (
                <CheckCircle2 className="w-4 h-4 text-green-400 flex-shrink-0" />
              ) : lesson.status === 'failed' ? (
                <XCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
              ) : (
                <Loader2 className="w-4 h-4 text-purple-400 animate-spin flex-shrink-0" />
              )}
              <span className="text-sm text-gray-300 truncate">
                {lesson.sceneIndex + 1}. {lesson.title}
              </span>
              {lesson.duration > 0 && (
                <span className="text-xs text-gray-500">
                  ({Math.floor(lesson.duration / 60)}:{(lesson.duration % 60).toString().padStart(2, '0')})
                </span>
              )}
            </div>
            {lesson.status === 'ready' && lesson.videoUrl && (
              <button
                onClick={() => onDownload(lesson.videoUrl, lesson.title)}
                className="text-xs px-2 py-1 rounded bg-blue-600 text-white hover:bg-blue-500 flex items-center gap-1"
              >
                <Download className="w-3 h-3" />
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

interface GenerationProgressProps {
  job: CourseJob;
  onDownload?: () => void;
  onPractice?: () => void;
  onEditLecture?: (lecture: Lecture) => void;
  onRetryFailed?: () => void;
  // New job management callbacks
  onCancelJob?: () => Promise<void>;
  onRetryLesson?: (sceneIndex: number) => Promise<void>;
  onUpdateLessonContent?: (sceneIndex: number, content: { voiceoverText?: string; title?: string }) => Promise<void>;
  onRebuildVideo?: () => Promise<void>;
  onGetErrors?: () => Promise<LessonError[] | null>;
  onGetLessons?: () => Promise<SceneVideo[] | null>;
  isCancelling?: boolean;
}

const STAGE_LABELS: Record<CourseStage, string> = {
  queued: 'Queued',
  planning: 'Planning curriculum...',
  generating_lectures: 'Generating lectures...',
  compiling: 'Compiling course...',
  completed: 'Completed!',
  partial_success: 'Partially completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
  cancelling: 'Cancelling...',
};

const LECTURE_STAGE_LABELS: Record<string, string> = {
  starting: 'Starting...',
  script: 'Generating script',
  slides: 'Creating slides',
  voiceover: 'Generating voiceover',
  creating_animations: 'Creating animations',
  composing: 'Composing video',
  completed: 'Completed',
  failed: 'Failed',
};

function LectureStatusIcon({ status }: { status: LectureStatus }) {
  switch (status) {
    case 'completed':
      return <CheckCircle2 className="w-4 h-4 text-green-400" />;
    case 'failed':
      return <XCircle className="w-4 h-4 text-red-400" />;
    case 'generating':
      return <Loader2 className="w-4 h-4 text-purple-400 animate-spin" />;
    case 'retrying':
      return <RefreshCw className="w-4 h-4 text-yellow-400 animate-spin" />;
    case 'cancelled':
      return <StopCircle className="w-4 h-4 text-orange-400" />;
    case 'skipped':
      return <Circle className="w-4 h-4 text-gray-400" />;
    case 'pending':
    default:
      return <Circle className="w-4 h-4 text-gray-500" />;
  }
}

interface LectureProgressItemProps {
  lecture: Lecture;
  index: number;
  onEdit?: (lecture: Lecture) => void;
  onRetry?: (sceneIndex: number) => void;
  onDownloadLesson?: (videoUrl: string, title: string) => void;
  isRetrying?: boolean;
}

function LectureProgressItem({ lecture, index, onEdit, onRetry, onDownloadLesson, isRetrying }: LectureProgressItemProps) {
  const isActive = lecture.status === 'generating' || lecture.status === 'retrying';
  const canEdit = lecture.status === 'completed' && lecture.hasComponents;
  const canRegenerate = (lecture.status === 'failed' || lecture.status === 'cancelled') && lecture.canRegenerate;
  const canDownload = lecture.status === 'completed' && lecture.videoUrl;
  const stageLabel = lecture.currentStage
    ? LECTURE_STAGE_LABELS[lecture.currentStage] || lecture.currentStage
    : '';

  return (
    <div className={`flex items-center gap-3 py-2 px-3 rounded-lg ${
      isActive ? 'bg-purple-500/10 border border-purple-500/30' :
      lecture.status === 'failed' ? 'bg-red-500/10 border border-red-500/30' :
      lecture.status === 'cancelled' ? 'bg-orange-500/10 border border-orange-500/30' :
      'bg-gray-800/30'
    }`}>
      <LectureStatusIcon status={lecture.status} />

      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span className={`text-sm truncate ${
            isActive ? 'text-white font-medium' :
            lecture.status === 'completed' ? 'text-gray-300' :
            lecture.status === 'failed' ? 'text-red-300' :
            lecture.status === 'cancelled' ? 'text-orange-300' : 'text-gray-500'
          }`}>
            {index + 1}. {lecture.title}
          </span>

          <div className="flex items-center gap-2">
            {lecture.isEdited && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-400">
                Modifié
              </span>
            )}
            {lecture.retryCount > 0 && lecture.status !== 'completed' && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-400">
                Retry {lecture.retryCount}/3
              </span>
            )}
            {/* Download button for completed lessons */}
            {canDownload && onDownloadLesson && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDownloadLesson(lecture.videoUrl!, lecture.title);
                }}
                className="text-xs px-2 py-1 rounded bg-blue-600 text-white hover:bg-blue-500 flex items-center gap-1"
              >
                <Download className="w-3 h-3" />
              </button>
            )}
            {/* Edit button for completed lectures with components */}
            {canEdit && onEdit && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onEdit(lecture);
                }}
                className="text-xs px-2 py-1 rounded bg-purple-600 text-white hover:bg-purple-500"
              >
                Éditer
              </button>
            )}
            {/* Retry button for failed/cancelled lectures */}
            {canRegenerate && onRetry && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onRetry(index);
                }}
                disabled={isRetrying}
                className="text-xs px-2 py-1 rounded bg-green-600 text-white hover:bg-green-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
              >
                {isRetrying ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : (
                  <RefreshCw className="w-3 h-3" />
                )}
                Relancer
              </button>
            )}
          </div>
        </div>

        {isActive && (
          <div className="mt-1 space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-purple-300">{stageLabel}</span>
              <span className="text-gray-400">{lecture.progressPercent.toFixed(0)}%</span>
            </div>
            <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-purple-500 to-blue-500 transition-all duration-300"
                style={{ width: `${lecture.progressPercent}%` }}
              />
            </div>
          </div>
        )}

        {lecture.status === 'failed' && lecture.error && (
          <p className="text-xs text-red-400 mt-1 truncate">{lecture.error}</p>
        )}
        {lecture.status === 'cancelled' && (
          <p className="text-xs text-orange-400 mt-1">Annulé par l'utilisateur</p>
        )}
      </div>
    </div>
  );
}

export function GenerationProgress({
  job,
  onDownload,
  onPractice,
  onEditLecture,
  onRetryFailed,
  onCancelJob,
  onRetryLesson,
  onUpdateLessonContent,
  onRebuildVideo,
  onGetErrors,
  onGetLessons,
  isCancelling = false,
}: GenerationProgressProps) {
  const [showLectureDetails, setShowLectureDetails] = useState(true);
  const [showTraceability, setShowTraceability] = useState(false);
  const [showErrorQueue, setShowErrorQueue] = useState(false);
  const [showProgressiveDownload, setShowProgressiveDownload] = useState(false);
  const [errorQueue, setErrorQueue] = useState<LessonError[]>([]);
  const [progressiveLessons, setProgressiveLessons] = useState<SceneVideo[]>([]);
  const [isLoadingErrors, setIsLoadingErrors] = useState(false);
  const [isLoadingLessons, setIsLoadingLessons] = useState(false);
  const [retryingLessonIndex, setRetryingLessonIndex] = useState<number | null>(null);
  const [editingError, setEditingError] = useState<LessonError | null>(null);
  const [editedContent, setEditedContent] = useState<{ voiceoverText: string; title: string }>({ voiceoverText: '', title: '' });
  const [isSavingEdit, setIsSavingEdit] = useState(false);

  const isComplete = job.status === 'completed';
  const isPartialSuccess = job.status === 'partial_success' || job.isPartialSuccess;
  const isFailed = job.status === 'failed';
  const isCancelled = job.status === 'cancelled';
  const isProcessing = job.status === 'processing' || job.status === 'queued';
  const canDownload = isComplete || (isPartialSuccess && job.canDownloadPartial) || (isCancelled && job.lecturesCompleted > 0);
  const canCancel = isProcessing && !isCancelling && !job.cancelRequested;

  // Load error queue
  const loadErrors = useCallback(async () => {
    if (!onGetErrors) return;
    setIsLoadingErrors(true);
    try {
      const errors = await onGetErrors();
      if (errors) {
        setErrorQueue(errors);
      }
    } finally {
      setIsLoadingErrors(false);
    }
  }, [onGetErrors]);

  // Load progressive download lessons
  const loadLessons = useCallback(async () => {
    if (!onGetLessons) return;
    setIsLoadingLessons(true);
    try {
      const lessons = await onGetLessons();
      if (lessons) {
        setProgressiveLessons(lessons);
      }
    } finally {
      setIsLoadingLessons(false);
    }
  }, [onGetLessons]);

  // Auto-load lessons periodically during processing
  useEffect(() => {
    if (isProcessing && onGetLessons) {
      loadLessons();
      const interval = setInterval(loadLessons, 5000);
      return () => clearInterval(interval);
    }
  }, [isProcessing, loadLessons, onGetLessons]);

  // Handle retry single lesson
  const handleRetryLesson = async (sceneIndex: number) => {
    if (!onRetryLesson) return;
    setRetryingLessonIndex(sceneIndex);
    try {
      await onRetryLesson(sceneIndex);
    } finally {
      setRetryingLessonIndex(null);
    }
  };

  // Handle save edited content
  const handleSaveEdit = async () => {
    if (!editingError || !onUpdateLessonContent) return;
    setIsSavingEdit(true);
    try {
      await onUpdateLessonContent(editingError.sceneIndex, editedContent);
      // Refresh error queue
      await loadErrors();
      setEditingError(null);
    } finally {
      setIsSavingEdit(false);
    }
  };

  // Handle download individual lesson
  const handleDownloadLesson = (videoUrl: string, title: string) => {
    const link = document.createElement('a');
    link.href = videoUrl;
    link.download = `${title.replace(/[^a-z0-9]/gi, '_')}.mp4`;
    link.target = '_blank';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Start editing an error
  const startEditingError = (error: LessonError) => {
    setEditingError(error);
    setEditedContent({
      voiceoverText: error.voiceoverText || '',
      title: error.title || '',
    });
  };

  // Flatten all lectures for display
  const allLectures = job.outline?.sections.flatMap(s => s.lectures) || [];

  const progressPercent = Math.min(Math.max(job.progress, 0), 100);

  // Time estimation
  const { elapsedSeconds, estimatedRemainingSeconds } = useEstimatedTime(
    job.createdAt,
    progressPercent,
    isProcessing
  );

  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          {isProcessing && !isCancelling && <Loader2 className="w-5 h-5 animate-spin text-purple-400" />}
          {isCancelling && <Loader2 className="w-5 h-5 animate-spin text-orange-400" />}
          {isComplete && <CheckCircle2 className="w-5 h-5 text-green-400" />}
          {isPartialSuccess && <CheckCircle2 className="w-5 h-5 text-yellow-400" />}
          {isCancelled && <StopCircle className="w-5 h-5 text-orange-400" />}
          {isFailed && <XCircle className="w-5 h-5 text-red-400" />}
          Progression de la génération
        </h3>
        <div className="flex items-center gap-2">
          {job.outline && (
            <span className="text-sm text-gray-400">
              {job.outline.title}
            </span>
          )}
          {/* Cancel button */}
          {canCancel && onCancelJob && (
            <button
              onClick={onCancelJob}
              disabled={isCancelling}
              className="flex items-center gap-1 px-3 py-1.5 text-sm bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors disabled:opacity-50"
            >
              <StopCircle className="w-4 h-4" />
              Annuler
            </button>
          )}
          {isCancelling && (
            <span className="flex items-center gap-1 px-3 py-1.5 text-sm bg-orange-600/20 text-orange-400 rounded-lg">
              <Loader2 className="w-4 h-4 animate-spin" />
              Annulation...
            </span>
          )}
        </div>
      </div>

      {/* Progress bar */}
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">{STAGE_LABELS[job.currentStage]}</span>
          <div className="flex items-center gap-3">
            {/* Time estimation */}
            {isProcessing && (
              <div className="flex items-center gap-2 text-gray-400">
                <Timer className="w-4 h-4" />
                <span>
                  {elapsedSeconds > 0 && (
                    <span className="text-gray-500">
                      {formatDuration(elapsedSeconds)} écoulé
                    </span>
                  )}
                  {estimatedRemainingSeconds !== null && estimatedRemainingSeconds > 0 && (
                    <span className="text-purple-400 ml-2">
                      ~{formatDuration(estimatedRemainingSeconds)} restant
                    </span>
                  )}
                </span>
              </div>
            )}
            <span className="text-white font-medium">{progressPercent.toFixed(1)}%</span>
          </div>
        </div>
        <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-500 ${
              isFailed
                ? 'bg-red-500'
                : isComplete
                ? 'bg-green-500'
                : isPartialSuccess
                ? 'bg-yellow-500'
                : 'bg-gradient-to-r from-purple-500 to-blue-500'
            }`}
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>

      {/* Lecture progress summary */}
      {job.lecturesTotal > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between py-2">
            <div className="flex items-center gap-2 flex-wrap">
              <FileVideo className="w-5 h-5 text-purple-400" />
              <span className="text-gray-300">
                Lectures: <span className="text-white font-medium">{job.lecturesCompleted}</span> / {job.lecturesTotal} terminées
              </span>
              {job.lecturesInProgress > 0 && (
                <span className="text-xs px-2 py-0.5 rounded bg-purple-500/20 text-purple-400 flex items-center gap-1">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  {job.lecturesInProgress} en cours
                </span>
              )}
              {allLectures.filter(l => l.status === 'failed').length > 0 && (
                <span className="text-xs px-2 py-0.5 rounded bg-red-500/20 text-red-400">
                  {allLectures.filter(l => l.status === 'failed').length} failed
                </span>
              )}
              {allLectures.filter(l => l.status === 'retrying').length > 0 && (
                <span className="text-xs px-2 py-0.5 rounded bg-yellow-500/20 text-yellow-400">
                  {allLectures.filter(l => l.status === 'retrying').length} retrying
                </span>
              )}
            </div>

            <button
              onClick={() => setShowLectureDetails(!showLectureDetails)}
              className="flex items-center gap-1 text-sm text-gray-400 hover:text-white transition-colors"
            >
              {showLectureDetails ? 'Hide details' : 'Show details'}
              {showLectureDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
          </div>

          {/* Show currently generating lectures */}
          {job.currentLectures && job.currentLectures.length > 0 && (
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
              <p className="text-sm text-purple-300 mb-2">En cours de génération:</p>
              <div className="flex flex-wrap gap-2">
                {job.currentLectures.map((title, idx) => (
                  <span key={idx} className="text-xs px-2 py-1 rounded bg-purple-600/30 text-purple-200 flex items-center gap-1">
                    <Loader2 className="w-3 h-3 animate-spin" />
                    {title.length > 40 ? title.substring(0, 40) + '...' : title}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Detailed lecture list */}
          {showLectureDetails && allLectures.length > 0 && (
            <div className="max-h-60 overflow-y-auto space-y-1 border-t border-gray-700 pt-3">
              {allLectures.map((lecture, index) => (
                <LectureProgressItem
                  key={lecture.id}
                  lecture={lecture}
                  index={index}
                  onEdit={onEditLecture}
                  onRetry={onRetryLesson ? handleRetryLesson : undefined}
                  onDownloadLesson={handleDownloadLesson}
                  isRetrying={retryingLessonIndex === index}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Status message */}
      {job.message && (
        <p className="text-sm text-gray-400">{job.message}</p>
      )}

      {/* Error display */}
      {isFailed && job.error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
          <p className="text-red-400 text-sm">{job.error}</p>
        </div>
      )}

      {/* Cancelled state */}
      {isCancelled && (
        <div className="space-y-4">
          <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg p-4">
            <p className="text-orange-400">
              Génération annulée. {job.lecturesCompleted} lecture{job.lecturesCompleted !== 1 ? 's' : ''} sur {job.lecturesTotal} générée{job.lecturesCompleted !== 1 ? 's' : ''} avant l'annulation.
              {job.lecturesCancelled > 0 && (
                <span className="block mt-1 text-sm">
                  {job.lecturesCancelled} lecture{job.lecturesCancelled !== 1 ? 's' : ''} annulée{job.lecturesCancelled !== 1 ? 's' : ''}.
                </span>
              )}
            </p>
          </div>

          {/* Action buttons for cancelled */}
          <div className="flex gap-3 flex-wrap">
            {/* Download completed button */}
            {canDownload && onDownload && (
              <button
                onClick={onDownload}
                className="flex-1 min-w-[150px] flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-700 text-white font-medium py-3 px-4 rounded-lg transition-colors"
              >
                <Download className="w-5 h-5" />
                Télécharger ({job.lecturesCompleted} lectures)
              </button>
            )}

            {/* Rebuild video button */}
            {job.lecturesCompleted > 0 && onRebuildVideo && (
              <button
                onClick={onRebuildVideo}
                className="flex-1 min-w-[150px] flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-lg transition-colors"
              >
                <Play className="w-5 h-5" />
                Reconstruire la vidéo
              </button>
            )}

            {/* Retry cancelled lessons button */}
            {job.lecturesCancelled > 0 && onRetryFailed && (
              <button
                onClick={onRetryFailed}
                className="flex-1 min-w-[150px] flex items-center justify-center gap-2 bg-green-600 hover:bg-green-700 text-white font-medium py-3 px-4 rounded-lg transition-colors"
              >
                <RefreshCw className="w-5 h-5" />
                Reprendre ({job.lecturesCancelled})
              </button>
            )}
          </div>
        </div>
      )}

      {/* Partial success state */}
      {isPartialSuccess && (
        <div className="space-y-4">
          <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
            <p className="text-yellow-400">
              Cours partiellement généré. {job.lecturesCompleted} lecture{job.lecturesCompleted !== 1 ? 's' : ''} sur {job.lecturesTotal} générée{job.lecturesCompleted !== 1 ? 's' : ''} avec succès.
              {allLectures.filter(l => l.status === 'failed').length > 0 && (
                <span className="block mt-1 text-sm">
                  {allLectures.filter(l => l.status === 'failed').length} lecture{allLectures.filter(l => l.status === 'failed').length !== 1 ? 's' : ''} en échec - vous pouvez les éditer et régénérer.
                </span>
              )}
            </p>
          </div>

          {/* Action buttons for partial success */}
          <div className="flex gap-3 flex-wrap">
            {/* Download partial button */}
            {canDownload && onDownload && (
              <button
                onClick={onDownload}
                className="flex-1 min-w-[150px] flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-700 text-white font-medium py-3 px-4 rounded-lg transition-colors"
              >
                <Download className="w-5 h-5" />
                Télécharger (partiel)
              </button>
            )}

            {/* Error queue button */}
            {onGetErrors && allLectures.filter(l => l.status === 'failed').length > 0 && (
              <button
                onClick={() => {
                  setShowErrorQueue(!showErrorQueue);
                  if (!showErrorQueue) loadErrors();
                }}
                className={`flex-1 min-w-[150px] flex items-center justify-center gap-2 font-medium py-3 px-4 rounded-lg transition-colors ${
                  showErrorQueue
                    ? 'bg-red-600 hover:bg-red-700 text-white'
                    : 'bg-red-600/20 hover:bg-red-600/30 text-red-400'
                }`}
              >
                <AlertTriangle className="w-5 h-5" />
                Erreurs ({allLectures.filter(l => l.status === 'failed').length})
              </button>
            )}

            {/* Retry all failed button */}
            {allLectures.filter(l => l.status === 'failed').length > 0 && onRetryFailed && (
              <button
                onClick={onRetryFailed}
                className="flex-1 min-w-[150px] flex items-center justify-center gap-2 bg-yellow-600 hover:bg-yellow-700 text-white font-medium py-3 px-4 rounded-lg transition-colors"
              >
                <RefreshCw className="w-5 h-5" />
                Régénérer les échecs ({allLectures.filter(l => l.status === 'failed').length})
              </button>
            )}

            {/* Traceability button */}
            <button
              onClick={() => setShowTraceability(!showTraceability)}
              className={`flex-1 min-w-[150px] flex items-center justify-center gap-2 font-medium py-3 px-4 rounded-lg transition-colors ${
                showTraceability
                  ? 'bg-blue-600 hover:bg-blue-700 text-white'
                  : 'bg-gray-700 hover:bg-gray-600 text-gray-200'
              }`}
            >
              <FileText className="w-5 h-5" />
              Traçabilité
            </button>
          </div>

          {/* Error Queue Panel */}
          {showErrorQueue && (
            <ErrorQueuePanel
              errors={errorQueue}
              isLoading={isLoadingErrors}
              onEditError={startEditingError}
              onRetryError={handleRetryLesson}
              onRefresh={loadErrors}
            />
          )}

          {/* Traceability Panel */}
          {showTraceability && (
            <TraceabilityPanel
              jobId={job.jobId}
              onClose={() => setShowTraceability(false)}
            />
          )}
        </div>
      )}

      {/* Completed state */}
      {isComplete && (
        <div className="space-y-4">
          <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
            <p className="text-green-400">
              Course generated successfully! {job.outputUrls.length} video{job.outputUrls.length !== 1 ? 's' : ''} created.
            </p>
          </div>

          {/* Action buttons */}
          <div className="flex gap-3">
            {/* Download button */}
            {job.status === 'completed' && onDownload && (
              <button
                onClick={onDownload}
                className="flex-1 flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-700 text-white font-medium py-3 px-4 rounded-lg transition-colors"
              >
                <Download className="w-5 h-5" />
                Download (ZIP)
              </button>
            )}

            {/* Practice mode button */}
            {job.status === 'completed' && onPractice && (
              <button
                onClick={onPractice}
                className="flex-1 flex items-center justify-center gap-2 bg-green-600 hover:bg-green-700 text-white font-medium py-3 px-4 rounded-lg transition-colors"
              >
                <GraduationCap className="w-5 h-5" />
                Mode Pratique
              </button>
            )}

            {/* Traceability button */}
            <button
              onClick={() => setShowTraceability(!showTraceability)}
              className={`flex-1 flex items-center justify-center gap-2 font-medium py-3 px-4 rounded-lg transition-colors ${
                showTraceability
                  ? 'bg-blue-600 hover:bg-blue-700 text-white'
                  : 'bg-gray-700 hover:bg-gray-600 text-gray-200'
              }`}
            >
              <FileText className="w-5 h-5" />
              Traçabilité
            </button>
          </div>

          {/* Traceability Panel */}
          {showTraceability && (
            <TraceabilityPanel
              jobId={job.jobId}
              onClose={() => setShowTraceability(false)}
            />
          )}

          {/* Video list */}
          {job.outputUrls.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-400">Generated Videos:</h4>
              <div className="max-h-40 overflow-y-auto space-y-2">
                {job.outputUrls.map((url, index) => (
                  <div key={index} className="flex items-center justify-between gap-2 py-1.5 px-2 rounded bg-gray-800/50">
                    <a
                      href={url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-sm text-purple-400 hover:text-purple-300 transition-colors flex-1"
                    >
                      <FileVideo className="w-4 h-4" />
                      Lecture {index + 1}
                      <ExternalLink className="w-3 h-3 opacity-50" />
                    </a>
                    <a
                      href={`/dashboard/studio/editor?videoUrl=${encodeURIComponent(url)}`}
                      className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-purple-600 text-white hover:bg-purple-500 transition-colors"
                    >
                      <Edit3 className="w-3 h-3" />
                      Éditer
                    </a>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Progressive Download Section (during processing) */}
      {isProcessing && onGetLessons && progressiveLessons.length > 0 && (
        <div className="space-y-3">
          <button
            onClick={() => setShowProgressiveDownload(!showProgressiveDownload)}
            className="flex items-center gap-2 text-sm text-blue-400 hover:text-blue-300"
          >
            <Download className="w-4 h-4" />
            Téléchargement progressif ({progressiveLessons.filter(l => l.status === 'ready').length} prêtes)
            {showProgressiveDownload ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {showProgressiveDownload && (
            <ProgressiveDownloadPanel
              lessons={progressiveLessons}
              isLoading={isLoadingLessons}
              onDownload={handleDownloadLesson}
              onRefresh={loadLessons}
            />
          )}
        </div>
      )}

      {/* Timestamps */}
      <div className="flex items-center gap-4 text-xs text-gray-500 pt-2 border-t border-gray-700">
        <div className="flex items-center gap-1">
          <Clock className="w-3 h-3" />
          Démarré: {new Date(job.createdAt).toLocaleTimeString()}
        </div>
        {job.completedAt && (
          <div>
            Terminé: {new Date(job.completedAt).toLocaleTimeString()}
          </div>
        )}
        {job.cancelledAt && (
          <div className="text-orange-400">
            Annulé: {new Date(job.cancelledAt).toLocaleTimeString()}
          </div>
        )}
      </div>

      {/* Edit Error Modal */}
      {editingError && (
        <EditErrorModal
          error={editingError}
          editedContent={editedContent}
          onContentChange={setEditedContent}
          onSave={handleSaveEdit}
          onCancel={() => setEditingError(null)}
          isSaving={isSavingEdit}
        />
      )}
    </div>
  );
}
