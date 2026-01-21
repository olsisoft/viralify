'use client';

import { useState } from 'react';
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
} from 'lucide-react';
import type { CourseJob, CourseStage, Lecture, LectureStatus } from '../lib/course-types';

interface GenerationProgressProps {
  job: CourseJob;
  onDownload?: () => void;
  onPractice?: () => void;
  onEditLecture?: (lecture: Lecture) => void;
  onRetryFailed?: () => void;
}

const STAGE_LABELS: Record<CourseStage, string> = {
  queued: 'Queued',
  planning: 'Planning curriculum...',
  generating_lectures: 'Generating lectures...',
  compiling: 'Compiling course...',
  completed: 'Completed!',
  partial_success: 'Partially completed',
  failed: 'Failed',
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
    case 'pending':
    default:
      return <Circle className="w-4 h-4 text-gray-500" />;
  }
}

interface LectureProgressItemProps {
  lecture: Lecture;
  index: number;
  onEdit?: (lecture: Lecture) => void;
}

function LectureProgressItem({ lecture, index, onEdit }: LectureProgressItemProps) {
  const isActive = lecture.status === 'generating' || lecture.status === 'retrying';
  const canEdit = lecture.status === 'completed' && lecture.hasComponents;
  const canRegenerate = lecture.status === 'failed' && lecture.canRegenerate;
  const stageLabel = lecture.currentStage
    ? LECTURE_STAGE_LABELS[lecture.currentStage] || lecture.currentStage
    : '';

  return (
    <div className={`flex items-center gap-3 py-2 px-3 rounded-lg ${
      isActive ? 'bg-purple-500/10 border border-purple-500/30' :
      lecture.status === 'failed' ? 'bg-red-500/10 border border-red-500/30' :
      'bg-gray-800/30'
    }`}>
      <LectureStatusIcon status={lecture.status} />

      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span className={`text-sm truncate ${
            isActive ? 'text-white font-medium' :
            lecture.status === 'completed' ? 'text-gray-300' :
            lecture.status === 'failed' ? 'text-red-300' : 'text-gray-500'
          }`}>
            {index + 1}. {lecture.title}
          </span>

          <div className="flex items-center gap-2">
            {lecture.isEdited && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-400">
                Modifi\u00e9
              </span>
            )}
            {lecture.retryCount > 0 && lecture.status !== 'completed' && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-400">
                Retry {lecture.retryCount}/3
              </span>
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
                \u00c9diter
              </button>
            )}
            {/* Regenerate button for failed lectures */}
            {canRegenerate && onEdit && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onEdit(lecture);
                }}
                className="text-xs px-2 py-1 rounded bg-green-600 text-white hover:bg-green-500"
              >
                R\u00e9g\u00e9n\u00e9rer
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
      </div>
    </div>
  );
}

export function GenerationProgress({ job, onDownload, onPractice, onEditLecture, onRetryFailed }: GenerationProgressProps) {
  const [showLectureDetails, setShowLectureDetails] = useState(true);
  const isComplete = job.status === 'completed';
  const isPartialSuccess = job.status === 'partial_success' || job.isPartialSuccess;
  const isFailed = job.status === 'failed';
  const isProcessing = job.status === 'processing' || job.status === 'queued';
  const canDownload = isComplete || (isPartialSuccess && job.canDownloadPartial);

  // Flatten all lectures for display
  const allLectures = job.outline?.sections.flatMap(s => s.lectures) || [];

  const progressPercent = Math.min(Math.max(job.progress, 0), 100);

  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          {isProcessing && <Loader2 className="w-5 h-5 animate-spin text-purple-400" />}
          {isComplete && <CheckCircle2 className="w-5 h-5 text-green-400" />}
          {isPartialSuccess && <CheckCircle2 className="w-5 h-5 text-yellow-400" />}
          {isFailed && <XCircle className="w-5 h-5 text-red-400" />}
          Progression de la génération
        </h3>
        {job.outline && (
          <span className="text-sm text-gray-400">
            {job.outline.title}
          </span>
        )}
      </div>

      {/* Progress bar */}
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">{STAGE_LABELS[job.currentStage]}</span>
          <span className="text-white font-medium">{progressPercent.toFixed(1)}%</span>
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
            <div className="flex items-center gap-2">
              <FileVideo className="w-5 h-5 text-purple-400" />
              <span className="text-gray-300">
                Lectures: <span className="text-white font-medium">{job.lecturesCompleted}</span> / {job.lecturesTotal}
              </span>
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

          {/* Detailed lecture list */}
          {showLectureDetails && allLectures.length > 0 && (
            <div className="max-h-60 overflow-y-auto space-y-1 border-t border-gray-700 pt-3">
              {allLectures.map((lecture, index) => (
                <LectureProgressItem key={lecture.id} lecture={lecture} index={index} onEdit={onEditLecture} />
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
          <div className="flex gap-3">
            {/* Download partial button */}
            {canDownload && onDownload && (
              <button
                onClick={onDownload}
                className="flex-1 flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-700 text-white font-medium py-3 px-4 rounded-lg transition-colors"
              >
                <Download className="w-5 h-5" />
                Télécharger (partiel)
              </button>
            )}

            {/* Retry all failed button */}
            {allLectures.filter(l => l.status === 'failed').length > 0 && onRetryFailed && (
              <button
                onClick={onRetryFailed}
                className="flex-1 flex items-center justify-center gap-2 bg-yellow-600 hover:bg-yellow-700 text-white font-medium py-3 px-4 rounded-lg transition-colors"
              >
                <RefreshCw className="w-5 h-5" />
                Régénérer les échecs ({allLectures.filter(l => l.status === 'failed').length})
              </button>
            )}
          </div>
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
          </div>

          {/* Video list */}
          {job.outputUrls.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-400">Generated Videos:</h4>
              <div className="max-h-40 overflow-y-auto space-y-1">
                {job.outputUrls.map((url, index) => (
                  <a
                    key={index}
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-sm text-purple-400 hover:text-purple-300 transition-colors"
                  >
                    <FileVideo className="w-4 h-4" />
                    Lecture {index + 1}
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Timestamps */}
      <div className="flex items-center gap-4 text-xs text-gray-500 pt-2 border-t border-gray-700">
        <div className="flex items-center gap-1">
          <Clock className="w-3 h-3" />
          Started: {new Date(job.createdAt).toLocaleTimeString()}
        </div>
        {job.completedAt && (
          <div>
            Completed: {new Date(job.completedAt).toLocaleTimeString()}
          </div>
        )}
      </div>
    </div>
  );
}
