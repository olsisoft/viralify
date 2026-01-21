'use client';

import React, { useEffect, useState, useCallback } from 'react';
import { useLectureEditor } from '../../hooks/useLectureEditor';
import { SlideTimeline } from './SlideTimeline';
import { SlidePreview } from './SlidePreview';
import { SlideProperties } from './SlideProperties';
import type { Lecture } from '../../lib/course-types';
import type { SlideComponent, UpdateSlideRequest } from '../../lib/lecture-editor-types';
import { formatTotalDuration } from '../../lib/lecture-editor-types';

interface LectureEditorProps {
  jobId: string;
  lecture: Lecture;
  onClose: () => void;
  onLectureUpdated?: (lecture: Lecture) => void;
}

export function LectureEditor({ jobId, lecture, onClose, onLectureUpdated }: LectureEditorProps) {
  const [showSuccessMessage, setShowSuccessMessage] = useState<string | null>(null);

  const {
    components,
    selectedSlide,
    isLoading,
    isSaving,
    isRegenerating,
    error,
    loadComponents,
    updateSlide,
    regenerateSlide,
    regenerateVoiceover,
    uploadCustomAudio,
    regenerateLecture,
    recomposeVideo,
    selectSlide,
  } = useLectureEditor({
    onSuccess: (message) => {
      setShowSuccessMessage(message);
      setTimeout(() => setShowSuccessMessage(null), 3000);
    },
    onError: (err) => {
      console.error('Lecture editor error:', err);
    },
  });

  // Load components on mount
  useEffect(() => {
    loadComponents(jobId, lecture.id);
  }, [jobId, lecture.id, loadComponents]);

  // Handle slide update
  const handleSlideUpdate = useCallback(async (updates: UpdateSlideRequest) => {
    if (!selectedSlide) return;
    await updateSlide(jobId, lecture.id, selectedSlide.id, updates);
  }, [jobId, lecture.id, selectedSlide, updateSlide]);

  // Handle slide regeneration
  const handleRegenerateSlide = useCallback(async () => {
    if (!selectedSlide) return;
    await regenerateSlide(jobId, lecture.id, selectedSlide.id, {
      regenerateImage: true,
      regenerateAnimation: selectedSlide.type === 'code' || selectedSlide.type === 'code_demo',
      useEditedContent: true,
    });
  }, [jobId, lecture.id, selectedSlide, regenerateSlide]);

  // Handle voiceover regeneration
  const handleRegenerateVoiceover = useCallback(async () => {
    await regenerateVoiceover(jobId, lecture.id, {});
  }, [jobId, lecture.id, regenerateVoiceover]);

  // Handle custom audio upload
  const handleUploadAudio = useCallback(async (file: File) => {
    await uploadCustomAudio(jobId, lecture.id, file);
  }, [jobId, lecture.id, uploadCustomAudio]);

  // Handle full lecture regeneration
  const handleRegenerateLecture = useCallback(async () => {
    const result = await regenerateLecture(jobId, lecture.id, {
      useEditedComponents: true,
      regenerateVoiceover: true,
    });
    if (result?.success && onLectureUpdated) {
      onLectureUpdated({
        ...lecture,
        status: 'completed',
        videoUrl: result.result?.video_url,
        isEdited: true,
      });
    }
  }, [jobId, lecture.id, regenerateLecture, onLectureUpdated, lecture]);

  // Handle video recomposition
  const handleRecomposeVideo = useCallback(async () => {
    const result = await recomposeVideo(jobId, lecture.id, {
      quality: 'high',
      includeTransitions: true,
    });
    if (result?.success && onLectureUpdated) {
      onLectureUpdated({
        ...lecture,
        videoUrl: result.result?.video_url,
        isEdited: true,
      });
    }
  }, [jobId, lecture.id, recomposeVideo, onLectureUpdated, lecture]);

  if (isLoading) {
    return (
      <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center">
        <div className="bg-gray-900 rounded-lg p-8 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500 mx-auto mb-4" />
          <p className="text-white">Chargement des composants...</p>
        </div>
      </div>
    );
  }

  if (error && !components) {
    return (
      <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center">
        <div className="bg-gray-900 rounded-lg p-8 text-center max-w-md">
          <div className="text-red-500 text-4xl mb-4">!</div>
          <h3 className="text-white text-lg font-semibold mb-2">Erreur</h3>
          <p className="text-gray-400 mb-4">{error}</p>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600"
          >
            Fermer
          </button>
        </div>
      </div>
    );
  }

  if (!components) return null;

  return (
    <div className="fixed inset-0 bg-black/90 z-50 flex flex-col">
      {/* Header */}
      <div className="bg-gray-900 border-b border-gray-800 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
          <div>
            <h2 className="text-white font-semibold">{lecture.title}</h2>
            <p className="text-gray-400 text-sm">
              {components.slides.length} slides - {formatTotalDuration(components.totalDuration)}
              {components.isEdited && <span className="ml-2 text-yellow-500">(modifi\u00e9)</span>}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Success message */}
          {showSuccessMessage && (
            <span className="text-green-400 text-sm">{showSuccessMessage}</span>
          )}

          {/* Regenerate voiceover */}
          <button
            onClick={handleRegenerateVoiceover}
            disabled={isRegenerating}
            className="px-3 py-1.5 text-sm bg-gray-700 text-white rounded hover:bg-gray-600 disabled:opacity-50"
          >
            {isRegenerating ? 'R\u00e9g\u00e9n\u00e9ration...' : 'R\u00e9g\u00e9n\u00e9rer voiceover'}
          </button>

          {/* Recompose video */}
          <button
            onClick={handleRecomposeVideo}
            disabled={isRegenerating || !components.isEdited}
            className="px-3 py-1.5 text-sm bg-purple-600 text-white rounded hover:bg-purple-500 disabled:opacity-50"
          >
            Recomposer vid\u00e9o
          </button>

          {/* Regenerate full lecture */}
          <button
            onClick={handleRegenerateLecture}
            disabled={isRegenerating}
            className="px-3 py-1.5 text-sm bg-green-600 text-white rounded hover:bg-green-500 disabled:opacity-50"
          >
            R\u00e9g\u00e9n\u00e9rer la le\u00e7on
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Timeline */}
        <div className="w-64 bg-gray-900 border-r border-gray-800 overflow-y-auto">
          <SlideTimeline
            slides={components.slides}
            selectedSlide={selectedSlide}
            onSelectSlide={selectSlide}
          />
        </div>

        {/* Center: Preview */}
        <div className="flex-1 bg-gray-950 flex items-center justify-center p-4">
          <SlidePreview
            slide={selectedSlide}
            voiceover={components.voiceover}
          />
        </div>

        {/* Right: Properties */}
        <div className="w-80 bg-gray-900 border-l border-gray-800 overflow-y-auto">
          <SlideProperties
            slide={selectedSlide}
            voiceover={components.voiceover}
            isSaving={isSaving}
            isRegenerating={isRegenerating}
            onUpdate={handleSlideUpdate}
            onRegenerate={handleRegenerateSlide}
            onUploadAudio={handleUploadAudio}
          />
        </div>
      </div>
    </div>
  );
}

export default LectureEditor;
