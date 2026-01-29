'use client';

import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useLectureEditor } from '../../hooks/useLectureEditor';
import { SlideTimeline } from './SlideTimeline';
import { SlidePreview } from './SlidePreview';
import { SlideProperties } from './SlideProperties';
import { EditorToolbar } from './EditorToolbar';
import type { Lecture } from '../../lib/course-types';
import type { SlideComponent, UpdateSlideRequest, MediaType } from '../../lib/lecture-editor-types';
import { formatTotalDuration, KEYBOARD_SHORTCUTS } from '../../lib/lecture-editor-types';

interface LectureEditorProps {
  jobId: string;
  lecture: Lecture;
  onClose: () => void;
  onLectureUpdated?: (lecture: Lecture) => void;
}

export function LectureEditor({ jobId, lecture, onClose, onLectureUpdated }: LectureEditorProps) {
  const [showSuccessMessage, setShowSuccessMessage] = useState<string | null>(null);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const editorRef = useRef<HTMLDivElement>(null);

  const {
    components,
    selectedSlide,
    isLoading,
    isSaving,
    isRegenerating,
    error,
    canUndo,
    canRedo,
    historyLength,
    futureLength,
    loadComponents,
    updateSlide,
    regenerateSlide,
    regenerateVoiceover,
    uploadCustomAudio,
    regenerateLecture,
    recomposeVideo,
    selectSlide,
    reorderSlide,
    deleteSlide,
    insertMediaSlide,
    uploadMediaToSlide,
    undo,
    redo,
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

  // Handle video recomposition (declared early for keyboard shortcuts)
  const handleRecomposeVideo = useCallback(async () => {
    const result = await recomposeVideo(jobId, lecture.id, {
      quality: 'high',
      includeTransitions: true,
    });
    if (result?.success && onLectureUpdated) {
      onLectureUpdated({
        ...lecture,
        videoUrl: result.result?.video_url as string | undefined,
        isEdited: true,
      });
    }
  }, [jobId, lecture.id, recomposeVideo, onLectureUpdated, lecture]);

  // Handle delete slide (declared early for keyboard shortcuts)
  const handleDeleteSlide = useCallback(async (slideId: string) => {
    if (components && components.slides.length <= 1) {
      setShowSuccessMessage('Impossible de supprimer le dernier slide');
      setTimeout(() => setShowSuccessMessage(null), 3000);
      return;
    }
    await deleteSlide(jobId, lecture.id, slideId);
  }, [jobId, lecture.id, deleteSlide, components]);

  // Keyboard shortcuts handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Build key combination string
      const key = e.key;
      const hasCtrlOrMeta = e.ctrlKey || e.metaKey;
      const hasShift = e.shiftKey;

      // Skip if we're in an input or textarea (except for specific shortcuts)
      const isInInput = document.activeElement?.tagName === 'INPUT' ||
                        document.activeElement?.tagName === 'TEXTAREA';

      // Check for Undo shortcut (Ctrl+Z / Cmd+Z)
      if (hasCtrlOrMeta && !hasShift && key.toLowerCase() === 'z') {
        e.preventDefault();
        if (canUndo && !isSaving && !isRegenerating) {
          undo();
        }
        return;
      }

      // Check for Redo shortcut (Ctrl+Y / Cmd+Y or Ctrl+Shift+Z / Cmd+Shift+Z)
      if (hasCtrlOrMeta && (key.toLowerCase() === 'y' || (hasShift && key.toLowerCase() === 'z'))) {
        e.preventDefault();
        if (canRedo && !isSaving && !isRegenerating) {
          redo();
        }
        return;
      }

      // Check for Save shortcut (Ctrl+S / Cmd+S)
      if (hasCtrlOrMeta && key.toLowerCase() === 's') {
        e.preventDefault();
        if (components?.isEdited) {
          handleRecomposeVideo();
        }
        return;
      }

      // Skip remaining shortcuts if in input field
      if (isInInput) return;

      // Check for Delete shortcut
      if ((KEYBOARD_SHORTCUTS.DELETE as readonly string[]).includes(key) && selectedSlide && !isRegenerating && !isSaving) {
        e.preventDefault();
        handleDeleteSlide(selectedSlide.id);
        return;
      }

      // Check for Escape to close
      if ((KEYBOARD_SHORTCUTS.ESCAPE as readonly string[]).includes(key)) {
        onClose();
        return;
      }

      // Check for Space to play/pause (if not editing)
      if (key === ' ') {
        e.preventDefault();
        // Handled by SlidePreview component
        return;
      }

      // Arrow keys for navigation
      if (key === 'ArrowLeft' && components && selectedSlide) {
        e.preventDefault();
        const currentIndex = components.slides.findIndex(s => s.id === selectedSlide.id);
        if (currentIndex > 0) {
          selectSlide(components.slides[currentIndex - 1]);
        }
        return;
      }
      if (key === 'ArrowRight' && components && selectedSlide) {
        e.preventDefault();
        const currentIndex = components.slides.findIndex(s => s.id === selectedSlide.id);
        if (currentIndex < components.slides.length - 1) {
          selectSlide(components.slides[currentIndex + 1]);
        }
        return;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedSlide, components, isRegenerating, isSaving, onClose, selectSlide, handleDeleteSlide, canUndo, canRedo, undo, redo, handleRecomposeVideo]);

  // Handle slide update
  const handleSlideUpdate = useCallback(async (updates: UpdateSlideRequest) => {
    if (!selectedSlide) return;
    await updateSlide(jobId, lecture.id, selectedSlide.id, updates);
  }, [jobId, lecture.id, selectedSlide, updateSlide]);

  // Handle slide regeneration
  const handleRegenerateSlide = useCallback(async (slideId?: string) => {
    const targetSlide = slideId ? components?.slides.find(s => s.id === slideId) : selectedSlide;
    if (!targetSlide) return;
    await regenerateSlide(jobId, lecture.id, targetSlide.id, {
      regenerateImage: true,
      regenerateAnimation: targetSlide.type === 'code' || targetSlide.type === 'code_demo',
      useEditedContent: true,
    });
  }, [jobId, lecture.id, selectedSlide, components, regenerateSlide]);

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
        videoUrl: result.result?.video_url as string | undefined,
        isEdited: true,
      });
    }
  }, [jobId, lecture.id, regenerateLecture, onLectureUpdated, lecture]);

  // Handle slide reorder
  const handleReorderSlide = useCallback(async (slideId: string, newIndex: number) => {
    await reorderSlide(jobId, lecture.id, slideId, newIndex);
  }, [jobId, lecture.id, reorderSlide]);

  // Handle insert media - opens file picker
  const handleInsertMedia = useCallback(async (type: MediaType, afterSlideId?: string) => {
    // Create hidden file input
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = type === 'image'
      ? 'image/jpeg,image/png,image/gif,image/webp'
      : type === 'video'
        ? 'video/mp4,video/webm,video/quicktime'
        : 'audio/mp3,audio/wav,audio/m4a';

    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        await insertMediaSlide(jobId, lecture.id, type, file, {
          insertAfterSlideId: afterSlideId,
          duration: type === 'image' ? 5.0 : undefined,
        });
      }
    };

    input.click();
  }, [jobId, lecture.id, insertMediaSlide]);

  // Handle media upload to existing slide
  const handleUploadMedia = useCallback(async (type: MediaType, file: File) => {
    if (!selectedSlide) return;
    await uploadMediaToSlide(jobId, lecture.id, selectedSlide.id, type, file);
  }, [jobId, lecture.id, selectedSlide, uploadMediaToSlide]);

  // Handle slide change from preview navigation
  const handleSlideChange = useCallback((index: number) => {
    if (components && index >= 0 && index < components.slides.length) {
      selectSlide(components.slides[index]);
    }
  }, [components, selectSlide]);

  // Get current slide index
  const currentSlideIndex = components?.slides.findIndex(s => s.id === selectedSlide?.id) ?? 0;

  if (isLoading) {
    return (
      <div className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center">
        <div className="text-center">
          <div className="relative w-16 h-16 mx-auto mb-4">
            <div className="absolute inset-0 border-4 border-purple-500/30 rounded-full" />
            <div className="absolute inset-0 border-4 border-purple-500 border-t-transparent rounded-full animate-spin" />
          </div>
          <p className="text-white font-medium">Chargement de l'éditeur...</p>
          <p className="text-gray-500 text-sm mt-1">{lecture.title}</p>
        </div>
      </div>
    );
  }

  if (error && !components) {
    return (
      <div className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center">
        <div className="bg-gray-900 rounded-xl p-8 text-center max-w-md mx-4">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-500/20 flex items-center justify-center">
            <svg className="w-8 h-8 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <h3 className="text-white text-lg font-semibold mb-2">Erreur de chargement</h3>
          <p className="text-gray-400 mb-6">{error}</p>
          <button
            onClick={onClose}
            className="px-6 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors"
          >
            Fermer
          </button>
        </div>
      </div>
    );
  }

  if (!components) return null;

  return (
    <div ref={editorRef} className="fixed inset-0 bg-black/95 z-50 flex flex-col">
      {/* Header */}
      <header className="bg-gray-900/80 backdrop-blur border-b border-gray-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-4">
          {/* Close button */}
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
            title="Fermer (Échap)"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>

          {/* Title */}
          <div>
            <h2 className="text-white font-semibold text-sm">{lecture.title}</h2>
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <span>{components.slides.length} slides</span>
              <span className="text-gray-600">•</span>
              <span>{formatTotalDuration(components.totalDuration)}</span>
              {components.isEdited && (
                <>
                  <span className="text-gray-600">•</span>
                  <span className="text-yellow-500 flex items-center gap-1">
                    <span className="w-1.5 h-1.5 bg-yellow-500 rounded-full" />
                    Modifié
                  </span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          {/* Success message */}
          {showSuccessMessage && (
            <span className="text-green-400 text-sm flex items-center gap-1 px-3 py-1 bg-green-500/10 rounded-lg">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
              </svg>
              {showSuccessMessage}
            </span>
          )}

          {/* Editor Toolbar with undo/redo */}
          <EditorToolbar
            canUndo={canUndo}
            canRedo={canRedo}
            historyLength={historyLength}
            futureLength={futureLength}
            onUndo={undo}
            onRedo={redo}
            onSave={handleRecomposeVideo}
            onInsertMedia={(type) => handleInsertMedia(type, selectedSlide?.id)}
            onRecompose={handleRecomposeVideo}
            isSaving={isSaving}
            isRegenerating={isRegenerating}
            hasUnsavedChanges={components.isEdited}
          />

          {/* Divider */}
          <div className="w-px h-6 bg-gray-700" />

          {/* Shortcuts help */}
          <button
            onClick={() => setShowShortcuts(!showShortcuts)}
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
            title="Raccourcis clavier"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
            </svg>
          </button>

          {/* Regenerate voiceover */}
          <button
            onClick={handleRegenerateVoiceover}
            disabled={isRegenerating}
            className="px-3 py-1.5 text-sm bg-gray-800 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 transition-colors flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
              <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
            </svg>
            {isRegenerating ? 'Régénération...' : 'Voiceover'}
          </button>

          {/* Regenerate full lecture */}
          <button
            onClick={handleRegenerateLecture}
            disabled={isRegenerating}
            className="px-3 py-1.5 text-sm bg-green-600 text-white rounded-lg hover:bg-green-500 disabled:opacity-50 transition-colors flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Régénérer tout
          </button>
        </div>
      </header>

      {/* Shortcuts panel */}
      {showShortcuts && (
        <div className="absolute top-16 right-4 bg-gray-900 border border-gray-700 rounded-xl p-4 shadow-2xl z-10">
          <h4 className="text-white font-medium text-sm mb-3">Raccourcis clavier</h4>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between gap-6">
              <span className="text-gray-400">Annuler</span>
              <kbd className="px-2 py-0.5 bg-gray-800 rounded text-gray-300">Ctrl+Z</kbd>
            </div>
            <div className="flex justify-between gap-6">
              <span className="text-gray-400">Rétablir</span>
              <kbd className="px-2 py-0.5 bg-gray-800 rounded text-gray-300">Ctrl+Y</kbd>
            </div>
            <div className="flex justify-between gap-6">
              <span className="text-gray-400">Navigation slides</span>
              <kbd className="px-2 py-0.5 bg-gray-800 rounded text-gray-300">← →</kbd>
            </div>
            <div className="flex justify-between gap-6">
              <span className="text-gray-400">Recomposer vidéo</span>
              <kbd className="px-2 py-0.5 bg-gray-800 rounded text-gray-300">Ctrl+S</kbd>
            </div>
            <div className="flex justify-between gap-6">
              <span className="text-gray-400">Supprimer slide</span>
              <kbd className="px-2 py-0.5 bg-gray-800 rounded text-gray-300">Suppr</kbd>
            </div>
            <div className="flex justify-between gap-6">
              <span className="text-gray-400">Fermer</span>
              <kbd className="px-2 py-0.5 bg-gray-800 rounded text-gray-300">Échap</kbd>
            </div>
          </div>
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Timeline */}
        <aside className="w-64 bg-gray-900/50 border-r border-gray-800 flex-shrink-0">
          <SlideTimeline
            slides={components.slides}
            selectedSlide={selectedSlide}
            onSelectSlide={selectSlide}
            onReorderSlide={handleReorderSlide}
            onDeleteSlide={handleDeleteSlide}
            onInsertMedia={handleInsertMedia}
            onRegenerateSlide={(slideId) => handleRegenerateSlide(slideId)}
          />
        </aside>

        {/* Center: Preview */}
        <main className="flex-1 bg-gray-950 overflow-hidden p-4">
          <SlidePreview
            slide={selectedSlide}
            voiceover={components.voiceover}
            lectureComponents={components}
            currentSlideIndex={currentSlideIndex}
            onSlideChange={handleSlideChange}
          />
        </main>

        {/* Right: Properties */}
        <aside className="w-80 bg-gray-900/50 border-l border-gray-800 flex-shrink-0">
          <SlideProperties
            slide={selectedSlide}
            voiceover={components.voiceover}
            isSaving={isSaving}
            isRegenerating={isRegenerating}
            onUpdate={handleSlideUpdate}
            onRegenerate={() => handleRegenerateSlide()}
            onUploadAudio={handleUploadAudio}
            onUploadMedia={handleUploadMedia}
          />
        </aside>
      </div>

      {/* Footer status bar */}
      <footer className="bg-gray-900/80 backdrop-blur border-t border-gray-800 px-4 py-2 flex items-center justify-between text-xs text-gray-400">
        <div className="flex items-center gap-4">
          {selectedSlide && (
            <span>
              Slide {currentSlideIndex + 1}/{components.slides.length}
            </span>
          )}
          {isSaving && (
            <span className="flex items-center gap-1 text-blue-400">
              <svg className="animate-spin h-3 w-3" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Sauvegarde...
            </span>
          )}
          {isRegenerating && (
            <span className="flex items-center gap-1 text-purple-400">
              <svg className="animate-spin h-3 w-3" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Régénération en cours...
            </span>
          )}
        </div>
        <div className="flex items-center gap-4">
          <span>Appuyez sur <kbd className="px-1.5 py-0.5 bg-gray-800 rounded text-gray-300">?</kbd> pour les raccourcis</span>
        </div>
      </footer>
    </div>
  );
}

export default LectureEditor;
