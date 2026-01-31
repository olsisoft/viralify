'use client';

import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useLectureEditor } from '../../hooks/useLectureEditor';
import { useAssetLibrary } from '../../hooks/useAssetLibrary';
import { SlideTimeline } from './SlideTimeline';
import { SlidePreview } from './SlidePreview';
import { SlideProperties } from './SlideProperties';
import { EditorToolbar } from './EditorToolbar';
import { SubtitlesEditor } from './SubtitlesEditor';
import { AudioTimeline } from './AudioTimeline';
import { AudioMixer } from './AudioMixer';
import { TransitionsPanel } from './TransitionsPanel';
import { VisualEffectsPanel } from './VisualEffectsPanel';
import { OverlaysEditor } from './OverlaysEditor';
import { ExportPanel } from './ExportPanel';
import { CollaborationPanel } from './CollaborationPanel';
import { AssetLibraryPanel } from './AssetLibraryPanel';
import { LayersPanel } from './LayersPanel';
import type { Lecture } from '../../lib/course-types';
import type {
  SlideComponent,
  SlideElement,
  UpdateSlideRequest,
  MediaType,
  SubtitleTrack,
  SubtitleCue,
  AudioTrack,
  SlideTransition,
  VisualEffect,
  Overlay,
  ExportSettings,
  CollaborationState,
  Comment,
  VersionHistoryEntry,
  AddElementRequest,
  UpdateElementRequest,
} from '../../lib/lecture-editor-types';
import { formatTotalDuration, KEYBOARD_SHORTCUTS } from '../../lib/lecture-editor-types';

// Panel tabs type
type PanelTab = 'properties' | 'layers' | 'subtitles' | 'audio' | 'transitions' | 'effects' | 'overlays' | 'export' | 'collaboration';

// Left sidebar tabs
type LeftSidebarTab = 'timeline' | 'assets';

interface LectureEditorProps {
  jobId: string;
  lecture: Lecture;
  onClose: () => void;
  onLectureUpdated?: (lecture: Lecture) => void;
}

export function LectureEditor({ jobId, lecture, onClose, onLectureUpdated }: LectureEditorProps) {
  const [showSuccessMessage, setShowSuccessMessage] = useState<string | null>(null);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [activePanel, setActivePanel] = useState<PanelTab>('properties');
  const [showBottomPanel, setShowBottomPanel] = useState(false);
  const [bottomPanelTab, setBottomPanelTab] = useState<'audio-timeline' | 'mixer'>('audio-timeline');
  const [leftSidebarTab, setLeftSidebarTab] = useState<LeftSidebarTab>('timeline');
  const [canvasSelectedElementId, setCanvasSelectedElementId] = useState<string | null>(null);
  const editorRef = useRef<HTMLDivElement>(null);

  // Professional editor state
  const [subtitleTracks, setSubtitleTracks] = useState<SubtitleTrack[]>([{
    id: 'default-track',
    language: 'fr',
    label: 'Français',
    cues: [],
    isDefault: true,
  }]);
  const [audioTracks, setAudioTracks] = useState<AudioTrack[]>([]);
  const [transitions, setTransitions] = useState<SlideTransition[]>([]);
  const [visualEffects, setVisualEffects] = useState<VisualEffect[]>([]);
  const [overlays, setOverlays] = useState<Overlay[]>([]);
  const [exportSettings, setExportSettings] = useState<ExportSettings>({
    resolution: '1080p',
    format: 'mp4',
    quality: 'high',
    fps: 30,
    aspectRatio: '16:9',
    includeSubtitles: true,
    burnSubtitles: false,
    watermark: { enabled: false },
  });
  const [collaboration, setCollaboration] = useState<CollaborationState>({
    comments: [],
    versionHistory: [],
    shareLinks: [],
    activeUsers: [],
  });

  // Asset library hook
  const assetLibrary = useAssetLibrary({
    jobId,
    onUploadComplete: (asset) => {
      setShowSuccessMessage(`Asset "${asset.filename}" uploadé`);
      setTimeout(() => setShowSuccessMessage(null), 2000);
    },
    onError: (error) => {
      console.error('Asset upload error:', error);
    },
  });

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
    // Element management for canvas
    addElement,
    updateElement,
    deleteElement,
    addImageElement,
    duplicateElement,
    bringElementToFront,
    sendElementToBack,
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
      quality: '1080p',
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

  // Professional editor handlers
  const handleSubtitleTrackChange = useCallback((track: SubtitleTrack) => {
    setSubtitleTracks(tracks =>
      tracks.map(t => t.id === track.id ? track : t)
    );
  }, []);

  const handleAddSubtitleCue = useCallback((trackId: string, cue: SubtitleCue) => {
    setSubtitleTracks(tracks =>
      tracks.map(t => t.id === trackId ? { ...t, cues: [...t.cues, cue] } : t)
    );
  }, []);

  const handleUpdateSubtitleCue = useCallback((trackId: string, cue: SubtitleCue) => {
    setSubtitleTracks(tracks =>
      tracks.map(t => t.id === trackId
        ? { ...t, cues: t.cues.map(c => c.id === cue.id ? cue : c) }
        : t
      )
    );
  }, []);

  const handleDeleteSubtitleCue = useCallback((trackId: string, cueId: string) => {
    setSubtitleTracks(tracks =>
      tracks.map(t => t.id === trackId
        ? { ...t, cues: t.cues.filter(c => c.id !== cueId) }
        : t
      )
    );
  }, []);

  const handleImportSubtitles = useCallback((trackId: string, srtContent: string) => {
    // Parse SRT format
    const lines = srtContent.trim().split('\n');
    const cues: SubtitleCue[] = [];
    let i = 0;

    while (i < lines.length) {
      const indexLine = lines[i]?.trim();
      if (!indexLine || !/^\d+$/.test(indexLine)) {
        i++;
        continue;
      }

      const timeLine = lines[i + 1]?.trim();
      if (!timeLine) {
        i++;
        continue;
      }

      const timeMatch = timeLine.match(/(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})/);
      if (!timeMatch) {
        i++;
        continue;
      }

      const startTime = parseInt(timeMatch[1]) * 3600 + parseInt(timeMatch[2]) * 60 + parseInt(timeMatch[3]) + parseInt(timeMatch[4]) / 1000;
      const endTime = parseInt(timeMatch[5]) * 3600 + parseInt(timeMatch[6]) * 60 + parseInt(timeMatch[7]) + parseInt(timeMatch[8]) / 1000;

      let text = '';
      let j = i + 2;
      while (j < lines.length && lines[j]?.trim() !== '') {
        text += (text ? '\n' : '') + lines[j];
        j++;
      }

      cues.push({
        id: `cue-${Date.now()}-${cues.length}`,
        startTime,
        endTime,
        text: text.trim(),
      });

      i = j + 1;
    }

    setSubtitleTracks(tracks =>
      tracks.map(t => t.id === trackId ? { ...t, cues } : t)
    );
  }, []);

  const handleAudioTracksChange = useCallback((tracks: AudioTrack[]) => {
    setAudioTracks(tracks);
  }, []);

  const handleTransitionsChange = useCallback((newTransitions: SlideTransition[]) => {
    setTransitions(newTransitions);
  }, []);

  const handleEffectsChange = useCallback((slideId: string, effects: VisualEffect[]) => {
    setVisualEffects(prev => {
      const filtered = prev.filter(e => e.slideId !== slideId);
      return [...filtered, ...effects.map(e => ({ ...e, slideId }))];
    });
  }, []);

  const handleOverlaysChange = useCallback((newOverlays: Overlay[]) => {
    setOverlays(newOverlays);
  }, []);

  const handleExportSettingsChange = useCallback((settings: ExportSettings) => {
    setExportSettings(settings);
  }, []);

  const handleStartExport = useCallback(async () => {
    setShowSuccessMessage('Export démarré...');
    // TODO: Implement actual export API call
    setTimeout(() => {
      setShowSuccessMessage('Export terminé!');
      setTimeout(() => setShowSuccessMessage(null), 3000);
    }, 3000);
  }, []);

  const handleAddComment = useCallback((comment: Omit<Comment, 'id' | 'createdAt' | 'replies'>) => {
    const newComment: Comment = {
      ...comment,
      id: `comment-${Date.now()}`,
      createdAt: new Date().toISOString(),
      replies: [],
    };
    setCollaboration(prev => ({
      ...prev,
      comments: [...prev.comments, newComment],
    }));
  }, []);

  const handleReplyToComment = useCallback((commentId: string, reply: Omit<Comment, 'id' | 'createdAt' | 'replies'>) => {
    const newReply: Comment = {
      ...reply,
      id: `reply-${Date.now()}`,
      createdAt: new Date().toISOString(),
      replies: [],
    };
    setCollaboration(prev => ({
      ...prev,
      comments: prev.comments.map(c =>
        c.id === commentId ? { ...c, replies: [...c.replies, newReply] } : c
      ),
    }));
  }, []);

  const handleResolveComment = useCallback((commentId: string) => {
    setCollaboration(prev => ({
      ...prev,
      comments: prev.comments.map(c =>
        c.id === commentId ? { ...c, resolved: true, resolvedAt: new Date().toISOString() } : c
      ),
    }));
  }, []);

  // Get effects for selected slide
  const selectedSlideEffects = selectedSlide
    ? visualEffects.filter(e => e.slideId === selectedSlide.id)
    : [];

  // =========================================================================
  // Element callbacks for InteractiveCanvas
  // These wrap the hook functions to provide jobId, lectureId, slideId
  // =========================================================================
  const handleAddElement = useCallback(async (request: AddElementRequest) => {
    if (!selectedSlide) return null;
    return addElement(jobId, lecture.id, selectedSlide.id, request);
  }, [jobId, lecture.id, selectedSlide, addElement]);

  const handleUpdateElement = useCallback(async (elementId: string, updates: UpdateElementRequest) => {
    if (!selectedSlide) return null;
    return updateElement(jobId, lecture.id, selectedSlide.id, elementId, updates);
  }, [jobId, lecture.id, selectedSlide, updateElement]);

  const handleDeleteElementFromCanvas = useCallback(async (elementId: string) => {
    if (!selectedSlide) return false;
    return deleteElement(jobId, lecture.id, selectedSlide.id, elementId);
  }, [jobId, lecture.id, selectedSlide, deleteElement]);

  const handleUploadImage = useCallback(async (file: File, position?: { x: number; y: number }) => {
    if (!selectedSlide) return null;
    return addImageElement(jobId, lecture.id, selectedSlide.id, file, position);
  }, [jobId, lecture.id, selectedSlide, addImageElement]);

  const handleDuplicateElement = useCallback(async (element: SlideElement) => {
    if (!selectedSlide) return null;
    return duplicateElement(jobId, lecture.id, selectedSlide.id, element);
  }, [jobId, lecture.id, selectedSlide, duplicateElement]);

  const handleBringToFront = useCallback(async (elementId: string) => {
    if (!selectedSlide) return false;
    return bringElementToFront(jobId, lecture.id, selectedSlide.id, elementId);
  }, [jobId, lecture.id, selectedSlide, bringElementToFront]);

  const handleSendToBack = useCallback(async (elementId: string) => {
    if (!selectedSlide) return false;
    return sendElementToBack(jobId, lecture.id, selectedSlide.id, elementId);
  }, [jobId, lecture.id, selectedSlide, sendElementToBack]);

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
        {/* Left: Timeline / Assets */}
        <aside className="w-64 bg-gray-900/50 border-r border-gray-800 flex-shrink-0 flex flex-col">
          {/* Left sidebar tabs */}
          <div className="flex border-b border-gray-800">
            <button
              onClick={() => setLeftSidebarTab('timeline')}
              className={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
                leftSidebarTab === 'timeline'
                  ? 'text-white bg-gray-800 border-b-2 border-purple-500'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
              }`}
            >
              <span className="flex items-center justify-center gap-1.5">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
                </svg>
                Timeline
              </span>
            </button>
            <button
              onClick={() => setLeftSidebarTab('assets')}
              className={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
                leftSidebarTab === 'assets'
                  ? 'text-white bg-gray-800 border-b-2 border-purple-500'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
              }`}
            >
              <span className="flex items-center justify-center gap-1.5">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                Assets
              </span>
            </button>
          </div>

          {/* Left sidebar content */}
          <div className="flex-1 overflow-hidden">
            {leftSidebarTab === 'timeline' && (
              <SlideTimeline
                slides={components.slides}
                selectedSlide={selectedSlide}
                onSelectSlide={selectSlide}
                onReorderSlide={handleReorderSlide}
                onDeleteSlide={handleDeleteSlide}
                onInsertMedia={handleInsertMedia}
                onRegenerateSlide={(slideId) => handleRegenerateSlide(slideId)}
              />
            )}
            {leftSidebarTab === 'assets' && (
              <AssetLibraryPanel
                assets={assetLibrary.assets}
                filteredAssets={assetLibrary.filteredAssets}
                uploadProgress={assetLibrary.uploadProgress}
                selectedAssetId={assetLibrary.selectedAssetId}
                searchQuery={assetLibrary.searchQuery}
                filterType={assetLibrary.filterType}
                isLoading={assetLibrary.isLoading}
                onUpload={(files) => assetLibrary.uploadMultipleAssets(files)}
                onDelete={assetLibrary.deleteAsset}
                onSelect={assetLibrary.selectAsset}
                onSearchChange={assetLibrary.setSearchQuery}
                onFilterChange={assetLibrary.setFilterType}
              />
            )}
          </div>
        </aside>

        {/* Center: Preview with Interactive Canvas */}
        <main className="flex-1 bg-gray-950 overflow-hidden p-4">
          <SlidePreview
            slide={selectedSlide}
            voiceover={components.voiceover}
            lectureComponents={components}
            currentSlideIndex={currentSlideIndex}
            onSlideChange={handleSlideChange}
            // Element editing callbacks for InteractiveCanvas
            onAddElement={handleAddElement}
            onUpdateElement={handleUpdateElement}
            onDeleteElement={handleDeleteElementFromCanvas}
            onUploadImage={handleUploadImage}
            onDuplicateElement={handleDuplicateElement}
            onBringToFront={handleBringToFront}
            onSendToBack={handleSendToBack}
            isEditing={isRegenerating}
            isSaving={isSaving}
          />
        </main>

        {/* Right: Panel with tabs */}
        <aside className="w-80 bg-gray-900/50 border-l border-gray-800 flex-shrink-0 flex flex-col">
          {/* Panel tabs */}
          <div className="flex border-b border-gray-800 overflow-x-auto">
            {[
              { id: 'properties' as const, icon: '⚙️', label: 'Propriétés' },
              { id: 'layers' as const, icon: '📚', label: 'Calques' },
              { id: 'subtitles' as const, icon: '💬', label: 'Sous-titres' },
              { id: 'transitions' as const, icon: '✨', label: 'Transitions' },
              { id: 'effects' as const, icon: '🎨', label: 'Effets' },
              { id: 'overlays' as const, icon: '📝', label: 'Overlays' },
              { id: 'export' as const, icon: '📤', label: 'Export' },
              { id: 'collaboration' as const, icon: '👥', label: 'Collab' },
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActivePanel(tab.id)}
                className={`flex-shrink-0 px-3 py-2 text-xs transition-colors ${
                  activePanel === tab.id
                    ? 'text-white bg-gray-800 border-b-2 border-purple-500'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
                }`}
                title={tab.label}
              >
                <span className="block">{tab.icon}</span>
              </button>
            ))}
          </div>

          {/* Panel content */}
          <div className="flex-1 overflow-hidden">
            {activePanel === 'properties' && (
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
            )}

            {activePanel === 'layers' && selectedSlide && (
              <LayersPanel
                elements={selectedSlide.elements}
                selectedElementId={canvasSelectedElementId}
                onSelectElement={setCanvasSelectedElementId}
                onReorderElements={async (newOrder) => {
                  // Update z-indices for all elements
                  for (const element of newOrder) {
                    await updateElement(jobId, lecture.id, selectedSlide.id, element.id, {
                      // z-index is managed by the LayersPanel reorder
                    });
                  }
                }}
                onToggleVisibility={async (elementId) => {
                  const element = selectedSlide.elements.find(e => e.id === elementId);
                  if (element) {
                    await updateElement(jobId, lecture.id, selectedSlide.id, elementId, {
                      visible: !element.visible,
                    });
                  }
                }}
                onToggleLock={async (elementId) => {
                  const element = selectedSlide.elements.find(e => e.id === elementId);
                  if (element) {
                    await updateElement(jobId, lecture.id, selectedSlide.id, elementId, {
                      locked: !element.locked,
                    });
                  }
                }}
                onDeleteElement={async (elementId) => {
                  await deleteElement(jobId, lecture.id, selectedSlide.id, elementId);
                }}
                onDuplicateElement={async (elementId) => {
                  const element = selectedSlide.elements.find(e => e.id === elementId);
                  if (element) {
                    await duplicateElement(jobId, lecture.id, selectedSlide.id, element);
                  }
                }}
                onBringToFront={async (elementId) => {
                  await bringElementToFront(jobId, lecture.id, selectedSlide.id, elementId);
                }}
                onSendToBack={async (elementId) => {
                  await sendElementToBack(jobId, lecture.id, selectedSlide.id, elementId);
                }}
                disabled={isRegenerating || isSaving}
              />
            )}

            {activePanel === 'layers' && !selectedSlide && (
              <div className="flex items-center justify-center h-full text-gray-500 text-sm p-4">
                Sélectionnez un slide pour gérer ses calques
              </div>
            )}

            {activePanel === 'subtitles' && (
              <SubtitlesEditor
                tracks={subtitleTracks}
                currentTime={0}
                duration={components.totalDuration}
                onTrackChange={handleSubtitleTrackChange}
                onAddCue={(cue) => handleAddSubtitleCue(subtitleTracks[0]?.id || '', cue)}
                onUpdateCue={(cue) => handleUpdateSubtitleCue(subtitleTracks[0]?.id || '', cue)}
                onDeleteCue={(cueId) => handleDeleteSubtitleCue(subtitleTracks[0]?.id || '', cueId)}
                onImportSRT={(srt) => handleImportSubtitles(subtitleTracks[0]?.id || '', srt)}
              />
            )}

            {activePanel === 'transitions' && (
              <TransitionsPanel
                slides={components.slides}
                transitions={transitions}
                selectedSlideId={selectedSlide?.id || null}
                onTransitionsChange={handleTransitionsChange}
              />
            )}

            {activePanel === 'effects' && selectedSlide && (
              <VisualEffectsPanel
                slide={selectedSlide}
                effects={selectedSlideEffects}
                onEffectsChange={(effects) => handleEffectsChange(selectedSlide.id, effects)}
              />
            )}

            {activePanel === 'effects' && !selectedSlide && (
              <div className="flex items-center justify-center h-full text-gray-500 text-sm p-4">
                Sélectionnez un slide pour modifier ses effets
              </div>
            )}

            {activePanel === 'overlays' && (
              <OverlaysEditor
                overlays={overlays}
                duration={components.totalDuration}
                currentTime={0}
                selectedSlideId={selectedSlide?.id || null}
                onOverlaysChange={handleOverlaysChange}
              />
            )}

            {activePanel === 'export' && (
              <ExportPanel
                settings={exportSettings}
                subtitleTracks={subtitleTracks}
                duration={components.totalDuration}
                onSettingsChange={handleExportSettingsChange}
                onStartExport={handleStartExport}
                isExporting={false}
                exportProgress={0}
              />
            )}

            {activePanel === 'collaboration' && (
              <CollaborationPanel
                comments={collaboration.comments}
                versionHistory={collaboration.versionHistory}
                shareLinks={collaboration.shareLinks}
                currentTime={0}
                onAddComment={(text, timestamp) => handleAddComment({
                  userId: 'current-user',
                  userName: 'Vous',
                  text,
                  timestamp,
                })}
                onReplyToComment={(commentId, text) => handleReplyToComment(commentId, {
                  userId: 'current-user',
                  userName: 'Vous',
                  text,
                })}
                onResolveComment={handleResolveComment}
                onSeekToTimestamp={() => {}}
                onRestoreVersion={() => {}}
              />
            )}
          </div>
        </aside>
      </div>

      {/* Bottom panel toggle button */}
      <button
        onClick={() => setShowBottomPanel(!showBottomPanel)}
        className="absolute bottom-12 left-1/2 transform -translate-x-1/2 px-4 py-1 bg-gray-800 text-gray-400 rounded-t-lg hover:text-white hover:bg-gray-700 transition-colors text-xs flex items-center gap-2 z-10"
      >
        <span>{showBottomPanel ? '▼' : '▲'}</span>
        <span>Audio Timeline</span>
      </button>

      {/* Bottom panel for audio */}
      {showBottomPanel && (
        <div className="h-64 bg-gray-900 border-t border-gray-800 flex flex-col">
          {/* Bottom panel tabs */}
          <div className="flex border-b border-gray-800">
            <button
              onClick={() => setBottomPanelTab('audio-timeline')}
              className={`px-4 py-2 text-sm ${
                bottomPanelTab === 'audio-timeline'
                  ? 'text-white bg-gray-800 border-b-2 border-purple-500'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              🎵 Timeline Audio
            </button>
            <button
              onClick={() => setBottomPanelTab('mixer')}
              className={`px-4 py-2 text-sm ${
                bottomPanelTab === 'mixer'
                  ? 'text-white bg-gray-800 border-b-2 border-purple-500'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              🎚️ Mixer
            </button>
          </div>

          {/* Bottom panel content */}
          <div className="flex-1 overflow-hidden">
            {bottomPanelTab === 'audio-timeline' && (
              <AudioTimeline
                tracks={audioTracks.length > 0 ? audioTracks : [{
                  id: 'voiceover-track',
                  name: 'Voiceover',
                  type: 'voiceover',
                  volume: 1,
                  pan: 0,
                  muted: false,
                  solo: false,
                  clips: components.voiceover ? [{
                    id: 'voiceover-clip',
                    trackId: 'voiceover-track',
                    startTime: 0,
                    duration: components.totalDuration,
                    offset: 0,
                    name: 'Voiceover principal',
                  }] : [],
                }]}
                duration={components.totalDuration}
                currentTime={0}
                zoom={1}
                onTracksChange={handleAudioTracksChange}
                onSeek={() => {}}
              />
            )}

            {bottomPanelTab === 'mixer' && (
              <AudioMixer
                tracks={audioTracks.length > 0 ? audioTracks : [{
                  id: 'voiceover-track',
                  name: 'Voiceover',
                  type: 'voiceover',
                  volume: 1,
                  pan: 0,
                  muted: false,
                  solo: false,
                  clips: [],
                }]}
                masterVolume={1}
                onTracksChange={handleAudioTracksChange}
                onMasterVolumeChange={() => {}}
              />
            )}
          </div>
        </div>
      )}

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
