'use client';

import React, { useRef, useState, useEffect, useCallback } from 'react';
import type { SlideComponent, VoiceoverComponent, LectureComponents, SlideElement, AddElementRequest, UpdateElementRequest } from '../../lib/lecture-editor-types';
import { getSlideTypeLabel, formatDuration } from '../../lib/lecture-editor-types';
import { InteractiveCanvas } from './InteractiveCanvas';

interface SlidePreviewProps {
  slide: SlideComponent | null;
  voiceover?: VoiceoverComponent;
  lectureComponents?: LectureComponents;
  currentSlideIndex?: number;
  onSlideChange?: (index: number) => void;
  // Element editing props (optional - enables canvas mode)
  onAddElement?: (request: AddElementRequest) => Promise<SlideElement | null>;
  onUpdateElement?: (elementId: string, updates: UpdateElementRequest) => Promise<SlideElement | null>;
  onDeleteElement?: (elementId: string) => Promise<boolean>;
  onUploadImage?: (file: File, position?: { x: number; y: number }) => Promise<SlideElement | null>;
  onDuplicateElement?: (element: SlideElement) => Promise<SlideElement | null>;
  onBringToFront?: (elementId: string) => Promise<boolean>;
  onSendToBack?: (elementId: string) => Promise<boolean>;
  isEditing?: boolean;
  isSaving?: boolean;
}

export function SlidePreview({
  slide,
  voiceover,
  lectureComponents,
  currentSlideIndex = 0,
  onSlideChange,
  onAddElement,
  onUpdateElement,
  onDeleteElement,
  onUploadImage,
  onDuplicateElement,
  onBringToFront,
  onSendToBack,
  isEditing = false,
  isSaving = false,
}: SlidePreviewProps) {
  // Check if canvas editing is enabled
  const canvasEnabled = Boolean(onAddElement && onUpdateElement && onDeleteElement && onUploadImage);
  const [editMode, setEditMode] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [isVideoMode, setIsVideoMode] = useState(false);
  const [showControls, setShowControls] = useState(true);

  // Reset play state when slide changes
  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
  }, [slide?.id]);

  // Toggle play/pause
  const togglePlay = useCallback(() => {
    if (isVideoMode && videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
    } else if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
    }
    setIsPlaying(!isPlaying);
  }, [isPlaying, isVideoMode]);

  // Handle time update
  const handleTimeUpdate = useCallback(() => {
    const element = isVideoMode ? videoRef.current : audioRef.current;
    if (element) {
      setCurrentTime(element.currentTime);
    }
  }, [isVideoMode]);

  // Handle duration loaded
  const handleLoadedMetadata = useCallback(() => {
    const element = isVideoMode ? videoRef.current : audioRef.current;
    if (element) {
      setDuration(element.duration);
    }
  }, [isVideoMode]);

  // Handle seek
  const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    const element = isVideoMode ? videoRef.current : audioRef.current;
    if (element) {
      element.currentTime = time;
      setCurrentTime(time);
    }
  }, [isVideoMode]);

  // Handle volume change
  const handleVolumeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const vol = parseFloat(e.target.value);
    setVolume(vol);
    if (audioRef.current) audioRef.current.volume = vol;
    if (videoRef.current) videoRef.current.volume = vol;
  }, []);

  // Toggle mute
  const toggleMute = useCallback(() => {
    setIsMuted(!isMuted);
    if (audioRef.current) audioRef.current.muted = !isMuted;
    if (videoRef.current) videoRef.current.muted = !isMuted;
  }, [isMuted]);

  // Format time
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Check if we have a video URL
  const hasVideo = lectureComponents?.videoUrl;

  if (!slide) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-gray-500">
          <svg className="w-16 h-16 mx-auto mb-4 opacity-30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          <p>Sélectionnez un slide pour le prévisualiser</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Mode toggle */}
      <div className="flex items-center justify-center gap-2 mb-4">
        {/* Edit/Preview toggle when canvas is enabled */}
        {canvasEnabled && (
          <>
            <button
              onClick={() => { setEditMode(false); setIsVideoMode(false); }}
              className={`px-4 py-1.5 rounded-lg text-sm transition-colors ${
                !editMode && !isVideoMode ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              Aperçu
            </button>
            <button
              onClick={() => { setEditMode(true); setIsVideoMode(false); }}
              className={`px-4 py-1.5 rounded-lg text-sm transition-colors flex items-center gap-1.5 ${
                editMode ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
              </svg>
              Éditer
            </button>
            {hasVideo && <div className="w-px h-6 bg-gray-700" />}
          </>
        )}
        {hasVideo && (
          <>
            {!canvasEnabled && (
              <button
                onClick={() => setIsVideoMode(false)}
                className={`px-4 py-1.5 rounded-lg text-sm transition-colors ${
                  !isVideoMode ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                Slide
              </button>
            )}
            <button
              onClick={() => { setIsVideoMode(true); setEditMode(false); }}
              className={`px-4 py-1.5 rounded-lg text-sm transition-colors ${
                isVideoMode ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              Vidéo
            </button>
          </>
        )}
      </div>

      {/* Preview area */}
      <div
        className="flex-1 flex items-center justify-center"
        onMouseEnter={() => setShowControls(true)}
        onMouseLeave={() => !isPlaying && setShowControls(true)}
      >
        <div className="relative w-full max-w-4xl aspect-video bg-gray-900 rounded-xl overflow-hidden shadow-2xl">
          {/* Edit mode - Interactive Canvas */}
          {editMode && canvasEnabled && onAddElement && onUpdateElement && onDeleteElement && onUploadImage ? (
            <InteractiveCanvas
              slide={slide}
              onAddElement={onAddElement}
              onUpdateElement={onUpdateElement}
              onDeleteElement={onDeleteElement}
              onUploadImage={onUploadImage}
              onDuplicateElement={onDuplicateElement}
              onBringToFront={onBringToFront}
              onSendToBack={onSendToBack}
              isLoading={isSaving}
              disabled={isEditing}
            />
          ) : isVideoMode && hasVideo ? (
            // Video player mode
            <video
              ref={videoRef}
              src={lectureComponents?.videoUrl}
              className="w-full h-full object-contain"
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              onEnded={() => setIsPlaying(false)}
              onClick={togglePlay}
            />
          ) : slide.type === 'media' && slide.mediaUrl ? (
            // Media slide (user-inserted image or video)
            slide.mediaType === 'video' ? (
              <video
                src={slide.mediaUrl}
                className="w-full h-full object-contain"
                controls
              />
            ) : (
              <img
                src={slide.mediaUrl}
                alt={slide.title || 'Media'}
                className="w-full h-full object-contain"
              />
            )
          ) : slide.animationUrl ? (
            // Animation preview
            <video
              src={slide.animationUrl}
              className="w-full h-full object-contain"
              controls
              loop
            />
          ) : slide.imageUrl ? (
            // Image preview
            <img
              src={slide.imageUrl}
              alt={slide.title || 'Slide preview'}
              className="w-full h-full object-contain"
            />
          ) : (
            // Fallback content preview
            <div className="w-full h-full flex flex-col items-center justify-center p-8 bg-gradient-to-br from-gray-900 to-gray-800">
              <span className="text-5xl mb-4">
                {slide.type === 'code' || slide.type === 'code_demo' ? '💻' : '📝'}
              </span>
              <h3 className="text-white text-xl font-bold text-center mb-2">
                {slide.title || getSlideTypeLabel(slide.type)}
              </h3>
              {slide.subtitle && (
                <p className="text-gray-400 text-center mb-4">{slide.subtitle}</p>
              )}
              {slide.bulletPoints.length > 0 && (
                <ul className="space-y-2 text-left max-w-md">
                  {slide.bulletPoints.slice(0, 4).map((point, idx) => (
                    <li key={idx} className="text-gray-300 flex items-start gap-2 text-sm">
                      <span className="text-purple-500 mt-0.5">•</span>
                      <span>{point}</span>
                    </li>
                  ))}
                  {slide.bulletPoints.length > 4 && (
                    <li className="text-gray-500 text-sm">+{slide.bulletPoints.length - 4} autres...</li>
                  )}
                </ul>
              )}
              {slide.codeBlocks.length > 0 && (
                <div className="mt-4 w-full max-w-md">
                  <pre className="bg-black/50 rounded-lg p-3 overflow-x-auto">
                    <code className="text-green-400 text-xs font-mono">
                      {slide.codeBlocks[0]?.code.slice(0, 150)}
                      {slide.codeBlocks[0]?.code.length > 150 && '...'}
                    </code>
                  </pre>
                </div>
              )}
            </div>
          )}

          {/* Play overlay (non-video mode) */}
          {!isVideoMode && voiceover?.audioUrl && (
            <button
              onClick={togglePlay}
              className="absolute inset-0 flex items-center justify-center bg-black/30 opacity-0 hover:opacity-100 transition-opacity"
            >
              <div className="w-16 h-16 rounded-full bg-white/20 backdrop-blur flex items-center justify-center">
                {isPlaying ? (
                  <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                  </svg>
                ) : (
                  <svg className="w-8 h-8 text-white ml-1" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z" />
                  </svg>
                )}
              </div>
            </button>
          )}

          {/* Duration badge */}
          <div className="absolute top-4 left-4 bg-black/70 px-2.5 py-1 rounded-lg text-white text-xs font-medium">
            {formatDuration(slide.duration)}
          </div>

          {/* Status badges */}
          <div className="absolute top-4 right-4 flex gap-2">
            {slide.isEdited && (
              <span className="bg-yellow-500 px-2.5 py-1 rounded-lg text-black text-xs font-medium">
                Modifié
              </span>
            )}
            {slide.status === 'failed' && (
              <span className="bg-red-500 px-2.5 py-1 rounded-lg text-white text-xs font-medium">
                Erreur
              </span>
            )}
          </div>

          {/* Slide navigation (when in lecture mode) */}
          {lectureComponents && lectureComponents.slides.length > 1 && !isVideoMode && (
            <>
              {currentSlideIndex > 0 && (
                <button
                  onClick={() => onSlideChange?.(currentSlideIndex - 1)}
                  className="absolute left-4 top-1/2 -translate-y-1/2 w-10 h-10 rounded-full bg-black/50 hover:bg-black/70 flex items-center justify-center text-white transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                </button>
              )}
              {currentSlideIndex < lectureComponents.slides.length - 1 && (
                <button
                  onClick={() => onSlideChange?.(currentSlideIndex + 1)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 w-10 h-10 rounded-full bg-black/50 hover:bg-black/70 flex items-center justify-center text-white transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              )}
            </>
          )}
        </div>
      </div>

      {/* Controls bar */}
      {(voiceover?.audioUrl || (isVideoMode && hasVideo)) && (
        <div className="mt-4 bg-gray-900 rounded-xl p-3">
          {/* Progress bar */}
          <div className="flex items-center gap-3 mb-2">
            <span className="text-gray-400 text-xs w-10">{formatTime(currentTime)}</span>
            <input
              type="range"
              min={0}
              max={duration || 100}
              value={currentTime}
              onChange={handleSeek}
              className="flex-1 h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer
                [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3
                [&::-webkit-slider-thumb]:bg-purple-500 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer"
            />
            <span className="text-gray-400 text-xs w-10 text-right">{formatTime(duration)}</span>
          </div>

          {/* Control buttons */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {/* Play/Pause */}
              <button
                onClick={togglePlay}
                className="w-10 h-10 rounded-full bg-purple-600 hover:bg-purple-500 flex items-center justify-center text-white transition-colors"
              >
                {isPlaying ? (
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5 ml-0.5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z" />
                  </svg>
                )}
              </button>

              {/* Skip backward */}
              <button
                onClick={() => {
                  const element = isVideoMode ? videoRef.current : audioRef.current;
                  if (element) {
                    element.currentTime = Math.max(0, element.currentTime - 5);
                  }
                }}
                className="p-2 text-gray-400 hover:text-white transition-colors"
                title="-5s"
              >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M11 18V6l-8.5 6 8.5 6zm.5-6l8.5 6V6l-8.5 6z" />
                </svg>
              </button>

              {/* Skip forward */}
              <button
                onClick={() => {
                  const element = isVideoMode ? videoRef.current : audioRef.current;
                  if (element) {
                    element.currentTime = Math.min(duration, element.currentTime + 5);
                  }
                }}
                className="p-2 text-gray-400 hover:text-white transition-colors"
                title="+5s"
              >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M4 18l8.5-6L4 6v12zm9-12v12l8.5-6L13 6z" />
                </svg>
              </button>
            </div>

            {/* Volume control */}
            <div className="flex items-center gap-2">
              <button
                onClick={toggleMute}
                className="p-2 text-gray-400 hover:text-white transition-colors"
              >
                {isMuted || volume === 0 ? (
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z" />
                  </svg>
                )}
              </button>
              <input
                type="range"
                min={0}
                max={1}
                step={0.1}
                value={isMuted ? 0 : volume}
                onChange={handleVolumeChange}
                className="w-20 h-1 bg-gray-700 rounded-full appearance-none cursor-pointer
                  [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-2.5 [&::-webkit-slider-thumb]:h-2.5
                  [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer"
              />
            </div>
          </div>
        </div>
      )}

      {/* Voiceover text */}
      <div className="mt-4 bg-gray-900 rounded-xl p-4">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-gray-400 text-sm font-medium flex items-center gap-2">
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.91-3c-.49 0-.9.36-.98.85C16.52 14.2 14.47 16 12 16s-4.52-1.8-4.93-4.15c-.08-.49-.49-.85-.98-.85-.61 0-1.09.54-1 1.14.49 3 2.89 5.35 5.91 5.78V20c0 .55.45 1 1 1s1-.45 1-1v-2.08c3.02-.43 5.42-2.78 5.91-5.78.1-.6-.39-1.14-1-1.14z" />
            </svg>
            Texte du voiceover
          </h4>
          <span className="text-gray-500 text-xs">
            {slide.voiceoverText?.split(' ').length || 0} mots
          </span>
        </div>
        <p className="text-gray-300 text-sm leading-relaxed">
          {slide.voiceoverText || (
            <span className="text-gray-600 italic">Aucun texte de voiceover défini</span>
          )}
        </p>
      </div>

      {/* Hidden audio element */}
      {voiceover?.audioUrl && !isVideoMode && (
        <audio
          ref={audioRef}
          src={voiceover.audioUrl}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={() => setIsPlaying(false)}
          className="hidden"
        />
      )}
    </div>
  );
}

export default SlidePreview;
