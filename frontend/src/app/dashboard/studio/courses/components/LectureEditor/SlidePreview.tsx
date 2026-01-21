'use client';

import React, { useRef, useState } from 'react';
import type { SlideComponent, VoiceoverComponent } from '../../lib/lecture-editor-types';
import { getSlideTypeLabel, formatDuration } from '../../lib/lecture-editor-types';

interface SlidePreviewProps {
  slide: SlideComponent | null;
  voiceover?: VoiceoverComponent;
}

export function SlidePreview({ slide, voiceover }: SlidePreviewProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  if (!slide) {
    return (
      <div className="text-gray-500 text-center">
        <p>S\u00e9lectionnez un slide pour le pr\u00e9visualiser</p>
      </div>
    );
  }

  const toggleAudio = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  return (
    <div className="flex flex-col items-center max-w-4xl w-full">
      {/* Slide preview */}
      <div className="relative w-full aspect-video bg-gray-800 rounded-lg overflow-hidden shadow-2xl">
        {slide.imageUrl ? (
          <img
            src={slide.imageUrl}
            alt={slide.title || 'Slide preview'}
            className="w-full h-full object-contain"
          />
        ) : slide.animationUrl ? (
          <video
            src={slide.animationUrl}
            className="w-full h-full object-contain"
            controls
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center p-8">
            <span className="text-6xl mb-4">
              {slide.type === 'code' || slide.type === 'code_demo' ? '\ud83d\udcbb' : '\ud83d\udcdd'}
            </span>
            <h3 className="text-white text-2xl font-bold text-center mb-2">
              {slide.title || getSlideTypeLabel(slide.type)}
            </h3>
            {slide.subtitle && (
              <p className="text-gray-400 text-lg text-center">{slide.subtitle}</p>
            )}
            {slide.content && (
              <p className="text-gray-300 text-center mt-4 max-w-lg">{slide.content}</p>
            )}
            {slide.bulletPoints.length > 0 && (
              <ul className="mt-4 space-y-2 text-left">
                {slide.bulletPoints.map((point, idx) => (
                  <li key={idx} className="text-gray-300 flex items-start gap-2">
                    <span className="text-purple-500">-</span>
                    {point}
                  </li>
                ))}
              </ul>
            )}
            {slide.codeBlocks.length > 0 && (
              <div className="mt-4 w-full max-w-lg">
                <pre className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
                  <code className="text-green-400 text-sm">
                    {slide.codeBlocks[0]?.code.slice(0, 200)}
                    {slide.codeBlocks[0]?.code.length > 200 && '...'}
                  </code>
                </pre>
              </div>
            )}
          </div>
        )}

        {/* Duration badge */}
        <div className="absolute bottom-4 right-4 bg-black/70 px-2 py-1 rounded text-white text-sm">
          {formatDuration(slide.duration)}
        </div>

        {/* Status badge */}
        {slide.isEdited && (
          <div className="absolute top-4 right-4 bg-yellow-500/90 px-2 py-1 rounded text-black text-sm font-medium">
            Modifi\u00e9
          </div>
        )}
      </div>

      {/* Voiceover text preview */}
      <div className="w-full mt-4 bg-gray-900 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-gray-400 text-sm font-medium">Texte du voiceover</h4>
          {voiceover?.audioUrl && (
            <button
              onClick={toggleAudio}
              className="flex items-center gap-2 px-3 py-1 bg-purple-600 text-white rounded text-sm hover:bg-purple-500"
            >
              {isPlaying ? (
                <>
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                  </svg>
                  Pause
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z" />
                  </svg>
                  \u00c9couter
                </>
              )}
            </button>
          )}
        </div>
        <p className="text-gray-300 text-sm leading-relaxed">
          {slide.voiceoverText || <span className="text-gray-500 italic">Aucun texte de voiceover</span>}
        </p>

        {/* Hidden audio element */}
        {voiceover?.audioUrl && (
          <audio
            ref={audioRef}
            src={voiceover.audioUrl}
            onEnded={() => setIsPlaying(false)}
            className="hidden"
          />
        )}
      </div>
    </div>
  );
}

export default SlidePreview;
