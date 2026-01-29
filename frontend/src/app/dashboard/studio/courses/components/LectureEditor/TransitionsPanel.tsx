'use client';

import React, { useState, useCallback } from 'react';
import type {
  SlideComponent,
  SlideTransition,
  Transition,
  TransitionType,
} from '../../lib/lecture-editor-types';
import { TRANSITION_PRESETS } from '../../lib/lecture-editor-types';

interface TransitionsPanelProps {
  slides: SlideComponent[];
  transitions: SlideTransition[];
  selectedSlideId: string | null;
  onTransitionsChange: (transitions: SlideTransition[]) => void;
}

const TRANSITION_TYPES = Object.keys(TRANSITION_PRESETS) as TransitionType[];

export function TransitionsPanel({
  slides,
  transitions,
  selectedSlideId,
  onTransitionsChange,
}: TransitionsPanelProps) {
  const [applyToAll, setApplyToAll] = useState(false);

  // Get transition for a slide
  const getTransition = (slideId: string): SlideTransition => {
    return transitions.find(t => t.slideId === slideId) || {
      slideId,
      inTransition: { id: `trans-${slideId}-in`, type: 'fade', duration: 0.5, easing: 'ease-in-out' },
    };
  };

  // Update transition for a slide
  const updateTransition = useCallback((slideId: string, updates: Partial<SlideTransition>) => {
    if (applyToAll) {
      // Apply to all slides
      onTransitionsChange(slides.map(slide => ({
        ...getTransition(slide.id),
        ...updates,
        slideId: slide.id,
      })));
    } else {
      // Apply to single slide
      const existing = transitions.find(t => t.slideId === slideId);
      if (existing) {
        onTransitionsChange(transitions.map(t =>
          t.slideId === slideId ? { ...t, ...updates } : t
        ));
      } else {
        onTransitionsChange([...transitions, { slideId, ...updates }]);
      }
    }
  }, [slides, transitions, applyToAll, onTransitionsChange]);

  // Update in-transition type
  const setTransitionType = useCallback((slideId: string, type: TransitionType) => {
    const current = getTransition(slideId);
    updateTransition(slideId, {
      inTransition: {
        ...current.inTransition!,
        type,
      },
    });
  }, [updateTransition]);

  // Update transition duration
  const setTransitionDuration = useCallback((slideId: string, duration: number) => {
    const current = getTransition(slideId);
    updateTransition(slideId, {
      inTransition: {
        ...current.inTransition!,
        duration,
      },
    });
  }, [updateTransition]);

  // Update transition easing
  const setTransitionEasing = useCallback((slideId: string, easing: Transition['easing']) => {
    const current = getTransition(slideId);
    updateTransition(slideId, {
      inTransition: {
        ...current.inTransition!,
        easing,
      },
    });
  }, [updateTransition]);

  const selectedSlide = slides.find(s => s.id === selectedSlideId);
  const selectedTransition = selectedSlideId ? getTransition(selectedSlideId) : null;

  return (
    <div className="flex flex-col h-full bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <h3 className="text-white font-semibold text-sm flex items-center gap-2">
          <span>✨</span>
          Transitions
        </h3>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={applyToAll}
            onChange={(e) => setApplyToAll(e.target.checked)}
            className="rounded border-gray-600 bg-gray-800 text-purple-600 focus:ring-purple-500"
          />
          <span className="text-gray-400 text-xs">Appliquer à tous</span>
        </label>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {!selectedSlide ? (
          <div className="flex items-center justify-center h-full text-gray-500 text-sm">
            Sélectionnez un slide pour modifier sa transition
          </div>
        ) : (
          <div className="space-y-6">
            {/* Current slide info */}
            <div className="bg-gray-800 rounded-lg p-3">
              <p className="text-white text-sm font-medium">
                Slide {selectedSlide.index + 1}: {selectedSlide.title || 'Sans titre'}
              </p>
              <p className="text-gray-400 text-xs mt-1">
                Transition d'entrée pour ce slide
              </p>
            </div>

            {/* Transition type grid */}
            <div>
              <label className="text-gray-400 text-xs font-medium block mb-3">Type de transition</label>
              <div className="grid grid-cols-3 gap-2">
                {TRANSITION_TYPES.map((type) => {
                  const preset = TRANSITION_PRESETS[type];
                  const isSelected = selectedTransition?.inTransition?.type === type;

                  return (
                    <button
                      key={type}
                      onClick={() => setTransitionType(selectedSlideId!, type)}
                      className={`p-3 rounded-lg text-center transition-colors ${
                        isSelected
                          ? 'bg-purple-600 text-white'
                          : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                      }`}
                    >
                      <span className="text-xl block mb-1">{preset.icon}</span>
                      <span className="text-xs block">{preset.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Duration */}
            {selectedTransition?.inTransition?.type !== 'none' && (
              <div>
                <label className="text-gray-400 text-xs font-medium block mb-2">
                  Durée: {selectedTransition?.inTransition?.duration || 0.5}s
                </label>
                <input
                  type="range"
                  min={0.1}
                  max={3}
                  step={0.1}
                  value={selectedTransition?.inTransition?.duration || 0.5}
                  onChange={(e) => setTransitionDuration(selectedSlideId!, parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>0.1s</span>
                  <span>3s</span>
                </div>
              </div>
            )}

            {/* Easing */}
            {selectedTransition?.inTransition?.type !== 'none' && (
              <div>
                <label className="text-gray-400 text-xs font-medium block mb-2">Accélération</label>
                <div className="grid grid-cols-2 gap-2">
                  {(['linear', 'ease-in', 'ease-out', 'ease-in-out'] as const).map((easing) => (
                    <button
                      key={easing}
                      onClick={() => setTransitionEasing(selectedSlideId!, easing)}
                      className={`py-2 px-3 rounded-lg text-sm transition-colors ${
                        selectedTransition?.inTransition?.easing === easing
                          ? 'bg-purple-600 text-white'
                          : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                      }`}
                    >
                      {easing === 'linear' ? 'Linéaire' :
                       easing === 'ease-in' ? 'Accélération' :
                       easing === 'ease-out' ? 'Décélération' :
                       'Les deux'}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Preview */}
            <div>
              <label className="text-gray-400 text-xs font-medium block mb-2">Aperçu</label>
              <div className="bg-gray-800 rounded-lg p-4 flex items-center justify-center h-32 relative overflow-hidden">
                <TransitionPreview
                  type={selectedTransition?.inTransition?.type || 'fade'}
                  duration={selectedTransition?.inTransition?.duration || 0.5}
                  easing={selectedTransition?.inTransition?.easing || 'ease-in-out'}
                />
              </div>
            </div>

            {/* Quick presets */}
            <div>
              <label className="text-gray-400 text-xs font-medium block mb-2">Presets rapides</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => {
                    setTransitionType(selectedSlideId!, 'fade');
                    setTransitionDuration(selectedSlideId!, 0.5);
                  }}
                  className="py-2 px-3 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 text-sm"
                >
                  Fondu court
                </button>
                <button
                  onClick={() => {
                    setTransitionType(selectedSlideId!, 'dissolve');
                    setTransitionDuration(selectedSlideId!, 1);
                  }}
                  className="py-2 px-3 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 text-sm"
                >
                  Dissolution
                </button>
                <button
                  onClick={() => {
                    setTransitionType(selectedSlideId!, 'slide-left');
                    setTransitionDuration(selectedSlideId!, 0.3);
                  }}
                  className="py-2 px-3 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 text-sm"
                >
                  Glissement rapide
                </button>
                <button
                  onClick={() => {
                    setTransitionType(selectedSlideId!, 'zoom-in');
                    setTransitionDuration(selectedSlideId!, 0.8);
                  }}
                  className="py-2 px-3 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 text-sm"
                >
                  Zoom dramatique
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* All slides transitions overview */}
      <div className="border-t border-gray-800 p-4">
        <label className="text-gray-400 text-xs font-medium block mb-2">Tous les slides</label>
        <div className="flex gap-1 overflow-x-auto pb-2">
          {slides.map((slide, index) => {
            const trans = getTransition(slide.id);
            const preset = TRANSITION_PRESETS[trans.inTransition?.type || 'none'];
            const isSelected = slide.id === selectedSlideId;

            return (
              <div
                key={slide.id}
                className={`flex-shrink-0 w-12 h-12 rounded-lg flex flex-col items-center justify-center cursor-pointer transition-colors ${
                  isSelected ? 'bg-purple-600' : 'bg-gray-800 hover:bg-gray-700'
                }`}
                title={`${slide.title || `Slide ${index + 1}`}: ${preset.label}`}
              >
                <span className="text-sm">{preset.icon}</span>
                <span className="text-xs text-gray-400">{index + 1}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// Transition preview component
interface TransitionPreviewProps {
  type: TransitionType;
  duration: number;
  easing: string;
}

function TransitionPreview({ type, duration, easing }: TransitionPreviewProps) {
  const [isAnimating, setIsAnimating] = useState(false);

  const playPreview = () => {
    setIsAnimating(true);
    setTimeout(() => setIsAnimating(false), duration * 1000 + 500);
  };

  // CSS animation based on type
  const getAnimationStyle = (): React.CSSProperties => {
    if (!isAnimating) return { opacity: 1 };

    const animations: Record<string, React.CSSProperties> = {
      'none': {},
      'fade': { animation: `fadeIn ${duration}s ${easing}` },
      'dissolve': { animation: `fadeIn ${duration}s ${easing}` },
      'slide-left': { animation: `slideInLeft ${duration}s ${easing}` },
      'slide-right': { animation: `slideInRight ${duration}s ${easing}` },
      'slide-up': { animation: `slideInUp ${duration}s ${easing}` },
      'slide-down': { animation: `slideInDown ${duration}s ${easing}` },
      'zoom-in': { animation: `zoomIn ${duration}s ${easing}` },
      'zoom-out': { animation: `zoomOut ${duration}s ${easing}` },
      'wipe-left': { animation: `wipeLeft ${duration}s ${easing}` },
      'wipe-right': { animation: `wipeRight ${duration}s ${easing}` },
      'wipe-up': { animation: `wipeUp ${duration}s ${easing}` },
      'wipe-down': { animation: `wipeDown ${duration}s ${easing}` },
      'blur': { animation: `blurIn ${duration}s ${easing}` },
      'flash': { animation: `flash ${duration}s ${easing}` },
    };

    return animations[type] || {};
  };

  return (
    <div className="relative w-full h-full flex items-center justify-center">
      <style>{`
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes slideInLeft { from { transform: translateX(-100%); } to { transform: translateX(0); } }
        @keyframes slideInRight { from { transform: translateX(100%); } to { transform: translateX(0); } }
        @keyframes slideInUp { from { transform: translateY(100%); } to { transform: translateY(0); } }
        @keyframes slideInDown { from { transform: translateY(-100%); } to { transform: translateY(0); } }
        @keyframes zoomIn { from { transform: scale(0); opacity: 0; } to { transform: scale(1); opacity: 1; } }
        @keyframes zoomOut { from { transform: scale(2); opacity: 0; } to { transform: scale(1); opacity: 1; } }
        @keyframes wipeLeft { from { clip-path: inset(0 100% 0 0); } to { clip-path: inset(0 0 0 0); } }
        @keyframes wipeRight { from { clip-path: inset(0 0 0 100%); } to { clip-path: inset(0 0 0 0); } }
        @keyframes wipeUp { from { clip-path: inset(100% 0 0 0); } to { clip-path: inset(0 0 0 0); } }
        @keyframes wipeDown { from { clip-path: inset(0 0 100% 0); } to { clip-path: inset(0 0 0 0); } }
        @keyframes blurIn { from { filter: blur(20px); opacity: 0; } to { filter: blur(0); opacity: 1; } }
        @keyframes flash { 0% { opacity: 0; } 50% { opacity: 1; background: white; } 100% { opacity: 1; background: transparent; } }
      `}</style>

      <div
        className="w-20 h-14 bg-purple-600 rounded flex items-center justify-center"
        style={getAnimationStyle()}
      >
        <span className="text-white text-xs">Slide</span>
      </div>

      <button
        onClick={playPreview}
        disabled={isAnimating}
        className="absolute bottom-2 right-2 p-1.5 bg-gray-700 rounded-full hover:bg-gray-600 disabled:opacity-50 transition-colors"
        title="Jouer l'aperçu"
      >
        <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24">
          <path d="M8 5v14l11-7z" />
        </svg>
      </button>
    </div>
  );
}

export default TransitionsPanel;
