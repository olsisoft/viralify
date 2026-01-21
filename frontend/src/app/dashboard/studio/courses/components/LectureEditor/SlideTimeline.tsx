'use client';

import React from 'react';
import type { SlideComponent } from '../../lib/lecture-editor-types';
import { getSlideTypeIcon, getSlideTypeLabel, formatDuration, getStatusColor } from '../../lib/lecture-editor-types';

interface SlideTimelineProps {
  slides: SlideComponent[];
  selectedSlide: SlideComponent | null;
  onSelectSlide: (slide: SlideComponent) => void;
}

export function SlideTimeline({ slides, selectedSlide, onSelectSlide }: SlideTimelineProps) {
  return (
    <div className="p-4">
      <h3 className="text-gray-400 text-sm font-medium mb-4">Timeline des slides</h3>

      <div className="space-y-2">
        {slides.map((slide, index) => (
          <button
            key={slide.id}
            onClick={() => onSelectSlide(slide)}
            className={`w-full text-left p-3 rounded-lg transition-colors ${
              selectedSlide?.id === slide.id
                ? 'bg-purple-600/20 border border-purple-500'
                : 'bg-gray-800 hover:bg-gray-700 border border-transparent'
            }`}
          >
            <div className="flex items-start gap-3">
              {/* Thumbnail placeholder */}
              <div className="w-16 h-10 bg-gray-700 rounded flex items-center justify-center text-xl flex-shrink-0">
                {slide.imageUrl ? (
                  <img
                    src={slide.imageUrl}
                    alt={`Slide ${index + 1}`}
                    className="w-full h-full object-cover rounded"
                  />
                ) : (
                  getSlideTypeIcon(slide.type)
                )}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-gray-500 text-xs">{index + 1}</span>
                  <span className="text-white text-sm font-medium truncate">
                    {slide.title || getSlideTypeLabel(slide.type)}
                  </span>
                </div>

                <div className="flex items-center gap-2 mt-1">
                  <span className="text-gray-500 text-xs">
                    {formatDuration(slide.duration)}
                  </span>
                  <span className="text-gray-600">|</span>
                  <span className={`text-xs ${getStatusColor(slide.status)}`}>
                    {slide.isEdited ? 'Modifi\u00e9' : getSlideTypeLabel(slide.type)}
                  </span>
                </div>
              </div>

              {/* Status indicator */}
              {slide.isEdited && (
                <div className="w-2 h-2 bg-yellow-500 rounded-full flex-shrink-0" />
              )}
              {slide.status === 'failed' && (
                <div className="w-2 h-2 bg-red-500 rounded-full flex-shrink-0" />
              )}
            </div>
          </button>
        ))}
      </div>

      {/* Duration summary */}
      <div className="mt-4 pt-4 border-t border-gray-800">
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Dur\u00e9e totale</span>
          <span className="text-white">
            {formatDuration(slides.reduce((acc, s) => acc + s.duration, 0))}
          </span>
        </div>
        <div className="flex justify-between text-sm mt-1">
          <span className="text-gray-400">Slides modifi\u00e9s</span>
          <span className="text-yellow-500">
            {slides.filter(s => s.isEdited).length} / {slides.length}
          </span>
        </div>
      </div>
    </div>
  );
}

export default SlideTimeline;
