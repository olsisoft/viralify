'use client';

import React, { useState, useCallback, useRef } from 'react';
import type { SlideComponent, MediaType } from '../../lib/lecture-editor-types';
import { getSlideTypeIcon, getSlideTypeLabel, formatDuration, getStatusColor, QUICK_ACTIONS } from '../../lib/lecture-editor-types';

interface SlideTimelineProps {
  slides: SlideComponent[];
  selectedSlide: SlideComponent | null;
  onSelectSlide: (slide: SlideComponent) => void;
  onReorderSlide?: (slideId: string, newIndex: number) => void;
  onDeleteSlide?: (slideId: string) => void;
  onInsertMedia?: (type: MediaType, afterSlideId?: string) => void;
  onRegenerateSlide?: (slideId: string) => void;
  isReadOnly?: boolean;
}

export function SlideTimeline({
  slides,
  selectedSlide,
  onSelectSlide,
  onReorderSlide,
  onDeleteSlide,
  onInsertMedia,
  onRegenerateSlide,
  isReadOnly = false,
}: SlideTimelineProps) {
  const [draggedSlide, setDraggedSlide] = useState<string | null>(null);
  const [dropTargetIndex, setDropTargetIndex] = useState<number | null>(null);
  const [showQuickActions, setShowQuickActions] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [pendingMediaType, setPendingMediaType] = useState<MediaType | null>(null);
  const [pendingInsertAfter, setPendingInsertAfter] = useState<string | undefined>(undefined);

  // Drag handlers
  const handleDragStart = useCallback((e: React.DragEvent, slideId: string) => {
    if (isReadOnly) return;
    setDraggedSlide(slideId);
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', slideId);
  }, [isReadOnly]);

  const handleDragOver = useCallback((e: React.DragEvent, index: number) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    setDropTargetIndex(index);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDropTargetIndex(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent, targetIndex: number) => {
    e.preventDefault();
    if (draggedSlide && onReorderSlide) {
      onReorderSlide(draggedSlide, targetIndex);
    }
    setDraggedSlide(null);
    setDropTargetIndex(null);
  }, [draggedSlide, onReorderSlide]);

  const handleDragEnd = useCallback(() => {
    setDraggedSlide(null);
    setDropTargetIndex(null);
  }, []);

  // Quick action handlers
  const handleQuickAction = useCallback((slideId: string, actionType: MediaType | 'regenerate') => {
    if (actionType === 'regenerate') {
      onRegenerateSlide?.(slideId);
    } else {
      setPendingMediaType(actionType);
      setPendingInsertAfter(slideId);
      fileInputRef.current?.click();
    }
    setShowQuickActions(null);
  }, [onRegenerateSlide]);

  // File input handler
  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && pendingMediaType) {
      onInsertMedia?.(pendingMediaType, pendingInsertAfter);
    }
    setPendingMediaType(null);
    setPendingInsertAfter(undefined);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [pendingMediaType, pendingInsertAfter, onInsertMedia]);

  // Get file accept string
  const getAcceptString = () => {
    switch (pendingMediaType) {
      case 'image': return 'image/jpeg,image/png,image/gif,image/webp';
      case 'video': return 'video/mp4,video/webm,video/mov';
      case 'audio': return 'audio/mp3,audio/wav,audio/m4a,audio/ogg';
      default: return '*/*';
    }
  };

  return (
    <div className="p-4 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-gray-400 text-sm font-medium">Timeline</h3>
        <span className="text-gray-500 text-xs">{slides.length} slides</span>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept={getAcceptString()}
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Slides list */}
      <div className="flex-1 overflow-y-auto space-y-1">
        {slides.map((slide, index) => {
          const isSelected = selectedSlide?.id === slide.id;
          const isDragging = draggedSlide === slide.id;
          const isDropTarget = dropTargetIndex === index;
          const showActions = showQuickActions === slide.id;

          return (
            <div key={slide.id} className="relative">
              {/* Drop indicator */}
              {isDropTarget && draggedSlide !== slide.id && (
                <div className="absolute -top-0.5 left-0 right-0 h-1 bg-purple-500 rounded-full z-10" />
              )}

              <div
                draggable={!isReadOnly}
                onDragStart={(e) => handleDragStart(e, slide.id)}
                onDragOver={(e) => handleDragOver(e, index)}
                onDragLeave={handleDragLeave}
                onDrop={(e) => handleDrop(e, index)}
                onDragEnd={handleDragEnd}
                onClick={() => onSelectSlide(slide)}
                onMouseEnter={() => !isReadOnly && setShowQuickActions(slide.id)}
                onMouseLeave={() => setShowQuickActions(null)}
                className={`
                  relative w-full text-left p-2 rounded-lg transition-all cursor-pointer
                  ${isSelected ? 'bg-purple-600/20 border-2 border-purple-500' : 'bg-gray-800 hover:bg-gray-750 border-2 border-transparent'}
                  ${isDragging ? 'opacity-50 scale-95' : ''}
                  ${!isReadOnly ? 'cursor-grab active:cursor-grabbing' : ''}
                `}
              >
                <div className="flex items-center gap-2">
                  {/* Drag handle */}
                  {!isReadOnly && (
                    <div className="text-gray-600 hover:text-gray-400 flex-shrink-0">
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M8 6a2 2 0 1 1-4 0 2 2 0 0 1 4 0zm0 6a2 2 0 1 1-4 0 2 2 0 0 1 4 0zm0 6a2 2 0 1 1-4 0 2 2 0 0 1 4 0zm8-12a2 2 0 1 1-4 0 2 2 0 0 1 4 0zm0 6a2 2 0 1 1-4 0 2 2 0 0 1 4 0zm0 6a2 2 0 1 1-4 0 2 2 0 0 1 4 0z" />
                      </svg>
                    </div>
                  )}

                  {/* Thumbnail */}
                  <div className="w-14 h-9 bg-gray-700 rounded flex items-center justify-center text-sm flex-shrink-0 overflow-hidden">
                    {slide.imageUrl ? (
                      <img
                        src={slide.imageUrl}
                        alt={`Slide ${index + 1}`}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <span>{getSlideTypeIcon(slide.type)}</span>
                    )}
                  </div>

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1">
                      <span className="text-gray-500 text-xs font-mono">{index + 1}</span>
                      <span className="text-white text-xs font-medium truncate">
                        {slide.title || getSlideTypeLabel(slide.type)}
                      </span>
                    </div>
                    <div className="flex items-center gap-1 mt-0.5">
                      <span className="text-gray-500 text-xs">{formatDuration(slide.duration)}</span>
                      {slide.isEdited && (
                        <span className="w-1.5 h-1.5 bg-yellow-500 rounded-full" title="Modifié" />
                      )}
                      {slide.status === 'failed' && (
                        <span className="w-1.5 h-1.5 bg-red-500 rounded-full" title="Erreur" />
                      )}
                    </div>
                  </div>

                  {/* Delete button (visible on hover) */}
                  {!isReadOnly && showActions && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteSlide?.(slide.id);
                      }}
                      className="p-1 text-gray-500 hover:text-red-500 transition-colors flex-shrink-0"
                      title="Supprimer"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  )}
                </div>

                {/* Quick actions bar (visible on hover) */}
                {!isReadOnly && showActions && (
                  <div className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 translate-y-full z-20">
                    <div className="flex items-center gap-1 bg-gray-900 border border-gray-700 rounded-lg p-1 shadow-lg">
                      {QUICK_ACTIONS.map((action) => (
                        <button
                          key={action.id}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleQuickAction(slide.id, action.type);
                          }}
                          className="p-1.5 hover:bg-gray-700 rounded text-sm transition-colors"
                          title={action.tooltip}
                        >
                          {action.icon}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {/* Drop zone at the end */}
        {draggedSlide && (
          <div
            onDragOver={(e) => handleDragOver(e, slides.length)}
            onDragLeave={handleDragLeave}
            onDrop={(e) => handleDrop(e, slides.length)}
            className={`h-12 border-2 border-dashed rounded-lg flex items-center justify-center transition-colors
              ${dropTargetIndex === slides.length ? 'border-purple-500 bg-purple-500/10' : 'border-gray-700'}
            `}
          >
            <span className="text-gray-500 text-xs">Déposer ici</span>
          </div>
        )}
      </div>

      {/* Footer with stats */}
      <div className="mt-4 pt-3 border-t border-gray-800 space-y-1">
        <div className="flex justify-between text-xs">
          <span className="text-gray-500">Durée totale</span>
          <span className="text-white font-medium">
            {formatDuration(slides.reduce((acc, s) => acc + s.duration, 0))}
          </span>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-gray-500">Modifiés</span>
          <span className="text-yellow-500">
            {slides.filter(s => s.isEdited).length}/{slides.length}
          </span>
        </div>
      </div>

      {/* Add media button */}
      {!isReadOnly && (
        <button
          onClick={() => {
            setPendingMediaType('image');
            setPendingInsertAfter(slides[slides.length - 1]?.id);
            fileInputRef.current?.click();
          }}
          className="mt-3 w-full py-2 border-2 border-dashed border-gray-700 rounded-lg text-gray-500 hover:border-purple-500 hover:text-purple-400 transition-colors text-sm flex items-center justify-center gap-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          Ajouter un média
        </button>
      )}
    </div>
  );
}

export default SlideTimeline;
