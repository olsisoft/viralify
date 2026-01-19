'use client';

/**
 * Segment Item Component
 * Displays a single segment in the timeline
 */

import React, { useState, useCallback } from 'react';
import {
  VideoSegment,
  formatDuration,
  getSegmentTypeLabel,
  getStatusColor,
} from '../lib/editor-types';

interface SegmentItemProps {
  segment: VideoSegment;
  isSelected: boolean;
  pixelsPerSecond: number;
  onSelect: (segmentId: string) => void;
  onUpdate: (segmentId: string, updates: Partial<VideoSegment>) => void;
  onRemove: (segmentId: string) => void;
  onDragStart: (segmentId: string) => void;
  onDragEnd: () => void;
}

export function SegmentItem({
  segment,
  isSelected,
  pixelsPerSecond,
  onSelect,
  onUpdate,
  onRemove,
  onDragStart,
  onDragEnd,
}: SegmentItemProps) {
  const [isHovered, setIsHovered] = useState(false);
  const [showContextMenu, setShowContextMenu] = useState(false);

  const width = Math.max(segment.duration * pixelsPerSecond, 60);

  const handleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onSelect(segment.id);
  }, [segment.id, onSelect]);

  const handleDragStart = useCallback((e: React.DragEvent) => {
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', segment.id);
    onDragStart(segment.id);
  }, [segment.id, onDragStart]);

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setShowContextMenu(true);
  }, []);

  const handleMuteToggle = useCallback(() => {
    onUpdate(segment.id, { is_audio_muted: !segment.is_audio_muted });
  }, [segment.id, segment.is_audio_muted, onUpdate]);

  const getSegmentColor = () => {
    const colors: Record<string, string> = {
      generated: 'bg-blue-500',
      user_video: 'bg-purple-500',
      user_audio: 'bg-green-500',
      slide: 'bg-yellow-500',
      transition: 'bg-gray-500',
      overlay: 'bg-pink-500',
    };
    return colors[segment.segment_type] || 'bg-gray-500';
  };

  return (
    <div
      className={`
        relative h-20 rounded-md cursor-pointer overflow-hidden
        ${getSegmentColor()}
        ${isSelected ? 'ring-2 ring-white ring-offset-2 ring-offset-gray-900' : ''}
        ${isHovered ? 'brightness-110' : ''}
        transition-all duration-150
      `}
      style={{ width: `${width}px`, minWidth: '60px' }}
      onClick={handleClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onContextMenu={handleContextMenu}
      draggable
      onDragStart={handleDragStart}
      onDragEnd={onDragEnd}
    >
      {/* Thumbnail */}
      {segment.thumbnail_url && (
        <div
          className="absolute inset-0 bg-cover bg-center opacity-40"
          style={{ backgroundImage: `url(${segment.thumbnail_url})` }}
        />
      )}

      {/* Content */}
      <div className="relative z-10 p-2 h-full flex flex-col justify-between">
        {/* Top row: Title and status */}
        <div className="flex items-start justify-between">
          <span className="text-xs font-medium text-white truncate max-w-[80%]">
            {segment.title || getSegmentTypeLabel(segment.segment_type)}
          </span>
          {segment.status !== 'ready' && (
            <span className={`
              px-1.5 py-0.5 text-[10px] rounded
              bg-${getStatusColor(segment.status)}-500 text-white
            `}>
              {segment.status}
            </span>
          )}
        </div>

        {/* Bottom row: Duration and controls */}
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-white/80">
            {formatDuration(segment.duration)}
          </span>

          <div className="flex items-center gap-1">
            {/* Audio indicator */}
            {(segment.segment_type === 'user_video' || segment.segment_type === 'generated') && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleMuteToggle();
                }}
                className={`
                  p-1 rounded text-white/80 hover:text-white
                  ${segment.is_audio_muted ? 'bg-red-500/50' : 'hover:bg-white/20'}
                `}
                title={segment.is_audio_muted ? 'Unmute' : 'Mute'}
              >
                <svg
                  className="w-3 h-3"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  {segment.is_audio_muted ? (
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"
                    />
                  ) : (
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15.536 8.464a5 5 0 010 7.072M18.364 5.636a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"
                    />
                  )}
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Trim handles (visible when selected) */}
      {isSelected && (
        <>
          {/* Left trim handle */}
          <div className="absolute left-0 top-0 bottom-0 w-2 bg-white/30 hover:bg-white/50 cursor-ew-resize" />
          {/* Right trim handle */}
          <div className="absolute right-0 top-0 bottom-0 w-2 bg-white/30 hover:bg-white/50 cursor-ew-resize" />
        </>
      )}

      {/* Context Menu */}
      {showContextMenu && (
        <div
          className="absolute z-50 top-full left-0 mt-1 bg-gray-800 rounded-md shadow-lg py-1 min-w-[150px]"
          onMouseLeave={() => setShowContextMenu(false)}
        >
          <button
            className="w-full px-3 py-2 text-left text-sm text-white hover:bg-gray-700"
            onClick={(e) => {
              e.stopPropagation();
              setShowContextMenu(false);
              onSelect(segment.id);
            }}
          >
            Edit Properties
          </button>
          <button
            className="w-full px-3 py-2 text-left text-sm text-white hover:bg-gray-700"
            onClick={(e) => {
              e.stopPropagation();
              setShowContextMenu(false);
              handleMuteToggle();
            }}
          >
            {segment.is_audio_muted ? 'Unmute Audio' : 'Mute Audio'}
          </button>
          <hr className="my-1 border-gray-700" />
          <button
            className="w-full px-3 py-2 text-left text-sm text-red-400 hover:bg-gray-700"
            onClick={(e) => {
              e.stopPropagation();
              setShowContextMenu(false);
              onRemove(segment.id);
            }}
          >
            Remove Segment
          </button>
        </div>
      )}
    </div>
  );
}
