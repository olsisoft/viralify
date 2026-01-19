'use client';

/**
 * Timeline Component
 * Main timeline view for the video editor
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { VideoSegment, VideoProject, formatDuration, formatDetailedDuration } from '../lib/editor-types';
import { SegmentItem } from './SegmentItem';

interface TimelineProps {
  project: VideoProject;
  selectedSegmentId: string | null;
  playheadPosition: number;
  onSelectSegment: (segmentId: string | null) => void;
  onUpdateSegment: (segmentId: string, updates: Partial<VideoSegment>) => void;
  onRemoveSegment: (segmentId: string) => void;
  onReorderSegments: (segmentIds: string[]) => void;
  onPlayheadChange: (position: number) => void;
  onUploadFile: (file: File, insertAfterSegmentId?: string) => void;
}

export function Timeline({
  project,
  selectedSegmentId,
  playheadPosition,
  onSelectSegment,
  onUpdateSegment,
  onRemoveSegment,
  onReorderSegments,
  onPlayheadChange,
  onUploadFile,
}: TimelineProps) {
  const [zoom, setZoom] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const [draggedSegmentId, setDraggedSegmentId] = useState<string | null>(null);
  const [dropTargetIndex, setDropTargetIndex] = useState<number | null>(null);

  const timelineRef = useRef<HTMLDivElement>(null);
  const rulerRef = useRef<HTMLDivElement>(null);

  // Pixels per second - adjustable with zoom
  const basePixelsPerSecond = 20;
  const pixelsPerSecond = basePixelsPerSecond * zoom;

  // Total timeline width
  const timelineWidth = Math.max(project.total_duration * pixelsPerSecond, 800);

  // Generate time markers
  const getTimeMarkers = useCallback(() => {
    const markers: { position: number; label: string }[] = [];
    const duration = project.total_duration || 60;
    const markerInterval = zoom >= 2 ? 5 : zoom >= 1 ? 10 : 30;

    for (let t = 0; t <= duration; t += markerInterval) {
      markers.push({
        position: t * pixelsPerSecond,
        label: formatDetailedDuration(t),
      });
    }

    return markers;
  }, [project.total_duration, pixelsPerSecond, zoom]);

  // Handle ruler click to move playhead
  const handleRulerClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const time = x / pixelsPerSecond;
    onPlayheadChange(Math.max(0, Math.min(time, project.total_duration)));
  }, [pixelsPerSecond, project.total_duration, onPlayheadChange]);

  // Handle drag over
  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>, index: number) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    setDropTargetIndex(index);
  }, []);

  // Handle drop
  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>, dropIndex: number) => {
    e.preventDefault();

    // Check if it's a file drop
    if (e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      const insertAfterSegmentId = dropIndex > 0 ? project.segments[dropIndex - 1]?.id : undefined;
      onUploadFile(file, insertAfterSegmentId);
      setDropTargetIndex(null);
      return;
    }

    // Handle segment reorder
    const segmentId = e.dataTransfer.getData('text/plain');
    if (!segmentId || !draggedSegmentId) {
      setDropTargetIndex(null);
      return;
    }

    const currentIndex = project.segments.findIndex((s) => s.id === segmentId);
    if (currentIndex === -1 || currentIndex === dropIndex) {
      setDropTargetIndex(null);
      return;
    }

    // Create new order
    const newSegments = [...project.segments];
    const [movedSegment] = newSegments.splice(currentIndex, 1);
    const newDropIndex = dropIndex > currentIndex ? dropIndex - 1 : dropIndex;
    newSegments.splice(newDropIndex, 0, movedSegment);

    onReorderSegments(newSegments.map((s) => s.id));
    setDropTargetIndex(null);
    setDraggedSegmentId(null);
  }, [project.segments, draggedSegmentId, onReorderSegments, onUploadFile]);

  // Handle drag start
  const handleDragStart = useCallback((segmentId: string) => {
    setIsDragging(true);
    setDraggedSegmentId(segmentId);
  }, []);

  // Handle drag end
  const handleDragEnd = useCallback(() => {
    setIsDragging(false);
    setDraggedSegmentId(null);
    setDropTargetIndex(null);
  }, []);

  // Zoom controls
  const handleZoomIn = useCallback(() => {
    setZoom((z) => Math.min(z * 1.5, 4));
  }, []);

  const handleZoomOut = useCallback(() => {
    setZoom((z) => Math.max(z / 1.5, 0.25));
  }, []);

  // Scroll playhead into view when it changes
  useEffect(() => {
    if (timelineRef.current) {
      const playheadX = playheadPosition * pixelsPerSecond;
      const container = timelineRef.current;
      const containerWidth = container.clientWidth;
      const scrollLeft = container.scrollLeft;

      if (playheadX < scrollLeft || playheadX > scrollLeft + containerWidth - 100) {
        container.scrollLeft = playheadX - containerWidth / 2;
      }
    }
  }, [playheadPosition, pixelsPerSecond]);

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-lg overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-400">
            Duration: {formatDetailedDuration(project.total_duration)}
          </span>
          <span className="text-sm text-gray-400">
            Segments: {project.segments.length}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Zoom controls */}
          <button
            onClick={handleZoomOut}
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
            title="Zoom Out"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
            </svg>
          </button>
          <span className="text-xs text-gray-400 w-12 text-center">
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={handleZoomIn}
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
            title="Zoom In"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
          </button>
        </div>
      </div>

      {/* Timeline content */}
      <div
        ref={timelineRef}
        className="flex-1 overflow-x-auto overflow-y-hidden relative"
      >
        <div style={{ width: `${timelineWidth}px`, minWidth: '100%' }}>
          {/* Time ruler */}
          <div
            ref={rulerRef}
            className="h-8 bg-gray-800 border-b border-gray-700 relative cursor-pointer"
            onClick={handleRulerClick}
          >
            {getTimeMarkers().map((marker, i) => (
              <div
                key={i}
                className="absolute top-0 bottom-0 flex flex-col items-center"
                style={{ left: `${marker.position}px` }}
              >
                <div className="h-full w-px bg-gray-600" />
                <span className="absolute bottom-1 text-[10px] text-gray-400 transform -translate-x-1/2">
                  {marker.label}
                </span>
              </div>
            ))}

            {/* Playhead */}
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-red-500 z-20"
              style={{ left: `${playheadPosition * pixelsPerSecond}px` }}
            >
              <div className="absolute -top-1 left-1/2 transform -translate-x-1/2 w-3 h-3 bg-red-500 rotate-45" />
            </div>
          </div>

          {/* Segments track */}
          <div
            className="h-24 bg-gray-850 relative flex items-center gap-1 px-2 py-2"
            onDragOver={(e) => e.preventDefault()}
          >
            {/* Drop zone at start */}
            <div
              className={`
                w-1 h-20 rounded transition-all
                ${dropTargetIndex === 0 ? 'bg-blue-500 w-2' : 'bg-transparent'}
              `}
              onDragOver={(e) => handleDragOver(e, 0)}
              onDrop={(e) => handleDrop(e, 0)}
            />

            {project.segments.map((segment, index) => (
              <React.Fragment key={segment.id}>
                <SegmentItem
                  segment={segment}
                  isSelected={segment.id === selectedSegmentId}
                  pixelsPerSecond={pixelsPerSecond}
                  onSelect={onSelectSegment}
                  onUpdate={(id, updates) => onUpdateSegment(id, updates as Partial<VideoSegment>)}
                  onRemove={onRemoveSegment}
                  onDragStart={handleDragStart}
                  onDragEnd={handleDragEnd}
                />

                {/* Drop zone after each segment */}
                <div
                  className={`
                    w-1 h-20 rounded transition-all
                    ${dropTargetIndex === index + 1 ? 'bg-blue-500 w-2' : 'bg-transparent'}
                  `}
                  onDragOver={(e) => handleDragOver(e, index + 1)}
                  onDrop={(e) => handleDrop(e, index + 1)}
                />
              </React.Fragment>
            ))}

            {/* Empty state */}
            {project.segments.length === 0 && (
              <div
                className="flex-1 h-20 border-2 border-dashed border-gray-600 rounded-lg flex items-center justify-center"
                onDragOver={(e) => {
                  e.preventDefault();
                  e.dataTransfer.dropEffect = 'copy';
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  if (e.dataTransfer.files.length > 0) {
                    onUploadFile(e.dataTransfer.files[0]);
                  }
                }}
              >
                <span className="text-gray-500 text-sm">
                  Drag and drop videos here, or import from course
                </span>
              </div>
            )}

            {/* Playhead line */}
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-red-500/50 pointer-events-none z-10"
              style={{ left: `${playheadPosition * pixelsPerSecond + 8}px` }}
            />
          </div>

          {/* Audio tracks (future enhancement) */}
          <div className="h-12 bg-gray-850 border-t border-gray-700 px-2 py-1">
            <div className="h-full flex items-center">
              <span className="text-xs text-gray-500">Audio Track</span>
            </div>
          </div>
        </div>
      </div>

      {/* Info bar */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-t border-gray-700">
        <span className="text-xs text-gray-400">
          Playhead: {formatDetailedDuration(playheadPosition)}
        </span>
        {selectedSegmentId && (
          <span className="text-xs text-gray-400">
            Selected: {project.segments.find((s) => s.id === selectedSegmentId)?.title || 'Untitled'}
          </span>
        )}
      </div>
    </div>
  );
}
