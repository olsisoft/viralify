'use client';

/**
 * Segment Properties Panel
 * Panel for editing selected segment properties
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  VideoSegment,
  UpdateSegmentRequest,
  TransitionType,
  formatDuration,
  getSegmentTypeLabel,
} from '../lib/editor-types';

interface SegmentPropertiesProps {
  segment: VideoSegment;
  onUpdate: (request: UpdateSegmentRequest) => void;
  onRemove: () => void;
  onSplit: (splitTime: number) => void;
}

export function SegmentProperties({
  segment,
  onUpdate,
  onRemove,
  onSplit,
}: SegmentPropertiesProps) {
  const [localValues, setLocalValues] = useState({
    trim_start: segment.trim_start,
    trim_end: segment.trim_end,
    original_audio_volume: segment.original_audio_volume,
    opacity: segment.opacity,
    transition_in: segment.transition_in,
    transition_in_duration: segment.transition_in_duration,
    transition_out: segment.transition_out,
    transition_out_duration: segment.transition_out_duration,
  });

  const [splitTime, setSplitTime] = useState(segment.duration / 2);

  // Update local values when segment changes
  useEffect(() => {
    setLocalValues({
      trim_start: segment.trim_start,
      trim_end: segment.trim_end,
      original_audio_volume: segment.original_audio_volume,
      opacity: segment.opacity,
      transition_in: segment.transition_in,
      transition_in_duration: segment.transition_in_duration,
      transition_out: segment.transition_out,
      transition_out_duration: segment.transition_out_duration,
    });
    setSplitTime(segment.duration / 2);
  }, [segment]);

  const handleChange = useCallback((field: keyof UpdateSegmentRequest, value: any) => {
    setLocalValues((prev) => ({ ...prev, [field]: value }));
  }, []);

  const handleApply = useCallback((field: keyof UpdateSegmentRequest) => {
    const value = localValues[field as keyof typeof localValues];
    if (value !== segment[field as keyof VideoSegment]) {
      onUpdate({ [field]: value });
    }
  }, [localValues, segment, onUpdate]);

  const handleSplit = useCallback(() => {
    if (splitTime > 0 && splitTime < segment.duration) {
      onSplit(splitTime);
    }
  }, [splitTime, segment.duration, onSplit]);

  const transitionOptions: { value: TransitionType; label: string }[] = [
    { value: 'none', label: 'None' },
    { value: 'fade', label: 'Fade' },
    { value: 'dissolve', label: 'Dissolve' },
    { value: 'wipe_left', label: 'Wipe Left' },
    { value: 'wipe_right', label: 'Wipe Right' },
    { value: 'zoom_in', label: 'Zoom In' },
    { value: 'zoom_out', label: 'Zoom Out' },
  ];

  return (
    <div className="h-full flex flex-col bg-gray-800 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-700">
        <h3 className="text-sm font-medium text-white">Segment Properties</h3>
        <p className="text-xs text-gray-400 mt-1">
          {getSegmentTypeLabel(segment.segment_type)} - {segment.title || 'Untitled'}
        </p>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Info */}
        <div className="space-y-2">
          <h4 className="text-xs font-medium text-gray-400 uppercase">Info</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-gray-400">Duration:</span>
              <span className="text-white ml-2">{formatDuration(segment.duration)}</span>
            </div>
            <div>
              <span className="text-gray-400">Start:</span>
              <span className="text-white ml-2">{formatDuration(segment.start_time)}</span>
            </div>
          </div>
        </div>

        {/* Trim */}
        <div className="space-y-3">
          <h4 className="text-xs font-medium text-gray-400 uppercase">Trim</h4>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Trim Start (seconds)</label>
            <input
              type="number"
              min={0}
              step={0.1}
              value={localValues.trim_start}
              onChange={(e) => handleChange('trim_start', parseFloat(e.target.value) || 0)}
              onBlur={() => handleApply('trim_start')}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
            />
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Trim End (seconds)</label>
            <input
              type="number"
              min={0}
              step={0.1}
              value={localValues.trim_end || ''}
              placeholder="End of source"
              onChange={(e) => handleChange('trim_end', e.target.value ? parseFloat(e.target.value) : null)}
              onBlur={() => handleApply('trim_end')}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
            />
          </div>
        </div>

        {/* Audio */}
        {(segment.segment_type === 'user_video' || segment.segment_type === 'generated') && (
          <div className="space-y-3">
            <h4 className="text-xs font-medium text-gray-400 uppercase">Audio</h4>

            <div className="flex items-center justify-between">
              <label className="text-xs text-gray-400">Mute Audio</label>
              <button
                onClick={() => onUpdate({ is_audio_muted: !segment.is_audio_muted })}
                className={`
                  relative w-10 h-5 rounded-full transition-colors
                  ${segment.is_audio_muted ? 'bg-red-500' : 'bg-gray-600'}
                `}
              >
                <div
                  className={`
                    absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform
                    ${segment.is_audio_muted ? 'translate-x-5' : 'translate-x-0.5'}
                  `}
                />
              </button>
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Volume: {Math.round(localValues.original_audio_volume * 100)}%
              </label>
              <input
                type="range"
                min={0}
                max={2}
                step={0.1}
                value={localValues.original_audio_volume}
                onChange={(e) => handleChange('original_audio_volume', parseFloat(e.target.value))}
                onMouseUp={() => handleApply('original_audio_volume')}
                className="w-full"
                disabled={segment.is_audio_muted}
              />
            </div>
          </div>
        )}

        {/* Visual */}
        <div className="space-y-3">
          <h4 className="text-xs font-medium text-gray-400 uppercase">Visual</h4>

          <div>
            <label className="block text-xs text-gray-400 mb-1">
              Opacity: {Math.round(localValues.opacity * 100)}%
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={localValues.opacity}
              onChange={(e) => handleChange('opacity', parseFloat(e.target.value))}
              onMouseUp={() => handleApply('opacity')}
              className="w-full"
            />
          </div>
        </div>

        {/* Transitions */}
        <div className="space-y-3">
          <h4 className="text-xs font-medium text-gray-400 uppercase">Transitions</h4>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Transition In</label>
            <select
              value={localValues.transition_in}
              onChange={(e) => {
                handleChange('transition_in', e.target.value as TransitionType);
                onUpdate({ transition_in: e.target.value as TransitionType });
              }}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
            >
              {transitionOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          {localValues.transition_in !== 'none' && (
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                In Duration: {localValues.transition_in_duration}s
              </label>
              <input
                type="range"
                min={0.1}
                max={3}
                step={0.1}
                value={localValues.transition_in_duration}
                onChange={(e) => handleChange('transition_in_duration', parseFloat(e.target.value))}
                onMouseUp={() => handleApply('transition_in_duration')}
                className="w-full"
              />
            </div>
          )}

          <div>
            <label className="block text-xs text-gray-400 mb-1">Transition Out</label>
            <select
              value={localValues.transition_out}
              onChange={(e) => {
                handleChange('transition_out', e.target.value as TransitionType);
                onUpdate({ transition_out: e.target.value as TransitionType });
              }}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
            >
              {transitionOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          {localValues.transition_out !== 'none' && (
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Out Duration: {localValues.transition_out_duration}s
              </label>
              <input
                type="range"
                min={0.1}
                max={3}
                step={0.1}
                value={localValues.transition_out_duration}
                onChange={(e) => handleChange('transition_out_duration', parseFloat(e.target.value))}
                onMouseUp={() => handleApply('transition_out_duration')}
                className="w-full"
              />
            </div>
          )}
        </div>

        {/* Split */}
        <div className="space-y-3">
          <h4 className="text-xs font-medium text-gray-400 uppercase">Split Segment</h4>

          <div>
            <label className="block text-xs text-gray-400 mb-1">
              Split at: {formatDuration(splitTime)}
            </label>
            <input
              type="range"
              min={0.5}
              max={segment.duration - 0.5}
              step={0.1}
              value={splitTime}
              onChange={(e) => setSplitTime(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          <button
            onClick={handleSplit}
            disabled={segment.duration < 2}
            className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded text-sm font-medium"
          >
            Split at {formatDuration(splitTime)}
          </button>
        </div>
      </div>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-gray-700">
        <button
          onClick={onRemove}
          className="w-full px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded text-sm font-medium"
        >
          Remove Segment
        </button>
      </div>
    </div>
  );
}
