'use client';

import { useEffect, useCallback } from 'react';
import { Shuffle, Clock, FolderOpen, FileVideo } from 'lucide-react';
import type { CourseStructureConfig } from '../lib/course-types';

interface StructureConfigProps {
  value: CourseStructureConfig;
  onChange: (value: CourseStructureConfig) => void;
}

// Calculate optimal structure based on duration
function calculateOptimalStructure(durationMinutes: number): { sections: number; lecturesPerSection: number } {
  // Target lecture duration: 5-10 minutes for short courses, 8-15 minutes for longer ones
  const targetLectureDuration = durationMinutes <= 60 ? 6 : durationMinutes <= 180 ? 10 : 12;

  const totalLectures = Math.max(2, Math.round(durationMinutes / targetLectureDuration));

  // Determine optimal sections based on total lectures
  let sections: number;
  let lecturesPerSection: number;

  if (totalLectures <= 4) {
    sections = 1;
    lecturesPerSection = totalLectures;
  } else if (totalLectures <= 8) {
    sections = 2;
    lecturesPerSection = Math.ceil(totalLectures / sections);
  } else if (totalLectures <= 15) {
    sections = 3;
    lecturesPerSection = Math.ceil(totalLectures / sections);
  } else if (totalLectures <= 24) {
    sections = 4;
    lecturesPerSection = Math.ceil(totalLectures / sections);
  } else if (totalLectures <= 35) {
    sections = 5;
    lecturesPerSection = Math.ceil(totalLectures / sections);
  } else if (totalLectures <= 48) {
    sections = 6;
    lecturesPerSection = Math.ceil(totalLectures / sections);
  } else {
    // For very long courses, cap sections at 10 and increase lectures per section
    sections = Math.min(10, Math.ceil(totalLectures / 6));
    lecturesPerSection = Math.ceil(totalLectures / sections);
  }

  return { sections, lecturesPerSection };
}

// Format duration for display
function formatDuration(minutes: number): string {
  if (minutes < 60) {
    return `${minutes} min`;
  }
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  if (mins === 0) {
    return `${hours}h`;
  }
  return `${hours}h ${mins}min`;
}

export function StructureConfig({ value, onChange }: StructureConfigProps) {
  const totalLectures = value.randomStructure
    ? '?'
    : value.numberOfSections * value.lecturesPerSection;

  const estimatedDurationPerLecture = value.randomStructure
    ? '?'
    : Math.round(value.totalDurationMinutes / (value.numberOfSections * value.lecturesPerSection));

  // Auto-adjust structure when duration changes (if not in random mode and user hasn't manually adjusted)
  const handleDurationChange = useCallback((newDuration: number) => {
    const optimal = calculateOptimalStructure(newDuration);
    onChange({
      ...value,
      totalDurationMinutes: newDuration,
      numberOfSections: optimal.sections,
      lecturesPerSection: optimal.lecturesPerSection,
    });
  }, [value, onChange]);

  // Toggle random structure
  const handleToggleRandom = useCallback(() => {
    const newRandomStructure = !value.randomStructure;
    if (newRandomStructure) {
      onChange({ ...value, randomStructure: true });
    } else {
      // When disabling random, calculate optimal structure
      const optimal = calculateOptimalStructure(value.totalDurationMinutes);
      onChange({
        ...value,
        randomStructure: false,
        numberOfSections: optimal.sections,
        lecturesPerSection: optimal.lecturesPerSection,
      });
    }
  }, [value, onChange]);

  return (
    <div className="space-y-4">
      {/* Random structure toggle */}
      <button
        type="button"
        onClick={handleToggleRandom}
        className="flex items-center gap-3 cursor-pointer group w-full text-left"
      >
        <div className={`relative w-12 h-6 rounded-full transition-colors ${
          value.randomStructure ? 'bg-purple-600' : 'bg-gray-700'
        }`}>
          <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
            value.randomStructure ? 'left-7' : 'left-1'
          }`} />
        </div>
        <div className="flex items-center gap-2">
          <Shuffle className="w-4 h-4 text-purple-400" />
          <span className="text-gray-300 group-hover:text-white transition-colors">
            Let AI decide structure
          </span>
        </div>
      </button>

      {/* Duration slider - Extended to 24 hours */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="flex items-center gap-2 text-gray-300">
            <Clock className="w-4 h-4 text-blue-400" />
            Total Duration
          </label>
          <span className="text-white font-medium">{formatDuration(value.totalDurationMinutes)}</span>
        </div>
        <input
          type="range"
          min="10"
          max="1440"
          step="10"
          value={value.totalDurationMinutes}
          onChange={(e) => handleDurationChange(parseInt(e.target.value))}
          className="w-full h-2 bg-gray-700 rounded-full appearance-none cursor-pointer accent-purple-500"
        />
        <div className="flex justify-between text-xs text-gray-500">
          <span>10 min</span>
          <span>24 hours</span>
        </div>
      </div>

      {!value.randomStructure && (
        <>
          {/* Sections slider - Extended to 50 */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="flex items-center gap-2 text-gray-300">
                <FolderOpen className="w-4 h-4 text-green-400" />
                Number of Sections
              </label>
              <span className="text-white font-medium">{value.numberOfSections}</span>
            </div>
            <input
              type="range"
              min="1"
              max="50"
              value={value.numberOfSections}
              onChange={(e) => onChange({ ...value, numberOfSections: parseInt(e.target.value) })}
              className="w-full h-2 bg-gray-700 rounded-full appearance-none cursor-pointer accent-purple-500"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>1</span>
              <span>50 sections</span>
            </div>
          </div>

          {/* Lectures per section slider - Extended to 20 */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="flex items-center gap-2 text-gray-300">
                <FileVideo className="w-4 h-4 text-yellow-400" />
                Lectures per Section
              </label>
              <span className="text-white font-medium">{value.lecturesPerSection}</span>
            </div>
            <input
              type="range"
              min="1"
              max="20"
              value={value.lecturesPerSection}
              onChange={(e) => onChange({ ...value, lecturesPerSection: parseInt(e.target.value) })}
              className="w-full h-2 bg-gray-700 rounded-full appearance-none cursor-pointer accent-purple-500"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>1</span>
              <span>20 lectures</span>
            </div>
          </div>
        </>
      )}

      {/* Summary */}
      <div className="bg-gray-800/50 rounded-lg p-4 mt-4">
        <h4 className="text-sm font-medium text-gray-400 mb-2">Course Summary</h4>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-2xl font-bold text-white">
              {value.randomStructure ? '?' : value.numberOfSections}
            </p>
            <p className="text-xs text-gray-500">Sections</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-white">{totalLectures}</p>
            <p className="text-xs text-gray-500">Lectures</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-white">~{estimatedDurationPerLecture}</p>
            <p className="text-xs text-gray-500">min/lecture</p>
          </div>
        </div>
        {!value.randomStructure && (
          <p className="text-xs text-gray-500 mt-3 text-center">
            Total: {formatDuration(value.totalDurationMinutes)} • {typeof totalLectures === 'number' ? totalLectures : value.numberOfSections * value.lecturesPerSection} videos
          </p>
        )}
      </div>
    </div>
  );
}
