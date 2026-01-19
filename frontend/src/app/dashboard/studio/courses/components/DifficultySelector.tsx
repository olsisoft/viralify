'use client';

import { useMemo } from 'react';
import type { DifficultyLevel } from '../lib/course-types';

const DIFFICULTY_LEVELS: { id: DifficultyLevel; name: string; color: string }[] = [
  { id: 'beginner', name: 'Beginner', color: 'bg-green-500' },
  { id: 'intermediate', name: 'Intermediate', color: 'bg-blue-500' },
  { id: 'advanced', name: 'Advanced', color: 'bg-yellow-500' },
  { id: 'very_advanced', name: 'Very Advanced', color: 'bg-orange-500' },
  { id: 'expert', name: 'Expert', color: 'bg-red-500' },
];

interface DifficultySelectorProps {
  startValue: DifficultyLevel;
  endValue: DifficultyLevel;
  onStartChange: (value: DifficultyLevel) => void;
  onEndChange: (value: DifficultyLevel) => void;
}

export function DifficultySelector({
  startValue,
  endValue,
  onStartChange,
  onEndChange,
}: DifficultySelectorProps) {
  const startIndex = DIFFICULTY_LEVELS.findIndex(d => d.id === startValue);
  const endIndex = DIFFICULTY_LEVELS.findIndex(d => d.id === endValue);

  const progressWidth = useMemo(() => {
    const start = (startIndex / (DIFFICULTY_LEVELS.length - 1)) * 100;
    const end = (endIndex / (DIFFICULTY_LEVELS.length - 1)) * 100;
    return { left: `${start}%`, width: `${end - start}%` };
  }, [startIndex, endIndex]);

  const handleStartChange = (index: number) => {
    if (index <= endIndex) {
      onStartChange(DIFFICULTY_LEVELS[index].id);
    }
  };

  const handleEndChange = (index: number) => {
    if (index >= startIndex) {
      onEndChange(DIFFICULTY_LEVELS[index].id);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between text-sm">
        <span className="text-gray-400">
          Start: <span className="text-white font-medium">{DIFFICULTY_LEVELS[startIndex]?.name}</span>
        </span>
        <span className="text-gray-400">
          End: <span className="text-white font-medium">{DIFFICULTY_LEVELS[endIndex]?.name}</span>
        </span>
      </div>

      {/* Range track */}
      <div className="relative h-3">
        {/* Background track */}
        <div className="absolute inset-0 bg-gray-700 rounded-full" />

        {/* Active range */}
        <div
          className="absolute h-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 rounded-full transition-all"
          style={progressWidth}
        />

        {/* Level markers */}
        <div className="absolute inset-0 flex justify-between items-center px-0">
          {DIFFICULTY_LEVELS.map((level, index) => (
            <button
              key={level.id}
              type="button"
              onClick={() => {
                // If clicking left of or at start, change start
                // If clicking right of or at end, change end
                // If clicking in between, change the closest one
                if (index <= startIndex) {
                  handleStartChange(index);
                } else if (index >= endIndex) {
                  handleEndChange(index);
                } else {
                  const distToStart = index - startIndex;
                  const distToEnd = endIndex - index;
                  if (distToStart <= distToEnd) {
                    handleStartChange(index);
                  } else {
                    handleEndChange(index);
                  }
                }
              }}
              className={`w-5 h-5 rounded-full border-2 transition-all z-10 ${
                index >= startIndex && index <= endIndex
                  ? `${level.color} border-white scale-110`
                  : 'bg-gray-600 border-gray-500 hover:scale-105'
              }`}
              title={level.name}
            />
          ))}
        </div>
      </div>

      {/* Level labels */}
      <div className="flex justify-between text-xs text-gray-500">
        {DIFFICULTY_LEVELS.map((level) => (
          <span key={level.id} className="w-16 text-center">
            {level.name.split(' ')[0]}
          </span>
        ))}
      </div>

      {/* Quick select buttons */}
      <div className="flex flex-wrap gap-2 pt-2">
        <button
          type="button"
          onClick={() => {
            onStartChange('beginner');
            onEndChange('intermediate');
          }}
          className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
            startValue === 'beginner' && endValue === 'intermediate'
              ? 'bg-purple-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          Beginner → Intermediate
        </button>
        <button
          type="button"
          onClick={() => {
            onStartChange('intermediate');
            onEndChange('advanced');
          }}
          className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
            startValue === 'intermediate' && endValue === 'advanced'
              ? 'bg-purple-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          Intermediate → Advanced
        </button>
        <button
          type="button"
          onClick={() => {
            onStartChange('beginner');
            onEndChange('expert');
          }}
          className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
            startValue === 'beginner' && endValue === 'expert'
              ? 'bg-purple-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          Full Range
        </button>
      </div>
    </div>
  );
}
