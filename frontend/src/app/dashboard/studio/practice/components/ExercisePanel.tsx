'use client';

import React, { useState } from 'react';
import {
  ChevronDown,
  ChevronUp,
  Lightbulb,
  Clock,
  Trophy,
  Tag,
  Eye,
  EyeOff,
} from 'lucide-react';
import {
  Exercise,
  getDifficultyLabel,
  getDifficultyColor,
  getCategoryLabel,
  getCategoryIcon,
  formatDuration,
} from '../lib/practice-types';

interface ExercisePanelProps {
  exercise: Exercise;
  onRequestHint: (level: number) => void;
  hintsUsed: number;
  isLoading?: boolean;
}

export function ExercisePanel({
  exercise,
  onRequestHint,
  hintsUsed,
  isLoading = false,
}: ExercisePanelProps) {
  const [showSolution, setShowSolution] = useState(false);
  const [expandedSection, setExpandedSection] = useState<'instructions' | 'hints' | null>(
    'instructions'
  );

  const availableHints = exercise.hints.length - hintsUsed;

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
      {/* Header */}
      <div className="p-4 bg-gray-800 border-b border-gray-700">
        <div className="flex items-start justify-between mb-2">
          <h2 className="text-lg font-semibold text-white">{exercise.title}</h2>
          <span
            className={`px-2 py-0.5 text-xs font-medium rounded ${getDifficultyColor(
              exercise.difficulty
            )}`}
          >
            {getDifficultyLabel(exercise.difficulty)}
          </span>
        </div>

        <p className="text-sm text-gray-400 mb-3">{exercise.description}</p>

        {/* Meta info */}
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <span className="flex items-center gap-1 text-gray-400">
            {getCategoryIcon(exercise.category)}
            {getCategoryLabel(exercise.category)}
          </span>

          <span className="flex items-center gap-1 text-gray-400">
            <Clock className="w-4 h-4" />
            {formatDuration(exercise.estimated_minutes)}
          </span>

          <span className="flex items-center gap-1 text-yellow-500">
            <Trophy className="w-4 h-4" />
            {exercise.points} pts
          </span>
        </div>

        {/* Tags */}
        {exercise.tags.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-3">
            {exercise.tags.map((tag) => (
              <span
                key={tag}
                className="flex items-center gap-1 px-2 py-0.5 text-xs text-gray-400 bg-gray-700 rounded"
              >
                <Tag className="w-3 h-3" />
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {/* Instructions Section */}
        <div className="border-b border-gray-700">
          <button
            onClick={() =>
              setExpandedSection(expandedSection === 'instructions' ? null : 'instructions')
            }
            className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-gray-800/50 transition-colors"
          >
            <span className="font-medium text-white">Instructions</span>
            {expandedSection === 'instructions' ? (
              <ChevronUp className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            )}
          </button>

          {expandedSection === 'instructions' && (
            <div className="px-4 pb-4">
              <div
                className="prose prose-invert prose-sm max-w-none"
                dangerouslySetInnerHTML={{
                  __html: exercise.instructions
                    .replace(/\n/g, '<br>')
                    .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
                    .replace(/`([^`]+)`/g, '<code>$1</code>')
                    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
                    .replace(/## (.+)/g, '<h3 class="text-white font-semibold mt-4 mb-2">$1</h3>')
                    .replace(/- (.+)/g, '<li class="text-gray-300">$1</li>'),
                }}
              />
            </div>
          )}
        </div>

        {/* Hints Section */}
        <div className="border-b border-gray-700">
          <button
            onClick={() =>
              setExpandedSection(expandedSection === 'hints' ? null : 'hints')
            }
            className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-gray-800/50 transition-colors"
          >
            <span className="flex items-center gap-2 font-medium text-white">
              <Lightbulb className="w-4 h-4 text-yellow-500" />
              Indices
              {availableHints > 0 && (
                <span className="text-xs text-gray-400">
                  ({availableHints} disponible{availableHints > 1 ? 's' : ''})
                </span>
              )}
            </span>
            {expandedSection === 'hints' ? (
              <ChevronUp className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            )}
          </button>

          {expandedSection === 'hints' && (
            <div className="px-4 pb-4 space-y-3">
              {/* Already revealed hints */}
              {hintsUsed > 0 && (
                <div className="space-y-2">
                  {exercise.hints.slice(0, hintsUsed).map((hint, i) => (
                    <div
                      key={i}
                      className="p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg"
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <Lightbulb className="w-4 h-4 text-yellow-500" />
                        <span className="text-sm font-medium text-yellow-500">
                          Indice {i + 1}
                        </span>
                      </div>
                      <p className="text-sm text-gray-300">{hint}</p>
                    </div>
                  ))}
                </div>
              )}

              {/* Request next hint button */}
              {availableHints > 0 && (
                <button
                  onClick={() => onRequestHint(hintsUsed + 1)}
                  disabled={isLoading}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-yellow-600/20 hover:bg-yellow-600/30 border border-yellow-600/50 text-yellow-500 rounded-lg transition-colors disabled:opacity-50"
                >
                  <Lightbulb className="w-4 h-4" />
                  Demander un indice (-10 pts)
                </button>
              )}

              {availableHints === 0 && hintsUsed > 0 && (
                <p className="text-sm text-gray-400 text-center">
                  Tous les indices ont été utilisés
                </p>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Solution toggle (only after completion or debug mode) */}
      {exercise.solution && (
        <div className="p-4 border-t border-gray-700">
          <button
            onClick={() => setShowSolution(!showSolution)}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 text-sm text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
          >
            {showSolution ? (
              <>
                <EyeOff className="w-4 h-4" />
                Masquer la solution
              </>
            ) : (
              <>
                <Eye className="w-4 h-4" />
                Voir la solution
              </>
            )}
          </button>

          {showSolution && (
            <div className="mt-3 p-3 bg-gray-800 rounded-lg">
              <pre className="text-sm text-gray-300 overflow-x-auto whitespace-pre-wrap">
                {exercise.solution}
              </pre>
              {exercise.solution_explanation && (
                <div className="mt-3 pt-3 border-t border-gray-700">
                  <p className="text-sm text-gray-400 whitespace-pre-wrap">
                    {exercise.solution_explanation}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
