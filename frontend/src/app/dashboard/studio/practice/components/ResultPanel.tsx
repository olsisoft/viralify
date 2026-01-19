'use client';

import React from 'react';
import {
  CheckCircle,
  XCircle,
  AlertCircle,
  Terminal,
  Trophy,
  ArrowRight,
} from 'lucide-react';
import { SubmitCodeResponse, Exercise } from '../lib/practice-types';

interface ResultPanelProps {
  result: SubmitCodeResponse | null;
  onNextExercise?: () => void;
  onRetry?: () => void;
}

export function ResultPanel({
  result,
  onNextExercise,
  onRetry,
}: ResultPanelProps) {
  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center h-full bg-gray-900 rounded-lg border border-gray-700 p-6">
        <Terminal className="w-12 h-12 text-gray-600 mb-3" />
        <p className="text-gray-400 text-sm text-center">
          Exécutez votre code pour voir les résultats ici
        </p>
      </div>
    );
  }

  const { passed, score, feedback, checks_passed, checks_failed, execution_output, next_exercise } = result;

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
      {/* Header */}
      <div
        className={`px-4 py-3 border-b border-gray-700 ${
          passed ? 'bg-green-900/20' : 'bg-red-900/20'
        }`}
      >
        <div className="flex items-center gap-3">
          {passed ? (
            <CheckCircle className="w-6 h-6 text-green-500" />
          ) : (
            <XCircle className="w-6 h-6 text-red-500" />
          )}
          <div>
            <h3 className={`font-semibold ${passed ? 'text-green-500' : 'text-red-500'}`}>
              {passed ? 'Exercice réussi!' : 'Pas encore...'}
            </h3>
            <div className="flex items-center gap-2 text-sm">
              <Trophy className="w-4 h-4 text-yellow-500" />
              <span className="text-yellow-500">{score} points</span>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Feedback */}
        <div className="p-3 bg-gray-800 rounded-lg">
          <p className="text-sm text-gray-300 whitespace-pre-wrap">{feedback}</p>
        </div>

        {/* Checks */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-400">Validations</h4>

          {/* Passed checks */}
          {checks_passed.length > 0 && (
            <div className="space-y-1">
              {checks_passed.map((check, i) => (
                <div key={i} className="flex items-center gap-2 text-sm">
                  <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
                  <span className="text-gray-300">{check}</span>
                </div>
              ))}
            </div>
          )}

          {/* Failed checks */}
          {checks_failed.length > 0 && (
            <div className="space-y-1 mt-2">
              {checks_failed.map((check, i) => (
                <div key={i} className="flex items-center gap-2 text-sm">
                  <XCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
                  <span className="text-gray-300">{check}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Execution output */}
        {execution_output && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-gray-400">Sortie d'exécution</h4>
            <div className="p-3 bg-black rounded-lg font-mono text-xs">
              <pre className="text-gray-300 whitespace-pre-wrap overflow-x-auto">
                {execution_output}
              </pre>
            </div>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="p-4 border-t border-gray-700 space-y-2">
        {passed && next_exercise ? (
          <button
            onClick={onNextExercise}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg transition-colors"
          >
            Exercice suivant
            <ArrowRight className="w-4 h-4" />
          </button>
        ) : !passed ? (
          <button
            onClick={onRetry}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
          >
            <AlertCircle className="w-4 h-4" />
            Réessayer
          </button>
        ) : null}
      </div>
    </div>
  );
}
