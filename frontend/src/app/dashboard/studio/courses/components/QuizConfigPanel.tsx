'use client';

import { useState } from 'react';
import {
  HelpCircle,
  Check,
  ChevronDown,
  ChevronUp,
  Info,
} from 'lucide-react';
import type { QuizConfig, QuizFrequency, QuizQuestionType } from '../lib/lesson-elements';
import { QUIZ_FREQUENCIES, QUIZ_QUESTION_TYPES, DEFAULT_QUIZ_CONFIG } from '../lib/lesson-elements';

interface QuizConfigPanelProps {
  value: QuizConfig;
  onChange: (value: QuizConfig) => void;
}

export function QuizConfigPanel({ value, onChange }: QuizConfigPanelProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleFrequencyChange = (frequency: QuizFrequency) => {
    onChange({ ...value, frequency });
  };

  const handleQuestionTypeToggle = (type: QuizQuestionType) => {
    const types = value.questionTypes.includes(type)
      ? value.questionTypes.filter((t) => t !== type)
      : [...value.questionTypes, type];

    // Ensure at least one type is selected
    if (types.length === 0) return;

    onChange({ ...value, questionTypes: types });
  };

  return (
    <div className="space-y-4">
      {/* Quiz Enabled Banner */}
      <div className="flex items-center gap-3 p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
        <HelpCircle className="w-5 h-5 text-green-400" />
        <div className="flex-1">
          <p className="text-sm text-green-300 font-medium">
            Quiz d'évaluation activés
          </p>
          <p className="text-xs text-green-400/70">
            Les quiz sont obligatoires pour tous les cours (format Udemy)
          </p>
        </div>
      </div>

      {/* Frequency Selection */}
      <div className="space-y-2">
        <label className="text-sm font-medium text-gray-300">
          Fréquence des quiz
        </label>
        <div className="grid grid-cols-2 gap-2">
          {QUIZ_FREQUENCIES.map((freq) => (
            <button
              key={freq.id}
              type="button"
              onClick={() => handleFrequencyChange(freq.id)}
              className={`p-3 rounded-lg border text-left transition-all ${
                value.frequency === freq.id
                  ? 'bg-purple-600/10 border-purple-500/50 text-white'
                  : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:border-gray-600'
              }`}
            >
              <p className="font-medium text-sm">{freq.name}</p>
              <p className="text-xs text-gray-500">{freq.description}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Custom Frequency Input */}
      {value.frequency === 'custom' && (
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-300">
            Quiz toutes les N lectures
          </label>
          <input
            type="number"
            min={1}
            max={10}
            value={value.customFrequency || 3}
            onChange={(e) =>
              onChange({ ...value, customFrequency: parseInt(e.target.value) || 3 })
            }
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:border-purple-500"
          />
        </div>
      )}

      {/* Questions per Quiz */}
      <div className="space-y-2">
        <label className="text-sm font-medium text-gray-300">
          Questions par quiz: {value.questionsPerQuiz}
        </label>
        <input
          type="range"
          min={3}
          max={15}
          value={value.questionsPerQuiz}
          onChange={(e) =>
            onChange({ ...value, questionsPerQuiz: parseInt(e.target.value) })
          }
          className="w-full accent-purple-500"
        />
        <div className="flex justify-between text-xs text-gray-500">
          <span>3</span>
          <span>15</span>
        </div>
      </div>

      {/* Question Types */}
      <div className="space-y-2">
        <label className="text-sm font-medium text-gray-300">
          Types de questions
        </label>
        <div className="flex flex-wrap gap-2">
          {QUIZ_QUESTION_TYPES.map((type) => {
            const isSelected = value.questionTypes.includes(type.id);
            return (
              <button
                key={type.id}
                type="button"
                onClick={() => handleQuestionTypeToggle(type.id)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm transition-colors ${
                  isSelected
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {isSelected && <Check className="w-3 h-3" />}
                {type.name}
              </button>
            );
          })}
        </div>
      </div>

      {/* Advanced Settings Toggle */}
      <button
        type="button"
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="w-full flex items-center justify-between text-sm text-gray-400 hover:text-gray-300"
      >
        <span>Options avancées</span>
        {showAdvanced ? (
          <ChevronUp className="w-4 h-4" />
        ) : (
          <ChevronDown className="w-4 h-4" />
        )}
      </button>

      {/* Advanced Settings */}
      {showAdvanced && (
        <div className="space-y-4 pt-2 border-t border-gray-700">
          {/* Passing Score */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">
              Score de réussite: {value.passingScore}%
            </label>
            <input
              type="range"
              min={50}
              max={100}
              step={5}
              value={value.passingScore}
              onChange={(e) =>
                onChange({ ...value, passingScore: parseInt(e.target.value) })
              }
              className="w-full accent-purple-500"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>50%</span>
              <span>100%</span>
            </div>
          </div>

          {/* Show Explanations */}
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={value.showExplanations}
              onChange={(e) =>
                onChange({ ...value, showExplanations: e.target.checked })
              }
              className="w-5 h-5 rounded border-gray-600 bg-gray-700 text-purple-600 focus:ring-purple-500"
            />
            <div>
              <p className="text-sm text-white">Afficher les explications</p>
              <p className="text-xs text-gray-500">
                Montrer les explications après chaque réponse
              </p>
            </div>
          </label>

          {/* Allow Retry */}
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={value.allowRetry}
              onChange={(e) =>
                onChange({ ...value, allowRetry: e.target.checked })
              }
              className="w-5 h-5 rounded border-gray-600 bg-gray-700 text-purple-600 focus:ring-purple-500"
            />
            <div>
              <p className="text-sm text-white">Autoriser les reprises</p>
              <p className="text-xs text-gray-500">
                Permettre de refaire le quiz en cas d'échec
              </p>
            </div>
          </label>
        </div>
      )}

      {/* Info */}
      <div className="flex items-start gap-2 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
        <Info className="w-4 h-4 text-blue-400 mt-0.5" />
        <p className="text-xs text-blue-300">
          Les quiz seront générés automatiquement par l'IA en fonction du contenu de chaque leçon.
        </p>
      </div>
    </div>
  );
}
