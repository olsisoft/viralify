'use client';

import {
  BookOpen,
  GitBranch,
  Code,
  Play,
  Mic,
  ListOrdered,
  Check,
  Lock,
} from 'lucide-react';
import type { LessonElementConfig } from '../lib/course-types';

interface ElementOption {
  id: keyof LessonElementConfig;
  name: string;
  description: string;
  icon: React.ReactNode;
  readonly?: boolean;
}

const ELEMENTS: ElementOption[] = [
  {
    id: 'curriculumSlide',
    name: 'Curriculum Slide',
    description: 'Shows lecture position in the course',
    icon: <ListOrdered className="w-5 h-5" />,
    readonly: true,
  },
  {
    id: 'conceptIntro',
    name: 'Concept Introduction',
    description: 'Start with theory explanation',
    icon: <BookOpen className="w-5 h-5" />,
  },
  {
    id: 'diagramSchema',
    name: 'Diagram/Schema',
    description: 'Visual diagrams and flowcharts',
    icon: <GitBranch className="w-5 h-5" />,
  },
  {
    id: 'codeTyping',
    name: 'Code Typing Animation',
    description: 'Show code being typed live',
    icon: <Code className="w-5 h-5" />,
  },
  {
    id: 'codeExecution',
    name: 'Code Execution',
    description: 'Execute code and show output',
    icon: <Play className="w-5 h-5" />,
  },
  {
    id: 'voiceoverExplanation',
    name: 'Voiceover During Code',
    description: 'Narration while typing code',
    icon: <Mic className="w-5 h-5" />,
  },
];

interface LessonElementsConfigProps {
  value: LessonElementConfig;
  onChange: (value: LessonElementConfig) => void;
}

export function LessonElementsConfig({ value, onChange }: LessonElementsConfigProps) {
  const toggleElement = (id: keyof LessonElementConfig) => {
    if (id === 'curriculumSlide') return; // readonly
    onChange({ ...value, [id]: !value[id] });
  };

  const enabledCount = Object.entries(value).filter(
    ([key, val]) => val && key !== 'curriculumSlide'
  ).length;

  return (
    <div className="space-y-3">
      {ELEMENTS.map((element) => {
        const isEnabled = value[element.id];
        const isReadonly = element.readonly;

        return (
          <button
            key={element.id}
            type="button"
            onClick={() => toggleElement(element.id)}
            disabled={isReadonly}
            className={`w-full flex items-center gap-4 p-3 rounded-lg border transition-all ${
              isEnabled
                ? 'bg-purple-600/10 border-purple-500/50 text-white'
                : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:border-gray-600'
            } ${isReadonly ? 'cursor-not-allowed opacity-75' : 'cursor-pointer'}`}
          >
            <div className={`p-2 rounded-lg ${
              isEnabled ? 'bg-purple-600/20 text-purple-400' : 'bg-gray-700 text-gray-500'
            }`}>
              {element.icon}
            </div>

            <div className="flex-1 text-left">
              <p className="font-medium">{element.name}</p>
              <p className="text-sm text-gray-500">{element.description}</p>
            </div>

            {isReadonly ? (
              <Lock className="w-5 h-5 text-gray-500" />
            ) : (
              <div className={`w-6 h-6 rounded-full border-2 flex items-center justify-center transition-colors ${
                isEnabled
                  ? 'bg-purple-600 border-purple-600'
                  : 'border-gray-600'
              }`}>
                {isEnabled && <Check className="w-4 h-4 text-white" />}
              </div>
            )}
          </button>
        );
      })}

      {/* Quick presets */}
      <div className="pt-3 border-t border-gray-700">
        <p className="text-sm text-gray-500 mb-2">Quick Presets:</p>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => onChange({
              ...value,
              conceptIntro: true,
              diagramSchema: true,
              codeTyping: true,
              codeExecution: false,
              voiceoverExplanation: true,
            })}
            className="px-3 py-1.5 rounded-full text-xs font-medium bg-gray-700 text-gray-300 hover:bg-gray-600 transition-colors"
          >
            Standard
          </button>
          <button
            type="button"
            onClick={() => onChange({
              ...value,
              conceptIntro: false,
              diagramSchema: false,
              codeTyping: true,
              codeExecution: true,
              voiceoverExplanation: true,
            })}
            className="px-3 py-1.5 rounded-full text-xs font-medium bg-gray-700 text-gray-300 hover:bg-gray-600 transition-colors"
          >
            Code-Focused
          </button>
          <button
            type="button"
            onClick={() => onChange({
              ...value,
              conceptIntro: true,
              diagramSchema: true,
              codeTyping: true,
              codeExecution: true,
              voiceoverExplanation: true,
            })}
            className="px-3 py-1.5 rounded-full text-xs font-medium bg-gray-700 text-gray-300 hover:bg-gray-600 transition-colors"
          >
            Full
          </button>
        </div>
      </div>

      <p className="text-sm text-gray-500 text-center">
        {enabledCount} elements enabled (+ curriculum slide)
      </p>
    </div>
  );
}
