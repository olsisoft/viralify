'use client';

import { useEffect, useMemo, useRef } from 'react';
import { HelpCircle, Sparkles, Loader2, Tag, Brain } from 'lucide-react';
import type { ContextQuestion, ProfileCategory, CourseContext } from '../lib/course-types';
import { CATEGORY_INFO } from '../lib/course-types';
import {
  getCategoryFromNiche,
  getBaseQuestions,
  areQuestionsAnswered,
} from '../lib/context-questions';
import type { CreatorProfile } from '@/lib/creator-profiles';

interface ContextQuestionsFormProps {
  profile: CreatorProfile | null;
  topic: string;
  answers: Record<string, string>;
  onAnswersChange: (answers: Record<string, string>) => void;
  onContextChange: (context: CourseContext | null) => void;
  aiQuestions?: ContextQuestion[];
  isLoadingAiQuestions?: boolean;
  // New props for auto-detection
  detectedCategory?: ProfileCategory | null;
  detectedDomain?: string | null;
  detectedDomainOptions?: string[];
  detectedKeywords?: string[];
  isDetecting?: boolean;
}

export function ContextQuestionsForm({
  profile,
  topic,
  answers,
  onAnswersChange,
  onContextChange,
  aiQuestions = [],
  isLoadingAiQuestions = false,
  detectedCategory,
  detectedDomain,
  detectedDomainOptions = [],
  detectedKeywords = [],
  isDetecting = false,
}: ContextQuestionsFormProps) {
  // Use detected category if available, otherwise fall back to profile niche
  const category = useMemo<ProfileCategory>(() => {
    if (detectedCategory) return detectedCategory;
    if (!profile?.niche) return 'lifestyle';
    return getCategoryFromNiche(profile.niche);
  }, [detectedCategory, profile?.niche]);

  // Get base questions for the category
  const staticBaseQuestions = useMemo(() => getBaseQuestions(category), [category]);

  // Domain question ID based on category
  const domainQuestionId = useMemo(() => {
    const domainQuestionIds: Record<ProfileCategory, string> = {
      tech: 'tech_domain',
      business: 'industry_focus',
      health: 'health_domain',
      creative: 'creative_domain',
      education: 'teaching_context',
      lifestyle: 'life_area',
    };
    return domainQuestionIds[category];
  }, [category]);

  // Override domain question options with AI-detected options
  const baseQuestions = useMemo(() => {
    if (!detectedDomainOptions || detectedDomainOptions.length === 0) {
      return staticBaseQuestions;
    }

    return staticBaseQuestions.map((q) => {
      if (q.id === domainQuestionId && q.type === 'select') {
        return {
          ...q,
          options: detectedDomainOptions,
        };
      }
      return q;
    });
  }, [staticBaseQuestions, domainQuestionId, detectedDomainOptions]);

  // All questions (base + AI)
  const allQuestions = useMemo(
    () => [...baseQuestions, ...aiQuestions],
    [baseQuestions, aiQuestions]
  );

  // Keep ref for onAnswersChange
  const onAnswersChangeRef = useRef(onAnswersChange);
  onAnswersChangeRef.current = onAnswersChange;

  // Auto-fill domain question when detected
  useEffect(() => {
    if (detectedDomain && domainQuestionId && !answers[domainQuestionId]) {
      console.log('[ContextQuestionsForm] Auto-filling domain:', detectedDomain);
      onAnswersChangeRef.current({
        ...answers,
        [domainQuestionId]: detectedDomain,
      });
    }
  }, [detectedDomain, domainQuestionId, answers]);

  // Check if form is complete
  const isComplete = useMemo(
    () => areQuestionsAnswered(baseQuestions, answers),
    [baseQuestions, answers]
  );

  // Use ref to store callback to avoid dependency issues
  const onContextChangeRef = useRef(onContextChange);
  onContextChangeRef.current = onContextChange;

  // Build context when answers change
  useEffect(() => {
    if (!profile || !isComplete) {
      onContextChangeRef.current(null);
      return;
    }

    const context: CourseContext = {
      category,
      profileNiche: profile.niche,
      profileTone: profile.tone,
      profileAudienceLevel: profile.audienceLevel,
      profileLanguageLevel: profile.languageLevel,
      profilePrimaryGoal: profile.primaryGoal,
      profileAudienceDescription: profile.audienceDescription || '',
      contextAnswers: answers,
      specificTools: answers.specific_tools || answers.tools_software,
      practicalFocus: answers.practical_focus,
      expectedOutcome: answers.expected_outcome || answers.transformation_goal,
    };

    onContextChangeRef.current(context);
  }, [profile, category, answers, isComplete]);

  const handleAnswerChange = (questionId: string, value: string) => {
    onAnswersChange({ ...answers, [questionId]: value });
  };

  const categoryInfo = CATEGORY_INFO[category];

  if (!profile) {
    return (
      <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6 text-center">
        <HelpCircle className="w-8 h-8 text-gray-500 mx-auto mb-2" />
        <p className="text-gray-400">
          Sélectionnez un profil pour voir les questions contextuelles
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Category Badge with Detection Status */}
      <div className="flex items-center justify-between pb-4 border-b border-gray-700">
        <div className="flex items-center gap-3">
          <span className="text-2xl">{categoryInfo.icon}</span>
          <div>
            <h3 className="text-lg font-semibold text-white">
              Questions pour votre cours {categoryInfo.label}
            </h3>
            <p className="text-sm text-gray-400">
              Ces questions nous aident à personnaliser le contenu du cours
            </p>
          </div>
        </div>
        {isDetecting && (
          <div className="flex items-center gap-2 text-purple-400">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-sm">Analyse IA...</span>
          </div>
        )}
        {!isDetecting && detectedCategory && (
          <div className="flex items-center gap-1 px-2 py-1 bg-purple-500/20 rounded-full">
            <Brain className="w-3 h-3 text-purple-400" />
            <span className="text-xs text-purple-300">Auto-détecté</span>
          </div>
        )}
      </div>

      {/* Detected Keywords */}
      {detectedKeywords && detectedKeywords.length > 0 && (
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Tag className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-medium text-gray-300">Mots-clés détectés</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {detectedKeywords.map((keyword, index) => (
              <span
                key={index}
                className="px-2 py-1 text-xs bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 rounded-full"
              >
                {keyword}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Base Questions */}
      <div className="space-y-4">
        {baseQuestions.map((question, index) => (
          <QuestionItem
            key={question.id}
            question={question}
            index={index + 1}
            value={answers[question.id] || ''}
            onChange={(value) => handleAnswerChange(question.id, value)}
          />
        ))}
      </div>

      {/* AI Questions */}
      {(aiQuestions.length > 0 || isLoadingAiQuestions) && (
        <div className="space-y-4 pt-4 border-t border-gray-700">
          <div className="flex items-center gap-2 text-purple-400">
            <Sparkles className="w-4 h-4" />
            <span className="text-sm font-medium">
              Questions spécifiques à votre sujet
            </span>
          </div>

          {isLoadingAiQuestions ? (
            <div className="animate-pulse space-y-3">
              <div className="h-16 bg-gray-700/50 rounded-lg" />
              <div className="h-16 bg-gray-700/50 rounded-lg" />
            </div>
          ) : (
            aiQuestions.map((question, index) => (
              <QuestionItem
                key={question.id}
                question={question}
                index={baseQuestions.length + index + 1}
                value={answers[question.id] || ''}
                onChange={(value) => handleAnswerChange(question.id, value)}
                isAiGenerated
              />
            ))
          )}
        </div>
      )}

      {/* Context Summary */}
      {isComplete && topic && (
        <div className="mt-6 p-4 bg-purple-500/10 border border-purple-500/30 rounded-lg">
          <h4 className="text-sm font-medium text-purple-300 mb-2">
            Résumé du contexte
          </h4>
          <div className="text-sm text-gray-300 space-y-1">
            <p>
              <span className="text-gray-500">Catégorie:</span> {categoryInfo.icon}{' '}
              {categoryInfo.label}
            </p>
            <p>
              <span className="text-gray-500">Ton:</span> {profile.tone}
            </p>
            <p>
              <span className="text-gray-500">Audience:</span> {profile.audienceLevel}
            </p>
            {Object.entries(answers)
              .filter(([_, v]) => v)
              .slice(0, 2)
              .map(([key, value]) => (
                <p key={key}>
                  <span className="text-gray-500">
                    {key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}:
                  </span>{' '}
                  {value}
                </p>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}

interface QuestionItemProps {
  question: ContextQuestion;
  index: number;
  value: string;
  onChange: (value: string) => void;
  isAiGenerated?: boolean;
}

function QuestionItem({
  question,
  index,
  value,
  onChange,
  isAiGenerated = false,
}: QuestionItemProps) {
  return (
    <div
      className={`p-4 rounded-lg ${
        isAiGenerated
          ? 'bg-purple-500/5 border border-purple-500/20'
          : 'bg-gray-800/50 border border-gray-700'
      }`}
    >
      <label className="block mb-2">
        <span className="text-sm text-gray-400">{index}.</span>{' '}
        <span className="text-white">{question.question}</span>
        {question.required !== false && (
          <span className="text-red-400 ml-1">*</span>
        )}
      </label>

      {question.type === 'select' && question.options ? (
        <div className="flex flex-wrap gap-2">
          {question.options.map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => onChange(option)}
              className={`px-3 py-1.5 rounded-full text-sm transition-colors ${
                value === option
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {option}
            </button>
          ))}
        </div>
      ) : (
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={question.placeholder}
          className="w-full bg-gray-900/50 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
        />
      )}
    </div>
  );
}
