'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  BookOpen,
  Target,
  Settings2,
  Film,
  ChevronDown,
  ChevronUp,
  Loader2,
  Eye,
  Rocket,
  MessageSquare,
  FileText,
  Sparkles,
  Brain,
} from 'lucide-react';
import { ProfileSelector } from './ProfileSelector';
import { DifficultySelector } from './DifficultySelector';
import { StructureConfig } from './StructureConfig';
import { LessonElementsConfig } from './LessonElementsConfig';
import { ContextQuestionsForm } from './ContextQuestionsForm';
import { DocumentUpload } from './DocumentUpload';
import { AdaptiveLessonElements } from './AdaptiveLessonElements';
import { SourceLibrary } from './SourceLibrary';
import { KeywordsInput } from './KeywordsInput';
import type { CourseFormState, CourseContext, ProfileCategory, DetectedCategory } from '../lib/course-types';
import type { Document } from '../lib/document-types';
import { getCreatorProfiles, type CreatorProfile } from '@/lib/creator-profiles';
import { Library, Globe, Mic } from 'lucide-react';
import { useVoices } from '../hooks/useVoices';
import { SUPPORTED_LANGUAGES, getLanguageFlag, getGenderLabel } from '../lib/voice-types';

const CATEGORY_LABELS: Record<ProfileCategory, { icon: string; label: string }> = {
  business: { icon: '💼', label: 'Business' },
  tech: { icon: '💻', label: 'Technique' },
  health: { icon: '🏃', label: 'Santé/Fitness' },
  creative: { icon: '🎨', label: 'Créatif' },
  education: { icon: '📚', label: 'Éducation' },
  lifestyle: { icon: '✨', label: 'Lifestyle' },
};

interface CourseFormProps {
  formState: CourseFormState;
  onFormChange: (state: CourseFormState | ((prev: CourseFormState) => CourseFormState)) => void;
  onPreview: () => void;
  onGenerate: () => void;
  isPreviewLoading: boolean;
  isGenerating: boolean;
  hasPreview: boolean;
}

export function CourseForm({
  formState,
  onFormChange,
  onPreview,
  onGenerate,
  isPreviewLoading,
  isGenerating,
  hasPreview,
}: CourseFormProps) {
  const [expandedSections, setExpandedSections] = useState({
    context: true,
    documents: false,
    sourceLibrary: true, // New source library section
    structure: true,
    elements: true, // Show elements by default now
    advanced: false,
  });
  const [linkedSourceIds, setLinkedSourceIds] = useState<string[]>([]);

  const [selectedProfile, setSelectedProfile] = useState<CreatorProfile | null>(null);
  const [isDetectingCategory, setIsDetectingCategory] = useState(false);
  const detectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastDetectedTopicRef = useRef<string>('');

  // Voice selection state
  const [selectedVoiceLanguage, setSelectedVoiceLanguage] = useState('fr');
  const { voices, isLoading: isLoadingVoices } = useVoices({ language: selectedVoiceLanguage });

  // Update voiceId when voices change and current voice is not available
  useEffect(() => {
    if (voices.length > 0) {
      const currentVoiceExists = voices.some((v) => v.id === formState.voiceId);
      if (!currentVoiceExists) {
        // Select first available voice
        onFormChange((prev) => ({ ...prev, voiceId: voices[0].id }));
      }
    }
  }, [voices, formState.voiceId, onFormChange]);

  // Load selected profile when profileId changes
  useEffect(() => {
    if (formState.profileId) {
      const profiles = getCreatorProfiles();
      const profile = profiles.find((p) => p.id === formState.profileId) || null;
      setSelectedProfile(profile);
    } else {
      setSelectedProfile(null);
    }
  }, [formState.profileId]);

  // Keep a ref to onFormChange to avoid stale closures
  const onFormChangeRef = useRef(onFormChange);
  onFormChangeRef.current = onFormChange;

  // Auto-detect category when topic changes (with debounce)
  useEffect(() => {
    // Clear previous timeout
    if (detectTimeoutRef.current) {
      clearTimeout(detectTimeoutRef.current);
    }

    // Only detect if topic is long enough
    if (formState.topic.trim().length < 5) {
      return;
    }

    // Debounce the detection
    detectTimeoutRef.current = setTimeout(async () => {
      // Skip if topic hasn't meaningfully changed (within 3 char difference)
      const normalizedTopic = formState.topic.trim().toLowerCase();
      const lastNormalized = lastDetectedTopicRef.current.trim().toLowerCase();

      if (lastNormalized && Math.abs(normalizedTopic.length - lastNormalized.length) <= 3 &&
          (normalizedTopic.startsWith(lastNormalized) || lastNormalized.startsWith(normalizedTopic))) {
        console.log('[CourseForm] Skipping category detection - topic too similar:', formState.topic);
        return;
      }

      lastDetectedTopicRef.current = formState.topic;
      console.log('[CourseForm] Starting category detection for topic:', formState.topic);

      setIsDetectingCategory(true);
      try {
        const params = new URLSearchParams({ topic: formState.topic });
        if (formState.description) {
          params.append('description', formState.description);
        }

        console.log('[CourseForm] Fetching category detection...');
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || '';
        const response = await fetch(
          `${apiUrl}/api/v1/courses/config/detect-category?${params}`
        );

        if (response.ok) {
          const data = await response.json();
          console.log('[CourseForm] Category detected:', data);
          onFormChangeRef.current((prev) => ({
            ...prev,
            detectedCategory: {
              category: data.category as ProfileCategory,
              confidence: data.confidence,
              domain: data.domain,
              domainOptions: data.domain_options,
              keywords: data.keywords,
            },
          }));
        } else {
          console.error('[CourseForm] Category detection failed:', response.status);
        }
      } catch (err) {
        console.error('[CourseForm] Error detecting category:', err);
      } finally {
        setIsDetectingCategory(false);
      }
    }, 1500); // Increased from 800ms to 1500ms for more stable typing

    return () => {
      if (detectTimeoutRef.current) {
        clearTimeout(detectTimeoutRef.current);
      }
    };
  }, [formState.topic, formState.description]);

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  const handleProfileChange = useCallback((profileId: string) => {
    onFormChange((prev) => ({
      ...prev,
      profileId,
      // Reset context answers when profile changes
      contextAnswers: {},
      context: null,
    }));
  }, [onFormChange]);

  const handleAnswersChange = useCallback((answers: Record<string, string>) => {
    onFormChange((prev) => ({ ...prev, contextAnswers: answers }));
  }, [onFormChange]);

  const handleContextChange = useCallback((context: CourseContext | null) => {
    onFormChange((prev) => ({ ...prev, context }));
  }, [onFormChange]);

  const handleTopicChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onFormChange((prev) => ({ ...prev, topic: e.target.value }));
  }, [onFormChange]);

  const handleDescriptionChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onFormChange((prev) => ({ ...prev, description: e.target.value }));
  }, [onFormChange]);

  const handleDocumentsChange = useCallback((documents: Document[]) => {
    onFormChange((prev) => ({ ...prev, documents }));
  }, [onFormChange]);

  const handleSourcesChange = useCallback((sourceIds: string[]) => {
    setLinkedSourceIds(sourceIds);
    // Store source IDs in form state for course generation
    onFormChange((prev) => ({
      ...prev,
      sourceIds: sourceIds,
    }));
  }, [onFormChange]);

  const handleKeywordsChange = useCallback((keywords: string[]) => {
    onFormChange((prev) => ({
      ...prev,
      customKeywords: keywords,
    }));
  }, [onFormChange]);

  const handleAdaptiveElementsChange = useCallback((elements: Record<string, boolean>) => {
    onFormChange((prev) => ({
      ...prev,
      adaptiveElements: {
        ...prev.adaptiveElements,
        categoryElements: elements,
      },
    }));
  }, [onFormChange]);

  // Get the effective category (detected or default)
  const effectiveCategory: ProfileCategory = formState.detectedCategory?.category || 'education';

  const canPreview = formState.topic.trim().length >= 5;
  // Require context for generation
  const canGenerate = canPreview && formState.profileId && formState.context !== null;

  return (
    <div className="space-y-6">
      {/* Profile Selection */}
      <div className="space-y-2">
        <label className="flex items-center gap-2 text-sm font-medium text-gray-300">
          <BookOpen className="w-4 h-4 text-purple-400" />
          Profil Créateur
        </label>
        <ProfileSelector value={formState.profileId} onChange={handleProfileChange} />
      </div>

      {/* Topic Input */}
      <div className="space-y-2">
        <label className="flex items-center gap-2 text-sm font-medium text-gray-300">
          <Target className="w-4 h-4 text-blue-400" />
          Sujet du Cours
        </label>
        <input
          type="text"
          value={formState.topic}
          onChange={handleTopicChange}
          placeholder="Ex: Techniques de négociation commerciale"
          className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:border-purple-500 focus:ring-1 focus:ring-purple-500 transition-colors"
        />
        <textarea
          value={formState.description}
          onChange={handleDescriptionChange}
          placeholder="Description ou contexte additionnel (optionnel)"
          rows={2}
          className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:border-purple-500 focus:ring-1 focus:ring-purple-500 transition-colors resize-none"
        />
      </div>

      {/* Context Questions Section */}
      {formState.profileId && (
        <div className="border border-gray-700 rounded-lg overflow-hidden">
          <button
            type="button"
            onClick={() => toggleSection('context')}
            className="w-full flex items-center justify-between p-4 bg-gray-800/50 hover:bg-gray-800 transition-colors"
          >
            <span className="flex items-center gap-2 text-white font-medium">
              <MessageSquare className="w-4 h-4 text-purple-400" />
              Questions Contextuelles
              {formState.context && (
                <span className="ml-2 px-2 py-0.5 text-xs bg-green-500/20 text-green-400 rounded">
                  Complété
                </span>
              )}
            </span>
            {expandedSections.context ? (
              <ChevronUp className="w-5 h-5 text-gray-400" />
            ) : (
              <ChevronDown className="w-5 h-5 text-gray-400" />
            )}
          </button>
          {expandedSections.context && (
            <div className="p-4 border-t border-gray-700">
              <ContextQuestionsForm
                profile={selectedProfile}
                topic={formState.topic}
                answers={formState.contextAnswers}
                onAnswersChange={handleAnswersChange}
                onContextChange={handleContextChange}
                detectedCategory={formState.detectedCategory?.category}
                detectedDomain={formState.detectedCategory?.domain}
                detectedDomainOptions={formState.detectedCategory?.domainOptions}
                detectedKeywords={formState.detectedCategory?.keywords}
                isDetecting={isDetectingCategory}
              />
            </div>
          )}
        </div>
      )}

      {/* Source Library Section (Multi-Source RAG) */}
      <div className="border border-gray-700 rounded-lg overflow-hidden">
        <button
          type="button"
          onClick={() => toggleSection('sourceLibrary')}
          className="w-full flex items-center justify-between p-4 bg-gray-800/50 hover:bg-gray-800 transition-colors"
        >
          <span className="flex items-center gap-2 text-white font-medium">
            <Library className="w-4 h-4 text-purple-400" />
            Bibliothèque de Sources
            {linkedSourceIds.length > 0 && (
              <span className="ml-2 px-2 py-0.5 text-xs bg-purple-500/20 text-purple-400 rounded">
                {linkedSourceIds.length} source{linkedSourceIds.length > 1 ? 's' : ''} liée{linkedSourceIds.length > 1 ? 's' : ''}
              </span>
            )}
          </span>
          {expandedSections.sourceLibrary ? (
            <ChevronUp className="w-5 h-5 text-gray-400" />
          ) : (
            <ChevronDown className="w-5 h-5 text-gray-400" />
          )}
        </button>
        {expandedSections.sourceLibrary && (
          <div className="p-4 border-t border-gray-700">
            <SourceLibrary
              userId={formState.profileId || 'anonymous'}
              topic={formState.topic}
              onSourcesChange={handleSourcesChange}
            />
          </div>
        )}
      </div>

      {/* Legacy Documents Section (RAG) - Hidden, kept for backward compatibility */}
      {formState.documents.length > 0 && (
        <div className="border border-gray-700 rounded-lg overflow-hidden">
          <button
            type="button"
            onClick={() => toggleSection('documents')}
            className="w-full flex items-center justify-between p-4 bg-gray-800/50 hover:bg-gray-800 transition-colors"
          >
            <span className="flex items-center gap-2 text-white font-medium">
              <FileText className="w-4 h-4 text-cyan-400" />
              Documents (Ancien système)
              <span className="ml-2 px-2 py-0.5 text-xs bg-cyan-500/20 text-cyan-400 rounded">
                {formState.documents.length} doc{formState.documents.length > 1 ? 's' : ''}
              </span>
            </span>
            {expandedSections.documents ? (
              <ChevronUp className="w-5 h-5 text-gray-400" />
            ) : (
              <ChevronDown className="w-5 h-5 text-gray-400" />
            )}
          </button>
          {expandedSections.documents && (
            <div className="p-4 border-t border-gray-700">
              <DocumentUpload
                userId={formState.profileId || 'anonymous'}
                courseId={undefined}
                documents={formState.documents}
                onDocumentsChange={handleDocumentsChange}
                maxDocuments={10}
              />
            </div>
          )}
        </div>
      )}

      {/* Difficulty Range */}
      <div className="space-y-2">
        <label className="flex items-center gap-2 text-sm font-medium text-gray-300">
          <Target className="w-4 h-4 text-yellow-400" />
          Plage de Difficulté
        </label>
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
          <DifficultySelector
            startValue={formState.difficultyStart}
            endValue={formState.difficultyEnd}
            onStartChange={(v) => onFormChange({ ...formState, difficultyStart: v })}
            onEndChange={(v) => onFormChange({ ...formState, difficultyEnd: v })}
          />
        </div>
      </div>

      {/* Structure Section */}
      <div className="border border-gray-700 rounded-lg overflow-hidden">
        <button
          type="button"
          onClick={() => toggleSection('structure')}
          className="w-full flex items-center justify-between p-4 bg-gray-800/50 hover:bg-gray-800 transition-colors"
        >
          <span className="flex items-center gap-2 text-white font-medium">
            <Settings2 className="w-4 h-4 text-green-400" />
            Structure du Cours
          </span>
          {expandedSections.structure ? (
            <ChevronUp className="w-5 h-5 text-gray-400" />
          ) : (
            <ChevronDown className="w-5 h-5 text-gray-400" />
          )}
        </button>
        {expandedSections.structure && (
          <div className="p-4 border-t border-gray-700">
            <StructureConfig
              value={formState.structure}
              onChange={(v) => onFormChange({ ...formState, structure: v })}
            />
          </div>
        )}
      </div>

      {/* Detected Category Display + Custom Keywords */}
      {formState.topic.trim().length >= 5 && (
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Brain className="w-4 h-4 text-purple-400" />
              <span className="text-sm font-medium text-gray-300">Domaine technique principal</span>
            </div>
            {isDetectingCategory ? (
              <div className="flex items-center gap-2 text-purple-400">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span className="text-sm">Détection...</span>
              </div>
            ) : formState.detectedCategory ? (
              <div className="flex items-center gap-2">
                <span className="text-xl">{CATEGORY_LABELS[formState.detectedCategory.category]?.icon}</span>
                <span className="text-white font-medium">
                  {CATEGORY_LABELS[formState.detectedCategory.category]?.label}
                </span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-400">
                  {Math.round(formState.detectedCategory.confidence * 100)}% confiance
                </span>
              </div>
            ) : (
              <span className="text-sm text-gray-500">En attente de sujet...</span>
            )}
          </div>

          {/* Custom Keywords Input */}
          <div className="space-y-2 pt-2 border-t border-gray-700">
            <label className="text-sm text-gray-400">
              Mots-clés personnalisés (max 5) - Affinez le contexte de la formation
            </label>
            <KeywordsInput
              keywords={formState.customKeywords}
              onChange={handleKeywordsChange}
              placeholder="Ex: React, TypeScript, API REST..."
              suggestions={formState.detectedCategory?.keywords || []}
            />
          </div>
        </div>
      )}

      {/* Adaptive Lesson Elements Section */}
      <div className="border border-gray-700 rounded-lg overflow-hidden">
        <button
          type="button"
          onClick={() => toggleSection('elements')}
          className="w-full flex items-center justify-between p-4 bg-gray-800/50 hover:bg-gray-800 transition-colors"
        >
          <span className="flex items-center gap-2 text-white font-medium">
            <Sparkles className="w-4 h-4 text-purple-400" />
            Éléments de Leçon Adaptatifs
            {formState.detectedCategory && (
              <span className="ml-2 px-2 py-0.5 text-xs bg-purple-500/20 text-purple-400 rounded">
                {CATEGORY_LABELS[formState.detectedCategory.category]?.label}
              </span>
            )}
          </span>
          {expandedSections.elements ? (
            <ChevronUp className="w-5 h-5 text-gray-400" />
          ) : (
            <ChevronDown className="w-5 h-5 text-gray-400" />
          )}
        </button>
        {expandedSections.elements && (
          <div className="p-4 border-t border-gray-700">
            <AdaptiveLessonElements
              category={effectiveCategory}
              topic={formState.topic}
              selectedElements={formState.adaptiveElements.categoryElements}
              onElementsChange={handleAdaptiveElementsChange}
              useAiSuggestions={formState.adaptiveElements.useAiSuggestions}
            />
          </div>
        )}
      </div>

      {/* Advanced Options */}
      <div className="border border-gray-700 rounded-lg overflow-hidden">
        <button
          type="button"
          onClick={() => toggleSection('advanced')}
          className="w-full flex items-center justify-between p-4 bg-gray-800/50 hover:bg-gray-800 transition-colors"
        >
          <span className="flex items-center gap-2 text-white font-medium">
            <Settings2 className="w-4 h-4 text-gray-400" />
            Options Avancées
          </span>
          {expandedSections.advanced ? (
            <ChevronUp className="w-5 h-5 text-gray-400" />
          ) : (
            <ChevronDown className="w-5 h-5 text-gray-400" />
          )}
        </button>
        {expandedSections.advanced && (
          <div className="p-4 border-t border-gray-700 space-y-4">
            {/* Voice Language Selection */}
            <div className="space-y-2">
              <label className="flex items-center gap-2 text-sm text-gray-400">
                <Globe className="w-4 h-4" />
                Langue de la Voix
              </label>
              <select
                value={selectedVoiceLanguage}
                onChange={(e) => setSelectedVoiceLanguage(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:border-purple-500"
              >
                {SUPPORTED_LANGUAGES.map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.flag} {lang.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Voice Selection */}
            <div className="space-y-2">
              <label className="flex items-center gap-2 text-sm text-gray-400">
                <Mic className="w-4 h-4" />
                Voix
                {isLoadingVoices && (
                  <Loader2 className="w-3 h-3 animate-spin ml-1" />
                )}
              </label>
              <select
                value={formState.voiceId}
                onChange={(e) => onFormChange({ ...formState, voiceId: e.target.value })}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:border-purple-500"
                disabled={isLoadingVoices}
              >
                {voices.length === 0 && !isLoadingVoices ? (
                  <option value="">Aucune voix disponible</option>
                ) : (
                  <>
                    {/* Group by gender */}
                    {['male', 'female', 'neutral'].map((gender) => {
                      const genderVoices = voices.filter((v) => v.gender === gender);
                      if (genderVoices.length === 0) return null;
                      return (
                        <optgroup key={gender} label={getGenderLabel(gender)}>
                          {genderVoices.map((voice) => (
                            <option key={voice.id} value={voice.id}>
                              {voice.name} - {voice.style !== 'default' ? voice.style : voice.description}
                            </option>
                          ))}
                        </optgroup>
                      );
                    })}
                  </>
                )}
              </select>
              {voices.length > 0 && (
                <p className="text-xs text-gray-500">
                  {voices.length} voix disponibles en {getLanguageFlag(selectedVoiceLanguage)} {selectedVoiceLanguage.toUpperCase()}
                </p>
              )}
            </div>

            {/* Style */}
            <div className="space-y-2">
              <label className="text-sm text-gray-400">Style Visuel</label>
              <select
                value={formState.style}
                onChange={(e) => onFormChange({ ...formState, style: e.target.value })}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:border-purple-500"
              >
                <option value="dark">Sombre</option>
                <option value="light">Clair</option>
                <option value="gradient">Dégradé</option>
                <option value="ocean">Océan</option>
              </select>
            </div>

            {/* Typing Speed */}
            <div className="space-y-2">
              <label className="text-sm text-gray-400">Vitesse de Frappe</label>
              <select
                value={formState.typingSpeed}
                onChange={(e) =>
                  onFormChange({ ...formState, typingSpeed: e.target.value })
                }
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:border-purple-500"
              >
                <option value="slow">Lent (Rythme pédagogique)</option>
                <option value="natural">Naturel (Par défaut)</option>
                <option value="moderate">Modéré</option>
                <option value="fast">Rapide</option>
              </select>
            </div>

            {/* Title Style */}
            <div className="space-y-2">
              <label className="text-sm text-gray-400">Style des Titres</label>
              <select
                value={formState.titleStyle}
                onChange={(e) =>
                  onFormChange({ ...formState, titleStyle: e.target.value as any })
                }
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:border-purple-500"
              >
                <option value="engaging">Engageant (Créateurs de contenu)</option>
                <option value="corporate">Corporate (Formation entreprise)</option>
                <option value="mentor">Mentor (Pédagogique)</option>
                <option value="expert">Expert (Technique précis)</option>
                <option value="storyteller">Narratif (Tutoriels)</option>
                <option value="direct">Direct (Documentation)</option>
              </select>
              <p className="text-xs text-gray-500">
                Détermine le ton et le style des titres de slides
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3 pt-4">
        <button
          onClick={onPreview}
          disabled={!canPreview || isPreviewLoading || isGenerating}
          className="flex-1 flex items-center justify-center gap-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-500 text-white font-medium py-3 px-4 rounded-lg transition-colors"
        >
          {isPreviewLoading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Eye className="w-5 h-5" />
          )}
          {hasPreview ? 'Mettre à jour' : 'Prévisualiser'}
        </button>

        <button
          onClick={onGenerate}
          disabled={!canGenerate || isGenerating}
          className="flex-1 flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-800 disabled:text-gray-500 text-white font-medium py-3 px-4 rounded-lg transition-colors"
        >
          {isGenerating ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Rocket className="w-5 h-5" />
          )}
          {isGenerating ? 'Génération...' : 'Générer le Cours'}
        </button>
      </div>

      {/* Validation hint */}
      {!canGenerate && formState.profileId && formState.topic.length >= 5 && (
        <p className="text-sm text-yellow-400 text-center">
          Répondez aux questions contextuelles pour générer le cours
        </p>
      )}
    </div>
  );
}
