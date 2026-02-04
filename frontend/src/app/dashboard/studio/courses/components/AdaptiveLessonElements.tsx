'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Check,
  CheckCircle2,
  Lock,
  Sparkles,
  Loader2,
  ChevronDown,
  ChevronUp,
  RefreshCw,
} from 'lucide-react';
import type { ProfileCategory } from '../lib/course-types';
import type {
  LessonElement,
  LessonElementType,
  CategoryElementsResponse,
  ElementSuggestionResponse,
} from '../lib/lesson-elements';
import { CATEGORIES, getCategoryInfo } from '../lib/lesson-elements';

interface AdaptiveLessonElementsProps {
  category: ProfileCategory;
  topic: string;
  selectedElements: Record<string, boolean>;
  onElementsChange: (elements: Record<string, boolean>) => void;
  useAiSuggestions?: boolean;
}

export function AdaptiveLessonElements({
  category,
  topic,
  selectedElements,
  onElementsChange,
  useAiSuggestions = true,
}: AdaptiveLessonElementsProps) {
  // Ensure selectedElements is always an object (defensive against undefined)
  const safeSelectedElements = selectedElements ?? {};

  const [commonElements, setCommonElements] = useState<LessonElement[]>([]);
  const [categoryElements, setCategoryElements] = useState<LessonElement[]>([]);
  const [aiSuggestions, setAiSuggestions] = useState<Record<string, { confidence: number; reason: string }>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingAi, setIsLoadingAi] = useState(false);
  const [showCategoryElements, setShowCategoryElements] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hasAutoSuggested, setHasAutoSuggested] = useState(false);
  const suggestTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastTopicRef = useRef<string>('');
  const lastCategoryRef = useRef<string>('');

  const categoryInfo = getCategoryInfo(category);

  // Fetch elements for category
  useEffect(() => {
    const fetchElements = async () => {
      setIsLoading(true);
      setError(null);
      // Reset AI suggestions when category changes
      setHasAutoSuggested(false);
      setAiSuggestions({});

      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || '';
        const response = await fetch(
          `${apiUrl}/api/v1/courses/config/elements/${category}`
        );

        if (!response.ok) {
          throw new Error('Failed to fetch elements');
        }

        const data = await response.json();
        console.log('[AdaptiveLessonElements] Elements received:', data);

        // API returns snake_case, convert to camelCase
        const commonEls = (data.common_elements || []).map((el: any) => ({
          id: el.id,
          name: el.name,
          description: el.description,
          icon: el.icon,
          isRequired: el.is_required,
          enabled: el.enabled,
        }));
        const categoryEls = (data.category_elements || []).map((el: any) => ({
          id: el.id,
          name: el.name,
          description: el.description,
          icon: el.icon,
          isRequired: el.is_required,
          enabled: el.enabled,
        }));

        setCommonElements(commonEls);
        setCategoryElements(categoryEls);

        // Initialize selected elements if empty or category changed
        const defaults: Record<string, boolean> = {};
        commonEls.forEach((el: any) => {
          defaults[el.id] = el.enabled;
        });
        categoryEls.forEach((el: any) => {
          defaults[el.id] = el.enabled;
        });
        onElementsChange(defaults);
      } catch (err) {
        console.error('Error fetching elements:', err);
        setError('Erreur lors du chargement des éléments');
      } finally {
        setIsLoading(false);
      }
    };

    fetchElements();
  }, [category]);

  // Fetch AI suggestions
  const fetchAiSuggestions = useCallback(async () => {
    if (!useAiSuggestions || !topic || topic.length < 5) return;

    setIsLoadingAi(true);
    try {
      // Use POST with query params as the endpoint expects
      const params = new URLSearchParams({
        topic,
        category,
      });

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || '';
      const response = await fetch(
        `${apiUrl}/api/v1/courses/config/suggest-elements?${params}`,
        { method: 'POST' }
      );

      if (!response.ok) {
        console.error('Failed to fetch suggestions:', response.status);
        throw new Error('Failed to fetch suggestions');
      }

      const data = await response.json();
      console.log('[AdaptiveLessonElements] AI suggestions received:', data);

      // Build suggestions map (API returns snake_case: element_id, not elementId)
      const suggestionsMap: Record<string, { confidence: number; reason: string }> = {};
      if (data.suggestions && Array.isArray(data.suggestions)) {
        data.suggestions.forEach((s: any) => {
          suggestionsMap[s.element_id] = {
            confidence: s.confidence,
            reason: s.reason,
          };
        });
      }
      setAiSuggestions(suggestionsMap);

      // Update selected elements based on AI suggestions
      const newSelected = { ...safeSelectedElements };
      if (data.suggestions && Array.isArray(data.suggestions)) {
        data.suggestions.forEach((s: any) => {
          if (s.enabled !== undefined) {
            newSelected[s.element_id] = s.enabled;
          }
        });
      }
      onElementsChange(newSelected);
      setHasAutoSuggested(true);
    } catch (err) {
      console.error('Error fetching AI suggestions:', err);
    } finally {
      setIsLoadingAi(false);
    }
  }, [topic, category, useAiSuggestions]);

  // Auto-fetch AI suggestions when topic or category changes (with debounce)
  useEffect(() => {
    // Clear previous timeout
    if (suggestTimeoutRef.current) {
      clearTimeout(suggestTimeoutRef.current);
    }

    // Only auto-fetch if topic is long enough and has changed
    if (!useAiSuggestions || !topic || topic.length < 5) {
      return;
    }

    // Debounce the suggestion fetch (wait for user to stop typing)
    suggestTimeoutRef.current = setTimeout(() => {
      // Skip if topic hasn't meaningfully changed (within 3 char difference, same prefix)
      const normalizedTopic = topic.trim().toLowerCase();
      const lastNormalized = lastTopicRef.current.trim().toLowerCase();

      const topicSimilar = lastNormalized &&
        Math.abs(normalizedTopic.length - lastNormalized.length) <= 3 &&
        (normalizedTopic.startsWith(lastNormalized) || lastNormalized.startsWith(normalizedTopic));

      if (topicSimilar && category === lastCategoryRef.current) {
        console.log('[AdaptiveLessonElements] Skipping suggestion - topic too similar:', topic);
        return;
      }

      lastTopicRef.current = topic;
      lastCategoryRef.current = category;
      fetchAiSuggestions();
    }, 1500); // Increased from 1000ms to 1500ms for more stable typing

    return () => {
      if (suggestTimeoutRef.current) {
        clearTimeout(suggestTimeoutRef.current);
      }
    };
  }, [topic, category, useAiSuggestions, fetchAiSuggestions]);

  const toggleElement = (elementId: string, isRequired: boolean) => {
    if (isRequired) return;
    onElementsChange({
      ...safeSelectedElements,
      [elementId]: !safeSelectedElements[elementId],
    });
  };

  const renderElement = (element: LessonElement, isAiSuggested: boolean = false) => {
    const isEnabled = safeSelectedElements[element.id] ?? element.enabled;
    const isRequired = element.isRequired;
    const suggestion = aiSuggestions[element.id];

    return (
      <button
        key={element.id}
        type="button"
        onClick={() => toggleElement(element.id, isRequired)}
        disabled={isRequired}
        className={`w-full flex items-center gap-3 p-3 rounded-lg border transition-all ${
          isEnabled
            ? 'bg-purple-600/10 border-purple-500/50 text-white'
            : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:border-gray-600'
        } ${isRequired ? 'cursor-not-allowed' : 'cursor-pointer'}`}
      >
        <span className="text-xl">{element.icon}</span>

        <div className="flex-1 text-left">
          <div className="flex items-center gap-2">
            <p className="font-medium">{element.name}</p>
            {isAiSuggested && suggestion && suggestion.confidence > 0.7 && (
              <span className="px-1.5 py-0.5 text-xs bg-purple-500/20 text-purple-400 rounded">
                IA recommandé
              </span>
            )}
          </div>
          <p className="text-sm text-gray-500">{element.description}</p>
          {suggestion && suggestion.reason && (
            <p className="text-xs text-purple-400 mt-1">{suggestion.reason}</p>
          )}
        </div>

        {isRequired ? (
          <Lock className="w-5 h-5 text-gray-500" />
        ) : (
          <div
            className={`w-6 h-6 rounded-full border-2 flex items-center justify-center transition-colors ${
              isEnabled ? 'bg-purple-600 border-purple-600' : 'border-gray-600'
            }`}
          >
            {isEnabled && <Check className="w-4 h-4 text-white" />}
          </div>
        )}
      </button>
    );
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="w-6 h-6 text-purple-400 animate-spin" />
        <span className="ml-2 text-gray-400">Chargement des éléments...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400">
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Category Header */}
      {categoryInfo && (
        <div className="flex items-center gap-2 pb-3 border-b border-gray-700">
          <span className="text-2xl">{categoryInfo.icon}</span>
          <div>
            <h4 className="font-medium text-white">Éléments pour {categoryInfo.name}</h4>
            <p className="text-sm text-gray-400">{categoryInfo.description}</p>
          </div>
        </div>
      )}

      {/* AI Suggestions Status */}
      {useAiSuggestions && topic && topic.length >= 5 && (
        <div className="space-y-2">
          {isLoadingAi ? (
            <div className="w-full flex items-center justify-center gap-2 p-3 rounded-lg bg-purple-600/10 border border-purple-500/30 text-purple-400">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Suggestion automatique des éléments en cours...</span>
            </div>
          ) : hasAutoSuggested && Object.keys(aiSuggestions).length > 0 ? (
            <div className="flex items-center justify-between p-2 rounded-lg bg-green-600/10 border border-green-500/30">
              <div className="flex items-center gap-2 text-green-400">
                <CheckCircle2 className="w-5 h-5" />
                <span className="text-sm font-medium">Éléments suggérés par l'IA</span>
                <span className="text-xs text-green-500">({Object.keys(aiSuggestions).length} éléments)</span>
              </div>
              <button
                type="button"
                onClick={fetchAiSuggestions}
                className="flex items-center gap-1 text-xs text-gray-400 hover:text-white transition-colors"
              >
                <RefreshCw className="w-3 h-3" />
                Actualiser
              </button>
            </div>
          ) : (
            <button
              type="button"
              onClick={fetchAiSuggestions}
              disabled={isLoadingAi}
              className="w-full flex items-center justify-center gap-2 p-2 rounded-lg bg-purple-600/10 border border-purple-500/30 text-purple-400 hover:bg-purple-600/20 transition-colors"
            >
              <Sparkles className="w-4 h-4" />
              <span>Suggérer les éléments avec l'IA</span>
            </button>
          )}
        </div>
      )}

      {/* Common Elements (Always visible) */}
      <div className="space-y-2">
        <h5 className="text-sm font-medium text-gray-400 uppercase tracking-wide">
          Éléments communs
        </h5>
        <div className="space-y-2">
          {commonElements.map((el) => renderElement(el, false))}
        </div>
      </div>

      {/* Category-Specific Elements (Collapsible) */}
      {categoryElements.length > 0 && (
        <div className="space-y-2">
          <button
            type="button"
            onClick={() => setShowCategoryElements(!showCategoryElements)}
            className="w-full flex items-center justify-between text-sm font-medium text-gray-400 uppercase tracking-wide hover:text-gray-300"
          >
            <span>Éléments {categoryInfo?.name || category}</span>
            {showCategoryElements ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </button>

          {showCategoryElements && (
            <div className="space-y-2">
              {categoryElements.map((el) =>
                renderElement(el, !!aiSuggestions[el.id])
              )}
            </div>
          )}
        </div>
      )}

      {/* Summary */}
      <div className="pt-3 border-t border-gray-700 text-center space-y-1">
        <p className="text-sm text-gray-500">
          {Object.values(safeSelectedElements).filter(Boolean).length} éléments activés
        </p>
        {hasAutoSuggested && Object.keys(aiSuggestions).length > 0 && (
          <p className="text-xs text-purple-400">
            {Object.keys(aiSuggestions).length} éléments analysés par l'IA
          </p>
        )}
      </div>
    </div>
  );
}
