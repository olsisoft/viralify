/**
 * useTraceability Hook
 *
 * Fetches traceability data for a course, including:
 * - Source usage per slide
 * - Knowledge graph concepts
 * - Cross-reference analysis
 */

import { useState, useCallback } from 'react';
import type {
  TraceabilityResponse,
  LectureTraceability,
  KnowledgeGraphResponse,
  CrossReferenceReport,
  Concept,
} from '../lib/traceability-types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';

interface UseTraceabilityResult {
  // Data
  traceability: TraceabilityResponse | null;
  knowledgeGraph: KnowledgeGraphResponse | null;
  crossReferences: CrossReferenceReport | null;
  selectedConcept: Concept | null;

  // Loading states
  isLoading: boolean;
  isLoadingKnowledgeGraph: boolean;
  isLoadingCrossReferences: boolean;
  isLoadingConcept: boolean;

  // Error
  error: string | null;

  // Actions
  fetchTraceability: (jobId: string) => Promise<void>;
  fetchLectureTraceability: (jobId: string, lectureId: string) => Promise<LectureTraceability | null>;
  fetchKnowledgeGraph: (jobId: string) => Promise<void>;
  fetchCrossReferences: (jobId: string) => Promise<void>;
  fetchConceptDetails: (jobId: string, conceptId: string) => Promise<void>;
  clearError: () => void;
}

export function useTraceability(): UseTraceabilityResult {
  const [traceability, setTraceability] = useState<TraceabilityResponse | null>(null);
  const [knowledgeGraph, setKnowledgeGraph] = useState<KnowledgeGraphResponse | null>(null);
  const [crossReferences, setCrossReferences] = useState<CrossReferenceReport | null>(null);
  const [selectedConcept, setSelectedConcept] = useState<Concept | null>(null);

  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingKnowledgeGraph, setIsLoadingKnowledgeGraph] = useState(false);
  const [isLoadingCrossReferences, setIsLoadingCrossReferences] = useState(false);
  const [isLoadingConcept, setIsLoadingConcept] = useState(false);

  const [error, setError] = useState<string | null>(null);

  const fetchTraceability = useCallback(async (jobId: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/api/v1/courses/${jobId}/traceability`);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to fetch traceability: ${response.status}`);
      }

      const data: TraceabilityResponse = await response.json();
      setTraceability(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch traceability';
      setError(message);
      console.error('[useTraceability] Error:', message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const fetchLectureTraceability = useCallback(async (
    jobId: string,
    lectureId: string
  ): Promise<LectureTraceability | null> => {
    try {
      const response = await fetch(
        `${API_URL}/api/v1/courses/${jobId}/lectures/${lectureId}/traceability`
      );

      if (!response.ok) {
        console.error('[useTraceability] Failed to fetch lecture traceability');
        return null;
      }

      const data = await response.json();
      return data.traceability;
    } catch (err) {
      console.error('[useTraceability] Error fetching lecture traceability:', err);
      return null;
    }
  }, []);

  const fetchKnowledgeGraph = useCallback(async (jobId: string) => {
    setIsLoadingKnowledgeGraph(true);

    try {
      const response = await fetch(`${API_URL}/api/v1/courses/${jobId}/knowledge-graph`);

      if (!response.ok) {
        console.error('[useTraceability] Failed to fetch knowledge graph');
        return;
      }

      const data: KnowledgeGraphResponse = await response.json();
      setKnowledgeGraph(data);
    } catch (err) {
      console.error('[useTraceability] Error fetching knowledge graph:', err);
    } finally {
      setIsLoadingKnowledgeGraph(false);
    }
  }, []);

  const fetchCrossReferences = useCallback(async (jobId: string) => {
    setIsLoadingCrossReferences(true);

    try {
      const response = await fetch(`${API_URL}/api/v1/courses/${jobId}/cross-references`);

      if (!response.ok) {
        console.error('[useTraceability] Failed to fetch cross-references');
        return;
      }

      const data: CrossReferenceReport = await response.json();
      setCrossReferences(data);
    } catch (err) {
      console.error('[useTraceability] Error fetching cross-references:', err);
    } finally {
      setIsLoadingCrossReferences(false);
    }
  }, []);

  const fetchConceptDetails = useCallback(async (jobId: string, conceptId: string) => {
    setIsLoadingConcept(true);

    try {
      const response = await fetch(
        `${API_URL}/api/v1/courses/${jobId}/knowledge-graph/concept/${conceptId}`
      );

      if (!response.ok) {
        console.error('[useTraceability] Failed to fetch concept details');
        return;
      }

      const data = await response.json();
      setSelectedConcept(data.concept);
    } catch (err) {
      console.error('[useTraceability] Error fetching concept details:', err);
    } finally {
      setIsLoadingConcept(false);
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    traceability,
    knowledgeGraph,
    crossReferences,
    selectedConcept,
    isLoading,
    isLoadingKnowledgeGraph,
    isLoadingCrossReferences,
    isLoadingConcept,
    error,
    fetchTraceability,
    fetchLectureTraceability,
    fetchKnowledgeGraph,
    fetchCrossReferences,
    fetchConceptDetails,
    clearError,
  };
}
