'use client';

import { useState, useCallback, useRef } from 'react';
import type {
  Source,
  CourseSource,
  SourceSuggestion,
  SourceType,
  SourceStatus,
} from '../lib/source-types';
import {
  mapSourceFromApi,
  mapCourseSourceFromApi,
  mapSuggestionFromApi,
} from '../lib/source-types';

const API_BASE = process.env.NEXT_PUBLIC_COURSE_GENERATOR_URL || 'http://localhost:8007';

interface UseSourceLibraryOptions {
  userId: string;
}

export function useSourceLibrary({ userId }: UseSourceLibraryOptions) {
  // State
  const [sources, setSources] = useState<Source[]>([]);
  const [courseSources, setCourseSources] = useState<CourseSource[]>([]);
  const [suggestions, setSuggestions] = useState<SourceSuggestion[]>([]);
  const [relevantExisting, setRelevantExisting] = useState<Source[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isSuggesting, setIsSuggesting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [totalSources, setTotalSources] = useState(0);

  // Ref to track ongoing operations
  const abortControllerRef = useRef<AbortController | null>(null);

  // ==========================================================================
  // Source Library CRUD
  // ==========================================================================

  const fetchSources = useCallback(async (options?: {
    sourceType?: SourceType;
    status?: SourceStatus;
    tags?: string[];
    search?: string;
    page?: number;
    pageSize?: number;
  }) => {
    setIsLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({ user_id: userId });
      if (options?.sourceType) params.append('source_type', options.sourceType);
      if (options?.status) params.append('status', options.status);
      if (options?.tags?.length) params.append('tags', options.tags.join(','));
      if (options?.search) params.append('search', options.search);
      if (options?.page) params.append('page', options.page.toString());
      if (options?.pageSize) params.append('page_size', options.pageSize.toString());

      const response = await fetch(`${API_BASE}/api/v1/sources?${params}`);
      if (!response.ok) throw new Error('Failed to fetch sources');

      const data = await response.json();
      const mappedSources = (data.sources || []).map(mapSourceFromApi);
      setSources(mappedSources);
      setTotalSources(data.total || 0);

      return mappedSources;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch sources';
      setError(message);
      return [];
    } finally {
      setIsLoading(false);
    }
  }, [userId]);

  const uploadFile = useCallback(async (
    file: File,
    name?: string,
    tags?: string[],
  ): Promise<Source | null> => {
    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('user_id', userId);
      if (name) formData.append('name', name);
      if (tags?.length) formData.append('tags', tags.join(','));

      const response = await fetch(`${API_BASE}/api/v1/sources/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await response.json();
      const source = mapSourceFromApi(data);

      // Add to local state
      setSources(prev => [source, ...prev]);
      setTotalSources(prev => prev + 1);

      return source;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Upload failed';
      setError(message);
      return null;
    } finally {
      setIsUploading(false);
    }
  }, [userId]);

  const createFromUrl = useCallback(async (
    url: string,
    name?: string,
    tags?: string[],
  ): Promise<Source | null> => {
    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('url', url);
      formData.append('user_id', userId);
      if (name) formData.append('name', name);
      if (tags?.length) formData.append('tags', tags.join(','));

      const response = await fetch(`${API_BASE}/api/v1/sources/url`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to create source from URL');
      }

      const data = await response.json();
      const source = mapSourceFromApi(data);

      setSources(prev => [source, ...prev]);
      setTotalSources(prev => prev + 1);

      return source;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to create source';
      setError(message);
      return null;
    } finally {
      setIsUploading(false);
    }
  }, [userId]);

  const createNote = useCallback(async (
    content: string,
    name: string,
    tags?: string[],
  ): Promise<Source | null> => {
    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('content', content);
      formData.append('name', name);
      formData.append('user_id', userId);
      if (tags?.length) formData.append('tags', tags.join(','));

      const response = await fetch(`${API_BASE}/api/v1/sources/note`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to create note');
      }

      const data = await response.json();
      const source = mapSourceFromApi(data);

      setSources(prev => [source, ...prev]);
      setTotalSources(prev => prev + 1);

      return source;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to create note';
      setError(message);
      return null;
    } finally {
      setIsUploading(false);
    }
  }, [userId]);

  const deleteSource = useCallback(async (sourceId: string): Promise<boolean> => {
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE}/api/v1/sources/${sourceId}?user_id=${userId}`,
        { method: 'DELETE' }
      );

      if (!response.ok) throw new Error('Failed to delete source');

      setSources(prev => prev.filter(s => s.id !== sourceId));
      setTotalSources(prev => prev - 1);

      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete source';
      setError(message);
      return false;
    }
  }, [userId]);

  const updateSource = useCallback(async (
    sourceId: string,
    updates: { name?: string; tags?: string[]; noteContent?: string },
  ): Promise<Source | null> => {
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE}/api/v1/sources/${sourceId}?user_id=${userId}`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: updates.name,
            tags: updates.tags,
            note_content: updates.noteContent,
          }),
        }
      );

      if (!response.ok) throw new Error('Failed to update source');

      const data = await response.json();
      const source = mapSourceFromApi(data);

      setSources(prev => prev.map(s => s.id === sourceId ? source : s));

      return source;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to update source';
      setError(message);
      return null;
    }
  }, [userId]);

  // ==========================================================================
  // Course-Source Linking
  // ==========================================================================

  const fetchCourseSources = useCallback(async (courseId: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE}/api/v1/courses/${courseId}/sources?user_id=${userId}`
      );
      if (!response.ok) throw new Error('Failed to fetch course sources');

      const data = await response.json();
      const mapped = (data.sources || []).map(mapCourseSourceFromApi);
      setCourseSources(mapped);

      return mapped;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch course sources';
      setError(message);
      return [];
    } finally {
      setIsLoading(false);
    }
  }, [userId]);

  const linkSourceToCourse = useCallback(async (
    courseId: string,
    sourceId: string,
    isPrimary: boolean = false,
  ): Promise<CourseSource | null> => {
    setError(null);

    try {
      const params = new URLSearchParams({
        source_id: sourceId,
        user_id: userId,
        is_primary: isPrimary.toString(),
      });

      const response = await fetch(
        `${API_BASE}/api/v1/courses/${courseId}/sources?${params}`,
        { method: 'POST' }
      );

      if (!response.ok) throw new Error('Failed to link source');

      const data = await response.json();
      const courseSource = mapCourseSourceFromApi(data);

      setCourseSources(prev => [...prev, courseSource]);

      return courseSource;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to link source';
      setError(message);
      return null;
    }
  }, [userId]);

  const unlinkSourceFromCourse = useCallback(async (
    courseId: string,
    sourceId: string,
  ): Promise<boolean> => {
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE}/api/v1/courses/${courseId}/sources/${sourceId}?user_id=${userId}`,
        { method: 'DELETE' }
      );

      if (!response.ok) throw new Error('Failed to unlink source');

      setCourseSources(prev => prev.filter(cs => cs.sourceId !== sourceId));

      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to unlink source';
      setError(message);
      return false;
    }
  }, [userId]);

  const linkSourcesBulk = useCallback(async (
    courseId: string,
    sourceIds: string[],
  ): Promise<{ linked: number; errors: number }> => {
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE}/api/v1/courses/${courseId}/sources/bulk`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            course_id: courseId,
            source_ids: sourceIds,
            user_id: userId,
          }),
        }
      );

      if (!response.ok) throw new Error('Failed to link sources');

      const data = await response.json();

      // Refresh course sources
      await fetchCourseSources(courseId);

      return {
        linked: data.total_linked || 0,
        errors: data.total_errors || 0,
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to link sources';
      setError(message);
      return { linked: 0, errors: sourceIds.length };
    }
  }, [userId, fetchCourseSources]);

  // ==========================================================================
  // AI Suggestions
  // ==========================================================================

  const suggestSources = useCallback(async (
    topic: string,
    description?: string,
    maxSuggestions: number = 5,
  ) => {
    // Abort previous request if any
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();
    setIsSuggesting(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/v1/sources/suggest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          topic,
          description,
          language: 'fr',
          max_suggestions: maxSuggestions,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) throw new Error('Failed to get suggestions');

      const data = await response.json();

      const mappedSuggestions = (data.suggestions || []).map(mapSuggestionFromApi);
      const mappedExisting = (data.existing_relevant_sources || []).map(mapSourceFromApi);

      setSuggestions(mappedSuggestions);
      setRelevantExisting(mappedExisting);

      return { suggestions: mappedSuggestions, existing: mappedExisting };
    } catch (err) {
      if ((err as Error).name === 'AbortError') return null;

      const message = err instanceof Error ? err.message : 'Failed to get suggestions';
      setError(message);
      return null;
    } finally {
      setIsSuggesting(false);
    }
  }, [userId]);

  const addSuggestionAsSource = useCallback(async (
    suggestion: SourceSuggestion,
  ): Promise<Source | null> => {
    if (suggestion.url) {
      return createFromUrl(
        suggestion.url,
        suggestion.title,
        suggestion.keywords,
      );
    }
    return null;
  }, [createFromUrl]);

  // ==========================================================================
  // Utility functions
  // ==========================================================================

  const clearError = useCallback(() => setError(null), []);

  const getSourceById = useCallback((sourceId: string): Source | undefined => {
    return sources.find(s => s.id === sourceId);
  }, [sources]);

  const getLinkedSourceIds = useCallback((): string[] => {
    return courseSources.map(cs => cs.sourceId);
  }, [courseSources]);

  return {
    // State
    sources,
    courseSources,
    suggestions,
    relevantExisting,
    isLoading,
    isUploading,
    isSuggesting,
    error,
    totalSources,

    // Source Library CRUD
    fetchSources,
    uploadFile,
    createFromUrl,
    createNote,
    deleteSource,
    updateSource,

    // Course-Source Linking
    fetchCourseSources,
    linkSourceToCourse,
    unlinkSourceFromCourse,
    linkSourcesBulk,

    // AI Suggestions
    suggestSources,
    addSuggestionAsSource,

    // Utilities
    clearError,
    getSourceById,
    getLinkedSourceIds,
  };
}
