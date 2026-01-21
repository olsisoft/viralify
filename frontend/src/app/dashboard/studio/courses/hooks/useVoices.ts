'use client';

import { useState, useEffect, useCallback } from 'react';
import type { Voice, VoicesResponse } from '../lib/voice-types';

interface UseVoicesOptions {
  language?: string;
  provider?: 'elevenlabs' | 'openai';
  gender?: 'male' | 'female' | 'neutral';
}

interface UseVoicesReturn {
  voices: Voice[];
  supportedLanguages: string[];
  isLoading: boolean;
  error: string | null;
  selectedLanguage: string;
  setSelectedLanguage: (lang: string) => void;
  refetch: () => Promise<void>;
}

export function useVoices(options: UseVoicesOptions = {}): UseVoicesReturn {
  const { provider = 'elevenlabs', gender } = options;

  const [voices, setVoices] = useState<Voice[]>([]);
  const [supportedLanguages, setSupportedLanguages] = useState<string[]>(['en', 'fr']);
  const [selectedLanguage, setSelectedLanguage] = useState(options.language || 'fr');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchVoices = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        language: selectedLanguage,
        provider: provider,
      });

      if (gender) {
        params.append('gender', gender);
      }

      const apiUrl = process.env.NEXT_PUBLIC_MEDIA_API_URL || process.env.NEXT_PUBLIC_API_URL || '';
      const response = await fetch(`${apiUrl}/api/v1/media/voices?${params}`);

      if (!response.ok) {
        throw new Error(`Failed to fetch voices: ${response.status}`);
      }

      const data: VoicesResponse = await response.json();
      setVoices(data.voices);
      setSupportedLanguages(data.supported_languages);
    } catch (err) {
      console.error('[useVoices] Error fetching voices:', err);
      setError(err instanceof Error ? err.message : 'Failed to load voices');
      // Set default OpenAI voices as fallback
      setVoices([
        { id: 'alloy', name: 'Alloy', provider: 'openai', gender: 'neutral', language: 'en', style: 'default', description: 'Voix neutre polyvalente' },
        { id: 'echo', name: 'Echo', provider: 'openai', gender: 'male', language: 'en', style: 'default', description: 'Voix masculine' },
        { id: 'fable', name: 'Fable', provider: 'openai', gender: 'neutral', language: 'en', style: 'british', description: 'Accent britannique' },
        { id: 'onyx', name: 'Onyx', provider: 'openai', gender: 'male', language: 'en', style: 'deep', description: 'Voix grave' },
        { id: 'nova', name: 'Nova', provider: 'openai', gender: 'female', language: 'en', style: 'default', description: 'Voix féminine' },
        { id: 'shimmer', name: 'Shimmer', provider: 'openai', gender: 'female', language: 'en', style: 'soft', description: 'Voix douce' },
      ]);
    } finally {
      setIsLoading(false);
    }
  }, [selectedLanguage, provider, gender]);

  useEffect(() => {
    fetchVoices();
  }, [fetchVoices]);

  return {
    voices,
    supportedLanguages,
    isLoading,
    error,
    selectedLanguage,
    setSelectedLanguage,
    refetch: fetchVoices,
  };
}
