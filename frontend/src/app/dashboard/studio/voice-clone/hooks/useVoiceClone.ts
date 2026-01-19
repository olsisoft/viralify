'use client';

/**
 * Voice Clone Hook
 * Manages voice cloning state and API interactions
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import {
  VoiceProfile,
  VoiceSample,
  VoiceProfileStatus,
  VoiceGenerationSettings,
  VoiceSampleRequirements,
  CreateVoiceProfileRequest,
  TrainingRequirements,
  defaultGenerationSettings,
} from '../lib/voice-types';

const MEDIA_API_URL = process.env.NEXT_PUBLIC_MEDIA_API_URL || 'http://localhost:8004';

interface UseVoiceCloneOptions {
  userId?: string;
  onError?: (error: string) => void;
}

interface UseVoiceCloneReturn {
  // State
  profiles: VoiceProfile[];
  selectedProfile: VoiceProfile | null;
  requirements: VoiceSampleRequirements | null;
  isLoading: boolean;
  isSaving: boolean;
  isTraining: boolean;
  isGenerating: boolean;
  error: string | null;

  // Profile actions
  loadProfiles: () => Promise<void>;
  createProfile: (request: CreateVoiceProfileRequest) => Promise<string | null>;
  selectProfile: (profileId: string) => Promise<void>;
  updateProfile: (profileId: string, updates: Partial<VoiceProfile>) => Promise<void>;
  deleteProfile: (profileId: string) => Promise<boolean>;

  // Sample actions
  uploadSample: (profileId: string, file: File) => Promise<boolean>;
  deleteSample: (profileId: string, sampleId: string) => Promise<boolean>;

  // Training actions
  startTraining: (profileId: string, consentConfirmed: boolean) => Promise<boolean>;
  checkTrainingStatus: (profileId: string) => Promise<void>;

  // Generation actions
  generateSpeech: (profileId: string, text: string, settings?: Partial<VoiceGenerationSettings>) => Promise<string | null>;
  previewVoice: (profileId: string, text?: string) => Promise<string | null>;

  // Utility
  clearError: () => void;
  getTrainingRequirements: (profile: VoiceProfile) => TrainingRequirements | null;
}

export function useVoiceClone(options: UseVoiceCloneOptions = {}): UseVoiceCloneReturn {
  const { userId = 'demo-user', onError } = options;

  const [profiles, setProfiles] = useState<VoiceProfile[]>([]);
  const [selectedProfile, setSelectedProfile] = useState<VoiceProfile | null>(null);
  const [requirements, setRequirements] = useState<VoiceSampleRequirements | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;

  const handleError = useCallback((message: string) => {
    setError(message);
    onErrorRef.current?.(message);
  }, []);

  // Fetch requirements on mount
  useEffect(() => {
    async function fetchRequirements() {
      try {
        const response = await fetch(`${MEDIA_API_URL}/api/v1/voice/requirements`);
        if (response.ok) {
          const data = await response.json();
          setRequirements(data);
        }
      } catch (e) {
        console.error('Failed to fetch voice requirements:', e);
      }
    }
    fetchRequirements();
  }, []);

  // ========================================
  // Profile Actions
  // ========================================

  const loadProfiles = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${MEDIA_API_URL}/api/v1/voice/profiles?user_id=${userId}`);

      if (!response.ok) {
        throw new Error('Failed to load profiles');
      }

      const data = await response.json();
      setProfiles(data.profiles || []);
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to load profiles');
    } finally {
      setIsLoading(false);
    }
  }, [userId, handleError]);

  const createProfile = useCallback(async (request: CreateVoiceProfileRequest): Promise<string | null> => {
    setIsSaving(true);
    setError(null);

    try {
      const response = await fetch(`${MEDIA_API_URL}/api/v1/voice/profiles?user_id=${userId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to create profile');
      }

      const data = await response.json();
      await loadProfiles();
      return data.profile_id;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to create profile');
      return null;
    } finally {
      setIsSaving(false);
    }
  }, [userId, handleError, loadProfiles]);

  const selectProfile = useCallback(async (profileId: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/voice/profiles/${profileId}?user_id=${userId}`
      );

      if (!response.ok) {
        throw new Error('Profile not found');
      }

      const data = await response.json();
      setSelectedProfile(data.profile);
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to load profile');
    } finally {
      setIsLoading(false);
    }
  }, [userId, handleError]);

  const updateProfile = useCallback(async (profileId: string, updates: Partial<VoiceProfile>) => {
    setIsSaving(true);

    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/voice/profiles/${profileId}?user_id=${userId}`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(updates),
        }
      );

      if (!response.ok) {
        throw new Error('Failed to update profile');
      }

      const data = await response.json();
      setSelectedProfile(data);
      await loadProfiles();
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to update profile');
    } finally {
      setIsSaving(false);
    }
  }, [userId, handleError, loadProfiles]);

  const deleteProfile = useCallback(async (profileId: string): Promise<boolean> => {
    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/voice/profiles/${profileId}?user_id=${userId}`,
        { method: 'DELETE' }
      );

      if (!response.ok) {
        throw new Error('Failed to delete profile');
      }

      if (selectedProfile?.id === profileId) {
        setSelectedProfile(null);
      }
      await loadProfiles();
      return true;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to delete profile');
      return false;
    }
  }, [userId, selectedProfile?.id, handleError, loadProfiles]);

  // ========================================
  // Sample Actions
  // ========================================

  const uploadSample = useCallback(async (profileId: string, file: File): Promise<boolean> => {
    setIsSaving(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('user_id', userId);

      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/voice/profiles/${profileId}/samples`,
        {
          method: 'POST',
          body: formData,
        }
      );

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to upload sample');
      }

      // Refresh profile
      await selectProfile(profileId);
      return true;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to upload sample');
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [userId, handleError, selectProfile]);

  const deleteSample = useCallback(async (profileId: string, sampleId: string): Promise<boolean> => {
    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/voice/profiles/${profileId}/samples/${sampleId}?user_id=${userId}`,
        { method: 'DELETE' }
      );

      if (!response.ok) {
        throw new Error('Failed to delete sample');
      }

      await selectProfile(profileId);
      return true;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Failed to delete sample');
      return false;
    }
  }, [userId, handleError, selectProfile]);

  // ========================================
  // Training Actions
  // ========================================

  const startTraining = useCallback(async (profileId: string, consentConfirmed: boolean): Promise<boolean> => {
    if (!consentConfirmed) {
      handleError('Voice ownership consent is required');
      return false;
    }

    setIsTraining(true);
    setError(null);

    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/voice/profiles/${profileId}/train?user_id=${userId}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            profile_id: profileId,
            consent_confirmed: consentConfirmed,
          }),
        }
      );

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Training failed');
      }

      await selectProfile(profileId);
      await loadProfiles();
      return true;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Training failed');
      return false;
    } finally {
      setIsTraining(false);
    }
  }, [userId, handleError, selectProfile, loadProfiles]);

  const checkTrainingStatus = useCallback(async (profileId: string) => {
    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/voice/profiles/${profileId}/training-status?user_id=${userId}`
      );

      if (response.ok) {
        const data = await response.json();
        if (selectedProfile?.id === profileId) {
          setSelectedProfile((prev) =>
            prev ? { ...prev, status: data.status, training_progress: data.progress } : prev
          );
        }
      }
    } catch (e) {
      console.error('Failed to check training status:', e);
    }
  }, [userId, selectedProfile?.id]);

  // ========================================
  // Generation Actions
  // ========================================

  const generateSpeech = useCallback(async (
    profileId: string,
    text: string,
    settings?: Partial<VoiceGenerationSettings>
  ): Promise<string | null> => {
    setIsGenerating(true);
    setError(null);

    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/voice/profiles/${profileId}/generate?user_id=${userId}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            profile_id: profileId,
            text,
            settings: settings ? { ...defaultGenerationSettings, ...settings } : undefined,
          }),
        }
      );

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Generation failed');
      }

      const data = await response.json();
      return data.audio_url;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Generation failed');
      return null;
    } finally {
      setIsGenerating(false);
    }
  }, [userId, handleError]);

  const previewVoice = useCallback(async (
    profileId: string,
    text?: string
  ): Promise<string | null> => {
    setIsGenerating(true);
    setError(null);

    try {
      const response = await fetch(
        `${MEDIA_API_URL}/api/v1/voice/profiles/${profileId}/preview?user_id=${userId}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(text ? { profile_id: profileId, text } : {}),
        }
      );

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Preview failed');
      }

      const data = await response.json();
      return data.audio_url;
    } catch (e) {
      handleError(e instanceof Error ? e.message : 'Preview failed');
      return null;
    } finally {
      setIsGenerating(false);
    }
  }, [userId, handleError]);

  // ========================================
  // Utility
  // ========================================

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const getTrainingRequirements = useCallback((profile: VoiceProfile): TrainingRequirements | null => {
    if (!requirements) return null;

    const validatedSamples = profile.samples.filter((s) => s.status === 'validated');
    const canTrain =
      validatedSamples.length >= requirements.min_samples &&
      profile.total_sample_duration >= requirements.min_duration_seconds;

    return {
      min_samples: requirements.min_samples,
      min_duration_seconds: requirements.min_duration_seconds,
      max_duration_seconds: requirements.max_duration_seconds,
      ideal_duration_seconds: requirements.ideal_duration_seconds,
      current_samples: validatedSamples.length,
      current_duration: profile.total_sample_duration,
      can_train: canTrain,
      training_message: canTrain
        ? 'Ready to start training!'
        : `Need ${requirements.min_duration_seconds - profile.total_sample_duration}s more audio`,
      progress_percent: Math.min(100, (profile.total_sample_duration / requirements.ideal_duration_seconds) * 100),
      requirements,
    };
  }, [requirements]);

  return {
    // State
    profiles,
    selectedProfile,
    requirements,
    isLoading,
    isSaving,
    isTraining,
    isGenerating,
    error,

    // Profile actions
    loadProfiles,
    createProfile,
    selectProfile,
    updateProfile,
    deleteProfile,

    // Sample actions
    uploadSample,
    deleteSample,

    // Training actions
    startTraining,
    checkTrainingStatus,

    // Generation actions
    generateSpeech,
    previewVoice,

    // Utility
    clearError,
    getTrainingRequirements,
  };
}
