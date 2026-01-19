/**
 * Voice Cloning Types
 * Phase 4: Voice Cloning feature
 */

// ========================================
// Enums
// ========================================

export type VoiceProvider = 'elevenlabs' | 'resemble' | 'coqui';

export type SampleStatus = 'pending' | 'uploading' | 'processing' | 'validated' | 'rejected' | 'error';

export type VoiceProfileStatus = 'draft' | 'training' | 'ready' | 'failed' | 'suspended';

export type VoiceGender = 'male' | 'female' | 'neutral';

export type VoiceAge = 'young' | 'middle' | 'mature';

export type VoiceAccent = 'american' | 'british' | 'australian' | 'indian' | 'french' | 'spanish' | 'german' | 'other';

// ========================================
// Models
// ========================================

export interface VoiceSample {
  id: string;
  profile_id: string;
  user_id: string;
  filename: string;
  file_path: string;
  file_size_bytes: number;
  duration_seconds: number;
  format: string;
  quality_score?: number;
  noise_level?: number;
  clarity_score?: number;
  status: SampleStatus;
  rejection_reason?: string;
  transcript?: string;
  is_transcript_verified: boolean;
  created_at: string;
  processed_at?: string;
}

export interface VoiceProfile {
  id: string;
  user_id: string;
  name: string;
  description?: string;
  gender: VoiceGender;
  age: VoiceAge;
  accent: VoiceAccent;
  language: string;
  samples: VoiceSample[];
  total_sample_duration: number;
  provider: VoiceProvider;
  provider_voice_id?: string;
  provider_model_id?: string;
  status: VoiceProfileStatus;
  training_progress: number;
  error_message?: string;
  default_stability: number;
  default_similarity: number;
  default_style: number;
  consent_given: boolean;
  consent_timestamp?: string;
  total_characters_generated: number;
  total_generations: number;
  last_used_at?: string;
  created_at: string;
  updated_at: string;
  trained_at?: string;
}

export interface VoiceGenerationSettings {
  stability: number;
  similarity_boost: number;
  style: number;
  use_speaker_boost: boolean;
  speed: number;
  emotion?: string;
  output_format: string;
}

export interface VoiceSampleRequirements {
  min_samples: number;
  max_samples: number;
  min_duration_seconds: number;
  max_duration_seconds: number;
  ideal_duration_seconds: number;
  supported_formats: string[];
  max_file_size_mb: number;
  sample_rate_hz: number;
  tips: string[];
}

// ========================================
// Request/Response Types
// ========================================

export interface CreateVoiceProfileRequest {
  name: string;
  description?: string;
  gender?: VoiceGender;
  age?: VoiceAge;
  accent?: VoiceAccent;
  language?: string;
  provider?: VoiceProvider;
}

export interface CreateVoiceProfileResponse {
  profile_id: string;
  name: string;
  status: VoiceProfileStatus;
  message: string;
  min_samples_required: number;
  min_duration_seconds: number;
}

export interface UploadSampleResponse {
  sample_id: string;
  profile_id: string;
  duration_seconds: number;
  quality_score?: number;
  status: SampleStatus;
  message: string;
  total_duration: number;
  can_start_training: boolean;
}

export interface StartTrainingRequest {
  profile_id: string;
  consent_confirmed: boolean;
}

export interface StartTrainingResponse {
  profile_id: string;
  status: VoiceProfileStatus;
  estimated_time_seconds: number;
  message: string;
}

export interface GenerateClonedSpeechRequest {
  profile_id: string;
  text: string;
  settings?: Partial<VoiceGenerationSettings>;
}

export interface GenerateClonedSpeechResponse {
  audio_url: string;
  duration_seconds: number;
  characters_used: number;
  profile_id: string;
}

export interface VoiceProfileListResponse {
  profiles: VoiceProfile[];
  total: number;
}

export interface VoiceProfileDetailResponse {
  profile: VoiceProfile;
  samples: VoiceSample[];
  can_train: boolean;
  training_requirements: TrainingRequirements;
}

export interface TrainingRequirements {
  min_samples: number;
  min_duration_seconds: number;
  max_duration_seconds: number;
  ideal_duration_seconds: number;
  current_samples?: number;
  current_duration?: number;
  can_train?: boolean;
  training_message?: string;
  progress_percent?: number;
  requirements: VoiceSampleRequirements;
}

export interface VoiceUsageStats {
  character_count: number;
  character_limit: number;
  voice_limit: number;
  tier: string;
  error?: string;
}

// ========================================
// UI State Types
// ========================================

export interface VoiceCloneState {
  profiles: VoiceProfile[];
  selectedProfileId: string | null;
  isLoading: boolean;
  isSaving: boolean;
  isTraining: boolean;
  isGenerating: boolean;
  error: string | null;
}

// ========================================
// Helper Functions
// ========================================

export function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function getStatusLabel(status: VoiceProfileStatus): string {
  const labels: Record<VoiceProfileStatus, string> = {
    draft: 'Collecting Samples',
    training: 'Training...',
    ready: 'Ready',
    failed: 'Failed',
    suspended: 'Suspended',
  };
  return labels[status] || status;
}

export function getStatusColor(status: VoiceProfileStatus): string {
  const colors: Record<VoiceProfileStatus, string> = {
    draft: 'yellow',
    training: 'blue',
    ready: 'green',
    failed: 'red',
    suspended: 'gray',
  };
  return colors[status] || 'gray';
}

export function getSampleStatusLabel(status: SampleStatus): string {
  const labels: Record<SampleStatus, string> = {
    pending: 'Pending',
    uploading: 'Uploading...',
    processing: 'Processing...',
    validated: 'Validated',
    rejected: 'Rejected',
    error: 'Error',
  };
  return labels[status] || status;
}

export function getSampleStatusColor(status: SampleStatus): string {
  const colors: Record<SampleStatus, string> = {
    pending: 'gray',
    uploading: 'blue',
    processing: 'yellow',
    validated: 'green',
    rejected: 'red',
    error: 'red',
  };
  return colors[status] || 'gray';
}

export function getQualityLabel(score: number): string {
  if (score >= 0.8) return 'Excellent';
  if (score >= 0.6) return 'Good';
  if (score >= 0.4) return 'Fair';
  return 'Poor';
}

export function getGenderLabel(gender: VoiceGender): string {
  const labels: Record<VoiceGender, string> = {
    male: 'Male',
    female: 'Female',
    neutral: 'Neutral',
  };
  return labels[gender] || gender;
}

export function getAccentLabel(accent: VoiceAccent): string {
  const labels: Record<VoiceAccent, string> = {
    american: 'American',
    british: 'British',
    australian: 'Australian',
    indian: 'Indian',
    french: 'French',
    spanish: 'Spanish',
    german: 'German',
    other: 'Other',
  };
  return labels[accent] || accent;
}

export const defaultGenerationSettings: VoiceGenerationSettings = {
  stability: 0.5,
  similarity_boost: 0.75,
  style: 0.0,
  use_speaker_boost: true,
  speed: 1.0,
  output_format: 'mp3_44100_128',
};
