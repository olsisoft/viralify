/**
 * Video Editor Types
 * Phase 3: User Video Editing feature
 */

// ========================================
// Enums
// ========================================

export type SegmentType = 'generated' | 'user_video' | 'user_audio' | 'slide' | 'transition' | 'overlay';

export type TransitionType = 'none' | 'fade' | 'dissolve' | 'wipe_left' | 'wipe_right' | 'zoom_in' | 'zoom_out';

export type SegmentStatus = 'pending' | 'uploading' | 'processing' | 'ready' | 'error';

export type ProjectStatus = 'draft' | 'editing' | 'rendering' | 'completed' | 'failed';

// ========================================
// Models
// ========================================

export interface AudioTrack {
  id: string;
  source_url: string;
  volume: number;
  start_time: number;
  duration?: number;
  fade_in: number;
  fade_out: number;
  is_muted: boolean;
}

export interface VideoSegment {
  id: string;
  project_id: string;
  segment_type: SegmentType;
  source_url?: string;
  source_lecture_id?: string;
  order: number;
  start_time: number;
  duration: number;
  trim_start: number;
  trim_end?: number;
  audio_tracks: AudioTrack[];
  original_audio_volume: number;
  is_audio_muted: boolean;
  opacity: number;
  scale: number;
  position_x: number;
  position_y: number;
  rotation: number;
  transition_in: TransitionType;
  transition_in_duration: number;
  transition_out: TransitionType;
  transition_out_duration: number;
  status: SegmentStatus;
  error_message?: string;
  title?: string;
  thumbnail_url?: string;
  original_filename?: string;
  file_size_bytes: number;
  created_at: string;
  updated_at: string;
}

export interface TextOverlay {
  id: string;
  text: string;
  font_family: string;
  font_size: number;
  font_color: string;
  background_color?: string;
  position_x: number;
  position_y: number;
  start_time: number;
  duration: number;
  animation: string;
}

export interface ImageOverlay {
  id: string;
  image_url: string;
  position_x: number;
  position_y: number;
  scale: number;
  opacity: number;
  start_time?: number;
  duration?: number;
}

export interface VideoProject {
  id: string;
  user_id: string;
  course_id?: string;
  course_job_id?: string;
  title: string;
  description?: string;
  segments: VideoSegment[];
  total_duration: number;
  text_overlays: TextOverlay[];
  image_overlays: ImageOverlay[];
  output_resolution: string;
  output_fps: number;
  output_format: string;
  output_quality: string;
  background_music_url?: string;
  background_music_volume: number;
  status: ProjectStatus;
  render_progress: number;
  render_message?: string;
  output_url?: string;
  created_at: string;
  updated_at: string;
  rendered_at?: string;
}

// ========================================
// Request/Response Types
// ========================================

export interface CreateProjectRequest {
  user_id: string;
  course_id?: string;
  course_job_id?: string;
  title: string;
  description?: string;
  import_course_videos?: boolean;
}

export interface CreateProjectResponse {
  project_id: string;
  title: string;
  status: ProjectStatus;
  segment_count: number;
  message: string;
}

export interface AddSegmentRequest {
  segment_type: SegmentType;
  source_url?: string;
  source_lecture_id?: string;
  insert_after_segment_id?: string;
  trim_start?: number;
  trim_end?: number;
  slide_image_url?: string;
  slide_duration?: number;
  title?: string;
}

export interface UpdateSegmentRequest {
  trim_start?: number;
  trim_end?: number;
  original_audio_volume?: number;
  is_audio_muted?: boolean;
  opacity?: number;
  scale?: number;
  position_x?: number;
  position_y?: number;
  transition_in?: TransitionType;
  transition_in_duration?: number;
  transition_out?: TransitionType;
  transition_out_duration?: number;
  title?: string;
}

export interface ReorderSegmentsRequest {
  segment_ids: string[];
}

export interface RenderProjectRequest {
  output_resolution?: string;
  output_fps?: number;
  output_quality?: string;
  include_watermark?: boolean;
  watermark_url?: string;
}

export interface ProjectListResponse {
  projects: VideoProject[];
  total: number;
  page: number;
  page_size: number;
}

export interface UploadSegmentResponse {
  segment_id: string;
  status: SegmentStatus;
  source_url: string;
  duration: number;
  thumbnail_url?: string;
  message: string;
}

export interface RenderJobStatus {
  job_id: string;
  project_id: string;
  user_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  created_at: string;
  completed_at?: string;
  output_url?: string;
  error?: string;
}

export interface SupportedFormats {
  video: string[];
  audio: string[];
  image: string[];
  max_sizes: {
    video_mb: number;
    audio_mb: number;
    image_mb: number;
  };
}

// ========================================
// UI State Types
// ========================================

export interface TimelineState {
  project: VideoProject | null;
  selectedSegmentId: string | null;
  playheadPosition: number;
  zoom: number;
  isPlaying: boolean;
  isDragging: boolean;
}

export interface EditorState {
  project: VideoProject | null;
  isLoading: boolean;
  isSaving: boolean;
  isRendering: boolean;
  renderJobId: string | null;
  error: string | null;
  undoStack: VideoProject[];
  redoStack: VideoProject[];
}

// ========================================
// Helper Functions
// ========================================

export function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function formatDetailedDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function getSegmentTypeLabel(type: SegmentType): string {
  const labels: Record<SegmentType, string> = {
    generated: 'Generated Video',
    user_video: 'User Video',
    user_audio: 'User Audio',
    slide: 'Slide/Image',
    transition: 'Transition',
    overlay: 'Overlay',
  };
  return labels[type] || type;
}

export function getSegmentTypeIcon(type: SegmentType): string {
  const icons: Record<SegmentType, string> = {
    generated: 'video',
    user_video: 'film',
    user_audio: 'music',
    slide: 'image',
    transition: 'shuffle',
    overlay: 'layers',
  };
  return icons[type] || 'file';
}

export function getStatusColor(status: SegmentStatus | ProjectStatus): string {
  const colors: Record<string, string> = {
    pending: 'gray',
    uploading: 'blue',
    processing: 'yellow',
    ready: 'green',
    error: 'red',
    draft: 'gray',
    editing: 'blue',
    rendering: 'yellow',
    completed: 'green',
    failed: 'red',
  };
  return colors[status] || 'gray';
}
