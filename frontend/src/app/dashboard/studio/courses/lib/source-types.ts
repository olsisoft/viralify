/**
 * Source Library TypeScript Types
 */

// Source Types
export type SourceType = 'file' | 'url' | 'youtube' | 'note';
export type SourceStatus = 'pending' | 'processing' | 'ready' | 'failed';

// Source interface - represents a saved source in user's library
export interface Source {
  id: string;
  userId: string;
  name: string;
  sourceType: SourceType;
  filename?: string;
  documentType?: string;
  fileSizeBytes: number;
  sourceUrl?: string;
  noteContent?: string;
  status: SourceStatus;
  errorMessage?: string;
  contentSummary?: string;
  wordCount: number;
  chunkCount: number;
  isVectorized: boolean;
  tags: string[];
  usageCount: number;
  lastUsedAt?: string;
  createdAt: string;
  updatedAt: string;
}

// CourseSource - link between source and course
export interface CourseSource {
  id: string;
  courseId: string;
  sourceId: string;
  source: Source;
  relevanceScore?: number;
  isPrimary: boolean;
  addedAt: string;
}

// Source suggestion from AI
export interface SourceSuggestion {
  suggestionType: SourceType;
  title: string;
  url?: string;
  description: string;
  relevanceScore: number;
  keywords: string[];
}

// API Response types
export interface SourceListResponse {
  sources: Source[];
  total: number;
  page: number;
  pageSize: number;
}

export interface CourseSourcesResponse {
  courseId: string;
  sources: CourseSource[];
  total: number;
}

export interface SuggestSourcesResponse {
  topic: string;
  suggestions: SourceSuggestion[];
  existingRelevantSources: Source[];
}

// Request types
export interface CreateSourceRequest {
  userId: string;
  name: string;
  sourceType: SourceType;
  sourceUrl?: string;
  noteContent?: string;
  tags?: string[];
}

export interface BulkLinkSourcesRequest {
  courseId: string;
  sourceIds: string[];
  userId: string;
}

// Source type info for UI
export interface SourceTypeInfo {
  id: SourceType;
  name: string;
  icon: string;
  description: string;
}

export const SOURCE_TYPES: SourceTypeInfo[] = [
  { id: 'file', name: 'Fichier', icon: '📄', description: 'PDF, Word, PowerPoint, Excel, etc.' },
  { id: 'url', name: 'Page Web', icon: '🌐', description: 'Article ou documentation en ligne' },
  { id: 'youtube', name: 'YouTube', icon: '🎬', description: 'Vidéo YouTube (transcription)' },
  { id: 'note', name: 'Note', icon: '📝', description: 'Texte personnel ou notes' },
];

// Helper functions
export function getSourceTypeInfo(type: SourceType): SourceTypeInfo {
  return SOURCE_TYPES.find(t => t.id === type) || SOURCE_TYPES[0];
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

export function getStatusColor(status: SourceStatus): string {
  switch (status) {
    case 'ready': return 'text-green-400';
    case 'processing': return 'text-yellow-400';
    case 'pending': return 'text-gray-400';
    case 'failed': return 'text-red-400';
    default: return 'text-gray-400';
  }
}

export function getStatusLabel(status: SourceStatus): string {
  switch (status) {
    case 'ready': return 'Prêt';
    case 'processing': return 'Traitement...';
    case 'pending': return 'En attente';
    case 'failed': return 'Erreur';
    default: return status;
  }
}

// Convert API response to frontend format (snake_case to camelCase)
export function mapSourceFromApi(data: Record<string, unknown>): Source {
  return {
    id: data.id as string,
    userId: data.user_id as string,
    name: data.name as string,
    sourceType: data.source_type as SourceType,
    filename: data.filename as string | undefined,
    documentType: data.document_type as string | undefined,
    fileSizeBytes: data.file_size_bytes as number || 0,
    sourceUrl: data.source_url as string | undefined,
    noteContent: data.note_content as string | undefined,
    status: data.status as SourceStatus,
    errorMessage: data.error_message as string | undefined,
    contentSummary: data.content_summary as string | undefined,
    wordCount: data.word_count as number || 0,
    chunkCount: data.chunk_count as number || 0,
    isVectorized: data.is_vectorized as boolean || false,
    tags: data.tags as string[] || [],
    usageCount: data.usage_count as number || 0,
    lastUsedAt: data.last_used_at as string | undefined,
    createdAt: data.created_at as string,
    updatedAt: data.updated_at as string,
  };
}

export function mapCourseSourceFromApi(data: Record<string, unknown>): CourseSource {
  return {
    id: data.id as string,
    courseId: data.course_id as string,
    sourceId: data.source_id as string,
    source: mapSourceFromApi(data.source as Record<string, unknown>),
    relevanceScore: data.relevance_score as number | undefined,
    isPrimary: data.is_primary as boolean || false,
    addedAt: data.added_at as string,
  };
}

export function mapSuggestionFromApi(data: Record<string, unknown>): SourceSuggestion {
  return {
    suggestionType: data.suggestion_type as SourceType,
    title: data.title as string,
    url: data.url as string | undefined,
    description: data.description as string,
    relevanceScore: data.relevance_score as number,
    keywords: data.keywords as string[] || [],
  };
}
