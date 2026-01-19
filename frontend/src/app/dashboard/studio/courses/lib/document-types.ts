/**
 * Document Types for RAG System (Phase 2)
 */

export type DocumentType =
  | 'pdf'
  | 'docx'
  | 'doc'
  | 'pptx'
  | 'ppt'
  | 'txt'
  | 'md'
  | 'xlsx'
  | 'xls'
  | 'csv'
  | 'url'
  | 'youtube';

export type DocumentStatus =
  | 'pending'
  | 'scanning'
  | 'scan_failed'
  | 'parsing'
  | 'parse_failed'
  | 'vectorizing'
  | 'ready'
  | 'failed';

export interface SecurityScanResult {
  is_safe: boolean;
  scan_timestamp: string;
  threats_found: string[];
  warnings: string[];
  mime_type_valid: boolean;
  extension_matches_content: boolean;
  no_macros: boolean;
  no_embedded_objects: boolean;
  file_size_ok: boolean;
}

export interface Document {
  id: string;
  user_id: string;
  course_id?: string;
  filename: string;
  document_type: DocumentType;
  file_size_bytes: number;
  file_path?: string;
  source_url?: string;
  status: DocumentStatus;
  error_message?: string;
  security_scan?: SecurityScanResult;
  content_summary?: string;
  page_count: number;
  word_count: number;
  chunk_count: number;
  uploaded_at: string;
  processed_at?: string;
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  document_type: DocumentType;
  status: DocumentStatus;
  message: string;
}

export interface DocumentListResponse {
  documents: Document[];
  total: number;
  page: number;
  page_size: number;
}

export interface RAGQueryRequest {
  query: string;
  document_ids?: string[];
  course_id?: string;
  user_id: string;
  top_k?: number;
  similarity_threshold?: number;
  include_metadata?: boolean;
  max_tokens?: number;
}

export interface RAGChunkResult {
  chunk_id: string;
  document_id: string;
  document_name: string;
  content: string;
  similarity_score: number;
  page_number?: number;
  section_title?: string;
  token_count: number;
}

export interface RAGQueryResponse {
  query: string;
  results: RAGChunkResult[];
  total_results: number;
  total_tokens: number;
  combined_context: string;
}

// File type info for UI
export const DOCUMENT_TYPE_INFO: Record<DocumentType, { icon: string; label: string; accept: string }> = {
  pdf: { icon: '📄', label: 'PDF', accept: '.pdf' },
  docx: { icon: '📝', label: 'Word', accept: '.docx' },
  doc: { icon: '📝', label: 'Word (Legacy)', accept: '.doc' },
  pptx: { icon: '📊', label: 'PowerPoint', accept: '.pptx' },
  ppt: { icon: '📊', label: 'PowerPoint (Legacy)', accept: '.ppt' },
  txt: { icon: '📃', label: 'Texte', accept: '.txt' },
  md: { icon: '📑', label: 'Markdown', accept: '.md' },
  xlsx: { icon: '📈', label: 'Excel', accept: '.xlsx' },
  xls: { icon: '📈', label: 'Excel (Legacy)', accept: '.xls' },
  csv: { icon: '📊', label: 'CSV', accept: '.csv' },
  url: { icon: '🌐', label: 'Page Web', accept: '' },
  youtube: { icon: '▶️', label: 'YouTube', accept: '' },
};

// Status info for UI
export const DOCUMENT_STATUS_INFO: Record<DocumentStatus, { label: string; color: string }> = {
  pending: { label: 'En attente', color: 'gray' },
  scanning: { label: 'Analyse sécurité...', color: 'blue' },
  scan_failed: { label: 'Échec sécurité', color: 'red' },
  parsing: { label: 'Extraction...', color: 'blue' },
  parse_failed: { label: 'Échec extraction', color: 'red' },
  vectorizing: { label: 'Indexation...', color: 'blue' },
  ready: { label: 'Prêt', color: 'green' },
  failed: { label: 'Échec', color: 'red' },
};

// Accepted file extensions
export const ACCEPTED_FILE_TYPES = [
  '.pdf',
  '.docx',
  '.doc',
  '.pptx',
  '.ppt',
  '.txt',
  '.md',
  '.xlsx',
  '.xls',
  '.csv',
].join(',');

// Max file size (50 MB)
export const MAX_FILE_SIZE = 50 * 1024 * 1024;

// Format file size for display
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}
