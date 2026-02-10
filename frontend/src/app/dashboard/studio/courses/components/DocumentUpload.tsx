'use client';

import { useState, useCallback, useRef } from 'react';
import {
  Upload,
  FileText,
  Link as LinkIcon,
  X,
  AlertCircle,
  CheckCircle,
  Loader2,
  Youtube,
  Globe,
} from 'lucide-react';
import {
  Document,
  DocumentUploadResponse,
  ACCEPTED_FILE_TYPES,
  MAX_FILE_SIZE,
  DOCUMENT_TYPE_INFO,
  DOCUMENT_STATUS_INFO,
  formatFileSize,
} from '../lib/document-types';

interface DocumentUploadProps {
  userId: string;
  courseId?: string;
  documents: Document[];
  onDocumentsChange: (documents: Document[]) => void;
  maxDocuments?: number;
}

type UploadMode = 'file' | 'url';

export function DocumentUpload({
  userId,
  courseId,
  documents,
  onDocumentsChange,
  maxDocuments = 10,
}: DocumentUploadProps) {
  const [uploadMode, setUploadMode] = useState<UploadMode>('file');
  const [urlInput, setUrlInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0) return;

      setError(null);
      setIsUploading(true);

      const newDocs: Document[] = [];

      for (const file of Array.from(files)) {
        // Check file size
        if (file.size > MAX_FILE_SIZE) {
          setError(`Le fichier "${file.name}" dépasse la limite de 50 MB`);
          continue;
        }

        // Check document limit
        if (documents.length + newDocs.length >= maxDocuments) {
          setError(`Limite de ${maxDocuments} documents atteinte`);
          break;
        }

        try {
          const formData = new FormData();
          formData.append('file', file);
          formData.append('user_id', userId);
          if (courseId) {
            formData.append('course_id', courseId);
          }

          const response = await fetch(
            `${process.env.NEXT_PUBLIC_API_URL || ''}/api/v1/documents/upload`,
            {
              method: 'POST',
              body: formData,
            }
          );

          if (!response.ok) {
            let errorMessage = 'Upload failed';
            try {
              const errorData = await response.json();
              errorMessage = errorData.detail || errorMessage;
            } catch {
              // Response wasn't JSON, use status text
              errorMessage = response.statusText || errorMessage;
            }
            throw new Error(errorMessage);
          }

          const result: DocumentUploadResponse = await response.json();

          // Create a Document object from the response
          const doc: Document = {
            id: result.document_id,
            user_id: userId,
            course_id: courseId,
            filename: result.filename,
            document_type: result.document_type,
            file_size_bytes: file.size,
            status: result.status,
            page_count: 0,
            word_count: 0,
            chunk_count: 0,
            uploaded_at: new Date().toISOString(),
          };

          newDocs.push(doc);
        } catch (err) {
          console.error('Upload error:', err);
          // Detect network errors specifically
          const isNetworkError = err instanceof TypeError &&
            (err.message.includes('Failed to fetch') || err.message.includes('NetworkError'));

          if (isNetworkError) {
            setError('Erreur réseau. Vérifiez votre connexion Internet.');
          } else {
            setError(
              err instanceof Error ? err.message : `Erreur lors de l'upload de "${file.name}"`
            );
          }
        }
      }

      if (newDocs.length > 0) {
        onDocumentsChange([...documents, ...newDocs]);
      }

      setIsUploading(false);

      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    },
    [documents, userId, courseId, maxDocuments, onDocumentsChange]
  );

  const handleUrlSubmit = useCallback(async () => {
    if (!urlInput.trim()) return;

    setError(null);
    setIsUploading(true);

    try {
      // Check document limit
      if (documents.length >= maxDocuments) {
        throw new Error(`Limite de ${maxDocuments} documents atteinte`);
      }

      const formData = new FormData();
      formData.append('url', urlInput.trim());
      formData.append('user_id', userId);
      if (courseId) {
        formData.append('course_id', courseId);
      }

      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || ''}/api/v1/documents/upload-url`,
        {
          method: 'POST',
          body: formData,
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'URL fetch failed');
      }

      const result: DocumentUploadResponse = await response.json();

      const doc: Document = {
        id: result.document_id,
        user_id: userId,
        course_id: courseId,
        filename: result.filename,
        document_type: result.document_type,
        source_url: urlInput.trim(),
        file_size_bytes: 0,
        status: result.status,
        page_count: 0,
        word_count: 0,
        chunk_count: 0,
        uploaded_at: new Date().toISOString(),
      };

      onDocumentsChange([...documents, doc]);
      setUrlInput('');
    } catch (err) {
      console.error('URL upload error:', err);
      setError(err instanceof Error ? err.message : "Erreur lors de l'import de l'URL");
    }

    setIsUploading(false);
  }, [urlInput, documents, userId, courseId, maxDocuments, onDocumentsChange]);

  const handleRemoveDocument = useCallback(
    async (documentId: string) => {
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL || ''}/api/v1/documents/${documentId}?user_id=${userId}`,
          {
            method: 'DELETE',
          }
        );

        if (response.ok) {
          onDocumentsChange(documents.filter((d) => d.id !== documentId));
        }
      } catch (err) {
        console.error('Delete error:', err);
      }
    },
    [documents, userId, onDocumentsChange]
  );

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);

      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files);
      }
    },
    [handleFileSelect]
  );

  const isYoutubeUrl = (url: string) => {
    return url.includes('youtube.com') || url.includes('youtu.be');
  };

  return (
    <div className="space-y-4">
      {/* Mode selector */}
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => setUploadMode('file')}
          className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-colors ${
            uploadMode === 'file'
              ? 'bg-purple-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          <FileText className="w-4 h-4 inline-block mr-2" />
          Fichier
        </button>
        <button
          type="button"
          onClick={() => setUploadMode('url')}
          className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-colors ${
            uploadMode === 'url'
              ? 'bg-purple-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          <LinkIcon className="w-4 h-4 inline-block mr-2" />
          URL / YouTube
        </button>
      </div>

      {/* Upload area */}
      {uploadMode === 'file' ? (
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
            dragActive
              ? 'border-purple-500 bg-purple-500/10'
              : 'border-gray-600 hover:border-gray-500 hover:bg-gray-800/50'
          } ${isUploading ? 'pointer-events-none opacity-60' : ''}`}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept={ACCEPTED_FILE_TYPES}
            onChange={(e) => handleFileSelect(e.target.files)}
            className="hidden"
          />

          {isUploading ? (
            <div className="flex flex-col items-center">
              <Loader2 className="w-12 h-12 text-purple-400 animate-spin mb-3" />
              <p className="text-gray-300">Upload en cours...</p>
            </div>
          ) : (
            <div className="flex flex-col items-center">
              <Upload className="w-12 h-12 text-gray-500 mb-3" />
              <p className="text-gray-300 mb-1">
                Glissez-déposez vos fichiers ici
              </p>
              <p className="text-gray-500 text-sm">
                ou cliquez pour sélectionner
              </p>
              <p className="text-gray-600 text-xs mt-2">
                PDF, Word, PowerPoint, Excel, TXT, Markdown, CSV (max 50 MB)
              </p>
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          <div className="flex gap-2">
            <input
              type="url"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              placeholder="https://example.com ou lien YouTube"
              className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
              disabled={isUploading}
            />
            <button
              type="button"
              onClick={handleUrlSubmit}
              disabled={!urlInput.trim() || isUploading}
              className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg transition-colors"
            >
              {isUploading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                'Ajouter'
              )}
            </button>
          </div>
          <div className="flex gap-4 text-sm text-gray-400">
            <span className="flex items-center gap-1">
              <Globe className="w-4 h-4" />
              Pages web
            </span>
            <span className="flex items-center gap-1">
              <Youtube className="w-4 h-4 text-red-500" />
              Transcription YouTube
            </span>
          </div>
        </div>
      )}

      {/* Error message */}
      {error && (
        <div className="flex items-center gap-2 text-red-400 text-sm bg-red-500/10 border border-red-500/30 rounded-lg p-3">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
          <button
            type="button"
            onClick={() => setError(null)}
            className="ml-auto text-red-300 hover:text-red-200"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Document list */}
      {documents.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-300">
            Documents source ({documents.length}/{maxDocuments})
          </h4>
          <div className="space-y-2">
            {documents.map((doc) => {
              const typeInfo = DOCUMENT_TYPE_INFO[doc.document_type];
              const statusInfo = DOCUMENT_STATUS_INFO[doc.status];

              return (
                <div
                  key={doc.id}
                  className="flex items-center gap-3 bg-gray-700/50 rounded-lg p-3"
                >
                  <span className="text-xl">{typeInfo?.icon || '📄'}</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-white truncate">{doc.filename}</p>
                    <div className="flex items-center gap-2 text-xs text-gray-400">
                      {doc.file_size_bytes > 0 && (
                        <span>{formatFileSize(doc.file_size_bytes)}</span>
                      )}
                      {doc.source_url && (
                        <span className="truncate max-w-[200px]">
                          {doc.source_url}
                        </span>
                      )}
                      {doc.chunk_count > 0 && (
                        <span>{doc.chunk_count} chunks</span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {doc.status === 'ready' ? (
                      <CheckCircle className="w-4 h-4 text-green-400" />
                    ) : doc.status.includes('failed') ? (
                      <AlertCircle className="w-4 h-4 text-red-400" />
                    ) : (
                      <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                    )}
                    <span
                      className={`text-xs ${
                        doc.status === 'ready'
                          ? 'text-green-400'
                          : doc.status.includes('failed')
                          ? 'text-red-400'
                          : 'text-blue-400'
                      }`}
                    >
                      {statusInfo?.label || doc.status}
                    </span>
                    <button
                      type="button"
                      onClick={() => handleRemoveDocument(doc.id)}
                      className="text-gray-400 hover:text-red-400 transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Info text */}
      <p className="text-xs text-gray-500">
        Les documents uploadés seront utilisés comme source de contenu pour
        générer le cours. L'IA extraira les informations pertinentes pour créer
        des leçons précises et complètes.
      </p>
    </div>
  );
}
