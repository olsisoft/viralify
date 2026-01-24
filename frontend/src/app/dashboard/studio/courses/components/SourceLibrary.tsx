'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Library,
  Upload,
  Link,
  FileText,
  Youtube,
  Globe,
  Plus,
  Trash2,
  Check,
  X,
  Loader2,
  Search,
  Tag,
  Sparkles,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  AlertCircle,
  CheckCircle2,
  Clock,
} from 'lucide-react';
import { useSourceLibrary } from '../hooks/useSourceLibrary';
import type {
  Source,
  SourceSuggestion,
  SourceType,
} from '../lib/source-types';
import {
  SOURCE_TYPES,
  getSourceTypeInfo,
  formatFileSize,
  getStatusColor,
  getStatusLabel,
} from '../lib/source-types';

interface SourceLibraryProps {
  userId: string;
  courseId?: string;
  topic?: string;
  onSourcesChange?: (sourceIds: string[]) => void;
}

export function SourceLibrary({
  userId,
  courseId,
  topic,
  onSourcesChange,
}: SourceLibraryProps) {
  // State
  const [selectedSourceIds, setSelectedSourceIds] = useState<Set<string>>(new Set());
  const [showAddModal, setShowAddModal] = useState(false);
  const [addType, setAddType] = useState<SourceType | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedSection, setExpandedSection] = useState<'library' | 'suggestions' | null>('library');
  const [isInitialized, setIsInitialized] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<{
    isActive: boolean;
    message: string;
    type: 'file' | 'url' | 'note' | null;
  }>({ isActive: false, message: '', type: null });
  // Track sources uploaded during this session (for RAG without courseId)
  // Using array instead of Set for proper React re-render detection
  const [sessionUploadedIds, setSessionUploadedIds] = useState<string[]>([]);

  // Hook
  const {
    sources,
    courseSources,
    suggestions,
    relevantExisting,
    isLoading,
    isUploading,
    isSuggesting,
    error,
    fetchSources,
    uploadFile,
    createFromUrl,
    createNote,
    deleteSource,
    fetchCourseSources,
    linkSourceToCourse,
    unlinkSourceFromCourse,
    suggestSources,
    addSuggestionAsSource,
    clearError,
  } = useSourceLibrary({ userId });

  // Initialize
  useEffect(() => {
    if (!isInitialized) {
      fetchSources();
      if (courseId) {
        fetchCourseSources(courseId);
      }
      setIsInitialized(true);
    }
  }, [isInitialized, fetchSources, fetchCourseSources, courseId]);

  // Update parent when course sources or session uploads change
  useEffect(() => {
    if (onSourcesChange) {
      // Combine course-linked sources with session-uploaded sources
      const courseSourceIds = courseSources.map(cs => cs.sourceId);
      // Merge and deduplicate
      const allIds = [...new Set([...courseSourceIds, ...sessionUploadedIds])];
      console.log('[SourceLibrary] Notifying sourceIds:', allIds);
      onSourcesChange(allIds);
    }
  }, [courseSources, sessionUploadedIds, onSourcesChange]);

  // Get AI suggestions when topic changes
  useEffect(() => {
    if (topic && topic.length > 3) {
      const timeout = setTimeout(() => {
        suggestSources(topic);
      }, 1000);
      return () => clearTimeout(timeout);
    }
  }, [topic, suggestSources]);

  // Filter sources by search
  const filteredSources = sources.filter(s =>
    s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.tags.some(t => t.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  // Check if source is linked to course
  const isLinked = useCallback((sourceId: string) => {
    return courseSources.some(cs => cs.sourceId === sourceId);
  }, [courseSources]);

  // Toggle source selection
  const toggleSourceSelection = useCallback((sourceId: string) => {
    setSelectedSourceIds(prev => {
      const next = new Set(prev);
      if (next.has(sourceId)) {
        next.delete(sourceId);
      } else {
        next.add(sourceId);
      }
      return next;
    });
  }, []);

  // Link selected sources to course
  const linkSelectedSources = useCallback(async () => {
    if (!courseId) return;

    for (const sourceId of selectedSourceIds) {
      if (!isLinked(sourceId)) {
        await linkSourceToCourse(courseId, sourceId);
      }
    }
    setSelectedSourceIds(new Set());
  }, [courseId, selectedSourceIds, isLinked, linkSourceToCourse]);

  // Handle suggestion click
  const handleAddSuggestion = useCallback(async (suggestion: SourceSuggestion) => {
    setUploadProgress({ isActive: true, message: 'Ajout de la suggestion...', type: 'url' });
    try {
      const source = await addSuggestionAsSource(suggestion);
      if (source && courseId) {
        await linkSourceToCourse(courseId, source.id);
      }
    } finally {
      setUploadProgress({ isActive: false, message: '', type: null });
    }
  }, [addSuggestionAsSource, courseId, linkSourceToCourse]);

  // Wrapped upload handlers with progress tracking
  const handleUploadFile = useCallback(async (file: File, name?: string, tags?: string[]) => {
    setUploadProgress({ isActive: true, message: `Chargement de "${file.name}"...`, type: 'file' });
    try {
      const result = await uploadFile(file, name, tags);
      setUploadProgress({ isActive: true, message: 'Traitement en cours...', type: 'file' });
      // Track this upload for RAG (even without courseId)
      if (result?.id) {
        console.log('[SourceLibrary] File uploaded, adding ID:', result.id);
        setSessionUploadedIds(prev => prev.includes(result.id) ? prev : [...prev, result.id]);
      }
      return result;
    } finally {
      setUploadProgress({ isActive: false, message: '', type: null });
    }
  }, [uploadFile]);

  const handleCreateFromUrl = useCallback(async (url: string, name?: string, tags?: string[]) => {
    const isYoutube = url.includes('youtube.com') || url.includes('youtu.be');
    setUploadProgress({
      isActive: true,
      message: isYoutube ? 'Extraction de la transcription YouTube...' : 'Chargement de la page web...',
      type: 'url'
    });
    try {
      const result = await createFromUrl(url, name, tags);
      // Track this upload for RAG (even without courseId)
      if (result?.id) {
        setSessionUploadedIds(prev => prev.includes(result.id) ? prev : [...prev, result.id]);
      }
      return result;
    } finally {
      setUploadProgress({ isActive: false, message: '', type: null });
    }
  }, [createFromUrl]);

  const handleCreateNote = useCallback(async (content: string, name: string, tags?: string[]) => {
    setUploadProgress({ isActive: true, message: 'Création de la note...', type: 'note' });
    try {
      const result = await createNote(content, name, tags);
      return result;
    } finally {
      setUploadProgress({ isActive: false, message: '', type: null });
    }
  }, [createNote]);

  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Library className="w-5 h-5 text-purple-400" />
            <h3 className="text-lg font-semibold text-white">Bibliothèque de Sources</h3>
          </div>
          <button
            onClick={() => setShowAddModal(true)}
            className="flex items-center gap-2 px-3 py-1.5 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-sm transition-colors"
          >
            <Plus className="w-4 h-4" />
            Ajouter
          </button>
        </div>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Rechercher dans vos sources..."
            className="w-full pl-10 pr-4 py-2 bg-gray-900/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
          />
        </div>

        {/* Error */}
        {error && (
          <div className="mt-3 p-2 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2 text-red-400 text-sm">
            <AlertCircle className="w-4 h-4" />
            {error}
            <button onClick={clearError} className="ml-auto">
              <X className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>

      {/* AI Suggestions Section */}
      {topic && (
        <div className="border-b border-gray-700">
          <button
            onClick={() => setExpandedSection(expandedSection === 'suggestions' ? null : 'suggestions')}
            className="w-full p-3 flex items-center justify-between hover:bg-gray-700/30 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-yellow-400" />
              <span className="text-sm font-medium text-gray-300">Suggestions IA</span>
              {isSuggesting && <Loader2 className="w-4 h-4 animate-spin text-gray-500" />}
              {suggestions.length > 0 && (
                <span className="px-2 py-0.5 bg-yellow-500/20 text-yellow-400 text-xs rounded-full">
                  {suggestions.length}
                </span>
              )}
            </div>
            {expandedSection === 'suggestions' ? (
              <ChevronUp className="w-4 h-4 text-gray-500" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-500" />
            )}
          </button>

          {expandedSection === 'suggestions' && (
            <div className="p-3 pt-0 space-y-2">
              {relevantExisting.length > 0 && (
                <div className="mb-3">
                  <p className="text-xs text-gray-500 mb-2">Sources existantes pertinentes:</p>
                  {relevantExisting.map(source => (
                    <SourceItem
                      key={source.id}
                      source={source}
                      isLinked={isLinked(source.id)}
                      isSelected={selectedSourceIds.has(source.id)}
                      onToggleSelect={() => toggleSourceSelection(source.id)}
                      onLink={courseId ? () => linkSourceToCourse(courseId, source.id) : undefined}
                      onUnlink={courseId ? () => unlinkSourceFromCourse(courseId, source.id) : undefined}
                      compact
                    />
                  ))}
                </div>
              )}

              {suggestions.length > 0 ? (
                suggestions.map((suggestion, idx) => (
                  <SuggestionItem
                    key={idx}
                    suggestion={suggestion}
                    onAdd={() => handleAddSuggestion(suggestion)}
                    isAdding={isUploading}
                  />
                ))
              ) : !isSuggesting ? (
                <p className="text-sm text-gray-500 text-center py-2">
                  Aucune suggestion pour ce sujet
                </p>
              ) : null}
            </div>
          )}
        </div>
      )}

      {/* Library Section */}
      <div>
        <button
          onClick={() => setExpandedSection(expandedSection === 'library' ? null : 'library')}
          className="w-full p-3 flex items-center justify-between hover:bg-gray-700/30 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Library className="w-4 h-4 text-purple-400" />
            <span className="text-sm font-medium text-gray-300">Mes Sources</span>
            <span className="px-2 py-0.5 bg-purple-500/20 text-purple-400 text-xs rounded-full">
              {sources.length}
            </span>
          </div>
          {expandedSection === 'library' ? (
            <ChevronUp className="w-4 h-4 text-gray-500" />
          ) : (
            <ChevronDown className="w-4 h-4 text-gray-500" />
          )}
        </button>

        {expandedSection === 'library' && (
          <div className="p-3 pt-0">
            {/* Selection actions */}
            {selectedSourceIds.size > 0 && courseId && (
              <div className="mb-3 p-2 bg-purple-500/10 border border-purple-500/30 rounded-lg flex items-center justify-between">
                <span className="text-sm text-purple-300">
                  {selectedSourceIds.size} source(s) sélectionnée(s)
                </span>
                <div className="flex gap-2">
                  <button
                    onClick={() => setSelectedSourceIds(new Set())}
                    className="px-2 py-1 text-sm text-gray-400 hover:text-white"
                  >
                    Annuler
                  </button>
                  <button
                    onClick={linkSelectedSources}
                    className="px-3 py-1 bg-purple-600 hover:bg-purple-500 text-white text-sm rounded-lg"
                  >
                    Lier au cours
                  </button>
                </div>
              </div>
            )}

            {/* Sources list */}
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 animate-spin text-purple-400" />
              </div>
            ) : filteredSources.length > 0 ? (
              <div className="space-y-2 max-h-[400px] overflow-y-auto">
                {filteredSources.map(source => (
                  <SourceItem
                    key={source.id}
                    source={source}
                    isLinked={isLinked(source.id)}
                    isSelected={selectedSourceIds.has(source.id)}
                    onToggleSelect={() => toggleSourceSelection(source.id)}
                    onLink={courseId ? () => linkSourceToCourse(courseId, source.id) : undefined}
                    onUnlink={courseId ? () => unlinkSourceFromCourse(courseId, source.id) : undefined}
                    onDelete={() => deleteSource(source.id)}
                  />
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <Library className="w-8 h-8 text-gray-600 mx-auto mb-2" />
                <p className="text-gray-500">Aucune source dans votre bibliothèque</p>
                <button
                  onClick={() => setShowAddModal(true)}
                  className="mt-2 text-purple-400 hover:text-purple-300 text-sm"
                >
                  Ajouter votre première source
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Linked sources summary */}
      {courseSources.length > 0 && (
        <div className="p-3 border-t border-gray-700 bg-gray-900/30">
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <CheckCircle2 className="w-4 h-4 text-green-400" />
            <span>{courseSources.length} source(s) liée(s) à ce cours</span>
          </div>
        </div>
      )}

      {/* Global Upload Progress Indicator */}
      {uploadProgress.isActive && (
        <div className="fixed bottom-4 right-4 z-50 flex items-center gap-3 px-4 py-3 bg-gray-800 border border-purple-500/50 rounded-lg shadow-lg">
          <Loader2 className="w-5 h-5 animate-spin text-purple-400" />
          <div>
            <p className="text-sm font-medium text-white">{uploadProgress.message}</p>
            <p className="text-xs text-gray-400">Veuillez patienter...</p>
          </div>
        </div>
      )}

      {/* Add Source Modal */}
      {showAddModal && (
        <AddSourceModal
          addType={addType}
          setAddType={setAddType}
          onClose={() => {
            if (!uploadProgress.isActive) {
              setShowAddModal(false);
              setAddType(null);
            }
          }}
          onUploadFile={handleUploadFile}
          onCreateFromUrl={handleCreateFromUrl}
          onCreateNote={handleCreateNote}
          isUploading={isUploading || uploadProgress.isActive}
        />
      )}
    </div>
  );
}

// =============================================================================
// Sub-components
// =============================================================================

interface SourceItemProps {
  source: Source;
  isLinked: boolean;
  isSelected: boolean;
  onToggleSelect: () => void;
  onLink?: () => void;
  onUnlink?: () => void;
  onDelete?: () => void;
  compact?: boolean;
}

function SourceItem({
  source,
  isLinked,
  isSelected,
  onToggleSelect,
  onLink,
  onUnlink,
  onDelete,
  compact = false,
}: SourceItemProps) {
  const typeInfo = getSourceTypeInfo(source.sourceType);

  return (
    <div
      className={`p-3 rounded-lg border transition-colors ${
        isLinked
          ? 'bg-green-500/10 border-green-500/30'
          : isSelected
          ? 'bg-purple-500/10 border-purple-500/30'
          : 'bg-gray-800/50 border-gray-700 hover:border-gray-600'
      }`}
    >
      <div className="flex items-start gap-3">
        {/* Checkbox */}
        {!isLinked && (
          <button
            onClick={onToggleSelect}
            className={`mt-1 w-4 h-4 rounded border flex items-center justify-center ${
              isSelected
                ? 'bg-purple-600 border-purple-600'
                : 'border-gray-600 hover:border-purple-500'
            }`}
          >
            {isSelected && <Check className="w-3 h-3 text-white" />}
          </button>
        )}

        {/* Icon */}
        <span className="text-xl">{typeInfo.icon}</span>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-white truncate">{source.name}</span>
            <span className={`text-xs ${getStatusColor(source.status)}`}>
              {source.status === 'processing' && <Loader2 className="w-3 h-3 animate-spin inline mr-1" />}
              {getStatusLabel(source.status)}
            </span>
          </div>

          {!compact && (
            <>
              {source.contentSummary && (
                <p className="text-sm text-gray-400 mt-1 line-clamp-2">{source.contentSummary}</p>
              )}

              <div className="flex items-center gap-3 mt-2 text-xs text-gray-500">
                {source.fileSizeBytes > 0 && (
                  <span>{formatFileSize(source.fileSizeBytes)}</span>
                )}
                {source.wordCount > 0 && (
                  <span>{source.wordCount.toLocaleString()} mots</span>
                )}
                {source.chunkCount > 0 && (
                  <span>{source.chunkCount} segments</span>
                )}
              </div>

              {source.tags.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {source.tags.slice(0, 3).map((tag, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-0.5 bg-gray-700 text-gray-400 text-xs rounded-full"
                    >
                      {tag}
                    </span>
                  ))}
                  {source.tags.length > 3 && (
                    <span className="px-2 py-0.5 text-gray-500 text-xs">
                      +{source.tags.length - 3}
                    </span>
                  )}
                </div>
              )}
            </>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1">
          {isLinked ? (
            onUnlink && (
              <button
                onClick={onUnlink}
                className="p-1.5 text-green-400 hover:text-red-400 rounded-lg hover:bg-gray-700/50 transition-colors"
                title="Retirer du cours"
              >
                <X className="w-4 h-4" />
              </button>
            )
          ) : (
            onLink && source.status === 'ready' && (
              <button
                onClick={onLink}
                className="p-1.5 text-gray-400 hover:text-green-400 rounded-lg hover:bg-gray-700/50 transition-colors"
                title="Lier au cours"
              >
                <Link className="w-4 h-4" />
              </button>
            )
          )}

          {onDelete && !isLinked && (
            <button
              onClick={onDelete}
              className="p-1.5 text-gray-400 hover:text-red-400 rounded-lg hover:bg-gray-700/50 transition-colors"
              title="Supprimer"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

interface SuggestionItemProps {
  suggestion: SourceSuggestion;
  onAdd: () => void;
  isAdding: boolean;
}

function SuggestionItem({ suggestion, onAdd, isAdding }: SuggestionItemProps) {
  const typeInfo = getSourceTypeInfo(suggestion.suggestionType);

  return (
    <div className="p-3 bg-yellow-500/5 border border-yellow-500/20 rounded-lg">
      <div className="flex items-start gap-3">
        <span className="text-xl">{typeInfo.icon}</span>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-white">{suggestion.title}</span>
            <span className="px-1.5 py-0.5 bg-yellow-500/20 text-yellow-400 text-xs rounded">
              {Math.round(suggestion.relevanceScore * 100)}% pertinent
            </span>
          </div>

          <p className="text-sm text-gray-400 mt-1">{suggestion.description}</p>

          {suggestion.url && (
            <a
              href={suggestion.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-xs text-purple-400 hover:text-purple-300 mt-2"
            >
              <ExternalLink className="w-3 h-3" />
              {suggestion.url.length > 50 ? suggestion.url.slice(0, 50) + '...' : suggestion.url}
            </a>
          )}

          {suggestion.keywords.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {suggestion.keywords.slice(0, 4).map((kw, idx) => (
                <span
                  key={idx}
                  className="px-2 py-0.5 bg-gray-700 text-gray-400 text-xs rounded-full"
                >
                  {kw}
                </span>
              ))}
            </div>
          )}
        </div>

        <button
          onClick={onAdd}
          disabled={isAdding}
          className="px-3 py-1.5 bg-yellow-600 hover:bg-yellow-500 disabled:opacity-50 text-white text-sm rounded-lg transition-colors flex items-center gap-1"
        >
          {isAdding ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Plus className="w-4 h-4" />
          )}
          Ajouter
        </button>
      </div>
    </div>
  );
}

interface AddSourceModalProps {
  addType: SourceType | null;
  setAddType: (type: SourceType | null) => void;
  onClose: () => void;
  onUploadFile: (file: File, name?: string, tags?: string[]) => Promise<Source | null>;
  onCreateFromUrl: (url: string, name?: string, tags?: string[]) => Promise<Source | null>;
  onCreateNote: (content: string, name: string, tags?: string[]) => Promise<Source | null>;
  isUploading: boolean;
}

function AddSourceModal({
  addType,
  setAddType,
  onClose,
  onUploadFile,
  onCreateFromUrl,
  onCreateNote,
  isUploading,
}: AddSourceModalProps) {
  const [url, setUrl] = useState('');
  const [name, setName] = useState('');
  const [noteContent, setNoteContent] = useState('');
  const [tags, setTags] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0] && !isUploading) {
      const file = e.dataTransfer.files[0];
      const tagList = tags ? tags.split(',').map(t => t.trim()) : undefined;
      const result = await onUploadFile(file, name || undefined, tagList);
      if (result) {
        onClose();
      }
    }
  }, [name, tags, onUploadFile, onClose, isUploading]);

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0] && !isUploading) {
      const file = e.target.files[0];
      const tagList = tags ? tags.split(',').map(t => t.trim()) : undefined;
      const result = await onUploadFile(file, name || undefined, tagList);
      if (result) {
        onClose();
      }
    }
  }, [name, tags, onUploadFile, onClose, isUploading]);

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (isUploading) return;

    const tagList = tags ? tags.split(',').map(t => t.trim()) : undefined;
    let result = null;

    if (addType === 'url' || addType === 'youtube') {
      result = await onCreateFromUrl(url, name || undefined, tagList);
    } else if (addType === 'note') {
      result = await onCreateNote(noteContent, name, tagList);
    }

    if (result) {
      onClose();
    }
  }, [addType, url, name, noteContent, tags, onCreateFromUrl, onCreateNote, onClose, isUploading]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-gray-800 border border-gray-700 rounded-xl w-full max-w-lg mx-4 overflow-hidden">
        <div className="p-4 border-b border-gray-700 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-white">Ajouter une Source</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-white">
            <X className="w-5 h-5" />
          </button>
        </div>

        {!addType ? (
          <div className="p-4 grid grid-cols-2 gap-3">
            {SOURCE_TYPES.map(type => (
              <button
                key={type.id}
                onClick={() => setAddType(type.id)}
                className="p-4 bg-gray-900/50 hover:bg-gray-700/50 border border-gray-700 hover:border-purple-500 rounded-lg text-left transition-colors"
              >
                <span className="text-2xl">{type.icon}</span>
                <div className="mt-2">
                  <p className="font-medium text-white">{type.name}</p>
                  <p className="text-sm text-gray-400">{type.description}</p>
                </div>
              </button>
            ))}
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="p-4 space-y-4">
            {/* Back button */}
            <button
              type="button"
              onClick={() => setAddType(null)}
              className="text-sm text-gray-400 hover:text-white flex items-center gap-1"
            >
              <ChevronDown className="w-4 h-4 rotate-90" />
              Retour
            </button>

            {addType === 'file' ? (
              <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                  isUploading
                    ? 'border-purple-500 bg-purple-500/10'
                    : dragActive
                    ? 'border-purple-500 bg-purple-500/10'
                    : 'border-gray-600 hover:border-gray-500'
                }`}
              >
                {isUploading ? (
                  <>
                    <Loader2 className="w-8 h-8 text-purple-400 mx-auto mb-3 animate-spin" />
                    <p className="text-purple-300 font-medium mb-1">Chargement en cours...</p>
                    <p className="text-sm text-gray-400">Analyse et traitement du fichier</p>
                  </>
                ) : (
                  <>
                    <Upload className="w-8 h-8 text-gray-500 mx-auto mb-3" />
                    <p className="text-gray-400 mb-2">
                      Glissez un fichier ici ou{' '}
                      <button
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                        className="text-purple-400 hover:text-purple-300"
                      >
                        parcourez
                      </button>
                    </p>
                    <p className="text-xs text-gray-500">PDF, Word, PowerPoint, Excel, TXT, Markdown</p>
                  </>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  onChange={handleFileSelect}
                  accept=".pdf,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.txt,.md,.csv"
                  className="hidden"
                  disabled={isUploading}
                />
              </div>
            ) : addType === 'url' || addType === 'youtube' ? (
              <div>
                <label className="block text-sm text-gray-400 mb-1">
                  {addType === 'youtube' ? 'URL YouTube' : 'URL de la page'}
                </label>
                <input
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder={addType === 'youtube' ? 'https://youtube.com/watch?v=...' : 'https://...'}
                  className="w-full px-4 py-2 bg-gray-900/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                  required
                />
              </div>
            ) : addType === 'note' ? (
              <div>
                <label className="block text-sm text-gray-400 mb-1">Contenu de la note</label>
                <textarea
                  value={noteContent}
                  onChange={(e) => setNoteContent(e.target.value)}
                  placeholder="Tapez vos notes ici..."
                  rows={6}
                  className="w-full px-4 py-2 bg-gray-900/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
                  required
                />
              </div>
            ) : null}

            {/* Name input */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Nom {addType === 'note' ? '(requis)' : '(optionnel)'}
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Nom de la source"
                className="w-full px-4 py-2 bg-gray-900/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                required={addType === 'note'}
              />
            </div>

            {/* Tags input */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">Tags (optionnel)</label>
              <div className="relative">
                <Tag className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                <input
                  type="text"
                  value={tags}
                  onChange={(e) => setTags(e.target.value)}
                  placeholder="python, tutorial, api (séparés par virgule)"
                  className="w-full pl-10 pr-4 py-2 bg-gray-900/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>
            </div>

            {/* Submit button */}
            {addType !== 'file' && (
              <button
                type="submit"
                disabled={isUploading}
                className="w-full py-2 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white rounded-lg flex items-center justify-center gap-2"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Traitement...
                  </>
                ) : (
                  <>
                    <Plus className="w-4 h-4" />
                    Ajouter
                  </>
                )}
              </button>
            )}
          </form>
        )}
      </div>
    </div>
  );
}
