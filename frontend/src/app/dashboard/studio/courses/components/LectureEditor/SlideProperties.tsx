'use client';

import React, { useState, useCallback, useRef, useEffect, memo } from 'react';
import type { SlideComponent, VoiceoverComponent, UpdateSlideRequest, MediaType, CodeBlockComponent } from '../../lib/lecture-editor-types';
import { getSlideTypeLabel, formatDuration, MEDIA_UPLOAD_CONFIG } from '../../lib/lecture-editor-types';
import { CodeBlockEditor } from './CodeBlockEditor';

// Separate EditableField component to prevent focus loss on parent re-renders
interface EditableFieldProps {
  field: string;
  value: string | number | undefined;
  label: string;
  multiline?: boolean;
  type?: 'text' | 'number';
  suffix?: string;
  min?: number;
  max?: number;
  isEditing: boolean;
  editValue: string | number | undefined;
  isReadOnly: boolean;
  isSaving: boolean;
  onStartEditing: (field: string, value: string | number) => void;
  onEditValueChange: (field: string, value: string | number) => void;
  onSave: (field: string) => void;
  onCancel: () => void;
}

const EditableField = memo(function EditableField({
  field,
  value,
  label,
  multiline = false,
  type = 'text',
  suffix = '',
  min,
  max,
  isEditing,
  editValue,
  isReadOnly,
  isSaving,
  onStartEditing,
  onEditValueChange,
  onSave,
  onCancel,
}: EditableFieldProps) {
  const inputRef = useRef<HTMLInputElement | HTMLTextAreaElement>(null);
  const displayValue = value ?? '';
  const currentValue = editValue ?? displayValue;

  // Focus input when editing starts
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      if (inputRef.current instanceof HTMLTextAreaElement) {
        inputRef.current.setSelectionRange(
          inputRef.current.value.length,
          inputRef.current.value.length
        );
      }
    }
  }, [isEditing]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSave(field);
    } else if (e.key === 'Escape') {
      onCancel();
    }
  };

  if (isEditing) {
    return (
      <div className="space-y-2">
        <label className="text-gray-400 text-xs font-medium block">{label}</label>
        <div className="relative">
          {multiline ? (
            <textarea
              ref={inputRef as React.RefObject<HTMLTextAreaElement>}
              value={currentValue}
              onChange={(e) => onEditValueChange(field, e.target.value)}
              onKeyDown={handleKeyDown}
              rows={4}
              className="w-full bg-gray-800 text-white rounded-lg px-3 py-2 text-sm border-2 border-purple-500 focus:outline-none resize-none"
            />
          ) : (
            <div className="flex items-center gap-2">
              <input
                ref={inputRef as React.RefObject<HTMLInputElement>}
                type={type}
                value={currentValue}
                onChange={(e) => onEditValueChange(field, type === 'number' ? Number(e.target.value) : e.target.value)}
                onKeyDown={handleKeyDown}
                min={min}
                max={max}
                className="flex-1 bg-gray-800 text-white rounded-lg px-3 py-2 text-sm border-2 border-purple-500 focus:outline-none"
              />
              {suffix && <span className="text-gray-400 text-sm">{suffix}</span>}
            </div>
          )}
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => onSave(field)}
            disabled={isSaving}
            className="flex-1 px-3 py-1.5 bg-purple-600 text-white rounded text-xs font-medium hover:bg-purple-500 disabled:opacity-50 transition-colors"
          >
            {isSaving ? 'Sauvegarde...' : 'Enregistrer'}
          </button>
          <button
            onClick={onCancel}
            className="px-3 py-1.5 bg-gray-700 text-white rounded text-xs hover:bg-gray-600 transition-colors"
          >
            Annuler
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      onClick={() => !isReadOnly && onStartEditing(field, displayValue)}
      className={`group ${!isReadOnly ? 'cursor-pointer' : ''}`}
    >
      <div className="flex items-center justify-between mb-1">
        <label className="text-gray-400 text-xs font-medium">{label}</label>
        {!isReadOnly && (
          <span className="text-purple-400 text-xs opacity-0 group-hover:opacity-100 transition-opacity">
            Cliquer pour modifier
          </span>
        )}
      </div>
      <div className={`bg-gray-800 rounded-lg px-3 py-2 text-sm transition-colors ${
        !isReadOnly ? 'hover:bg-gray-750 hover:ring-1 hover:ring-purple-500/50' : ''
      }`}>
        {displayValue ? (
          <span className="text-white">{displayValue}{suffix}</span>
        ) : (
          <span className="text-gray-500 italic">Non défini</span>
        )}
      </div>
    </div>
  );
});

interface SlidePropertiesProps {
  slide: SlideComponent | null;
  voiceover?: VoiceoverComponent;
  isSaving: boolean;
  isRegenerating: boolean;
  onUpdate: (updates: UpdateSlideRequest) => void;
  onRegenerate: () => void;
  onUploadAudio: (file: File) => void;
  onUploadMedia?: (type: MediaType, file: File) => void;
  isReadOnly?: boolean;
}

export function SlideProperties({
  slide,
  voiceover,
  isSaving,
  isRegenerating,
  onUpdate,
  onRegenerate,
  onUploadAudio,
  onUploadMedia,
  isReadOnly = false,
}: SlidePropertiesProps) {
  // Edit states
  const [editingField, setEditingField] = useState<string | null>(null);
  const [editValues, setEditValues] = useState<Record<string, string | number>>({});
  const [isDraggingAudio, setIsDraggingAudio] = useState(false);
  const [isDraggingMedia, setIsDraggingMedia] = useState(false);
  const [isDraggingImage, setIsDraggingImage] = useState(false);
  const [editingBulletPoints, setEditingBulletPoints] = useState(false);
  const [bulletPointsEdit, setBulletPointsEdit] = useState<string[]>([]);
  const [editingCodeBlock, setEditingCodeBlock] = useState<CodeBlockComponent | null>(null);
  const [codeBlockIndex, setCodeBlockIndex] = useState<number>(0);

  const audioInputRef = useRef<HTMLInputElement>(null);
  const mediaInputRef = useRef<HTMLInputElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);

  // Start editing a field
  const startEditing = useCallback((field: string, value: string | number) => {
    if (isReadOnly) return;
    setEditingField(field);
    setEditValues({ ...editValues, [field]: value });
  }, [isReadOnly, editValues]);

  // Save edited field
  const saveField = useCallback((field: string) => {
    const value = editValues[field];
    const updates: UpdateSlideRequest = {};

    switch (field) {
      case 'title':
        updates.title = value as string;
        break;
      case 'subtitle':
        updates.subtitle = value as string;
        break;
      case 'content':
        updates.content = value as string;
        break;
      case 'voiceoverText':
        updates.voiceoverText = value as string;
        break;
      case 'duration':
        updates.duration = Number(value);
        break;
    }

    onUpdate(updates);
    setEditingField(null);
  }, [editValues, onUpdate]);

  // Cancel editing
  const cancelEditing = useCallback(() => {
    setEditingField(null);
    setEditValues({});
  }, []);

  // Handle audio file selection
  const handleAudioChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onUploadAudio(file);
    }
    if (audioInputRef.current) {
      audioInputRef.current.value = '';
    }
  }, [onUploadAudio]);

  // Handle audio drag & drop
  const handleAudioDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingAudio(false);

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('audio/')) {
      onUploadAudio(file);
    }
  }, [onUploadAudio]);

  // Handle media drag & drop
  const handleMediaDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingMedia(false);

    const file = e.dataTransfer.files[0];
    if (file && onUploadMedia) {
      if (file.type.startsWith('image/')) {
        onUploadMedia('image', file);
      } else if (file.type.startsWith('video/')) {
        onUploadMedia('video', file);
      }
    }
  }, [onUploadMedia]);

  // Handle image replacement drag & drop
  const handleImageDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingImage(false);

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/') && onUploadMedia) {
      onUploadMedia('image', file);
    }
  }, [onUploadMedia]);

  // Handle image file selection
  const handleImageChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && onUploadMedia) {
      onUploadMedia('image', file);
    }
    if (imageInputRef.current) {
      imageInputRef.current.value = '';
    }
  }, [onUploadMedia]);

  // Start editing bullet points
  const startEditingBulletPoints = useCallback(() => {
    if (isReadOnly || !slide) return;
    setBulletPointsEdit([...slide.bulletPoints]);
    setEditingBulletPoints(true);
  }, [isReadOnly, slide]);

  // Save bullet points
  const saveBulletPoints = useCallback(() => {
    const filtered = bulletPointsEdit.filter(bp => bp.trim() !== '');
    onUpdate({ bulletPoints: filtered });
    setEditingBulletPoints(false);
  }, [bulletPointsEdit, onUpdate]);

  // Cancel bullet point editing
  const cancelBulletPointEditing = useCallback(() => {
    setEditingBulletPoints(false);
    setBulletPointsEdit([]);
  }, []);

  // Add bullet point
  const addBulletPoint = useCallback(() => {
    setBulletPointsEdit([...bulletPointsEdit, '']);
  }, [bulletPointsEdit]);

  // Remove bullet point
  const removeBulletPoint = useCallback((index: number) => {
    setBulletPointsEdit(bulletPointsEdit.filter((_, i) => i !== index));
  }, [bulletPointsEdit]);

  // Update bullet point
  const updateBulletPoint = useCallback((index: number, value: string) => {
    const updated = [...bulletPointsEdit];
    updated[index] = value;
    setBulletPointsEdit(updated);
  }, [bulletPointsEdit]);

  // Move bullet point up
  const moveBulletPointUp = useCallback((index: number) => {
    if (index === 0) return;
    const updated = [...bulletPointsEdit];
    [updated[index - 1], updated[index]] = [updated[index], updated[index - 1]];
    setBulletPointsEdit(updated);
  }, [bulletPointsEdit]);

  // Move bullet point down
  const moveBulletPointDown = useCallback((index: number) => {
    if (index >= bulletPointsEdit.length - 1) return;
    const updated = [...bulletPointsEdit];
    [updated[index], updated[index + 1]] = [updated[index + 1], updated[index]];
    setBulletPointsEdit(updated);
  }, [bulletPointsEdit]);

  // Start editing code block
  const startEditingCodeBlock = useCallback((codeBlock: CodeBlockComponent, index: number) => {
    if (isReadOnly) return;
    setEditingCodeBlock(codeBlock);
    setCodeBlockIndex(index);
  }, [isReadOnly]);

  // Save code block
  const saveCodeBlock = useCallback((updated: CodeBlockComponent) => {
    if (!slide) return;
    const newCodeBlocks = [...slide.codeBlocks];
    newCodeBlocks[codeBlockIndex] = updated;
    onUpdate({ codeBlocks: newCodeBlocks });
    setEditingCodeBlock(null);
  }, [slide, codeBlockIndex, onUpdate]);

  // Handler for EditableField value changes
  const handleEditValueChange = useCallback((field: string, value: string | number) => {
    setEditValues(prev => ({ ...prev, [field]: value }));
  }, []);

  // Helper to render EditableField with common props
  const renderEditableField = (
    field: string,
    value: string | number | undefined,
    label: string,
    options?: {
      multiline?: boolean;
      type?: 'text' | 'number';
      suffix?: string;
      min?: number;
      max?: number;
    }
  ) => (
    <EditableField
      key={field}
      field={field}
      value={value}
      label={label}
      multiline={options?.multiline}
      type={options?.type}
      suffix={options?.suffix}
      min={options?.min}
      max={options?.max}
      isEditing={editingField === field}
      editValue={editValues[field]}
      isReadOnly={isReadOnly}
      isSaving={isSaving}
      onStartEditing={startEditing}
      onEditValueChange={handleEditValueChange}
      onSave={saveField}
      onCancel={cancelEditing}
    />
  );

  if (!slide) {
    return (
      <div className="p-4 h-full flex items-center justify-center">
        <div className="text-center text-gray-500">
          <svg className="w-12 h-12 mx-auto mb-3 opacity-30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
          </svg>
          <p className="text-sm">Sélectionnez un slide pour modifier ses propriétés</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 h-full overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-white font-semibold text-sm">Propriétés</h3>
        {slide.isEdited && (
          <span className="px-2 py-0.5 bg-yellow-500/20 text-yellow-400 text-xs rounded-full">
            Modifié
          </span>
        )}
      </div>

      {/* Slide type info */}
      <div className="bg-gray-800/50 rounded-lg p-3 mb-4">
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-400">Type</span>
          <span className="text-white font-medium">{getSlideTypeLabel(slide.type)}</span>
        </div>
        <div className="flex items-center justify-between text-xs mt-1">
          <span className="text-gray-400">Index</span>
          <span className="text-white">{slide.index + 1}</span>
        </div>
      </div>

      {/* Editable fields */}
      <div className="space-y-4">
        {/* Title */}
        {renderEditableField('title', slide.title, 'Titre')}

        {/* Subtitle (optional) */}
        {(slide.type === 'title' || slide.type === 'content') &&
          renderEditableField('subtitle', slide.subtitle, 'Sous-titre')
        }

        {/* Duration */}
        {renderEditableField('duration', slide.duration, 'Durée', {
          type: 'number',
          suffix: ' sec',
          min: 1,
          max: 300,
        })}

        {/* Content (for content slides) */}
        {(slide.type === 'content' || slide.type === 'split') &&
          renderEditableField('content', slide.content, 'Contenu', { multiline: true })
        }

        {/* Voiceover text */}
        {renderEditableField('voiceoverText', slide.voiceoverText, 'Texte du voiceover', { multiline: true })}

        {/* Bullet points editable */}
        {(slide.bulletPoints.length > 0 || !isReadOnly) && (
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-gray-400 text-xs font-medium">Points clés</label>
              {!isReadOnly && !editingBulletPoints && (
                <button
                  onClick={startEditingBulletPoints}
                  className="text-purple-400 text-xs hover:text-purple-300 transition-colors"
                >
                  Modifier
                </button>
              )}
            </div>

            {editingBulletPoints ? (
              <div className="bg-gray-800 rounded-lg p-3 space-y-2">
                {bulletPointsEdit.map((point, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    {/* Move buttons */}
                    <div className="flex flex-col">
                      <button
                        onClick={() => moveBulletPointUp(idx)}
                        disabled={idx === 0}
                        className="p-0.5 text-gray-500 hover:text-gray-300 disabled:opacity-30 disabled:cursor-not-allowed"
                      >
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                        </svg>
                      </button>
                      <button
                        onClick={() => moveBulletPointDown(idx)}
                        disabled={idx >= bulletPointsEdit.length - 1}
                        className="p-0.5 text-gray-500 hover:text-gray-300 disabled:opacity-30 disabled:cursor-not-allowed"
                      >
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                      </button>
                    </div>

                    {/* Input */}
                    <input
                      type="text"
                      value={point}
                      onChange={(e) => updateBulletPoint(idx, e.target.value)}
                      placeholder="Point clé..."
                      className="flex-1 bg-gray-700 text-white text-xs rounded px-2 py-1.5 border border-gray-600 focus:border-purple-500 focus:outline-none"
                    />

                    {/* Remove button */}
                    <button
                      onClick={() => removeBulletPoint(idx)}
                      className="p-1 text-gray-500 hover:text-red-400 transition-colors"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                ))}

                {/* Add button */}
                <button
                  onClick={addBulletPoint}
                  className="w-full py-1.5 border border-dashed border-gray-600 rounded text-gray-500 hover:border-purple-500 hover:text-purple-400 text-xs transition-colors flex items-center justify-center gap-1"
                >
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                  Ajouter un point
                </button>

                {/* Action buttons */}
                <div className="flex gap-2 pt-2">
                  <button
                    onClick={saveBulletPoints}
                    disabled={isSaving}
                    className="flex-1 px-3 py-1.5 bg-purple-600 text-white rounded text-xs font-medium hover:bg-purple-500 disabled:opacity-50 transition-colors"
                  >
                    {isSaving ? 'Sauvegarde...' : 'Enregistrer'}
                  </button>
                  <button
                    onClick={cancelBulletPointEditing}
                    className="px-3 py-1.5 bg-gray-700 text-white rounded text-xs hover:bg-gray-600 transition-colors"
                  >
                    Annuler
                  </button>
                </div>
              </div>
            ) : slide.bulletPoints.length > 0 ? (
              <div
                onClick={() => !isReadOnly && startEditingBulletPoints()}
                className={`bg-gray-800 rounded-lg p-3 ${!isReadOnly ? 'cursor-pointer hover:bg-gray-750 hover:ring-1 hover:ring-purple-500/50' : ''}`}
              >
                <ul className="space-y-1">
                  {slide.bulletPoints.map((point, idx) => (
                    <li key={idx} className="text-white text-xs flex items-start gap-2">
                      <span className="text-purple-500 mt-0.5">•</span>
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ) : !isReadOnly && (
              <button
                onClick={startEditingBulletPoints}
                className="w-full py-3 border border-dashed border-gray-700 rounded-lg text-gray-500 hover:border-purple-500 hover:text-purple-400 text-xs transition-colors"
              >
                + Ajouter des points clés
              </button>
            )}
          </div>
        )}

        {/* Code blocks editable (for code slides) */}
        {(slide.type === 'code' || slide.type === 'code_demo') && slide.codeBlocks.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-gray-400 text-xs font-medium">Code</label>
              {slide.codeBlocks.length > 1 && (
                <span className="text-gray-500 text-xs">{slide.codeBlocks.length} blocs</span>
              )}
            </div>
            <div className="space-y-2">
              {slide.codeBlocks.map((codeBlock, idx) => (
                <div key={codeBlock.id} className="relative group">
                  <div
                    onClick={() => !isReadOnly && startEditingCodeBlock(codeBlock, idx)}
                    className={`bg-gray-900 rounded-lg p-3 max-h-32 overflow-y-auto ${!isReadOnly ? 'cursor-pointer hover:ring-1 hover:ring-purple-500/50' : ''}`}
                  >
                    {/* Language & filename badge */}
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-purple-400 font-medium">
                        {codeBlock.language}
                        {codeBlock.filename && (
                          <span className="text-gray-500 ml-2">{codeBlock.filename}</span>
                        )}
                      </span>
                      {!isReadOnly && (
                        <span className="text-purple-400 text-xs opacity-0 group-hover:opacity-100 transition-opacity">
                          Cliquer pour modifier
                        </span>
                      )}
                    </div>
                    <pre className="text-green-400 text-xs font-mono">
                      <code>{codeBlock.code.slice(0, 200)}</code>
                      {codeBlock.code.length > 200 && (
                        <span className="text-gray-500">...</span>
                      )}
                    </pre>
                    {codeBlock.highlightLines.length > 0 && (
                      <p className="text-gray-500 text-xs mt-2">
                        Lignes en évidence: {codeBlock.highlightLines.join(', ')}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Slide Image preview & replacement */}
      {!isReadOnly && onUploadMedia && (
        <div className="mt-6 pt-4 border-t border-gray-800">
          <label className="text-gray-400 text-xs font-medium block mb-2">Image du slide</label>
          <input
            ref={imageInputRef}
            type="file"
            accept={MEDIA_UPLOAD_CONFIG.image.accept}
            onChange={handleImageChange}
            className="hidden"
          />

          {slide.imageUrl ? (
            <div className="relative group">
              {/* Image preview */}
              <div className="aspect-video bg-gray-800 rounded-lg overflow-hidden">
                <img
                  src={slide.imageUrl}
                  alt="Slide preview"
                  className="w-full h-full object-cover"
                />
              </div>

              {/* Overlay with replace button */}
              <div
                onClick={() => imageInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setIsDraggingImage(true); }}
                onDragLeave={() => setIsDraggingImage(false)}
                onDrop={handleImageDrop}
                className={`
                  absolute inset-0 flex items-center justify-center rounded-lg cursor-pointer transition-all
                  ${isDraggingImage
                    ? 'bg-purple-500/30 border-2 border-purple-500'
                    : 'bg-black/50 opacity-0 group-hover:opacity-100'
                  }
                `}
              >
                <div className="text-center">
                  <svg className="w-8 h-8 mx-auto text-white mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <span className="text-white text-xs font-medium">Remplacer l'image</span>
                </div>
              </div>
            </div>
          ) : (
            <div
              onClick={() => imageInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setIsDraggingImage(true); }}
              onDragLeave={() => setIsDraggingImage(false)}
              onDrop={handleImageDrop}
              className={`
                border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors
                ${isDraggingImage
                  ? 'border-purple-500 bg-purple-500/10'
                  : 'border-gray-700 hover:border-gray-600'
                }
              `}
            >
              <svg className="w-10 h-10 mx-auto text-gray-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p className="text-gray-400 text-xs">
                Glissez-déposez ou cliquez pour ajouter une image
              </p>
              <p className="text-gray-500 text-xs mt-1">
                JPG, PNG, GIF, WebP (max {MEDIA_UPLOAD_CONFIG.image.maxSizeMB}MB)
              </p>
            </div>
          )}
        </div>
      )}

      {/* Audio upload */}
      {!isReadOnly && (
        <div className="mt-6 pt-4 border-t border-gray-800">
          <label className="text-gray-400 text-xs font-medium block mb-2">Audio personnalisé</label>
          <input
            ref={audioInputRef}
            type="file"
            accept={MEDIA_UPLOAD_CONFIG.audio.accept}
            onChange={handleAudioChange}
            className="hidden"
          />
          <div
            onClick={() => audioInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setIsDraggingAudio(true); }}
            onDragLeave={() => setIsDraggingAudio(false)}
            onDrop={handleAudioDrop}
            className={`
              border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors
              ${isDraggingAudio
                ? 'border-purple-500 bg-purple-500/10'
                : 'border-gray-700 hover:border-gray-600'
              }
            `}
          >
            {voiceover?.isCustomAudio ? (
              <div className="flex items-center justify-center gap-2 text-green-400">
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
                </svg>
                <span className="text-sm">{voiceover.originalFilename || 'Audio personnalisé'}</span>
              </div>
            ) : (
              <>
                <svg className="w-8 h-8 mx-auto text-gray-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                </svg>
                <p className="text-gray-400 text-xs">
                  Glissez-déposez ou cliquez pour uploader
                </p>
                <p className="text-gray-500 text-xs mt-1">
                  MP3, WAV, M4A (max {MEDIA_UPLOAD_CONFIG.audio.maxSizeMB}MB)
                </p>
              </>
            )}
          </div>
        </div>
      )}

      {/* Media upload */}
      {!isReadOnly && onUploadMedia && (
        <div className="mt-4">
          <label className="text-gray-400 text-xs font-medium block mb-2">Ajouter un média</label>
          <input
            ref={mediaInputRef}
            type="file"
            accept={`${MEDIA_UPLOAD_CONFIG.image.accept},${MEDIA_UPLOAD_CONFIG.video.accept}`}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) {
                if (file.type.startsWith('image/')) {
                  onUploadMedia('image', file);
                } else if (file.type.startsWith('video/')) {
                  onUploadMedia('video', file);
                }
              }
              if (mediaInputRef.current) {
                mediaInputRef.current.value = '';
              }
            }}
            className="hidden"
          />
          <div
            onClick={() => mediaInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setIsDraggingMedia(true); }}
            onDragLeave={() => setIsDraggingMedia(false)}
            onDrop={handleMediaDrop}
            className={`
              border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors
              ${isDraggingMedia
                ? 'border-purple-500 bg-purple-500/10'
                : 'border-gray-700 hover:border-gray-600'
              }
            `}
          >
            <svg className="w-8 h-8 mx-auto text-gray-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <p className="text-gray-400 text-xs">Image ou vidéo</p>
          </div>
        </div>
      )}

      {/* Regenerate button */}
      {!isReadOnly && (
        <div className="mt-6 pt-4 border-t border-gray-800">
          <button
            onClick={onRegenerate}
            disabled={isRegenerating || !slide.isEdited}
            className="w-full px-4 py-2.5 bg-purple-600 text-white rounded-lg hover:bg-purple-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
          >
            {isRegenerating ? (
              <>
                <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span>Régénération...</span>
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                <span>Régénérer ce slide</span>
              </>
            )}
          </button>
          {!slide.isEdited && (
            <p className="text-gray-500 text-xs text-center mt-2">
              Modifiez le contenu pour activer la régénération
            </p>
          )}
        </div>
      )}

      {/* Edit history */}
      {slide.editedFields.length > 0 && (
        <div className="mt-4 text-xs text-gray-500">
          <p className="flex items-center gap-1">
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Champs modifiés: {slide.editedFields.join(', ')}
          </p>
          {slide.editedAt && (
            <p className="mt-1">
              Dernière modification: {new Date(slide.editedAt).toLocaleString('fr-FR')}
            </p>
          )}
        </div>
      )}

      {/* Code block editor modal */}
      {editingCodeBlock && (
        <CodeBlockEditor
          codeBlock={editingCodeBlock}
          onChange={setEditingCodeBlock}
          onClose={() => setEditingCodeBlock(null)}
          onSave={() => {
            if (editingCodeBlock) {
              saveCodeBlock(editingCodeBlock);
            }
          }}
        />
      )}
    </div>
  );
}

export default SlideProperties;
