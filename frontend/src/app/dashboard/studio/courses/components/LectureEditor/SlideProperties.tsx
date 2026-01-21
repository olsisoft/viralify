'use client';

import React, { useState, useCallback, useRef } from 'react';
import type { SlideComponent, VoiceoverComponent, UpdateSlideRequest } from '../../lib/lecture-editor-types';
import { getSlideTypeLabel, formatDuration } from '../../lib/lecture-editor-types';

interface SlidePropertiesProps {
  slide: SlideComponent | null;
  voiceover?: VoiceoverComponent;
  isSaving: boolean;
  isRegenerating: boolean;
  onUpdate: (updates: UpdateSlideRequest) => void;
  onRegenerate: () => void;
  onUploadAudio: (file: File) => void;
}

export function SlideProperties({
  slide,
  voiceover,
  isSaving,
  isRegenerating,
  onUpdate,
  onRegenerate,
  onUploadAudio,
}: SlidePropertiesProps) {
  const [editedTitle, setEditedTitle] = useState('');
  const [editedContent, setEditedContent] = useState('');
  const [editedVoiceover, setEditedVoiceover] = useState('');
  const [editedDuration, setEditedDuration] = useState(10);
  const [isEditing, setIsEditing] = useState<'title' | 'content' | 'voiceover' | 'duration' | null>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);

  // Start editing a field
  const startEditing = useCallback((field: 'title' | 'content' | 'voiceover' | 'duration') => {
    if (!slide) return;
    setIsEditing(field);
    switch (field) {
      case 'title':
        setEditedTitle(slide.title || '');
        break;
      case 'content':
        setEditedContent(slide.content || '');
        break;
      case 'voiceover':
        setEditedVoiceover(slide.voiceoverText || '');
        break;
      case 'duration':
        setEditedDuration(slide.duration);
        break;
    }
  }, [slide]);

  // Save edited field
  const saveField = useCallback((field: 'title' | 'content' | 'voiceover' | 'duration') => {
    const updates: UpdateSlideRequest = {};
    switch (field) {
      case 'title':
        updates.title = editedTitle;
        break;
      case 'content':
        updates.content = editedContent;
        break;
      case 'voiceover':
        updates.voiceoverText = editedVoiceover;
        break;
      case 'duration':
        updates.duration = editedDuration;
        break;
    }
    onUpdate(updates);
    setIsEditing(null);
  }, [editedTitle, editedContent, editedVoiceover, editedDuration, onUpdate]);

  // Cancel editing
  const cancelEditing = useCallback(() => {
    setIsEditing(null);
  }, []);

  // Handle audio file selection
  const handleAudioChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onUploadAudio(file);
    }
  }, [onUploadAudio]);

  if (!slide) {
    return (
      <div className="p-4 text-gray-500 text-center">
        <p>S\u00e9lectionnez un slide pour voir ses propri\u00e9t\u00e9s</p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-6">
      <h3 className="text-white font-semibold">Propri\u00e9t\u00e9s du slide</h3>

      {/* Slide type info */}
      <div className="bg-gray-800 rounded-lg p-3">
        <div className="flex items-center justify-between">
          <span className="text-gray-400 text-sm">Type</span>
          <span className="text-white">{getSlideTypeLabel(slide.type)}</span>
        </div>
        <div className="flex items-center justify-between mt-2">
          <span className="text-gray-400 text-sm">Index</span>
          <span className="text-white">{slide.index + 1}</span>
        </div>
      </div>

      {/* Title */}
      <div>
        <label className="text-gray-400 text-sm block mb-2">Titre</label>
        {isEditing === 'title' ? (
          <div className="space-y-2">
            <input
              type="text"
              value={editedTitle}
              onChange={(e) => setEditedTitle(e.target.value)}
              className="w-full bg-gray-800 text-white rounded-lg px-3 py-2 border border-gray-700 focus:border-purple-500 focus:outline-none"
              autoFocus
            />
            <div className="flex gap-2">
              <button
                onClick={() => saveField('title')}
                disabled={isSaving}
                className="flex-1 px-3 py-1.5 bg-purple-600 text-white rounded text-sm hover:bg-purple-500 disabled:opacity-50"
              >
                {isSaving ? 'Enregistrement...' : 'Enregistrer'}
              </button>
              <button
                onClick={cancelEditing}
                className="px-3 py-1.5 bg-gray-700 text-white rounded text-sm hover:bg-gray-600"
              >
                Annuler
              </button>
            </div>
          </div>
        ) : (
          <div
            onClick={() => startEditing('title')}
            className="bg-gray-800 text-white rounded-lg px-3 py-2 cursor-pointer hover:bg-gray-700"
          >
            {slide.title || <span className="text-gray-500 italic">Cliquer pour ajouter</span>}
          </div>
        )}
      </div>

      {/* Duration */}
      <div>
        <label className="text-gray-400 text-sm block mb-2">Dur\u00e9e</label>
        {isEditing === 'duration' ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <input
                type="number"
                value={editedDuration}
                onChange={(e) => setEditedDuration(Number(e.target.value))}
                min={1}
                max={300}
                className="w-24 bg-gray-800 text-white rounded-lg px-3 py-2 border border-gray-700 focus:border-purple-500 focus:outline-none"
                autoFocus
              />
              <span className="text-gray-400">secondes</span>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => saveField('duration')}
                disabled={isSaving}
                className="flex-1 px-3 py-1.5 bg-purple-600 text-white rounded text-sm hover:bg-purple-500 disabled:opacity-50"
              >
                {isSaving ? 'Enregistrement...' : 'Enregistrer'}
              </button>
              <button
                onClick={cancelEditing}
                className="px-3 py-1.5 bg-gray-700 text-white rounded text-sm hover:bg-gray-600"
              >
                Annuler
              </button>
            </div>
          </div>
        ) : (
          <div
            onClick={() => startEditing('duration')}
            className="bg-gray-800 text-white rounded-lg px-3 py-2 cursor-pointer hover:bg-gray-700"
          >
            {formatDuration(slide.duration)}
          </div>
        )}
      </div>

      {/* Voiceover text */}
      <div>
        <label className="text-gray-400 text-sm block mb-2">Texte du voiceover</label>
        {isEditing === 'voiceover' ? (
          <div className="space-y-2">
            <textarea
              value={editedVoiceover}
              onChange={(e) => setEditedVoiceover(e.target.value)}
              rows={6}
              className="w-full bg-gray-800 text-white rounded-lg px-3 py-2 border border-gray-700 focus:border-purple-500 focus:outline-none resize-none"
              autoFocus
            />
            <div className="flex gap-2">
              <button
                onClick={() => saveField('voiceover')}
                disabled={isSaving}
                className="flex-1 px-3 py-1.5 bg-purple-600 text-white rounded text-sm hover:bg-purple-500 disabled:opacity-50"
              >
                {isSaving ? 'Enregistrement...' : 'Enregistrer'}
              </button>
              <button
                onClick={cancelEditing}
                className="px-3 py-1.5 bg-gray-700 text-white rounded text-sm hover:bg-gray-600"
              >
                Annuler
              </button>
            </div>
          </div>
        ) : (
          <div
            onClick={() => startEditing('voiceover')}
            className="bg-gray-800 text-white rounded-lg px-3 py-2 cursor-pointer hover:bg-gray-700 max-h-32 overflow-y-auto"
          >
            {slide.voiceoverText || <span className="text-gray-500 italic">Cliquer pour ajouter</span>}
          </div>
        )}
      </div>

      {/* Content (for content slides) */}
      {(slide.type === 'content' || slide.type === 'split') && (
        <div>
          <label className="text-gray-400 text-sm block mb-2">Contenu</label>
          {isEditing === 'content' ? (
            <div className="space-y-2">
              <textarea
                value={editedContent}
                onChange={(e) => setEditedContent(e.target.value)}
                rows={4}
                className="w-full bg-gray-800 text-white rounded-lg px-3 py-2 border border-gray-700 focus:border-purple-500 focus:outline-none resize-none"
                autoFocus
              />
              <div className="flex gap-2">
                <button
                  onClick={() => saveField('content')}
                  disabled={isSaving}
                  className="flex-1 px-3 py-1.5 bg-purple-600 text-white rounded text-sm hover:bg-purple-500 disabled:opacity-50"
                >
                  {isSaving ? 'Enregistrement...' : 'Enregistrer'}
                </button>
                <button
                  onClick={cancelEditing}
                  className="px-3 py-1.5 bg-gray-700 text-white rounded text-sm hover:bg-gray-600"
                >
                  Annuler
                </button>
              </div>
            </div>
          ) : (
            <div
              onClick={() => startEditing('content')}
              className="bg-gray-800 text-white rounded-lg px-3 py-2 cursor-pointer hover:bg-gray-700 max-h-24 overflow-y-auto"
            >
              {slide.content || <span className="text-gray-500 italic">Cliquer pour ajouter</span>}
            </div>
          )}
        </div>
      )}

      {/* Code blocks preview (for code slides) */}
      {(slide.type === 'code' || slide.type === 'code_demo') && slide.codeBlocks.length > 0 && (
        <div>
          <label className="text-gray-400 text-sm block mb-2">Code</label>
          <div className="bg-gray-800 rounded-lg p-3 max-h-32 overflow-y-auto">
            <pre className="text-green-400 text-xs">
              <code>{slide.codeBlocks[0].code.slice(0, 300)}</code>
              {slide.codeBlocks[0].code.length > 300 && '...'}
            </pre>
          </div>
          <p className="text-gray-500 text-xs mt-1">
            L'&eacute;dition du code n\u00e9cessite la r\u00e9g\u00e9n\u00e9ration du slide
          </p>
        </div>
      )}

      {/* Custom audio upload */}
      <div>
        <label className="text-gray-400 text-sm block mb-2">Audio personnalis\u00e9</label>
        <input
          ref={audioInputRef}
          type="file"
          accept="audio/*"
          onChange={handleAudioChange}
          className="hidden"
        />
        <button
          onClick={() => audioInputRef.current?.click()}
          disabled={isRegenerating}
          className="w-full px-3 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 text-sm"
        >
          {voiceover?.isCustomAudio ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 24 24">
                <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
              </svg>
              {voiceover.originalFilename || 'Audio personnalis\u00e9'}
            </span>
          ) : (
            'Uploader un audio personnalis\u00e9'
          )}
        </button>
      </div>

      {/* Regenerate button */}
      <div className="pt-4 border-t border-gray-800">
        <button
          onClick={onRegenerate}
          disabled={isRegenerating || !slide.isEdited}
          className="w-full px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isRegenerating ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              R\u00e9g\u00e9n\u00e9ration...
            </span>
          ) : (
            'R\u00e9g\u00e9n\u00e9rer ce slide'
          )}
        </button>
        {!slide.isEdited && (
          <p className="text-gray-500 text-xs text-center mt-2">
            Modifiez le contenu pour activer la r\u00e9g\u00e9n\u00e9ration
          </p>
        )}
      </div>

      {/* Edit history */}
      {slide.editedFields.length > 0 && (
        <div className="text-xs text-gray-500">
          <p>Champs modifi\u00e9s: {slide.editedFields.join(', ')}</p>
          {slide.editedAt && (
            <p>Derni\u00e8re modification: {new Date(slide.editedAt).toLocaleString()}</p>
          )}
        </div>
      )}
    </div>
  );
}

export default SlideProperties;
