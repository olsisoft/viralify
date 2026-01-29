'use client';

import React, { useState, useCallback, useRef, useEffect } from 'react';
import type {
  SubtitleCue,
  SubtitleTrack,
  SubtitleStyle,
  SubtitlePosition,
} from '../../lib/lecture-editor-types';
import {
  DEFAULT_SUBTITLE_STYLE,
  SUBTITLE_FONTS,
  formatDuration,
} from '../../lib/lecture-editor-types';

interface SubtitlesEditorProps {
  tracks: SubtitleTrack[];
  currentTime: number;
  totalDuration: number;
  onTracksChange: (tracks: SubtitleTrack[]) => void;
  onSeek: (time: number) => void;
}

// Generate unique ID
const generateId = () => `cue-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

// Parse SRT format
function parseSRT(content: string): SubtitleCue[] {
  const cues: SubtitleCue[] = [];
  const blocks = content.trim().split(/\n\n+/);

  for (const block of blocks) {
    const lines = block.split('\n');
    if (lines.length < 3) continue;

    const timeMatch = lines[1].match(/(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})/);
    if (!timeMatch) continue;

    const startTime = parseInt(timeMatch[1]) * 3600 + parseInt(timeMatch[2]) * 60 + parseInt(timeMatch[3]) + parseInt(timeMatch[4]) / 1000;
    const endTime = parseInt(timeMatch[5]) * 3600 + parseInt(timeMatch[6]) * 60 + parseInt(timeMatch[7]) + parseInt(timeMatch[8]) / 1000;
    const text = lines.slice(2).join('\n');

    cues.push({
      id: generateId(),
      startTime,
      endTime,
      text,
      position: 'bottom',
      style: { ...DEFAULT_SUBTITLE_STYLE },
    });
  }

  return cues;
}

// Export to SRT format
function exportSRT(cues: SubtitleCue[]): string {
  return cues.map((cue, index) => {
    const formatTime = (seconds: number) => {
      const h = Math.floor(seconds / 3600);
      const m = Math.floor((seconds % 3600) / 60);
      const s = Math.floor(seconds % 60);
      const ms = Math.floor((seconds % 1) * 1000);
      return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')},${ms.toString().padStart(3, '0')}`;
    };
    return `${index + 1}\n${formatTime(cue.startTime)} --> ${formatTime(cue.endTime)}\n${cue.text}`;
  }).join('\n\n');
}

export function SubtitlesEditor({
  tracks,
  currentTime,
  totalDuration,
  onTracksChange,
  onSeek,
}: SubtitlesEditorProps) {
  const [activeTrackId, setActiveTrackId] = useState<string | null>(tracks[0]?.id || null);
  const [selectedCueId, setSelectedCueId] = useState<string | null>(null);
  const [editingCueId, setEditingCueId] = useState<string | null>(null);
  const [showStylePanel, setShowStylePanel] = useState(false);
  const [showImportExport, setShowImportExport] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);

  const activeTrack = tracks.find(t => t.id === activeTrackId);
  const selectedCue = activeTrack?.cues.find(c => c.id === selectedCueId);

  // Get current cue (for live preview)
  const currentCue = activeTrack?.cues.find(
    c => currentTime >= c.startTime && currentTime <= c.endTime
  );

  // Create new track
  const createTrack = useCallback(() => {
    const newTrack: SubtitleTrack = {
      id: `track-${Date.now()}`,
      language: 'fr',
      label: `Piste ${tracks.length + 1}`,
      cues: [],
      isDefault: tracks.length === 0,
    };
    onTracksChange([...tracks, newTrack]);
    setActiveTrackId(newTrack.id);
  }, [tracks, onTracksChange]);

  // Delete track
  const deleteTrack = useCallback((trackId: string) => {
    const newTracks = tracks.filter(t => t.id !== trackId);
    onTracksChange(newTracks);
    if (activeTrackId === trackId) {
      setActiveTrackId(newTracks[0]?.id || null);
    }
  }, [tracks, activeTrackId, onTracksChange]);

  // Add new cue at current time
  const addCue = useCallback(() => {
    if (!activeTrack) return;

    const newCue: SubtitleCue = {
      id: generateId(),
      startTime: currentTime,
      endTime: Math.min(currentTime + 3, totalDuration),
      text: '',
      position: 'bottom',
      style: { ...DEFAULT_SUBTITLE_STYLE },
    };

    const updatedTrack = {
      ...activeTrack,
      cues: [...activeTrack.cues, newCue].sort((a, b) => a.startTime - b.startTime),
    };

    onTracksChange(tracks.map(t => t.id === activeTrackId ? updatedTrack : t));
    setSelectedCueId(newCue.id);
    setEditingCueId(newCue.id);
  }, [activeTrack, activeTrackId, currentTime, totalDuration, tracks, onTracksChange]);

  // Update cue
  const updateCue = useCallback((cueId: string, updates: Partial<SubtitleCue>) => {
    if (!activeTrack) return;

    const updatedTrack = {
      ...activeTrack,
      cues: activeTrack.cues.map(c =>
        c.id === cueId ? { ...c, ...updates } : c
      ).sort((a, b) => a.startTime - b.startTime),
    };

    onTracksChange(tracks.map(t => t.id === activeTrackId ? updatedTrack : t));
  }, [activeTrack, activeTrackId, tracks, onTracksChange]);

  // Delete cue
  const deleteCue = useCallback((cueId: string) => {
    if (!activeTrack) return;

    const updatedTrack = {
      ...activeTrack,
      cues: activeTrack.cues.filter(c => c.id !== cueId),
    };

    onTracksChange(tracks.map(t => t.id === activeTrackId ? updatedTrack : t));
    if (selectedCueId === cueId) {
      setSelectedCueId(null);
    }
  }, [activeTrack, activeTrackId, selectedCueId, tracks, onTracksChange]);

  // Import SRT file
  const handleImport = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      const cues = parseSRT(content);

      if (activeTrack) {
        const updatedTrack = {
          ...activeTrack,
          cues: [...activeTrack.cues, ...cues].sort((a, b) => a.startTime - b.startTime),
        };
        onTracksChange(tracks.map(t => t.id === activeTrackId ? updatedTrack : t));
      } else {
        const newTrack: SubtitleTrack = {
          id: `track-${Date.now()}`,
          language: 'fr',
          label: file.name.replace('.srt', ''),
          cues,
          isDefault: true,
        };
        onTracksChange([...tracks, newTrack]);
        setActiveTrackId(newTrack.id);
      }
    };
    reader.readAsText(file);

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [activeTrack, activeTrackId, tracks, onTracksChange]);

  // Export SRT file
  const handleExport = useCallback(() => {
    if (!activeTrack || activeTrack.cues.length === 0) return;

    const srtContent = exportSRT(activeTrack.cues);
    const blob = new Blob([srtContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${activeTrack.label || 'subtitles'}.srt`;
    a.click();
    URL.revokeObjectURL(url);
  }, [activeTrack]);

  // Auto-generate subtitles from voiceover (placeholder)
  const autoGenerate = useCallback(() => {
    // This would call an API to transcribe the audio
    alert('La génération automatique nécessite une connexion à un service de transcription.');
  }, []);

  // Timeline click handler
  const handleTimelineClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!timelineRef.current) return;
    const rect = timelineRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const time = (x / rect.width) * totalDuration;
    onSeek(Math.max(0, Math.min(time, totalDuration)));
  }, [totalDuration, onSeek]);

  return (
    <div className="flex flex-col h-full bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <h3 className="text-white font-semibold text-sm flex items-center gap-2">
          <span>💬</span>
          Sous-titres
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={autoGenerate}
            className="px-3 py-1.5 text-xs bg-purple-600 text-white rounded-lg hover:bg-purple-500 transition-colors"
          >
            Auto-générer
          </button>
          <button
            onClick={() => setShowImportExport(!showImportExport)}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
          </button>
        </div>
      </div>

      {/* Import/Export panel */}
      {showImportExport && (
        <div className="px-4 py-3 border-b border-gray-800 bg-gray-800/50">
          <div className="flex items-center gap-2">
            <input
              ref={fileInputRef}
              type="file"
              accept=".srt,.vtt"
              onChange={handleImport}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex-1 px-3 py-2 text-xs bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors"
            >
              Importer SRT
            </button>
            <button
              onClick={handleExport}
              disabled={!activeTrack || activeTrack.cues.length === 0}
              className="flex-1 px-3 py-2 text-xs bg-gray-700 text-white rounded-lg hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Exporter SRT
            </button>
          </div>
        </div>
      )}

      {/* Track selector */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-gray-800">
        <select
          value={activeTrackId || ''}
          onChange={(e) => setActiveTrackId(e.target.value || null)}
          className="flex-1 bg-gray-800 text-white text-sm rounded px-2 py-1.5 border border-gray-700 focus:border-purple-500 focus:outline-none"
        >
          {tracks.length === 0 && (
            <option value="">Aucune piste</option>
          )}
          {tracks.map(track => (
            <option key={track.id} value={track.id}>
              {track.label} ({track.language})
            </option>
          ))}
        </select>
        <button
          onClick={createTrack}
          className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
          title="Nouvelle piste"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
        </button>
        {activeTrack && (
          <button
            onClick={() => deleteTrack(activeTrackId!)}
            className="p-1.5 text-gray-400 hover:text-red-400 hover:bg-gray-800 rounded-lg transition-colors"
            title="Supprimer la piste"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        )}
      </div>

      {/* Current subtitle preview */}
      {currentCue && (
        <div className="px-4 py-3 border-b border-gray-800 bg-purple-900/20">
          <p className="text-gray-400 text-xs mb-1">Sous-titre actuel:</p>
          <p
            className="text-center py-2 rounded"
            style={{
              fontFamily: currentCue.style.fontFamily,
              fontSize: `${Math.min(currentCue.style.fontSize, 18)}px`,
              fontWeight: currentCue.style.fontWeight,
              color: currentCue.style.color,
              backgroundColor: currentCue.style.backgroundColor,
            }}
          >
            {currentCue.text}
          </p>
        </div>
      )}

      {/* Mini timeline */}
      <div className="px-4 py-2 border-b border-gray-800">
        <div
          ref={timelineRef}
          onClick={handleTimelineClick}
          className="relative h-8 bg-gray-800 rounded cursor-pointer"
        >
          {/* Cues on timeline */}
          {activeTrack?.cues.map(cue => (
            <div
              key={cue.id}
              onClick={(e) => {
                e.stopPropagation();
                setSelectedCueId(cue.id);
                onSeek(cue.startTime);
              }}
              className={`absolute top-1 bottom-1 rounded cursor-pointer transition-colors ${
                selectedCueId === cue.id ? 'bg-purple-500' : 'bg-blue-500/70 hover:bg-blue-500'
              }`}
              style={{
                left: `${(cue.startTime / totalDuration) * 100}%`,
                width: `${((cue.endTime - cue.startTime) / totalDuration) * 100}%`,
                minWidth: '4px',
              }}
              title={cue.text}
            />
          ))}
          {/* Playhead */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-red-500"
            style={{ left: `${(currentTime / totalDuration) * 100}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>{formatDuration(0)}</span>
          <span>{formatDuration(totalDuration)}</span>
        </div>
      </div>

      {/* Cues list */}
      <div className="flex-1 overflow-y-auto">
        {!activeTrack ? (
          <div className="flex items-center justify-center h-full text-gray-500 text-sm">
            <div className="text-center">
              <p className="mb-2">Aucune piste de sous-titres</p>
              <button
                onClick={createTrack}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-500 transition-colors text-sm"
              >
                Créer une piste
              </button>
            </div>
          </div>
        ) : activeTrack.cues.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500 text-sm">
            <div className="text-center">
              <p className="mb-2">Aucun sous-titre</p>
              <button
                onClick={addCue}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-500 transition-colors text-sm"
              >
                Ajouter un sous-titre
              </button>
            </div>
          </div>
        ) : (
          <div className="divide-y divide-gray-800">
            {activeTrack.cues.map((cue, index) => (
              <CueItem
                key={cue.id}
                cue={cue}
                index={index}
                isSelected={selectedCueId === cue.id}
                isEditing={editingCueId === cue.id}
                onSelect={() => {
                  setSelectedCueId(cue.id);
                  onSeek(cue.startTime);
                }}
                onEdit={() => setEditingCueId(cue.id)}
                onSave={() => setEditingCueId(null)}
                onUpdate={(updates) => updateCue(cue.id, updates)}
                onDelete={() => deleteCue(cue.id)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Bottom toolbar */}
      <div className="flex items-center justify-between px-4 py-3 border-t border-gray-800">
        <button
          onClick={addCue}
          disabled={!activeTrack}
          className="flex items-center gap-2 px-3 py-1.5 text-sm bg-purple-600 text-white rounded-lg hover:bg-purple-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          Ajouter à {formatDuration(currentTime)}
        </button>
        <button
          onClick={() => setShowStylePanel(!showStylePanel)}
          disabled={!selectedCue}
          className="flex items-center gap-2 px-3 py-1.5 text-sm bg-gray-700 text-white rounded-lg hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
          </svg>
          Style
        </button>
      </div>

      {/* Style panel modal */}
      {showStylePanel && selectedCue && (
        <StylePanel
          style={selectedCue.style}
          position={selectedCue.position}
          onStyleChange={(style) => updateCue(selectedCue.id, { style })}
          onPositionChange={(position) => updateCue(selectedCue.id, { position })}
          onClose={() => setShowStylePanel(false)}
        />
      )}
    </div>
  );
}

// Cue item component
interface CueItemProps {
  cue: SubtitleCue;
  index: number;
  isSelected: boolean;
  isEditing: boolean;
  onSelect: () => void;
  onEdit: () => void;
  onSave: () => void;
  onUpdate: (updates: Partial<SubtitleCue>) => void;
  onDelete: () => void;
}

function CueItem({
  cue,
  index,
  isSelected,
  isEditing,
  onSelect,
  onEdit,
  onSave,
  onUpdate,
  onDelete,
}: CueItemProps) {
  const [text, setText] = useState(cue.text);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setText(cue.text);
  }, [cue.text]);

  useEffect(() => {
    if (isEditing && textareaRef.current) {
      textareaRef.current.focus();
      textareaRef.current.select();
    }
  }, [isEditing]);

  const handleSave = () => {
    onUpdate({ text });
    onSave();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSave();
    }
    if (e.key === 'Escape') {
      setText(cue.text);
      onSave();
    }
  };

  return (
    <div
      onClick={onSelect}
      className={`p-3 cursor-pointer transition-colors ${
        isSelected ? 'bg-purple-900/30' : 'hover:bg-gray-800/50'
      }`}
    >
      <div className="flex items-start gap-3">
        {/* Index */}
        <span className="text-gray-500 text-xs font-mono w-6">{index + 1}</span>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Timing */}
          <div className="flex items-center gap-2 mb-1">
            <input
              type="text"
              value={formatDuration(cue.startTime)}
              onChange={(e) => {
                // Parse time input
                const parts = e.target.value.split(':');
                if (parts.length === 2) {
                  const mins = parseInt(parts[0]) || 0;
                  const secs = parseFloat(parts[1]) || 0;
                  onUpdate({ startTime: mins * 60 + secs });
                }
              }}
              onClick={(e) => e.stopPropagation()}
              className="w-16 bg-gray-800 text-green-400 text-xs font-mono rounded px-1.5 py-0.5 border border-gray-700 focus:border-purple-500 focus:outline-none text-center"
            />
            <span className="text-gray-600">→</span>
            <input
              type="text"
              value={formatDuration(cue.endTime)}
              onChange={(e) => {
                const parts = e.target.value.split(':');
                if (parts.length === 2) {
                  const mins = parseInt(parts[0]) || 0;
                  const secs = parseFloat(parts[1]) || 0;
                  onUpdate({ endTime: mins * 60 + secs });
                }
              }}
              onClick={(e) => e.stopPropagation()}
              className="w-16 bg-gray-800 text-green-400 text-xs font-mono rounded px-1.5 py-0.5 border border-gray-700 focus:border-purple-500 focus:outline-none text-center"
            />
            <span className="text-gray-500 text-xs">
              ({((cue.endTime - cue.startTime)).toFixed(1)}s)
            </span>
          </div>

          {/* Text */}
          {isEditing ? (
            <textarea
              ref={textareaRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKeyDown}
              onBlur={handleSave}
              onClick={(e) => e.stopPropagation()}
              rows={2}
              className="w-full bg-gray-800 text-white text-sm rounded px-2 py-1.5 border-2 border-purple-500 focus:outline-none resize-none"
              placeholder="Texte du sous-titre..."
            />
          ) : (
            <p
              onDoubleClick={(e) => {
                e.stopPropagation();
                onEdit();
              }}
              className="text-white text-sm leading-snug"
            >
              {cue.text || <span className="text-gray-500 italic">Double-clic pour éditer</span>}
            </p>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1">
          {isEditing ? (
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleSave();
              }}
              className="p-1 text-green-400 hover:bg-gray-800 rounded transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </button>
          ) : (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onEdit();
              }}
              className="p-1 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
            </button>
          )}
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
            className="p-1 text-gray-400 hover:text-red-400 hover:bg-gray-800 rounded transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

// Style panel component
interface StylePanelProps {
  style: SubtitleStyle;
  position: SubtitlePosition;
  onStyleChange: (style: SubtitleStyle) => void;
  onPositionChange: (position: SubtitlePosition) => void;
  onClose: () => void;
}

function StylePanel({
  style,
  position,
  onStyleChange,
  onPositionChange,
  onClose,
}: StylePanelProps) {
  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 rounded-xl border border-gray-700 shadow-2xl w-full max-w-md">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
          <h4 className="text-white font-semibold text-sm">Style du sous-titre</h4>
          <button
            onClick={onClose}
            className="p-1 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4 max-h-[60vh] overflow-y-auto">
          {/* Position */}
          <div>
            <label className="text-gray-400 text-xs font-medium block mb-2">Position</label>
            <div className="flex gap-2">
              {(['top', 'middle', 'bottom'] as SubtitlePosition[]).map((pos) => (
                <button
                  key={pos}
                  onClick={() => onPositionChange(pos)}
                  className={`flex-1 py-2 rounded text-sm transition-colors ${
                    position === pos
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                >
                  {pos === 'top' ? 'Haut' : pos === 'middle' ? 'Milieu' : 'Bas'}
                </button>
              ))}
            </div>
          </div>

          {/* Font family */}
          <div>
            <label className="text-gray-400 text-xs font-medium block mb-2">Police</label>
            <select
              value={style.fontFamily}
              onChange={(e) => onStyleChange({ ...style, fontFamily: e.target.value })}
              className="w-full bg-gray-800 text-white text-sm rounded px-3 py-2 border border-gray-700 focus:border-purple-500 focus:outline-none"
            >
              {SUBTITLE_FONTS.map(font => (
                <option key={font} value={font}>{font}</option>
              ))}
            </select>
          </div>

          {/* Font size */}
          <div>
            <label className="text-gray-400 text-xs font-medium block mb-2">
              Taille: {style.fontSize}px
            </label>
            <input
              type="range"
              min={12}
              max={72}
              value={style.fontSize}
              onChange={(e) => onStyleChange({ ...style, fontSize: parseInt(e.target.value) })}
              className="w-full"
            />
          </div>

          {/* Font weight & style */}
          <div className="flex gap-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={style.fontWeight === 'bold'}
                onChange={(e) => onStyleChange({ ...style, fontWeight: e.target.checked ? 'bold' : 'normal' })}
                className="rounded border-gray-600 bg-gray-800 text-purple-600 focus:ring-purple-500"
              />
              <span className="text-gray-300 text-sm font-bold">Gras</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={style.fontStyle === 'italic'}
                onChange={(e) => onStyleChange({ ...style, fontStyle: e.target.checked ? 'italic' : 'normal' })}
                className="rounded border-gray-600 bg-gray-800 text-purple-600 focus:ring-purple-500"
              />
              <span className="text-gray-300 text-sm italic">Italique</span>
            </label>
          </div>

          {/* Colors */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-gray-400 text-xs font-medium block mb-2">Couleur du texte</label>
              <input
                type="color"
                value={style.color}
                onChange={(e) => onStyleChange({ ...style, color: e.target.value })}
                className="w-full h-10 rounded border border-gray-700 cursor-pointer"
              />
            </div>
            <div>
              <label className="text-gray-400 text-xs font-medium block mb-2">Couleur du fond</label>
              <input
                type="color"
                value={style.backgroundColor.replace(/rgba?\([^)]+\)/, '#000000')}
                onChange={(e) => onStyleChange({ ...style, backgroundColor: e.target.value + 'cc' })}
                className="w-full h-10 rounded border border-gray-700 cursor-pointer"
              />
            </div>
          </div>

          {/* Text align */}
          <div>
            <label className="text-gray-400 text-xs font-medium block mb-2">Alignement</label>
            <div className="flex gap-2">
              {(['left', 'center', 'right'] as const).map((align) => (
                <button
                  key={align}
                  onClick={() => onStyleChange({ ...style, textAlign: align })}
                  className={`flex-1 py-2 rounded text-sm transition-colors ${
                    style.textAlign === align
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                >
                  {align === 'left' ? 'Gauche' : align === 'center' ? 'Centre' : 'Droite'}
                </button>
              ))}
            </div>
          </div>

          {/* Effects */}
          <div className="flex gap-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={style.textShadow}
                onChange={(e) => onStyleChange({ ...style, textShadow: e.target.checked })}
                className="rounded border-gray-600 bg-gray-800 text-purple-600 focus:ring-purple-500"
              />
              <span className="text-gray-300 text-sm">Ombre</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={style.outline}
                onChange={(e) => onStyleChange({ ...style, outline: e.target.checked })}
                className="rounded border-gray-600 bg-gray-800 text-purple-600 focus:ring-purple-500"
              />
              <span className="text-gray-300 text-sm">Contour</span>
            </label>
          </div>

          {/* Preview */}
          <div>
            <label className="text-gray-400 text-xs font-medium block mb-2">Aperçu</label>
            <div className="bg-gray-800 rounded-lg p-4 flex items-center justify-center min-h-[80px]">
              <p
                style={{
                  fontFamily: style.fontFamily,
                  fontSize: `${Math.min(style.fontSize, 24)}px`,
                  fontWeight: style.fontWeight,
                  fontStyle: style.fontStyle,
                  color: style.color,
                  backgroundColor: style.backgroundColor,
                  textAlign: style.textAlign,
                  textShadow: style.textShadow ? '2px 2px 4px rgba(0,0,0,0.8)' : 'none',
                  WebkitTextStroke: style.outline ? `1px ${style.outlineColor}` : 'none',
                  padding: '4px 8px',
                  borderRadius: '4px',
                }}
              >
                Exemple de sous-titre
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end px-4 py-3 border-t border-gray-800">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-500 transition-colors text-sm"
          >
            Fermer
          </button>
        </div>
      </div>
    </div>
  );
}

export default SubtitlesEditor;
