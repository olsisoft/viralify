'use client';

import React, { useState, useCallback, useRef } from 'react';
import type { AudioTrack, AudioMixerState } from '../../lib/lecture-editor-types';
import { formatDuration } from '../../lib/lecture-editor-types';

interface AudioMixerProps {
  state: AudioMixerState;
  onStateChange: (state: AudioMixerState) => void;
  onAddTrack: (file: File, type: 'music' | 'sfx') => void;
}

export function AudioMixer({
  state,
  onStateChange,
  onAddTrack,
}: AudioMixerProps) {
  const [draggedTrackId, setDraggedTrackId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [pendingTrackType, setPendingTrackType] = useState<'music' | 'sfx'>('music');

  // Update master volume
  const updateMasterVolume = useCallback((volume: number) => {
    onStateChange({ ...state, masterVolume: volume });
  }, [state, onStateChange]);

  // Update track
  const updateTrack = useCallback((trackId: string, updates: Partial<AudioTrack>) => {
    onStateChange({
      ...state,
      tracks: state.tracks.map(t => t.id === trackId ? { ...t, ...updates } : t),
    });
  }, [state, onStateChange]);

  // Delete track
  const deleteTrack = useCallback((trackId: string) => {
    onStateChange({
      ...state,
      tracks: state.tracks.filter(t => t.id !== trackId),
    });
  }, [state, onStateChange]);

  // Solo track
  const soloTrack = useCallback((trackId: string) => {
    const track = state.tracks.find(t => t.id === trackId);
    if (!track) return;

    const newSoloState = !track.isSolo;
    onStateChange({
      ...state,
      tracks: state.tracks.map(t => ({
        ...t,
        isSolo: t.id === trackId ? newSoloState : false,
        isMuted: newSoloState && t.id !== trackId,
      })),
    });
  }, [state, onStateChange]);

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onAddTrack(file, pendingTrackType);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Open file picker
  const openFilePicker = (type: 'music' | 'sfx') => {
    setPendingTrackType(type);
    fileInputRef.current?.click();
  };

  // Group tracks by type
  const voiceoverTracks = state.tracks.filter(t => t.type === 'voiceover');
  const musicTracks = state.tracks.filter(t => t.type === 'music');
  const sfxTracks = state.tracks.filter(t => t.type === 'sfx');

  return (
    <div className="flex flex-col h-full bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <h3 className="text-white font-semibold text-sm flex items-center gap-2">
          <span>🎚️</span>
          Mixeur Audio
        </h3>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Master Volume */}
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-white text-sm font-medium">Volume Master</span>
            <span className="text-purple-400 text-sm font-mono">{Math.round(state.masterVolume * 100)}%</span>
          </div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={state.masterVolume}
            onChange={(e) => updateMasterVolume(parseFloat(e.target.value))}
            className="w-full h-3 bg-gray-700 rounded-full appearance-none cursor-pointer
              [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:h-5
              [&::-webkit-slider-thumb]:bg-purple-500 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:shadow-lg"
          />
          {/* Level meter visualization */}
          <div className="flex gap-1 mt-3 h-2">
            {Array.from({ length: 20 }).map((_, i) => (
              <div
                key={i}
                className={`flex-1 rounded-sm transition-colors ${
                  i < state.masterVolume * 20
                    ? i < 14 ? 'bg-green-500' : i < 18 ? 'bg-yellow-500' : 'bg-red-500'
                    : 'bg-gray-700'
                }`}
              />
            ))}
          </div>
        </div>

        {/* Quick Balance */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h4 className="text-white text-sm font-medium mb-3">Balance rapide</h4>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <span className="text-gray-400 text-xs w-16">Voix</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={state.voiceoverVolume}
                onChange={(e) => onStateChange({ ...state, voiceoverVolume: parseFloat(e.target.value) })}
                className="flex-1 h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer
                  [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3
                  [&::-webkit-slider-thumb]:bg-blue-500 [&::-webkit-slider-thumb]:rounded-full"
              />
              <span className="text-gray-300 text-xs w-10 text-right">{Math.round(state.voiceoverVolume * 100)}%</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-gray-400 text-xs w-16">Musique</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={state.musicVolume}
                onChange={(e) => onStateChange({ ...state, musicVolume: parseFloat(e.target.value) })}
                className="flex-1 h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer
                  [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3
                  [&::-webkit-slider-thumb]:bg-green-500 [&::-webkit-slider-thumb]:rounded-full"
              />
              <span className="text-gray-300 text-xs w-10 text-right">{Math.round(state.musicVolume * 100)}%</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-gray-400 text-xs w-16">Effets</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={state.sfxVolume}
                onChange={(e) => onStateChange({ ...state, sfxVolume: parseFloat(e.target.value) })}
                className="flex-1 h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer
                  [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3
                  [&::-webkit-slider-thumb]:bg-yellow-500 [&::-webkit-slider-thumb]:rounded-full"
              />
              <span className="text-gray-300 text-xs w-10 text-right">{Math.round(state.sfxVolume * 100)}%</span>
            </div>
          </div>
        </div>

        {/* Voiceover Tracks */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-blue-400 text-xs font-medium uppercase tracking-wider">Voiceover</h4>
            <span className="text-gray-500 text-xs">{voiceoverTracks.length} piste(s)</span>
          </div>
          <div className="space-y-2">
            {voiceoverTracks.map(track => (
              <TrackMixerItem
                key={track.id}
                track={track}
                color="blue"
                onUpdate={(updates) => updateTrack(track.id, updates)}
                onSolo={() => soloTrack(track.id)}
                onDelete={() => deleteTrack(track.id)}
                canDelete={false}
              />
            ))}
            {voiceoverTracks.length === 0 && (
              <p className="text-gray-500 text-xs italic py-2">Aucune piste voiceover</p>
            )}
          </div>
        </div>

        {/* Music Tracks */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-green-400 text-xs font-medium uppercase tracking-wider">Musique</h4>
            <button
              onClick={() => openFilePicker('music')}
              className="text-green-400 text-xs hover:text-green-300"
            >
              + Ajouter
            </button>
          </div>
          <div className="space-y-2">
            {musicTracks.map(track => (
              <TrackMixerItem
                key={track.id}
                track={track}
                color="green"
                onUpdate={(updates) => updateTrack(track.id, updates)}
                onSolo={() => soloTrack(track.id)}
                onDelete={() => deleteTrack(track.id)}
                canDelete={true}
              />
            ))}
            {musicTracks.length === 0 && (
              <button
                onClick={() => openFilePicker('music')}
                className="w-full py-3 border border-dashed border-gray-700 rounded-lg text-gray-500 hover:border-green-500 hover:text-green-400 text-xs transition-colors"
              >
                Ajouter une musique de fond
              </button>
            )}
          </div>
        </div>

        {/* SFX Tracks */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-yellow-400 text-xs font-medium uppercase tracking-wider">Effets sonores</h4>
            <button
              onClick={() => openFilePicker('sfx')}
              className="text-yellow-400 text-xs hover:text-yellow-300"
            >
              + Ajouter
            </button>
          </div>
          <div className="space-y-2">
            {sfxTracks.map(track => (
              <TrackMixerItem
                key={track.id}
                track={track}
                color="yellow"
                onUpdate={(updates) => updateTrack(track.id, updates)}
                onSolo={() => soloTrack(track.id)}
                onDelete={() => deleteTrack(track.id)}
                canDelete={true}
              />
            ))}
            {sfxTracks.length === 0 && (
              <button
                onClick={() => openFilePicker('sfx')}
                className="w-full py-3 border border-dashed border-gray-700 rounded-lg text-gray-500 hover:border-yellow-500 hover:text-yellow-400 text-xs transition-colors"
              >
                Ajouter un effet sonore
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  );
}

// Track mixer item component
interface TrackMixerItemProps {
  track: AudioTrack;
  color: 'blue' | 'green' | 'yellow';
  onUpdate: (updates: Partial<AudioTrack>) => void;
  onSolo: () => void;
  onDelete: () => void;
  canDelete: boolean;
}

function TrackMixerItem({
  track,
  color,
  onUpdate,
  onSolo,
  onDelete,
  canDelete,
}: TrackMixerItemProps) {
  const [showDetails, setShowDetails] = useState(false);

  const colorClasses = {
    blue: {
      bg: 'bg-blue-500/20',
      border: 'border-blue-500/50',
      text: 'text-blue-400',
      slider: '[&::-webkit-slider-thumb]:bg-blue-500',
    },
    green: {
      bg: 'bg-green-500/20',
      border: 'border-green-500/50',
      text: 'text-green-400',
      slider: '[&::-webkit-slider-thumb]:bg-green-500',
    },
    yellow: {
      bg: 'bg-yellow-500/20',
      border: 'border-yellow-500/50',
      text: 'text-yellow-400',
      slider: '[&::-webkit-slider-thumb]:bg-yellow-500',
    },
  };

  const classes = colorClasses[color];

  return (
    <div className={`${classes.bg} border ${classes.border} rounded-lg p-3`}>
      <div className="flex items-center gap-3">
        {/* Track name */}
        <div className="flex-1 min-w-0">
          <p className={`${classes.text} text-sm font-medium truncate`}>{track.name}</p>
          <p className="text-gray-500 text-xs">{formatDuration(track.duration)}</p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-1">
          {/* Mute */}
          <button
            onClick={() => onUpdate({ isMuted: !track.isMuted })}
            className={`w-6 h-6 flex items-center justify-center rounded text-xs font-bold transition-colors ${
              track.isMuted ? 'bg-red-500/30 text-red-400' : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
            }`}
            title="Mute"
          >
            M
          </button>
          {/* Solo */}
          <button
            onClick={onSolo}
            className={`w-6 h-6 flex items-center justify-center rounded text-xs font-bold transition-colors ${
              track.isSolo ? 'bg-yellow-500/30 text-yellow-400' : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
            }`}
            title="Solo"
          >
            S
          </button>
          {/* Details toggle */}
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="w-6 h-6 flex items-center justify-center rounded text-gray-400 hover:bg-gray-700 transition-colors"
          >
            <svg className={`w-4 h-4 transition-transform ${showDetails ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
      </div>

      {/* Volume slider */}
      <div className="flex items-center gap-2 mt-2">
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={track.volume}
          onChange={(e) => onUpdate({ volume: parseFloat(e.target.value) })}
          className={`flex-1 h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer
            [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3
            [&::-webkit-slider-thumb]:rounded-full ${classes.slider}`}
        />
        <span className="text-gray-300 text-xs w-10 text-right">{Math.round(track.volume * 100)}%</span>
      </div>

      {/* Expanded details */}
      {showDetails && (
        <div className="mt-3 pt-3 border-t border-gray-700 space-y-3">
          {/* Fade In */}
          <div className="flex items-center gap-2">
            <span className="text-gray-400 text-xs w-16">Fade In</span>
            <input
              type="range"
              min={0}
              max={5}
              step={0.1}
              value={track.fadeIn}
              onChange={(e) => onUpdate({ fadeIn: parseFloat(e.target.value) })}
              className="flex-1 h-1 bg-gray-700 rounded-full appearance-none cursor-pointer"
            />
            <span className="text-gray-300 text-xs w-10 text-right">{track.fadeIn}s</span>
          </div>

          {/* Fade Out */}
          <div className="flex items-center gap-2">
            <span className="text-gray-400 text-xs w-16">Fade Out</span>
            <input
              type="range"
              min={0}
              max={5}
              step={0.1}
              value={track.fadeOut}
              onChange={(e) => onUpdate({ fadeOut: parseFloat(e.target.value) })}
              className="flex-1 h-1 bg-gray-700 rounded-full appearance-none cursor-pointer"
            />
            <span className="text-gray-300 text-xs w-10 text-right">{track.fadeOut}s</span>
          </div>

          {/* Start time */}
          <div className="flex items-center gap-2">
            <span className="text-gray-400 text-xs w-16">Début</span>
            <input
              type="number"
              min={0}
              step={0.1}
              value={track.startTime}
              onChange={(e) => onUpdate({ startTime: parseFloat(e.target.value) || 0 })}
              className="flex-1 bg-gray-700 text-white text-xs rounded px-2 py-1 border border-gray-600 focus:border-purple-500 focus:outline-none"
            />
            <span className="text-gray-400 text-xs">sec</span>
          </div>

          {/* Delete button */}
          {canDelete && (
            <button
              onClick={onDelete}
              className="w-full py-1.5 text-xs bg-red-500/20 text-red-400 rounded hover:bg-red-500/30 transition-colors"
            >
              Supprimer cette piste
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export default AudioMixer;
