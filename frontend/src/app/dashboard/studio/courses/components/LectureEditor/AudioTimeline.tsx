'use client';

import React, { useState, useCallback, useRef, useEffect } from 'react';
import type { AudioTrack, AudioMixerState } from '../../lib/lecture-editor-types';
import { formatDuration } from '../../lib/lecture-editor-types';

interface AudioTimelineProps {
  tracks: AudioTrack[];
  masterVolume: number;
  currentTime: number;
  totalDuration: number;
  isPlaying: boolean;
  onTracksChange: (tracks: AudioTrack[]) => void;
  onMasterVolumeChange: (volume: number) => void;
  onSeek: (time: number) => void;
  onAddTrack: (type: 'music' | 'sfx') => void;
}

// Generate fake waveform data for visualization
function generateWaveform(length: number): number[] {
  const waveform: number[] = [];
  for (let i = 0; i < length; i++) {
    // Generate somewhat realistic waveform pattern
    const base = 0.3 + Math.random() * 0.4;
    const spike = Math.random() > 0.9 ? 0.3 : 0;
    waveform.push(Math.min(1, base + spike));
  }
  return waveform;
}

export function AudioTimeline({
  tracks,
  masterVolume,
  currentTime,
  totalDuration,
  isPlaying,
  onTracksChange,
  onMasterVolumeChange,
  onSeek,
  onAddTrack,
}: AudioTimelineProps) {
  const [selectedTrackId, setSelectedTrackId] = useState<string | null>(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [scrollLeft, setScrollLeft] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [dragType, setDragType] = useState<'seek' | 'fade-in' | 'fade-out' | null>(null);

  const timelineRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const selectedTrack = tracks.find(t => t.id === selectedTrackId);

  // Calculate pixels per second based on zoom
  const pixelsPerSecond = 50 * zoomLevel;
  const timelineWidth = totalDuration * pixelsPerSecond;

  // Generate waveform data for tracks that don't have it
  useEffect(() => {
    const tracksNeedingWaveform = tracks.filter(t => !t.waveformData);
    if (tracksNeedingWaveform.length > 0) {
      const updatedTracks = tracks.map(track => {
        if (!track.waveformData) {
          return {
            ...track,
            waveformData: generateWaveform(Math.floor(track.duration * 10)),
          };
        }
        return track;
      });
      onTracksChange(updatedTracks);
    }
  }, [tracks, onTracksChange]);

  // Update track
  const updateTrack = useCallback((trackId: string, updates: Partial<AudioTrack>) => {
    onTracksChange(tracks.map(t => t.id === trackId ? { ...t, ...updates } : t));
  }, [tracks, onTracksChange]);

  // Delete track
  const deleteTrack = useCallback((trackId: string) => {
    onTracksChange(tracks.filter(t => t.id !== trackId));
    if (selectedTrackId === trackId) {
      setSelectedTrackId(null);
    }
  }, [tracks, selectedTrackId, onTracksChange]);

  // Handle timeline click for seeking
  const handleTimelineClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!timelineRef.current || isDragging) return;
    const rect = timelineRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left + scrollLeft;
    const time = x / pixelsPerSecond;
    onSeek(Math.max(0, Math.min(time, totalDuration)));
  }, [pixelsPerSecond, totalDuration, scrollLeft, isDragging, onSeek]);

  // Handle scroll
  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    setScrollLeft(e.currentTarget.scrollLeft);
  }, []);

  // Solo track (mute all others)
  const handleSolo = useCallback((trackId: string) => {
    const track = tracks.find(t => t.id === trackId);
    if (!track) return;

    const newSoloState = !track.isSolo;
    onTracksChange(tracks.map(t => ({
      ...t,
      isSolo: t.id === trackId ? newSoloState : false,
      isMuted: newSoloState && t.id !== trackId,
    })));
  }, [tracks, onTracksChange]);

  // Time markers
  const timeMarkers = [];
  const markerInterval = zoomLevel < 0.5 ? 10 : zoomLevel < 1 ? 5 : zoomLevel < 2 ? 2 : 1;
  for (let t = 0; t <= totalDuration; t += markerInterval) {
    timeMarkers.push(t);
  }

  return (
    <div className="flex flex-col h-full bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <h3 className="text-white font-semibold text-sm flex items-center gap-2">
          <span>🎵</span>
          Timeline Audio
        </h3>
        <div className="flex items-center gap-3">
          {/* Zoom controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => setZoomLevel(Math.max(0.25, zoomLevel / 1.5))}
              className="p-1 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
              title="Zoom arrière"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" />
              </svg>
            </button>
            <span className="text-gray-500 text-xs w-12 text-center">{Math.round(zoomLevel * 100)}%</span>
            <button
              onClick={() => setZoomLevel(Math.min(4, zoomLevel * 1.5))}
              className="p-1 text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors"
              title="Zoom avant"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v6m3-3H7" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Master volume */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-gray-800 bg-gray-800/50">
        <span className="text-gray-400 text-xs w-20">Master</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={masterVolume}
          onChange={(e) => onMasterVolumeChange(parseFloat(e.target.value))}
          className="flex-1 h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer
            [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3
            [&::-webkit-slider-thumb]:bg-purple-500 [&::-webkit-slider-thumb]:rounded-full"
        />
        <span className="text-gray-300 text-xs w-10 text-right">{Math.round(masterVolume * 100)}%</span>
      </div>

      {/* Timeline area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Track labels */}
        <div className="w-48 flex-shrink-0 border-r border-gray-800">
          {/* Time ruler header */}
          <div className="h-8 border-b border-gray-800 bg-gray-800/50" />

          {/* Track labels */}
          {tracks.map(track => (
            <div
              key={track.id}
              onClick={() => setSelectedTrackId(track.id)}
              className={`h-20 border-b border-gray-800 px-2 py-2 cursor-pointer transition-colors ${
                selectedTrackId === track.id ? 'bg-purple-900/30' : 'hover:bg-gray-800/50'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-white text-xs font-medium truncate">{track.name}</span>
                <span className={`text-xs px-1.5 py-0.5 rounded ${
                  track.type === 'voiceover' ? 'bg-blue-500/20 text-blue-400' :
                  track.type === 'music' ? 'bg-green-500/20 text-green-400' :
                  'bg-yellow-500/20 text-yellow-400'
                }`}>
                  {track.type === 'voiceover' ? 'Voix' : track.type === 'music' ? 'Musique' : 'SFX'}
                </span>
              </div>

              {/* Track controls */}
              <div className="flex items-center gap-1">
                {/* Mute */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    updateTrack(track.id, { isMuted: !track.isMuted });
                  }}
                  className={`p-1 rounded text-xs transition-colors ${
                    track.isMuted ? 'bg-red-500/20 text-red-400' : 'text-gray-400 hover:bg-gray-700'
                  }`}
                  title="Mute"
                >
                  M
                </button>
                {/* Solo */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleSolo(track.id);
                  }}
                  className={`p-1 rounded text-xs transition-colors ${
                    track.isSolo ? 'bg-yellow-500/20 text-yellow-400' : 'text-gray-400 hover:bg-gray-700'
                  }`}
                  title="Solo"
                >
                  S
                </button>
                {/* Volume slider */}
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={track.volume}
                  onChange={(e) => {
                    e.stopPropagation();
                    updateTrack(track.id, { volume: parseFloat(e.target.value) });
                  }}
                  onClick={(e) => e.stopPropagation()}
                  className="flex-1 h-1 bg-gray-700 rounded-full appearance-none cursor-pointer
                    [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-2 [&::-webkit-slider-thumb]:h-2
                    [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:rounded-full"
                />
                <span className="text-gray-500 text-xs w-8 text-right">
                  {Math.round(track.volume * 100)}%
                </span>
              </div>

              {/* Fade controls */}
              <div className="flex items-center gap-2 mt-1">
                <div className="flex items-center gap-1">
                  <span className="text-gray-500 text-xs">In:</span>
                  <input
                    type="number"
                    min={0}
                    max={5}
                    step={0.1}
                    value={track.fadeIn}
                    onChange={(e) => updateTrack(track.id, { fadeIn: parseFloat(e.target.value) || 0 })}
                    onClick={(e) => e.stopPropagation()}
                    className="w-12 bg-gray-800 text-white text-xs rounded px-1 py-0.5 border border-gray-700 focus:border-purple-500 focus:outline-none"
                  />
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-gray-500 text-xs">Out:</span>
                  <input
                    type="number"
                    min={0}
                    max={5}
                    step={0.1}
                    value={track.fadeOut}
                    onChange={(e) => updateTrack(track.id, { fadeOut: parseFloat(e.target.value) || 0 })}
                    onClick={(e) => e.stopPropagation()}
                    className="w-12 bg-gray-800 text-white text-xs rounded px-1 py-0.5 border border-gray-700 focus:border-purple-500 focus:outline-none"
                  />
                </div>
              </div>
            </div>
          ))}

          {/* Add track button */}
          <div className="p-2 border-b border-gray-800">
            <div className="flex gap-1">
              <button
                onClick={() => onAddTrack('music')}
                className="flex-1 py-1.5 text-xs bg-gray-800 text-gray-400 rounded hover:bg-gray-700 hover:text-white transition-colors"
              >
                + Musique
              </button>
              <button
                onClick={() => onAddTrack('sfx')}
                className="flex-1 py-1.5 text-xs bg-gray-800 text-gray-400 rounded hover:bg-gray-700 hover:text-white transition-colors"
              >
                + SFX
              </button>
            </div>
          </div>
        </div>

        {/* Timeline content */}
        <div
          ref={containerRef}
          className="flex-1 overflow-x-auto overflow-y-hidden"
          onScroll={handleScroll}
        >
          <div
            ref={timelineRef}
            style={{ width: `${timelineWidth}px`, minWidth: '100%' }}
            onClick={handleTimelineClick}
            className="relative"
          >
            {/* Time ruler */}
            <div className="h-8 border-b border-gray-800 bg-gray-800/50 relative">
              {timeMarkers.map(time => (
                <div
                  key={time}
                  className="absolute top-0 bottom-0 flex flex-col items-center"
                  style={{ left: `${time * pixelsPerSecond}px` }}
                >
                  <div className="h-2 w-px bg-gray-600" />
                  <span className="text-gray-500 text-xs">{formatDuration(time)}</span>
                </div>
              ))}
            </div>

            {/* Tracks */}
            {tracks.map(track => (
              <TrackWaveform
                key={track.id}
                track={track}
                pixelsPerSecond={pixelsPerSecond}
                isSelected={selectedTrackId === track.id}
                onClick={() => setSelectedTrackId(track.id)}
              />
            ))}

            {/* Playhead */}
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-red-500 z-10 pointer-events-none"
              style={{ left: `${currentTime * pixelsPerSecond}px` }}
            >
              <div className="absolute -top-0 left-1/2 -translate-x-1/2 w-3 h-3 bg-red-500 rounded-full" />
            </div>
          </div>
        </div>
      </div>

      {/* Selected track details */}
      {selectedTrack && (
        <div className="px-4 py-3 border-t border-gray-800 bg-gray-800/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <span className="text-white text-sm font-medium">{selectedTrack.name}</span>
              <span className="text-gray-500 text-xs">
                Durée: {formatDuration(selectedTrack.duration)}
              </span>
              <span className="text-gray-500 text-xs">
                Début: {formatDuration(selectedTrack.startTime)}
              </span>
            </div>
            <div className="flex items-center gap-2">
              {selectedTrack.type !== 'voiceover' && (
                <button
                  onClick={() => deleteTrack(selectedTrack.id)}
                  className="px-3 py-1 text-xs bg-red-500/20 text-red-400 rounded hover:bg-red-500/30 transition-colors"
                >
                  Supprimer
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Track waveform component
interface TrackWaveformProps {
  track: AudioTrack;
  pixelsPerSecond: number;
  isSelected: boolean;
  onClick: () => void;
}

function TrackWaveform({ track, pixelsPerSecond, isSelected, onClick }: TrackWaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const width = track.duration * pixelsPerSecond;

  // Draw waveform
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !track.waveformData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = 60 * dpr;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.clearRect(0, 0, width, 60);

    // Draw waveform
    const barWidth = Math.max(2, width / track.waveformData.length);
    const color = track.type === 'voiceover' ? '#3b82f6' :
                  track.type === 'music' ? '#22c55e' : '#eab308';
    const mutedColor = track.isMuted ? 'rgba(156, 163, 175, 0.3)' : color;

    ctx.fillStyle = mutedColor;

    track.waveformData.forEach((value, i) => {
      const x = (i / track.waveformData!.length) * width;
      const height = value * 50 * track.volume;
      const y = (60 - height) / 2;
      ctx.fillRect(x, y, barWidth - 1, height);
    });

    // Draw fade in gradient
    if (track.fadeIn > 0) {
      const fadeWidth = track.fadeIn * pixelsPerSecond;
      const gradient = ctx.createLinearGradient(0, 0, fadeWidth, 0);
      gradient.addColorStop(0, 'rgba(0, 0, 0, 0.7)');
      gradient.addColorStop(1, 'transparent');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, fadeWidth, 60);
    }

    // Draw fade out gradient
    if (track.fadeOut > 0) {
      const fadeWidth = track.fadeOut * pixelsPerSecond;
      const fadeStart = width - fadeWidth;
      const gradient = ctx.createLinearGradient(fadeStart, 0, width, 0);
      gradient.addColorStop(0, 'transparent');
      gradient.addColorStop(1, 'rgba(0, 0, 0, 0.7)');
      ctx.fillStyle = gradient;
      ctx.fillRect(fadeStart, 0, fadeWidth, 60);
    }
  }, [track, width, pixelsPerSecond]);

  return (
    <div
      onClick={onClick}
      className={`h-20 border-b border-gray-800 relative cursor-pointer ${
        isSelected ? 'bg-purple-900/20' : 'hover:bg-gray-800/30'
      }`}
    >
      <div
        className="absolute top-2 bottom-2 rounded overflow-hidden"
        style={{
          left: `${track.startTime * pixelsPerSecond}px`,
          width: `${width}px`,
        }}
      >
        <canvas
          ref={canvasRef}
          style={{ width: `${width}px`, height: '60px' }}
          className={`rounded ${track.isMuted ? 'opacity-30' : ''}`}
        />

        {/* Track name overlay */}
        <div className="absolute top-1 left-2 text-xs text-white/70 truncate max-w-[100px]">
          {track.name}
        </div>

        {/* Fade markers */}
        {track.fadeIn > 0 && (
          <div
            className="absolute top-0 bottom-0 border-r-2 border-yellow-500 border-dashed"
            style={{ left: `${track.fadeIn * pixelsPerSecond}px` }}
          />
        )}
        {track.fadeOut > 0 && (
          <div
            className="absolute top-0 bottom-0 border-l-2 border-yellow-500 border-dashed"
            style={{ right: `${track.fadeOut * pixelsPerSecond}px` }}
          />
        )}
      </div>
    </div>
  );
}

export default AudioTimeline;
