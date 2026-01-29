'use client';

import React, { useState, useCallback } from 'react';
import type {
  ExportSettings,
  ExportResolution,
  ExportFormat,
  ExportAspectRatio,
  ExportQuality,
  WatermarkOverlay,
} from '../../lib/lecture-editor-types';
import {
  DEFAULT_EXPORT_SETTINGS,
  RESOLUTION_CONFIG,
  ASPECT_RATIO_CONFIG,
  QUALITY_PRESETS,
  formatDuration,
} from '../../lib/lecture-editor-types';

interface ExportPanelProps {
  settings: ExportSettings;
  totalDuration: number;
  onSettingsChange: (settings: ExportSettings) => void;
  onExport: (settings: ExportSettings) => Promise<void>;
  isExporting: boolean;
  exportProgress?: number;
  exportError?: string;
}

export function ExportPanel({
  settings,
  totalDuration,
  onSettingsChange,
  onExport,
  isExporting,
  exportProgress = 0,
  exportError,
}: ExportPanelProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showWatermarkSettings, setShowWatermarkSettings] = useState(false);

  // Calculate estimated file size
  const estimatedSize = useCallback(() => {
    const videoBitrate = settings.videoBitrate * 1000; // kbps to bps
    const audioBitrate = settings.audioBitrate * 1000;
    const totalBitrate = videoBitrate + audioBitrate;
    const sizeBytes = (totalBitrate * totalDuration) / 8;
    const sizeMB = sizeBytes / (1024 * 1024);
    return sizeMB.toFixed(1);
  }, [settings.videoBitrate, settings.audioBitrate, totalDuration]);

  // Get resolution dimensions based on aspect ratio
  const getAdjustedResolution = useCallback(() => {
    const base = RESOLUTION_CONFIG[settings.resolution];
    const [ratioW, ratioH] = settings.aspectRatio.split(':').map(Number);

    if (ratioW > ratioH) {
      // Landscape or square
      const height = Math.round(base.width * (ratioH / ratioW));
      return { width: base.width, height };
    } else {
      // Portrait
      const width = Math.round(base.height * (ratioW / ratioH));
      return { width, height: base.height };
    }
  }, [settings.resolution, settings.aspectRatio]);

  const adjustedRes = getAdjustedResolution();

  // Handle export click
  const handleExport = async () => {
    await onExport(settings);
  };

  // Update watermark
  const updateWatermark = (updates: Partial<WatermarkOverlay>) => {
    const currentWatermark = settings.watermark || {
      type: 'watermark' as const,
      opacity: 0.5,
      position: 'bottom-right' as const,
    };
    onSettingsChange({
      ...settings,
      watermark: { ...currentWatermark, ...updates },
    });
  };

  return (
    <div className="flex flex-col h-full bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <h3 className="text-white font-semibold text-sm flex items-center gap-2">
          <span>📤</span>
          Exporter la vidéo
        </h3>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Resolution */}
        <div>
          <label className="text-gray-400 text-xs font-medium block mb-2">Résolution</label>
          <div className="grid grid-cols-2 gap-2">
            {(Object.keys(RESOLUTION_CONFIG) as ExportResolution[]).map((res) => (
              <button
                key={res}
                onClick={() => onSettingsChange({ ...settings, resolution: res })}
                className={`py-2 px-3 rounded-lg text-sm transition-colors ${
                  settings.resolution === res
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                {RESOLUTION_CONFIG[res].label}
              </button>
            ))}
          </div>
        </div>

        {/* Aspect Ratio */}
        <div>
          <label className="text-gray-400 text-xs font-medium block mb-2">Format (Ratio)</label>
          <div className="grid grid-cols-3 gap-2">
            {(Object.keys(ASPECT_RATIO_CONFIG) as ExportAspectRatio[]).map((ratio) => (
              <button
                key={ratio}
                onClick={() => onSettingsChange({ ...settings, aspectRatio: ratio })}
                className={`py-2 px-3 rounded-lg text-sm transition-colors flex flex-col items-center ${
                  settings.aspectRatio === ratio
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                <span className="font-medium">{ASPECT_RATIO_CONFIG[ratio].label}</span>
                <span className="text-xs opacity-70">{ASPECT_RATIO_CONFIG[ratio].description}</span>
              </button>
            ))}
          </div>
          <p className="text-gray-500 text-xs mt-2">
            Dimension finale: {adjustedRes.width} x {adjustedRes.height}
          </p>
        </div>

        {/* Format */}
        <div>
          <label className="text-gray-400 text-xs font-medium block mb-2">Format de fichier</label>
          <div className="flex gap-2">
            {(['mp4', 'webm', 'mov'] as ExportFormat[]).map((format) => (
              <button
                key={format}
                onClick={() => onSettingsChange({ ...settings, format })}
                className={`flex-1 py-2 rounded-lg text-sm uppercase transition-colors ${
                  settings.format === format
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                {format}
              </button>
            ))}
          </div>
        </div>

        {/* Quality */}
        <div>
          <label className="text-gray-400 text-xs font-medium block mb-2">Qualité</label>
          <div className="space-y-2">
            {(Object.keys(QUALITY_PRESETS) as ExportQuality[]).map((quality) => (
              <button
                key={quality}
                onClick={() => onSettingsChange({
                  ...settings,
                  quality,
                  videoBitrate: QUALITY_PRESETS[quality].videoBitrate,
                  audioBitrate: QUALITY_PRESETS[quality].audioBitrate,
                })}
                className={`w-full py-2 px-3 rounded-lg text-sm text-left flex items-center justify-between transition-colors ${
                  settings.quality === quality
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                <span>{QUALITY_PRESETS[quality].label}</span>
                <span className="text-xs opacity-70">
                  {QUALITY_PRESETS[quality].videoBitrate / 1000} Mbps
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* FPS */}
        <div>
          <label className="text-gray-400 text-xs font-medium block mb-2">Images par seconde</label>
          <div className="flex gap-2">
            {([24, 30, 60] as const).map((fps) => (
              <button
                key={fps}
                onClick={() => onSettingsChange({ ...settings, fps })}
                className={`flex-1 py-2 rounded-lg text-sm transition-colors ${
                  settings.fps === fps
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                {fps} FPS
              </button>
            ))}
          </div>
        </div>

        {/* Subtitles */}
        <div>
          <label className="text-gray-400 text-xs font-medium block mb-2">Sous-titres</label>
          <div className="space-y-2">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.includeSubtitles}
                onChange={(e) => onSettingsChange({ ...settings, includeSubtitles: e.target.checked })}
                className="rounded border-gray-600 bg-gray-800 text-purple-600 focus:ring-purple-500"
              />
              <span className="text-gray-300 text-sm">Inclure les sous-titres</span>
            </label>
            {settings.includeSubtitles && (
              <label className="flex items-center gap-2 cursor-pointer ml-6">
                <input
                  type="checkbox"
                  checked={settings.burnSubtitles}
                  onChange={(e) => onSettingsChange({ ...settings, burnSubtitles: e.target.checked })}
                  className="rounded border-gray-600 bg-gray-800 text-purple-600 focus:ring-purple-500"
                />
                <span className="text-gray-300 text-sm">Incruster dans la vidéo (hardcode)</span>
              </label>
            )}
          </div>
        </div>

        {/* Watermark */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-gray-400 text-xs font-medium">Filigrane</label>
            <button
              onClick={() => setShowWatermarkSettings(!showWatermarkSettings)}
              className="text-purple-400 text-xs hover:text-purple-300"
            >
              {settings.watermark ? 'Modifier' : 'Ajouter'}
            </button>
          </div>

          {settings.watermark && (
            <div className="bg-gray-800 rounded-lg p-3 flex items-center justify-between">
              <span className="text-gray-300 text-sm">
                {settings.watermark.text || settings.watermark.imageUrl ? 'Actif' : 'Configuré'}
              </span>
              <button
                onClick={() => onSettingsChange({ ...settings, watermark: undefined })}
                className="text-red-400 text-xs hover:text-red-300"
              >
                Supprimer
              </button>
            </div>
          )}

          {showWatermarkSettings && (
            <div className="mt-3 space-y-3 bg-gray-800 rounded-lg p-3">
              <div>
                <label className="text-gray-400 text-xs block mb-1">Texte du filigrane</label>
                <input
                  type="text"
                  value={settings.watermark?.text || ''}
                  onChange={(e) => updateWatermark({ text: e.target.value })}
                  placeholder="Ex: @moncompte"
                  className="w-full bg-gray-700 text-white text-sm rounded px-3 py-2 border border-gray-600 focus:border-purple-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="text-gray-400 text-xs block mb-1">Position</label>
                <select
                  value={settings.watermark?.position || 'bottom-right'}
                  onChange={(e) => updateWatermark({ position: e.target.value as WatermarkOverlay['position'] })}
                  className="w-full bg-gray-700 text-white text-sm rounded px-3 py-2 border border-gray-600 focus:border-purple-500 focus:outline-none"
                >
                  <option value="top-left">Haut gauche</option>
                  <option value="top-right">Haut droite</option>
                  <option value="bottom-left">Bas gauche</option>
                  <option value="bottom-right">Bas droite</option>
                  <option value="center">Centre</option>
                </select>
              </div>
              <div>
                <label className="text-gray-400 text-xs block mb-1">
                  Opacité: {Math.round((settings.watermark?.opacity || 0.5) * 100)}%
                </label>
                <input
                  type="range"
                  min={0.1}
                  max={1}
                  step={0.1}
                  value={settings.watermark?.opacity || 0.5}
                  onChange={(e) => updateWatermark({ opacity: parseFloat(e.target.value) })}
                  className="w-full"
                />
              </div>
            </div>
          )}
        </div>

        {/* Advanced settings */}
        <div>
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-gray-400 text-xs flex items-center gap-1 hover:text-white transition-colors"
          >
            <svg
              className={`w-4 h-4 transition-transform ${showAdvanced ? 'rotate-90' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            Paramètres avancés
          </button>

          {showAdvanced && (
            <div className="mt-3 space-y-4 bg-gray-800 rounded-lg p-3">
              <div>
                <label className="text-gray-400 text-xs block mb-1">
                  Débit vidéo: {settings.videoBitrate} kbps
                </label>
                <input
                  type="range"
                  min={1000}
                  max={50000}
                  step={500}
                  value={settings.videoBitrate}
                  onChange={(e) => onSettingsChange({ ...settings, videoBitrate: parseInt(e.target.value) })}
                  className="w-full"
                />
              </div>
              <div>
                <label className="text-gray-400 text-xs block mb-1">
                  Débit audio: {settings.audioBitrate} kbps
                </label>
                <input
                  type="range"
                  min={64}
                  max={320}
                  step={32}
                  value={settings.audioBitrate}
                  onChange={(e) => onSettingsChange({ ...settings, audioBitrate: parseInt(e.target.value) })}
                  className="w-full"
                />
              </div>
            </div>
          )}
        </div>

        {/* Summary */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h4 className="text-white text-sm font-medium mb-3">Résumé de l'export</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Format</span>
              <span className="text-white">{settings.format.toUpperCase()} - {adjustedRes.width}x{adjustedRes.height}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Qualité</span>
              <span className="text-white">{QUALITY_PRESETS[settings.quality].label}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Durée</span>
              <span className="text-white">{formatDuration(totalDuration)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Taille estimée</span>
              <span className="text-white">~{estimatedSize()} MB</span>
            </div>
          </div>
        </div>

        {/* Error message */}
        {exportError && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
            {exportError}
          </div>
        )}
      </div>

      {/* Export button */}
      <div className="p-4 border-t border-gray-800">
        {isExporting ? (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-400">Export en cours...</span>
              <span className="text-white">{Math.round(exportProgress)}%</span>
            </div>
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-purple-600 transition-all duration-300"
                style={{ width: `${exportProgress}%` }}
              />
            </div>
          </div>
        ) : (
          <button
            onClick={handleExport}
            className="w-full py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-500 transition-colors font-medium flex items-center justify-center gap-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Exporter la vidéo
          </button>
        )}
      </div>
    </div>
  );
}

export default ExportPanel;
