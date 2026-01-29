'use client';

import React, { useState, useCallback } from 'react';
import type {
  SlideComponent,
  VisualEffect,
  KenBurnsEffect,
  ColorGrading,
  FilterPreset,
} from '../../lib/lecture-editor-types';
import {
  FILTER_PRESETS,
  DEFAULT_COLOR_GRADING,
} from '../../lib/lecture-editor-types';

interface VisualEffectsPanelProps {
  slides: SlideComponent[];
  effects: VisualEffect[];
  selectedSlideId: string | null;
  onEffectsChange: (effects: VisualEffect[]) => void;
}

export function VisualEffectsPanel({
  slides,
  effects,
  selectedSlideId,
  onEffectsChange,
}: VisualEffectsPanelProps) {
  const [activeTab, setActiveTab] = useState<'filters' | 'kenburns' | 'speed'>('filters');

  // Get effect for a slide
  const getEffect = (slideId: string): VisualEffect => {
    return effects.find(e => e.slideId === slideId) || {
      id: `effect-${slideId}`,
      slideId,
      filterPreset: 'none',
      speed: 1,
      reverse: false,
    };
  };

  // Update effect for a slide
  const updateEffect = useCallback((slideId: string, updates: Partial<VisualEffect>) => {
    const existing = effects.find(e => e.slideId === slideId);
    if (existing) {
      onEffectsChange(effects.map(e =>
        e.slideId === slideId ? { ...e, ...updates } : e
      ));
    } else {
      onEffectsChange([...effects, {
        id: `effect-${slideId}`,
        slideId,
        filterPreset: 'none',
        speed: 1,
        reverse: false,
        ...updates,
      }]);
    }
  }, [effects, onEffectsChange]);

  const selectedSlide = slides.find(s => s.id === selectedSlideId);
  const selectedEffect = selectedSlideId ? getEffect(selectedSlideId) : null;

  return (
    <div className="flex flex-col h-full bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <h3 className="text-white font-semibold text-sm flex items-center gap-2">
          <span>🎨</span>
          Effets visuels
        </h3>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-800">
        {(['filters', 'kenburns', 'speed'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 py-2 text-sm transition-colors ${
              activeTab === tab
                ? 'text-purple-400 border-b-2 border-purple-400'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            {tab === 'filters' ? 'Filtres' : tab === 'kenburns' ? 'Ken Burns' : 'Vitesse'}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {!selectedSlide ? (
          <div className="flex items-center justify-center h-full text-gray-500 text-sm">
            Sélectionnez un slide pour modifier ses effets
          </div>
        ) : (
          <>
            {/* Filters tab */}
            {activeTab === 'filters' && (
              <FiltersTab
                effect={selectedEffect!}
                slideId={selectedSlideId!}
                onUpdate={updateEffect}
              />
            )}

            {/* Ken Burns tab */}
            {activeTab === 'kenburns' && (
              <KenBurnsTab
                effect={selectedEffect!}
                slideId={selectedSlideId!}
                onUpdate={updateEffect}
              />
            )}

            {/* Speed tab */}
            {activeTab === 'speed' && (
              <SpeedTab
                effect={selectedEffect!}
                slideId={selectedSlideId!}
                onUpdate={updateEffect}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}

// Filters tab
interface FiltersTabProps {
  effect: VisualEffect;
  slideId: string;
  onUpdate: (slideId: string, updates: Partial<VisualEffect>) => void;
}

function FiltersTab({ effect, slideId, onUpdate }: FiltersTabProps) {
  const [showManual, setShowManual] = useState(false);

  const grading = effect.colorGrading || DEFAULT_COLOR_GRADING;

  const updateGrading = (updates: Partial<ColorGrading>) => {
    onUpdate(slideId, {
      colorGrading: { ...grading, ...updates },
      filterPreset: 'none', // Reset preset when manually adjusting
    });
  };

  return (
    <div className="space-y-6">
      {/* Filter presets */}
      <div>
        <label className="text-gray-400 text-xs font-medium block mb-3">Presets</label>
        <div className="grid grid-cols-2 gap-2">
          {(Object.keys(FILTER_PRESETS) as FilterPreset[]).map((preset) => (
            <button
              key={preset}
              onClick={() => {
                onUpdate(slideId, {
                  filterPreset: preset,
                  colorGrading: {
                    ...DEFAULT_COLOR_GRADING,
                    ...FILTER_PRESETS[preset].grading,
                  },
                });
              }}
              className={`py-2 px-3 rounded-lg text-sm transition-colors ${
                effect.filterPreset === preset
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {FILTER_PRESETS[preset].label}
            </button>
          ))}
        </div>
      </div>

      {/* Manual adjustments toggle */}
      <button
        onClick={() => setShowManual(!showManual)}
        className="text-purple-400 text-xs flex items-center gap-1 hover:text-purple-300"
      >
        <svg
          className={`w-4 h-4 transition-transform ${showManual ? 'rotate-90' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        Ajustements manuels
      </button>

      {showManual && (
        <div className="space-y-4 bg-gray-800 rounded-lg p-4">
          {/* Brightness */}
          <SliderControl
            label="Luminosité"
            value={grading.brightness}
            min={-100}
            max={100}
            onChange={(v) => updateGrading({ brightness: v })}
          />

          {/* Contrast */}
          <SliderControl
            label="Contraste"
            value={grading.contrast}
            min={-100}
            max={100}
            onChange={(v) => updateGrading({ contrast: v })}
          />

          {/* Saturation */}
          <SliderControl
            label="Saturation"
            value={grading.saturation}
            min={-100}
            max={100}
            onChange={(v) => updateGrading({ saturation: v })}
          />

          {/* Temperature */}
          <SliderControl
            label="Température"
            value={grading.temperature}
            min={-100}
            max={100}
            onChange={(v) => updateGrading({ temperature: v })}
            leftLabel="Froid"
            rightLabel="Chaud"
          />

          {/* Tint */}
          <SliderControl
            label="Teinte"
            value={grading.tint}
            min={-100}
            max={100}
            onChange={(v) => updateGrading({ tint: v })}
            leftLabel="Vert"
            rightLabel="Magenta"
          />

          {/* Highlights */}
          <SliderControl
            label="Hautes lumières"
            value={grading.highlights}
            min={-100}
            max={100}
            onChange={(v) => updateGrading({ highlights: v })}
          />

          {/* Shadows */}
          <SliderControl
            label="Ombres"
            value={grading.shadows}
            min={-100}
            max={100}
            onChange={(v) => updateGrading({ shadows: v })}
          />

          {/* Vignette */}
          <SliderControl
            label="Vignette"
            value={grading.vignette}
            min={0}
            max={100}
            onChange={(v) => updateGrading({ vignette: v })}
          />

          {/* Reset button */}
          <button
            onClick={() => onUpdate(slideId, {
              filterPreset: 'none',
              colorGrading: DEFAULT_COLOR_GRADING,
            })}
            className="w-full py-2 text-xs bg-gray-700 text-gray-300 rounded hover:bg-gray-600 transition-colors"
          >
            Réinitialiser
          </button>
        </div>
      )}
    </div>
  );
}

// Ken Burns tab
interface KenBurnsTabProps {
  effect: VisualEffect;
  slideId: string;
  onUpdate: (slideId: string, updates: Partial<VisualEffect>) => void;
}

function KenBurnsTab({ effect, slideId, onUpdate }: KenBurnsTabProps) {
  const kenBurns = effect.kenBurns || {
    enabled: false,
    startScale: 1,
    endScale: 1.2,
    startPosition: { x: 0, y: 0 },
    endPosition: { x: 0, y: 0 },
  };

  const updateKenBurns = (updates: Partial<KenBurnsEffect>) => {
    onUpdate(slideId, {
      kenBurns: { ...kenBurns, ...updates },
    });
  };

  return (
    <div className="space-y-6">
      {/* Enable toggle */}
      <label className="flex items-center justify-between cursor-pointer">
        <span className="text-white text-sm">Activer Ken Burns</span>
        <div className="relative">
          <input
            type="checkbox"
            checked={kenBurns.enabled}
            onChange={(e) => updateKenBurns({ enabled: e.target.checked })}
            className="sr-only"
          />
          <div className={`w-10 h-6 rounded-full transition-colors ${kenBurns.enabled ? 'bg-purple-600' : 'bg-gray-700'}`}>
            <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${kenBurns.enabled ? 'translate-x-5' : 'translate-x-1'}`} />
          </div>
        </div>
      </label>

      {kenBurns.enabled && (
        <>
          {/* Presets */}
          <div>
            <label className="text-gray-400 text-xs font-medium block mb-2">Presets</label>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => updateKenBurns({
                  startScale: 1,
                  endScale: 1.3,
                  startPosition: { x: 0, y: 0 },
                  endPosition: { x: 0, y: 0 },
                })}
                className="py-2 px-3 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 text-sm"
              >
                Zoom avant
              </button>
              <button
                onClick={() => updateKenBurns({
                  startScale: 1.3,
                  endScale: 1,
                  startPosition: { x: 0, y: 0 },
                  endPosition: { x: 0, y: 0 },
                })}
                className="py-2 px-3 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 text-sm"
              >
                Zoom arrière
              </button>
              <button
                onClick={() => updateKenBurns({
                  startScale: 1.2,
                  endScale: 1.2,
                  startPosition: { x: -0.2, y: 0 },
                  endPosition: { x: 0.2, y: 0 },
                })}
                className="py-2 px-3 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 text-sm"
              >
                Pan gauche-droite
              </button>
              <button
                onClick={() => updateKenBurns({
                  startScale: 1.2,
                  endScale: 1.2,
                  startPosition: { x: 0, y: -0.2 },
                  endPosition: { x: 0, y: 0.2 },
                })}
                className="py-2 px-3 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 text-sm"
              >
                Pan haut-bas
              </button>
            </div>
          </div>

          {/* Start scale */}
          <SliderControl
            label="Zoom initial"
            value={(kenBurns.startScale - 1) * 100}
            min={0}
            max={100}
            onChange={(v) => updateKenBurns({ startScale: 1 + v / 100 })}
            suffix="%"
          />

          {/* End scale */}
          <SliderControl
            label="Zoom final"
            value={(kenBurns.endScale - 1) * 100}
            min={0}
            max={100}
            onChange={(v) => updateKenBurns({ endScale: 1 + v / 100 })}
            suffix="%"
          />

          {/* Position controls */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-gray-400 text-xs font-medium block mb-2">Position initiale</label>
              <div className="space-y-2">
                <SliderControl
                  label="X"
                  value={kenBurns.startPosition.x * 100}
                  min={-50}
                  max={50}
                  onChange={(v) => updateKenBurns({
                    startPosition: { ...kenBurns.startPosition, x: v / 100 }
                  })}
                  compact
                />
                <SliderControl
                  label="Y"
                  value={kenBurns.startPosition.y * 100}
                  min={-50}
                  max={50}
                  onChange={(v) => updateKenBurns({
                    startPosition: { ...kenBurns.startPosition, y: v / 100 }
                  })}
                  compact
                />
              </div>
            </div>
            <div>
              <label className="text-gray-400 text-xs font-medium block mb-2">Position finale</label>
              <div className="space-y-2">
                <SliderControl
                  label="X"
                  value={kenBurns.endPosition.x * 100}
                  min={-50}
                  max={50}
                  onChange={(v) => updateKenBurns({
                    endPosition: { ...kenBurns.endPosition, x: v / 100 }
                  })}
                  compact
                />
                <SliderControl
                  label="Y"
                  value={kenBurns.endPosition.y * 100}
                  min={-50}
                  max={50}
                  onChange={(v) => updateKenBurns({
                    endPosition: { ...kenBurns.endPosition, y: v / 100 }
                  })}
                  compact
                />
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// Speed tab
interface SpeedTabProps {
  effect: VisualEffect;
  slideId: string;
  onUpdate: (slideId: string, updates: Partial<VisualEffect>) => void;
}

function SpeedTab({ effect, slideId, onUpdate }: SpeedTabProps) {
  const speedPresets = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 4];

  return (
    <div className="space-y-6">
      {/* Speed presets */}
      <div>
        <label className="text-gray-400 text-xs font-medium block mb-3">Vitesse de lecture</label>
        <div className="grid grid-cols-4 gap-2">
          {speedPresets.map((speed) => (
            <button
              key={speed}
              onClick={() => onUpdate(slideId, { speed })}
              className={`py-2 px-3 rounded-lg text-sm transition-colors ${
                effect.speed === speed
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {speed}x
            </button>
          ))}
        </div>
      </div>

      {/* Custom speed slider */}
      <div>
        <label className="text-gray-400 text-xs font-medium block mb-2">
          Vitesse personnalisée: {effect.speed}x
        </label>
        <input
          type="range"
          min={0.25}
          max={4}
          step={0.05}
          value={effect.speed}
          onChange={(e) => onUpdate(slideId, { speed: parseFloat(e.target.value) })}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>0.25x (Ralenti)</span>
          <span>4x (Accéléré)</span>
        </div>
      </div>

      {/* Reverse toggle */}
      <label className="flex items-center justify-between cursor-pointer bg-gray-800 rounded-lg p-3">
        <span className="text-white text-sm">Lecture inversée</span>
        <div className="relative">
          <input
            type="checkbox"
            checked={effect.reverse}
            onChange={(e) => onUpdate(slideId, { reverse: e.target.checked })}
            className="sr-only"
          />
          <div className={`w-10 h-6 rounded-full transition-colors ${effect.reverse ? 'bg-purple-600' : 'bg-gray-700'}`}>
            <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${effect.reverse ? 'translate-x-5' : 'translate-x-1'}`} />
          </div>
        </div>
      </label>

      {/* Duration info */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h4 className="text-gray-400 text-xs font-medium mb-2">Information</h4>
        <p className="text-gray-300 text-sm">
          {effect.speed < 1
            ? `Le slide sera ralenti de ${Math.round((1 - effect.speed) * 100)}%`
            : effect.speed > 1
            ? `Le slide sera accéléré de ${Math.round((effect.speed - 1) * 100)}%`
            : 'Vitesse normale'}
        </p>
        {effect.reverse && (
          <p className="text-yellow-400 text-sm mt-1">
            La lecture sera inversée
          </p>
        )}
      </div>
    </div>
  );
}

// Slider control component
interface SliderControlProps {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
  leftLabel?: string;
  rightLabel?: string;
  suffix?: string;
  compact?: boolean;
}

function SliderControl({
  label,
  value,
  min,
  max,
  onChange,
  leftLabel,
  rightLabel,
  suffix = '',
  compact = false,
}: SliderControlProps) {
  return (
    <div className={compact ? 'flex items-center gap-2' : ''}>
      <div className={`flex items-center justify-between ${compact ? 'w-8' : 'mb-1'}`}>
        <span className="text-gray-400 text-xs">{label}</span>
        {!compact && (
          <span className="text-gray-300 text-xs">{Math.round(value)}{suffix}</span>
        )}
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className={`${compact ? 'flex-1' : 'w-full'} h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer
          [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3
          [&::-webkit-slider-thumb]:bg-purple-500 [&::-webkit-slider-thumb]:rounded-full`}
      />
      {(leftLabel || rightLabel) && !compact && (
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>{leftLabel}</span>
          <span>{rightLabel}</span>
        </div>
      )}
    </div>
  );
}

export default VisualEffectsPanel;
