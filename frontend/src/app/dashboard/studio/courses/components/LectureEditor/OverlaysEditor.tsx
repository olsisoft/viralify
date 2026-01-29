'use client';

import React, { useState, useCallback } from 'react';
import type {
  Overlay,
  OverlayType,
  TextOverlay,
  LowerThirdOverlay,
  CalloutOverlay,
  ShapeOverlay,
  OverlayPosition,
  OverlayTiming,
  OverlayAnimation,
} from '../../lib/lecture-editor-types';
import {
  DEFAULT_SUBTITLE_STYLE,
  LOWER_THIRD_STYLES,
} from '../../lib/lecture-editor-types';

interface OverlaysEditorProps {
  overlays: Overlay[];
  currentTime: number;
  totalDuration: number;
  selectedSlideId: string | null;
  onOverlaysChange: (overlays: Overlay[]) => void;
  onSeek: (time: number) => void;
}

// Generate unique ID
const generateId = () => `overlay-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

export function OverlaysEditor({
  overlays,
  currentTime,
  totalDuration,
  selectedSlideId,
  onOverlaysChange,
  onSeek,
}: OverlaysEditorProps) {
  const [selectedOverlayId, setSelectedOverlayId] = useState<string | null>(null);
  const [showAddMenu, setShowAddMenu] = useState(false);

  const selectedOverlay = overlays.find(o => o.id === selectedOverlayId);

  // Add new overlay
  const addOverlay = useCallback((type: OverlayType) => {
    const basePosition: OverlayPosition = {
      x: 50,
      y: type === 'lower-third' ? 80 : 50,
      width: type === 'lower-third' ? 60 : 30,
      height: type === 'lower-third' ? 15 : 10,
      rotation: 0,
    };

    const baseTiming: OverlayTiming = {
      startTime: currentTime,
      endTime: Math.min(currentTime + 5, totalDuration),
      fadeIn: 0.3,
      fadeOut: 0.3,
    };

    const baseAnimation: OverlayAnimation = {
      type: 'fade',
      duration: 0.3,
    };

    let content: Overlay['content'];

    switch (type) {
      case 'text':
        content = {
          type: 'text',
          text: 'Nouveau texte',
          style: { ...DEFAULT_SUBTITLE_STYLE },
        } as TextOverlay;
        break;
      case 'lower-third':
        content = {
          type: 'lower-third',
          title: 'Titre',
          subtitle: 'Sous-titre',
          style: 'modern',
          primaryColor: '#8B5CF6',
          secondaryColor: '#1F2937',
        } as LowerThirdOverlay;
        break;
      case 'callout':
        content = {
          type: 'callout',
          text: 'Callout',
          shape: 'rounded',
          backgroundColor: '#8B5CF6',
          borderColor: '#7C3AED',
          textColor: '#FFFFFF',
        } as CalloutOverlay;
        break;
      case 'shape':
        content = {
          type: 'shape',
          shape: 'rectangle',
          fillColor: '#8B5CF680',
          strokeColor: '#8B5CF6',
          strokeWidth: 2,
          opacity: 0.8,
        } as ShapeOverlay;
        break;
      default:
        return;
    }

    const newOverlay: Overlay = {
      id: generateId(),
      slideId: selectedSlideId || undefined,
      position: basePosition,
      timing: baseTiming,
      animation: baseAnimation,
      content,
      isLocked: false,
      isVisible: true,
    };

    onOverlaysChange([...overlays, newOverlay]);
    setSelectedOverlayId(newOverlay.id);
    setShowAddMenu(false);
  }, [currentTime, totalDuration, selectedSlideId, overlays, onOverlaysChange]);

  // Update overlay
  const updateOverlay = useCallback((overlayId: string, updates: Partial<Overlay>) => {
    onOverlaysChange(overlays.map(o =>
      o.id === overlayId ? { ...o, ...updates } : o
    ));
  }, [overlays, onOverlaysChange]);

  // Delete overlay
  const deleteOverlay = useCallback((overlayId: string) => {
    onOverlaysChange(overlays.filter(o => o.id !== overlayId));
    if (selectedOverlayId === overlayId) {
      setSelectedOverlayId(null);
    }
  }, [overlays, selectedOverlayId, onOverlaysChange]);

  // Duplicate overlay
  const duplicateOverlay = useCallback((overlayId: string) => {
    const overlay = overlays.find(o => o.id === overlayId);
    if (!overlay) return;

    const newOverlay: Overlay = {
      ...JSON.parse(JSON.stringify(overlay)),
      id: generateId(),
      position: {
        ...overlay.position,
        x: overlay.position.x + 5,
        y: overlay.position.y + 5,
      },
    };

    onOverlaysChange([...overlays, newOverlay]);
    setSelectedOverlayId(newOverlay.id);
  }, [overlays, onOverlaysChange]);

  // Get active overlays at current time
  const activeOverlays = overlays.filter(o =>
    o.isVisible && currentTime >= o.timing.startTime && currentTime <= o.timing.endTime
  );

  return (
    <div className="flex flex-col h-full bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <h3 className="text-white font-semibold text-sm flex items-center gap-2">
          <span>📝</span>
          Overlays
        </h3>
        <div className="relative">
          <button
            onClick={() => setShowAddMenu(!showAddMenu)}
            className="flex items-center gap-1 px-3 py-1.5 text-xs bg-purple-600 text-white rounded-lg hover:bg-purple-500 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            Ajouter
          </button>

          {showAddMenu && (
            <div className="absolute top-full right-0 mt-1 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-10 py-1 min-w-[150px]">
              <button
                onClick={() => addOverlay('text')}
                className="w-full px-4 py-2 text-left text-sm text-white hover:bg-gray-700 flex items-center gap-2"
              >
                <span>📄</span> Texte
              </button>
              <button
                onClick={() => addOverlay('lower-third')}
                className="w-full px-4 py-2 text-left text-sm text-white hover:bg-gray-700 flex items-center gap-2"
              >
                <span>📺</span> Lower Third
              </button>
              <button
                onClick={() => addOverlay('callout')}
                className="w-full px-4 py-2 text-left text-sm text-white hover:bg-gray-700 flex items-center gap-2"
              >
                <span>💬</span> Callout
              </button>
              <button
                onClick={() => addOverlay('shape')}
                className="w-full px-4 py-2 text-left text-sm text-white hover:bg-gray-700 flex items-center gap-2"
              >
                <span>⬜</span> Forme
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Active overlays indicator */}
      {activeOverlays.length > 0 && (
        <div className="px-4 py-2 bg-purple-900/20 border-b border-gray-800">
          <p className="text-purple-400 text-xs">
            {activeOverlays.length} overlay(s) actif(s) à ce moment
          </p>
        </div>
      )}

      <div className="flex-1 overflow-y-auto">
        {/* Overlays list */}
        <div className="p-4 space-y-2">
          {overlays.length === 0 ? (
            <div className="text-center py-8 text-gray-500 text-sm">
              <p className="mb-3">Aucun overlay</p>
              <button
                onClick={() => setShowAddMenu(true)}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-500 text-sm"
              >
                Ajouter un overlay
              </button>
            </div>
          ) : (
            overlays.map(overlay => (
              <OverlayListItem
                key={overlay.id}
                overlay={overlay}
                isSelected={selectedOverlayId === overlay.id}
                isActive={activeOverlays.some(o => o.id === overlay.id)}
                onSelect={() => {
                  setSelectedOverlayId(overlay.id);
                  onSeek(overlay.timing.startTime);
                }}
                onToggleVisibility={() => updateOverlay(overlay.id, { isVisible: !overlay.isVisible })}
                onToggleLock={() => updateOverlay(overlay.id, { isLocked: !overlay.isLocked })}
                onDuplicate={() => duplicateOverlay(overlay.id)}
                onDelete={() => deleteOverlay(overlay.id)}
              />
            ))
          )}
        </div>

        {/* Selected overlay editor */}
        {selectedOverlay && (
          <div className="border-t border-gray-800 p-4">
            <OverlayEditor
              overlay={selectedOverlay}
              totalDuration={totalDuration}
              onUpdate={(updates) => updateOverlay(selectedOverlay.id, updates)}
            />
          </div>
        )}
      </div>
    </div>
  );
}

// Overlay list item
interface OverlayListItemProps {
  overlay: Overlay;
  isSelected: boolean;
  isActive: boolean;
  onSelect: () => void;
  onToggleVisibility: () => void;
  onToggleLock: () => void;
  onDuplicate: () => void;
  onDelete: () => void;
}

function OverlayListItem({
  overlay,
  isSelected,
  isActive,
  onSelect,
  onToggleVisibility,
  onToggleLock,
  onDuplicate,
  onDelete,
}: OverlayListItemProps) {
  const getOverlayLabel = () => {
    switch (overlay.content.type) {
      case 'text':
        return (overlay.content as TextOverlay).text.slice(0, 20);
      case 'lower-third':
        return (overlay.content as LowerThirdOverlay).title;
      case 'callout':
        return (overlay.content as CalloutOverlay).text.slice(0, 20);
      case 'shape':
        return `Forme: ${(overlay.content as ShapeOverlay).shape}`;
      default:
        return 'Overlay';
    }
  };

  const getOverlayIcon = () => {
    switch (overlay.content.type) {
      case 'text': return '📄';
      case 'lower-third': return '📺';
      case 'callout': return '💬';
      case 'shape': return '⬜';
      default: return '📝';
    }
  };

  return (
    <div
      onClick={onSelect}
      className={`p-3 rounded-lg cursor-pointer transition-colors ${
        isSelected ? 'bg-purple-600/20 border border-purple-500' :
        isActive ? 'bg-green-900/20 border border-green-500/50' :
        'bg-gray-800 border border-transparent hover:bg-gray-750'
      } ${overlay.isLocked ? 'opacity-60' : ''} ${!overlay.isVisible ? 'opacity-40' : ''}`}
    >
      <div className="flex items-center gap-3">
        <span className="text-lg">{getOverlayIcon()}</span>
        <div className="flex-1 min-w-0">
          <p className="text-white text-sm font-medium truncate">{getOverlayLabel()}</p>
          <p className="text-gray-500 text-xs">
            {overlay.timing.startTime.toFixed(1)}s - {overlay.timing.endTime.toFixed(1)}s
          </p>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={(e) => { e.stopPropagation(); onToggleVisibility(); }}
            className={`p-1 rounded transition-colors ${overlay.isVisible ? 'text-gray-400 hover:text-white' : 'text-red-400'}`}
            title={overlay.isVisible ? 'Masquer' : 'Afficher'}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              {overlay.isVisible ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
              )}
            </svg>
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onToggleLock(); }}
            className={`p-1 rounded transition-colors ${overlay.isLocked ? 'text-yellow-400' : 'text-gray-400 hover:text-white'}`}
            title={overlay.isLocked ? 'Déverrouiller' : 'Verrouiller'}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              {overlay.isLocked ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 11V7a4 4 0 118 0m-4 8v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2z" />
              )}
            </svg>
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onDuplicate(); }}
            className="p-1 text-gray-400 hover:text-white rounded transition-colors"
            title="Dupliquer"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onDelete(); }}
            className="p-1 text-gray-400 hover:text-red-400 rounded transition-colors"
            title="Supprimer"
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

// Overlay editor
interface OverlayEditorProps {
  overlay: Overlay;
  totalDuration: number;
  onUpdate: (updates: Partial<Overlay>) => void;
}

function OverlayEditor({ overlay, totalDuration, onUpdate }: OverlayEditorProps) {
  const updateTiming = (updates: Partial<OverlayTiming>) => {
    onUpdate({ timing: { ...overlay.timing, ...updates } });
  };

  const updatePosition = (updates: Partial<OverlayPosition>) => {
    onUpdate({ position: { ...overlay.position, ...updates } });
  };

  const updateContent = (updates: Partial<Overlay['content']>) => {
    onUpdate({ content: { ...overlay.content, ...updates } as Overlay['content'] });
  };

  return (
    <div className="space-y-4">
      <h4 className="text-white text-sm font-medium">Éditer l'overlay</h4>

      {/* Timing */}
      <div className="bg-gray-800 rounded-lg p-3 space-y-3">
        <p className="text-gray-400 text-xs font-medium">Timing</p>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-gray-500 text-xs block mb-1">Début (s)</label>
            <input
              type="number"
              min={0}
              max={overlay.timing.endTime - 0.1}
              step={0.1}
              value={overlay.timing.startTime}
              onChange={(e) => updateTiming({ startTime: parseFloat(e.target.value) || 0 })}
              className="w-full bg-gray-700 text-white text-sm rounded px-2 py-1.5 border border-gray-600 focus:border-purple-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="text-gray-500 text-xs block mb-1">Fin (s)</label>
            <input
              type="number"
              min={overlay.timing.startTime + 0.1}
              max={totalDuration}
              step={0.1}
              value={overlay.timing.endTime}
              onChange={(e) => updateTiming({ endTime: parseFloat(e.target.value) || totalDuration })}
              className="w-full bg-gray-700 text-white text-sm rounded px-2 py-1.5 border border-gray-600 focus:border-purple-500 focus:outline-none"
            />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-gray-500 text-xs block mb-1">Fade In (s)</label>
            <input
              type="number"
              min={0}
              max={2}
              step={0.1}
              value={overlay.timing.fadeIn}
              onChange={(e) => updateTiming({ fadeIn: parseFloat(e.target.value) || 0 })}
              className="w-full bg-gray-700 text-white text-sm rounded px-2 py-1.5 border border-gray-600 focus:border-purple-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="text-gray-500 text-xs block mb-1">Fade Out (s)</label>
            <input
              type="number"
              min={0}
              max={2}
              step={0.1}
              value={overlay.timing.fadeOut}
              onChange={(e) => updateTiming({ fadeOut: parseFloat(e.target.value) || 0 })}
              className="w-full bg-gray-700 text-white text-sm rounded px-2 py-1.5 border border-gray-600 focus:border-purple-500 focus:outline-none"
            />
          </div>
        </div>
      </div>

      {/* Position */}
      <div className="bg-gray-800 rounded-lg p-3 space-y-3">
        <p className="text-gray-400 text-xs font-medium">Position (%)</p>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-gray-500 text-xs block mb-1">X</label>
            <input
              type="number"
              min={0}
              max={100}
              value={overlay.position.x}
              onChange={(e) => updatePosition({ x: parseFloat(e.target.value) || 0 })}
              className="w-full bg-gray-700 text-white text-sm rounded px-2 py-1.5 border border-gray-600 focus:border-purple-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="text-gray-500 text-xs block mb-1">Y</label>
            <input
              type="number"
              min={0}
              max={100}
              value={overlay.position.y}
              onChange={(e) => updatePosition({ y: parseFloat(e.target.value) || 0 })}
              className="w-full bg-gray-700 text-white text-sm rounded px-2 py-1.5 border border-gray-600 focus:border-purple-500 focus:outline-none"
            />
          </div>
        </div>
      </div>

      {/* Content-specific editor */}
      {overlay.content.type === 'text' && (
        <div className="bg-gray-800 rounded-lg p-3 space-y-3">
          <p className="text-gray-400 text-xs font-medium">Texte</p>
          <textarea
            value={(overlay.content as TextOverlay).text}
            onChange={(e) => updateContent({ text: e.target.value })}
            rows={3}
            className="w-full bg-gray-700 text-white text-sm rounded px-2 py-1.5 border border-gray-600 focus:border-purple-500 focus:outline-none resize-none"
          />
        </div>
      )}

      {overlay.content.type === 'lower-third' && (
        <div className="bg-gray-800 rounded-lg p-3 space-y-3">
          <p className="text-gray-400 text-xs font-medium">Lower Third</p>
          <div>
            <label className="text-gray-500 text-xs block mb-1">Titre</label>
            <input
              type="text"
              value={(overlay.content as LowerThirdOverlay).title}
              onChange={(e) => updateContent({ title: e.target.value })}
              className="w-full bg-gray-700 text-white text-sm rounded px-2 py-1.5 border border-gray-600 focus:border-purple-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="text-gray-500 text-xs block mb-1">Sous-titre</label>
            <input
              type="text"
              value={(overlay.content as LowerThirdOverlay).subtitle}
              onChange={(e) => updateContent({ subtitle: e.target.value })}
              className="w-full bg-gray-700 text-white text-sm rounded px-2 py-1.5 border border-gray-600 focus:border-purple-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="text-gray-500 text-xs block mb-1">Style</label>
            <select
              value={(overlay.content as LowerThirdOverlay).style}
              onChange={(e) => updateContent({ style: e.target.value as LowerThirdOverlay['style'] })}
              className="w-full bg-gray-700 text-white text-sm rounded px-2 py-1.5 border border-gray-600 focus:border-purple-500 focus:outline-none"
            >
              {Object.entries(LOWER_THIRD_STYLES).map(([key, value]) => (
                <option key={key} value={key}>{value.label}</option>
              ))}
            </select>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-gray-500 text-xs block mb-1">Couleur principale</label>
              <input
                type="color"
                value={(overlay.content as LowerThirdOverlay).primaryColor}
                onChange={(e) => updateContent({ primaryColor: e.target.value })}
                className="w-full h-8 rounded border border-gray-600 cursor-pointer"
              />
            </div>
            <div>
              <label className="text-gray-500 text-xs block mb-1">Couleur secondaire</label>
              <input
                type="color"
                value={(overlay.content as LowerThirdOverlay).secondaryColor}
                onChange={(e) => updateContent({ secondaryColor: e.target.value })}
                className="w-full h-8 rounded border border-gray-600 cursor-pointer"
              />
            </div>
          </div>
        </div>
      )}

      {overlay.content.type === 'callout' && (
        <div className="bg-gray-800 rounded-lg p-3 space-y-3">
          <p className="text-gray-400 text-xs font-medium">Callout</p>
          <textarea
            value={(overlay.content as CalloutOverlay).text}
            onChange={(e) => updateContent({ text: e.target.value })}
            rows={2}
            className="w-full bg-gray-700 text-white text-sm rounded px-2 py-1.5 border border-gray-600 focus:border-purple-500 focus:outline-none resize-none"
          />
          <div className="grid grid-cols-3 gap-2">
            <div>
              <label className="text-gray-500 text-xs block mb-1">Fond</label>
              <input
                type="color"
                value={(overlay.content as CalloutOverlay).backgroundColor}
                onChange={(e) => updateContent({ backgroundColor: e.target.value })}
                className="w-full h-8 rounded border border-gray-600 cursor-pointer"
              />
            </div>
            <div>
              <label className="text-gray-500 text-xs block mb-1">Bordure</label>
              <input
                type="color"
                value={(overlay.content as CalloutOverlay).borderColor}
                onChange={(e) => updateContent({ borderColor: e.target.value })}
                className="w-full h-8 rounded border border-gray-600 cursor-pointer"
              />
            </div>
            <div>
              <label className="text-gray-500 text-xs block mb-1">Texte</label>
              <input
                type="color"
                value={(overlay.content as CalloutOverlay).textColor}
                onChange={(e) => updateContent({ textColor: e.target.value })}
                className="w-full h-8 rounded border border-gray-600 cursor-pointer"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default OverlaysEditor;
