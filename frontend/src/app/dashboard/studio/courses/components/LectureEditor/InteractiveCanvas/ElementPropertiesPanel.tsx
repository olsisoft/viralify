'use client';

import React, { memo, useCallback, useState } from 'react';
import type { SlideElement, UpdateElementRequest, TextBlockContent, ShapeContent, ImageElementContent } from '../../../lib/lecture-editor-types';

interface ElementPropertiesPanelProps {
  element: SlideElement | null;
  onUpdate: (updates: UpdateElementRequest) => void;
  disabled?: boolean;
}

// Color presets for quick selection
const COLOR_PRESETS = [
  '#FFFFFF', '#000000', '#6366F1', '#8B5CF6', '#EC4899',
  '#EF4444', '#F97316', '#EAB308', '#22C55E', '#14B8A6',
  '#3B82F6', '#6B7280', '#1F2937', '#F3F4F6',
];

// Font options
const FONT_FAMILIES = [
  { value: 'Inter', label: 'Inter' },
  { value: 'Arial', label: 'Arial' },
  { value: 'Georgia', label: 'Georgia' },
  { value: 'Courier New', label: 'Courier' },
  { value: 'Times New Roman', label: 'Times' },
];

const FONT_WEIGHTS = [
  { value: 'normal', label: 'Normal' },
  { value: 'medium', label: 'Medium' },
  { value: 'semibold', label: 'Semi-bold' },
  { value: 'bold', label: 'Gras' },
];

const TEXT_ALIGNS = [
  { value: 'left', icon: '⬅️', label: 'Gauche' },
  { value: 'center', icon: '↔️', label: 'Centre' },
  { value: 'right', icon: '➡️', label: 'Droite' },
];

export const ElementPropertiesPanel = memo(function ElementPropertiesPanel({
  element,
  onUpdate,
  disabled = false,
}: ElementPropertiesPanelProps) {
  const [showColorPicker, setShowColorPicker] = useState<'text' | 'fill' | 'stroke' | 'bg' | null>(null);

  // Update text content
  const updateTextContent = useCallback((updates: Partial<TextBlockContent>) => {
    if (!element?.textContent) return;
    onUpdate({
      textContent: {
        ...element.textContent,
        ...updates,
      },
    });
  }, [element, onUpdate]);

  // Update shape content
  const updateShapeContent = useCallback((updates: Partial<ShapeContent>) => {
    if (!element?.shapeContent) return;
    onUpdate({
      shapeContent: {
        ...element.shapeContent,
        ...updates,
      },
    });
  }, [element, onUpdate]);

  // Update image content
  const updateImageContent = useCallback((updates: Partial<ImageElementContent>) => {
    if (!element?.imageContent) return;
    onUpdate({
      imageContent: {
        ...element.imageContent,
        ...updates,
      },
    });
  }, [element, onUpdate]);

  if (!element) {
    return (
      <div className="p-4 text-center text-gray-500 text-sm">
        <svg className="w-8 h-8 mx-auto mb-2 opacity-30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
        </svg>
        <p>Sélectionnez un élément pour modifier ses propriétés</p>
      </div>
    );
  }

  // Color picker component
  const ColorPicker = ({
    value,
    onChange,
    label,
    pickerKey,
  }: {
    value: string;
    onChange: (color: string) => void;
    label: string;
    pickerKey: 'text' | 'fill' | 'stroke' | 'bg';
  }) => (
    <div className="relative">
      <label className="block text-xs text-gray-400 mb-1">{label}</label>
      <button
        onClick={() => setShowColorPicker(showColorPicker === pickerKey ? null : pickerKey)}
        disabled={disabled}
        className="w-full flex items-center gap-2 px-2 py-1.5 bg-gray-800 rounded border border-gray-700 hover:border-gray-600 disabled:opacity-50"
      >
        <div
          className="w-5 h-5 rounded border border-gray-600"
          style={{ backgroundColor: value }}
        />
        <span className="text-xs text-gray-300 flex-1 text-left font-mono">{value}</span>
      </button>

      {showColorPicker === pickerKey && (
        <div className="absolute top-full left-0 mt-1 p-2 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-50">
          <div className="grid grid-cols-7 gap-1 mb-2">
            {COLOR_PRESETS.map((color) => (
              <button
                key={color}
                onClick={() => {
                  onChange(color);
                  setShowColorPicker(null);
                }}
                className="w-6 h-6 rounded border border-gray-600 hover:scale-110 transition-transform"
                style={{ backgroundColor: color }}
                title={color}
              />
            ))}
          </div>
          <input
            type="color"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            className="w-full h-8 rounded cursor-pointer"
          />
        </div>
      )}
    </div>
  );

  return (
    <div className="p-3 space-y-4 text-sm">
      {/* Header */}
      <div className="flex items-center gap-2 pb-2 border-b border-gray-700">
        <div className={`w-3 h-3 rounded ${
          element.type === 'image' ? 'bg-blue-500' :
          element.type === 'text_block' ? 'bg-green-500' :
          'bg-purple-500'
        }`} />
        <span className="text-white font-medium">
          {element.type === 'image' ? 'Image' :
           element.type === 'text_block' ? 'Texte' :
           'Forme'}
        </span>
      </div>

      {/* Position & Size */}
      <div>
        <h4 className="text-xs text-gray-400 uppercase tracking-wide mb-2">Position & Taille</h4>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-xs text-gray-500 mb-1">X (%)</label>
            <input
              type="number"
              value={Math.round(element.x)}
              onChange={(e) => onUpdate({ x: parseFloat(e.target.value) || 0 })}
              disabled={disabled}
              className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-white text-xs focus:border-purple-500 focus:outline-none disabled:opacity-50"
              min={0}
              max={100}
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Y (%)</label>
            <input
              type="number"
              value={Math.round(element.y)}
              onChange={(e) => onUpdate({ y: parseFloat(e.target.value) || 0 })}
              disabled={disabled}
              className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-white text-xs focus:border-purple-500 focus:outline-none disabled:opacity-50"
              min={0}
              max={100}
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Largeur (%)</label>
            <input
              type="number"
              value={Math.round(element.width)}
              onChange={(e) => onUpdate({ width: parseFloat(e.target.value) || 5 })}
              disabled={disabled}
              className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-white text-xs focus:border-purple-500 focus:outline-none disabled:opacity-50"
              min={5}
              max={100}
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Hauteur (%)</label>
            <input
              type="number"
              value={Math.round(element.height)}
              onChange={(e) => onUpdate({ height: parseFloat(e.target.value) || 5 })}
              disabled={disabled}
              className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-white text-xs focus:border-purple-500 focus:outline-none disabled:opacity-50"
              min={5}
              max={100}
            />
          </div>
        </div>
      </div>

      {/* Text-specific properties */}
      {element.type === 'text_block' && element.textContent && (
        <>
          {/* Text content */}
          <div>
            <h4 className="text-xs text-gray-400 uppercase tracking-wide mb-2">Contenu</h4>
            <textarea
              value={element.textContent.text}
              onChange={(e) => updateTextContent({ text: e.target.value })}
              disabled={disabled}
              rows={3}
              className="w-full px-2 py-1.5 bg-gray-800 border border-gray-700 rounded text-white text-xs focus:border-purple-500 focus:outline-none resize-none disabled:opacity-50"
              placeholder="Texte..."
            />
          </div>

          {/* Typography */}
          <div>
            <h4 className="text-xs text-gray-400 uppercase tracking-wide mb-2">Typographie</h4>
            <div className="space-y-2">
              {/* Font family */}
              <div>
                <label className="block text-xs text-gray-500 mb-1">Police</label>
                <select
                  value={element.textContent.fontFamily || 'Inter'}
                  onChange={(e) => updateTextContent({ fontFamily: e.target.value })}
                  disabled={disabled}
                  className="w-full px-2 py-1.5 bg-gray-800 border border-gray-700 rounded text-white text-xs focus:border-purple-500 focus:outline-none disabled:opacity-50"
                >
                  {FONT_FAMILIES.map((font) => (
                    <option key={font.value} value={font.value}>{font.label}</option>
                  ))}
                </select>
              </div>

              {/* Font size and weight */}
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Taille (px)</label>
                  <input
                    type="number"
                    value={element.textContent.fontSize || 16}
                    onChange={(e) => updateTextContent({ fontSize: parseInt(e.target.value) || 16 })}
                    disabled={disabled}
                    className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-white text-xs focus:border-purple-500 focus:outline-none disabled:opacity-50"
                    min={8}
                    max={200}
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Graisse</label>
                  <select
                    value={element.textContent.fontWeight || 'normal'}
                    onChange={(e) => updateTextContent({ fontWeight: e.target.value as any })}
                    disabled={disabled}
                    className="w-full px-2 py-1.5 bg-gray-800 border border-gray-700 rounded text-white text-xs focus:border-purple-500 focus:outline-none disabled:opacity-50"
                  >
                    {FONT_WEIGHTS.map((weight) => (
                      <option key={weight.value} value={weight.value}>{weight.label}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Text align */}
              <div>
                <label className="block text-xs text-gray-500 mb-1">Alignement</label>
                <div className="flex gap-1">
                  {TEXT_ALIGNS.map((align) => (
                    <button
                      key={align.value}
                      onClick={() => updateTextContent({ textAlign: align.value as any })}
                      disabled={disabled}
                      className={`flex-1 py-1.5 rounded text-xs transition-colors ${
                        element.textContent?.textAlign === align.value
                          ? 'bg-purple-600 text-white'
                          : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                      } disabled:opacity-50`}
                      title={align.label}
                    >
                      {align.icon}
                    </button>
                  ))}
                </div>
              </div>

              {/* Line height */}
              <div>
                <label className="block text-xs text-gray-500 mb-1">
                  Interligne: {element.textContent.lineHeight || 1.5}
                </label>
                <input
                  type="range"
                  value={element.textContent.lineHeight || 1.5}
                  onChange={(e) => updateTextContent({ lineHeight: parseFloat(e.target.value) })}
                  disabled={disabled}
                  min={1}
                  max={3}
                  step={0.1}
                  className="w-full h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer disabled:opacity-50"
                />
              </div>
            </div>
          </div>

          {/* Colors */}
          <div>
            <h4 className="text-xs text-gray-400 uppercase tracking-wide mb-2">Couleurs</h4>
            <div className="space-y-2">
              <ColorPicker
                value={element.textContent.color || '#FFFFFF'}
                onChange={(color) => updateTextContent({ color })}
                label="Couleur du texte"
                pickerKey="text"
              />
              <ColorPicker
                value={element.textContent.backgroundColor || 'transparent'}
                onChange={(color) => updateTextContent({ backgroundColor: color === 'transparent' ? undefined : color })}
                label="Arrière-plan"
                pickerKey="bg"
              />
            </div>
          </div>
        </>
      )}

      {/* Shape-specific properties */}
      {element.type === 'shape' && element.shapeContent && (
        <div>
          <h4 className="text-xs text-gray-400 uppercase tracking-wide mb-2">Apparence</h4>
          <div className="space-y-2">
            <ColorPicker
              value={element.shapeContent.fillColor || '#6366F1'}
              onChange={(color) => updateShapeContent({ fillColor: color })}
              label="Couleur de remplissage"
              pickerKey="fill"
            />

            <ColorPicker
              value={element.shapeContent.strokeColor || 'transparent'}
              onChange={(color) => updateShapeContent({ strokeColor: color === 'transparent' ? undefined : color })}
              label="Couleur de bordure"
              pickerKey="stroke"
            />

            <div>
              <label className="block text-xs text-gray-500 mb-1">Épaisseur bordure (px)</label>
              <input
                type="number"
                value={element.shapeContent.strokeWidth || 0}
                onChange={(e) => updateShapeContent({ strokeWidth: parseInt(e.target.value) || 0 })}
                disabled={disabled}
                className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-white text-xs focus:border-purple-500 focus:outline-none disabled:opacity-50"
                min={0}
                max={20}
              />
            </div>

            <div>
              <label className="block text-xs text-gray-500 mb-1">
                Opacité: {Math.round((element.shapeContent.opacity ?? 1) * 100)}%
              </label>
              <input
                type="range"
                value={element.shapeContent.opacity ?? 1}
                onChange={(e) => updateShapeContent({ opacity: parseFloat(e.target.value) })}
                disabled={disabled}
                min={0}
                max={1}
                step={0.05}
                className="w-full h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer disabled:opacity-50"
              />
            </div>

            {element.shapeContent.shape === 'rounded_rect' && (
              <div>
                <label className="block text-xs text-gray-500 mb-1">Arrondi (px)</label>
                <input
                  type="number"
                  value={element.shapeContent.borderRadius || 8}
                  onChange={(e) => updateShapeContent({ borderRadius: parseInt(e.target.value) || 0 })}
                  disabled={disabled}
                  className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-white text-xs focus:border-purple-500 focus:outline-none disabled:opacity-50"
                  min={0}
                  max={100}
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Image-specific properties */}
      {element.type === 'image' && element.imageContent && (
        <div>
          <h4 className="text-xs text-gray-400 uppercase tracking-wide mb-2">Image</h4>
          <div className="space-y-2">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Ajustement</label>
              <select
                value={element.imageContent.fit || 'cover'}
                onChange={(e) => updateImageContent({ fit: e.target.value as any })}
                disabled={disabled}
                className="w-full px-2 py-1.5 bg-gray-800 border border-gray-700 rounded text-white text-xs focus:border-purple-500 focus:outline-none disabled:opacity-50"
              >
                <option value="cover">Couvrir</option>
                <option value="contain">Contenir</option>
                <option value="fill">Remplir</option>
              </select>
            </div>

            <div>
              <label className="block text-xs text-gray-500 mb-1">
                Opacité: {Math.round((element.imageContent.opacity ?? 1) * 100)}%
              </label>
              <input
                type="range"
                value={element.imageContent.opacity ?? 1}
                onChange={(e) => updateImageContent({ opacity: parseFloat(e.target.value) })}
                disabled={disabled}
                min={0}
                max={1}
                step={0.05}
                className="w-full h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer disabled:opacity-50"
              />
            </div>

            <div>
              <label className="block text-xs text-gray-500 mb-1">Arrondi (%)</label>
              <input
                type="number"
                value={element.imageContent.borderRadius || 0}
                onChange={(e) => updateImageContent({ borderRadius: parseInt(e.target.value) || 0 })}
                disabled={disabled}
                className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-white text-xs focus:border-purple-500 focus:outline-none disabled:opacity-50"
                min={0}
                max={50}
              />
            </div>
          </div>
        </div>
      )}

      {/* Visibility toggle */}
      <div className="pt-2 border-t border-gray-700">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={element.visible !== false}
            onChange={(e) => onUpdate({ visible: e.target.checked })}
            disabled={disabled}
            className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-purple-500 focus:ring-purple-500 focus:ring-offset-0"
          />
          <span className="text-xs text-gray-300">Visible</span>
        </label>
      </div>
    </div>
  );
});

export default ElementPropertiesPanel;
