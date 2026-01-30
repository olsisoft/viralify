'use client';

import React, { memo, useState, useRef, useEffect, useCallback } from 'react';
import type { SlideElement, TextBlockContent } from '../../../lib/lecture-editor-types';

type ResizeHandle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w';

interface SelectableElementProps {
  element: SlideElement;
  isSelected: boolean;
  onSelect: (e: React.MouseEvent) => void;
  onStartDrag: (e: React.MouseEvent) => void;
  onStartResize: (handle: ResizeHandle, e: React.MouseEvent) => void;
  onStartRotate?: (e: React.MouseEvent) => void;
  onTextChange?: (text: string) => void;
  onToggleLock?: () => void;
  onContextMenu?: (e: React.MouseEvent) => void;
  disabled?: boolean;
}

// Resize handle positions
const HANDLE_POSITIONS: Record<ResizeHandle, { cursor: string; style: React.CSSProperties }> = {
  nw: { cursor: 'nw-resize', style: { top: -4, left: -4 } },
  n: { cursor: 'n-resize', style: { top: -4, left: '50%', transform: 'translateX(-50%)' } },
  ne: { cursor: 'ne-resize', style: { top: -4, right: -4 } },
  e: { cursor: 'e-resize', style: { top: '50%', right: -4, transform: 'translateY(-50%)' } },
  se: { cursor: 'se-resize', style: { bottom: -4, right: -4 } },
  s: { cursor: 's-resize', style: { bottom: -4, left: '50%', transform: 'translateX(-50%)' } },
  sw: { cursor: 'sw-resize', style: { bottom: -4, left: -4 } },
  w: { cursor: 'w-resize', style: { top: '50%', left: -4, transform: 'translateY(-50%)' } },
};

export const SelectableElement = memo(function SelectableElement({
  element,
  isSelected,
  onSelect,
  onStartDrag,
  onStartResize,
  onStartRotate,
  onTextChange,
  onToggleLock,
  onContextMenu,
  disabled = false,
}: SelectableElementProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Handle double-click to start editing text
  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    if (element.type !== 'text_block' || !element.textContent || disabled || element.locked) return;
    e.stopPropagation();
    setEditText(element.textContent.text);
    setIsEditing(true);
  }, [element, disabled]);

  // Focus textarea when editing starts
  useEffect(() => {
    if (isEditing && textareaRef.current) {
      textareaRef.current.focus();
      textareaRef.current.select();
    }
  }, [isEditing]);

  // Handle text editing completion
  const finishEditing = useCallback(() => {
    if (isEditing && onTextChange && editText !== element.textContent?.text) {
      onTextChange(editText);
    }
    setIsEditing(false);
  }, [isEditing, editText, element.textContent?.text, onTextChange]);

  // Handle keyboard in edit mode
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      setIsEditing(false);
      setEditText(element.textContent?.text || '');
    } else if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      finishEditing();
    }
    // Prevent event from bubbling to canvas keyboard handler
    e.stopPropagation();
  }, [element.textContent?.text, finishEditing]);

  if (!element.visible) return null;

  const renderContent = () => {
    switch (element.type) {
      case 'image':
        if (!element.imageContent?.url) return null;
        return (
          <img
            src={element.imageContent.url}
            alt=""
            className="w-full h-full pointer-events-none select-none"
            style={{
              objectFit: element.imageContent.fit || 'cover',
              opacity: element.imageContent.opacity ?? 1,
              borderRadius: `${element.imageContent.borderRadius || 0}%`,
            }}
            draggable={false}
          />
        );

      case 'text_block':
        if (!element.textContent) return null;

        // Inline editing mode
        if (isEditing) {
          return (
            <textarea
              ref={textareaRef}
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              onBlur={finishEditing}
              onKeyDown={handleKeyDown}
              className="w-full h-full resize-none border-none outline-none bg-transparent"
              style={{
                fontSize: `${element.textContent.fontSize || 16}px`,
                fontWeight: element.textContent.fontWeight || 'normal',
                fontFamily: element.textContent.fontFamily || 'Inter',
                color: element.textContent.color || '#FFFFFF',
                textAlign: element.textContent.textAlign || 'left',
                lineHeight: element.textContent.lineHeight || 1.5,
                padding: `${element.textContent.padding || 8}px`,
              }}
            />
          );
        }

        // Display mode
        return (
          <div
            className="w-full h-full overflow-hidden pointer-events-none select-none"
            style={{
              fontSize: `${element.textContent.fontSize || 16}px`,
              fontWeight: element.textContent.fontWeight || 'normal',
              fontFamily: element.textContent.fontFamily || 'Inter',
              color: element.textContent.color || '#FFFFFF',
              backgroundColor: element.textContent.backgroundColor || 'transparent',
              textAlign: element.textContent.textAlign || 'left',
              lineHeight: element.textContent.lineHeight || 1.5,
              padding: `${element.textContent.padding || 8}px`,
            }}
          >
            {element.textContent.text}
          </div>
        );

      case 'shape':
        if (!element.shapeContent) return null;
        const { shape, fillColor, strokeColor, strokeWidth, opacity, borderRadius } = element.shapeContent;

        if (shape === 'circle') {
          return (
            <div
              className="w-full h-full pointer-events-none"
              style={{
                backgroundColor: fillColor || '#6366F1',
                border: strokeColor ? `${strokeWidth || 1}px solid ${strokeColor}` : 'none',
                borderRadius: '50%',
                opacity: opacity ?? 1,
              }}
            />
          );
        }

        if (shape === 'line') {
          return (
            <div
              className="w-full pointer-events-none"
              style={{
                height: `${strokeWidth || 2}px`,
                backgroundColor: strokeColor || fillColor || '#6366F1',
                opacity: opacity ?? 1,
                position: 'absolute',
                top: '50%',
                transform: 'translateY(-50%)',
              }}
            />
          );
        }

        // Rectangle or rounded rect
        return (
          <div
            className="w-full h-full pointer-events-none"
            style={{
              backgroundColor: fillColor || '#6366F1',
              border: strokeColor ? `${strokeWidth || 1}px solid ${strokeColor}` : 'none',
              borderRadius: shape === 'rounded_rect' ? `${borderRadius || 8}px` : 0,
              opacity: opacity ?? 1,
            }}
          />
        );

      default:
        return null;
    }
  };

  return (
    <div
      className={`
        absolute transition-shadow
        ${isSelected ? 'ring-2 ring-purple-500 ring-offset-1 ring-offset-transparent' : ''}
        ${element.locked ? 'cursor-not-allowed opacity-75' : isEditing ? 'cursor-text' : 'cursor-move'}
        ${disabled ? 'pointer-events-none' : ''}
      `}
      style={{
        left: `${element.x}%`,
        top: `${element.y}%`,
        width: `${element.width}%`,
        height: `${element.height}%`,
        transform: element.rotation ? `rotate(${element.rotation}deg)` : undefined,
        zIndex: element.zIndex,
      }}
      onClick={onSelect}
      onDoubleClick={handleDoubleClick}
      onContextMenu={(e) => {
        e.preventDefault();
        onContextMenu?.(e);
      }}
      onMouseDown={(e) => {
        if (e.button !== 0) return; // Only left click
        if (isEditing) return; // Don't drag while editing
        onSelect(e);
        if (!element.locked && !disabled) {
          onStartDrag(e);
        }
      }}
    >
      {/* Element content */}
      <div className="w-full h-full overflow-hidden">
        {renderContent()}
      </div>

      {/* Selection handles */}
      {isSelected && !element.locked && !disabled && (
        <>
          {/* Corner and edge handles */}
          {(Object.keys(HANDLE_POSITIONS) as ResizeHandle[]).map((handle) => (
            <div
              key={handle}
              className="absolute w-2 h-2 bg-white border-2 border-purple-500 rounded-sm z-10 hover:bg-purple-100"
              style={{
                ...HANDLE_POSITIONS[handle].style,
                cursor: HANDLE_POSITIONS[handle].cursor,
              }}
              onMouseDown={(e) => {
                e.stopPropagation();
                onStartResize(handle, e);
              }}
            />
          ))}

          {/* Rotation handle */}
          {onStartRotate && (
            <div className="absolute left-1/2 -top-8 flex flex-col items-center z-20">
              {/* Connecting line */}
              <div className="w-px h-4 bg-purple-500" />
              {/* Rotation handle */}
              <div
                className="w-4 h-4 bg-white border-2 border-purple-500 rounded-full cursor-grab hover:bg-purple-100 flex items-center justify-center"
                onMouseDown={(e) => {
                  e.stopPropagation();
                  onStartRotate(e);
                }}
                title="Rotation"
              >
                <svg className="w-2.5 h-2.5 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </div>
            </div>
          )}

          {/* Delete button */}
          <button
            className="absolute -top-3 -right-3 w-5 h-5 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center text-xs z-20 shadow-md transition-colors"
            onClick={(e) => {
              e.stopPropagation();
              // Trigger delete via keyboard event simulation
              const event = new KeyboardEvent('keydown', { key: 'Delete', bubbles: true });
              window.dispatchEvent(event);
            }}
            title="Supprimer"
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>

          {/* Lock/Unlock button */}
          {onToggleLock && (
            <button
              className="absolute -top-3 -left-3 w-5 h-5 bg-gray-700 hover:bg-gray-600 text-white rounded-full flex items-center justify-center text-xs z-20 shadow-md transition-colors"
              onClick={(e) => {
                e.stopPropagation();
                onToggleLock();
              }}
              title={element.locked ? 'Déverrouiller' : 'Verrouiller'}
            >
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                {element.locked ? (
                  <path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3.1-9H8.9V6c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2z" />
                ) : (
                  <path d="M12 17c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm6-9h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6h1.9c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm0 12H6V10h12v10z" />
                )}
              </svg>
            </button>
          )}
        </>
      )}
    </div>
  );
});

export default SelectableElement;
