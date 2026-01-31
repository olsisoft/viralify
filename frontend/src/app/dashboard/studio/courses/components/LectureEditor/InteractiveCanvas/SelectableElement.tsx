'use client';

import React, { memo, useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { SlideElement, TextBlockContent } from '../../../lib/lecture-editor-types';
import { IMAGE_CLIP_SHAPES } from '../../../lib/lecture-editor-types';

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

// Spring config for smooth animations
const springConfig = {
  type: 'spring' as const,
  stiffness: 300,
  damping: 30,
  mass: 0.8,
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
  const [isHovered, setIsHovered] = useState(false);
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

  // Get clip-path value for the current shape
  const getClipPath = (shapeId: string | undefined): string | undefined => {
    if (!shapeId || shapeId === 'none') return undefined;
    const shape = IMAGE_CLIP_SHAPES.find(s => s.id === shapeId);
    return shape?.clipPath !== 'none' ? shape?.clipPath : undefined;
  };

  const renderContent = () => {
    switch (element.type) {
      case 'image':
        if (!element.imageContent?.url) return null;
        const clipPath = getClipPath(element.imageContent.clipShape);
        return (
          <motion.img
            src={element.imageContent.url}
            alt=""
            className="w-full h-full pointer-events-none select-none"
            style={{
              objectFit: element.imageContent.fit || 'cover',
              opacity: element.imageContent.opacity ?? 1,
              borderRadius: clipPath ? undefined : `${element.imageContent.borderRadius || 0}%`,
              clipPath: clipPath,
              WebkitClipPath: clipPath, // Safari support
            }}
            draggable={false}
            initial={{ opacity: 0, clipPath: clipPath }}
            animate={{
              opacity: element.imageContent.opacity ?? 1,
              clipPath: clipPath,
            }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
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
            <motion.div
              className="w-full h-full pointer-events-none"
              style={{
                backgroundColor: fillColor || '#6366F1',
                border: strokeColor ? `${strokeWidth || 1}px solid ${strokeColor}` : 'none',
                borderRadius: '50%',
                opacity: opacity ?? 1,
              }}
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: opacity ?? 1 }}
              transition={springConfig}
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
          <motion.div
            className="w-full h-full pointer-events-none"
            style={{
              backgroundColor: fillColor || '#6366F1',
              border: strokeColor ? `${strokeWidth || 1}px solid ${strokeColor}` : 'none',
              borderRadius: shape === 'rounded_rect' ? `${borderRadius || 8}px` : 0,
              opacity: opacity ?? 1,
            }}
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: opacity ?? 1 }}
            transition={springConfig}
          />
        );

      default:
        return null;
    }
  };

  return (
    <motion.div
      className={`
        absolute
        ${element.locked ? 'cursor-not-allowed' : isEditing ? 'cursor-text' : 'cursor-move'}
        ${disabled ? 'pointer-events-none' : ''}
      `}
      style={{
        left: `${element.x}%`,
        top: `${element.y}%`,
        width: `${element.width}%`,
        height: `${element.height}%`,
        zIndex: element.zIndex,
      }}
      initial={false}
      animate={{
        rotate: element.rotation || 0,
        scale: isHovered && !isEditing ? 1.01 : 1,
      }}
      transition={springConfig}
      whileTap={!element.locked && !disabled ? { scale: 0.99 } : undefined}
      onClick={onSelect}
      onDoubleClick={handleDoubleClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
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
      <motion.div
        className="w-full h-full overflow-hidden"
        animate={{
          boxShadow: isSelected
            ? '0 0 0 2px rgb(168 85 247), 0 0 0 3px rgba(168, 85, 247, 0.3)'
            : isHovered
              ? '0 0 0 1px rgba(168, 85, 247, 0.5)'
              : 'none',
        }}
        transition={{ duration: 0.15 }}
        style={{
          opacity: element.locked ? 0.75 : 1,
        }}
      >
        {renderContent()}
      </motion.div>

      {/* Selection handles */}
      <AnimatePresence>
        {isSelected && !element.locked && !disabled && (
          <>
            {/* Corner and edge handles */}
            {(Object.keys(HANDLE_POSITIONS) as ResizeHandle[]).map((handle) => (
              <motion.div
                key={handle}
                className="absolute w-2 h-2 bg-white border-2 border-purple-500 rounded-sm z-10"
                style={{
                  ...HANDLE_POSITIONS[handle].style,
                  cursor: HANDLE_POSITIONS[handle].cursor,
                }}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0, opacity: 0 }}
                transition={{ duration: 0.15 }}
                whileHover={{ scale: 1.3, backgroundColor: '#E9D5FF' }}
                onMouseDown={(e) => {
                  e.stopPropagation();
                  onStartResize(handle, e);
                }}
              />
            ))}

            {/* Rotation handle */}
            {onStartRotate && (
              <motion.div
                className="absolute left-1/2 -top-8 flex flex-col items-center z-20"
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 5 }}
                transition={{ duration: 0.15 }}
                style={{ marginLeft: -8 }}
              >
                {/* Connecting line */}
                <div className="w-px h-4 bg-purple-500" />
                {/* Rotation handle */}
                <motion.div
                  className="w-4 h-4 bg-white border-2 border-purple-500 rounded-full cursor-grab flex items-center justify-center"
                  whileHover={{ scale: 1.2, backgroundColor: '#E9D5FF' }}
                  whileTap={{ scale: 0.95 }}
                  onMouseDown={(e) => {
                    e.stopPropagation();
                    onStartRotate(e);
                  }}
                  title="Rotation"
                >
                  <svg className="w-2.5 h-2.5 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                </motion.div>
              </motion.div>
            )}

            {/* Delete button */}
            <motion.button
              className="absolute -top-3 -right-3 w-5 h-5 bg-red-500 text-white rounded-full flex items-center justify-center text-xs z-20 shadow-md"
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0, opacity: 0 }}
              transition={{ duration: 0.15 }}
              whileHover={{ scale: 1.15, backgroundColor: '#DC2626' }}
              whileTap={{ scale: 0.95 }}
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
            </motion.button>

            {/* Lock/Unlock button */}
            {onToggleLock && (
              <motion.button
                className="absolute -top-3 -left-3 w-5 h-5 bg-gray-700 text-white rounded-full flex items-center justify-center text-xs z-20 shadow-md"
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0, opacity: 0 }}
                transition={{ duration: 0.15 }}
                whileHover={{ scale: 1.15, backgroundColor: '#4B5563' }}
                whileTap={{ scale: 0.95 }}
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
              </motion.button>
            )}
          </>
        )}
      </AnimatePresence>

      {/* Locked indicator overlay */}
      {element.locked && isSelected && (
        <motion.div
          className="absolute inset-0 bg-gray-900/20 flex items-center justify-center pointer-events-none"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <div className="bg-gray-900/80 px-2 py-1 rounded text-xs text-gray-300 flex items-center gap-1">
            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
              <path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3.1-9H8.9V6c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2z" />
            </svg>
            Verrouillé
          </div>
        </motion.div>
      )}
    </motion.div>
  );
});

export default SelectableElement;
