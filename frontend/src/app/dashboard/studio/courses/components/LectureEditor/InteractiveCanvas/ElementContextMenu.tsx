'use client';

import React, { memo, useEffect, useRef } from 'react';

interface ContextMenuPosition {
  x: number;
  y: number;
}

interface ElementContextMenuProps {
  position: ContextMenuPosition;
  onClose: () => void;
  // Actions
  onCopy: () => void;
  onPaste: () => void;
  onDuplicate: () => void;
  onDelete: () => void;
  onBringToFront: () => void;
  onSendToBack: () => void;
  onToggleLock: () => void;
  // State
  isLocked: boolean;
  hasClipboard: boolean;
}

export const ElementContextMenu = memo(function ElementContextMenu({
  position,
  onClose,
  onCopy,
  onPaste,
  onDuplicate,
  onDelete,
  onBringToFront,
  onSendToBack,
  onToggleLock,
  isLocked,
  hasClipboard,
}: ElementContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);

  // Close on click outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [onClose]);

  const MenuItem = ({
    onClick,
    icon,
    label,
    shortcut,
    disabled = false,
    danger = false,
  }: {
    onClick: () => void;
    icon: React.ReactNode;
    label: string;
    shortcut?: string;
    disabled?: boolean;
    danger?: boolean;
  }) => (
    <button
      onClick={() => {
        if (!disabled) {
          onClick();
          onClose();
        }
      }}
      disabled={disabled}
      className={`w-full flex items-center gap-3 px-3 py-2 text-left text-sm transition-colors ${
        disabled
          ? 'text-gray-600 cursor-not-allowed'
          : danger
            ? 'text-red-400 hover:bg-red-500/20'
            : 'text-gray-300 hover:bg-gray-700'
      }`}
    >
      <span className="w-4 h-4 flex items-center justify-center">{icon}</span>
      <span className="flex-1">{label}</span>
      {shortcut && (
        <span className="text-xs text-gray-500">{shortcut}</span>
      )}
    </button>
  );

  const Divider = () => <div className="h-px bg-gray-700 my-1" />;

  return (
    <div
      ref={menuRef}
      className="fixed bg-gray-800 border border-gray-700 rounded-lg shadow-xl py-1 z-[100] min-w-[180px]"
      style={{
        left: position.x,
        top: position.y,
      }}
    >
      <MenuItem
        onClick={onCopy}
        icon={
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
        }
        label="Copier"
        shortcut="Ctrl+C"
      />
      <MenuItem
        onClick={onPaste}
        disabled={!hasClipboard}
        icon={
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
        }
        label="Coller"
        shortcut="Ctrl+V"
      />
      <MenuItem
        onClick={onDuplicate}
        icon={
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
          </svg>
        }
        label="Dupliquer"
        shortcut="Ctrl+D"
      />

      <Divider />

      <MenuItem
        onClick={onBringToFront}
        icon={
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 11l7-7 7 7M5 19l7-7 7 7" />
          </svg>
        }
        label="Premier plan"
        shortcut="Ctrl+]"
      />
      <MenuItem
        onClick={onSendToBack}
        icon={
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 13l-7 7-7-7m14-8l-7 7-7-7" />
          </svg>
        }
        label="Arrière-plan"
        shortcut="Ctrl+["
      />

      <Divider />

      <MenuItem
        onClick={onToggleLock}
        icon={
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
            {isLocked ? (
              <path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3.1-9H8.9V6c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2z" />
            ) : (
              <path d="M12 17c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm6-9h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6h1.9c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm0 12H6V10h12v10z" />
            )}
          </svg>
        }
        label={isLocked ? 'Déverrouiller' : 'Verrouiller'}
      />

      <Divider />

      <MenuItem
        onClick={onDelete}
        danger
        icon={
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        }
        label="Supprimer"
        shortcut="Suppr"
      />
    </div>
  );
});

export default ElementContextMenu;
