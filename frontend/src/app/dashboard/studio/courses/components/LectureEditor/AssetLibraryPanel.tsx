'use client';

import React, { useCallback, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { Asset, UploadProgress } from '../../hooks/useAssetLibrary';

interface AssetLibraryPanelProps {
  assets: Asset[];
  filteredAssets: Asset[];
  uploadProgress: UploadProgress[];
  selectedAssetId: string | null;
  searchQuery: string;
  filterType: 'all' | 'image' | 'video' | 'audio';
  isLoading: boolean;
  onUpload: (files: FileList) => void;
  onDelete: (assetId: string) => void;
  onSelect: (assetId: string | null) => void;
  onSearchChange: (query: string) => void;
  onFilterChange: (type: 'all' | 'image' | 'video' | 'audio') => void;
  onDragStart?: (asset: Asset, e: React.DragEvent) => void;
}

// Format file size
function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function AssetLibraryPanel({
  assets,
  filteredAssets,
  uploadProgress,
  selectedAssetId,
  searchQuery,
  filterType,
  isLoading,
  onUpload,
  onDelete,
  onSelect,
  onSearchChange,
  onFilterChange,
  onDragStart,
}: AssetLibraryPanelProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  // Handle file input change
  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onUpload(e.target.files);
      // Reset input
      e.target.value = '';
    }
  }, [onUpload]);

  // Handle drop zone
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onUpload(e.dataTransfer.files);
    }
  }, [onUpload]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  // Handle asset drag start for canvas drop
  const handleAssetDragStart = useCallback((asset: Asset, e: React.DragEvent) => {
    e.dataTransfer.setData('application/json', JSON.stringify({
      type: 'asset',
      assetId: asset.id,
      assetType: asset.type,
      url: asset.url,
      width: asset.width,
      height: asset.height,
    }));
    e.dataTransfer.effectAllowed = 'copy';
    onDragStart?.(asset, e);
  }, [onDragStart]);

  // Get type icon
  const getTypeIcon = (type: 'image' | 'video' | 'audio') => {
    switch (type) {
      case 'image': return '🖼️';
      case 'video': return '🎬';
      case 'audio': return '🎵';
    }
  };

  return (
    <div className="h-full flex flex-col bg-gray-900/50">
      {/* Header */}
      <div className="p-3 border-b border-gray-800">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-white">Bibliothèque</h3>
          <span className="text-xs text-gray-500">{assets.length} assets</span>
        </div>

        {/* Search */}
        <div className="relative mb-2">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Rechercher..."
            className="w-full px-3 py-1.5 pl-8 bg-gray-800 border border-gray-700 rounded-lg text-sm text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 transition-colors"
          />
          <svg
            className="absolute left-2.5 top-1/2 transform -translate-y-1/2 w-3.5 h-3.5 text-gray-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>

        {/* Filter tabs */}
        <div className="flex gap-1">
          {(['all', 'image', 'video', 'audio'] as const).map((type) => (
            <button
              key={type}
              onClick={() => onFilterChange(type)}
              className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                filterType === type
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}
            >
              {type === 'all' ? 'Tous' : type === 'image' ? '🖼️' : type === 'video' ? '🎬' : '🎵'}
            </button>
          ))}
        </div>
      </div>

      {/* Upload zone */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`m-3 p-4 border-2 border-dashed rounded-lg transition-all cursor-pointer ${
          isDragOver
            ? 'border-purple-500 bg-purple-500/10'
            : 'border-gray-700 hover:border-gray-600'
        }`}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept="image/*,video/*,audio/*"
          onChange={handleFileChange}
          className="hidden"
        />
        <div className="text-center">
          <svg
            className={`w-8 h-8 mx-auto mb-2 ${isDragOver ? 'text-purple-400' : 'text-gray-500'}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <p className="text-xs text-gray-400">
            {isDragOver ? 'Déposez les fichiers' : 'Glissez ou cliquez'}
          </p>
        </div>
      </div>

      {/* Upload progress */}
      <AnimatePresence>
        {uploadProgress.map((progress) => (
          <motion.div
            key={progress.assetId}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mx-3 mb-2"
          >
            <div className="bg-gray-800 rounded-lg p-2">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-gray-400 truncate">
                  {progress.status === 'uploading' ? 'Upload...' : progress.status === 'processing' ? 'Traitement...' : 'Terminé'}
                </span>
                <span className="text-xs text-gray-500">{progress.progress}%</span>
              </div>
              <div className="h-1 bg-gray-700 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${progress.progress}%` }}
                  className={`h-full rounded-full ${
                    progress.status === 'failed' ? 'bg-red-500' : 'bg-purple-500'
                  }`}
                />
              </div>
              {progress.error && (
                <p className="text-xs text-red-400 mt-1">{progress.error}</p>
              )}
            </div>
          </motion.div>
        ))}
      </AnimatePresence>

      {/* Assets grid */}
      <div className="flex-1 overflow-y-auto p-3">
        {filteredAssets.length === 0 ? (
          <div className="text-center py-8">
            <svg
              className="w-12 h-12 mx-auto mb-2 text-gray-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <p className="text-sm text-gray-500">
              {searchQuery ? 'Aucun résultat' : 'Aucun asset'}
            </p>
            <p className="text-xs text-gray-600 mt-1">
              Uploadez des images, vidéos ou sons
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-2">
            <AnimatePresence>
              {filteredAssets.map((asset) => (
                <motion.div
                  key={asset.id}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  whileHover={{ scale: 1.02 }}
                  draggable
                  onDragStart={(e) => handleAssetDragStart(asset, e as unknown as React.DragEvent)}
                  onClick={() => onSelect(asset.id === selectedAssetId ? null : asset.id)}
                  className={`relative group cursor-grab active:cursor-grabbing rounded-lg overflow-hidden border-2 transition-colors ${
                    selectedAssetId === asset.id
                      ? 'border-purple-500'
                      : 'border-transparent hover:border-gray-600'
                  }`}
                >
                  {/* Thumbnail */}
                  <div className="aspect-square bg-gray-800 flex items-center justify-center">
                    {asset.type === 'image' && asset.thumbnailUrl ? (
                      <img
                        src={asset.thumbnailUrl}
                        alt={asset.filename}
                        className="w-full h-full object-cover"
                        draggable={false}
                      />
                    ) : (
                      <span className="text-2xl">{getTypeIcon(asset.type)}</span>
                    )}
                  </div>

                  {/* Overlay with info */}
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
                    <div className="absolute bottom-0 left-0 right-0 p-1.5">
                      <p className="text-xs text-white truncate">{asset.filename}</p>
                      <p className="text-xs text-gray-400">{formatFileSize(asset.size)}</p>
                    </div>
                  </div>

                  {/* Delete button */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(asset.id);
                    }}
                    className="absolute top-1 right-1 p-1 bg-red-500/80 hover:bg-red-500 text-white rounded opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>

                  {/* Type badge */}
                  <div className="absolute top-1 left-1 px-1 py-0.5 bg-black/60 rounded text-xs">
                    {getTypeIcon(asset.type)}
                  </div>

                  {/* Drag hint */}
                  <div className="absolute inset-0 flex items-center justify-center bg-purple-600/20 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                    <span className="text-xs text-white bg-purple-600 px-2 py-1 rounded">
                      Glisser vers canvas
                    </span>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Footer stats */}
      <div className="p-3 border-t border-gray-800">
        <div className="flex justify-between text-xs text-gray-500">
          <span>
            {filteredAssets.length} sur {assets.length}
          </span>
          <span>
            {formatFileSize(assets.reduce((acc, a) => acc + a.size, 0))}
          </span>
        </div>
      </div>
    </div>
  );
}

export default AssetLibraryPanel;
