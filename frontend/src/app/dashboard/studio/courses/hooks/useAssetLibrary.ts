'use client';

import { useState, useCallback, useMemo } from 'react';

/**
 * Asset types for the library
 */
export interface Asset {
  id: string;
  type: 'image' | 'video' | 'audio';
  url: string;
  thumbnailUrl?: string;
  filename: string;
  size: number; // bytes
  width?: number;
  height?: number;
  duration?: number; // seconds for video/audio
  createdAt: string;
  tags?: string[];
}

export interface UploadProgress {
  assetId: string;
  progress: number; // 0-100
  status: 'uploading' | 'processing' | 'completed' | 'failed';
  error?: string;
}

export interface UseAssetLibraryOptions {
  jobId: string;
  onUploadComplete?: (asset: Asset) => void;
  onError?: (error: string) => void;
}

export interface UseAssetLibraryReturn {
  // State
  assets: Asset[];
  isLoading: boolean;
  uploadProgress: UploadProgress[];
  selectedAssetId: string | null;
  searchQuery: string;
  filterType: 'all' | 'image' | 'video' | 'audio';

  // Computed
  filteredAssets: Asset[];

  // Actions
  uploadAsset: (file: File) => Promise<Asset | null>;
  uploadMultipleAssets: (files: FileList) => Promise<Asset[]>;
  deleteAsset: (assetId: string) => void;
  selectAsset: (assetId: string | null) => void;
  setSearchQuery: (query: string) => void;
  setFilterType: (type: 'all' | 'image' | 'video' | 'audio') => void;
  getAssetById: (assetId: string) => Asset | undefined;
  clearAssets: () => void;
}

/**
 * Hook for managing the asset library
 * Stores assets in memory for the current editing session
 */
export function useAssetLibrary({
  jobId,
  onUploadComplete,
  onError,
}: UseAssetLibraryOptions): UseAssetLibraryReturn {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress[]>([]);
  const [selectedAssetId, setSelectedAssetId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'image' | 'video' | 'audio'>('all');

  // Generate unique asset ID
  const generateAssetId = useCallback(() => {
    return `asset-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }, []);

  // Determine asset type from file
  const getAssetType = useCallback((file: File): 'image' | 'video' | 'audio' => {
    if (file.type.startsWith('image/')) return 'image';
    if (file.type.startsWith('video/')) return 'video';
    if (file.type.startsWith('audio/')) return 'audio';
    return 'image'; // fallback
  }, []);

  // Create thumbnail for images
  const createThumbnail = useCallback((file: File): Promise<string> => {
    return new Promise((resolve) => {
      if (!file.type.startsWith('image/')) {
        resolve('');
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement('canvas');
          const maxSize = 150;
          let width = img.width;
          let height = img.height;

          if (width > height) {
            if (width > maxSize) {
              height = (height * maxSize) / width;
              width = maxSize;
            }
          } else {
            if (height > maxSize) {
              width = (width * maxSize) / height;
              height = maxSize;
            }
          }

          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d');
          ctx?.drawImage(img, 0, 0, width, height);
          resolve(canvas.toDataURL('image/jpeg', 0.7));
        };
        img.src = e.target?.result as string;
      };
      reader.readAsDataURL(file);
    });
  }, []);

  // Get image dimensions
  const getImageDimensions = useCallback((file: File): Promise<{ width: number; height: number }> => {
    return new Promise((resolve) => {
      if (!file.type.startsWith('image/')) {
        resolve({ width: 0, height: 0 });
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          resolve({ width: img.width, height: img.height });
        };
        img.onerror = () => resolve({ width: 0, height: 0 });
        img.src = e.target?.result as string;
      };
      reader.readAsDataURL(file);
    });
  }, []);

  // Upload a single asset
  const uploadAsset = useCallback(async (file: File): Promise<Asset | null> => {
    const assetId = generateAssetId();
    const assetType = getAssetType(file);

    // Add to progress tracking
    setUploadProgress((prev) => [
      ...prev,
      { assetId, progress: 0, status: 'uploading' },
    ]);

    try {
      // Simulate upload progress (in real app, this would be actual upload)
      setUploadProgress((prev) =>
        prev.map((p) =>
          p.assetId === assetId ? { ...p, progress: 30 } : p
        )
      );

      // Create object URL for the file (in-memory storage)
      const url = URL.createObjectURL(file);

      // Get thumbnail and dimensions for images
      const [thumbnailUrl, dimensions] = await Promise.all([
        createThumbnail(file),
        getImageDimensions(file),
      ]);

      setUploadProgress((prev) =>
        prev.map((p) =>
          p.assetId === assetId ? { ...p, progress: 70, status: 'processing' } : p
        )
      );

      // Create asset object
      const asset: Asset = {
        id: assetId,
        type: assetType,
        url,
        thumbnailUrl: thumbnailUrl || undefined,
        filename: file.name,
        size: file.size,
        width: dimensions.width || undefined,
        height: dimensions.height || undefined,
        createdAt: new Date().toISOString(),
      };

      // Complete upload
      setUploadProgress((prev) =>
        prev.map((p) =>
          p.assetId === assetId ? { ...p, progress: 100, status: 'completed' } : p
        )
      );

      // Add to assets
      setAssets((prev) => [...prev, asset]);

      // Remove from progress after a delay
      setTimeout(() => {
        setUploadProgress((prev) => prev.filter((p) => p.assetId !== assetId));
      }, 1000);

      onUploadComplete?.(asset);
      return asset;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';

      setUploadProgress((prev) =>
        prev.map((p) =>
          p.assetId === assetId ? { ...p, status: 'failed', error: errorMessage } : p
        )
      );

      onError?.(errorMessage);
      return null;
    }
  }, [generateAssetId, getAssetType, createThumbnail, getImageDimensions, onUploadComplete, onError]);

  // Upload multiple assets
  const uploadMultipleAssets = useCallback(async (files: FileList): Promise<Asset[]> => {
    setIsLoading(true);
    const results: Asset[] = [];

    for (const file of Array.from(files)) {
      const asset = await uploadAsset(file);
      if (asset) {
        results.push(asset);
      }
    }

    setIsLoading(false);
    return results;
  }, [uploadAsset]);

  // Delete an asset
  const deleteAsset = useCallback((assetId: string) => {
    const asset = assets.find((a) => a.id === assetId);
    if (asset) {
      // Revoke object URL to free memory
      URL.revokeObjectURL(asset.url);
      if (asset.thumbnailUrl) {
        URL.revokeObjectURL(asset.thumbnailUrl);
      }
    }

    setAssets((prev) => prev.filter((a) => a.id !== assetId));

    if (selectedAssetId === assetId) {
      setSelectedAssetId(null);
    }
  }, [assets, selectedAssetId]);

  // Select an asset
  const selectAsset = useCallback((assetId: string | null) => {
    setSelectedAssetId(assetId);
  }, []);

  // Get asset by ID
  const getAssetById = useCallback((assetId: string): Asset | undefined => {
    return assets.find((a) => a.id === assetId);
  }, [assets]);

  // Clear all assets
  const clearAssets = useCallback(() => {
    // Revoke all object URLs
    assets.forEach((asset) => {
      URL.revokeObjectURL(asset.url);
      if (asset.thumbnailUrl) {
        URL.revokeObjectURL(asset.thumbnailUrl);
      }
    });

    setAssets([]);
    setSelectedAssetId(null);
  }, [assets]);

  // Filter assets based on search and type
  const filteredAssets = useMemo(() => {
    return assets.filter((asset) => {
      // Type filter
      if (filterType !== 'all' && asset.type !== filterType) {
        return false;
      }

      // Search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        const matchesFilename = asset.filename.toLowerCase().includes(query);
        const matchesTags = asset.tags?.some((tag) => tag.toLowerCase().includes(query));
        return matchesFilename || matchesTags;
      }

      return true;
    });
  }, [assets, filterType, searchQuery]);

  return {
    // State
    assets,
    isLoading,
    uploadProgress,
    selectedAssetId,
    searchQuery,
    filterType,

    // Computed
    filteredAssets,

    // Actions
    uploadAsset,
    uploadMultipleAssets,
    deleteAsset,
    selectAsset,
    setSearchQuery,
    setFilterType,
    getAssetById,
    clearAssets,
  };
}

export default useAssetLibrary;
