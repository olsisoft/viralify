'use client';

/**
 * Voice Sample Upload Component
 * Allows users to upload voice samples for cloning
 */

import React, { useState, useRef, useCallback } from 'react';
import {
  VoiceSample,
  VoiceSampleRequirements,
  TrainingRequirements,
  formatDuration,
  getSampleStatusLabel,
  getSampleStatusColor,
  getQualityLabel,
} from '../lib/voice-types';

interface VoiceSampleUploadProps {
  profileId: string;
  samples: VoiceSample[];
  requirements: VoiceSampleRequirements | null;
  trainingRequirements: TrainingRequirements | null;
  isUploading: boolean;
  onUpload: (file: File) => Promise<boolean>;
  onDelete: (sampleId: string) => Promise<boolean>;
}

export function VoiceSampleUpload({
  profileId,
  samples,
  requirements,
  trainingRequirements,
  isUploading,
  onUpload,
  onDelete,
}: VoiceSampleUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validatedSamples = samples.filter((s) => s.status === 'validated');
  const totalDuration = validatedSamples.reduce((sum, s) => sum + s.duration_seconds, 0);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    setUploadError(null);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      const success = await onUpload(files[0]);
      if (!success) {
        setUploadError('Upload failed. Check the file format and try again.');
      }
    }
  }, [onUpload]);

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    setUploadError(null);
    const file = e.target.files?.[0];
    if (file) {
      const success = await onUpload(file);
      if (!success) {
        setUploadError('Upload failed. Check the file format and try again.');
      }
    }
    e.target.value = '';
  }, [onUpload]);

  const handleDeleteSample = useCallback(async (sampleId: string) => {
    if (window.confirm('Delete this sample?')) {
      await onDelete(sampleId);
    }
  }, [onDelete]);

  const progressPercent = trainingRequirements?.progress_percent || 0;
  const canTrain = trainingRequirements?.can_train || false;

  return (
    <div className="space-y-6">
      {/* Progress indicator */}
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-400">Recording Progress</span>
          <span className="text-sm text-white">
            {formatDuration(totalDuration)} / {formatDuration(requirements?.ideal_duration_seconds || 60)}
          </span>
        </div>

        <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-300 ${
              canTrain ? 'bg-green-500' : 'bg-blue-500'
            }`}
            style={{ width: `${Math.min(100, progressPercent)}%` }}
          />
        </div>

        <div className="flex items-center justify-between mt-2">
          <span className="text-xs text-gray-500">
            {validatedSamples.length} sample(s) validated
          </span>
          {canTrain ? (
            <span className="text-xs text-green-400">Ready for training!</span>
          ) : (
            <span className="text-xs text-gray-500">
              Need {(requirements?.min_duration_seconds || 30) - totalDuration}s more
            </span>
          )}
        </div>
      </div>

      {/* Upload area */}
      <div
        className={`
          border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer
          ${isDragging ? 'border-blue-500 bg-blue-500/10' : 'border-gray-600 hover:border-gray-500'}
          ${isUploading ? 'opacity-50 pointer-events-none' : ''}
        `}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={requirements?.supported_formats.map((f) => `.${f}`).join(',') || 'audio/*'}
          onChange={handleFileSelect}
          className="hidden"
        />

        <div className="flex flex-col items-center">
          {isUploading ? (
            <>
              <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4" />
              <p className="text-gray-400">Uploading and analyzing...</p>
            </>
          ) : (
            <>
              <svg
                className="w-12 h-12 text-gray-500 mb-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                />
              </svg>
              <p className="text-white font-medium mb-1">
                Drag and drop audio file here
              </p>
              <p className="text-gray-400 text-sm">
                or click to browse
              </p>
              <p className="text-gray-500 text-xs mt-2">
                {requirements?.supported_formats.join(', ').toUpperCase()} - Max {requirements?.max_file_size_mb}MB
              </p>
            </>
          )}
        </div>
      </div>

      {/* Upload error */}
      {uploadError && (
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-3">
          <p className="text-sm text-red-300">{uploadError}</p>
        </div>
      )}

      {/* Tips */}
      {requirements?.tips && requirements.tips.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h4 className="text-sm font-medium text-white mb-2">Recording Tips</h4>
          <ul className="space-y-1">
            {requirements.tips.slice(0, 4).map((tip, i) => (
              <li key={i} className="text-xs text-gray-400 flex items-start gap-2">
                <span className="text-green-500 mt-0.5">-</span>
                {tip}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Samples list */}
      {samples.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-white">Uploaded Samples</h4>

          {samples.map((sample) => (
            <div
              key={sample.id}
              className={`
                bg-gray-800 rounded-lg p-4 flex items-center justify-between
                ${sample.status === 'rejected' ? 'border border-red-700' : ''}
              `}
            >
              <div className="flex items-center gap-4">
                {/* Audio icon */}
                <div className={`
                  w-10 h-10 rounded-full flex items-center justify-center
                  ${sample.status === 'validated' ? 'bg-green-500/20' : 'bg-gray-700'}
                `}>
                  <svg className="w-5 h-5 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                  </svg>
                </div>

                {/* Info */}
                <div>
                  <p className="text-sm text-white truncate max-w-[200px]">
                    {sample.filename}
                  </p>
                  <div className="flex items-center gap-3 text-xs text-gray-400">
                    <span>{formatDuration(sample.duration_seconds)}</span>
                    <span className={`text-${getSampleStatusColor(sample.status)}-400`}>
                      {getSampleStatusLabel(sample.status)}
                    </span>
                    {sample.quality_score !== undefined && sample.status === 'validated' && (
                      <span>Quality: {getQualityLabel(sample.quality_score)}</span>
                    )}
                  </div>
                  {sample.rejection_reason && (
                    <p className="text-xs text-red-400 mt-1">{sample.rejection_reason}</p>
                  )}
                </div>
              </div>

              {/* Delete button */}
              <button
                onClick={() => handleDeleteSample(sample.id)}
                className="p-2 text-gray-400 hover:text-red-400 hover:bg-gray-700 rounded"
                title="Delete sample"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
