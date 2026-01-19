'use client';

/**
 * Video Editor Page
 * Phase 3: User Video Editing feature
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { useVideoEditor } from './hooks/useVideoEditor';
import { Timeline } from './components/Timeline';
import { SegmentProperties } from './components/SegmentProperties';
import {
  VideoProject,
  VideoSegment,
  formatDetailedDuration,
  CreateProjectRequest,
} from './lib/editor-types';

export default function VideoEditorPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const projectId = searchParams.get('projectId');
  const courseId = searchParams.get('courseId');
  const courseJobId = searchParams.get('courseJobId');

  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null);
  const [playheadPosition, setPlayheadPosition] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showNewProjectModal, setShowNewProjectModal] = useState(!projectId);
  const [newProjectTitle, setNewProjectTitle] = useState('');

  const fileInputRef = useRef<HTMLInputElement>(null);
  const playIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const {
    project,
    isLoading,
    isSaving,
    isRendering,
    renderJob,
    error,
    supportedFormats,
    createProject,
    loadProject,
    updateProjectSettings,
    uploadSegment,
    updateSegment,
    removeSegment,
    reorderSegments,
    splitSegment,
    startRender,
    checkRenderStatus,
    clearError,
  } = useVideoEditor({
    onError: (err) => console.error('Editor error:', err),
  });

  // Load project on mount
  useEffect(() => {
    if (projectId) {
      loadProject(projectId);
    }
  }, [projectId, loadProject]);

  // Poll render status
  useEffect(() => {
    if (renderJob && renderJob.status === 'processing') {
      const interval = setInterval(() => {
        checkRenderStatus(renderJob.job_id);
      }, 2000);

      return () => clearInterval(interval);
    }
  }, [renderJob, checkRenderStatus]);

  // Playback simulation
  useEffect(() => {
    if (isPlaying && project) {
      playIntervalRef.current = setInterval(() => {
        setPlayheadPosition((pos) => {
          const newPos = pos + 0.1;
          if (newPos >= project.total_duration) {
            setIsPlaying(false);
            return 0;
          }
          return newPos;
        });
      }, 100);
    }

    return () => {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
      }
    };
  }, [isPlaying, project]);

  // Handlers
  const handleCreateProject = useCallback(async () => {
    if (!newProjectTitle.trim()) return;

    const request: CreateProjectRequest = {
      user_id: 'demo-user',
      title: newProjectTitle.trim(),
      course_id: courseId || undefined,
      course_job_id: courseJobId || undefined,
      import_course_videos: !!courseJobId,
    };

    const newProjectId = await createProject(request);
    if (newProjectId) {
      setShowNewProjectModal(false);
      router.push(`/dashboard/studio/editor?projectId=${newProjectId}`);
    }
  }, [newProjectTitle, courseId, courseJobId, createProject, router]);

  const handleFileUpload = useCallback(async (file: File, insertAfterSegmentId?: string) => {
    await uploadSegment(file, insertAfterSegmentId);
  }, [uploadSegment]);

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
    e.target.value = '';
  }, [handleFileUpload]);

  const handleUpdateSegment = useCallback(async (segmentId: string, updates: Partial<VideoSegment>) => {
    await updateSegment(segmentId, updates);
  }, [updateSegment]);

  const handleRemoveSegment = useCallback(async (segmentId: string) => {
    await removeSegment(segmentId);
    if (selectedSegmentId === segmentId) {
      setSelectedSegmentId(null);
    }
  }, [removeSegment, selectedSegmentId]);

  const handleSplitSegment = useCallback(async (splitTime: number) => {
    if (selectedSegmentId) {
      await splitSegment(selectedSegmentId, splitTime);
    }
  }, [selectedSegmentId, splitSegment]);

  const handleStartRender = useCallback(async () => {
    await startRender({});
  }, [startRender]);

  const handlePlayPause = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  const handleStop = useCallback(() => {
    setIsPlaying(false);
    setPlayheadPosition(0);
  }, []);

  const selectedSegment = project?.segments.find((s) => s.id === selectedSegmentId);

  // New project modal
  if (showNewProjectModal) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
        <div className="bg-gray-800 rounded-lg shadow-xl w-full max-w-md p-6">
          <h2 className="text-xl font-bold text-white mb-4">Create New Project</h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">Project Title</label>
              <input
                type="text"
                value={newProjectTitle}
                onChange={(e) => setNewProjectTitle(e.target.value)}
                placeholder="My Video Project"
                className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400"
                autoFocus
              />
            </div>

            {courseJobId && (
              <div className="bg-blue-900/30 border border-blue-700 rounded-lg p-3">
                <p className="text-sm text-blue-300">
                  This project will import videos from your course generation.
                </p>
              </div>
            )}

            <div className="flex gap-3 pt-4">
              <button
                onClick={() => router.push('/dashboard/studio')}
                className="flex-1 px-4 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-medium"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateProject}
                disabled={!newProjectTitle.trim() || isLoading}
                className="flex-1 px-4 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium"
              >
                {isLoading ? 'Creating...' : 'Create Project'}
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Loading state
  if (isLoading && !project) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4" />
          <p className="text-gray-400">Loading project...</p>
        </div>
      </div>
    );
  }

  // No project
  if (!project) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-400 mb-4">Project not found</p>
          <button
            onClick={() => setShowNewProjectModal(true)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg"
          >
            Create New Project
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => router.push('/dashboard/studio')}
              className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <div>
              <h1 className="text-lg font-semibold text-white">{project.title}</h1>
              <p className="text-xs text-gray-400">
                {project.segments.length} segments - {formatDetailedDuration(project.total_duration)}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Save indicator */}
            {isSaving && (
              <span className="text-xs text-gray-400 flex items-center gap-1">
                <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
                Saving...
              </span>
            )}

            {/* Render button */}
            <button
              onClick={handleStartRender}
              disabled={isRendering || project.segments.length === 0}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg text-sm font-medium flex items-center gap-2"
            >
              {isRendering ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Rendering {renderJob?.progress || 0}%
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                  </svg>
                  Export Video
                </>
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Error banner */}
      {error && (
        <div className="bg-red-900/50 border-b border-red-700 px-4 py-2 flex items-center justify-between">
          <span className="text-sm text-red-300">{error}</span>
          <button onClick={clearError} className="text-red-300 hover:text-white">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      )}

      {/* Render complete banner */}
      {renderJob?.status === 'completed' && renderJob.output_url && (
        <div className="bg-green-900/50 border-b border-green-700 px-4 py-2 flex items-center justify-between">
          <span className="text-sm text-green-300">
            Video exported successfully!
          </span>
          <a
            href={renderJob.output_url}
            download
            className="text-sm text-green-300 hover:text-white underline"
          >
            Download Video
          </a>
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 flex">
        {/* Left panel: Preview & Controls */}
        <div className="w-1/3 min-w-[300px] max-w-[500px] border-r border-gray-700 flex flex-col">
          {/* Preview area */}
          <div className="flex-1 bg-black flex items-center justify-center p-4">
            <div className="aspect-video w-full max-h-full bg-gray-900 rounded-lg flex items-center justify-center">
              {selectedSegment?.thumbnail_url ? (
                <img
                  src={selectedSegment.thumbnail_url}
                  alt="Preview"
                  className="max-w-full max-h-full object-contain rounded"
                />
              ) : (
                <span className="text-gray-500">Select a segment to preview</span>
              )}
            </div>
          </div>

          {/* Playback controls */}
          <div className="bg-gray-800 border-t border-gray-700 px-4 py-3">
            <div className="flex items-center justify-center gap-4">
              <button
                onClick={handleStop}
                className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                title="Stop"
              >
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                  <rect x="6" y="6" width="12" height="12" />
                </svg>
              </button>

              <button
                onClick={handlePlayPause}
                className="p-3 bg-blue-600 hover:bg-blue-700 text-white rounded-full"
                title={isPlaying ? 'Pause' : 'Play'}
              >
                {isPlaying ? (
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <rect x="6" y="5" width="4" height="14" />
                    <rect x="14" y="5" width="4" height="14" />
                  </svg>
                ) : (
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z" />
                  </svg>
                )}
              </button>

              <span className="text-sm text-gray-400 min-w-[80px] text-center">
                {formatDetailedDuration(playheadPosition)} / {formatDetailedDuration(project.total_duration)}
              </span>
            </div>
          </div>

          {/* Upload button */}
          <div className="px-4 py-3 border-t border-gray-700">
            <input
              ref={fileInputRef}
              type="file"
              accept={supportedFormats ? [
                ...supportedFormats.video,
                ...supportedFormats.audio,
                ...supportedFormats.image
              ].join(',') : '*'}
              onChange={handleFileInputChange}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="w-full px-4 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm font-medium flex items-center justify-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Upload Media
            </button>
            {supportedFormats && (
              <p className="text-xs text-gray-500 mt-2 text-center">
                Supports: Video ({supportedFormats.max_sizes.video_mb}MB), Audio ({supportedFormats.max_sizes.audio_mb}MB), Images ({supportedFormats.max_sizes.image_mb}MB)
              </p>
            )}
          </div>
        </div>

        {/* Right panel: Timeline & Properties */}
        <div className="flex-1 flex flex-col">
          {/* Timeline */}
          <div className="flex-1 min-h-[200px]">
            <Timeline
              project={project}
              selectedSegmentId={selectedSegmentId}
              playheadPosition={playheadPosition}
              onSelectSegment={setSelectedSegmentId}
              onUpdateSegment={handleUpdateSegment}
              onRemoveSegment={handleRemoveSegment}
              onReorderSegments={reorderSegments}
              onPlayheadChange={setPlayheadPosition}
              onUploadFile={handleFileUpload}
            />
          </div>

          {/* Segment properties panel */}
          {selectedSegment && (
            <div className="h-80 border-t border-gray-700">
              <SegmentProperties
                segment={selectedSegment}
                onUpdate={(request) => updateSegment(selectedSegment.id, request)}
                onRemove={() => handleRemoveSegment(selectedSegment.id)}
                onSplit={handleSplitSegment}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
