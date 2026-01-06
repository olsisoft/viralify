'use client';
/* eslint-disable @next/next/no-img-element */

import { useState } from 'react';
import Image from 'next/image';
import { motion } from 'framer-motion';
import {
  Video, Search, Play, Plus, Trash2, Music, Mic,
  Loader2, Sparkles, Info, Download, Clock, Volume2,
  Image as ImageIcon, ChevronDown, ChevronUp, Zap, Layers
} from 'lucide-react';
import toast from 'react-hot-toast';
import { api } from '@/lib/api';

interface StockVideo {
  id: string;
  url: string;
  preview_url: string;
  duration: number;
  width: number;
  height: number;
  user: string;
}

interface TimelineItem {
  id: string;
  type: 'video' | 'image' | 'voiceover';
  content: StockVideo | { url: string; duration: number };
  startTime: number;
  duration: number;
}


const videoCategories = [
  { id: 'business', name: 'Business' },
  { id: 'nature', name: 'Nature' },
  { id: 'technology', name: 'Technology' },
  { id: 'lifestyle', name: 'Lifestyle' },
  { id: 'food', name: 'Food' },
  { id: 'fitness', name: 'Fitness' },
];

export default function VideoComposerPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [stockVideos, setStockVideos] = useState<StockVideo[]>([]);
  const [timeline, setTimeline] = useState<TimelineItem[]>([]);
  const [isExporting, setIsExporting] = useState(false);
  const [showVoiceoverPanel, setShowVoiceoverPanel] = useState(false);
  const [previewIndex, setPreviewIndex] = useState<number | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const totalDuration = timeline.reduce((acc, item) => acc + item.duration, 0);

  const handleSearch = async () => {
    const query = searchQuery.trim() || selectedCategory || 'trending';

    setIsSearching(true);

    try {
      const response = await api.media.searchStockVideos(query, {
        orientation: 'portrait',
        per_page: 12,
      });

      // API returns array directly
      const videos: StockVideo[] = (Array.isArray(response) ? response : []).map((v: any) => ({
        id: String(v.id),
        url: v.url,
        preview_url: v.preview_url,
        duration: v.duration,
        width: v.width,
        height: v.height,
        user: v.user,
      }));

      setStockVideos(videos);
      toast.success(`Found ${videos.length} videos from Pexels`);
    } catch (error) {
      console.error('Search error:', error);
      toast.error('Search failed. Check if media-generator is running.');
    } finally {
      setIsSearching(false);
    }
  };

  const addToTimeline = (video: StockVideo) => {
    const newItem: TimelineItem = {
      id: `tl-${Date.now()}`,
      type: 'video',
      content: video,
      startTime: totalDuration,
      duration: video.duration,
    };
    setTimeline([...timeline, newItem]);
    toast.success('Added to timeline');
  };

  const removeFromTimeline = (id: string) => {
    setTimeline(timeline.filter(item => item.id !== id));
  };

  const moveItem = (index: number, direction: 'up' | 'down') => {
    const newTimeline = [...timeline];
    const newIndex = direction === 'up' ? index - 1 : index + 1;
    if (newIndex < 0 || newIndex >= timeline.length) return;
    [newTimeline[index], newTimeline[newIndex]] = [newTimeline[newIndex], newTimeline[index]];
    setTimeline(newTimeline);
  };

  const handleExport = async () => {
    if (timeline.length === 0) {
      toast.error('Add some clips to the timeline first');
      return;
    }

    setIsExporting(true);

    try {
      // Prepare video URLs for composition
      const videoUrls = timeline
        .filter(item => item.type === 'video')
        .map(item => (item.content as StockVideo).url);

      const response = await api.media.composeVideo({
        video_urls: videoUrls,
        output_format: '9:16',
        quality: '1080p',
      });

      // Poll for job completion
      const jobId = response.job_id;
      const pollJob = async (): Promise<any> => {
        const status = await api.media.getJobStatus(jobId);
        if (status.status === 'completed') return status;
        else if (status.status === 'failed') throw new Error(status.error_message || 'Export failed');
        await new Promise(resolve => setTimeout(resolve, 2000));
        return pollJob();
      };

      toast.success('Video export started...');
      const result = await pollJob();

      if (result.output_data?.url) {
        toast.success('Video exported successfully!');
        // Open in new tab for download
        window.open(result.output_data.url, '_blank');
      }
    } catch (error) {
      console.error('Export error:', error);
      toast.error('Export failed. Try again later.');
    } finally {
      setIsExporting(false);
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="min-h-screen bg-gray-900 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-gradient-to-r from-orange-500 to-red-500 rounded-xl">
              <Video className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-white">Video Composer</h1>
          </div>
          <p className="text-gray-400">
            Combine stock footage, images, and audio to create viral videos
          </p>
        </div>

        {/* Connected Status Banner */}
        <div className="mb-6 p-3 bg-green-500/20 border border-green-500/30 rounded-xl flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-green-400 flex-shrink-0" />
          <p className="text-sm text-green-200">
            <strong>Connected to Pexels API</strong> - Search millions of free stock videos
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Stock Video Search Panel */}
          <div className="lg:col-span-2 space-y-6">
            {/* Search */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Search className="w-5 h-5 text-gray-400" />
                Stock Footage Search
              </h2>

              <div className="flex gap-3 mb-4">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder="Search for stock videos..."
                  className="flex-1 px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-orange-500"
                />
                <button
                  onClick={handleSearch}
                  disabled={isSearching}
                  className="px-6 py-3 bg-orange-600 text-white rounded-xl hover:bg-orange-700 transition disabled:opacity-50 flex items-center gap-2"
                >
                  {isSearching ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Search className="w-5 h-5" />
                  )}
                  Search
                </button>
              </div>

              {/* Categories */}
              <div className="flex flex-wrap gap-2 mb-6">
                <button
                  onClick={() => { setSelectedCategory(''); handleSearch(); }}
                  className={`px-3 py-1.5 rounded-lg text-sm transition ${
                    !selectedCategory
                      ? 'bg-orange-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  All
                </button>
                {videoCategories.map((cat) => (
                  <button
                    key={cat.id}
                    onClick={() => { setSelectedCategory(cat.id); }}
                    className={`px-3 py-1.5 rounded-lg text-sm transition ${
                      selectedCategory === cat.id
                        ? 'bg-orange-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    {cat.name}
                  </button>
                ))}
              </div>

              {/* Results Grid */}
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {stockVideos.length === 0 ? (
                  <div className="col-span-full py-12 text-center">
                    <Search className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                    <p className="text-gray-400">Search for videos to get started</p>
                    <p className="text-gray-500 text-sm">Try &quot;nature&quot;, &quot;business&quot;, or &quot;technology&quot;</p>
                  </div>
                ) : (
                  stockVideos.map((video) => (
                    <motion.div
                      key={video.id}
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="group relative bg-gray-700 rounded-xl overflow-hidden cursor-pointer"
                    >
                      {/* Thumbnail image from Pexels */}
                      <img
                        src={video.preview_url}
                        alt={`Video by ${video.user}`}
                        className="w-full aspect-video object-cover"
                      />
                      {/* Play icon overlay */}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="p-2 bg-black/40 rounded-full">
                          <Play className="w-6 h-6 text-white" />
                        </div>
                      </div>
                      {/* Hover actions */}
                      <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition flex items-center justify-center gap-2">
                        <a
                          href={video.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="p-2 bg-blue-600 rounded-full hover:bg-blue-700 transition"
                          title="Preview video"
                        >
                          <Play className="w-5 h-5 text-white" />
                        </a>
                        <button
                          onClick={() => addToTimeline(video)}
                          className="p-2 bg-orange-600 rounded-full hover:bg-orange-700 transition"
                          title="Add to timeline"
                        >
                          <Plus className="w-5 h-5 text-white" />
                        </button>
                      </div>
                      <div className="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-black/80">
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-gray-400">{formatDuration(video.duration)}</span>
                          <span className="text-xs text-gray-400">by {video.user}</span>
                        </div>
                      </div>
                    </motion.div>
                  ))
                )}
              </div>
            </div>

            {/* Timeline */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                  <Layers className="w-5 h-5 text-gray-400" />
                  Timeline
                </h2>
                <div className="flex items-center gap-4">
                  {timeline.length > 0 && (
                    <button
                      onClick={() => {
                        setPreviewIndex(0);
                        setIsPlaying(true);
                      }}
                      className="px-3 py-1.5 bg-orange-600 text-white text-sm rounded-lg hover:bg-orange-700 transition flex items-center gap-1"
                    >
                      <Play className="w-4 h-4" />
                      Play All
                    </button>
                  )}
                  <span className="text-sm text-gray-400 flex items-center gap-1">
                    <Clock className="w-4 h-4" />
                    {formatDuration(totalDuration)}
                  </span>
                  <span className="text-sm text-gray-400">{timeline.length} clips</span>
                </div>
              </div>

              {timeline.length > 0 ? (
                <div className="space-y-2">
                  {timeline.map((item, index) => (
                    <motion.div
                      key={item.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="flex items-center gap-3 p-3 bg-gray-700/50 rounded-xl"
                    >
                      {/* Thumbnail with play button */}
                      <div className="w-20 h-12 bg-gray-600 rounded-lg overflow-hidden flex-shrink-0 relative group/thumb">
                        {item.type === 'video' && (
                          <>
                            <img
                              src={(item.content as StockVideo).preview_url}
                              alt="Video thumbnail"
                              className="w-full h-full object-cover"
                            />
                            <button
                              onClick={() => {
                                setPreviewIndex(index);
                                setIsPlaying(true);
                              }}
                              className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 group-hover/thumb:opacity-100 transition"
                            >
                              <Play className="w-4 h-4 text-white" />
                            </button>
                          </>
                        )}
                        {item.type === 'voiceover' && (
                          <div className="w-full h-full flex items-center justify-center bg-blue-600/20">
                            <Mic className="w-5 h-5 text-blue-400" />
                          </div>
                        )}
                      </div>

                      {/* Info */}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-white truncate">
                          {item.type === 'video' ? `Video by ${(item.content as StockVideo).user}` : 'Voiceover'}
                        </p>
                        <p className="text-xs text-gray-400">{formatDuration(item.duration)}</p>
                      </div>

                      {/* Actions */}
                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => moveItem(index, 'up')}
                          disabled={index === 0}
                          className="p-1.5 hover:bg-gray-600 rounded transition disabled:opacity-30"
                        >
                          <ChevronUp className="w-4 h-4 text-gray-400" />
                        </button>
                        <button
                          onClick={() => moveItem(index, 'down')}
                          disabled={index === timeline.length - 1}
                          className="p-1.5 hover:bg-gray-600 rounded transition disabled:opacity-30"
                        >
                          <ChevronDown className="w-4 h-4 text-gray-400" />
                        </button>
                        <button
                          onClick={() => removeFromTimeline(item.id)}
                          className="p-1.5 hover:bg-red-600/20 rounded transition"
                        >
                          <Trash2 className="w-4 h-4 text-red-400" />
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              ) : (
                <div className="py-12 text-center">
                  <Video className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                  <p className="text-gray-400">No clips yet</p>
                  <p className="text-gray-500 text-sm">Search and add stock videos above</p>
                </div>
              )}
            </div>
          </div>

          {/* Right Panel - Tools & Export */}
          <div className="space-y-6">
            {/* Quick Add */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4">Quick Add</h3>
              <div className="space-y-3">
                <button
                  onClick={() => setShowVoiceoverPanel(!showVoiceoverPanel)}
                  className="w-full p-3 bg-gray-700 rounded-xl hover:bg-gray-600 transition flex items-center gap-3"
                >
                  <div className="p-2 bg-blue-600/20 rounded-lg">
                    <Mic className="w-5 h-5 text-blue-400" />
                  </div>
                  <div className="text-left">
                    <p className="text-white text-sm font-medium">Add Voiceover</p>
                    <p className="text-gray-400 text-xs">Import from Voiceover Creator</p>
                  </div>
                </button>

                <button className="w-full p-3 bg-gray-700 rounded-xl hover:bg-gray-600 transition flex items-center gap-3">
                  <div className="p-2 bg-purple-600/20 rounded-lg">
                    <Music className="w-5 h-5 text-purple-400" />
                  </div>
                  <div className="text-left">
                    <p className="text-white text-sm font-medium">Add Music</p>
                    <p className="text-gray-400 text-xs">Background audio tracks</p>
                  </div>
                </button>

                <button className="w-full p-3 bg-gray-700 rounded-xl hover:bg-gray-600 transition flex items-center gap-3">
                  <div className="p-2 bg-pink-600/20 rounded-lg">
                    <ImageIcon className="w-5 h-5 text-pink-400" />
                  </div>
                  <div className="text-left">
                    <p className="text-white text-sm font-medium">Add Image</p>
                    <p className="text-gray-400 text-xs">From Image Generator</p>
                  </div>
                </button>
              </div>
            </div>

            {/* Export Settings */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4">Export Settings</h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Format</label>
                  <select className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-orange-500">
                    <option value="9:16">9:16 (TikTok/Reels)</option>
                    <option value="16:9">16:9 (YouTube)</option>
                    <option value="1:1">1:1 (Square)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-2">Quality</label>
                  <select className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-orange-500">
                    <option value="1080p">1080p (HD)</option>
                    <option value="720p">720p</option>
                    <option value="4k">4K (Premium)</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Export Button */}
            <button
              onClick={handleExport}
              disabled={isExporting || timeline.length === 0}
              className="w-full py-4 bg-gradient-to-r from-orange-600 to-red-600 text-white font-semibold rounded-xl hover:from-orange-700 hover:to-red-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isExporting ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Exporting...
                </>
              ) : (
                <>
                  <Download className="w-5 h-5" />
                  Export Video (5 credits)
                </>
              )}
            </button>

            {/* Credits Info */}
            <div className="p-4 bg-gray-700/50 rounded-xl">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Estimated cost:</span>
                <span className="text-white flex items-center gap-1">
                  <Zap className="w-4 h-4 text-yellow-400" />
                  5 credits
                </span>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Based on {timeline.length} clips, {formatDuration(totalDuration)} duration
              </p>
            </div>
          </div>
        </div>

        {/* Video Preview Modal */}
        {previewIndex !== null && timeline[previewIndex] && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80">
            <div className="relative w-full max-w-2xl mx-4">
              {/* Close button */}
              <button
                onClick={() => {
                  setPreviewIndex(null);
                  setIsPlaying(false);
                }}
                className="absolute -top-12 right-0 p-2 text-white hover:text-gray-300 transition"
              >
                <span className="text-lg">✕ Close</span>
              </button>

              {/* Video player */}
              <div className="bg-gray-900 rounded-2xl overflow-hidden">
                <video
                  src={(timeline[previewIndex].content as StockVideo).url}
                  className="w-full aspect-video"
                  controls
                  autoPlay
                  onEnded={() => {
                    // Auto-play next clip if available
                    if (previewIndex < timeline.length - 1) {
                      setPreviewIndex(previewIndex + 1);
                    } else {
                      setIsPlaying(false);
                    }
                  }}
                />

                {/* Timeline navigation */}
                <div className="p-4 bg-gray-800">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-white font-medium">
                      Clip {previewIndex + 1} of {timeline.length}
                    </span>
                    <span className="text-gray-400 text-sm">
                      {(timeline[previewIndex].content as StockVideo).user}
                    </span>
                  </div>

                  {/* Clip thumbnails */}
                  <div className="flex gap-2 overflow-x-auto pb-2">
                    {timeline.map((item, idx) => (
                      <button
                        key={item.id}
                        onClick={() => setPreviewIndex(idx)}
                        className={`flex-shrink-0 w-16 h-10 rounded-lg overflow-hidden border-2 transition ${
                          idx === previewIndex ? 'border-orange-500' : 'border-transparent opacity-60 hover:opacity-100'
                        }`}
                      >
                        {item.type === 'video' && (
                          <img
                            src={(item.content as StockVideo).preview_url}
                            alt={`Clip ${idx + 1}`}
                            className="w-full h-full object-cover"
                          />
                        )}
                      </button>
                    ))}
                  </div>

                  {/* Navigation buttons */}
                  <div className="flex justify-center gap-4 mt-3">
                    <button
                      onClick={() => setPreviewIndex(Math.max(0, previewIndex - 1))}
                      disabled={previewIndex === 0}
                      className="px-4 py-2 bg-gray-700 text-white rounded-lg disabled:opacity-50 hover:bg-gray-600 transition"
                    >
                      ← Previous
                    </button>
                    <button
                      onClick={() => setPreviewIndex(Math.min(timeline.length - 1, previewIndex + 1))}
                      disabled={previewIndex === timeline.length - 1}
                      className="px-4 py-2 bg-gray-700 text-white rounded-lg disabled:opacity-50 hover:bg-gray-600 transition"
                    >
                      Next →
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
