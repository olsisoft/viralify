'use client';
/* eslint-disable @next/next/no-img-element */

import { useState, useEffect } from 'react';
import Image from 'next/image';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import {
  Calendar, Clock, Video, Plus, Trash2, Check,
  X, ChevronLeft, ChevronRight, Eye, Loader2,
  Instagram, Youtube, Info, ExternalLink
} from 'lucide-react';
import { format, addDays, startOfWeek, addWeeks, isSameDay, parseISO, isToday, isPast } from 'date-fns';
import toast from 'react-hot-toast';
import { DEMO_MODE, DEMO_SCHEDULED_POSTS, DEMO_PLATFORM_ACCOUNTS } from '@/lib/demo-mode';
import { api, ScheduledPost } from '@/lib/api';
import { DashboardLayout } from '@/components/layout/dashboard-layout';

type PlatformType = 'TIKTOK' | 'INSTAGRAM' | 'YOUTUBE';

interface PlatformAccount {
  id: string;
  platform: PlatformType;
  platformUsername: string;
  platformDisplayName?: string;
  followerCount: number;
  accountStatus: string;
}

export default function SchedulerPage() {
  const [posts, setPosts] = useState<ScheduledPost[]>([]);
  const [currentWeekStart, setCurrentWeekStart] = useState(startOfWeek(new Date(), { weekStartsOn: 1 }));
  const [selectedPost, setSelectedPost] = useState<ScheduledPost | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [connectedAccounts, setConnectedAccounts] = useState<PlatformAccount[]>([]);
  const [viewMode, setViewMode] = useState<'calendar' | 'list'>('calendar');

  const weekDays = Array.from({ length: 7 }, (_, i) => addDays(currentWeekStart, i));

  useEffect(() => {
    fetchPosts();
    fetchConnectedAccounts();
  }, []);

  const fetchConnectedAccounts = async () => {
    try {
      const accounts = await api.platforms.getConnectedAccounts();
      setConnectedAccounts(accounts as PlatformAccount[]);
    } catch (error) {
      console.error('Failed to fetch accounts:', error);
    }
  };

  const fetchPosts = async () => {
    setIsLoading(true);
    try {
      const data = await api.scheduler.getPosts();
      setPosts(data);
    } catch (error) {
      console.error('Failed to fetch posts:', error);
      setPosts([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancelPost = async (postId: string) => {
    try {
      await api.scheduler.cancelPost(postId);
      setPosts(posts.filter(p => p.id !== postId));

      // Also remove from DEMO_SCHEDULED_POSTS if in demo mode
      if (DEMO_MODE) {
        const index = DEMO_SCHEDULED_POSTS.findIndex(p => p.id === postId);
        if (index !== -1) {
          DEMO_SCHEDULED_POSTS.splice(index, 1);
        }
      }

      toast.success('Post cancelled');
      setSelectedPost(null);
    } catch (error) {
      toast.error('Failed to cancel post');
    }
  };

  const getPostsForDate = (date: Date) => {
    return posts.filter(post => {
      try {
        return isSameDay(parseISO(post.scheduledAt), date);
      } catch {
        return false;
      }
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'processing': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      case 'published': return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'failed': return 'bg-red-500/20 text-red-400 border-red-500/30';
      case 'cancelled': return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
      default: return 'bg-gray-500/20 text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending': return <Clock className="w-4 h-4" />;
      case 'processing': return <Loader2 className="w-4 h-4 animate-spin" />;
      case 'published': return <Check className="w-4 h-4" />;
      case 'failed': return <X className="w-4 h-4" />;
      default: return <Clock className="w-4 h-4" />;
    }
  };

  const getPlatformIcon = (platform: string) => {
    switch (platform) {
      case 'TIKTOK':
        return (
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
            <path d="M19.59 6.69a4.83 4.83 0 0 1-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 0 1-5.2 1.74 2.89 2.89 0 0 1 2.31-4.64 2.93 2.93 0 0 1 .88.13V9.4a6.84 6.84 0 0 0-1-.05A6.33 6.33 0 0 0 5 20.1a6.34 6.34 0 0 0 10.86-4.43v-7a8.16 8.16 0 0 0 4.77 1.52v-3.4a4.85 4.85 0 0 1-1-.1z"/>
          </svg>
        );
      case 'INSTAGRAM':
        return <Instagram className="w-4 h-4 text-pink-400" />;
      case 'YOUTUBE':
        return <Youtube className="w-4 h-4 text-red-500" />;
      default:
        return <Video className="w-4 h-4" />;
    }
  };

  const pendingPosts = posts.filter(p => p.status === 'pending' || p.status === 'processing');
  const publishedPosts = posts.filter(p => p.status === 'published');
  const failedPosts = posts.filter(p => p.status === 'failed');

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Demo Mode Banner */}
        {DEMO_MODE && (
          <div className="p-3 bg-yellow-500/20 border border-yellow-500/30 rounded-xl flex items-center gap-2">
            <Info className="w-5 h-5 text-yellow-400 flex-shrink-0" />
            <p className="text-sm text-yellow-200">
              <strong>Demo Mode:</strong> Posts are stored locally. Create posts from the{' '}
              <Link href="/dashboard/create" className="underline">Create page</Link> to see them here.
            </p>
          </div>
        )}

        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center">
              <Calendar className="mr-3 h-8 w-8 text-purple-400" />
              Content Scheduler
            </h1>
            <p className="mt-1 text-gray-400">
              {pendingPosts.length} scheduled • {publishedPosts.length} published
            </p>
          </div>

          <div className="flex items-center gap-3 mt-4 md:mt-0">
            {/* View Toggle */}
            <div className="flex bg-gray-800 rounded-lg p-1">
              <button
                onClick={() => setViewMode('calendar')}
                className={`px-3 py-1.5 rounded-md text-sm transition ${
                  viewMode === 'calendar' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white'
                }`}
              >
                Calendar
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`px-3 py-1.5 rounded-md text-sm transition ${
                  viewMode === 'list' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white'
                }`}
              >
                List
              </button>
            </div>

            <Link
              href="/dashboard/create"
              className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-xl transition flex items-center"
            >
              <Plus className="mr-2 h-5 w-5" />
              Create Post
            </Link>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-yellow-500/20 rounded-lg">
                <Clock className="w-5 h-5 text-yellow-400" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{pendingPosts.length}</p>
                <p className="text-sm text-gray-400">Scheduled</p>
              </div>
            </div>
          </div>
          <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-green-500/20 rounded-lg">
                <Check className="w-5 h-5 text-green-400" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{publishedPosts.length}</p>
                <p className="text-sm text-gray-400">Published</p>
              </div>
            </div>
          </div>
          <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-red-500/20 rounded-lg">
                <X className="w-5 h-5 text-red-400" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{failedPosts.length}</p>
                <p className="text-sm text-gray-400">Failed</p>
              </div>
            </div>
          </div>
          <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-500/20 rounded-lg">
                <Video className="w-5 h-5 text-purple-400" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{posts.length}</p>
                <p className="text-sm text-gray-400">Total</p>
              </div>
            </div>
          </div>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-8 h-8 text-purple-400 animate-spin" />
          </div>
        ) : viewMode === 'calendar' ? (
          <>
            {/* Calendar Navigation */}
            <div className="flex items-center justify-between">
              <button
                onClick={() => setCurrentWeekStart(addWeeks(currentWeekStart, -1))}
                className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition"
              >
                <ChevronLeft className="h-5 w-5 text-gray-400" />
              </button>

              <h2 className="text-lg font-semibold text-white">
                {format(currentWeekStart, 'MMMM d')} - {format(addDays(currentWeekStart, 6), 'MMMM d, yyyy')}
              </h2>

              <button
                onClick={() => setCurrentWeekStart(addWeeks(currentWeekStart, 1))}
                className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition"
              >
                <ChevronRight className="h-5 w-5 text-gray-400" />
              </button>
            </div>

            {/* Calendar Grid */}
            <div className="grid grid-cols-7 gap-3">
              {weekDays.map((day, index) => {
                const dayPosts = getPostsForDate(day);
                const isCurrentDay = isToday(day);
                const isPastDay = isPast(day) && !isCurrentDay;

                return (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className={`bg-gray-800 rounded-xl p-3 border min-h-[180px] ${
                      isCurrentDay ? 'border-purple-500 ring-1 ring-purple-500/50' :
                      isPastDay ? 'border-gray-700 opacity-60' : 'border-gray-700'
                    }`}
                  >
                    <div className={`text-center mb-3 ${isCurrentDay ? 'text-purple-400' : 'text-gray-400'}`}>
                      <div className="text-xs font-medium uppercase">{format(day, 'EEE')}</div>
                      <div className={`text-xl font-bold ${isCurrentDay ? 'text-white' : ''}`}>
                        {format(day, 'd')}
                      </div>
                    </div>

                    <div className="space-y-2">
                      {dayPosts.slice(0, 3).map((post) => (
                        <button
                          key={post.id}
                          onClick={() => setSelectedPost(post)}
                          className={`w-full p-2 rounded-lg border text-left text-xs transition hover:scale-[1.02] ${getStatusColor(post.status)}`}
                        >
                          <div className="font-medium truncate">{post.title}</div>
                          <div className="flex items-center justify-between mt-1">
                            <span className="text-gray-400">
                              {format(parseISO(post.scheduledAt), 'HH:mm')}
                            </span>
                            <div className="flex gap-0.5">
                              {post.targetPlatforms?.slice(0, 2).map((platform) => (
                                <span key={platform} className="opacity-70">
                                  {getPlatformIcon(platform)}
                                </span>
                              ))}
                            </div>
                          </div>
                        </button>
                      ))}

                      {dayPosts.length > 3 && (
                        <div className="text-xs text-gray-500 text-center">
                          +{dayPosts.length - 3} more
                        </div>
                      )}

                      {dayPosts.length === 0 && !isPastDay && (
                        <Link
                          href="/dashboard/create"
                          className="block w-full p-2 border border-dashed border-gray-600 rounded-lg text-gray-500 hover:border-purple-500 hover:text-purple-400 transition text-xs text-center"
                        >
                          + Add
                        </Link>
                      )}
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </>
        ) : (
          /* List View */
          <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
            <div className="divide-y divide-gray-700">
              {posts.length === 0 ? (
                <div className="p-12 text-center">
                  <Calendar className="h-16 w-16 text-gray-600 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-white mb-2">No scheduled posts</h3>
                  <p className="text-gray-400 mb-6">Create your first post to get started</p>
                  <Link
                    href="/dashboard/create"
                    className="inline-flex items-center px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-xl transition"
                  >
                    <Plus className="mr-2 h-5 w-5" />
                    Create Post
                  </Link>
                </div>
              ) : (
                posts.map((post) => (
                  <div
                    key={post.id}
                    className="p-4 flex items-center justify-between hover:bg-gray-700/50 transition cursor-pointer"
                    onClick={() => setSelectedPost(post)}
                  >
                    <div className="flex items-center gap-4">
                      {/* Thumbnail */}
                      <div className="w-16 h-20 bg-gray-700 rounded-lg flex items-center justify-center overflow-hidden">
                        {post.thumbnailUrl ? (
                          <img src={post.thumbnailUrl} alt="" className="w-full h-full object-cover" />
                        ) : (
                          <Video className="h-6 w-6 text-gray-500" />
                        )}
                      </div>

                      {/* Info */}
                      <div>
                        <h4 className="font-medium text-white">{post.title}</h4>
                        {post.caption && (
                          <p className="text-sm text-gray-400 truncate max-w-md">{post.caption}</p>
                        )}
                        <div className="flex items-center gap-4 mt-2 text-sm text-gray-500">
                          <span className="flex items-center gap-1">
                            <Calendar className="h-4 w-4" />
                            {format(parseISO(post.scheduledAt), 'MMM d, yyyy')}
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="h-4 w-4" />
                            {format(parseISO(post.scheduledAt), 'HH:mm')}
                          </span>
                        </div>

                        {/* Platforms */}
                        <div className="flex items-center gap-2 mt-2">
                          {post.targetPlatforms?.map((platform) => {
                            const status = post.platformStatuses?.find(s => s.platform === platform);
                            return (
                              <div
                                key={platform}
                                className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs ${
                                  status?.status === 'published'
                                    ? 'bg-green-500/20 text-green-400'
                                    : status?.status === 'failed'
                                    ? 'bg-red-500/20 text-red-400'
                                    : 'bg-gray-600/50 text-gray-300'
                                }`}
                              >
                                {getPlatformIcon(platform)}
                                <span className="capitalize">{platform.toLowerCase()}</span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </div>

                    {/* Status & Actions */}
                    <div className="flex items-center gap-3">
                      <span className={`px-3 py-1.5 rounded-full text-xs font-medium flex items-center gap-1.5 ${getStatusColor(post.status)}`}>
                        {getStatusIcon(post.status)}
                        {post.status}
                      </span>
                      {post.status === 'pending' && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCancelPost(post.id);
                          }}
                          className="p-2 text-gray-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* Post Detail Modal */}
        <AnimatePresence>
          {selectedPost && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4"
              onClick={() => setSelectedPost(null)}
            >
              <motion.div
                initial={{ scale: 0.95, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.95, opacity: 0 }}
                onClick={(e) => e.stopPropagation()}
                className="bg-gray-800 rounded-2xl w-full max-w-lg overflow-hidden"
              >
                {/* Header */}
                <div className="p-4 border-b border-gray-700 flex items-center justify-between">
                  <h2 className="text-lg font-semibold text-white">Post Details</h2>
                  <button
                    onClick={() => setSelectedPost(null)}
                    className="p-2 text-gray-400 hover:text-white rounded-lg hover:bg-gray-700"
                  >
                    <X className="h-5 w-5" />
                  </button>
                </div>

                {/* Content */}
                <div className="p-6 space-y-4">
                  {/* Thumbnail */}
                  {selectedPost.thumbnailUrl && (
                    <div className="aspect-[9/16] max-h-48 bg-gray-700 rounded-xl overflow-hidden mx-auto w-fit">
                      <img
                        src={selectedPost.thumbnailUrl}
                        alt=""
                        className="h-full w-auto object-cover"
                      />
                    </div>
                  )}

                  {/* Title */}
                  <div>
                    <label className="text-sm text-gray-400">Title</label>
                    <p className="text-white font-medium">{selectedPost.title}</p>
                  </div>

                  {/* Caption */}
                  {selectedPost.caption && (
                    <div>
                      <label className="text-sm text-gray-400">Caption</label>
                      <p className="text-white">{selectedPost.caption}</p>
                    </div>
                  )}

                  {/* Hashtags */}
                  {selectedPost.hashtags && selectedPost.hashtags.length > 0 && (
                    <div>
                      <label className="text-sm text-gray-400">Hashtags</label>
                      <div className="flex flex-wrap gap-2 mt-1">
                        {selectedPost.hashtags.map((tag, i) => (
                          <span key={i} className="px-2 py-1 bg-purple-500/20 text-purple-300 rounded-full text-sm">
                            #{tag.replace('#', '')}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Schedule */}
                  <div className="flex gap-4">
                    <div>
                      <label className="text-sm text-gray-400">Date</label>
                      <p className="text-white">{format(parseISO(selectedPost.scheduledAt), 'MMMM d, yyyy')}</p>
                    </div>
                    <div>
                      <label className="text-sm text-gray-400">Time</label>
                      <p className="text-white">{format(parseISO(selectedPost.scheduledAt), 'HH:mm')}</p>
                    </div>
                  </div>

                  {/* Platforms */}
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Platforms</label>
                    <div className="space-y-2">
                      {selectedPost.targetPlatforms?.map((platform) => {
                        const status = selectedPost.platformStatuses?.find(s => s.platform === platform);
                        return (
                          <div
                            key={platform}
                            className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg"
                          >
                            <div className="flex items-center gap-2">
                              {getPlatformIcon(platform)}
                              <span className="text-white capitalize">{platform.toLowerCase()}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className={`px-2 py-1 rounded text-xs ${getStatusColor(status?.status || 'pending')}`}>
                                {status?.status || 'pending'}
                              </span>
                              {status?.platformShareUrl && (
                                <a
                                  href={status.platformShareUrl}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="p-1 text-gray-400 hover:text-white"
                                >
                                  <ExternalLink className="w-4 h-4" />
                                </a>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Status */}
                  <div className="flex items-center justify-between pt-4 border-t border-gray-700">
                    <span className={`px-4 py-2 rounded-full text-sm font-medium flex items-center gap-2 ${getStatusColor(selectedPost.status)}`}>
                      {getStatusIcon(selectedPost.status)}
                      {selectedPost.status.charAt(0).toUpperCase() + selectedPost.status.slice(1)}
                    </span>

                    {selectedPost.status === 'pending' && (
                      <button
                        onClick={() => handleCancelPost(selectedPost.id)}
                        className="px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition flex items-center gap-2"
                      >
                        <Trash2 className="w-4 h-4" />
                        Cancel Post
                      </button>
                    )}
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </DashboardLayout>
  );
}
