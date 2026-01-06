'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import {
  Wand2, Image as ImageIcon, Video, Mic, FileText, Sparkles,
  ArrowRight, Zap, Clock, Star, TrendingUp, Palette, Brain
} from 'lucide-react';

const studioTools = [
  {
    id: 'ai-video',
    title: 'AI Video Generator',
    description: 'Create complete videos from text prompts with AI',
    icon: Sparkles,
    href: '/dashboard/studio/ai-video',
    gradient: 'from-pink-500 to-rose-500',
    features: ['Text to Video', 'Auto Scenes', 'Voice Narration'],
    credits: 10,
  },
  {
    id: 'image',
    title: 'Image Generator',
    description: 'Create stunning images with DALL-E 3 for thumbnails, posts, and more',
    icon: ImageIcon,
    href: '/dashboard/studio/image',
    gradient: 'from-purple-500 to-pink-500',
    features: ['DALL-E 3 Integration', 'Thumbnail Presets', 'Custom Styles'],
    credits: 1,
  },
  {
    id: 'voiceover',
    title: 'Voiceover Creator',
    description: 'Generate natural-sounding voiceovers with AI voices',
    icon: Mic,
    href: '/dashboard/studio/voiceover',
    gradient: 'from-blue-500 to-cyan-500',
    features: ['ElevenLabs Voices', 'Multiple Languages', 'Emotion Control'],
    credits: 2,
  },
  {
    id: 'video',
    title: 'Video Composer',
    description: 'Combine stock footage, images, and audio into viral videos',
    icon: Video,
    href: '/dashboard/studio/video',
    gradient: 'from-orange-500 to-red-500',
    features: ['Stock Footage', 'Auto-Edit', 'Music Library'],
    credits: 5,
  },
  {
    id: 'articles',
    title: 'Article Writer',
    description: 'Generate SEO-optimized articles and blog posts with AI',
    icon: FileText,
    href: '/dashboard/studio/articles',
    gradient: 'from-green-500 to-emerald-500',
    features: ['SEO Optimized', 'Multiple Formats', 'Fact Checking'],
    credits: 1,
  },
];

const quickActions = [
  { label: 'Generate Thumbnail', icon: Palette, href: '/dashboard/studio/image?preset=thumbnail' },
  { label: 'Quick Voiceover', icon: Mic, href: '/dashboard/studio/voiceover?quick=true' },
  { label: 'Trending Images', icon: TrendingUp, href: '/dashboard/studio/image?preset=trending' },
  { label: 'AI Script to Video', icon: Brain, href: '/dashboard/studio/video?mode=script' },
];

const recentGenerations = [
  { id: '1', type: 'image', title: 'Thumbnail - Morning Routine', time: '2 hours ago', status: 'completed' },
  { id: '2', type: 'voiceover', title: 'Script narration', time: '5 hours ago', status: 'completed' },
  { id: '3', type: 'article', title: 'Top 10 Productivity Tips', time: '1 day ago', status: 'completed' },
];

export default function StudioPage() {
  return (
    <div className="min-h-screen bg-gray-900 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-10">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center justify-center p-3 bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl mb-4"
          >
            <Wand2 className="w-8 h-8 text-white" />
          </motion.div>
          <motion.h1
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-4xl font-bold text-white mb-2"
          >
            Content Studio
          </motion.h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-gray-400 text-lg max-w-2xl mx-auto"
          >
            Create professional content with AI-powered tools. Generate images, videos, voiceovers, and articles in minutes.
          </motion.p>
        </div>

        {/* Credits Banner */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-gradient-to-r from-purple-600/20 to-pink-600/20 border border-purple-500/30 rounded-2xl p-4 mb-8 flex items-center justify-between"
        >
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-600/30 rounded-xl">
              <Zap className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <p className="text-white font-medium">500 Credits Available</p>
              <p className="text-gray-400 text-sm">Pro Plan - Resets in 15 days</p>
            </div>
          </div>
          <Link
            href="/dashboard/settings"
            className="px-4 py-2 bg-purple-600 text-white rounded-xl hover:bg-purple-700 transition text-sm font-medium"
          >
            Upgrade Plan
          </Link>
        </motion.div>

        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="mb-10"
        >
          <h2 className="text-lg font-semibold text-white mb-4">Quick Actions</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {quickActions.map((action, index) => (
              <Link
                key={action.label}
                href={action.href}
                className="group p-4 bg-gray-800 border border-gray-700 rounded-xl hover:border-purple-500 transition-all"
              >
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-gray-700 rounded-lg group-hover:bg-purple-600/20 transition">
                    <action.icon className="w-5 h-5 text-gray-400 group-hover:text-purple-400 transition" />
                  </div>
                  <span className="text-sm text-gray-300 group-hover:text-white transition">
                    {action.label}
                  </span>
                </div>
              </Link>
            ))}
          </div>
        </motion.div>

        {/* Studio Tools Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="grid md:grid-cols-2 gap-6 mb-10"
        >
          {studioTools.map((tool, index) => (
            <motion.div
              key={tool.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 + index * 0.1 }}
            >
              <Link href={tool.href}>
                <div className="group h-full p-6 bg-gray-800 border border-gray-700 rounded-2xl hover:border-purple-500 transition-all cursor-pointer">
                  <div className="flex items-start justify-between mb-4">
                    <div className={`p-3 rounded-xl bg-gradient-to-r ${tool.gradient}`}>
                      <tool.icon className="w-6 h-6 text-white" />
                    </div>
                    <div className="flex items-center gap-1 px-3 py-1 bg-gray-700 rounded-full">
                      <Zap className="w-3 h-3 text-yellow-400" />
                      <span className="text-xs text-gray-300">{tool.credits} credits</span>
                    </div>
                  </div>

                  <h3 className="text-xl font-semibold text-white mb-2 group-hover:text-purple-400 transition">
                    {tool.title}
                  </h3>
                  <p className="text-gray-400 text-sm mb-4">
                    {tool.description}
                  </p>

                  <div className="flex flex-wrap gap-2 mb-4">
                    {tool.features.map((feature) => (
                      <span
                        key={feature}
                        className="px-2 py-1 bg-gray-700 rounded-lg text-xs text-gray-300"
                      >
                        {feature}
                      </span>
                    ))}
                  </div>

                  <div className="flex items-center text-purple-400 text-sm font-medium group-hover:text-purple-300">
                    Start Creating
                    <ArrowRight className="ml-2 w-4 h-4 group-hover:translate-x-1 transition" />
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </motion.div>

        {/* Recent Generations */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9 }}
          className="bg-gray-800 border border-gray-700 rounded-2xl p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <Clock className="w-5 h-5 text-gray-400" />
              Recent Generations
            </h2>
            <Link
              href="/dashboard/studio/history"
              className="text-sm text-purple-400 hover:text-purple-300"
            >
              View All
            </Link>
          </div>

          <div className="space-y-3">
            {recentGenerations.map((item) => (
              <div
                key={item.id}
                className="flex items-center justify-between p-3 bg-gray-700/50 rounded-xl"
              >
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${
                    item.type === 'image' ? 'bg-purple-600/20 text-purple-400' :
                    item.type === 'voiceover' ? 'bg-blue-600/20 text-blue-400' :
                    item.type === 'article' ? 'bg-green-600/20 text-green-400' :
                    'bg-orange-600/20 text-orange-400'
                  }`}>
                    {item.type === 'image' && <ImageIcon className="w-4 h-4" />}
                    {item.type === 'voiceover' && <Mic className="w-4 h-4" />}
                    {item.type === 'article' && <FileText className="w-4 h-4" />}
                    {item.type === 'video' && <Video className="w-4 h-4" />}
                  </div>
                  <div>
                    <p className="text-white text-sm font-medium">{item.title}</p>
                    <p className="text-gray-400 text-xs">{item.time}</p>
                  </div>
                </div>
                <span className={`px-2 py-1 rounded-full text-xs ${
                  item.status === 'completed' ? 'bg-green-600/20 text-green-400' :
                  item.status === 'processing' ? 'bg-yellow-600/20 text-yellow-400' :
                  'bg-red-600/20 text-red-400'
                }`}>
                  {item.status}
                </span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Pro Tips */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1 }}
          className="mt-8 p-4 bg-gradient-to-r from-blue-600/10 to-purple-600/10 border border-blue-500/20 rounded-xl"
        >
          <div className="flex items-start gap-3">
            <Star className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-white font-medium mb-1">Pro Tip</p>
              <p className="text-gray-400 text-sm">
                Combine AI-generated images with voiceovers to create engaging video content in minutes.
                Start with an image, add a voiceover, and let the video composer do the rest!
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
