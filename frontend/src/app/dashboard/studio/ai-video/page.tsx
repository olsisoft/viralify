'use client';
/* eslint-disable @next/next/no-img-element */

import { useState, useEffect } from 'react';
import Image from 'next/image';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Sparkles, Video, Loader2, Play, Pause, Plus, Trash2, Edit3,
  Music, Mic, Image as ImageIcon, Clock, Download, RefreshCw,
  ChevronRight, Check, AlertCircle, Volume2, Settings, Wand2,
  FileText, Lightbulb, Eye, Table
} from 'lucide-react';
import toast from 'react-hot-toast';
import { api } from '@/lib/api';

interface Scene {
  id: string;
  order: number;
  start_time: number;
  duration: number;
  scene_type: 'video' | 'image' | 'ai_image';
  description: string;
  search_keywords: string[];
  text_overlay?: string;
  media_url?: string;
  thumbnail_url?: string;
}

interface ScriptSegment {
  time_range: string;
  visual: string;
  audio: string;
  start_seconds: number;
  end_seconds: number;
  duration: number;
}

interface GeneratedScript {
  title: string;
  hook: string;
  segments: ScriptSegment[];
  cta: string;
  music_mood: string;
  hashtags: string[];
  total_duration: number;
}

type GenerationMode = 'prompt' | 'topic';

interface VideoProject {
  id: string;
  title: string;
  description: string;
  duration: number;
  format: string;
  style: string;
  script: string;
  voiceover_text: string;
  music_style?: string;
  music_url?: string;
  scenes: Scene[];
}

interface StageProgress {
  stage: string;
  progress: number;
  message: string;
}

interface GenerationJob {
  job_id: string;
  status: string;
  stages: Record<string, StageProgress>;
  project?: VideoProject;
  output_url?: string;
  error_message?: string;
}

const styleOptions = [
  { id: 'cinematic', name: 'Cinematic', description: 'Professional, movie-like quality' },
  { id: 'energetic', name: 'Energetic', description: 'Fast-paced, dynamic content' },
  { id: 'calm', name: 'Calm', description: 'Peaceful, relaxing atmosphere' },
  { id: 'professional', name: 'Professional', description: 'Corporate, business-oriented' },
  { id: 'fun', name: 'Fun', description: 'Playful, entertaining vibe' },
];

const voiceOptions = [
  { id: '21m00Tcm4TlvDq8ikWAM', name: 'Rachel', provider: 'elevenlabs', description: 'Calm female voice' },
  { id: 'EXAVITQu4vr4xnSDxMaL', name: 'Bella', provider: 'elevenlabs', description: 'Soft female voice' },
  { id: 'ErXwobaYiN019PkySvjV', name: 'Antoni', provider: 'elevenlabs', description: 'Warm male voice' },
  { id: 'nova', name: 'Nova', provider: 'openai', description: 'Friendly, upbeat' },
  { id: 'onyx', name: 'Onyx', provider: 'openai', description: 'Deep, authoritative' },
];

const formatOptions = [
  { id: '9:16', name: 'Portrait (9:16)', description: 'TikTok, Reels, Shorts' },
  { id: '16:9', name: 'Landscape (16:9)', description: 'YouTube, Vimeo' },
  { id: '1:1', name: 'Square (1:1)', description: 'Instagram, Facebook' },
];

// TikTok-style caption presets
const captionStyles = [
  {
    id: 'none',
    name: 'No Captions',
    preview: '—',
    config: null
  },
  {
    id: 'classic',
    name: 'Classic',
    preview: 'Aa',
    config: {
      fontColor: 'white',
      bgColor: 'black',
      bgOpacity: 0.7,
      fontSize: 'medium',
      position: 'bottom',
      animation: 'none'
    }
  },
  {
    id: 'bold',
    name: 'Bold Impact',
    preview: 'Aa',
    config: {
      fontColor: 'white',
      bgColor: 'none',
      bgOpacity: 0,
      fontSize: 'large',
      position: 'center',
      animation: 'pop',
      stroke: true,
      strokeColor: 'black'
    }
  },
  {
    id: 'neon',
    name: 'Neon Glow',
    preview: 'Aa',
    config: {
      fontColor: '#00ff88',
      bgColor: 'none',
      bgOpacity: 0,
      fontSize: 'medium',
      position: 'bottom',
      animation: 'glow',
      glow: true,
      glowColor: '#00ff88'
    }
  },
  {
    id: 'minimal',
    name: 'Minimal',
    preview: 'Aa',
    config: {
      fontColor: 'white',
      bgColor: 'none',
      bgOpacity: 0,
      fontSize: 'small',
      position: 'bottom',
      animation: 'fade'
    }
  },
  {
    id: 'karaoke',
    name: 'Karaoke',
    preview: 'Aa',
    config: {
      fontColor: 'yellow',
      bgColor: 'black',
      bgOpacity: 0.8,
      fontSize: 'large',
      position: 'center',
      animation: 'highlight',
      highlightColor: '#ff3366'
    }
  },
  {
    id: 'boxed',
    name: 'Boxed',
    preview: 'Aa',
    config: {
      fontColor: 'black',
      bgColor: 'white',
      bgOpacity: 1,
      fontSize: 'medium',
      position: 'bottom',
      animation: 'slide',
      rounded: true
    }
  },
  {
    id: 'gradient',
    name: 'Gradient',
    preview: 'Aa',
    config: {
      fontColor: 'white',
      bgColor: 'gradient',
      bgOpacity: 0.9,
      fontSize: 'medium',
      position: 'bottom',
      animation: 'fade',
      gradientColors: ['#667eea', '#764ba2']
    }
  },
];

export default function AIVideoGeneratorPage() {
  // Generation mode
  const [mode, setMode] = useState<GenerationMode>('prompt');

  // Form state
  const [prompt, setPrompt] = useState('');
  const [topic, setTopic] = useState('');
  const [duration, setDuration] = useState(30);
  const [style, setStyle] = useState('cinematic');
  const [format, setFormat] = useState('9:16');
  const [selectedVoice, setSelectedVoice] = useState(voiceOptions[0]);
  const [includeMusic, setIncludeMusic] = useState(true);
  const [musicStyle, setMusicStyle] = useState('');
  const [preferAiImages, setPreferAiImages] = useState(false);
  const [captionStyle, setCaptionStyle] = useState(captionStyles[1]); // Default to 'classic'
  const [targetAudience, setTargetAudience] = useState('general');

  // Script state (for topic mode)
  const [generatedScript, setGeneratedScript] = useState<GeneratedScript | null>(null);
  const [isGeneratingScript, setIsGeneratingScript] = useState(false);
  const [editingSegmentIndex, setEditingSegmentIndex] = useState<number | null>(null);

  // Generation state
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentJob, setCurrentJob] = useState<GenerationJob | null>(null);
  const [editingScene, setEditingScene] = useState<string | null>(null);
  const [editedDescription, setEditedDescription] = useState('');

  // Polling for job status
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (currentJob && !['completed', 'failed'].includes(currentJob.status)) {
      interval = setInterval(async () => {
        try {
          const response = await fetch(
            `http://localhost:8004/api/v1/media/video/generate/${currentJob.job_id}`
          );
          const data = await response.json();
          setCurrentJob(data);

          if (data.status === 'completed') {
            toast.success('Video generated successfully!');
            setIsGenerating(false);
          } else if (data.status === 'failed') {
            toast.error(data.error_message || 'Generation failed');
            setIsGenerating(false);
          }
        } catch (error) {
          console.error('Polling error:', error);
        }
      }, 2000);
    }

    return () => clearInterval(interval);
  }, [currentJob]);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      toast.error('Please enter a description for your video');
      return;
    }

    setIsGenerating(true);

    try {
      const response = await fetch('http://localhost:8004/api/v1/media/video/generate-from-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          duration,
          style,
          format,
          voice_id: selectedVoice.id,
          voice_provider: selectedVoice.provider,
          include_music: includeMusic,
          music_style: musicStyle || undefined,
          prefer_ai_images: preferAiImages,
          caption_style: captionStyle.id !== 'none' ? captionStyle.id : null,
          caption_config: captionStyle.config,
        }),
      });

      if (!response.ok) throw new Error('Failed to start generation');

      const data = await response.json();
      setCurrentJob(data);
      toast.success('Video generation started!');
    } catch (error) {
      toast.error('Failed to start generation');
      setIsGenerating(false);
    }
  };

  const handleUpdateScene = async (sceneId: string) => {
    if (!currentJob || !editedDescription.trim()) return;

    try {
      const response = await fetch(
        `http://localhost:8004/api/v1/media/video/generate/${currentJob.job_id}/scenes/${sceneId}`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            description: editedDescription,
            search_keywords: editedDescription.split(' ').filter(w => w.length > 3),
          }),
        }
      );

      if (response.ok) {
        toast.success('Scene updated');
        setEditingScene(null);
        // Refresh job
        const jobResponse = await fetch(
          `http://localhost:8004/api/v1/media/video/generate/${currentJob.job_id}`
        );
        setCurrentJob(await jobResponse.json());
      }
    } catch (error) {
      toast.error('Failed to update scene');
    }
  };

  const handleAddScene = async () => {
    if (!currentJob) return;

    try {
      const response = await fetch(
        `http://localhost:8004/api/v1/media/video/generate/${currentJob.job_id}/scenes`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            description: 'New scene - click to edit',
            duration: 5,
            scene_type: 'video',
            search_keywords: ['new', 'scene'],
          }),
        }
      );

      if (response.ok) {
        toast.success('Scene added');
        // Refresh job
        const jobResponse = await fetch(
          `http://localhost:8004/api/v1/media/video/generate/${currentJob.job_id}`
        );
        setCurrentJob(await jobResponse.json());
      }
    } catch (error) {
      toast.error('Failed to add scene');
    }
  };

  const handleRemoveScene = async (sceneId: string) => {
    if (!currentJob) return;

    try {
      const response = await fetch(
        `http://localhost:8004/api/v1/media/video/generate/${currentJob.job_id}/scenes/${sceneId}`,
        { method: 'DELETE' }
      );

      if (response.ok) {
        toast.success('Scene removed');
        // Refresh job
        const jobResponse = await fetch(
          `http://localhost:8004/api/v1/media/video/generate/${currentJob.job_id}`
        );
        setCurrentJob(await jobResponse.json());
      }
    } catch (error) {
      toast.error('Failed to remove scene');
    }
  };

  const handleRegenerate = async () => {
    if (!currentJob) return;

    setIsGenerating(true);

    try {
      const response = await fetch(
        `http://localhost:8004/api/v1/media/video/generate/${currentJob.job_id}/regenerate`,
        { method: 'POST' }
      );

      if (response.ok) {
        const data = await response.json();
        setCurrentJob(data);
        toast.success('Regenerating video with updated scenes...');
      }
    } catch (error) {
      toast.error('Failed to regenerate');
      setIsGenerating(false);
    }
  };

  // Generate script from topic
  const handleGenerateScript = async () => {
    if (!topic.trim()) {
      toast.error('Please enter a topic for your video');
      return;
    }

    setIsGeneratingScript(true);

    try {
      const response = await fetch('http://localhost:8004/api/v1/media/script/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic,
          duration,
          style,
          target_audience: targetAudience,
        }),
      });

      if (!response.ok) throw new Error('Failed to generate script');

      const data = await response.json();
      setGeneratedScript(data);
      toast.success('Script generated! Review and edit before creating video.');
    } catch (error) {
      toast.error('Failed to generate script');
    } finally {
      setIsGeneratingScript(false);
    }
  };

  // Generate video from script
  const handleGenerateFromScript = async () => {
    if (!generatedScript) {
      toast.error('Please generate a script first');
      return;
    }

    setIsGenerating(true);

    try {
      const response = await fetch('http://localhost:8004/api/v1/media/video/generate-from-script', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          script: generatedScript,
          format,
          voice_id: selectedVoice.id,
          voice_provider: selectedVoice.provider,
          include_music: includeMusic,
          prefer_ai_images: preferAiImages,
          caption_style: captionStyle.id !== 'none' ? captionStyle.id : null,
          caption_config: captionStyle.config,
        }),
      });

      if (!response.ok) throw new Error('Failed to start generation');

      const data = await response.json();
      setCurrentJob(data);
      toast.success('Video generation started from script!');
    } catch (error) {
      toast.error('Failed to start generation');
      setIsGenerating(false);
    }
  };

  // Update a script segment
  const handleUpdateSegment = (index: number, field: 'visual' | 'audio', value: string) => {
    if (!generatedScript) return;

    const updatedSegments = [...generatedScript.segments];
    updatedSegments[index] = { ...updatedSegments[index], [field]: value };
    setGeneratedScript({ ...generatedScript, segments: updatedSegments });
  };

  const getStageIcon = (stage: string, status: string, progress: number) => {
    if (progress === 100) return <Check className="w-5 h-5 text-green-400" />;
    if (status === stage) return <Loader2 className="w-5 h-5 text-orange-400 animate-spin" />;
    return <div className="w-5 h-5 rounded-full border-2 border-gray-600" />;
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
  };

  return (
    <div className="min-h-screen bg-gray-900 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl">
              <Wand2 className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-white">AI Video Generator</h1>
          </div>
          <p className="text-gray-400 mb-4">
            Create viral videos with AI - from prompt or structured script
          </p>

          {/* Mode Toggle */}
          <div className="flex gap-2 p-1 bg-gray-800 rounded-xl w-fit">
            <button
              onClick={() => { setMode('prompt'); setGeneratedScript(null); }}
              className={`px-4 py-2 rounded-lg flex items-center gap-2 transition ${
                mode === 'prompt'
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <Sparkles className="w-4 h-4" />
              Quick Prompt
            </button>
            <button
              onClick={() => setMode('topic')}
              className={`px-4 py-2 rounded-lg flex items-center gap-2 transition ${
                mode === 'topic'
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <FileText className="w-4 h-4" />
              Script from Topic
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Panel - Input Form */}
          <div className="space-y-6">
            {/* Mode-specific Input */}
            {mode === 'prompt' ? (
              /* Prompt Input */
              <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                <label className="block text-lg font-semibold text-white mb-3">
                  Describe Your Video
                </label>
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Example: Create a 30-second motivational video about the power of morning routines, showing sunrise, meditation, exercise, and healthy breakfast..."
                  className="w-full h-32 px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                />
                <p className="text-sm text-gray-500 mt-2">
                  Be specific about the mood, visuals, and message you want to convey
                </p>
              </div>
            ) : (
              /* Topic-based Script Generation */
              <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                <label className="block text-lg font-semibold text-white mb-3 flex items-center gap-2">
                  <Lightbulb className="w-5 h-5 text-yellow-400" />
                  Enter Your Topic
                </label>
                <textarea
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  placeholder="Example: 5 productivity hacks that will change your life, or the psychology of viral content, or how AI is transforming the creative industry..."
                  className="w-full h-24 px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                />
                <div className="mt-4">
                  <label className="block text-sm text-gray-400 mb-2">Target Audience</label>
                  <select
                    value={targetAudience}
                    onChange={(e) => setTargetAudience(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="general">General Audience</option>
                    <option value="professionals">Professionals</option>
                    <option value="students">Students</option>
                    <option value="entrepreneurs">Entrepreneurs</option>
                    <option value="creators">Content Creators</option>
                    <option value="tech">Tech Enthusiasts</option>
                  </select>
                </div>
                <button
                  onClick={handleGenerateScript}
                  disabled={isGeneratingScript || !topic.trim()}
                  className="mt-4 w-full py-3 bg-gradient-to-r from-yellow-500 to-orange-500 text-white font-semibold rounded-xl hover:from-yellow-600 hover:to-orange-600 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isGeneratingScript ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Generating Script...
                    </>
                  ) : (
                    <>
                      <FileText className="w-5 h-5" />
                      Generate Script
                    </>
                  )}
                </button>
                <p className="text-sm text-gray-500 mt-2">
                  AI will create a structured script with Time, Visual, and Audio columns
                </p>
              </div>
            )}

            {/* Settings */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Settings className="w-5 h-5 text-gray-400" />
                Video Settings
              </h3>

              <div className="space-y-4">
                {/* Duration */}
                <div>
                  <label className="block text-sm text-gray-400 mb-2">
                    Duration: {Math.floor(duration / 60)}:{(duration % 60).toString().padStart(2, '0')}
                    <span className="text-gray-500 ml-1">({duration}s)</span>
                  </label>
                  <input
                    type="range"
                    min="15"
                    max="2700"
                    step="15"
                    value={duration}
                    onChange={(e) => setDuration(parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>15s</span>
                    <span>5min</span>
                    <span>15min</span>
                    <span>30min</span>
                    <span>45min</span>
                  </div>
                </div>

                {/* Format */}
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Format</label>
                  <div className="grid grid-cols-3 gap-2">
                    {formatOptions.map((f) => (
                      <button
                        key={f.id}
                        onClick={() => setFormat(f.id)}
                        className={`p-3 rounded-lg text-center transition ${
                          format === f.id
                            ? 'bg-purple-600 text-white'
                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                        }`}
                      >
                        <div className="text-sm font-medium">{f.name}</div>
                        <div className="text-xs opacity-70">{f.description}</div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Style */}
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Visual Style</label>
                  <select
                    value={style}
                    onChange={(e) => setStyle(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    {styleOptions.map((s) => (
                      <option key={s.id} value={s.id}>{s.name} - {s.description}</option>
                    ))}
                  </select>
                </div>

                {/* Voice */}
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Voiceover Voice</label>
                  <select
                    value={selectedVoice.id}
                    onChange={(e) => setSelectedVoice(voiceOptions.find(v => v.id === e.target.value) || voiceOptions[0])}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    {voiceOptions.map((v) => (
                      <option key={v.id} value={v.id}>{v.name} ({v.provider}) - {v.description}</option>
                    ))}
                  </select>
                </div>

                {/* Music */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm text-white">Include Background Music</label>
                    <p className="text-xs text-gray-500">Royalty-free music matching your video mood</p>
                  </div>
                  <button
                    onClick={() => setIncludeMusic(!includeMusic)}
                    className={`w-12 h-6 rounded-full transition ${
                      includeMusic ? 'bg-purple-600' : 'bg-gray-600'
                    }`}
                  >
                    <div className={`w-5 h-5 bg-white rounded-full transition transform ${
                      includeMusic ? 'translate-x-6' : 'translate-x-1'
                    }`} />
                  </button>
                </div>

                {/* AI Images */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm text-white">Prefer AI-Generated Images</label>
                    <p className="text-xs text-gray-500">Use DALL-E instead of stock footage</p>
                  </div>
                  <button
                    onClick={() => setPreferAiImages(!preferAiImages)}
                    className={`w-12 h-6 rounded-full transition ${
                      preferAiImages ? 'bg-purple-600' : 'bg-gray-600'
                    }`}
                  >
                    <div className={`w-5 h-5 bg-white rounded-full transition transform ${
                      preferAiImages ? 'translate-x-6' : 'translate-x-1'
                    }`} />
                  </button>
                </div>

                {/* Caption Styles */}
                <div>
                  <label className="block text-sm text-gray-400 mb-3">Caption Style</label>
                  <div className="grid grid-cols-4 gap-2">
                    {captionStyles.map((cs) => (
                      <button
                        key={cs.id}
                        onClick={() => setCaptionStyle(cs)}
                        className={`relative p-3 rounded-xl text-center transition border-2 ${
                          captionStyle.id === cs.id
                            ? 'border-purple-500 bg-purple-600/20'
                            : 'border-gray-600 bg-gray-700 hover:border-gray-500'
                        }`}
                      >
                        {/* Style preview */}
                        <div
                          className={`text-lg font-bold mb-1 ${
                            cs.id === 'none' ? 'text-gray-500' :
                            cs.id === 'neon' ? 'text-green-400' :
                            cs.id === 'karaoke' ? 'text-yellow-400' :
                            cs.id === 'boxed' ? 'text-gray-900 bg-white px-1 rounded' :
                            cs.id === 'gradient' ? 'bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent' :
                            cs.id === 'bold' ? 'text-white drop-shadow-[2px_2px_0px_rgba(0,0,0,1)]' :
                            'text-white'
                          }`}
                          style={cs.id === 'neon' ? { textShadow: '0 0 10px #00ff88' } : undefined}
                        >
                          {cs.preview}
                        </div>
                        <div className="text-xs text-gray-400">{cs.name}</div>
                        {captionStyle.id === cs.id && (
                          <div className="absolute -top-1 -right-1 w-4 h-4 bg-purple-500 rounded-full flex items-center justify-center">
                            <Check className="w-3 h-3 text-white" />
                          </div>
                        )}
                      </button>
                    ))}
                  </div>
                  {captionStyle.id !== 'none' && (
                    <p className="text-xs text-gray-500 mt-2">
                      Captions will be auto-generated from voiceover text
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Generate Button - Only show in prompt mode */}
            {mode === 'prompt' && (
              <button
                onClick={handleGenerate}
                disabled={isGenerating || !prompt.trim()}
                className="w-full py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold rounded-xl hover:from-purple-700 hover:to-pink-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-5 h-5" />
                    Generate Video
                  </>
                )}
              </button>
            )}

            {/* Script Preview - Only show in topic mode with generated script */}
            {mode === 'topic' && generatedScript && (
              <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                    <Table className="w-5 h-5 text-purple-400" />
                    Generated Script
                  </h3>
                  <span className="text-sm text-gray-400">
                    {generatedScript.total_duration}s total
                  </span>
                </div>

                {/* Script Title & Hook */}
                <div className="mb-4 p-3 bg-gray-700/50 rounded-xl">
                  <h4 className="text-white font-medium">{generatedScript.title}</h4>
                  <p className="text-sm text-yellow-400 mt-1">🪝 {generatedScript.hook}</p>
                </div>

                {/* Script Table */}
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-600">
                        <th className="text-left py-2 px-2 text-gray-400 font-medium w-20">Time</th>
                        <th className="text-left py-2 px-2 text-gray-400 font-medium">Visual</th>
                        <th className="text-left py-2 px-2 text-gray-400 font-medium">Audio</th>
                      </tr>
                    </thead>
                    <tbody>
                      {generatedScript.segments.map((segment, index) => (
                        <tr
                          key={index}
                          className="border-b border-gray-700/50 hover:bg-gray-700/30"
                        >
                          <td className="py-2 px-2 text-purple-400 font-mono text-xs whitespace-nowrap">
                            {segment.time_range}
                          </td>
                          <td className="py-2 px-2">
                            {editingSegmentIndex === index ? (
                              <input
                                type="text"
                                value={segment.visual}
                                onChange={(e) => handleUpdateSegment(index, 'visual', e.target.value)}
                                className="w-full px-2 py-1 bg-gray-600 border border-gray-500 rounded text-white text-xs"
                                onBlur={() => setEditingSegmentIndex(null)}
                              />
                            ) : (
                              <p
                                className="text-gray-300 text-xs cursor-pointer hover:text-white"
                                onClick={() => setEditingSegmentIndex(index)}
                              >
                                {segment.visual}
                              </p>
                            )}
                          </td>
                          <td className="py-2 px-2">
                            {editingSegmentIndex === index ? (
                              <input
                                type="text"
                                value={segment.audio}
                                onChange={(e) => handleUpdateSegment(index, 'audio', e.target.value)}
                                className="w-full px-2 py-1 bg-gray-600 border border-gray-500 rounded text-white text-xs"
                                onBlur={() => setEditingSegmentIndex(null)}
                              />
                            ) : (
                              <p
                                className="text-blue-300 text-xs cursor-pointer hover:text-white"
                                onClick={() => setEditingSegmentIndex(index)}
                              >
                                &quot;{segment.audio}&quot;
                              </p>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* CTA & Music */}
                <div className="mt-4 grid grid-cols-2 gap-3">
                  <div className="p-3 bg-gray-700/50 rounded-xl">
                    <span className="text-xs text-gray-400">Call to Action</span>
                    <p className="text-sm text-green-400">{generatedScript.cta}</p>
                  </div>
                  <div className="p-3 bg-gray-700/50 rounded-xl">
                    <span className="text-xs text-gray-400">Music Mood</span>
                    <p className="text-sm text-pink-400">{generatedScript.music_mood}</p>
                  </div>
                </div>

                {/* Hashtags */}
                {generatedScript.hashtags.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {generatedScript.hashtags.map((tag, i) => (
                      <span key={i} className="px-2 py-1 bg-purple-600/30 text-purple-300 text-xs rounded-full">
                        #{tag}
                      </span>
                    ))}
                  </div>
                )}

                {/* Generate Video from Script Button */}
                <button
                  onClick={handleGenerateFromScript}
                  disabled={isGenerating}
                  className="mt-4 w-full py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold rounded-xl hover:from-purple-700 hover:to-pink-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isGenerating ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Creating Video...
                    </>
                  ) : (
                    <>
                      <Video className="w-5 h-5" />
                      Create Video from Script
                    </>
                  )}
                </button>

                <p className="text-xs text-gray-500 mt-2 text-center">
                  Click on any cell to edit before generating
                </p>
              </div>
            )}
          </div>

          {/* Right Panel - Progress & Results */}
          <div className="space-y-6">
            {/* Generation Progress */}
            {currentJob && (
              <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-4">Generation Progress</h3>

                <div className="space-y-3">
                  {['planning', 'fetching_assets', 'generating_voiceover', 'fetching_music', 'composing'].map((stage) => {
                    const stageData = currentJob.stages[stage];
                    const isActive = currentJob.status === stage;

                    return (
                      <div
                        key={stage}
                        className={`flex items-center gap-3 p-3 rounded-lg transition ${
                          isActive ? 'bg-gray-700' : 'bg-gray-800'
                        }`}
                      >
                        {getStageIcon(stage, currentJob.status, stageData?.progress || 0)}
                        <div className="flex-1">
                          <div className="flex items-center justify-between">
                            <span className="text-white capitalize">
                              {stage.replace('_', ' ')}
                            </span>
                            <span className="text-sm text-gray-400">
                              {stageData?.progress || 0}%
                            </span>
                          </div>
                          {isActive && stageData?.message && (
                            <p className="text-sm text-gray-400">{stageData.message}</p>
                          )}
                          {isActive && (
                            <div className="mt-2 h-1 bg-gray-600 rounded-full overflow-hidden">
                              <motion.div
                                className="h-full bg-purple-500"
                                initial={{ width: 0 }}
                                animate={{ width: `${stageData?.progress || 0}%` }}
                              />
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {currentJob.status === 'failed' && (
                  <div className="mt-4 p-3 bg-red-500/20 border border-red-500/30 rounded-lg flex items-center gap-2">
                    <AlertCircle className="w-5 h-5 text-red-400" />
                    <span className="text-red-200">{currentJob.error_message}</span>
                  </div>
                )}

                {currentJob.status === 'completed' && currentJob.output_url && (
                  <div className="mt-4 space-y-3">
                    {/* Video Preview - aspect ratio based on format */}
                    <div className={`${
                      format === '16:9' ? 'aspect-video' :
                      format === '1:1' ? 'aspect-square' :
                      'aspect-[9/16]'
                    } max-h-96 bg-black rounded-xl overflow-hidden mx-auto`}>
                      <video
                        src={`http://localhost:8004/api/v1/media/video/download/${currentJob.job_id}`}
                        controls
                        className="w-full h-full object-contain"
                      />
                    </div>
                    <a
                      href={`http://localhost:8004/api/v1/media/video/download/${currentJob.job_id}`}
                      download={`${currentJob.project?.title || 'video'}.mp4`}
                      className="w-full py-3 bg-green-600 text-white font-semibold rounded-xl hover:bg-green-700 transition flex items-center justify-center gap-2"
                    >
                      <Download className="w-5 h-5" />
                      Download Video
                    </a>
                  </div>
                )}
              </div>
            )}

            {/* Scene Editor */}
            {currentJob?.project && (
              <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white">Scenes</h3>
                  <div className="flex gap-2">
                    <button
                      onClick={handleAddScene}
                      className="px-3 py-1.5 bg-gray-700 text-white text-sm rounded-lg hover:bg-gray-600 transition flex items-center gap-1"
                    >
                      <Plus className="w-4 h-4" />
                      Add Scene
                    </button>
                    <button
                      onClick={handleRegenerate}
                      disabled={isGenerating}
                      className="px-3 py-1.5 bg-purple-600 text-white text-sm rounded-lg hover:bg-purple-700 transition flex items-center gap-1 disabled:opacity-50"
                    >
                      <RefreshCw className="w-4 h-4" />
                      Regenerate
                    </button>
                  </div>
                </div>

                <div className="space-y-3">
                  {currentJob.project.scenes.map((scene, index) => (
                    <motion.div
                      key={scene.id}
                      layout
                      className="bg-gray-700/50 rounded-xl p-4"
                    >
                      <div className="flex items-start gap-3">
                        {/* Scene number */}
                        <div className="w-8 h-8 bg-purple-600/30 rounded-lg flex items-center justify-center text-purple-400 font-bold">
                          {index + 1}
                        </div>

                        {/* Thumbnail/Preview */}
                        <div className="w-20 h-14 bg-gray-600 rounded-lg overflow-hidden flex-shrink-0">
                          {scene.thumbnail_url || scene.media_url ? (
                            <img
                              src={scene.thumbnail_url || scene.media_url}
                              alt={`Scene ${index + 1}`}
                              className="w-full h-full object-cover"
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center">
                              {scene.scene_type === 'video' ? (
                                <Video className="w-6 h-6 text-gray-400" />
                              ) : (
                                <ImageIcon className="w-6 h-6 text-gray-400" />
                              )}
                            </div>
                          )}
                        </div>

                        {/* Scene details */}
                        <div className="flex-1 min-w-0">
                          {editingScene === scene.id ? (
                            <div className="flex gap-2">
                              <input
                                type="text"
                                value={editedDescription}
                                onChange={(e) => setEditedDescription(e.target.value)}
                                className="flex-1 px-3 py-1 bg-gray-600 border border-gray-500 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                                placeholder="Scene description..."
                              />
                              <button
                                onClick={() => handleUpdateScene(scene.id)}
                                className="px-3 py-1 bg-green-600 text-white rounded-lg text-sm"
                              >
                                Save
                              </button>
                              <button
                                onClick={() => setEditingScene(null)}
                                className="px-3 py-1 bg-gray-600 text-white rounded-lg text-sm"
                              >
                                Cancel
                              </button>
                            </div>
                          ) : (
                            <>
                              <p className="text-white text-sm line-clamp-2">{scene.description}</p>
                              <div className="flex items-center gap-3 mt-1">
                                <span className="text-xs text-gray-400 flex items-center gap-1">
                                  <Clock className="w-3 h-3" />
                                  {formatDuration(scene.duration)}
                                </span>
                                <span className="text-xs text-purple-400 capitalize">
                                  {scene.scene_type.replace('_', ' ')}
                                </span>
                              </div>
                            </>
                          )}
                        </div>

                        {/* Actions */}
                        {editingScene !== scene.id && (
                          <div className="flex gap-1">
                            <button
                              onClick={() => {
                                setEditingScene(scene.id);
                                setEditedDescription(scene.description);
                              }}
                              className="p-2 hover:bg-gray-600 rounded-lg transition"
                              title="Edit scene"
                            >
                              <Edit3 className="w-4 h-4 text-gray-400" />
                            </button>
                            <button
                              onClick={() => handleRemoveScene(scene.id)}
                              className="p-2 hover:bg-red-600/20 rounded-lg transition"
                              title="Remove scene"
                            >
                              <Trash2 className="w-4 h-4 text-red-400" />
                            </button>
                          </div>
                        )}
                      </div>

                      {/* Text overlay preview */}
                      {scene.text_overlay && (
                        <div className="mt-2 ml-11 p-2 bg-gray-600/50 rounded-lg">
                          <span className="text-xs text-gray-400">Text overlay: </span>
                          <span className="text-xs text-white">{scene.text_overlay}</span>
                        </div>
                      )}
                    </motion.div>
                  ))}
                </div>

                {/* Script Preview */}
                {currentJob.project.voiceover_text && (
                  <div className="mt-6 p-4 bg-gray-700/30 rounded-xl">
                    <h4 className="text-sm font-medium text-gray-400 mb-2 flex items-center gap-2">
                      <Mic className="w-4 h-4" />
                      Voiceover Script
                    </h4>
                    <p className="text-gray-300 text-sm">{currentJob.project.voiceover_text}</p>
                  </div>
                )}

                {/* Music */}
                {currentJob.project.music_style && (
                  <div className="mt-4 p-4 bg-gray-700/30 rounded-xl">
                    <h4 className="text-sm font-medium text-gray-400 mb-1 flex items-center gap-2">
                      <Music className="w-4 h-4" />
                      Background Music
                    </h4>
                    <p className="text-gray-300 text-sm capitalize">{currentJob.project.music_style}</p>
                  </div>
                )}
              </div>
            )}

            {/* Empty State */}
            {!currentJob && (
              <div className="bg-gray-800 rounded-2xl p-12 border border-gray-700 text-center">
                <Video className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-400 mb-2">No Video Yet</h3>
                <p className="text-gray-500">
                  Enter a description and click &quot;Generate Video&quot; to create your AI-powered video
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
