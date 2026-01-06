'use client';
/* eslint-disable @next/next/no-img-element */

import { useState } from 'react';
import Image from 'next/image';
import { motion } from 'framer-motion';
import {
  Image as ImageIcon, Wand2, Download, Copy, RefreshCw,
  Loader2, Sparkles, Grid, Maximize2, Info, Check,
  ChevronDown, Palette, Zap
} from 'lucide-react';
import toast from 'react-hot-toast';
import { api } from '@/lib/api';

interface GeneratedImage {
  id: string;
  url: string;
  prompt: string;
  style: string;
  createdAt: Date;
}

const presets = [
  { id: 'thumbnail', name: 'Thumbnail', description: 'YouTube/TikTok thumbnail', aspectRatio: '16:9' },
  { id: 'post', name: 'Social Post', description: 'Instagram/Facebook post', aspectRatio: '1:1' },
  { id: 'story', name: 'Story/Reel', description: 'Vertical content', aspectRatio: '9:16' },
  { id: 'banner', name: 'Banner', description: 'Channel art', aspectRatio: '16:9' },
];

const styles = [
  { id: 'photorealistic', name: 'Photorealistic', icon: '📷' },
  { id: 'digital-art', name: 'Digital Art', icon: '🎨' },
  { id: 'anime', name: 'Anime', icon: '🎌' },
  { id: 'illustration', name: 'Illustration', icon: '✏️' },
  { id: '3d-render', name: '3D Render', icon: '🧊' },
  { id: 'cinematic', name: 'Cinematic', icon: '🎬' },
  { id: 'pop-art', name: 'Pop Art', icon: '🌈' },
  { id: 'minimalist', name: 'Minimalist', icon: '⬜' },
];


export default function ImageGeneratorPage() {
  const [prompt, setPrompt] = useState('');
  const [selectedPreset, setSelectedPreset] = useState('thumbnail');
  const [selectedStyle, setSelectedStyle] = useState('photorealistic');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImages, setGeneratedImages] = useState<GeneratedImage[]>([]);
  const [selectedImage, setSelectedImage] = useState<GeneratedImage | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Advanced options
  const [negativePrompt, setNegativePrompt] = useState('');
  const [quality, setQuality] = useState<'standard' | 'hd'>('standard');
  const [imageCount, setImageCount] = useState(1);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      toast.error('Please enter a prompt');
      return;
    }

    setIsGenerating(true);

    try {
      // Call real API
      const jobResponse = await api.media.generateImage({
        prompt,
        style: selectedStyle,
        preset: selectedPreset,
        quality,
        negative_prompt: negativePrompt,
      });

      toast.success('Image generation started...');

      // Poll for job completion
      const jobId = jobResponse.job_id;
      let attempts = 0;
      const maxAttempts = 30; // 30 seconds max

      const pollJob = async (): Promise<any> => {
        const status = await api.media.getJobStatus(jobId);

        if (status.status === 'completed') {
          return status;
        } else if (status.status === 'failed') {
          throw new Error(status.error_message || 'Generation failed');
        } else if (attempts >= maxAttempts) {
          throw new Error('Generation timed out');
        }

        attempts++;
        await new Promise(resolve => setTimeout(resolve, 1000));
        return pollJob();
      };

      const result = await pollJob();

      // Extract image URL from result
      const imageUrl = result.output_data?.url || result.output_data?.image_url;

      if (imageUrl) {
        const newImage: GeneratedImage = {
          id: jobId,
          url: imageUrl,
          prompt: result.output_data?.revised_prompt || prompt,
          style: selectedStyle,
          createdAt: new Date(),
        };

        setGeneratedImages(prev => [newImage, ...prev]);
        setSelectedImage(newImage);
        toast.success('Image generated successfully!');
      }
    } catch (error: any) {
      console.error('Generation error:', error);
      toast.error(error.message || 'Failed to generate image');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = async (image: GeneratedImage) => {
    try {
      const response = await fetch(image.url);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `viralify-${image.id}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      toast.success('Image downloaded!');
    } catch (error) {
      toast.error('Download failed');
    }
  };

  const copyPrompt = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    toast.success('Prompt copied!');
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="min-h-screen bg-gray-900 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl">
              <ImageIcon className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-white">Image Generator</h1>
          </div>
          <p className="text-gray-400">
            Create stunning AI-generated images with DALL-E 3
          </p>
        </div>

        {/* API Status Banner */}
        <div className="mb-6 p-3 bg-green-500/20 border border-green-500/30 rounded-xl flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-green-400 flex-shrink-0" />
          <p className="text-sm text-green-200">
            <strong>Connected to DALL-E 3:</strong> Real AI image generation is active.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Generation Form */}
          <div className="space-y-6">
            {/* Prompt Input */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                Describe your image
              </label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="A futuristic cityscape at sunset with neon lights, flying cars, and tall skyscrapers..."
                rows={4}
                className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
              />
              <div className="flex justify-between mt-2 text-xs text-gray-500">
                <span>Be specific for better results</span>
                <span>{prompt.length} / 1000</span>
              </div>
            </div>

            {/* Preset Selection */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                Format Preset
              </label>
              <div className="grid grid-cols-2 gap-3">
                {presets.map((preset) => (
                  <button
                    key={preset.id}
                    onClick={() => setSelectedPreset(preset.id)}
                    className={`p-3 rounded-xl border text-left transition ${
                      selectedPreset === preset.id
                        ? 'bg-purple-600/20 border-purple-500 text-white'
                        : 'bg-gray-700 border-gray-600 text-gray-300 hover:border-gray-500'
                    }`}
                  >
                    <div className="font-medium text-sm">{preset.name}</div>
                    <div className="text-xs text-gray-400">{preset.aspectRatio}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Style Selection */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                Style
              </label>
              <div className="grid grid-cols-4 gap-2">
                {styles.map((style) => (
                  <button
                    key={style.id}
                    onClick={() => setSelectedStyle(style.id)}
                    className={`p-3 rounded-xl border text-center transition ${
                      selectedStyle === style.id
                        ? 'bg-purple-600/20 border-purple-500'
                        : 'bg-gray-700 border-gray-600 hover:border-gray-500'
                    }`}
                  >
                    <div className="text-xl mb-1">{style.icon}</div>
                    <div className="text-xs text-gray-300">{style.name}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Advanced Options */}
            <div className="bg-gray-800 rounded-2xl border border-gray-700 overflow-hidden">
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="w-full p-4 flex items-center justify-between text-gray-300 hover:bg-gray-700/50 transition"
              >
                <span className="font-medium">Advanced Options</span>
                <ChevronDown className={`w-5 h-5 transition ${showAdvanced ? 'rotate-180' : ''}`} />
              </button>

              {showAdvanced && (
                <div className="p-4 pt-0 space-y-4 border-t border-gray-700">
                  {/* Negative Prompt */}
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-2">
                      Negative Prompt (What to avoid)
                    </label>
                    <input
                      type="text"
                      value={negativePrompt}
                      onChange={(e) => setNegativePrompt(e.target.value)}
                      placeholder="blurry, low quality, distorted..."
                      className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>

                  {/* Quality */}
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-2">
                      Quality
                    </label>
                    <div className="flex gap-3">
                      <button
                        onClick={() => setQuality('standard')}
                        className={`flex-1 py-2 rounded-xl border transition ${
                          quality === 'standard'
                            ? 'bg-purple-600/20 border-purple-500 text-white'
                            : 'bg-gray-700 border-gray-600 text-gray-300'
                        }`}
                      >
                        Standard (1 credit)
                      </button>
                      <button
                        onClick={() => setQuality('hd')}
                        className={`flex-1 py-2 rounded-xl border transition ${
                          quality === 'hd'
                            ? 'bg-purple-600/20 border-purple-500 text-white'
                            : 'bg-gray-700 border-gray-600 text-gray-300'
                        }`}
                      >
                        HD (2 credits)
                      </button>
                    </div>
                  </div>

                  {/* Image Count */}
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-2">
                      Number of Images
                    </label>
                    <div className="flex gap-2">
                      {[1, 2, 3, 4].map((num) => (
                        <button
                          key={num}
                          onClick={() => setImageCount(num)}
                          className={`flex-1 py-2 rounded-xl border transition ${
                            imageCount === num
                              ? 'bg-purple-600/20 border-purple-500 text-white'
                              : 'bg-gray-700 border-gray-600 text-gray-300'
                          }`}
                        >
                          {num}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Generate Button */}
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
                  Generate Image ({quality === 'hd' ? 2 * imageCount : imageCount} credits)
                </>
              )}
            </button>
          </div>

          {/* Results Panel */}
          <div className="space-y-6">
            {/* Selected Image Preview */}
            {selectedImage ? (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-gray-800 rounded-2xl p-4 border border-gray-700"
              >
                <div className="relative aspect-square rounded-xl overflow-hidden bg-gray-900 mb-4">
                  <img
                    src={selectedImage.url}
                    alt={selectedImage.prompt}
                    className="w-full h-full object-cover"
                  />
                  <button
                    className="absolute top-2 right-2 p-2 bg-black/50 rounded-lg hover:bg-black/70 transition"
                    onClick={() => window.open(selectedImage.url, '_blank')}
                  >
                    <Maximize2 className="w-4 h-4 text-white" />
                  </button>
                </div>

                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Prompt</span>
                    <button
                      onClick={() => copyPrompt(selectedImage.prompt, selectedImage.id)}
                      className="p-1 hover:bg-gray-700 rounded transition"
                    >
                      {copiedId === selectedImage.id ? (
                        <Check className="w-4 h-4 text-green-400" />
                      ) : (
                        <Copy className="w-4 h-4 text-gray-400" />
                      )}
                    </button>
                  </div>
                  <p className="text-sm text-gray-300 bg-gray-700/50 p-3 rounded-lg">
                    {selectedImage.prompt}
                  </p>

                  <div className="flex gap-3">
                    <button
                      onClick={() => handleDownload(selectedImage)}
                      className="flex-1 py-3 bg-gray-700 text-white rounded-xl hover:bg-gray-600 transition flex items-center justify-center gap-2"
                    >
                      <Download className="w-4 h-4" />
                      Download
                    </button>
                    <button
                      onClick={() => {
                        setPrompt(selectedImage.prompt);
                        toast.success('Prompt loaded');
                      }}
                      className="flex-1 py-3 bg-purple-600 text-white rounded-xl hover:bg-purple-700 transition flex items-center justify-center gap-2"
                    >
                      <RefreshCw className="w-4 h-4" />
                      Regenerate
                    </button>
                  </div>
                </div>
              </motion.div>
            ) : (
              <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700 text-center">
                <div className="w-16 h-16 bg-gray-700 rounded-2xl flex items-center justify-center mx-auto mb-4">
                  <Palette className="w-8 h-8 text-gray-500" />
                </div>
                <h3 className="text-lg font-medium text-white mb-2">No images yet</h3>
                <p className="text-gray-400 text-sm">
                  Enter a prompt and click generate to create your first image
                </p>
              </div>
            )}

            {/* Generated Images Grid */}
            {generatedImages.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
                  <Grid className="w-4 h-4" />
                  Generated Images ({generatedImages.length})
                </h3>
                <div className="grid grid-cols-4 gap-3">
                  {generatedImages.map((image) => (
                    <button
                      key={image.id}
                      onClick={() => setSelectedImage(image)}
                      className={`aspect-square rounded-xl overflow-hidden border-2 transition ${
                        selectedImage?.id === image.id
                          ? 'border-purple-500'
                          : 'border-transparent hover:border-gray-600'
                      }`}
                    >
                      <img
                        src={image.url}
                        alt={image.prompt}
                        className="w-full h-full object-cover"
                      />
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Prompt Suggestions */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <h3 className="text-sm font-medium text-gray-300 mb-3">Prompt Ideas</h3>
              <div className="space-y-2">
                {[
                  'A surprised person reacting to their phone, vibrant colors, social media style',
                  'Professional YouTube thumbnail with bold text space, tech theme',
                  'Aesthetic lifestyle flat lay with coffee and notebook, warm tones',
                ].map((suggestion, i) => (
                  <button
                    key={i}
                    onClick={() => setPrompt(suggestion)}
                    className="w-full p-3 bg-gray-700/50 rounded-xl text-left text-sm text-gray-300 hover:bg-gray-700 transition"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
