'use client';
/* eslint-disable @next/next/no-img-element */

import { useState, useRef, useEffect } from 'react';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Sparkles, Wand2, Upload, Clock, Calendar,
  Video, FileText, Target, Loader2, X,
  Copy, Check, RefreshCw, ChevronRight, ChevronLeft, Info,
  Play, Pause, Volume2, VolumeX, Image as ImageIcon,
  Mic, Music, Type, Layers, Download, Eye,
  Zap, ArrowRight, Plus, Trash2, Settings, Search, User, Star
} from 'lucide-react';
import toast from 'react-hot-toast';
import Link from 'next/link';
import { DEMO_MODE } from '@/lib/demo-mode';
import { api, GeneratedScript, GeneratedCaption } from '@/lib/api';
import {
  CreatorProfile,
  getCreatorProfiles,
  getDefaultProfile,
  buildSystemPrompt,
  TONE_OPTIONS,
} from '@/lib/creator-profiles';

// Types
type ContentType = 'video-ai' | 'video-upload' | 'video-stock' | 'image' | 'image-text' | 'carousel';
type Step = 1 | 2 | 3 | 4;

interface MediaAsset {
  id: string;
  type: 'video' | 'image' | 'voiceover';
  url: string;
  thumbnailUrl?: string;
  duration?: number;
  prompt?: string;
}

interface TextOverlay {
  id: string;
  text: string;
  position: 'top' | 'center' | 'bottom';
  style: 'bold' | 'minimal' | 'gradient';
}

interface StockVideo {
  id: string;
  url: string;
  preview_url: string;
  duration: number;
  width: number;
  height: number;
  user: string;
}

// AI Video Script Scene structure
type VisualType = 'image' | 'diagram' | 'video' | 'text-overlay';

interface ScriptScene {
  time: string;
  visual: string;
  audio: string;
  visualType: VisualType;
  visualPrompt?: string; // Optimized prompt for media generation
  diagramDescription?: string; // For diagrams: detailed description
  media?: MediaAsset | null;
  mediaOptions?: StockVideo[];
  isSearching?: boolean;
  isGeneratingMedia?: boolean;
  generatedMediaUrl?: string;
}

// Content type configurations
const contentTypes = [
  {
    id: 'video-ai' as ContentType,
    title: 'AI Video',
    description: 'Generate a complete video from a text prompt',
    icon: Sparkles,
    gradient: 'from-pink-500 to-rose-500',
    credits: 10,
    features: ['Text to Video', 'Auto Scenes', 'Voice Narration'],
  },
  {
    id: 'video-stock' as ContentType,
    title: 'Stock Video',
    description: 'Combine stock footage into a viral video',
    icon: Video,
    gradient: 'from-orange-500 to-red-500',
    credits: 5,
    features: ['Pexels Library', 'Auto-Edit', 'Add Voiceover'],
  },
  {
    id: 'video-upload' as ContentType,
    title: 'Upload Video',
    description: 'Upload your own video and enhance it',
    icon: Upload,
    gradient: 'from-blue-500 to-cyan-500',
    credits: 0,
    features: ['MP4/MOV/WEBM', 'Add Captions', 'Add Music'],
  },
  {
    id: 'image' as ContentType,
    title: 'AI Image',
    description: 'Generate stunning images with DALL-E 3',
    icon: ImageIcon,
    gradient: 'from-purple-500 to-pink-500',
    credits: 1,
    features: ['DALL-E 3', 'Multiple Styles', 'HD Quality'],
  },
  {
    id: 'image-text' as ContentType,
    title: 'Image + Text',
    description: 'Create images with text overlays',
    icon: Type,
    gradient: 'from-green-500 to-emerald-500',
    credits: 1,
    features: ['Text Overlay', 'Quote Cards', 'Announcements'],
  },
  {
    id: 'carousel' as ContentType,
    title: 'Carousel',
    description: 'Create a multi-image carousel post',
    icon: Layers,
    gradient: 'from-indigo-500 to-purple-500',
    credits: 3,
    features: ['Multiple Images', 'Slide Format', 'Story Mode'],
  },
];

const imageStyles = [
  { id: 'photorealistic', name: 'Realistic', icon: '📷' },
  { id: 'digital-art', name: 'Digital Art', icon: '🎨' },
  { id: 'anime', name: 'Anime', icon: '🎌' },
  { id: 'cinematic', name: 'Cinematic', icon: '🎬' },
  { id: 'minimalist', name: 'Minimalist', icon: '⬜' },
  { id: '3d-render', name: '3D Render', icon: '🧊' },
];

const voiceOptions = [
  { id: 'rachel', name: 'Rachel', gender: 'Female', accent: 'American' },
  { id: 'josh', name: 'Josh', gender: 'Male', accent: 'American' },
  { id: 'bella', name: 'Bella', gender: 'Female', accent: 'British' },
  { id: 'adam', name: 'Adam', gender: 'Male', accent: 'American' },
  { id: 'nova', name: 'Nova', gender: 'Female', accent: 'Neutral', provider: 'openai' },
  { id: 'onyx', name: 'Onyx', gender: 'Male', accent: 'Deep', provider: 'openai' },
];

const platforms = [
  { id: 'TIKTOK', name: 'TikTok', color: 'bg-black' },
  { id: 'INSTAGRAM', name: 'Instagram', color: 'bg-gradient-to-r from-purple-500 to-pink-500' },
  { id: 'YOUTUBE', name: 'YouTube Shorts', color: 'bg-red-600' },
];

export default function CreatePostPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Step management
  const [currentStep, setCurrentStep] = useState<Step>(1);
  const [contentType, setContentType] = useState<ContentType | null>(null);

  // Media assets
  const [mediaAssets, setMediaAssets] = useState<MediaAsset[]>([]);
  const [textOverlays, setTextOverlays] = useState<TextOverlay[]>([]);
  const [voiceoverText, setVoiceoverText] = useState('');
  const [selectedVoice, setSelectedVoice] = useState('rachel');
  const [hasVoiceover, setHasVoiceover] = useState(false);

  // Generation states
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationProgress, setGenerationProgress] = useState(0);
  const [generationStatus, setGenerationStatus] = useState('');

  // Form data
  const [prompt, setPrompt] = useState('');
  const [imageStyle, setImageStyle] = useState('photorealistic');
  const [videoSearchQuery, setVideoSearchQuery] = useState('');
  const [stockVideos, setStockVideos] = useState<StockVideo[]>([]);
  const [selectedStockVideos, setSelectedStockVideos] = useState<StockVideo[]>([]);

  // Caption & hashtags
  const [caption, setCaption] = useState('');
  const [hashtags, setHashtags] = useState<string[]>([]);
  const [isGeneratingCaption, setIsGeneratingCaption] = useState(false);

  // Schedule
  const [targetPlatforms, setTargetPlatforms] = useState<string[]>(['TIKTOK']);
  const [scheduledDate, setScheduledDate] = useState('');
  const [scheduledTime, setScheduledTime] = useState('');
  const [isPublishing, setIsPublishing] = useState(false);

  // Upload state
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);

  // Preview
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(true);

  // Carousel state
  const [carouselPrompts, setCarouselPrompts] = useState<string[]>(['']);

  // AI Video workflow state
  const [aiVideoStep, setAiVideoStep] = useState<0 | 1 | 2 | 3 | 4 | 5 | 6>(0); // 0 = profile selection
  const [creatorProfiles, setCreatorProfiles] = useState<CreatorProfile[]>([]);
  const [selectedProfile, setSelectedProfile] = useState<CreatorProfile | null>(null);
  const [videoTopic, setVideoTopic] = useState('');
  const [scriptScenes, setScriptScenes] = useState<ScriptScene[]>([]);
  const [videoDuration, setVideoDuration] = useState<15 | 30 | 45 | 60>(30);
  const [videoStyle, setVideoStyle] = useState('dynamic');
  const [isGeneratingScript, setIsGeneratingScript] = useState(false);
  const [isGeneratingVideo, setIsGeneratingVideo] = useState(false);
  const [videoGenerationProgress, setVideoGenerationProgress] = useState(0);
  const [activeSceneIndex, setActiveSceneIndex] = useState<number | null>(null);

  // Load creator profiles on mount
  useEffect(() => {
    const profiles = getCreatorProfiles();
    setCreatorProfiles(profiles);
    const defaultProfile = profiles.find(p => p.isDefault) || profiles[0];
    if (defaultProfile) {
      setSelectedProfile(defaultProfile);
    }
  }, []);

  // Step navigation
  const canProceed = () => {
    switch (currentStep) {
      case 1:
        return contentType !== null;
      case 2:
        return mediaAssets.length > 0;
      case 3:
        return caption.trim().length > 0;
      case 4:
        return targetPlatforms.length > 0 && scheduledDate && scheduledTime;
      default:
        return false;
    }
  };

  const nextStep = () => {
    if (canProceed() && currentStep < 4) {
      setCurrentStep((currentStep + 1) as Step);
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep((currentStep - 1) as Step);
    }
  };

  // ============ STEP 2: Media Creation Functions ============

  // Generate AI Image
  const handleGenerateImage = async () => {
    if (!prompt.trim()) {
      toast.error('Please enter a prompt');
      return;
    }

    setIsGenerating(true);
    setGenerationStatus('Starting image generation...');
    setGenerationProgress(10);

    try {
      const response = await api.media.generateImage({
        prompt,
        style: imageStyle,
        preset: 'post',
        quality: 'hd',
      });

      setGenerationProgress(30);
      setGenerationStatus('AI is creating your image...');

      // Poll for completion
      const jobId = response.job_id;
      let attempts = 0;

      const pollJob = async (): Promise<any> => {
        const status = await api.media.getJobStatus(jobId);
        setGenerationProgress(30 + (attempts * 5));

        if (status.status === 'completed') {
          return status;
        } else if (status.status === 'failed') {
          throw new Error(status.error_message || 'Generation failed');
        }

        attempts++;
        if (attempts > 30) throw new Error('Generation timed out');

        await new Promise(resolve => setTimeout(resolve, 1000));
        return pollJob();
      };

      const result = await pollJob();
      setGenerationProgress(100);

      const imageUrl = result.output_data?.url || result.output_data?.image_url;

      if (imageUrl) {
        const newAsset: MediaAsset = {
          id: `img-${Date.now()}`,
          type: 'image',
          url: imageUrl,
          prompt: result.output_data?.revised_prompt || prompt,
        };

        if (contentType === 'carousel') {
          setMediaAssets(prev => [...prev, newAsset]);
        } else {
          setMediaAssets([newAsset]);
        }

        toast.success('Image generated!');
      }
    } catch (error: any) {
      toast.error(error.message || 'Failed to generate image');
    } finally {
      setIsGenerating(false);
      setGenerationProgress(0);
      setGenerationStatus('');
    }
  };

  // Generate voiceover
  const handleGenerateVoiceover = async () => {
    if (!voiceoverText.trim()) {
      toast.error('Please enter text for voiceover');
      return;
    }

    setIsGenerating(true);
    setGenerationStatus('Generating voiceover...');

    try {
      const response = await api.media.generateVoiceover({
        text: voiceoverText,
        voice: selectedVoice,
      });

      // Poll for completion
      const jobId = response.job_id;
      let attempts = 0;

      const pollJob = async (): Promise<any> => {
        const status = await api.media.getJobStatus(jobId);
        if (status.status === 'completed') return status;
        if (status.status === 'failed') throw new Error(status.error_message);
        attempts++;
        if (attempts > 60) throw new Error('Timed out');
        await new Promise(resolve => setTimeout(resolve, 1000));
        return pollJob();
      };

      const result = await pollJob();
      const audioUrl = result.output_data?.url;

      if (audioUrl) {
        const voiceoverAsset: MediaAsset = {
          id: `vo-${Date.now()}`,
          type: 'voiceover',
          url: audioUrl,
          duration: result.output_data?.duration,
        };
        setMediaAssets(prev => [...prev, voiceoverAsset]);
        setHasVoiceover(true);
        toast.success('Voiceover generated!');
      }
    } catch (error: any) {
      toast.error(error.message || 'Failed to generate voiceover');
    } finally {
      setIsGenerating(false);
      setGenerationStatus('');
    }
  };

  // Search stock videos
  const handleSearchStockVideos = async () => {
    if (!videoSearchQuery.trim()) {
      toast.error('Please enter a search query');
      return;
    }

    setIsGenerating(true);
    try {
      const response = await api.media.searchStockVideos(videoSearchQuery, {
        orientation: 'portrait',
        per_page: 12,
      });

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
      toast.success(`Found ${videos.length} videos`);
    } catch (error) {
      toast.error('Failed to search videos');
    } finally {
      setIsGenerating(false);
    }
  };

  // Add stock video to selection
  const handleAddStockVideo = (video: StockVideo) => {
    if (selectedStockVideos.find(v => v.id === video.id)) {
      setSelectedStockVideos(prev => prev.filter(v => v.id !== video.id));
    } else {
      setSelectedStockVideos(prev => [...prev, video]);
    }
  };

  // Compose video from stock footage
  const handleComposeVideo = async () => {
    if (selectedStockVideos.length === 0) {
      toast.error('Please select at least one video');
      return;
    }

    setIsGenerating(true);
    setGenerationStatus('Composing video...');

    try {
      const videoUrls = selectedStockVideos.map(v => v.url);

      const response = await api.media.composeVideo({
        video_urls: videoUrls,
        output_format: '9:16',
        quality: '1080p',
      });

      // Poll for completion
      const jobId = response.job_id;
      const pollJob = async (): Promise<any> => {
        const status = await api.media.getJobStatus(jobId);
        if (status.status === 'completed') return status;
        if (status.status === 'failed') throw new Error(status.error_message);
        await new Promise(resolve => setTimeout(resolve, 2000));
        return pollJob();
      };

      const result = await pollJob();

      if (result.output_data?.url) {
        const videoAsset: MediaAsset = {
          id: `vid-${Date.now()}`,
          type: 'video',
          url: result.output_data.url,
          duration: selectedStockVideos.reduce((acc, v) => acc + v.duration, 0),
        };
        setMediaAssets([videoAsset]);
        toast.success('Video composed!');
      }
    } catch (error: any) {
      toast.error(error.message || 'Failed to compose video');
    } finally {
      setIsGenerating(false);
      setGenerationStatus('');
    }
  };

  // Handle file upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const isVideo = file.type.startsWith('video/');
    const isImage = file.type.startsWith('image/');

    if (!isVideo && !isImage) {
      toast.error('Please select a video or image file');
      return;
    }

    if (file.size > 500 * 1024 * 1024) {
      toast.error('File must be less than 500MB');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    try {
      const url = URL.createObjectURL(file);

      // Simulate upload progress
      const interval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            return 100;
          }
          return prev + 20;
        });
      }, 200);

      await new Promise(resolve => setTimeout(resolve, 1000));

      let duration = 0;
      let thumbnailUrl = url;

      if (isVideo) {
        // Get video duration
        const video = document.createElement('video');
        video.src = url;
        await new Promise<void>((resolve) => {
          video.onloadedmetadata = () => {
            duration = video.duration;
            // Generate thumbnail
            video.currentTime = 1;
            video.onseeked = () => {
              const canvas = document.createElement('canvas');
              canvas.width = video.videoWidth;
              canvas.height = video.videoHeight;
              const ctx = canvas.getContext('2d');
              ctx?.drawImage(video, 0, 0);
              thumbnailUrl = canvas.toDataURL('image/jpeg');
              resolve();
            };
          };
        });
      }

      const newAsset: MediaAsset = {
        id: `upload-${Date.now()}`,
        type: isVideo ? 'video' : 'image',
        url,
        thumbnailUrl,
        duration,
      };

      setMediaAssets([newAsset]);
      toast.success(`${isVideo ? 'Video' : 'Image'} uploaded!`);
    } catch (error) {
      toast.error('Failed to process file');
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  // ============ AI VIDEO WORKFLOW FUNCTIONS ============

  // Generate structured script with Time, Visual, Audio format
  const handleGenerateScript = async () => {
    if (!videoTopic.trim()) {
      toast.error('Please enter a topic for your video');
      return;
    }

    if (!selectedProfile) {
      toast.error('Please select a creator profile first');
      setAiVideoStep(0);
      return;
    }

    setIsGeneratingScript(true);
    try {
      // Build system prompt from selected profile
      const systemPrompt = buildSystemPrompt(selectedProfile, videoTopic, videoDuration);

      // Try to get structured script from API
      const result = await api.content.generateScript({
        topic: videoTopic,
        duration_seconds: videoDuration,
        style: selectedProfile.tone,
        platform: 'tiktok',
        format: 'structured',
        system_prompt: systemPrompt,
        profile: {
          brandName: selectedProfile.brandName,
          niche: selectedProfile.niche,
          tone: selectedProfile.tone,
          hookStyle: selectedProfile.hookStyle,
          audienceLevel: selectedProfile.audienceLevel,
        },
      });

      // If API returns structured scenes, use them with proper typing
      if (result.scenes && Array.isArray(result.scenes)) {
        const apiScenes: ScriptScene[] = result.scenes.map((scene: any) => ({
          time: scene.time || '0:00-0:05',
          visual: scene.visual || '',
          audio: scene.audio || '',
          visualType: (scene.visualType as VisualType) || 'image',
          visualPrompt: scene.visualPrompt || extractImagePrompt(scene.visual || ''),
          diagramDescription: scene.diagramDescription,
        }));
        setScriptScenes(apiScenes);

        // Auto-generate media for all scenes
        setTimeout(() => {
          generateAllMedia(apiScenes);
        }, 500);
      } else {
        // Generate demo structured script using profile
        generateDemoStructuredScript();
      }

      setAiVideoStep(2);
      toast.success('Script generated!');
    } catch (error: any) {
      console.error('Script generation error:', error);
      // Fallback to demo structured script using profile
      generateDemoStructuredScript();
      setAiVideoStep(2);
      toast.success('Script generated! (Demo)');
    } finally {
      setIsGeneratingScript(false);
    }
  };

  // Classify visual description to determine media type
  const classifyVisualType = (visual: string): { type: VisualType; prompt: string; diagramDesc?: string } => {
    const lowerVisual = visual.toLowerCase();

    // Diagram keywords
    const diagramKeywords = [
      'diagram', 'flowchart', 'flow chart', 'architecture', 'pipeline',
      'process flow', 'workflow', 'schema', 'data flow', 'sequence',
      'hierarchy', 'tree structure', 'org chart', 'network diagram',
      'er diagram', 'class diagram', 'state diagram', 'system design',
      'infographic', 'chart showing', 'visual representation of steps',
      'arrows showing', 'boxes connected', 'nodes and edges'
    ];

    // Video keywords
    const videoKeywords = [
      'video of', 'footage of', 'clip of', 'motion', 'animation',
      'b-roll', 'b roll', 'timelapse', 'time lapse', 'screen recording',
      'demo video', 'walkthrough video', 'tutorial footage',
      'show video', 'play video', 'moving', 'action shot'
    ];

    // Text overlay only (no real image needed)
    const textOnlyKeywords = [
      'text overlay only', 'just text', 'text on screen', 'title card',
      'text animation', 'kinetic typography', 'words on screen'
    ];

    // Check for diagram
    if (diagramKeywords.some(kw => lowerVisual.includes(kw))) {
      // Extract diagram description
      const diagramDesc = extractDiagramDescription(visual);
      return {
        type: 'diagram',
        prompt: `Technical diagram: ${visual}`,
        diagramDesc
      };
    }

    // Check for video
    if (videoKeywords.some(kw => lowerVisual.includes(kw))) {
      return {
        type: 'video',
        prompt: extractVideoSearchTerms(visual)
      };
    }

    // Check for text only
    if (textOnlyKeywords.some(kw => lowerVisual.includes(kw))) {
      return {
        type: 'text-overlay',
        prompt: visual
      };
    }

    // Default to image - extract image generation prompt
    return {
      type: 'image',
      prompt: extractImagePrompt(visual)
    };
  };

  // Extract optimized prompt for image generation
  const extractImagePrompt = (visual: string): string => {
    let prompt = visual;

    // Step 1: Remove meta-instructions and labels
    prompt = prompt
      .replace(/^(Hook:|CTA:|Visual:|Scene:|Image:)\s*/gi, '')
      .replace(/Text overlay[:\s]*["'"].*?["'"]/gi, '') // Remove text overlay instructions
      .replace(/\(.*?text.*?\)/gi, '') // Remove parenthetical text instructions
      .trim();

    // Step 2: Extract the core visual description
    // Keep the full description if it describes a person/action
    const hasPersonAction = /\b(you|person|someone|man|woman|hand|holding|showing|pointing|standing|sitting|looking|wearing)\b/i.test(prompt);

    if (hasPersonAction) {
      // Clean up but keep the full action description
      prompt = prompt
        .replace(/\.\s*(Text|Add|Include|Show text).*$/gi, '') // Remove trailing text instructions
        .replace(/["'"]/g, '') // Remove quotes
        .trim();
    } else {
      // For non-person visuals, try to extract the main subject
      const subjectPatterns = [
        /(?:show|display|image of|photo of|picture of)\s+(.+?)(?:\.|$)/i,
        /^(.+?)(?:with text|text overlay|\.)/i,
      ];

      for (const pattern of subjectPatterns) {
        const match = prompt.match(pattern);
        if (match && match[1]) {
          prompt = match[1].trim();
          break;
        }
      }
    }

    // Step 3: Ensure prompt is substantial
    if (prompt.length < 20) {
      // If too short, use the original cleaned visual
      prompt = visual
        .replace(/^(Hook:|CTA:|Visual:|Scene:)\s*/gi, '')
        .replace(/Text overlay.*$/gi, '')
        .trim();
    }

    // Step 4: Add professional styling for DALL-E
    const styleGuide = [
      "high quality photograph",
      "professional lighting",
      "social media style",
      "vibrant colors",
      "9:16 vertical format",
      "sharp focus",
      "modern aesthetic"
    ].join(", ");

    return `${prompt}. ${styleGuide}`;
  };

  // Extract video search terms
  const extractVideoSearchTerms = (visual: string): string => {
    return visual
      .replace(/^(Hook:|CTA:|Visual:|Scene:)\s*/i, '')
      .replace(/Text overlay:.*?["""]/gi, '')
      .replace(/["""]/g, '')
      .replace(/b-roll|footage|video|clip/gi, '')
      .trim()
      .split(/[,.]/)
      .map(s => s.trim())
      .filter(s => s.length > 3)
      .slice(0, 3)
      .join(' ');
  };

  // Extract diagram description for generation
  const extractDiagramDescription = (visual: string): string => {
    // Parse the visual to create a structured diagram description
    let desc = visual
      .replace(/^(Hook:|CTA:|Visual:|Scene:)\s*/i, '')
      .replace(/Text overlay:.*?["""]/gi, '')
      .trim();

    // Create a Mermaid-compatible description hint
    return `Create a clear, professional diagram showing: ${desc}. Use clean lines, labeled boxes/nodes, and directional arrows where applicable.`;
  };

  // Generate media for a scene based on its type
  const generateMediaForScene = async (sceneIndex: number, scene: ScriptScene) => {
    setScriptScenes(prev => prev.map((s, i) =>
      i === sceneIndex ? { ...s, isGeneratingMedia: true } : s
    ));

    try {
      let mediaUrl = '';
      const visualType = scene.visualType || 'image';

      switch (visualType) {
        case 'image':
          // Generate image using AI
          mediaUrl = await generateImageForScene(scene.visualPrompt || scene.visual);
          break;

        case 'diagram':
          // Generate diagram
          mediaUrl = await generateDiagramForScene(scene.diagramDescription || scene.visual);
          break;

        case 'video':
          // Search for stock video
          const videos = await searchVideoForScene(scene.visualPrompt || scene.visual);
          if (videos.length > 0) {
            setScriptScenes(prev => prev.map((s, i) =>
              i === sceneIndex ? {
                ...s,
                mediaOptions: videos,
                isGeneratingMedia: false,
                generatedMediaUrl: videos[0].preview_url
              } : s
            ));
            return;
          }
          break;

        case 'text-overlay':
          // For text overlay, we'll generate a background
          mediaUrl = await generateTextOverlayBackground(scene.visual);
          break;
      }

      setScriptScenes(prev => prev.map((s, i) =>
        i === sceneIndex ? { ...s, generatedMediaUrl: mediaUrl, isGeneratingMedia: false } : s
      ));

    } catch (error) {
      console.error('Media generation error:', error);
      setScriptScenes(prev => prev.map((s, i) =>
        i === sceneIndex ? { ...s, isGeneratingMedia: false } : s
      ));
      toast.error(`Failed to generate media for scene ${sceneIndex + 1}`);
    }
  };

  // Generate image using API
  const generateImageForScene = async (prompt: string): Promise<string> => {
    try {
      const response = await api.media.generateImage({
        prompt,
        style: 'realistic',
        aspect_ratio: '9:16',
      });
      return response.url || response.image_url || '';
    } catch (error) {
      console.error('Image generation error:', error);
      // Return placeholder for demo
      return `https://placehold.co/1080x1920/1a1a2e/ffffff?text=${encodeURIComponent('Generated Image')}`;
    }
  };

  // Generate diagram using API or Mermaid
  const generateDiagramForScene = async (description: string): Promise<string> => {
    try {
      // Try to generate via API
      const response = await api.media.generateDiagram({
        description,
        style: 'modern',
        format: 'png',
      });
      return response.url || response.diagram_url || '';
    } catch (error) {
      console.error('Diagram generation error:', error);
      // Return placeholder for demo
      return `https://placehold.co/1080x1920/1a1a2e/ffffff?text=${encodeURIComponent('Diagram')}`;
    }
  };

  // Search for stock video
  const searchVideoForScene = async (searchTerms: string): Promise<StockVideo[]> => {
    try {
      const response = await api.media.searchStockVideos(searchTerms, {
        orientation: 'portrait',
        per_page: 6,
      });
      return (Array.isArray(response) ? response : []).map((v: any) => ({
        id: String(v.id),
        url: v.url,
        preview_url: v.preview_url,
        duration: v.duration,
        width: v.width,
        height: v.height,
        user: v.user,
      }));
    } catch (error) {
      console.error('Video search error:', error);
      return [];
    }
  };

  // Generate background for text overlay
  const generateTextOverlayBackground = async (text: string): Promise<string> => {
    // Generate a gradient or simple background
    return `https://placehold.co/1080x1920/667eea/ffffff?text=${encodeURIComponent('Text Overlay')}`;
  };

  // Auto-generate all media for scenes
  const generateAllMedia = async (scenes: ScriptScene[]) => {
    toast.loading('Generating media for all scenes...', { id: 'media-gen' });

    for (let i = 0; i < scenes.length; i++) {
      const scene = scenes[i];
      if (scene.visualType !== 'text-overlay') {
        await generateMediaForScene(i, scene);
        // Small delay between requests to avoid rate limiting
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    }

    toast.success('Media generation complete!', { id: 'media-gen' });
  };

  // Extract core topic from user input (remove "Generate a video on", "Create a tutorial about", etc.)
  const extractCoreTopic = (input: string): string => {
    let topic = input;

    // Remove common prefixes like "Generate a video on", "Create a tutorial about", etc.
    const prefixPatterns = [
      /^(generate|create|make|build|write|produce)\s+(a\s+)?(video|tutorial|guide|content|post)\s+(on|about|for|explaining|showing|covering)\s+(how\s+to\s+)?/i,
      /^(how\s+to\s+)(create|build|make|develop|implement)\s+(a\s+)?/i,
      /^(explain|show|teach|demonstrate)\s+(how\s+to\s+)?/i,
    ];

    for (const pattern of prefixPatterns) {
      topic = topic.replace(pattern, '');
    }

    // Clean up the result
    topic = topic.trim();

    // If still too long or looks like a sentence, try to extract key noun phrases
    if (topic.length > 50 || topic.split(' ').length > 8) {
      // Try to get the last meaningful part after common verbs
      const verbMatch = topic.match(/(?:build|create|implement|develop|use|set up|configure)\s+(.+?)(?:\s+with|\s+using|\s+in|\s+for|$)/i);
      if (verbMatch) {
        topic = verbMatch[1];
      }
    }

    // Capitalize first letter
    return topic.charAt(0).toUpperCase() + topic.slice(1);
  };

  // Generate demo structured script based on topic, duration, and profile
  const generateDemoStructuredScript = () => {
    const sceneDuration = videoDuration <= 30 ? 5 : videoDuration <= 45 ? 7 : 10;
    const numScenes = Math.ceil(videoDuration / sceneDuration);
    const profile = selectedProfile;

    // Extract the core topic for use in templates
    const coreTopic = extractCoreTopic(videoTopic);

    const demoScenes: ScriptScene[] = [];

    // Hook templates based on profile hookStyle - using coreTopic instead of videoTopic
    const hookTemplates: Record<string, { visual: string; audio: string }> = {
      'question': {
        visual: `Person looking directly at camera with curious, engaging expression. Clean background with subtle tech elements.`,
        audio: `What if I told you everything you know about ${coreTopic} is wrong?`,
      },
      'controversial': {
        visual: `Confident person pointing at camera with bold expression. Dynamic lighting, professional setting.`,
        audio: `Hot take: Most advice about ${coreTopic} is completely outdated.`,
      },
      'statistic': {
        visual: `Bold infographic showing "90%" statistic with eye-catching design. Person gesturing towards the number.`,
        audio: `90% of people make this mistake with ${coreTopic}. Here's what they're missing.`,
      },
      'story': {
        visual: `Person in casual setting, leaning in as if sharing a secret. Warm, inviting atmosphere.`,
        audio: `Last week, I discovered something about ${coreTopic} that changed everything.`,
      },
      'pattern-interrupt': {
        visual: `Person holding up hand in "STOP" gesture or holding a red stop sign. Urgent, attention-grabbing expression.`,
        audio: `STOP! Before you scroll, you need to hear this about ${coreTopic}.`,
      },
      'curiosity-gap': {
        visual: `Person with mysterious expression, leaning towards camera. Dramatic lighting creating intrigue.`,
        audio: `There's one thing about ${coreTopic} that nobody tells you...`,
      },
      'bold-claim': {
        visual: `Energetic person with confident pose. Bold text graphics appearing. High-energy setting.`,
        audio: `This is the ONLY method for ${coreTopic} that actually works in ${new Date().getFullYear()}.`,
      },
    };

    // Get hook based on profile or default
    const hookStyle = profile?.hookStyle || 'pattern-interrupt';
    const hook = hookTemplates[hookStyle] || hookTemplates['pattern-interrupt'];

    // Classify and add hook scene
    const hookClassification = classifyVisualType(hook.visual);
    demoScenes.push({
      time: '0:00-0:05',
      visual: hook.visual,
      audio: hook.audio,
      visualType: hookClassification.type,
      visualPrompt: hookClassification.prompt,
      diagramDescription: hookClassification.diagramDesc,
    });

    // Content scenes - adapted to profile tone and using coreTopic
    const tonePrefix = profile?.tone === 'humorous' ? 'Here\'s the funny thing - ' :
                       profile?.tone === 'professional' ? 'The data shows that ' :
                       profile?.tone === 'motivational' ? 'Here\'s the truth - ' :
                       profile?.tone === 'provocative' ? 'Nobody wants to admit that ' : '';

    // Generate contextual visuals based on the niche
    const nicheContext = profile?.niche || 'Technology';
    const isCodeRelated = /code|programming|software|app|api|database|streaming/i.test(coreTopic);

    const contentPoints = [
      {
        visual: isCodeRelated
          ? `Split screen comparison: Left side shows messy, complicated code or architecture. Right side shows clean, elegant ${coreTopic} implementation. Modern IDE or diagram style.`
          : `Before and after comparison related to ${coreTopic}. Left side shows the problem, right side shows the solution. Professional ${nicheContext} setting.`,
        audio: `${tonePrefix}Here's what most people get wrong about ${coreTopic}...`,
      },
      {
        visual: isCodeRelated
          ? `Clean diagram or flowchart showing the key components of ${coreTopic}. Arrows connecting different parts. Modern tech aesthetic with dark theme.`
          : `Person pointing at key information displayed on screen. Bold text overlays highlighting the main concept. Professional lighting.`,
        audio: `The secret that ${profile?.audienceDescription || 'developers'} need to understand is actually quite simple.`,
      },
      {
        visual: isCodeRelated
          ? `Screen recording style visual showing ${coreTopic} in action. Code editor or terminal with real examples. Highlighted sections showing key parts.`
          : `Demonstration of the concept with real examples. B-roll footage of the process. Step by step visualization.`,
        audio: `Once you understand this, everything about ${coreTopic} changes.`,
      },
      {
        visual: isCodeRelated
          ? `Dashboard or metrics showing successful ${coreTopic} implementation. Green checkmarks, performance graphs, success indicators.`
          : `Results and transformation visualization. Success metrics, happy outcomes, achievement graphics.`,
        audio: `This is what happens when you apply this correctly.`,
      },
    ];

    let currentTime = 5;
    for (let i = 0; i < Math.min(numScenes - 2, contentPoints.length); i++) {
      const endTime = currentTime + sceneDuration;
      const contentClassification = classifyVisualType(contentPoints[i].visual);
      demoScenes.push({
        time: `0:${currentTime.toString().padStart(2, '0')}-0:${endTime.toString().padStart(2, '0')}`,
        visual: contentPoints[i].visual,
        audio: contentPoints[i].audio,
        visualType: contentClassification.type,
        visualPrompt: contentClassification.prompt,
        diagramDescription: contentClassification.diagramDesc,
      });
      currentTime = endTime;
    }

    // CTA scene - use profile's CTA style if available
    const ctaText = profile?.ctaStyle || 'Follow for more tips like this!';
    const ctaVisual = `CTA: Text overlay "${ctaText}" Point to follow button or use engaging gesture.`;
    const ctaClassification = classifyVisualType(ctaVisual);
    demoScenes.push({
      time: `0:${currentTime.toString().padStart(2, '0')}-0:${videoDuration.toString().padStart(2, '0')}`,
      visual: ctaVisual,
      audio: `${ctaText} Drop a comment if you want part 2.`,
      visualType: ctaClassification.type,
      visualPrompt: ctaClassification.prompt,
      diagramDescription: ctaClassification.diagramDesc,
    });

    setScriptScenes(demoScenes);

    // Auto-generate media for all scenes after a short delay
    setTimeout(() => {
      generateAllMedia(demoScenes);
    }, 500);
  };

  // Search media for a specific scene
  const handleSearchMediaForScene = async (sceneIndex: number) => {
    const scene = scriptScenes[sceneIndex];
    if (!scene) return;

    // Extract keywords from visual description
    const keywords = scene.visual
      .replace(/["""]/g, '')
      .replace(/Hook:|CTA:|Text:|Visual:/gi, '')
      .split(/[,.]/)
      .map(s => s.trim())
      .filter(s => s.length > 3)
      .slice(0, 2)
      .join(' ') || videoTopic;

    // Update scene to show searching state
    setScriptScenes(prev => prev.map((s, i) =>
      i === sceneIndex ? { ...s, isSearching: true } : s
    ));

    try {
      const response = await api.media.searchStockVideos(keywords, {
        orientation: 'portrait',
        per_page: 6,
      });

      const videos: StockVideo[] = (Array.isArray(response) ? response : []).map((v: any) => ({
        id: String(v.id),
        url: v.url,
        preview_url: v.preview_url,
        duration: v.duration,
        width: v.width,
        height: v.height,
        user: v.user,
      }));

      setScriptScenes(prev => prev.map((s, i) =>
        i === sceneIndex ? { ...s, mediaOptions: videos, isSearching: false } : s
      ));
      setActiveSceneIndex(sceneIndex);

      if (videos.length === 0) {
        toast.error('No videos found. Try different keywords.');
      }
    } catch (error) {
      setScriptScenes(prev => prev.map((s, i) =>
        i === sceneIndex ? { ...s, isSearching: false } : s
      ));
      toast.error('Failed to search media');
    }
  };

  // Select media for a scene
  const handleSelectMediaForScene = (sceneIndex: number, video: StockVideo) => {
    const mediaAsset: MediaAsset = {
      id: `scene-${sceneIndex}-${video.id}`,
      type: 'video',
      url: video.url,
      thumbnailUrl: video.preview_url,
      duration: video.duration,
    };

    setScriptScenes(prev => prev.map((s, i) =>
      i === sceneIndex ? { ...s, media: mediaAsset, mediaOptions: undefined } : s
    ));
    setActiveSceneIndex(null);
    toast.success(`Media selected for scene ${sceneIndex + 1}`);
  };

  // Check if all scenes have media
  const allScenesHaveMedia = () => {
    return scriptScenes.length > 0 && scriptScenes.every(scene => scene.media);
  };

  // Generate final AI Video from scenes
  const handleGenerateAIVideo = async () => {
    if (scriptScenes.length === 0) {
      toast.error('Please generate a script first');
      return;
    }

    // Check if all scenes have media (optional - can proceed without)
    const scenesWithMedia = scriptScenes.filter(s => s.media);
    if (scenesWithMedia.length === 0) {
      toast.error('Please select at least one media for your scenes');
      return;
    }

    setIsGeneratingVideo(true);
    setVideoGenerationProgress(0);

    try {
      // Step 1: Generate voiceover from all audio scripts
      setVideoGenerationProgress(10);
      toast('Generating voiceover...', { icon: '🎙️' });

      const fullScript = scriptScenes.map(s => s.audio).join(' ');

      const voiceoverResponse = await api.media.generateVoiceover({
        text: fullScript,
        voice: selectedVoice,
      });

      let voiceoverUrl = '';
      if (voiceoverResponse.job_id) {
        const voiceResult = await pollJobCompletion(voiceoverResponse.job_id);
        voiceoverUrl = voiceResult.output_data?.url || '';
      }
      setVideoGenerationProgress(40);

      // Step 2: Compose video with selected media
      toast('Composing your video...', { icon: '✨' });
      setVideoGenerationProgress(50);

      const videoUrls = scriptScenes
        .filter(s => s.media)
        .map(s => s.media!.url);

      const composeResponse = await api.media.composeVideo({
        video_urls: videoUrls,
        audio_url: voiceoverUrl || undefined,
        output_format: '9:16',
        quality: '1080p',
      });

      // Poll for video completion
      if (composeResponse.job_id) {
        setVideoGenerationProgress(70);
        const videoResult = await pollJobCompletion(composeResponse.job_id, 120);
        setVideoGenerationProgress(90);

        if (videoResult.output_data?.url) {
          const videoAsset: MediaAsset = {
            id: `ai-video-${Date.now()}`,
            type: 'video',
            url: videoResult.output_data.url,
            thumbnailUrl: videoResult.output_data.thumbnail_url,
            duration: videoDuration,
            prompt: fullScript,
          };
          setMediaAssets([videoAsset]);
          setVideoGenerationProgress(100);
          toast.success('AI Video created successfully!');
        }
      } else {
        // Demo mode - create a placeholder
        const demoAsset: MediaAsset = {
          id: `ai-video-demo-${Date.now()}`,
          type: 'video',
          url: 'https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4',
          duration: videoDuration,
          prompt: fullScript,
        };
        setMediaAssets([demoAsset]);
        setVideoGenerationProgress(100);
        toast.success('AI Video created! (Demo)');
      }

      setAiVideoStep(5);
    } catch (error: any) {
      console.error('AI Video generation error:', error);
      toast.error(error.message || 'Failed to generate video. Please try again.');
    } finally {
      setIsGeneratingVideo(false);
    }
  };

  // Helper function to poll job completion
  const pollJobCompletion = async (jobId: string, maxAttempts = 60): Promise<any> => {
    let attempts = 0;
    while (attempts < maxAttempts) {
      const status = await api.media.getJobStatus(jobId);
      if (status.status === 'completed') return status;
      if (status.status === 'failed') throw new Error(status.error_message || 'Job failed');
      attempts++;
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    throw new Error('Job timed out');
  };

  // Add text overlay
  const handleAddTextOverlay = () => {
    const newOverlay: TextOverlay = {
      id: `text-${Date.now()}`,
      text: '',
      position: 'center',
      style: 'bold',
    };
    setTextOverlays(prev => [...prev, newOverlay]);
  };

  // ============ STEP 3: Caption Functions ============

  const handleGenerateCaption = async () => {
    setIsGeneratingCaption(true);
    try {
      // Use the prompt or first asset's prompt for context
      const context = prompt || mediaAssets[0]?.prompt || 'viral social media content';

      const result = await api.content.generateCaption({
        script: context,
        max_length: 150,
        include_hashtags: true,
        include_cta: true,
      });

      setCaption(result.caption);
      setHashtags(result.hashtags);
      toast.success('Caption generated!');
    } catch (error) {
      toast.error('Failed to generate caption');
    } finally {
      setIsGeneratingCaption(false);
    }
  };

  const handleGenerateHashtags = async () => {
    try {
      const result = await api.content.generateHashtags(caption || prompt, '', 10);
      setHashtags(result.hashtags);
      toast.success('Hashtags generated!');
    } catch (error) {
      toast.error('Failed to generate hashtags');
    }
  };

  // ============ STEP 4: Publish Functions ============

  const handlePublish = async () => {
    if (!scheduledDate || !scheduledTime) {
      toast.error('Please select a date and time');
      return;
    }

    if (targetPlatforms.length === 0) {
      toast.error('Please select at least one platform');
      return;
    }

    setIsPublishing(true);

    try {
      const scheduledAt = new Date(`${scheduledDate}T${scheduledTime}`).toISOString();
      const mainAsset = mediaAssets.find(a => a.type === 'video' || a.type === 'image');

      await api.scheduler.createPost({
        title: caption.substring(0, 50) || 'Untitled Post',
        caption,
        hashtags,
        videoUrl: mainAsset?.url || '',
        thumbnailUrl: mainAsset?.thumbnailUrl,
        videoDurationSeconds: mainAsset?.duration || 0,
        scheduledAt,
        targetPlatforms,
      });

      toast.success('Post scheduled successfully!' + (DEMO_MODE ? ' (Demo)' : ''));
      router.push('/dashboard/scheduler');
    } catch (error) {
      toast.error('Failed to schedule post');
    } finally {
      setIsPublishing(false);
    }
  };

  // Toggle platform
  const togglePlatform = (platformId: string) => {
    setTargetPlatforms(prev =>
      prev.includes(platformId)
        ? prev.filter(p => p !== platformId)
        : [...prev, platformId]
    );
  };

  // Get min date
  const getMinDate = () => new Date().toISOString().split('T')[0];

  // Format duration
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Remove asset
  const removeAsset = (id: string) => {
    setMediaAssets(prev => prev.filter(a => a.id !== id));
  };

  return (
    <div className="min-h-screen bg-gray-900 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-5xl mx-auto">
        {/* Demo Mode Banner */}
        {DEMO_MODE && (
          <div className="mb-6 p-3 bg-yellow-500/20 border border-yellow-500/30 rounded-xl flex items-center gap-2">
            <Info className="w-5 h-5 text-yellow-400 flex-shrink-0" />
            <p className="text-sm text-yellow-200">
              <strong>Demo Mode:</strong> All features work with simulated data.
            </p>
          </div>
        )}

        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white flex items-center justify-center gap-3">
            <div className="p-2 bg-gradient-to-r from-purple-600 to-pink-600 rounded-xl">
              <Wand2 className="h-6 w-6 text-white" />
            </div>
            Create Post
          </h1>
          <p className="mt-2 text-gray-400">
            Create complete content from scratch, step by step
          </p>
        </div>

        {/* Progress Steps */}
        <div className="flex items-center justify-center mb-10">
          {[1, 2, 3, 4].map((step, idx) => (
            <div key={step} className="flex items-center">
              <button
                onClick={() => step < currentStep && setCurrentStep(step as Step)}
                disabled={step > currentStep}
                className={`relative w-12 h-12 rounded-full flex items-center justify-center font-semibold transition-all ${
                  currentStep === step
                    ? 'bg-purple-600 text-white scale-110'
                    : currentStep > step
                    ? 'bg-green-600 text-white cursor-pointer hover:bg-green-500'
                    : 'bg-gray-700 text-gray-400'
                }`}
              >
                {currentStep > step ? <Check className="w-5 h-5" /> : step}
                <span className="absolute -bottom-6 text-xs text-gray-400 whitespace-nowrap">
                  {step === 1 && 'Type'}
                  {step === 2 && 'Create'}
                  {step === 3 && 'Caption'}
                  {step === 4 && 'Publish'}
                </span>
              </button>
              {idx < 3 && (
                <div className={`w-16 sm:w-24 h-1 mx-2 rounded ${
                  currentStep > step ? 'bg-green-600' : 'bg-gray-700'
                }`} />
              )}
            </div>
          ))}
        </div>

        {/* Step Content */}
        <AnimatePresence mode="wait">
          {/* ============ STEP 1: Choose Content Type ============ */}
          {currentStep === 1 && (
            <motion.div
              key="step1"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-6"
            >
              <h2 className="text-xl font-semibold text-white text-center mb-6">
                What do you want to create?
              </h2>

              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                {contentTypes.map((type) => (
                  <button
                    key={type.id}
                    onClick={() => setContentType(type.id)}
                    className={`group p-5 rounded-2xl border-2 text-left transition-all ${
                      contentType === type.id
                        ? 'bg-purple-600/20 border-purple-500'
                        : 'bg-gray-800 border-gray-700 hover:border-gray-600'
                    }`}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className={`p-2.5 rounded-xl bg-gradient-to-r ${type.gradient}`}>
                        <type.icon className="w-5 h-5 text-white" />
                      </div>
                      <div className="flex items-center gap-1 px-2 py-1 bg-gray-700 rounded-full">
                        <Zap className="w-3 h-3 text-yellow-400" />
                        <span className="text-xs text-gray-300">{type.credits}</span>
                      </div>
                    </div>
                    <h3 className="text-lg font-semibold text-white mb-1">{type.title}</h3>
                    <p className="text-sm text-gray-400 mb-3">{type.description}</p>
                    <div className="flex flex-wrap gap-1.5">
                      {type.features.map((feature) => (
                        <span
                          key={feature}
                          className="px-2 py-0.5 bg-gray-700/50 rounded text-xs text-gray-300"
                        >
                          {feature}
                        </span>
                      ))}
                    </div>
                  </button>
                ))}
              </div>
            </motion.div>
          )}

          {/* ============ STEP 2: Create Media ============ */}
          {currentStep === 2 && (
            <motion.div
              key="step2"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-6"
            >
              <div className="grid lg:grid-cols-2 gap-6">
                {/* Left: Creation Panel */}
                <div className="space-y-6">
                  {/* AI Video Workflow */}
                  {contentType === 'video-ai' && (
                    <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                      {/* Step 0: Profile Selection */}
                      {aiVideoStep === 0 && (
                        <div className="space-y-4">
                          <div className="flex items-center justify-between">
                            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                              <User className="w-5 h-5 text-pink-400" />
                              Select Creator Profile
                            </h3>
                            <Link
                              href="/dashboard/settings/profiles"
                              className="text-sm text-pink-400 hover:text-pink-300 flex items-center gap-1"
                            >
                              <Settings className="w-4 h-4" />
                              Manage Profiles
                            </Link>
                          </div>
                          <p className="text-sm text-gray-400">
                            Choose the profile that matches your content style. This will personalize the AI-generated script.
                          </p>

                          {creatorProfiles.length === 0 ? (
                            <div className="text-center py-8">
                              <User className="w-12 h-12 mx-auto text-gray-600 mb-3" />
                              <p className="text-gray-400 mb-4">No profiles created yet</p>
                              <Link
                                href="/dashboard/settings/profiles"
                                className="inline-flex items-center gap-2 px-4 py-2 bg-pink-600 text-white rounded-xl hover:bg-pink-700 transition"
                              >
                                <Plus className="w-4 h-4" />
                                Create Your First Profile
                              </Link>
                            </div>
                          ) : (
                            <div className="space-y-3">
                              {creatorProfiles.map((profile) => (
                                <button
                                  key={profile.id}
                                  onClick={() => setSelectedProfile(profile)}
                                  className={`w-full text-left p-4 rounded-xl border transition ${
                                    selectedProfile?.id === profile.id
                                      ? 'bg-pink-600/20 border-pink-500'
                                      : 'bg-gray-700/50 border-gray-600 hover:border-gray-500'
                                  }`}
                                >
                                  <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                      <div className="flex items-center gap-2">
                                        <span className="font-medium text-white">{profile.name}</span>
                                        {profile.isDefault && (
                                          <span className="text-xs bg-pink-600/30 text-pink-300 px-2 py-0.5 rounded-full">
                                            Default
                                          </span>
                                        )}
                                      </div>
                                      <p className="text-sm text-gray-400 mt-1">{profile.brandName} - {profile.tagline}</p>
                                      <div className="flex items-center gap-2 mt-2">
                                        <span className="text-xs bg-gray-600 text-gray-300 px-2 py-0.5 rounded-full">
                                          {profile.niche}
                                        </span>
                                        <span className="text-xs bg-gray-600 text-gray-300 px-2 py-0.5 rounded-full">
                                          {TONE_OPTIONS.find(t => t.id === profile.tone)?.emoji} {profile.tone}
                                        </span>
                                      </div>
                                    </div>
                                    <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                                      selectedProfile?.id === profile.id
                                        ? 'border-pink-500 bg-pink-500'
                                        : 'border-gray-500'
                                    }`}>
                                      {selectedProfile?.id === profile.id && (
                                        <Check className="w-3 h-3 text-white" />
                                      )}
                                    </div>
                                  </div>
                                </button>
                              ))}
                            </div>
                          )}

                          {creatorProfiles.length > 0 && (
                            <button
                              onClick={() => setAiVideoStep(1)}
                              disabled={!selectedProfile}
                              className="w-full py-3 bg-gradient-to-r from-pink-600 to-rose-600 text-white font-semibold rounded-xl hover:from-pink-700 hover:to-rose-700 transition disabled:opacity-50 flex items-center justify-center gap-2"
                            >
                              Continue with {selectedProfile?.name || 'Selected Profile'}
                              <ArrowRight className="w-4 h-4" />
                            </button>
                          )}
                        </div>
                      )}

                      {/* AI Video Sub-steps indicator (shown after profile selection) */}
                      {aiVideoStep >= 1 && (
                        <div className="flex items-center gap-2 mb-6">
                          <button
                            onClick={() => setAiVideoStep(0)}
                            className="text-xs text-gray-400 hover:text-white flex items-center gap-1 mr-2"
                          >
                            <User className="w-3 h-3" />
                            {selectedProfile?.name}
                          </button>
                          <div className="h-4 w-px bg-gray-600 mr-2" />
                          {[1, 2, 3, 4, 5].map((step) => (
                            <div key={step} className="flex items-center">
                              <div
                                className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition ${
                                  aiVideoStep === step
                                    ? 'bg-pink-600 text-white'
                                    : aiVideoStep > step
                                    ? 'bg-green-600 text-white'
                                    : 'bg-gray-700 text-gray-400'
                                }`}
                              >
                                {aiVideoStep > step ? <Check className="w-4 h-4" /> : step}
                              </div>
                              {step < 5 && (
                                <div className={`w-6 h-0.5 ${aiVideoStep > step ? 'bg-green-600' : 'bg-gray-700'}`} />
                              )}
                            </div>
                          ))}
                          <span className="ml-3 text-sm text-gray-400">
                            {aiVideoStep === 1 && 'Topic'}
                            {aiVideoStep === 2 && 'Script'}
                            {aiVideoStep === 3 && 'Media'}
                            {aiVideoStep === 4 && 'Voice'}
                            {aiVideoStep === 5 && 'Done'}
                          </span>
                        </div>
                      )}

                      {/* Step 1: Topic Input */}
                      {aiVideoStep === 1 && (
                        <div className="space-y-4">
                          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                            <Target className="w-5 h-5 text-pink-400" />
                            What&apos;s your video about?
                          </h3>
                          <div>
                            <label className="block text-sm text-gray-400 mb-2">Video Topic</label>
                            <input
                              type="text"
                              value={videoTopic}
                              onChange={(e) => setVideoTopic(e.target.value)}
                              placeholder="e.g., Best software architecture for startups, How to choose tech stack..."
                              className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-pink-500"
                            />
                          </div>
                          <div>
                            <label className="block text-sm text-gray-400 mb-2">Video Duration</label>
                            <div className="grid grid-cols-4 gap-2">
                              {[15, 30, 45, 60].map((dur) => (
                                <button
                                  key={dur}
                                  onClick={() => setVideoDuration(dur as 15 | 30 | 45 | 60)}
                                  className={`py-3 rounded-xl border font-medium transition ${
                                    videoDuration === dur
                                      ? 'bg-pink-600/20 border-pink-500 text-pink-300'
                                      : 'bg-gray-700 border-gray-600 text-gray-300 hover:border-gray-500'
                                  }`}
                                >
                                  {dur}s
                                </button>
                              ))}
                            </div>
                          </div>
                          <button
                            onClick={handleGenerateScript}
                            disabled={isGeneratingScript || !videoTopic.trim()}
                            className="w-full py-3 bg-gradient-to-r from-pink-600 to-rose-600 text-white font-semibold rounded-xl hover:from-pink-700 hover:to-rose-700 transition disabled:opacity-50 flex items-center justify-center gap-2"
                          >
                            {isGeneratingScript ? (
                              <>
                                <Loader2 className="w-5 h-5 animate-spin" />
                                Generating Script...
                              </>
                            ) : (
                              <>
                                <Sparkles className="w-5 h-5" />
                                Generate Script
                              </>
                            )}
                          </button>
                        </div>
                      )}

                      {/* Step 2: Structured Script Table */}
                      {aiVideoStep === 2 && (
                        <div className="space-y-4">
                          <div className="flex items-center justify-between">
                            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                              <FileText className="w-5 h-5 text-pink-400" />
                              Video Script
                            </h3>
                            <button
                              onClick={() => setAiVideoStep(1)}
                              className="text-sm text-gray-400 hover:text-white"
                            >
                              ← Back
                            </button>
                          </div>

                          {/* Script Table with Media Preview */}
                          {scriptScenes.length === 0 ? (
                            <div className="text-center py-8 text-gray-400">
                              <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
                              <p>No script generated yet. Click &quot;Generate Script&quot; to start.</p>
                            </div>
                          ) : (
                          <div className="overflow-x-auto rounded-xl border border-gray-700">
                            <table className="w-full text-sm min-w-[800px]">
                              <thead className="bg-gray-700/50">
                                <tr>
                                  <th className="px-2 py-2 text-left text-gray-300 font-medium w-16">Time</th>
                                  <th className="px-2 py-2 text-left text-gray-300 font-medium w-20">Type</th>
                                  <th className="px-2 py-2 text-left text-gray-300 font-medium w-1/3">Visual</th>
                                  <th className="px-2 py-2 text-left text-gray-300 font-medium w-1/3">Audio (Script)</th>
                                  <th className="px-2 py-2 text-left text-gray-300 font-medium w-28">Media</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-gray-700">
                                {scriptScenes.map((scene, idx) => (
                                  <tr key={idx} className="bg-gray-800/50 hover:bg-gray-700/30">
                                    <td className="px-3 py-3 text-pink-400 font-mono text-xs align-top">
                                      {scene.time}
                                    </td>
                                    <td className="px-2 py-2 align-top">
                                      <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                                        scene.visualType === 'image' ? 'bg-blue-500/20 text-blue-300' :
                                        scene.visualType === 'diagram' ? 'bg-purple-500/20 text-purple-300' :
                                        scene.visualType === 'video' ? 'bg-green-500/20 text-green-300' :
                                        scene.visualType === 'text-overlay' ? 'bg-yellow-500/20 text-yellow-300' :
                                        'bg-gray-500/20 text-gray-300'
                                      }`}>
                                        {scene.visualType === 'image' && <ImageIcon className="w-3 h-3" />}
                                        {scene.visualType === 'diagram' && <Layers className="w-3 h-3" />}
                                        {scene.visualType === 'video' && <Video className="w-3 h-3" />}
                                        {scene.visualType === 'text-overlay' && <Type className="w-3 h-3" />}
                                        {!scene.visualType && <ImageIcon className="w-3 h-3" />}
                                        {scene.visualType || 'image'}
                                      </span>
                                    </td>
                                    <td className="px-3 py-3 text-gray-300 align-top">
                                      <textarea
                                        value={scene.visual}
                                        onChange={(e) => {
                                          const updated = [...scriptScenes];
                                          updated[idx].visual = e.target.value;
                                          // Re-classify on change
                                          const newClass = classifyVisualType(e.target.value);
                                          updated[idx].visualType = newClass.type;
                                          updated[idx].visualPrompt = newClass.prompt;
                                          updated[idx].diagramDescription = newClass.diagramDesc;
                                          setScriptScenes(updated);
                                        }}
                                        rows={2}
                                        className="w-full bg-transparent border-none text-gray-300 text-sm resize-none focus:outline-none focus:ring-1 focus:ring-pink-500 rounded p-1"
                                      />
                                    </td>
                                    <td className="px-3 py-3 text-white align-top">
                                      <textarea
                                        value={scene.audio}
                                        onChange={(e) => {
                                          const updated = [...scriptScenes];
                                          updated[idx].audio = e.target.value;
                                          setScriptScenes(updated);
                                        }}
                                        rows={2}
                                        className="w-full bg-transparent border-none text-white text-sm resize-none focus:outline-none focus:ring-1 focus:ring-pink-500 rounded p-1"
                                      />
                                    </td>
                                    <td className="px-2 py-2 align-top">
                                      <div className="relative w-20 h-28 rounded-lg overflow-hidden bg-gray-700/50 border border-gray-600">
                                        {scene.isGeneratingMedia ? (
                                          <div className="absolute inset-0 flex flex-col items-center justify-center">
                                            <Loader2 className="w-5 h-5 text-pink-400 animate-spin" />
                                            <span className="text-[10px] text-gray-400 mt-1">...</span>
                                          </div>
                                        ) : scene.generatedMediaUrl ? (
                                          <>
                                            {scene.visualType === 'video' ? (
                                              <video
                                                src={scene.generatedMediaUrl}
                                                className="w-full h-full object-cover"
                                                muted
                                                loop
                                                autoPlay
                                              />
                                            ) : (
                                              <img
                                                src={scene.generatedMediaUrl}
                                                alt={`Scene ${idx + 1}`}
                                                className="w-full h-full object-cover"
                                                onError={(e) => {
                                                  (e.target as HTMLImageElement).src = 'https://placehold.co/200x300/374151/9ca3af?text=Error';
                                                }}
                                              />
                                            )}
                                            <button
                                              onClick={() => generateMediaForScene(idx, scene)}
                                              className="absolute bottom-1 right-1 p-1 bg-gray-900/80 rounded-full hover:bg-gray-800 transition"
                                              title="Regenerate"
                                            >
                                              <RefreshCw className="w-3 h-3 text-white" />
                                            </button>
                                          </>
                                        ) : (
                                          <button
                                            onClick={() => generateMediaForScene(idx, scene)}
                                            className="absolute inset-0 flex flex-col items-center justify-center hover:bg-gray-600/50 transition"
                                          >
                                            <Sparkles className="w-4 h-4 text-pink-400 mb-1" />
                                            <span className="text-[10px] text-gray-400">Generate</span>
                                          </button>
                                        )}
                                      </div>
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                          )}

                          {/* Action buttons */}
                          <div className="flex gap-3">
                            <button
                              onClick={handleGenerateScript}
                              disabled={isGeneratingScript}
                              className="py-3 px-4 bg-gray-700 text-white rounded-xl hover:bg-gray-600 transition flex items-center justify-center gap-2"
                            >
                              <RefreshCw className="w-4 h-4" />
                              Regenerate Script
                            </button>
                            <button
                              onClick={() => generateAllMedia(scriptScenes)}
                              disabled={scriptScenes.length === 0}
                              className="py-3 px-4 bg-purple-600 text-white rounded-xl hover:bg-purple-700 transition flex items-center justify-center gap-2"
                            >
                              <Sparkles className="w-4 h-4" />
                              Generate All Media
                            </button>
                            <button
                              onClick={() => setAiVideoStep(3)}
                              disabled={scriptScenes.length === 0}
                              className="flex-1 py-3 bg-pink-600 text-white font-semibold rounded-xl hover:bg-pink-700 transition disabled:opacity-50 flex items-center justify-center gap-2"
                            >
                              Next: Review Media
                              <ChevronRight className="w-4 h-4" />
                            </button>
                          </div>
                        </div>
                      )}

                      {/* Step 3: Media Selection for Each Scene */}
                      {aiVideoStep === 3 && (
                        <div className="space-y-4">
                          <div className="flex items-center justify-between">
                            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                              <Video className="w-5 h-5 text-pink-400" />
                              Select Media for Each Scene
                            </h3>
                            <button
                              onClick={() => setAiVideoStep(2)}
                              className="text-sm text-gray-400 hover:text-white"
                            >
                              ← Back to script
                            </button>
                          </div>

                          <p className="text-sm text-gray-400">
                            Click &quot;Find Media&quot; to search for videos matching each visual description.
                          </p>

                          <div className="space-y-3 max-h-[400px] overflow-y-auto">
                            {scriptScenes.map((scene, idx) => (
                              <div key={idx} className="p-4 bg-gray-700/50 rounded-xl border border-gray-600">
                                <div className="flex items-start justify-between gap-4 mb-3">
                                  <div className="flex-1">
                                    <div className="flex items-center gap-2 mb-1">
                                      <span className="text-pink-400 font-mono text-xs">{scene.time}</span>
                                      {scene.media && (
                                        <span className="px-2 py-0.5 bg-green-600/20 text-green-400 text-xs rounded-full">
                                          Media selected
                                        </span>
                                      )}
                                    </div>
                                    <p className="text-gray-300 text-sm">{scene.visual}</p>
                                  </div>
                                  {scene.media ? (
                                    <div className="relative w-20 h-20 rounded-lg overflow-hidden flex-shrink-0">
                                      <img
                                        src={scene.media.thumbnailUrl || scene.media.url}
                                        alt="Selected"
                                        className="w-full h-full object-cover"
                                      />
                                      <button
                                        onClick={() => {
                                          setScriptScenes(prev => prev.map((s, i) =>
                                            i === idx ? { ...s, media: null } : s
                                          ));
                                        }}
                                        className="absolute top-1 right-1 p-1 bg-red-600 rounded-full"
                                      >
                                        <X className="w-3 h-3 text-white" />
                                      </button>
                                    </div>
                                  ) : (
                                    <button
                                      onClick={() => handleSearchMediaForScene(idx)}
                                      disabled={scene.isSearching}
                                      className="px-3 py-2 bg-pink-600 text-white text-sm rounded-lg hover:bg-pink-700 transition disabled:opacity-50 flex items-center gap-1 flex-shrink-0"
                                    >
                                      {scene.isSearching ? (
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                      ) : (
                                        <>
                                          <ImageIcon className="w-4 h-4" />
                                          Find Media
                                        </>
                                      )}
                                    </button>
                                  )}
                                </div>

                                {/* Media Options Grid */}
                                {scene.mediaOptions && scene.mediaOptions.length > 0 && (
                                  <div className="grid grid-cols-3 gap-2 mt-3 pt-3 border-t border-gray-600">
                                    {scene.mediaOptions.map((video) => (
                                      <button
                                        key={video.id}
                                        onClick={() => handleSelectMediaForScene(idx, video)}
                                        className="relative rounded-lg overflow-hidden aspect-video border-2 border-transparent hover:border-pink-500 transition"
                                      >
                                        <img
                                          src={video.preview_url}
                                          alt={`Video by ${video.user}`}
                                          className="w-full h-full object-cover"
                                        />
                                        <div className="absolute bottom-0 left-0 right-0 p-1 bg-black/60">
                                          <span className="text-white text-xs">{Math.round(video.duration)}s</span>
                                        </div>
                                      </button>
                                    ))}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>

                          <div className="flex gap-3">
                            <button
                              onClick={() => setAiVideoStep(4)}
                              className="flex-1 py-3 bg-pink-600 text-white font-semibold rounded-xl hover:bg-pink-700 transition flex items-center justify-center gap-2"
                            >
                              Next: Select Voice
                              <ChevronRight className="w-4 h-4" />
                            </button>
                          </div>

                          <p className="text-xs text-gray-500 text-center">
                            {scriptScenes.filter(s => s.media).length} / {scriptScenes.length} scenes have media selected
                          </p>
                        </div>
                      )}

                      {/* Step 4: Voice Selection & Generate */}
                      {aiVideoStep === 4 && (
                        <div className="space-y-4">
                          <div className="flex items-center justify-between">
                            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                              <Mic className="w-5 h-5 text-pink-400" />
                              Select Voice & Generate
                            </h3>
                            <button
                              onClick={() => setAiVideoStep(3)}
                              className="text-sm text-gray-400 hover:text-white"
                            >
                              ← Back to media
                            </button>
                          </div>

                          {/* Voice Selection */}
                          <div>
                            <label className="block text-sm text-gray-400 mb-2">Narrator Voice</label>
                            <div className="grid grid-cols-3 gap-2">
                              {voiceOptions.map((voice) => (
                                <button
                                  key={voice.id}
                                  onClick={() => setSelectedVoice(voice.id)}
                                  className={`p-3 rounded-xl border text-center transition ${
                                    selectedVoice === voice.id
                                      ? 'bg-pink-600/20 border-pink-500'
                                      : 'bg-gray-700 border-gray-600 hover:border-gray-500'
                                  }`}
                                >
                                  <div className="text-sm font-medium text-white">{voice.name}</div>
                                  <div className="text-xs text-gray-400">{voice.gender} • {voice.accent}</div>
                                </button>
                              ))}
                            </div>
                          </div>

                          {/* Summary */}
                          <div className="p-4 bg-gray-700/50 rounded-xl">
                            <h4 className="text-sm font-medium text-white mb-2">Summary</h4>
                            <div className="grid grid-cols-2 gap-2 text-sm">
                              <div className="text-gray-400">Scenes:</div>
                              <div className="text-white">{scriptScenes.length}</div>
                              <div className="text-gray-400">Media selected:</div>
                              <div className="text-white">{scriptScenes.filter(s => s.media).length}</div>
                              <div className="text-gray-400">Duration:</div>
                              <div className="text-white">{videoDuration}s</div>
                              <div className="text-gray-400">Voice:</div>
                              <div className="text-white">{voiceOptions.find(v => v.id === selectedVoice)?.name}</div>
                            </div>
                          </div>

                          <button
                            onClick={handleGenerateAIVideo}
                            disabled={isGeneratingVideo || scriptScenes.filter(s => s.media).length === 0}
                            className="w-full py-4 bg-gradient-to-r from-pink-600 to-rose-600 text-white font-semibold rounded-xl hover:from-pink-700 hover:to-rose-700 transition disabled:opacity-50 flex items-center justify-center gap-2 text-lg"
                          >
                            {isGeneratingVideo ? (
                              <>
                                <Loader2 className="w-5 h-5 animate-spin" />
                                Creating Your Video...
                              </>
                            ) : (
                              <>
                                <Wand2 className="w-5 h-5" />
                                Generate AI Video
                              </>
                            )}
                          </button>

                          {isGeneratingVideo && (
                            <div className="space-y-2">
                              <div className="flex justify-between text-sm">
                                <span className="text-gray-400">Progress</span>
                                <span className="text-pink-400">{videoGenerationProgress}%</span>
                              </div>
                              <div className="w-full bg-gray-700 rounded-full h-2">
                                <div
                                  className="bg-gradient-to-r from-pink-500 to-rose-500 h-2 rounded-full transition-all"
                                  style={{ width: `${videoGenerationProgress}%` }}
                                />
                              </div>
                              <p className="text-xs text-gray-500 text-center">
                                {videoGenerationProgress < 40 && 'Generating voiceover...'}
                                {videoGenerationProgress >= 40 && videoGenerationProgress < 70 && 'Composing video...'}
                                {videoGenerationProgress >= 70 && videoGenerationProgress < 100 && 'Finalizing...'}
                                {videoGenerationProgress === 100 && 'Complete!'}
                              </p>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Step 5: Video Ready */}
                      {aiVideoStep === 5 && mediaAssets.length > 0 && (
                        <div className="space-y-4 text-center">
                          <div className="w-16 h-16 bg-green-600/20 rounded-full flex items-center justify-center mx-auto">
                            <Check className="w-8 h-8 text-green-400" />
                          </div>
                          <h3 className="text-lg font-semibold text-white">Video Created!</h3>
                          <p className="text-gray-400 text-sm">
                            Your AI video is ready. Preview it on the right and proceed to add captions.
                          </p>
                          <button
                            onClick={() => {
                              setAiVideoStep(1);
                              setVideoTopic('');
                              setScriptScenes([]);
                              setMediaAssets([]);
                            }}
                            className="px-4 py-2 bg-gray-700 text-gray-300 rounded-xl hover:bg-gray-600 transition"
                          >
                            Start Over
                          </button>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Image Generation */}
                  {(contentType === 'image' || contentType === 'image-text' || contentType === 'carousel') && (
                    <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                      <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <ImageIcon className="w-5 h-5 text-purple-400" />
                        Generate Image
                      </h3>

                      <div className="space-y-4">
                        <div>
                          <label className="block text-sm text-gray-400 mb-2">Describe your image</label>
                          <textarea
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder="A stunning sunset over mountains with vibrant colors..."
                            rows={3}
                            className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                          />
                        </div>

                        <div>
                          <label className="block text-sm text-gray-400 mb-2">Style</label>
                          <div className="grid grid-cols-3 gap-2">
                            {imageStyles.map((style) => (
                              <button
                                key={style.id}
                                onClick={() => setImageStyle(style.id)}
                                className={`p-2 rounded-xl border text-center transition ${
                                  imageStyle === style.id
                                    ? 'bg-purple-600/20 border-purple-500'
                                    : 'bg-gray-700 border-gray-600 hover:border-gray-500'
                                }`}
                              >
                                <div className="text-lg mb-0.5">{style.icon}</div>
                                <div className="text-xs text-gray-300">{style.name}</div>
                              </button>
                            ))}
                          </div>
                        </div>

                        <button
                          onClick={handleGenerateImage}
                          disabled={isGenerating || !prompt.trim()}
                          className="w-full py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold rounded-xl hover:from-purple-700 hover:to-pink-700 transition disabled:opacity-50 flex items-center justify-center gap-2"
                        >
                          {isGenerating ? (
                            <>
                              <Loader2 className="w-5 h-5 animate-spin" />
                              {generationStatus || 'Generating...'}
                            </>
                          ) : (
                            <>
                              <Sparkles className="w-5 h-5" />
                              Generate Image
                            </>
                          )}
                        </button>

                        {isGenerating && generationProgress > 0 && (
                          <div className="w-full bg-gray-700 rounded-full h-2">
                            <div
                              className="bg-purple-500 h-2 rounded-full transition-all"
                              style={{ width: `${generationProgress}%` }}
                            />
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Video Upload */}
                  {contentType === 'video-upload' && (
                    <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                      <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <Upload className="w-5 h-5 text-blue-400" />
                        Upload Video
                      </h3>

                      <div
                        onClick={() => fileInputRef.current?.click()}
                        className="border-2 border-dashed border-gray-600 rounded-xl p-8 text-center cursor-pointer hover:border-purple-500 transition"
                      >
                        {isUploading ? (
                          <>
                            <Loader2 className="w-12 h-12 text-purple-500 mx-auto mb-4 animate-spin" />
                            <p className="text-white font-medium mb-2">Uploading...</p>
                            <div className="w-full bg-gray-700 rounded-full h-2 mt-4">
                              <div
                                className="bg-purple-500 h-2 rounded-full transition-all"
                                style={{ width: `${uploadProgress}%` }}
                              />
                            </div>
                          </>
                        ) : (
                          <>
                            <Upload className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                            <p className="text-white font-medium mb-2">Drop your video here</p>
                            <p className="text-gray-400 text-sm mb-4">or click to browse</p>
                            <p className="text-gray-500 text-xs">MP4, MOV, WEBM • Max 500MB</p>
                          </>
                        )}
                        <input
                          ref={fileInputRef}
                          type="file"
                          accept="video/*,image/*"
                          onChange={handleFileUpload}
                          className="hidden"
                        />
                      </div>
                    </div>
                  )}

                  {/* Stock Video Search */}
                  {contentType === 'video-stock' && (
                    <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                      <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <Video className="w-5 h-5 text-orange-400" />
                        Stock Videos
                      </h3>

                      <div className="space-y-4">
                        <div className="flex gap-3">
                          <input
                            type="text"
                            value={videoSearchQuery}
                            onChange={(e) => setVideoSearchQuery(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSearchStockVideos()}
                            placeholder="Search for videos..."
                            className="flex-1 px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-orange-500"
                          />
                          <button
                            onClick={handleSearchStockVideos}
                            disabled={isGenerating}
                            className="px-4 py-3 bg-orange-600 text-white rounded-xl hover:bg-orange-700 transition disabled:opacity-50"
                          >
                            Search
                          </button>
                        </div>

                        {stockVideos.length > 0 && (
                          <div className="grid grid-cols-2 gap-3 max-h-80 overflow-y-auto">
                            {stockVideos.map((video) => (
                              <button
                                key={video.id}
                                onClick={() => handleAddStockVideo(video)}
                                className={`relative rounded-xl overflow-hidden border-2 transition ${
                                  selectedStockVideos.find(v => v.id === video.id)
                                    ? 'border-orange-500'
                                    : 'border-transparent hover:border-gray-600'
                                }`}
                              >
                                <img
                                  src={video.preview_url}
                                  alt={`Video by ${video.user}`}
                                  className="w-full aspect-video object-cover"
                                />
                                <div className="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-black/80">
                                  <span className="text-xs text-white">{formatDuration(video.duration)}</span>
                                </div>
                                {selectedStockVideos.find(v => v.id === video.id) && (
                                  <div className="absolute top-2 right-2 w-6 h-6 bg-orange-500 rounded-full flex items-center justify-center">
                                    <Check className="w-4 h-4 text-white" />
                                  </div>
                                )}
                              </button>
                            ))}
                          </div>
                        )}

                        {selectedStockVideos.length > 0 && (
                          <button
                            onClick={handleComposeVideo}
                            disabled={isGenerating}
                            className="w-full py-3 bg-gradient-to-r from-orange-600 to-red-600 text-white font-semibold rounded-xl hover:from-orange-700 hover:to-red-700 transition disabled:opacity-50 flex items-center justify-center gap-2"
                          >
                            {isGenerating ? (
                              <>
                                <Loader2 className="w-5 h-5 animate-spin" />
                                {generationStatus || 'Composing...'}
                              </>
                            ) : (
                              <>
                                <Video className="w-5 h-5" />
                                Compose {selectedStockVideos.length} Videos
                              </>
                            )}
                          </button>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Voiceover Section (for video types) */}
                  {(contentType === 'video-stock' || contentType === 'video-upload') && (
                    <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                      <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <Mic className="w-5 h-5 text-blue-400" />
                        Add Voiceover (Optional)
                      </h3>

                      <div className="space-y-4">
                        <div>
                          <label className="block text-sm text-gray-400 mb-2">Voice</label>
                          <div className="grid grid-cols-3 gap-2">
                            {voiceOptions.slice(0, 6).map((voice) => (
                              <button
                                key={voice.id}
                                onClick={() => setSelectedVoice(voice.id)}
                                className={`p-2 rounded-xl border text-center transition ${
                                  selectedVoice === voice.id
                                    ? 'bg-blue-600/20 border-blue-500'
                                    : 'bg-gray-700 border-gray-600 hover:border-gray-500'
                                }`}
                              >
                                <div className="text-sm font-medium text-white">{voice.name}</div>
                                <div className="text-xs text-gray-400">{voice.gender}</div>
                              </button>
                            ))}
                          </div>
                        </div>

                        <div>
                          <label className="block text-sm text-gray-400 mb-2">Script</label>
                          <textarea
                            value={voiceoverText}
                            onChange={(e) => setVoiceoverText(e.target.value)}
                            placeholder="Enter the text you want to convert to speech..."
                            rows={3}
                            className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                          />
                        </div>

                        <button
                          onClick={handleGenerateVoiceover}
                          disabled={isGenerating || !voiceoverText.trim()}
                          className="w-full py-3 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-700 transition disabled:opacity-50 flex items-center justify-center gap-2"
                        >
                          {isGenerating ? (
                            <>
                              <Loader2 className="w-5 h-5 animate-spin" />
                              Generating...
                            </>
                          ) : (
                            <>
                              <Mic className="w-5 h-5" />
                              Generate Voiceover
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Text Overlay (for image-text) */}
                  {contentType === 'image-text' && (
                    <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                          <Type className="w-5 h-5 text-green-400" />
                          Text Overlays
                        </h3>
                        <button
                          onClick={handleAddTextOverlay}
                          className="p-2 bg-gray-700 rounded-lg hover:bg-gray-600 transition"
                        >
                          <Plus className="w-4 h-4 text-white" />
                        </button>
                      </div>

                      <div className="space-y-3">
                        {textOverlays.map((overlay, idx) => (
                          <div key={overlay.id} className="flex gap-3">
                            <input
                              type="text"
                              value={overlay.text}
                              onChange={(e) => {
                                const updated = [...textOverlays];
                                updated[idx].text = e.target.value;
                                setTextOverlays(updated);
                              }}
                              placeholder="Enter text..."
                              className="flex-1 px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500"
                            />
                            <select
                              value={overlay.position}
                              onChange={(e) => {
                                const updated = [...textOverlays];
                                updated[idx].position = e.target.value as any;
                                setTextOverlays(updated);
                              }}
                              className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white focus:outline-none"
                            >
                              <option value="top">Top</option>
                              <option value="center">Center</option>
                              <option value="bottom">Bottom</option>
                            </select>
                            <button
                              onClick={() => setTextOverlays(prev => prev.filter(o => o.id !== overlay.id))}
                              className="p-2 bg-red-600/20 rounded-lg hover:bg-red-600/40 transition"
                            >
                              <Trash2 className="w-4 h-4 text-red-400" />
                            </button>
                          </div>
                        ))}

                        {textOverlays.length === 0 && (
                          <p className="text-gray-500 text-sm text-center py-4">
                            Click + to add text overlays
                          </p>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {/* Right: Preview Panel */}
                <div className="space-y-6">
                  <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <Eye className="w-5 h-5 text-gray-400" />
                      Preview
                    </h3>

                    {mediaAssets.length > 0 ? (
                      <div className="space-y-4">
                        {mediaAssets.filter(a => a.type !== 'voiceover').map((asset) => (
                          <div key={asset.id} className="relative">
                            {asset.type === 'image' && (
                              <div className="relative rounded-xl overflow-hidden aspect-square bg-gray-900">
                                <img
                                  src={asset.url}
                                  alt="Generated"
                                  className="w-full h-full object-cover"
                                />
                                {/* Text overlays preview */}
                                {textOverlays.map((overlay) => (
                                  <div
                                    key={overlay.id}
                                    className={`absolute left-0 right-0 p-4 text-center ${
                                      overlay.position === 'top' ? 'top-4' :
                                      overlay.position === 'bottom' ? 'bottom-4' :
                                      'top-1/2 -translate-y-1/2'
                                    }`}
                                  >
                                    <span className="px-4 py-2 bg-black/60 text-white font-bold text-lg rounded-lg">
                                      {overlay.text || 'Your text here'}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            )}
                            {asset.type === 'video' && (
                              <div className="relative rounded-xl overflow-hidden aspect-[9/16] max-h-96 bg-black">
                                <video
                                  ref={videoRef}
                                  src={asset.url}
                                  className="w-full h-full object-contain"
                                  loop
                                  muted={isMuted}
                                  playsInline
                                />
                                <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/80">
                                  <div className="flex items-center justify-between">
                                    <button
                                      onClick={() => {
                                        if (videoRef.current) {
                                          isPlaying ? videoRef.current.pause() : videoRef.current.play();
                                          setIsPlaying(!isPlaying);
                                        }
                                      }}
                                      className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center"
                                    >
                                      {isPlaying ? <Pause className="w-4 h-4 text-white" /> : <Play className="w-4 h-4 text-white" />}
                                    </button>
                                    <span className="text-white text-sm">{formatDuration(asset.duration || 0)}</span>
                                    <button
                                      onClick={() => setIsMuted(!isMuted)}
                                      className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center"
                                    >
                                      {isMuted ? <VolumeX className="w-4 h-4 text-white" /> : <Volume2 className="w-4 h-4 text-white" />}
                                    </button>
                                  </div>
                                </div>
                              </div>
                            )}
                            <button
                              onClick={() => removeAsset(asset.id)}
                              className="absolute top-2 right-2 p-2 bg-red-500/80 rounded-full hover:bg-red-500 transition"
                            >
                              <X className="w-4 h-4 text-white" />
                            </button>
                          </div>
                        ))}

                        {/* Voiceover indicator */}
                        {hasVoiceover && (
                          <div className="flex items-center gap-3 p-3 bg-blue-600/20 border border-blue-500/30 rounded-xl">
                            <Mic className="w-5 h-5 text-blue-400" />
                            <span className="text-blue-300 text-sm">Voiceover added</span>
                          </div>
                        )}

                        {/* Carousel thumbnails */}
                        {contentType === 'carousel' && mediaAssets.length > 1 && (
                          <div className="flex gap-2 overflow-x-auto pb-2">
                            {mediaAssets.filter(a => a.type === 'image').map((asset, idx) => (
                              <div
                                key={asset.id}
                                className="relative flex-shrink-0 w-16 h-16 rounded-lg overflow-hidden"
                              >
                                <img
                                  src={asset.url}
                                  alt={`Slide ${idx + 1}`}
                                  className="w-full h-full object-cover"
                                />
                                <span className="absolute bottom-0.5 right-0.5 text-xs bg-black/60 text-white px-1 rounded">
                                  {idx + 1}
                                </span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="aspect-square rounded-xl bg-gray-900 flex items-center justify-center">
                        <div className="text-center">
                          <ImageIcon className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                          <p className="text-gray-500">No media yet</p>
                          <p className="text-gray-600 text-sm">Generate or upload content</p>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Asset list */}
                  {mediaAssets.length > 0 && (
                    <div className="bg-gray-800 rounded-2xl p-4 border border-gray-700">
                      <h4 className="text-sm font-medium text-gray-400 mb-3">Assets ({mediaAssets.length})</h4>
                      <div className="space-y-2">
                        {mediaAssets.map((asset) => (
                          <div key={asset.id} className="flex items-center gap-3 p-2 bg-gray-700/50 rounded-lg">
                            <div className={`p-1.5 rounded ${
                              asset.type === 'image' ? 'bg-purple-600/20' :
                              asset.type === 'video' ? 'bg-orange-600/20' :
                              'bg-blue-600/20'
                            }`}>
                              {asset.type === 'image' && <ImageIcon className="w-4 h-4 text-purple-400" />}
                              {asset.type === 'video' && <Video className="w-4 h-4 text-orange-400" />}
                              {asset.type === 'voiceover' && <Mic className="w-4 h-4 text-blue-400" />}
                            </div>
                            <span className="flex-1 text-sm text-white capitalize">{asset.type}</span>
                            {asset.duration && (
                              <span className="text-xs text-gray-400">{formatDuration(asset.duration)}</span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}

          {/* ============ STEP 3: Caption & Hashtags ============ */}
          {currentStep === 3 && (
            <motion.div
              key="step3"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-6"
            >
              <div className="grid lg:grid-cols-2 gap-6">
                {/* Caption Section */}
                <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                      <FileText className="w-5 h-5 text-purple-400" />
                      Caption
                    </h3>
                    <button
                      onClick={handleGenerateCaption}
                      disabled={isGeneratingCaption}
                      className="px-3 py-1.5 bg-purple-600 text-white text-sm rounded-lg hover:bg-purple-700 transition disabled:opacity-50 flex items-center gap-1"
                    >
                      {isGeneratingCaption ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Sparkles className="w-4 h-4" />
                      )}
                      AI Generate
                    </button>
                  </div>

                  <textarea
                    value={caption}
                    onChange={(e) => setCaption(e.target.value)}
                    placeholder="Write a compelling caption for your post..."
                    rows={5}
                    className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                  />
                  <div className="flex justify-between mt-2 text-xs text-gray-500">
                    <span>Tip: Include a call-to-action</span>
                    <span>{caption.length} characters</span>
                  </div>
                </div>

                {/* Hashtags Section */}
                <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                      <Target className="w-5 h-5 text-pink-400" />
                      Hashtags
                    </h3>
                    <button
                      onClick={handleGenerateHashtags}
                      className="px-3 py-1.5 bg-pink-600 text-white text-sm rounded-lg hover:bg-pink-700 transition flex items-center gap-1"
                    >
                      <Sparkles className="w-4 h-4" />
                      AI Suggest
                    </button>
                  </div>

                  <div className="flex flex-wrap gap-2 mb-4 min-h-[80px] p-3 bg-gray-700/50 rounded-xl">
                    {hashtags.map((tag, idx) => (
                      <span
                        key={idx}
                        className="px-3 py-1.5 bg-purple-600/20 text-purple-300 rounded-full text-sm flex items-center gap-1"
                      >
                        {tag}
                        <button
                          onClick={() => setHashtags(prev => prev.filter((_, i) => i !== idx))}
                          className="hover:text-red-400"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </span>
                    ))}
                    {hashtags.length === 0 && (
                      <span className="text-gray-500 text-sm">No hashtags yet. Click AI Suggest or add manually.</span>
                    )}
                  </div>

                  <div className="flex gap-2">
                    <input
                      type="text"
                      placeholder="Add hashtag..."
                      className="flex-1 px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-pink-500"
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          const input = e.target as HTMLInputElement;
                          const tag = input.value.trim();
                          if (tag) {
                            setHashtags(prev => [...prev, tag.startsWith('#') ? tag : `#${tag}`]);
                            input.value = '';
                          }
                        }
                      }}
                    />
                    <button
                      onClick={(e) => {
                        const input = (e.target as HTMLElement).previousElementSibling as HTMLInputElement;
                        const tag = input.value.trim();
                        if (tag) {
                          setHashtags(prev => [...prev, tag.startsWith('#') ? tag : `#${tag}`]);
                          input.value = '';
                        }
                      }}
                      className="px-4 py-2 bg-gray-700 text-white rounded-xl hover:bg-gray-600 transition"
                    >
                      <Plus className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              </div>

              {/* Preview with caption */}
              <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-4">Post Preview</h3>
                <div className="flex gap-6">
                  {/* Media thumbnail */}
                  <div className="w-32 h-32 rounded-xl overflow-hidden bg-gray-900 flex-shrink-0">
                    {mediaAssets[0]?.type === 'image' && (
                      <img src={mediaAssets[0].url} alt="Preview" className="w-full h-full object-cover" />
                    )}
                    {mediaAssets[0]?.type === 'video' && (
                      <video src={mediaAssets[0].url} className="w-full h-full object-cover" />
                    )}
                  </div>
                  {/* Caption preview */}
                  <div className="flex-1">
                    <p className="text-white mb-3">{caption || 'Your caption will appear here...'}</p>
                    <div className="flex flex-wrap gap-1">
                      {hashtags.map((tag, idx) => (
                        <span key={idx} className="text-purple-400 text-sm">{tag}</span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* ============ STEP 4: Schedule & Publish ============ */}
          {currentStep === 4 && (
            <motion.div
              key="step4"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-6"
            >
              <div className="grid lg:grid-cols-2 gap-6">
                {/* Platforms */}
                <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <Target className="w-5 h-5 text-purple-400" />
                    Target Platforms
                  </h3>

                  <div className="space-y-3">
                    {platforms.map((platform) => (
                      <button
                        key={platform.id}
                        onClick={() => togglePlatform(platform.id)}
                        className={`w-full p-4 rounded-xl border transition flex items-center gap-3 ${
                          targetPlatforms.includes(platform.id)
                            ? 'bg-purple-600/20 border-purple-500'
                            : 'bg-gray-700 border-gray-600 hover:border-gray-500'
                        }`}
                      >
                        <div className={`w-10 h-10 rounded-full ${platform.color} flex items-center justify-center`}>
                          <span className="text-white font-bold text-sm">{platform.name[0]}</span>
                        </div>
                        <span className="text-white font-medium">{platform.name}</span>
                        {targetPlatforms.includes(platform.id) && (
                          <Check className="w-5 h-5 text-purple-400 ml-auto" />
                        )}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Schedule */}
                <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <Calendar className="w-5 h-5 text-blue-400" />
                    Schedule
                  </h3>

                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Date</label>
                      <input
                        type="date"
                        value={scheduledDate}
                        onChange={(e) => setScheduledDate(e.target.value)}
                        min={getMinDate()}
                        className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Time</label>
                      <input
                        type="time"
                        value={scheduledTime}
                        onChange={(e) => setScheduledTime(e.target.value)}
                        className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>

                    {scheduledDate && scheduledTime && (
                      <div className="p-4 bg-blue-600/10 border border-blue-500/30 rounded-xl">
                        <p className="text-blue-300 text-sm">
                          Your post will be published on{' '}
                          <strong>
                            {new Date(`${scheduledDate}T${scheduledTime}`).toLocaleDateString('en-US', {
                              weekday: 'long',
                              month: 'long',
                              day: 'numeric',
                              hour: 'numeric',
                              minute: '2-digit',
                            })}
                          </strong>
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Final Summary */}
              <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-4">Summary</h3>

                <div className="grid md:grid-cols-4 gap-4">
                  <div className="p-4 bg-gray-700/50 rounded-xl">
                    <p className="text-gray-400 text-sm mb-1">Content</p>
                    <p className="text-white font-medium capitalize">
                      {contentType?.replace('-', ' ') || 'Not selected'}
                    </p>
                  </div>
                  <div className="p-4 bg-gray-700/50 rounded-xl">
                    <p className="text-gray-400 text-sm mb-1">Assets</p>
                    <p className="text-white font-medium">{mediaAssets.length} files</p>
                  </div>
                  <div className="p-4 bg-gray-700/50 rounded-xl">
                    <p className="text-gray-400 text-sm mb-1">Platforms</p>
                    <p className="text-white font-medium">{targetPlatforms.length} selected</p>
                  </div>
                  <div className="p-4 bg-gray-700/50 rounded-xl">
                    <p className="text-gray-400 text-sm mb-1">Hashtags</p>
                    <p className="text-white font-medium">{hashtags.length} tags</p>
                  </div>
                </div>
              </div>

              {/* Publish Button */}
              <button
                onClick={handlePublish}
                disabled={isPublishing || !canProceed()}
                className="w-full py-4 bg-gradient-to-r from-green-600 to-emerald-600 text-white font-semibold rounded-xl hover:from-green-700 hover:to-emerald-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 text-lg"
              >
                {isPublishing ? (
                  <>
                    <Loader2 className="w-6 h-6 animate-spin" />
                    Scheduling...
                  </>
                ) : (
                  <>
                    <Calendar className="w-6 h-6" />
                    Schedule Post
                  </>
                )}
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Navigation Buttons */}
        <div className="flex justify-between mt-8">
          <button
            onClick={prevStep}
            disabled={currentStep === 1}
            className="px-6 py-3 bg-gray-700 text-white rounded-xl hover:bg-gray-600 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <ChevronLeft className="w-5 h-5" />
            Back
          </button>

          {currentStep < 4 && (
            <button
              onClick={nextStep}
              disabled={!canProceed()}
              className="px-6 py-3 bg-purple-600 text-white rounded-xl hover:bg-purple-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              Next
              <ChevronRight className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
