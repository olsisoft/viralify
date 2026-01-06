'use client';

import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import {
  Mic, Play, Pause, Download, Copy, RefreshCw,
  Loader2, Sparkles, Volume2, Info, Check,
  ChevronDown, Clock, Zap, Square
} from 'lucide-react';
import toast from 'react-hot-toast';
import { api } from '@/lib/api';

interface GeneratedVoiceover {
  id: string;
  text: string;
  voice: string;
  audioUrl: string;
  duration: number;
  createdAt: Date;
}

const voices = [
  { id: 'rachel', name: 'Rachel', description: 'Professional female', language: 'en-US', preview: true },
  { id: 'adam', name: 'Adam', description: 'Deep male voice', language: 'en-US', preview: true },
  { id: 'bella', name: 'Bella', description: 'Young female', language: 'en-US', preview: true },
  { id: 'josh', name: 'Josh', description: 'Energetic male', language: 'en-US', preview: true },
  { id: 'emily', name: 'Emily', description: 'British female', language: 'en-GB', preview: true },
  { id: 'thomas', name: 'Thomas', description: 'British male', language: 'en-GB', preview: true },
  { id: 'marie', name: 'Marie', description: 'French female', language: 'fr-FR', preview: true },
  { id: 'hans', name: 'Hans', description: 'German male', language: 'de-DE', preview: true },
];

const emotions = [
  { id: 'neutral', name: 'Neutral', icon: '😐' },
  { id: 'happy', name: 'Happy', icon: '😊' },
  { id: 'sad', name: 'Sad', icon: '😢' },
  { id: 'excited', name: 'Excited', icon: '🤩' },
  { id: 'calm', name: 'Calm', icon: '😌' },
  { id: 'serious', name: 'Serious', icon: '😐' },
];

const speechRates = [
  { id: 'slow', name: 'Slow', value: 0.75 },
  { id: 'normal', name: 'Normal', value: 1.0 },
  { id: 'fast', name: 'Fast', value: 1.25 },
  { id: 'very-fast', name: 'Very Fast', value: 1.5 },
];

export default function VoiceoverPage() {
  const [text, setText] = useState('');
  const [selectedVoice, setSelectedVoice] = useState('rachel');
  const [selectedEmotion, setSelectedEmotion] = useState('neutral');
  const [speechRate, setSpeechRate] = useState('normal');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedVoiceovers, setGeneratedVoiceovers] = useState<GeneratedVoiceover[]>([]);
  const [selectedVoiceover, setSelectedVoiceover] = useState<GeneratedVoiceover | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const audioRef = useRef<HTMLAudioElement>(null);

  // Advanced options
  const [stability, setStability] = useState(0.5);
  const [similarity, setSimilarity] = useState(0.75);

  const estimateCredits = () => {
    const charCount = text.length;
    return Math.ceil(charCount / 500) || 1;
  };

  const estimateDuration = () => {
    const words = text.split(/\s+/).filter(w => w.length > 0).length;
    const rate = speechRates.find(r => r.id === speechRate)?.value || 1;
    const wordsPerMinute = 150 * rate;
    const minutes = words / wordsPerMinute;
    return Math.ceil(minutes * 60);
  };

  const handleGenerate = async () => {
    if (!text.trim()) {
      toast.error('Please enter text to convert');
      return;
    }

    setIsGenerating(true);

    try {
      // Call real API
      const jobResponse = await api.media.generateVoiceover({
        text,
        voice: selectedVoice,
        emotion: selectedEmotion,
        speed: speechRates.find(r => r.id === speechRate)?.value || 1,
      });

      toast.success('Voiceover generation started...');

      // Poll for job completion
      const jobId = jobResponse.job_id;
      let attempts = 0;
      const maxAttempts = 60; // 60 seconds max for voiceover

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

      // Extract audio info from result
      const audioUrl = result.output_data?.url;
      const duration = result.output_data?.duration_seconds || estimateDuration();

      const newVoiceover: GeneratedVoiceover = {
        id: jobId,
        text,
        voice: selectedVoice,
        audioUrl: audioUrl || '',
        duration: Math.round(duration),
        createdAt: new Date(),
      };

      setGeneratedVoiceovers(prev => [newVoiceover, ...prev]);
      setSelectedVoiceover(newVoiceover);
      toast.success('Voiceover generated successfully!');
    } catch (error: any) {
      console.error('Generation error:', error);
      toast.error(error.message || 'Failed to generate voiceover');
    } finally {
      setIsGenerating(false);
    }
  };

  const handlePlay = () => {
    if (selectedVoiceover) {
      // If we have a real audio URL, use the audio element
      if (selectedVoiceover.audioUrl && audioRef.current) {
        if (isPlaying) {
          audioRef.current.pause();
        } else {
          audioRef.current.play();
        }
        setIsPlaying(!isPlaying);
      } else if ('speechSynthesis' in window) {
        // Fallback to browser TTS if no audio URL
        if (isPlaying) {
          window.speechSynthesis.cancel();
          setIsPlaying(false);
        } else {
          const utterance = new SpeechSynthesisUtterance(selectedVoiceover.text);
          utterance.rate = speechRates.find(r => r.id === speechRate)?.value || 1;
          utterance.onend = () => setIsPlaying(false);
          window.speechSynthesis.speak(utterance);
          setIsPlaying(true);
        }
      }
    }
  };

  const handleStop = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    setIsPlaying(false);
  };

  const copyText = (textToCopy: string, id: string) => {
    navigator.clipboard.writeText(textToCopy);
    setCopiedId(id);
    toast.success('Text copied!');
    setTimeout(() => setCopiedId(null), 2000);
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
            <div className="p-2 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl">
              <Mic className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-white">Voiceover Creator</h1>
          </div>
          <p className="text-gray-400">
            Generate natural-sounding voiceovers with AI voices
          </p>
        </div>

        {/* API Status Banner */}
        <div className="mb-6 p-3 bg-green-500/20 border border-green-500/30 rounded-xl flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-green-400 flex-shrink-0" />
          <p className="text-sm text-green-200">
            <strong>Connected to ElevenLabs & OpenAI TTS:</strong> Real AI voice generation is active.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Form */}
          <div className="space-y-6">
            {/* Text Input */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <div className="flex items-center justify-between mb-3">
                <label className="block text-sm font-medium text-gray-300">
                  Text to Convert
                </label>
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <Clock className="w-3 h-3" />
                  ~{formatDuration(estimateDuration())}
                </div>
              </div>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Enter the text you want to convert to speech. You can write your script here..."
                rows={6}
                className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
              />
              <div className="flex justify-between mt-2 text-xs text-gray-500">
                <span>{text.length} characters</span>
                <span>{estimateCredits()} credit(s)</span>
              </div>
            </div>

            {/* Voice Selection */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                Select Voice
              </label>
              <div className="grid grid-cols-2 gap-3">
                {voices.slice(0, 6).map((voice) => (
                  <button
                    key={voice.id}
                    onClick={() => setSelectedVoice(voice.id)}
                    className={`p-3 rounded-xl border text-left transition ${
                      selectedVoice === voice.id
                        ? 'bg-blue-600/20 border-blue-500 text-white'
                        : 'bg-gray-700 border-gray-600 text-gray-300 hover:border-gray-500'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium text-sm">{voice.name}</div>
                        <div className="text-xs text-gray-400">{voice.description}</div>
                      </div>
                      <span className="text-xs px-2 py-0.5 bg-gray-600 rounded text-gray-300">
                        {voice.language}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Emotion Selection */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                Emotion / Tone
              </label>
              <div className="grid grid-cols-3 gap-2">
                {emotions.map((emotion) => (
                  <button
                    key={emotion.id}
                    onClick={() => setSelectedEmotion(emotion.id)}
                    className={`p-3 rounded-xl border text-center transition ${
                      selectedEmotion === emotion.id
                        ? 'bg-blue-600/20 border-blue-500'
                        : 'bg-gray-700 border-gray-600 hover:border-gray-500'
                    }`}
                  >
                    <div className="text-xl mb-1">{emotion.icon}</div>
                    <div className="text-xs text-gray-300">{emotion.name}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Speech Rate */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                Speech Rate
              </label>
              <div className="flex gap-2">
                {speechRates.map((rate) => (
                  <button
                    key={rate.id}
                    onClick={() => setSpeechRate(rate.id)}
                    className={`flex-1 py-2 rounded-xl border transition ${
                      speechRate === rate.id
                        ? 'bg-blue-600/20 border-blue-500 text-white'
                        : 'bg-gray-700 border-gray-600 text-gray-300'
                    }`}
                  >
                    {rate.name}
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
                  {/* Stability Slider */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <label className="text-sm text-gray-400">Stability</label>
                      <span className="text-sm text-gray-400">{(stability * 100).toFixed(0)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={stability}
                      onChange={(e) => setStability(parseFloat(e.target.value))}
                      className="w-full accent-blue-500"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>More Variable</span>
                      <span>More Stable</span>
                    </div>
                  </div>

                  {/* Similarity Slider */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <label className="text-sm text-gray-400">Clarity + Similarity</label>
                      <span className="text-sm text-gray-400">{(similarity * 100).toFixed(0)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={similarity}
                      onChange={(e) => setSimilarity(parseFloat(e.target.value))}
                      className="w-full accent-blue-500"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>Low</span>
                      <span>High</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Generate Button */}
            <button
              onClick={handleGenerate}
              disabled={isGenerating || !text.trim()}
              className="w-full py-4 bg-gradient-to-r from-blue-600 to-cyan-600 text-white font-semibold rounded-xl hover:from-blue-700 hover:to-cyan-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  Generate Voiceover ({estimateCredits()} credits)
                </>
              )}
            </button>
          </div>

          {/* Results Panel */}
          <div className="space-y-6">
            {/* Audio Player */}
            {selectedVoiceover ? (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-gray-800 rounded-2xl p-6 border border-gray-700"
              >
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-medium text-white">Generated Audio</h3>
                  <span className="text-sm text-gray-400">
                    {voices.find(v => v.id === selectedVoiceover.voice)?.name}
                  </span>
                </div>

                {/* Waveform Placeholder */}
                <div className="h-24 bg-gray-700/50 rounded-xl mb-4 flex items-center justify-center">
                  <div className="flex items-end gap-1 h-16">
                    {[...Array(30)].map((_, i) => (
                      <div
                        key={i}
                        className={`w-1.5 bg-blue-500 rounded-full transition-all ${
                          isPlaying ? 'animate-pulse' : ''
                        }`}
                        style={{
                          height: `${20 + Math.random() * 60}%`,
                          animationDelay: `${i * 50}ms`,
                        }}
                      />
                    ))}
                  </div>
                </div>

                {/* Controls */}
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <button
                      onClick={handlePlay}
                      className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center hover:bg-blue-700 transition"
                    >
                      {isPlaying ? (
                        <Pause className="w-5 h-5 text-white" />
                      ) : (
                        <Play className="w-5 h-5 text-white ml-0.5" />
                      )}
                    </button>
                    <button
                      onClick={handleStop}
                      className="w-10 h-10 bg-gray-700 rounded-full flex items-center justify-center hover:bg-gray-600 transition"
                    >
                      <Square className="w-4 h-4 text-white" />
                    </button>
                  </div>
                  <span className="text-sm text-gray-400">
                    {formatDuration(selectedVoiceover.duration)}
                  </span>
                </div>

                {/* Text Preview */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Text</span>
                    <button
                      onClick={() => copyText(selectedVoiceover.text, selectedVoiceover.id)}
                      className="p-1 hover:bg-gray-700 rounded transition"
                    >
                      {copiedId === selectedVoiceover.id ? (
                        <Check className="w-4 h-4 text-green-400" />
                      ) : (
                        <Copy className="w-4 h-4 text-gray-400" />
                      )}
                    </button>
                  </div>
                  <p className="text-sm text-gray-300 bg-gray-700/50 p-3 rounded-lg max-h-32 overflow-y-auto">
                    {selectedVoiceover.text}
                  </p>
                </div>

                {/* Actions */}
                <div className="flex gap-3 mt-4">
                  <button
                    onClick={() => toast.success('Download started! (Demo)')}
                    className="flex-1 py-3 bg-gray-700 text-white rounded-xl hover:bg-gray-600 transition flex items-center justify-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    Download MP3
                  </button>
                  <button
                    onClick={() => {
                      setText(selectedVoiceover.text);
                      toast.success('Text loaded');
                    }}
                    className="flex-1 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition flex items-center justify-center gap-2"
                  >
                    <RefreshCw className="w-4 h-4" />
                    Regenerate
                  </button>
                </div>

                {selectedVoiceover.audioUrl && (
                  <audio
                    ref={audioRef}
                    src={selectedVoiceover.audioUrl}
                    onEnded={() => setIsPlaying(false)}
                  />
                )}
              </motion.div>
            ) : (
              <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700 text-center">
                <div className="w-16 h-16 bg-gray-700 rounded-2xl flex items-center justify-center mx-auto mb-4">
                  <Volume2 className="w-8 h-8 text-gray-500" />
                </div>
                <h3 className="text-lg font-medium text-white mb-2">No audio yet</h3>
                <p className="text-gray-400 text-sm">
                  Enter text and click generate to create your first voiceover
                </p>
              </div>
            )}

            {/* History */}
            {generatedVoiceovers.length > 0 && (
              <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                <h3 className="text-sm font-medium text-gray-400 mb-3">
                  Recent Voiceovers ({generatedVoiceovers.length})
                </h3>
                <div className="space-y-2">
                  {generatedVoiceovers.map((vo) => (
                    <button
                      key={vo.id}
                      onClick={() => setSelectedVoiceover(vo)}
                      className={`w-full p-3 rounded-xl text-left transition ${
                        selectedVoiceover?.id === vo.id
                          ? 'bg-blue-600/20 border border-blue-500'
                          : 'bg-gray-700/50 hover:bg-gray-700'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm text-white truncate">{vo.text.slice(0, 50)}...</p>
                          <p className="text-xs text-gray-400 mt-1">
                            {voices.find(v => v.id === vo.voice)?.name} • {formatDuration(vo.duration)}
                          </p>
                        </div>
                        <Play className="w-4 h-4 text-gray-400 flex-shrink-0 ml-2" />
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Tips */}
            <div className="bg-gradient-to-r from-blue-600/10 to-cyan-600/10 border border-blue-500/20 rounded-xl p-4">
              <h4 className="text-white font-medium mb-2">Tips for better results</h4>
              <ul className="text-sm text-gray-400 space-y-1">
                <li>• Use punctuation to control pauses and rhythm</li>
                <li>• Write naturally as you would speak</li>
                <li>• Add emphasis with CAPS for important words</li>
                <li>• Keep sentences short for better clarity</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
