'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'next/navigation';
import {
  Play,
  BookOpen,
  MessageSquare,
  Settings,
  Trophy,
  Flame,
  Target,
  X,
  ChevronLeft,
  Wifi,
  WifiOff,
} from 'lucide-react';

import { usePractice } from './hooks/usePractice';
import { CodeEditor } from './components/CodeEditor';
import { ExercisePanel } from './components/ExercisePanel';
import { ChatInterface } from './components/ChatInterface';
import { ResultPanel } from './components/ResultPanel';
import {
  DifficultyLevel,
  ExerciseCategory,
  Exercise,
  SubmitCodeResponse,
  getDifficultyLabel,
  getCategoryLabel,
  getCategoryIcon,
} from './lib/practice-types';

// Mock user ID - in production, get from auth context
const MOCK_USER_ID = 'user-123';

export default function PracticePage() {
  const searchParams = useSearchParams();
  const courseId = searchParams.get('courseId');
  const sessionId = searchParams.get('sessionId');

  const {
    session,
    currentExercise,
    progress,
    messages,
    isLoading,
    isExecuting,
    error,
    wsConnected,
    createSession,
    loadSession,
    endSession,
    listExercises,
    selectExercise,
    sendMessage,
    submitCode,
    requestHint,
    loadProgress,
    clearError,
  } = usePractice(MOCK_USER_ID);

  const [code, setCode] = useState('');
  const [lastResult, setLastResult] = useState<SubmitCodeResponse | null>(null);
  const [showSettings, setShowSettings] = useState(!session);
  const [exercises, setExercises] = useState<Exercise[]>([]);
  const [activeTab, setActiveTab] = useState<'exercise' | 'chat' | 'result'>('exercise');

  // Settings state
  const [selectedDifficulty, setSelectedDifficulty] = useState<DifficultyLevel>(
    DifficultyLevel.BEGINNER
  );
  const [selectedCategories, setSelectedCategories] = useState<ExerciseCategory[]>([
    ExerciseCategory.DOCKER,
  ]);
  const [pairProgramming, setPairProgramming] = useState(false);

  // Load session from URL or create new one
  useEffect(() => {
    if (sessionId) {
      loadSession(sessionId);
    }
  }, [sessionId, loadSession]);

  // Load progress
  useEffect(() => {
    loadProgress();
  }, [loadProgress]);

  // Set starter code when exercise changes
  useEffect(() => {
    if (currentExercise?.starter_code) {
      setCode(currentExercise.starter_code);
      setLastResult(null);
    }
  }, [currentExercise?.id]);

  // Load exercises for selection
  useEffect(() => {
    const loadExercises = async () => {
      const exerciseList = await listExercises();
      setExercises(exerciseList);
    };
    loadExercises();
  }, [listExercises]);

  const handleStartSession = useCallback(async () => {
    await createSession(
      courseId || undefined,
      selectedDifficulty,
      selectedCategories.length > 0 ? selectedCategories : undefined,
      pairProgramming
    );
    setShowSettings(false);
  }, [courseId, selectedDifficulty, selectedCategories, pairProgramming, createSession]);

  const handleSubmitCode = useCallback(async () => {
    if (!code.trim()) return;
    const result = await submitCode(code);
    if (result) {
      setLastResult(result);
      setActiveTab('result');
    }
  }, [code, submitCode]);

  const handleRequestHint = useCallback(
    (level: number) => {
      requestHint(level);
    },
    [requestHint]
  );

  const handleNextExercise = useCallback(() => {
    if (lastResult?.next_exercise) {
      setCode(lastResult.next_exercise.starter_code || '');
      setLastResult(null);
      setActiveTab('exercise');
    }
  }, [lastResult]);

  const handleSelectExercise = useCallback(
    async (exerciseId: string) => {
      await selectExercise(exerciseId);
      setShowSettings(false);
    },
    [selectExercise]
  );

  const toggleCategory = useCallback((category: ExerciseCategory) => {
    setSelectedCategories((prev) =>
      prev.includes(category)
        ? prev.filter((c) => c !== category)
        : [...prev, category]
    );
  }, []);

  // Settings/New Session Modal
  if (showSettings || !session) {
    return (
      <div className="min-h-screen bg-gray-950 p-6">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">Mode Pratique</h1>
            <p className="text-gray-400">
              Entraînez-vous avec des exercices interactifs et un assistant IA
            </p>
          </div>

          {/* Session configuration */}
          <div className="bg-gray-900 rounded-xl border border-gray-700 p-6 mb-6">
            <h2 className="text-xl font-semibold text-white mb-4">
              Configurer votre session
            </h2>

            {/* Difficulty */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Niveau de difficulté
              </label>
              <div className="grid grid-cols-4 gap-2">
                {Object.values(DifficultyLevel).map((level) => (
                  <button
                    key={level}
                    onClick={() => setSelectedDifficulty(level)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                      selectedDifficulty === level
                        ? 'bg-purple-600 text-white'
                        : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                    }`}
                  >
                    {getDifficultyLabel(level)}
                  </button>
                ))}
              </div>
            </div>

            {/* Categories */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Catégories (sélectionnez une ou plusieurs)
              </label>
              <div className="flex flex-wrap gap-2">
                {Object.values(ExerciseCategory).map((category) => (
                  <button
                    key={category}
                    onClick={() => toggleCategory(category)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm transition-colors ${
                      selectedCategories.includes(category)
                        ? 'bg-purple-600 text-white'
                        : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                    }`}
                  >
                    {getCategoryIcon(category)}
                    {getCategoryLabel(category)}
                  </button>
                ))}
              </div>
            </div>

            {/* Pair programming toggle */}
            <div className="mb-6">
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={pairProgramming}
                  onChange={(e) => setPairProgramming(e.target.checked)}
                  className="w-5 h-5 rounded border-gray-600 bg-gray-700 text-purple-600 focus:ring-purple-500"
                />
                <div>
                  <span className="text-white font-medium">
                    Mode Pair Programming
                  </span>
                  <p className="text-sm text-gray-400">
                    L'assistant code avec vous en temps réel
                  </p>
                </div>
              </label>
            </div>

            {/* Start button */}
            <button
              onClick={handleStartSession}
              disabled={isLoading || selectedCategories.length === 0}
              className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors"
            >
              {isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Démarrage...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  Démarrer la session
                </>
              )}
            </button>
          </div>

          {/* Exercise browser */}
          <div className="bg-gray-900 rounded-xl border border-gray-700 p-6">
            <h2 className="text-xl font-semibold text-white mb-4">
              Ou choisissez un exercice
            </h2>

            <div className="space-y-2 max-h-96 overflow-y-auto">
              {exercises.map((exercise) => (
                <button
                  key={exercise.id}
                  onClick={() => handleSelectExercise(exercise.id)}
                  className="w-full flex items-center gap-4 p-4 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors text-left"
                >
                  <div className="text-2xl">{getCategoryIcon(exercise.category)}</div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium text-white truncate">
                      {exercise.title}
                    </h3>
                    <p className="text-sm text-gray-400 truncate">
                      {exercise.description}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-yellow-500 text-sm">
                      {exercise.points} pts
                    </span>
                    <span className="px-2 py-0.5 text-xs bg-gray-700 text-gray-300 rounded">
                      {getDifficultyLabel(exercise.difficulty)}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Progress summary */}
          {progress && (
            <div className="mt-6 bg-gray-900 rounded-xl border border-gray-700 p-6">
              <h2 className="text-xl font-semibold text-white mb-4">
                Votre progression
              </h2>
              <div className="grid grid-cols-4 gap-4">
                <div className="text-center">
                  <Trophy className="w-8 h-8 text-yellow-500 mx-auto mb-2" />
                  <div className="text-2xl font-bold text-white">
                    {progress.total_points}
                  </div>
                  <div className="text-sm text-gray-400">Points</div>
                </div>
                <div className="text-center">
                  <Target className="w-8 h-8 text-green-500 mx-auto mb-2" />
                  <div className="text-2xl font-bold text-white">
                    {progress.exercises_completed}
                  </div>
                  <div className="text-sm text-gray-400">Exercices</div>
                </div>
                <div className="text-center">
                  <Flame className="w-8 h-8 text-orange-500 mx-auto mb-2" />
                  <div className="text-2xl font-bold text-white">
                    {progress.current_streak}
                  </div>
                  <div className="text-sm text-gray-400">Série actuelle</div>
                </div>
                <div className="text-center">
                  <BookOpen className="w-8 h-8 text-blue-500 mx-auto mb-2" />
                  <div className="text-2xl font-bold text-white">
                    {progress.badges.length}
                  </div>
                  <div className="text-sm text-gray-400">Badges</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  // Main practice interface
  return (
    <div className="h-screen bg-gray-950 flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 bg-gray-900 border-b border-gray-700">
        <div className="flex items-center gap-4">
          <button
            onClick={() => setShowSettings(true)}
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
          <div>
            <h1 className="font-semibold text-white">
              {currentExercise?.title || 'Mode Pratique'}
            </h1>
            <p className="text-xs text-gray-400">
              Session: {session.id.slice(0, 8)}...
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Connection status */}
          <div className="flex items-center gap-2 text-sm">
            {wsConnected ? (
              <>
                <Wifi className="w-4 h-4 text-green-500" />
                <span className="text-green-500">Connecté</span>
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4 text-red-500" />
                <span className="text-red-500">Déconnecté</span>
              </>
            )}
          </div>

          {/* Stats */}
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1 text-yellow-500">
              <Trophy className="w-4 h-4" />
              {session.points_earned}
            </span>
            <span className="flex items-center gap-1 text-green-500">
              <Target className="w-4 h-4" />
              {session.exercises_completed.length}
            </span>
          </div>

          {/* Settings */}
          <button
            onClick={() => setShowSettings(true)}
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
          >
            <Settings className="w-5 h-5" />
          </button>

          {/* End session */}
          <button
            onClick={endSession}
            className="px-3 py-1.5 text-sm text-red-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
          >
            Terminer
          </button>
        </div>
      </header>

      {/* Error banner */}
      {error && (
        <div className="px-4 py-2 bg-red-900/50 border-b border-red-500/50 flex items-center justify-between">
          <span className="text-sm text-red-200">{error}</span>
          <button onClick={clearError} className="text-red-200 hover:text-white">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left panel - Code Editor */}
        <div className="w-1/2 p-4 border-r border-gray-700">
          <CodeEditor
            code={code}
            onChange={setCode}
            onSubmit={handleSubmitCode}
            onReset={() => setCode(currentExercise?.starter_code || '')}
            language={currentExercise?.sandbox_type || 'yaml'}
            isExecuting={isExecuting}
            disabled={!currentExercise}
            starterCode={currentExercise?.starter_code || ''}
          />
        </div>

        {/* Right panel - Tabbed interface */}
        <div className="w-1/2 flex flex-col">
          {/* Tabs */}
          <div className="flex border-b border-gray-700">
            <button
              onClick={() => setActiveTab('exercise')}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === 'exercise'
                  ? 'text-purple-500 border-b-2 border-purple-500'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <BookOpen className="w-4 h-4" />
              Exercice
            </button>
            <button
              onClick={() => setActiveTab('chat')}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === 'chat'
                  ? 'text-purple-500 border-b-2 border-purple-500'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <MessageSquare className="w-4 h-4" />
              Assistant
            </button>
            <button
              onClick={() => setActiveTab('result')}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === 'result'
                  ? 'text-purple-500 border-b-2 border-purple-500'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <Target className="w-4 h-4" />
              Résultats
            </button>
          </div>

          {/* Tab content */}
          <div className="flex-1 p-4 overflow-hidden">
            {activeTab === 'exercise' && currentExercise && (
              <ExercisePanel
                exercise={currentExercise}
                onRequestHint={handleRequestHint}
                hintsUsed={session.hints_used_total}
                isLoading={isLoading}
              />
            )}

            {activeTab === 'chat' && (
              <ChatInterface
                messages={messages}
                onSendMessage={sendMessage}
                isLoading={isLoading}
                disabled={!currentExercise}
              />
            )}

            {activeTab === 'result' && (
              <ResultPanel
                result={lastResult}
                onNextExercise={handleNextExercise}
                onRetry={() => setActiveTab('exercise')}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
