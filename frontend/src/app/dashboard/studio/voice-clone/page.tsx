'use client';

/**
 * Voice Cloning Page
 * Phase 4: Voice Cloning feature
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useVoiceClone } from './hooks/useVoiceClone';
import { VoiceSampleUpload } from './components/VoiceSampleUpload';
import {
  VoiceProfile,
  VoiceGender,
  VoiceAccent,
  getStatusLabel,
  getStatusColor,
  getGenderLabel,
  getAccentLabel,
  formatDuration,
} from './lib/voice-types';

export default function VoiceClonePage() {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showConsentModal, setShowConsentModal] = useState(false);
  const [profileToTrain, setProfileToTrain] = useState<string | null>(null);
  const [testText, setTestText] = useState("Hello! This is a test of my cloned voice. How does it sound?");
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  // Form state for new profile
  const [newProfileName, setNewProfileName] = useState('');
  const [newProfileGender, setNewProfileGender] = useState<VoiceGender>('neutral');
  const [newProfileAccent, setNewProfileAccent] = useState<VoiceAccent>('american');

  const audioRef = useRef<HTMLAudioElement>(null);

  const {
    profiles,
    selectedProfile,
    requirements,
    isLoading,
    isSaving,
    isTraining,
    isGenerating,
    error,
    loadProfiles,
    createProfile,
    selectProfile,
    deleteProfile,
    uploadSample,
    deleteSample,
    startTraining,
    previewVoice,
    generateSpeech,
    clearError,
    getTrainingRequirements,
  } = useVoiceClone({
    onError: (err) => console.error('Voice clone error:', err),
  });

  // Load profiles on mount
  useEffect(() => {
    loadProfiles();
  }, [loadProfiles]);

  // Handlers
  const handleCreateProfile = useCallback(async () => {
    if (!newProfileName.trim()) return;

    const profileId = await createProfile({
      name: newProfileName.trim(),
      gender: newProfileGender,
      accent: newProfileAccent,
    });

    if (profileId) {
      setShowCreateModal(false);
      setNewProfileName('');
      await selectProfile(profileId);
    }
  }, [newProfileName, newProfileGender, newProfileAccent, createProfile, selectProfile]);

  const handleSelectProfile = useCallback(async (profile: VoiceProfile) => {
    await selectProfile(profile.id);
  }, [selectProfile]);

  const handleDeleteProfile = useCallback(async (profileId: string) => {
    if (window.confirm('Delete this voice profile? This action cannot be undone.')) {
      await deleteProfile(profileId);
    }
  }, [deleteProfile]);

  const handleStartTraining = useCallback((profileId: string) => {
    setProfileToTrain(profileId);
    setShowConsentModal(true);
  }, []);

  const handleConfirmTraining = useCallback(async () => {
    if (profileToTrain) {
      await startTraining(profileToTrain, true);
      setShowConsentModal(false);
      setProfileToTrain(null);
    }
  }, [profileToTrain, startTraining]);

  const handlePreviewVoice = useCallback(async () => {
    if (!selectedProfile) return;

    const url = await previewVoice(selectedProfile.id, testText);
    if (url) {
      setAudioUrl(url);
      // Auto-play
      setTimeout(() => audioRef.current?.play(), 100);
    }
  }, [selectedProfile, testText, previewVoice]);

  const handleGenerateSpeech = useCallback(async () => {
    if (!selectedProfile || !testText) return;

    const url = await generateSpeech(selectedProfile.id, testText);
    if (url) {
      setAudioUrl(url);
      setTimeout(() => audioRef.current?.play(), 100);
    }
  }, [selectedProfile, testText, generateSpeech]);

  const trainingRequirements = selectedProfile ? getTrainingRequirements(selectedProfile) : null;

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold text-white">Voice Cloning</h1>
            <p className="text-gray-400 mt-1">Create your own custom AI voice</p>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium flex items-center gap-2"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            New Voice Profile
          </button>
        </div>

        {/* Error banner */}
        {error && (
          <div className="mb-6 bg-red-900/50 border border-red-700 rounded-lg p-4 flex items-center justify-between">
            <span className="text-red-300">{error}</span>
            <button onClick={clearError} className="text-red-300 hover:text-white">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}

        <div className="grid grid-cols-3 gap-6">
          {/* Profiles list */}
          <div className="col-span-1">
            <div className="bg-gray-800 rounded-lg p-4">
              <h2 className="text-lg font-medium text-white mb-4">Voice Profiles</h2>

              {isLoading && profiles.length === 0 ? (
                <div className="text-center py-8">
                  <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
                  <p className="text-gray-400 text-sm">Loading...</p>
                </div>
              ) : profiles.length === 0 ? (
                <div className="text-center py-8">
                  <svg className="w-12 h-12 text-gray-600 mx-auto mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                  </svg>
                  <p className="text-gray-400 text-sm">No voice profiles yet</p>
                  <button
                    onClick={() => setShowCreateModal(true)}
                    className="mt-3 text-blue-400 hover:text-blue-300 text-sm"
                  >
                    Create your first voice
                  </button>
                </div>
              ) : (
                <div className="space-y-2">
                  {profiles.map((profile) => (
                    <button
                      key={profile.id}
                      onClick={() => handleSelectProfile(profile)}
                      className={`
                        w-full text-left p-3 rounded-lg transition-colors
                        ${selectedProfile?.id === profile.id
                          ? 'bg-blue-600'
                          : 'bg-gray-700 hover:bg-gray-600'
                        }
                      `}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-white">{profile.name}</span>
                        <span className={`
                          text-xs px-2 py-0.5 rounded
                          bg-${getStatusColor(profile.status)}-500/20
                          text-${getStatusColor(profile.status)}-400
                        `}>
                          {getStatusLabel(profile.status)}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-gray-400 mt-1">
                        <span>{getGenderLabel(profile.gender)}</span>
                        <span>-</span>
                        <span>{formatDuration(profile.total_sample_duration)}</span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Profile details / editor */}
          <div className="col-span-2">
            {selectedProfile ? (
              <div className="space-y-6">
                {/* Profile header */}
                <div className="bg-gray-800 rounded-lg p-6">
                  <div className="flex items-start justify-between">
                    <div>
                      <h2 className="text-xl font-bold text-white">{selectedProfile.name}</h2>
                      <div className="flex items-center gap-3 text-sm text-gray-400 mt-2">
                        <span>{getGenderLabel(selectedProfile.gender)}</span>
                        <span>-</span>
                        <span>{getAccentLabel(selectedProfile.accent)}</span>
                        <span>-</span>
                        <span className={`text-${getStatusColor(selectedProfile.status)}-400`}>
                          {getStatusLabel(selectedProfile.status)}
                        </span>
                      </div>
                    </div>
                    <button
                      onClick={() => handleDeleteProfile(selectedProfile.id)}
                      className="p-2 text-gray-400 hover:text-red-400 hover:bg-gray-700 rounded"
                      title="Delete profile"
                    >
                      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                </div>

                {/* Sample upload (for draft profiles) */}
                {selectedProfile.status === 'draft' && (
                  <div className="bg-gray-800 rounded-lg p-6">
                    <h3 className="text-lg font-medium text-white mb-4">Upload Voice Samples</h3>
                    <VoiceSampleUpload
                      profileId={selectedProfile.id}
                      samples={selectedProfile.samples}
                      requirements={requirements}
                      trainingRequirements={trainingRequirements}
                      isUploading={isSaving}
                      onUpload={(file) => uploadSample(selectedProfile.id, file)}
                      onDelete={(sampleId) => deleteSample(selectedProfile.id, sampleId)}
                    />

                    {/* Train button */}
                    {trainingRequirements?.can_train && (
                      <div className="mt-6 pt-6 border-t border-gray-700">
                        <button
                          onClick={() => handleStartTraining(selectedProfile.id)}
                          disabled={isTraining}
                          className="w-full px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium flex items-center justify-center gap-2"
                        >
                          {isTraining ? (
                            <>
                              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                              Training...
                            </>
                          ) : (
                            <>
                              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                              </svg>
                              Start Voice Training
                            </>
                          )}
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {/* Training progress */}
                {selectedProfile.status === 'training' && (
                  <div className="bg-gray-800 rounded-lg p-6 text-center">
                    <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-white">Training Your Voice...</h3>
                    <p className="text-gray-400 mt-2">This usually takes less than a minute</p>
                    <div className="mt-4 h-2 bg-gray-700 rounded-full overflow-hidden max-w-xs mx-auto">
                      <div
                        className="h-full bg-blue-500 transition-all"
                        style={{ width: `${selectedProfile.training_progress}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Ready - Test voice */}
                {selectedProfile.status === 'ready' && (
                  <div className="bg-gray-800 rounded-lg p-6">
                    <h3 className="text-lg font-medium text-white mb-4">Test Your Voice</h3>

                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm text-gray-400 mb-2">
                          Enter text to synthesize
                        </label>
                        <textarea
                          value={testText}
                          onChange={(e) => setTestText(e.target.value)}
                          rows={3}
                          className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 resize-none"
                          placeholder="Type something to hear your cloned voice..."
                        />
                      </div>

                      <div className="flex items-center gap-3">
                        <button
                          onClick={handlePreviewVoice}
                          disabled={isGenerating || !testText}
                          className="flex-1 px-4 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium flex items-center justify-center gap-2"
                        >
                          {isGenerating ? (
                            <>
                              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                              Generating...
                            </>
                          ) : (
                            <>
                              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                              </svg>
                              Generate Speech
                            </>
                          )}
                        </button>
                      </div>

                      {/* Audio player */}
                      {audioUrl && (
                        <div className="bg-gray-700 rounded-lg p-4">
                          <audio
                            ref={audioRef}
                            src={audioUrl}
                            controls
                            className="w-full"
                          />
                        </div>
                      )}

                      {/* Usage stats */}
                      <div className="text-xs text-gray-500 flex items-center justify-between pt-4 border-t border-gray-700">
                        <span>Total generations: {selectedProfile.total_generations}</span>
                        <span>Characters used: {selectedProfile.total_characters_generated.toLocaleString()}</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Failed state */}
                {selectedProfile.status === 'failed' && (
                  <div className="bg-red-900/30 border border-red-700 rounded-lg p-6">
                    <h3 className="text-lg font-medium text-red-300 mb-2">Training Failed</h3>
                    <p className="text-red-400">{selectedProfile.error_message || 'An error occurred during training.'}</p>
                    <button
                      onClick={() => handleStartTraining(selectedProfile.id)}
                      className="mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm"
                    >
                      Retry Training
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-gray-800 rounded-lg p-12 text-center">
                <svg className="w-16 h-16 text-gray-600 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                <p className="text-gray-400">Select a voice profile or create a new one</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Create profile modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-lg shadow-xl w-full max-w-md p-6">
            <h2 className="text-xl font-bold text-white mb-4">Create Voice Profile</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-2">Profile Name</label>
                <input
                  type="text"
                  value={newProfileName}
                  onChange={(e) => setNewProfileName(e.target.value)}
                  placeholder="My Voice"
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400"
                  autoFocus
                />
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">Voice Gender</label>
                <select
                  value={newProfileGender}
                  onChange={(e) => setNewProfileGender(e.target.value as VoiceGender)}
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white"
                >
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="neutral">Neutral</option>
                </select>
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">Accent</label>
                <select
                  value={newProfileAccent}
                  onChange={(e) => setNewProfileAccent(e.target.value as VoiceAccent)}
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white"
                >
                  <option value="american">American</option>
                  <option value="british">British</option>
                  <option value="australian">Australian</option>
                  <option value="indian">Indian</option>
                  <option value="french">French</option>
                  <option value="spanish">Spanish</option>
                  <option value="german">German</option>
                  <option value="other">Other</option>
                </select>
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowCreateModal(false)}
                className="flex-1 px-4 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-medium"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateProfile}
                disabled={!newProfileName.trim() || isSaving}
                className="flex-1 px-4 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium"
              >
                {isSaving ? 'Creating...' : 'Create'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Consent modal */}
      {showConsentModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-lg shadow-xl w-full max-w-lg p-6">
            <h2 className="text-xl font-bold text-white mb-4">Voice Ownership Confirmation</h2>

            <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-4 mb-4">
              <p className="text-yellow-300 text-sm">
                By proceeding, you confirm that:
              </p>
              <ul className="mt-2 space-y-1 text-sm text-yellow-200">
                <li>- This is your own voice</li>
                <li>- OR you have explicit permission from the voice owner</li>
                <li>- You understand this voice will be used for AI content generation</li>
              </ul>
            </div>

            <p className="text-gray-400 text-sm mb-6">
              Misuse of voice cloning technology is prohibited. Viralify reserves the right
              to suspend accounts that violate our terms of service.
            </p>

            <div className="flex gap-3">
              <button
                onClick={() => {
                  setShowConsentModal(false);
                  setProfileToTrain(null);
                }}
                className="flex-1 px-4 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-medium"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmTraining}
                disabled={isTraining}
                className="flex-1 px-4 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium"
              >
                {isTraining ? 'Starting...' : 'I Confirm & Start Training'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
