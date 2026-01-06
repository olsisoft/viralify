'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  User, Plus, Edit2, Trash2, Star, Check, X,
  ChevronDown, ChevronUp, Target, MessageSquare,
  Users, Sparkles, Save, ArrowLeft
} from 'lucide-react';
import Link from 'next/link';
import toast from 'react-hot-toast';
import {
  CreatorProfile,
  getCreatorProfiles,
  createProfile,
  updateProfile,
  deleteProfile,
  createEmptyProfile,
  NICHE_OPTIONS,
  TONE_OPTIONS,
  HOOK_OPTIONS,
  GOAL_OPTIONS,
  ContentTone,
  HookStyle,
  ContentGoal,
  AgeRange,
  AudienceLevel,
  LanguageLevel,
} from '@/lib/creator-profiles';

export default function CreatorProfilesPage() {
  const [profiles, setProfiles] = useState<CreatorProfile[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [formData, setFormData] = useState(createEmptyProfile());

  useEffect(() => {
    setProfiles(getCreatorProfiles());
  }, []);

  const handleCreate = () => {
    setFormData(createEmptyProfile());
    setIsCreating(true);
    setEditingId(null);
  };

  const handleEdit = (profile: CreatorProfile) => {
    setFormData(profile);
    setEditingId(profile.id);
    setIsCreating(false);
  };

  const handleSave = () => {
    if (!formData.name.trim()) {
      toast.error('Profile name is required');
      return;
    }
    if (!formData.brandName.trim()) {
      toast.error('Brand name is required');
      return;
    }
    if (!formData.niche) {
      toast.error('Please select a niche');
      return;
    }

    if (isCreating) {
      const newProfile = createProfile(formData);
      setProfiles(prev => [...prev, newProfile]);
      toast.success('Profile created!');
    } else if (editingId) {
      const updated = updateProfile(editingId, formData);
      if (updated) {
        setProfiles(getCreatorProfiles());
        toast.success('Profile updated!');
      }
    }

    setIsCreating(false);
    setEditingId(null);
    setFormData(createEmptyProfile());
  };

  const handleDelete = (id: string) => {
    if (confirm('Are you sure you want to delete this profile?')) {
      deleteProfile(id);
      setProfiles(getCreatorProfiles());
      toast.success('Profile deleted');
    }
  };

  const handleSetDefault = (id: string) => {
    updateProfile(id, { isDefault: true });
    setProfiles(getCreatorProfiles());
    toast.success('Default profile updated');
  };

  const handleCancel = () => {
    setIsCreating(false);
    setEditingId(null);
    setFormData(createEmptyProfile());
  };

  const updateFormField = (field: string, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const addToArray = (field: string, value: string) => {
    if (!value.trim()) return;
    setFormData(prev => ({
      ...prev,
      [field]: [...(prev as any)[field], value.trim()],
    }));
  };

  const removeFromArray = (field: string, index: number) => {
    setFormData(prev => ({
      ...prev,
      [field]: (prev as any)[field].filter((_: any, i: number) => i !== index),
    }));
  };

  const isEditing = isCreating || editingId !== null;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link
            href="/dashboard/settings"
            className="p-2 text-gray-400 hover:text-white hover:bg-white/5 rounded-xl transition"
          >
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-white">Creator Profiles</h1>
            <p className="text-gray-400 text-sm">
              Manage your content personas for AI-generated scripts
            </p>
          </div>
        </div>
        {!isEditing && (
          <button
            onClick={handleCreate}
            className="flex items-center gap-2 px-4 py-2 bg-pink-600 text-white rounded-xl hover:bg-pink-700 transition"
          >
            <Plus className="w-4 h-4" />
            New Profile
          </button>
        )}
      </div>

      {/* Profile Editor */}
      <AnimatePresence>
        {isEditing && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="bg-gray-800 rounded-2xl border border-gray-700 overflow-hidden"
          >
            <div className="p-6 border-b border-gray-700">
              <h2 className="text-lg font-semibold text-white">
                {isCreating ? 'Create New Profile' : 'Edit Profile'}
              </h2>
            </div>

            <div className="p-6 space-y-6 max-h-[70vh] overflow-y-auto">
              {/* Basic Info */}
              <div className="space-y-4">
                <h3 className="text-sm font-medium text-pink-400 uppercase tracking-wider">
                  Basic Information
                </h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Profile Name *</label>
                    <input
                      type="text"
                      value={formData.name}
                      onChange={(e) => updateFormField('name', e.target.value)}
                      placeholder="e.g., Tech Expert, Business Coach"
                      className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-pink-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Brand Name *</label>
                    <input
                      type="text"
                      value={formData.brandName}
                      onChange={(e) => updateFormField('brandName', e.target.value)}
                      placeholder="e.g., Tech with Alex"
                      className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-pink-500"
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Tagline</label>
                  <input
                    type="text"
                    value={formData.tagline}
                    onChange={(e) => updateFormField('tagline', e.target.value)}
                    placeholder="e.g., Making tech simple for everyone"
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-pink-500"
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Unique Value Proposition</label>
                  <textarea
                    value={formData.uniqueValue}
                    onChange={(e) => updateFormField('uniqueValue', e.target.value)}
                    placeholder="What makes your content different? e.g., I explain complex concepts in 60 seconds"
                    rows={2}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-pink-500 resize-none"
                  />
                </div>
              </div>

              {/* Content Style */}
              <div className="space-y-4">
                <h3 className="text-sm font-medium text-pink-400 uppercase tracking-wider">
                  Content Style
                </h3>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Niche *</label>
                  <select
                    value={formData.niche}
                    onChange={(e) => updateFormField('niche', e.target.value)}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-pink-500"
                  >
                    <option value="">Select a niche...</option>
                    {NICHE_OPTIONS.map(niche => (
                      <option key={niche} value={niche}>{niche}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Sub-niches</label>
                  <div className="flex flex-wrap gap-2 mb-2">
                    {formData.subNiches.map((sub, idx) => (
                      <span key={idx} className="px-3 py-1 bg-gray-700 text-gray-300 rounded-full text-sm flex items-center gap-1">
                        {sub}
                        <button onClick={() => removeFromArray('subNiches', idx)} className="text-gray-500 hover:text-red-400">
                          <X className="w-3 h-3" />
                        </button>
                      </span>
                    ))}
                  </div>
                  <input
                    type="text"
                    placeholder="Add sub-niche and press Enter"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        addToArray('subNiches', (e.target as HTMLInputElement).value);
                        (e.target as HTMLInputElement).value = '';
                      }
                    }}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-pink-500"
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Tone</label>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    {TONE_OPTIONS.map(tone => (
                      <button
                        key={tone.id}
                        onClick={() => updateFormField('tone', tone.id)}
                        className={`p-3 rounded-xl border text-left transition ${
                          formData.tone === tone.id
                            ? 'bg-pink-600/20 border-pink-500'
                            : 'bg-gray-700 border-gray-600 hover:border-gray-500'
                        }`}
                      >
                        <div className="text-lg mb-1">{tone.emoji}</div>
                        <div className="text-sm font-medium text-white">{tone.name}</div>
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Hook Style</label>
                  <div className="grid grid-cols-2 gap-2">
                    {HOOK_OPTIONS.map(hook => (
                      <button
                        key={hook.id}
                        onClick={() => updateFormField('hookStyle', hook.id)}
                        className={`p-3 rounded-xl border text-left transition ${
                          formData.hookStyle === hook.id
                            ? 'bg-pink-600/20 border-pink-500'
                            : 'bg-gray-700 border-gray-600 hover:border-gray-500'
                        }`}
                      >
                        <div className="text-sm font-medium text-white">{hook.name}</div>
                        <div className="text-xs text-gray-400 mt-1">{hook.example}</div>
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">CTA Style</label>
                  <input
                    type="text"
                    value={formData.ctaStyle}
                    onChange={(e) => updateFormField('ctaStyle', e.target.value)}
                    placeholder="e.g., Follow for more tips! DM me 'START' for..."
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-pink-500"
                  />
                </div>
              </div>

              {/* Target Audience */}
              <div className="space-y-4">
                <h3 className="text-sm font-medium text-pink-400 uppercase tracking-wider">
                  Target Audience
                </h3>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Audience Description</label>
                  <textarea
                    value={formData.audienceDescription}
                    onChange={(e) => updateFormField('audienceDescription', e.target.value)}
                    placeholder="Describe your ideal viewer... e.g., Junior developers who want to level up"
                    rows={2}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-pink-500 resize-none"
                  />
                </div>
                <div className="grid md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Age Range</label>
                    <select
                      value={formData.audienceAgeRange}
                      onChange={(e) => updateFormField('audienceAgeRange', e.target.value as AgeRange)}
                      className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-pink-500"
                    >
                      <option value="18-24">18-24</option>
                      <option value="25-34">25-34</option>
                      <option value="35-44">35-44</option>
                      <option value="45-54">45-54</option>
                      <option value="55+">55+</option>
                      <option value="all">All ages</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Expertise Level</label>
                    <select
                      value={formData.audienceLevel}
                      onChange={(e) => updateFormField('audienceLevel', e.target.value as AudienceLevel)}
                      className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-pink-500"
                    >
                      <option value="beginner">Beginner</option>
                      <option value="intermediate">Intermediate</option>
                      <option value="advanced">Advanced</option>
                      <option value="mixed">Mixed</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Language Level</label>
                    <select
                      value={formData.languageLevel}
                      onChange={(e) => updateFormField('languageLevel', e.target.value as LanguageLevel)}
                      className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-pink-500"
                    >
                      <option value="simple">Simple (no jargon)</option>
                      <option value="moderate">Moderate (some terms OK)</option>
                      <option value="technical">Technical</option>
                      <option value="jargon-heavy">Jargon-heavy (expert)</option>
                    </select>
                  </div>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Pain Points</label>
                  <div className="flex flex-wrap gap-2 mb-2">
                    {formData.audiencePainPoints.map((point, idx) => (
                      <span key={idx} className="px-3 py-1 bg-red-900/30 text-red-300 rounded-full text-sm flex items-center gap-1">
                        {point}
                        <button onClick={() => removeFromArray('audiencePainPoints', idx)} className="text-red-400 hover:text-red-300">
                          <X className="w-3 h-3" />
                        </button>
                      </span>
                    ))}
                  </div>
                  <input
                    type="text"
                    placeholder="Add pain point and press Enter"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        addToArray('audiencePainPoints', (e.target as HTMLInputElement).value);
                        (e.target as HTMLInputElement).value = '';
                      }
                    }}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-pink-500"
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Audience Goals</label>
                  <div className="flex flex-wrap gap-2 mb-2">
                    {formData.audienceGoals.map((goal, idx) => (
                      <span key={idx} className="px-3 py-1 bg-green-900/30 text-green-300 rounded-full text-sm flex items-center gap-1">
                        {goal}
                        <button onClick={() => removeFromArray('audienceGoals', idx)} className="text-green-400 hover:text-green-300">
                          <X className="w-3 h-3" />
                        </button>
                      </span>
                    ))}
                  </div>
                  <input
                    type="text"
                    placeholder="Add goal and press Enter"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        addToArray('audienceGoals', (e.target as HTMLInputElement).value);
                        (e.target as HTMLInputElement).value = '';
                      }
                    }}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-pink-500"
                  />
                </div>
              </div>

              {/* Content Goals */}
              <div className="space-y-4">
                <h3 className="text-sm font-medium text-pink-400 uppercase tracking-wider">
                  Content Goals
                </h3>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Primary Goal</label>
                  <div className="grid grid-cols-4 gap-2">
                    {GOAL_OPTIONS.map(goal => (
                      <button
                        key={goal.id}
                        onClick={() => updateFormField('primaryGoal', goal.id)}
                        className={`p-3 rounded-xl border text-center transition ${
                          formData.primaryGoal === goal.id
                            ? 'bg-pink-600/20 border-pink-500'
                            : 'bg-gray-700 border-gray-600 hover:border-gray-500'
                        }`}
                      >
                        <div className="text-xl mb-1">{goal.emoji}</div>
                        <div className="text-xs text-white">{goal.name}</div>
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Secondary Goals</label>
                  <div className="grid grid-cols-4 gap-2">
                    {GOAL_OPTIONS.filter(g => g.id !== formData.primaryGoal).map(goal => (
                      <button
                        key={goal.id}
                        onClick={() => {
                          const current = formData.secondaryGoals;
                          if (current.includes(goal.id)) {
                            updateFormField('secondaryGoals', current.filter(g => g !== goal.id));
                          } else {
                            updateFormField('secondaryGoals', [...current, goal.id]);
                          }
                        }}
                        className={`p-3 rounded-xl border text-center transition ${
                          formData.secondaryGoals.includes(goal.id)
                            ? 'bg-purple-600/20 border-purple-500'
                            : 'bg-gray-700 border-gray-600 hover:border-gray-500'
                        }`}
                      >
                        <div className="text-xl mb-1">{goal.emoji}</div>
                        <div className="text-xs text-white">{goal.name}</div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Additional */}
              <div className="space-y-4">
                <h3 className="text-sm font-medium text-pink-400 uppercase tracking-wider">
                  Additional Info
                </h3>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Competitors / Inspiration</label>
                  <div className="flex flex-wrap gap-2 mb-2">
                    {formData.competitors.map((comp, idx) => (
                      <span key={idx} className="px-3 py-1 bg-gray-700 text-gray-300 rounded-full text-sm flex items-center gap-1">
                        {comp}
                        <button onClick={() => removeFromArray('competitors', idx)} className="text-gray-500 hover:text-red-400">
                          <X className="w-3 h-3" />
                        </button>
                      </span>
                    ))}
                  </div>
                  <input
                    type="text"
                    placeholder="Add @handle and press Enter"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        addToArray('competitors', (e.target as HTMLInputElement).value);
                        (e.target as HTMLInputElement).value = '';
                      }
                    }}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-pink-500"
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Style Notes</label>
                  <textarea
                    value={formData.styleNotes}
                    onChange={(e) => updateFormField('styleNotes', e.target.value)}
                    placeholder="Any specific style preferences... e.g., Direct, no fluff, lots of examples"
                    rows={2}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-pink-500 resize-none"
                  />
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="p-6 border-t border-gray-700 flex justify-end gap-3">
              <button
                onClick={handleCancel}
                className="px-4 py-2 bg-gray-700 text-gray-300 rounded-xl hover:bg-gray-600 transition"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                className="px-4 py-2 bg-pink-600 text-white rounded-xl hover:bg-pink-700 transition flex items-center gap-2"
              >
                <Save className="w-4 h-4" />
                {isCreating ? 'Create Profile' : 'Save Changes'}
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Profiles List */}
      {!isEditing && (
        <div className="space-y-4">
          {profiles.length === 0 ? (
            <div className="text-center py-12 bg-gray-800/50 rounded-2xl border border-gray-700">
              <User className="w-12 h-12 text-gray-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-white mb-2">No profiles yet</h3>
              <p className="text-gray-400 mb-4">Create your first creator profile to get started</p>
              <button
                onClick={handleCreate}
                className="px-4 py-2 bg-pink-600 text-white rounded-xl hover:bg-pink-700 transition"
              >
                Create Profile
              </button>
            </div>
          ) : (
            profiles.map((profile) => (
              <motion.div
                key={profile.id}
                layout
                className="bg-gray-800 rounded-2xl border border-gray-700 overflow-hidden"
              >
                {/* Profile Header */}
                <div
                  className="p-4 flex items-center justify-between cursor-pointer hover:bg-gray-700/30 transition"
                  onClick={() => setExpandedId(expandedId === profile.id ? null : profile.id)}
                >
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-pink-500 to-purple-500 flex items-center justify-center">
                      <User className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="font-semibold text-white">{profile.name}</h3>
                        {profile.isDefault && (
                          <span className="px-2 py-0.5 bg-yellow-500/20 text-yellow-400 text-xs rounded-full flex items-center gap-1">
                            <Star className="w-3 h-3" />
                            Default
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-400">{profile.brandName} - {profile.niche}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {!profile.isDefault && (
                      <button
                        onClick={(e) => { e.stopPropagation(); handleSetDefault(profile.id); }}
                        className="p-2 text-gray-400 hover:text-yellow-400 transition"
                        title="Set as default"
                      >
                        <Star className="w-4 h-4" />
                      </button>
                    )}
                    <button
                      onClick={(e) => { e.stopPropagation(); handleEdit(profile); }}
                      className="p-2 text-gray-400 hover:text-white transition"
                    >
                      <Edit2 className="w-4 h-4" />
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); handleDelete(profile.id); }}
                      className="p-2 text-gray-400 hover:text-red-400 transition"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                    {expandedId === profile.id ? (
                      <ChevronUp className="w-5 h-5 text-gray-400" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-gray-400" />
                    )}
                  </div>
                </div>

                {/* Expanded Details */}
                <AnimatePresence>
                  {expandedId === profile.id && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="overflow-hidden"
                    >
                      <div className="p-4 pt-0 grid md:grid-cols-3 gap-4 text-sm">
                        <div className="space-y-3">
                          <div>
                            <span className="text-gray-500">Tagline:</span>
                            <p className="text-gray-300">{profile.tagline || '-'}</p>
                          </div>
                          <div>
                            <span className="text-gray-500">Unique Value:</span>
                            <p className="text-gray-300">{profile.uniqueValue || '-'}</p>
                          </div>
                          <div>
                            <span className="text-gray-500">Tone:</span>
                            <p className="text-gray-300 capitalize">{profile.tone}</p>
                          </div>
                        </div>
                        <div className="space-y-3">
                          <div>
                            <span className="text-gray-500">Audience:</span>
                            <p className="text-gray-300">{profile.audienceDescription || '-'}</p>
                          </div>
                          <div>
                            <span className="text-gray-500">Age / Level:</span>
                            <p className="text-gray-300">{profile.audienceAgeRange} / {profile.audienceLevel}</p>
                          </div>
                          <div>
                            <span className="text-gray-500">Hook Style:</span>
                            <p className="text-gray-300 capitalize">{profile.hookStyle.replace('-', ' ')}</p>
                          </div>
                        </div>
                        <div className="space-y-3">
                          <div>
                            <span className="text-gray-500">Primary Goal:</span>
                            <p className="text-gray-300 capitalize">{profile.primaryGoal.replace('-', ' ')}</p>
                          </div>
                          <div>
                            <span className="text-gray-500">CTA:</span>
                            <p className="text-gray-300">{profile.ctaStyle || '-'}</p>
                          </div>
                          <div>
                            <span className="text-gray-500">Sub-niches:</span>
                            <p className="text-gray-300">{profile.subNiches.join(', ') || '-'}</p>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
