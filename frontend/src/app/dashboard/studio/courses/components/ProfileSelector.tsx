'use client';

import { useEffect, useState } from 'react';
import { User, ChevronDown } from 'lucide-react';
import { getCreatorProfiles, CreatorProfile } from '@/lib/creator-profiles';

interface ProfileSelectorProps {
  value: string;
  onChange: (profileId: string) => void;
}

export function ProfileSelector({ value, onChange }: ProfileSelectorProps) {
  const [profiles, setProfiles] = useState<CreatorProfile[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Load profiles using the shared creator-profiles library
    const loadProfiles = () => {
      try {
        const loadedProfiles = getCreatorProfiles();
        setProfiles(loadedProfiles);
        // Select default profile if none selected
        if (!value && loadedProfiles.length > 0) {
          const defaultProfile = loadedProfiles.find(p => p.isDefault) || loadedProfiles[0];
          onChange(defaultProfile.id);
        }
      } catch (err) {
        console.error('Error loading profiles:', err);
      } finally {
        setIsLoading(false);
      }
    };

    loadProfiles();
  }, [value, onChange]);

  const selectedProfile = profiles.find(p => p.id === value);

  if (isLoading) {
    return (
      <div className="animate-pulse bg-gray-800 rounded-lg h-12" />
    );
  }

  if (profiles.length === 0) {
    return (
      <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4 text-center">
        <User className="w-8 h-8 text-gray-500 mx-auto mb-2" />
        <p className="text-gray-400 text-sm">No profiles created yet</p>
        <a
          href="/dashboard/settings/profiles"
          className="text-purple-400 hover:text-purple-300 text-sm mt-1 inline-block"
        >
          Create a profile in Settings
        </a>
      </div>
    );
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-left hover:border-gray-600 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-purple-600/20 rounded-full flex items-center justify-center">
            <User className="w-5 h-5 text-purple-400" />
          </div>
          <div>
            <p className="text-white font-medium">
              {selectedProfile?.name || 'Select Profile'}
            </p>
            {selectedProfile && (
              <p className="text-gray-400 text-sm">
                {selectedProfile.niche} • {selectedProfile.tone}
              </p>
            )}
          </div>
        </div>
        <ChevronDown className={`w-5 h-5 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute z-10 w-full mt-2 bg-gray-800 border border-gray-700 rounded-lg shadow-xl overflow-hidden">
          {profiles.map((profile) => (
            <button
              key={profile.id}
              type="button"
              onClick={() => {
                onChange(profile.id);
                setIsOpen(false);
              }}
              className={`w-full flex items-center gap-3 px-4 py-3 hover:bg-gray-700 transition-colors ${
                value === profile.id ? 'bg-gray-700/50' : ''
              }`}
            >
              <div className="w-10 h-10 bg-purple-600/20 rounded-full flex items-center justify-center">
                <User className="w-5 h-5 text-purple-400" />
              </div>
              <div className="text-left">
                <p className="text-white font-medium">{profile.name}</p>
                <p className="text-gray-400 text-sm">
                  {profile.niche} • {profile.tone}
                </p>
              </div>
              {value === profile.id && (
                <div className="ml-auto w-2 h-2 bg-purple-500 rounded-full" />
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
