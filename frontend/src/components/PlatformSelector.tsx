'use client';

import React from 'react';
import { PlatformType, PlatformAccount, PLATFORMS } from '@/services/api';
import { Video, Instagram, Youtube, Check, AlertCircle } from 'lucide-react';

interface PlatformSelectorProps {
  selectedPlatforms: PlatformType[];
  onSelectionChange: (platforms: PlatformType[]) => void;
  connectedAccounts?: PlatformAccount[];
  disabled?: boolean;
}

const PlatformIcon: React.FC<{ platform: PlatformType; className?: string }> = ({ platform, className }) => {
  switch (platform) {
    case 'TIKTOK':
      return <Video className={className} />;
    case 'INSTAGRAM':
      return <Instagram className={className} />;
    case 'YOUTUBE':
      return <Youtube className={className} />;
    default:
      return <Video className={className} />;
  }
};

export function PlatformSelector({
  selectedPlatforms,
  onSelectionChange,
  connectedAccounts = [],
  disabled = false,
}: PlatformSelectorProps) {
  const isConnected = (platform: PlatformType) => {
    return connectedAccounts.some(
      (account) => account.platform === platform && account.accountStatus === 'active'
    );
  };

  const getAccount = (platform: PlatformType) => {
    return connectedAccounts.find((account) => account.platform === platform);
  };

  const togglePlatform = (platform: PlatformType) => {
    if (disabled) return;

    if (selectedPlatforms.includes(platform)) {
      // Don't allow deselecting if it's the last one
      if (selectedPlatforms.length > 1) {
        onSelectionChange(selectedPlatforms.filter((p) => p !== platform));
      }
    } else {
      // Only allow selecting connected platforms
      if (isConnected(platform)) {
        onSelectionChange([...selectedPlatforms, platform]);
      }
    }
  };

  return (
    <div className="space-y-3">
      <label className="block text-sm font-medium text-gray-300">
        Publish to platforms
      </label>
      <div className="grid grid-cols-3 gap-3">
        {(Object.keys(PLATFORMS) as PlatformType[]).map((platform) => {
          const info = PLATFORMS[platform];
          const connected = isConnected(platform);
          const selected = selectedPlatforms.includes(platform);
          const account = getAccount(platform);

          return (
            <button
              key={platform}
              type="button"
              onClick={() => togglePlatform(platform)}
              disabled={disabled || !connected}
              className={`
                relative p-4 rounded-xl border-2 transition-all duration-200
                ${selected
                  ? 'border-purple-500 bg-purple-500/20'
                  : connected
                    ? 'border-gray-600 bg-gray-700/50 hover:border-gray-500'
                    : 'border-gray-700 bg-gray-800/50 opacity-50 cursor-not-allowed'
                }
              `}
            >
              {/* Selection indicator */}
              {selected && (
                <div className="absolute -top-2 -right-2 w-5 h-5 bg-purple-500 rounded-full flex items-center justify-center">
                  <Check className="w-3 h-3 text-white" />
                </div>
              )}

              {/* Platform icon and name */}
              <div className="flex flex-col items-center gap-2">
                <div
                  className="w-10 h-10 rounded-full flex items-center justify-center"
                  style={{ backgroundColor: `${info.color}20` }}
                >
                  <PlatformIcon
                    platform={platform}
                    className="w-5 h-5"
                    style={{ color: info.color } as any}
                  />
                </div>
                <span className="text-sm font-medium text-white">{info.displayName}</span>

                {/* Connection status */}
                {connected ? (
                  <span className="text-xs text-gray-400">
                    @{account?.platformUsername || 'connected'}
                  </span>
                ) : (
                  <span className="text-xs text-yellow-500 flex items-center gap-1">
                    <AlertCircle className="w-3 h-3" />
                    Not connected
                  </span>
                )}
              </div>

              {/* Platform limits */}
              <div className="mt-2 text-xs text-gray-500 text-center">
                Max {info.maxDurationSeconds}s
              </div>
            </button>
          );
        })}
      </div>

      {/* Selection summary */}
      {selectedPlatforms.length > 0 && (
        <p className="text-sm text-gray-400">
          Posting to {selectedPlatforms.length} platform{selectedPlatforms.length > 1 ? 's' : ''}:{' '}
          {selectedPlatforms.map((p) => PLATFORMS[p].displayName).join(', ')}
        </p>
      )}
    </div>
  );
}

export default PlatformSelector;
