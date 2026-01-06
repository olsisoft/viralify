'use client';

import React, { useState } from 'react';
import { PlatformType, PLATFORMS, ContentAdaptation } from '@/services/api';
import { Video, Instagram, Youtube, AlertTriangle, Check, Info } from 'lucide-react';

interface ContentAdaptationPreviewProps {
  adaptations: ContentAdaptation[];
  originalDuration?: number;
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

export function ContentAdaptationPreview({
  adaptations,
  originalDuration,
}: ContentAdaptationPreviewProps) {
  const [selectedPlatform, setSelectedPlatform] = useState<PlatformType>(
    adaptations[0]?.platform || 'TIKTOK'
  );

  const selectedAdaptation = adaptations.find((a) => a.platform === selectedPlatform);

  if (adaptations.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-300">Content Preview by Platform</h3>
        {adaptations.some((a) => a.contentWasModified) && (
          <span className="flex items-center gap-1 text-xs text-yellow-500">
            <AlertTriangle className="w-3 h-3" />
            Content will be adapted
          </span>
        )}
      </div>

      {/* Platform tabs */}
      <div className="flex gap-2">
        {adaptations.map((adaptation) => {
          const platformInfo = PLATFORMS[adaptation.platform];
          const isSelected = adaptation.platform === selectedPlatform;

          return (
            <button
              key={adaptation.platform}
              onClick={() => setSelectedPlatform(adaptation.platform)}
              className={`
                flex items-center gap-2 px-3 py-2 rounded-lg transition-all
                ${isSelected
                  ? 'bg-purple-500/20 border border-purple-500'
                  : 'bg-gray-700/50 border border-transparent hover:bg-gray-700'
                }
              `}
            >
              <PlatformIcon
                platform={adaptation.platform}
                className="w-4 h-4"
                style={{ color: platformInfo.color } as any}
              />
              <span className="text-sm text-white">{platformInfo.displayName}</span>
              {adaptation.contentWasModified && (
                <span className="w-2 h-2 rounded-full bg-yellow-500" />
              )}
            </button>
          );
        })}
      </div>

      {/* Adaptation preview */}
      {selectedAdaptation && (
        <div className="bg-gray-800/50 rounded-xl p-4 space-y-4">
          {/* Duration warning */}
          {selectedAdaptation.suggestedDurationSeconds &&
            originalDuration &&
            selectedAdaptation.suggestedDurationSeconds < originalDuration && (
              <div className="flex items-start gap-3 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                <AlertTriangle className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-yellow-500">Duration Adjustment</p>
                  <p className="text-sm text-gray-400">
                    Video will be trimmed from {originalDuration}s to{' '}
                    {selectedAdaptation.suggestedDurationSeconds}s for{' '}
                    {PLATFORMS[selectedAdaptation.platform].displayName}
                  </p>
                </div>
              </div>
            )}

          {/* Title (if applicable) */}
          {selectedAdaptation.title && (
            <div>
              <label className="block text-xs font-medium text-gray-400 mb-1">Title</label>
              <p className="text-sm text-white bg-gray-700/50 rounded-lg p-3">
                {selectedAdaptation.title}
              </p>
            </div>
          )}

          {/* Caption/Description */}
          <div>
            <label className="block text-xs font-medium text-gray-400 mb-1">
              {selectedAdaptation.platform === 'YOUTUBE' ? 'Description' : 'Caption'}
            </label>
            <p className="text-sm text-white bg-gray-700/50 rounded-lg p-3 whitespace-pre-wrap">
              {selectedAdaptation.platform === 'YOUTUBE'
                ? selectedAdaptation.description || selectedAdaptation.caption
                : selectedAdaptation.caption}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              {(selectedAdaptation.platform === 'YOUTUBE'
                ? selectedAdaptation.description || selectedAdaptation.caption
                : selectedAdaptation.caption
              )?.length || 0}{' '}
              / {PLATFORMS[selectedAdaptation.platform].maxCaptionLength} characters
            </p>
          </div>

          {/* Hashtags/Tags */}
          <div>
            <label className="block text-xs font-medium text-gray-400 mb-1">
              {selectedAdaptation.platform === 'YOUTUBE' ? 'Tags' : 'Hashtags'}
            </label>
            <div className="flex flex-wrap gap-2">
              {(selectedAdaptation.platform === 'YOUTUBE'
                ? selectedAdaptation.tags
                : selectedAdaptation.hashtags
              )?.map((tag, index) => (
                <span
                  key={index}
                  className="px-2 py-1 text-xs bg-purple-500/20 text-purple-300 rounded-full"
                >
                  {selectedAdaptation.platform === 'YOUTUBE' ? tag : `#${tag.replace(/^#/, '')}`}
                </span>
              )) || <span className="text-xs text-gray-500">No hashtags</span>}
            </div>
            <p className="text-xs text-gray-500 mt-1">
              {(selectedAdaptation.platform === 'YOUTUBE'
                ? selectedAdaptation.tags
                : selectedAdaptation.hashtags
              )?.length || 0}{' '}
              {selectedAdaptation.platform === 'YOUTUBE' ? 'tags' : 'hashtags'}
            </p>
          </div>

          {/* Adaptation notes */}
          {selectedAdaptation.adaptationNotes?.length > 0 && (
            <div className="border-t border-gray-700 pt-4">
              <label className="flex items-center gap-1 text-xs font-medium text-gray-400 mb-2">
                <Info className="w-3 h-3" />
                Adaptation Notes
              </label>
              <ul className="space-y-1">
                {selectedAdaptation.adaptationNotes.map((note, index) => (
                  <li key={index} className="flex items-start gap-2 text-xs text-gray-400">
                    <Check className="w-3 h-3 text-green-500 flex-shrink-0 mt-0.5" />
                    {note}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ContentAdaptationPreview;
