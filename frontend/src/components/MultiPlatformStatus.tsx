'use client';

import React from 'react';
import { PlatformStatus, PlatformType, PLATFORMS } from '@/services/api';
import {
  Video,
  Instagram,
  Youtube,
  Check,
  Clock,
  Loader2,
  XCircle,
  ExternalLink,
  AlertTriangle,
} from 'lucide-react';

interface MultiPlatformStatusProps {
  platformStatuses: PlatformStatus[];
  targetPlatforms: PlatformType[];
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

const StatusIcon: React.FC<{ status: PlatformStatus['status']; className?: string }> = ({
  status,
  className,
}) => {
  switch (status) {
    case 'published':
      return <Check className={`${className} text-green-500`} />;
    case 'pending':
      return <Clock className={`${className} text-yellow-500`} />;
    case 'processing':
    case 'uploading':
      return <Loader2 className={`${className} text-blue-500 animate-spin`} />;
    case 'failed':
      return <XCircle className={`${className} text-red-500`} />;
    case 'cancelled':
      return <XCircle className={`${className} text-gray-500`} />;
    case 'skipped':
      return <AlertTriangle className={`${className} text-yellow-500`} />;
    default:
      return <Clock className={`${className} text-gray-500`} />;
  }
};

const statusLabels: Record<PlatformStatus['status'], string> = {
  pending: 'Pending',
  processing: 'Processing',
  uploading: 'Uploading',
  published: 'Published',
  failed: 'Failed',
  cancelled: 'Cancelled',
  skipped: 'Skipped',
};

const statusColors: Record<PlatformStatus['status'], string> = {
  pending: 'bg-yellow-500/20 text-yellow-500',
  processing: 'bg-blue-500/20 text-blue-500',
  uploading: 'bg-blue-500/20 text-blue-500',
  published: 'bg-green-500/20 text-green-500',
  failed: 'bg-red-500/20 text-red-500',
  cancelled: 'bg-gray-500/20 text-gray-500',
  skipped: 'bg-yellow-500/20 text-yellow-500',
};

export function MultiPlatformStatus({ platformStatuses, targetPlatforms }: MultiPlatformStatusProps) {
  // Create a map of statuses by platform
  const statusMap = new Map(platformStatuses.map((s) => [s.platform, s]));

  return (
    <div className="space-y-2">
      {targetPlatforms.map((platform) => {
        const status = statusMap.get(platform);
        const platformInfo = PLATFORMS[platform];

        return (
          <div
            key={platform}
            className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg"
          >
            <div className="flex items-center gap-3">
              {/* Platform icon */}
              <div
                className="w-8 h-8 rounded-full flex items-center justify-center"
                style={{ backgroundColor: `${platformInfo.color}20` }}
              >
                <PlatformIcon
                  platform={platform}
                  className="w-4 h-4"
                  style={{ color: platformInfo.color } as any}
                />
              </div>

              {/* Platform name */}
              <span className="text-sm font-medium text-white">{platformInfo.displayName}</span>
            </div>

            <div className="flex items-center gap-3">
              {/* Status badge */}
              {status ? (
                <>
                  <div
                    className={`flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium ${
                      statusColors[status.status]
                    }`}
                  >
                    <StatusIcon status={status.status} className="w-3 h-3" />
                    {statusLabels[status.status]}
                  </div>

                  {/* Link to post if published */}
                  {status.status === 'published' && status.platformShareUrl && (
                    <a
                      href={status.platformShareUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="p-1.5 text-gray-400 hover:text-white transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                    </a>
                  )}

                  {/* Error message if failed */}
                  {status.status === 'failed' && status.errorMessage && (
                    <span
                      className="text-xs text-red-400 max-w-[150px] truncate"
                      title={status.errorMessage}
                    >
                      {status.errorMessage}
                    </span>
                  )}
                </>
              ) : (
                <div className="flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium bg-gray-600/50 text-gray-400">
                  <Clock className="w-3 h-3" />
                  Queued
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Compact version for calendar/list views
export function PlatformStatusIcons({ platformStatuses, targetPlatforms }: MultiPlatformStatusProps) {
  const statusMap = new Map(platformStatuses.map((s) => [s.platform, s]));

  return (
    <div className="flex items-center gap-1">
      {targetPlatforms.map((platform) => {
        const status = statusMap.get(platform);
        const platformInfo = PLATFORMS[platform];

        return (
          <div
            key={platform}
            className={`
              relative w-6 h-6 rounded-full flex items-center justify-center
              ${status?.status === 'published' ? 'bg-green-500/20' : 'bg-gray-600/50'}
            `}
            title={`${platformInfo.displayName}: ${status ? statusLabels[status.status] : 'Pending'}`}
          >
            <PlatformIcon
              platform={platform}
              className="w-3 h-3"
              style={{ color: status?.status === 'published' ? '#22c55e' : platformInfo.color } as any}
            />
            {status && status.status !== 'pending' && (
              <div className="absolute -bottom-0.5 -right-0.5">
                <StatusIcon status={status.status} className="w-2.5 h-2.5" />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default MultiPlatformStatus;
