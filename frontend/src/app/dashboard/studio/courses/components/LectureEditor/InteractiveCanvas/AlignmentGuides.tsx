'use client';

import React, { memo } from 'react';

interface AlignmentGuide {
  type: 'vertical' | 'horizontal';
  position: number; // percentage
}

interface AlignmentGuidesProps {
  guides: AlignmentGuide[];
}

export const AlignmentGuides = memo(function AlignmentGuides({ guides }: AlignmentGuidesProps) {
  if (guides.length === 0) return null;

  return (
    <div className="absolute inset-0 pointer-events-none z-50">
      {guides.map((guide, index) => (
        <div
          key={`${guide.type}-${guide.position}-${index}`}
          className={`absolute bg-purple-500 ${
            guide.type === 'vertical' ? 'w-px h-full' : 'h-px w-full'
          }`}
          style={{
            [guide.type === 'vertical' ? 'left' : 'top']: `${guide.position}%`,
            opacity: 0.7,
          }}
        >
          {/* Guide indicator dots at ends */}
          <div
            className={`absolute w-1.5 h-1.5 bg-purple-500 rounded-full ${
              guide.type === 'vertical' ? '-left-[2px] top-0' : 'left-0 -top-[2px]'
            }`}
          />
          <div
            className={`absolute w-1.5 h-1.5 bg-purple-500 rounded-full ${
              guide.type === 'vertical' ? '-left-[2px] bottom-0' : 'right-0 -top-[2px]'
            }`}
          />
        </div>
      ))}
    </div>
  );
});

export default AlignmentGuides;
