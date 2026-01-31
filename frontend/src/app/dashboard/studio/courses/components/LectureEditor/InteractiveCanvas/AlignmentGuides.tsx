'use client';

import React, { memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface AlignmentGuide {
  type: 'vertical' | 'horizontal';
  position: number; // percentage
}

interface AlignmentGuidesProps {
  guides: AlignmentGuide[];
}

// Spring animation config for smooth appearance
const guideSpring = {
  type: 'spring' as const,
  stiffness: 400,
  damping: 30,
  mass: 0.5,
};

export const AlignmentGuides = memo(function AlignmentGuides({ guides }: AlignmentGuidesProps) {
  return (
    <div className="absolute inset-0 pointer-events-none z-50">
      <AnimatePresence>
        {guides.map((guide, index) => {
          const key = `${guide.type}-${guide.position.toFixed(2)}-${index}`;

          return (
            <motion.div
              key={key}
              className={`absolute ${
                guide.type === 'vertical' ? 'w-px h-full' : 'h-px w-full'
              }`}
              style={{
                [guide.type === 'vertical' ? 'left' : 'top']: `${guide.position}%`,
              }}
              initial={{
                opacity: 0,
                scale: 0.9,
              }}
              animate={{
                opacity: 1,
                scale: 1,
              }}
              exit={{
                opacity: 0,
                scale: 0.9,
              }}
              transition={guideSpring}
            >
              {/* Main guide line with gradient */}
              <motion.div
                className={`absolute ${
                  guide.type === 'vertical' ? 'w-px h-full' : 'h-px w-full'
                }`}
                style={{
                  background: guide.type === 'vertical'
                    ? 'linear-gradient(to bottom, transparent, rgb(168, 85, 247), rgb(168, 85, 247), transparent)'
                    : 'linear-gradient(to right, transparent, rgb(168, 85, 247), rgb(168, 85, 247), transparent)',
                }}
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.8 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.1 }}
              />

              {/* Glow effect */}
              <motion.div
                className={`absolute ${
                  guide.type === 'vertical' ? 'w-1 h-full -left-px' : 'h-1 w-full -top-px'
                }`}
                style={{
                  background: guide.type === 'vertical'
                    ? 'linear-gradient(to bottom, transparent, rgba(168, 85, 247, 0.3), rgba(168, 85, 247, 0.3), transparent)'
                    : 'linear-gradient(to right, transparent, rgba(168, 85, 247, 0.3), rgba(168, 85, 247, 0.3), transparent)',
                  filter: 'blur(2px)',
                }}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.15 }}
              />

              {/* Top/Left indicator dot */}
              <motion.div
                className={`absolute w-2 h-2 rounded-full ${
                  guide.type === 'vertical' ? '-left-[3px] top-2' : 'left-2 -top-[3px]'
                }`}
                style={{
                  background: 'rgb(168, 85, 247)',
                  boxShadow: '0 0 6px rgba(168, 85, 247, 0.6)',
                }}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0 }}
                transition={guideSpring}
              />

              {/* Bottom/Right indicator dot */}
              <motion.div
                className={`absolute w-2 h-2 rounded-full ${
                  guide.type === 'vertical' ? '-left-[3px] bottom-2' : 'right-2 -top-[3px]'
                }`}
                style={{
                  background: 'rgb(168, 85, 247)',
                  boxShadow: '0 0 6px rgba(168, 85, 247, 0.6)',
                }}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0 }}
                transition={guideSpring}
              />

              {/* Center indicator for center alignment */}
              {Math.abs(guide.position - 50) < 0.5 && (
                <motion.div
                  className={`absolute flex items-center justify-center ${
                    guide.type === 'vertical'
                      ? 'top-1/2 -translate-y-1/2 -left-3'
                      : 'left-1/2 -translate-x-1/2 -top-3'
                  }`}
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0, opacity: 0 }}
                  transition={guideSpring}
                >
                  <div
                    className="w-6 h-6 rounded-full flex items-center justify-center text-xs"
                    style={{
                      background: 'rgba(168, 85, 247, 0.9)',
                      boxShadow: '0 2px 8px rgba(168, 85, 247, 0.4)',
                    }}
                  >
                    <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z" />
                    </svg>
                  </div>
                </motion.div>
              )}
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
});

export default AlignmentGuides;
