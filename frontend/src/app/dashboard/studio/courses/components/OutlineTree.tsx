'use client';

import { useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  GripVertical,
  FileVideo,
  FolderOpen,
  Clock,
} from 'lucide-react';
import type { CourseOutline, Section, Lecture } from '../lib/course-types';

interface OutlineTreeProps {
  outline: CourseOutline;
  onReorder?: (sections: Section[]) => void;
  readonly?: boolean;
}

export function OutlineTree({ outline, onReorder, readonly = false }: OutlineTreeProps) {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(outline.sections.map(s => s.id))
  );
  const [draggedItem, setDraggedItem] = useState<{ type: 'section' | 'lecture'; id: string } | null>(null);

  const toggleSection = (sectionId: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(sectionId)) {
      newExpanded.delete(sectionId);
    } else {
      newExpanded.add(sectionId);
    }
    setExpandedSections(newExpanded);
  };

  const handleDragStart = (type: 'section' | 'lecture', id: string) => {
    if (readonly) return;
    setDraggedItem({ type, id });
  };

  const handleDragOver = (e: React.DragEvent) => {
    if (readonly) return;
    e.preventDefault();
  };

  const handleDropOnSection = (targetSectionId: string, targetIndex: number) => {
    if (readonly || !draggedItem || !onReorder) return;

    const newSections = [...outline.sections];

    if (draggedItem.type === 'section') {
      const sourceIndex = newSections.findIndex(s => s.id === draggedItem.id);
      if (sourceIndex === -1 || sourceIndex === targetIndex) return;

      const [removed] = newSections.splice(sourceIndex, 1);
      const adjustedIndex = sourceIndex < targetIndex ? targetIndex - 1 : targetIndex;
      newSections.splice(adjustedIndex, 0, removed);

      // Update order
      newSections.forEach((s, i) => s.order = i);
      onReorder(newSections);
    }

    setDraggedItem(null);
  };

  const handleDropOnLecture = (
    targetSectionId: string,
    targetLectureIndex: number
  ) => {
    if (readonly || !draggedItem || draggedItem.type !== 'lecture' || !onReorder) return;

    const newSections = [...outline.sections];

    // Find source section and lecture
    let sourceSection: Section | undefined;
    let sourceLectureIndex = -1;

    for (const section of newSections) {
      const idx = section.lectures.findIndex(l => l.id === draggedItem.id);
      if (idx !== -1) {
        sourceSection = section;
        sourceLectureIndex = idx;
        break;
      }
    }

    if (!sourceSection || sourceLectureIndex === -1) return;

    const targetSection = newSections.find(s => s.id === targetSectionId);
    if (!targetSection) return;

    // Remove from source
    const [movedLecture] = sourceSection.lectures.splice(sourceLectureIndex, 1);

    // Calculate adjusted target index
    let adjustedIndex = targetLectureIndex;
    if (sourceSection.id === targetSection.id && sourceLectureIndex < targetLectureIndex) {
      adjustedIndex--;
    }

    // Insert at target
    targetSection.lectures.splice(adjustedIndex, 0, movedLecture);

    // Update order in both sections
    sourceSection.lectures.forEach((l, i) => l.order = i);
    targetSection.lectures.forEach((l, i) => l.order = i);

    onReorder(newSections);
    setDraggedItem(null);
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'text-green-400';
      case 'intermediate': return 'text-blue-400';
      case 'advanced': return 'text-yellow-400';
      case 'very_advanced': return 'text-orange-400';
      case 'expert': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="space-y-2">
      {outline.sections.map((section, sectionIndex) => (
        <div
          key={section.id}
          className={`border rounded-lg overflow-hidden transition-colors ${
            draggedItem?.type === 'section' && draggedItem.id === section.id
              ? 'border-purple-500 bg-purple-500/10'
              : 'border-gray-700 bg-gray-800/50'
          }`}
          draggable={!readonly}
          onDragStart={() => handleDragStart('section', section.id)}
          onDragOver={handleDragOver}
          onDrop={() => handleDropOnSection(section.id, sectionIndex)}
        >
          {/* Section header */}
          <button
            type="button"
            onClick={() => toggleSection(section.id)}
            className="w-full flex items-center gap-3 p-3 hover:bg-gray-700/50 transition-colors"
          >
            {!readonly && (
              <GripVertical className="w-4 h-4 text-gray-500 cursor-grab" />
            )}
            <div className="p-1.5 bg-purple-600/20 rounded">
              <FolderOpen className="w-4 h-4 text-purple-400" />
            </div>
            <div className="flex-1 text-left">
              <p className="text-white font-medium">
                Section {sectionIndex + 1}: {section.title}
              </p>
              <p className="text-sm text-gray-500">
                {section.lectures.length} lecture{section.lectures.length !== 1 ? 's' : ''}
              </p>
            </div>
            {expandedSections.has(section.id) ? (
              <ChevronDown className="w-5 h-5 text-gray-400" />
            ) : (
              <ChevronRight className="w-5 h-5 text-gray-400" />
            )}
          </button>

          {/* Lectures */}
          {expandedSections.has(section.id) && (
            <div className="border-t border-gray-700">
              {section.lectures.map((lecture, lectureIndex) => (
                <div
                  key={lecture.id}
                  className={`flex items-center gap-3 px-3 py-2 pl-8 border-b border-gray-700/50 last:border-b-0 transition-colors ${
                    draggedItem?.type === 'lecture' && draggedItem.id === lecture.id
                      ? 'bg-purple-500/10'
                      : 'hover:bg-gray-700/30'
                  }`}
                  draggable={!readonly}
                  onDragStart={() => handleDragStart('lecture', lecture.id)}
                  onDragOver={handleDragOver}
                  onDrop={() => handleDropOnLecture(section.id, lectureIndex)}
                >
                  {!readonly && (
                    <GripVertical className="w-3 h-3 text-gray-600 cursor-grab" />
                  )}
                  <FileVideo className="w-4 h-4 text-gray-500" />
                  <div className="flex-1 min-w-0">
                    <p className="text-gray-300 text-sm truncate">
                      {sectionIndex + 1}.{lectureIndex + 1} {lecture.title}
                    </p>
                  </div>
                  <span className={`text-xs ${getDifficultyColor(lecture.difficulty)}`}>
                    {lecture.difficulty}
                  </span>
                  <div className="flex items-center gap-1 text-xs text-gray-500">
                    <Clock className="w-3 h-3" />
                    {Math.round(lecture.durationSeconds / 60)}m
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}

      {/* Summary */}
      <div className="flex justify-between items-center pt-4 text-sm text-gray-500">
        <span>
          {outline.sections.length} sections, {outline.sections.reduce((acc, s) => acc + s.lectures.length, 0)} lectures
        </span>
        <span>
          ~{outline.totalDurationMinutes} minutes total
        </span>
      </div>
    </div>
  );
}
