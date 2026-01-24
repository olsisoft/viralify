'use client';

import { useState, useEffect } from 'react';
import {
  FileText,
  BookOpen,
  Lightbulb,
  Network,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  CheckCircle2,
  AlertCircle,
  Loader2,
  X,
} from 'lucide-react';
import { useTraceability } from '../hooks/useTraceability';
import type {
  TraceabilityResponse,
  SourceSummary,
  SlideTraceability,
  LectureTraceability,
  ConceptSummary,
  TopicCrossRefSummary,
  PedagogicalRole,
} from '../lib/traceability-types';
import {
  getPedagogicalRoleIcon,
  getPedagogicalRoleLabel,
  getCoverageColor,
  getCoverageBgColor,
  formatCoverage,
} from '../lib/traceability-types';

type TabType = 'sources' | 'slides' | 'concepts' | 'cross-refs';

interface TraceabilityPanelProps {
  jobId: string;
  onClose: () => void;
}

export function TraceabilityPanel({ jobId, onClose }: TraceabilityPanelProps) {
  const [activeTab, setActiveTab] = useState<TabType>('sources');
  const [expandedLectures, setExpandedLectures] = useState<Set<string>>(new Set());
  const [expandedSlides, setExpandedSlides] = useState<Set<string>>(new Set());

  const {
    traceability,
    knowledgeGraph,
    crossReferences,
    isLoading,
    isLoadingKnowledgeGraph,
    isLoadingCrossReferences,
    error,
    fetchTraceability,
    fetchKnowledgeGraph,
    fetchCrossReferences,
  } = useTraceability();

  // Fetch data on mount
  useEffect(() => {
    fetchTraceability(jobId);
  }, [jobId, fetchTraceability]);

  // Fetch additional data when tab changes
  useEffect(() => {
    if (activeTab === 'concepts' && !knowledgeGraph) {
      fetchKnowledgeGraph(jobId);
    }
    if (activeTab === 'cross-refs' && !crossReferences) {
      fetchCrossReferences(jobId);
    }
  }, [activeTab, jobId, knowledgeGraph, crossReferences, fetchKnowledgeGraph, fetchCrossReferences]);

  const toggleLecture = (lectureId: string) => {
    setExpandedLectures((prev) => {
      const next = new Set(prev);
      if (next.has(lectureId)) {
        next.delete(lectureId);
      } else {
        next.add(lectureId);
      }
      return next;
    });
  };

  const toggleSlide = (slideKey: string) => {
    setExpandedSlides((prev) => {
      const next = new Set(prev);
      if (next.has(slideKey)) {
        next.delete(slideKey);
      } else {
        next.add(slideKey);
      }
      return next;
    });
  };

  const tabs = [
    { id: 'sources' as TabType, label: 'Sources', icon: FileText },
    { id: 'slides' as TabType, label: 'Par Slide', icon: BookOpen },
    { id: 'concepts' as TabType, label: 'Concepts', icon: Lightbulb },
    { id: 'cross-refs' as TabType, label: 'Croisements', icon: Network },
  ];

  return (
    <div className="bg-gray-800/90 backdrop-blur border border-gray-700 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <FileText className="w-5 h-5 text-purple-400" />
          Traçabilité des Sources
        </h3>
        <button
          onClick={onClose}
          className="p-1 hover:bg-gray-700 rounded transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-700">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'text-purple-400 border-b-2 border-purple-400 bg-purple-400/10'
                : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="p-4 max-h-[60vh] overflow-y-auto">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-6 h-6 animate-spin text-purple-400" />
            <span className="ml-2 text-gray-400">Chargement...</span>
          </div>
        ) : error ? (
          <div className="flex items-center gap-2 p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <span className="text-red-400">{error}</span>
          </div>
        ) : (
          <>
            {activeTab === 'sources' && traceability && (
              <SourcesTab traceability={traceability} />
            )}
            {activeTab === 'slides' && traceability && (
              <SlidesTab
                traceability={traceability}
                expandedLectures={expandedLectures}
                expandedSlides={expandedSlides}
                toggleLecture={toggleLecture}
                toggleSlide={toggleSlide}
              />
            )}
            {activeTab === 'concepts' && (
              <ConceptsTab
                knowledgeGraph={knowledgeGraph}
                isLoading={isLoadingKnowledgeGraph}
              />
            )}
            {activeTab === 'cross-refs' && (
              <CrossRefsTab
                crossReferences={crossReferences}
                isLoading={isLoadingCrossReferences}
              />
            )}
          </>
        )}
      </div>

      {/* Footer with overall stats */}
      {traceability && (
        <div className="flex items-center justify-between p-4 border-t border-gray-700 bg-gray-800/50">
          <div className="flex items-center gap-4 text-sm">
            <span className="text-gray-400">
              {traceability.sources_summary.length} sources
            </span>
            <span className="text-gray-400">
              {traceability.traceability.total_references} références
            </span>
          </div>
          <div className={`flex items-center gap-2 ${getCoverageColor(traceability.traceability.overall_source_coverage)}`}>
            <span className="text-sm">Couverture globale:</span>
            <span className="font-semibold">
              {formatCoverage(traceability.traceability.overall_source_coverage)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

// Sources Tab Component
function SourcesTab({ traceability }: { traceability: TraceabilityResponse }) {
  const { sources_summary } = traceability;

  if (!sources_summary.length) {
    return (
      <div className="text-center py-8 text-gray-400">
        Aucune source utilisée pour ce cours
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {sources_summary.map((source) => (
        <SourceCard key={source.id} source={source} />
      ))}
    </div>
  );
}

function SourceCard({ source }: { source: SourceSummary }) {
  const icon = getPedagogicalRoleIcon(source.pedagogical_role);
  const roleLabel = getPedagogicalRoleLabel(source.pedagogical_role);

  return (
    <div className="bg-gray-700/50 border border-gray-600 rounded-lg p-4">
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-3">
          <span className="text-2xl">{icon}</span>
          <div>
            <h4 className="font-medium text-white">{source.name}</h4>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-xs px-2 py-0.5 bg-purple-500/20 text-purple-300 rounded">
                {roleLabel}
              </span>
              <span className="text-xs text-gray-400">{source.type}</span>
            </div>
          </div>
        </div>
        {source.usage_stats && (
          <div className="text-right text-sm">
            <div className="text-gray-400">
              {source.usage_stats.total_references} refs
            </div>
            <div className="text-gray-500 text-xs">
              {source.usage_stats.lectures_referenced} lectures
            </div>
          </div>
        )}
      </div>

      {source.usage_stats?.top_concepts && source.usage_stats.top_concepts.length > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-600">
          <div className="text-xs text-gray-400 mb-2">Concepts clés:</div>
          <div className="flex flex-wrap gap-1">
            {source.usage_stats.top_concepts.slice(0, 5).map((concept, i) => (
              <span
                key={i}
                className="text-xs px-2 py-0.5 bg-gray-600 text-gray-300 rounded"
              >
                {concept}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Slides Tab Component
interface SlidesTabProps {
  traceability: TraceabilityResponse;
  expandedLectures: Set<string>;
  expandedSlides: Set<string>;
  toggleLecture: (id: string) => void;
  toggleSlide: (key: string) => void;
}

function SlidesTab({
  traceability,
  expandedLectures,
  expandedSlides,
  toggleLecture,
  toggleSlide,
}: SlidesTabProps) {
  const { lectures } = traceability.traceability;

  if (!lectures.length) {
    return (
      <div className="text-center py-8 text-gray-400">
        Aucune donnée de traçabilité par slide
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {lectures.map((lecture) => (
        <LectureAccordion
          key={lecture.lecture_id}
          lecture={lecture}
          isExpanded={expandedLectures.has(lecture.lecture_id)}
          onToggle={() => toggleLecture(lecture.lecture_id)}
          expandedSlides={expandedSlides}
          toggleSlide={toggleSlide}
        />
      ))}
    </div>
  );
}

function LectureAccordion({
  lecture,
  isExpanded,
  onToggle,
  expandedSlides,
  toggleSlide,
}: {
  lecture: LectureTraceability;
  isExpanded: boolean;
  onToggle: () => void;
  expandedSlides: Set<string>;
  toggleSlide: (key: string) => void;
}) {
  return (
    <div className="border border-gray-600 rounded-lg overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between p-3 bg-gray-700/50 hover:bg-gray-700 transition-colors"
      >
        <div className="flex items-center gap-2">
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-400" />
          )}
          <span className="font-medium">{lecture.lecture_title}</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-400">
            {lecture.slides.length} slides
          </span>
          <span className={`text-sm ${getCoverageColor(lecture.overall_source_coverage)}`}>
            {formatCoverage(lecture.overall_source_coverage)}
          </span>
        </div>
      </button>

      {isExpanded && (
        <div className="p-3 space-y-2 bg-gray-800/50">
          {lecture.slides.map((slide) => {
            const slideKey = `${lecture.lecture_id}-${slide.slide_index}`;
            return (
              <SlideRow
                key={slideKey}
                slide={slide}
                isExpanded={expandedSlides.has(slideKey)}
                onToggle={() => toggleSlide(slideKey)}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

function SlideRow({
  slide,
  isExpanded,
  onToggle,
}: {
  slide: SlideTraceability;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const hasReferences = slide.content_references.length > 0 || slide.voiceover_references.length > 0;

  return (
    <div className="bg-gray-700/30 rounded border border-gray-600/50">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between p-2 hover:bg-gray-700/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          {hasReferences ? (
            isExpanded ? (
              <ChevronDown className="w-3 h-3 text-gray-400" />
            ) : (
              <ChevronRight className="w-3 h-3 text-gray-400" />
            )
          ) : (
            <span className="w-3 h-3" />
          )}
          <span className="text-sm">
            Slide {slide.slide_index + 1}
            {slide.slide_title && `: ${slide.slide_title}`}
          </span>
          <span className="text-xs px-1.5 py-0.5 bg-gray-600 text-gray-300 rounded">
            {slide.slide_type}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">
            {slide.content_references.length + slide.voiceover_references.length} refs
          </span>
          <span className={`text-xs ${getCoverageColor(slide.source_coverage)}`}>
            {formatCoverage(slide.source_coverage)}
          </span>
        </div>
      </button>

      {isExpanded && hasReferences && (
        <div className="px-3 pb-3 space-y-2">
          {slide.content_references.map((ref, i) => (
            <ReferenceRow key={`content-${i}`} reference={ref} type="Contenu" />
          ))}
          {slide.voiceover_references.map((ref, i) => (
            <ReferenceRow key={`voice-${i}`} reference={ref} type="Voiceover" />
          ))}
        </div>
      )}
    </div>
  );
}

function ReferenceRow({
  reference,
  type,
}: {
  reference: any;
  type: string;
}) {
  const icon = getPedagogicalRoleIcon(reference.pedagogical_role);

  return (
    <div className="flex items-start gap-2 text-xs bg-gray-800/50 rounded p-2">
      <span>{icon}</span>
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <span className="text-gray-300">{reference.source_name}</span>
          <span className="text-gray-500">({type})</span>
        </div>
        {reference.quote_excerpt && (
          <p className="text-gray-400 mt-1 italic">
            "{reference.quote_excerpt}"
          </p>
        )}
        {reference.matched_concepts?.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-1">
            {reference.matched_concepts.map((concept: string, i: number) => (
              <span
                key={i}
                className="px-1.5 py-0.5 bg-purple-500/20 text-purple-300 rounded"
              >
                {concept}
              </span>
            ))}
          </div>
        )}
      </div>
      <span className={getCoverageColor(reference.similarity_score)}>
        {formatCoverage(reference.similarity_score)}
      </span>
    </div>
  );
}

// Concepts Tab Component
function ConceptsTab({
  knowledgeGraph,
  isLoading,
}: {
  knowledgeGraph: any;
  isLoading: boolean;
}) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="w-6 h-6 animate-spin text-purple-400" />
        <span className="ml-2 text-gray-400">Construction du graphe...</span>
      </div>
    );
  }

  if (!knowledgeGraph) {
    return (
      <div className="text-center py-8 text-gray-400">
        Cliquez pour charger le graphe de connaissances
      </div>
    );
  }

  const { concepts, total_concepts, total_cross_references } = knowledgeGraph;

  return (
    <div className="space-y-4">
      {/* Stats */}
      <div className="flex items-center gap-4 p-3 bg-gray-700/30 rounded-lg">
        <div className="text-center">
          <div className="text-2xl font-bold text-purple-400">{total_concepts}</div>
          <div className="text-xs text-gray-400">Concepts</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-400">{total_cross_references}</div>
          <div className="text-xs text-gray-400">Cross-refs</div>
        </div>
      </div>

      {/* Concepts list */}
      <div className="space-y-2">
        {concepts?.slice(0, 20).map((concept: ConceptSummary) => (
          <div
            key={concept.id}
            className="flex items-center justify-between p-3 bg-gray-700/30 rounded-lg border border-gray-600/50"
          >
            <div>
              <h4 className="font-medium text-white">{concept.name}</h4>
              <div className="flex items-center gap-2 mt-1 text-xs">
                <span className="text-gray-400">
                  Complexité: {concept.complexity}/5
                </span>
                <span className="text-gray-400">
                  {concept.sources_count} sources
                </span>
                {concept.has_consolidated_definition && (
                  <CheckCircle2 className="w-3 h-3 text-green-400" />
                )}
              </div>
            </div>
            {concept.prerequisites_count > 0 && (
              <span className="text-xs text-gray-400">
                {concept.prerequisites_count} prérequis
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// Cross References Tab Component
function CrossRefsTab({
  crossReferences,
  isLoading,
}: {
  crossReferences: CrossReferenceReport | null;
  isLoading: boolean;
}) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="w-6 h-6 animate-spin text-purple-400" />
        <span className="ml-2 text-gray-400">Analyse des croisements...</span>
      </div>
    );
  }

  if (!crossReferences) {
    return (
      <div className="text-center py-8 text-gray-400">
        Cliquez pour analyser les références croisées
      </div>
    );
  }

  const { topic_cross_refs, average_coverage, sources_analyzed } = crossReferences;

  return (
    <div className="space-y-4">
      {/* Stats */}
      <div className="flex items-center gap-4 p-3 bg-gray-700/30 rounded-lg">
        <div className="text-center">
          <div className="text-2xl font-bold text-purple-400">{sources_analyzed}</div>
          <div className="text-xs text-gray-400">Sources analysées</div>
        </div>
        <div className="text-center">
          <div className={`text-2xl font-bold ${getCoverageColor(average_coverage)}`}>
            {formatCoverage(average_coverage)}
          </div>
          <div className="text-xs text-gray-400">Couverture moy.</div>
        </div>
      </div>

      {/* Topics */}
      <div className="space-y-2">
        {topic_cross_refs?.map((topic: TopicCrossRefSummary, i: number) => (
          <div
            key={i}
            className="p-3 bg-gray-700/30 rounded-lg border border-gray-600/50"
          >
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-medium text-white">{topic.topic}</h4>
              <span className={getCoverageColor(topic.coverage_score)}>
                {formatCoverage(topic.coverage_score)}
              </span>
            </div>
            <div className="flex items-center gap-3 text-xs">
              <span className="text-gray-400">{topic.sources_count} sources</span>
              {topic.has_agreement && (
                <span className="flex items-center gap-1 text-green-400">
                  <CheckCircle2 className="w-3 h-3" />
                  Accord
                </span>
              )}
              {topic.has_conflicts && (
                <span className="flex items-center gap-1 text-yellow-400">
                  <AlertCircle className="w-3 h-3" />
                  Conflits
                </span>
              )}
            </div>
            {topic.missing_aspects?.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1">
                {topic.missing_aspects.map((aspect, j) => (
                  <span
                    key={j}
                    className="text-xs px-2 py-0.5 bg-red-500/20 text-red-300 rounded"
                  >
                    Manque: {aspect}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
