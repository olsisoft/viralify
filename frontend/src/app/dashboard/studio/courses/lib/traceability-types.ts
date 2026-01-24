/**
 * Traceability Types
 *
 * Types for source traceability and citation system.
 * Allows users to see exactly which sources were used for each slide/concept.
 */

// Pedagogical role of a source
export type PedagogicalRole =
  | 'theory'      // Definitions, concepts, explanations
  | 'example'     // Practical examples, demos, tutorials
  | 'reference'   // Official documentation, specifications
  | 'opinion'     // Personal notes, perspectives
  | 'data'        // Statistics, studies, research
  | 'context'     // Background information, history, prerequisites
  | 'auto';       // AI determines the role

// Citation style for voiceover
export type CitationStyle = 'natural' | 'academic' | 'minimal' | 'none';

// Source citation configuration
export interface SourceCitationConfig {
  enable_vocal_citations: boolean;
  citation_style: CitationStyle;
  show_traceability_panel: boolean;
  include_page_numbers: boolean;
  include_timestamps: boolean;
  include_quote_excerpts: boolean;
}

// Reference to a source document chunk
export interface ContentReference {
  source_id: string;
  source_name: string;
  source_type: string;
  pedagogical_role: PedagogicalRole;
  chunk_id?: string;
  page_number?: number;
  timestamp?: string;
  quote_excerpt?: string;
  similarity_score: number;
  matched_concepts: string[];
}

// Traceability for a single slide
export interface SlideTraceability {
  slide_index: number;
  slide_type: string;
  slide_title?: string;
  content_references: ContentReference[];
  voiceover_references: ContentReference[];
  source_coverage: number; // 0-1
  primary_source_id?: string;
  generated_content_preview?: string;
}

// Traceability for a lecture
export interface LectureTraceability {
  lecture_id: string;
  lecture_title: string;
  slides: SlideTraceability[];
  sources_used: string[];
  primary_sources: string[];
  overall_source_coverage: number; // 0-1
  key_concepts: string[];
  concept_source_map: Record<string, string[]>;
}

// Complete course traceability
export interface CourseTraceability {
  course_id: string;
  course_title: string;
  citation_config: SourceCitationConfig;
  lectures: LectureTraceability[];
  all_sources_used: string[];
  source_usage_stats: Record<string, SourceUsageStats>;
  overall_source_coverage: number; // 0-1
  total_references: number;
}

// Usage statistics for a source
export interface SourceUsageStats {
  source_id: string;
  total_references: number;
  slides_referenced: number;
  lectures_referenced: number;
  average_similarity: number;
  top_concepts: string[];
}

// Source summary in traceability response
export interface SourceSummary {
  id: string;
  name: string;
  type: string;
  pedagogical_role: PedagogicalRole;
  usage_stats?: SourceUsageStats;
}

// API response for traceability
export interface TraceabilityResponse {
  course_id: string;
  course_title: string;
  traceability: CourseTraceability;
  sources_summary: SourceSummary[];
}

// Knowledge graph concept
export interface Concept {
  id: string;
  name: string;
  canonical_name: string;
  aliases: string[];
  complexity_level: number;
  frequency: number;
  definitions: ConceptDefinition[];
  consolidated_definition?: string;
  prerequisites: string[];
  related_concepts: string[];
}

// Definition from a specific source
export interface ConceptDefinition {
  source_id: string;
  source_name: string;
  source_type: string;
  pedagogical_role: PedagogicalRole;
  definition_text: string;
  context: string;
  confidence: number;
}

// Knowledge graph response
export interface KnowledgeGraphResponse {
  course_id: string;
  total_concepts: number;
  total_cross_references: number;
  sources_analyzed: number;
  concepts: ConceptSummary[];
  cross_references: CrossReferenceSummary[];
}

export interface ConceptSummary {
  id: string;
  name: string;
  complexity: number;
  sources_count: number;
  has_consolidated_definition: boolean;
  prerequisites_count: number;
}

export interface CrossReferenceSummary {
  concept: string;
  sources_count: number;
  agreement_score: number;
}

// Cross-reference analysis
export interface CrossReferenceReport {
  course_topic: string;
  sources_analyzed: number;
  total_concepts_covered: number;
  concepts_with_multiple_sources: number;
  average_coverage: number;
  source_summaries: SourceContributionSummary[];
  topic_cross_refs: TopicCrossRefSummary[];
}

export interface SourceContributionSummary {
  source_id: string;
  source_name: string;
  pedagogical_role: string;
  key_insights_count: number;
  unique_content_count: number;
}

export interface TopicCrossRefSummary {
  topic: string;
  sources_count: number;
  coverage_score: number;
  has_agreement: boolean;
  has_conflicts: boolean;
  missing_aspects: string[];
}

// Helper functions
export function getPedagogicalRoleIcon(role: PedagogicalRole): string {
  const icons: Record<PedagogicalRole, string> = {
    theory: '📚',
    example: '💡',
    reference: '📖',
    opinion: '💭',
    data: '📊',
    context: '🔍',
    auto: '🤖',
  };
  return icons[role] || '📄';
}

export function getPedagogicalRoleLabel(role: PedagogicalRole): string {
  const labels: Record<PedagogicalRole, string> = {
    theory: 'Théorie',
    example: 'Exemple',
    reference: 'Référence',
    opinion: 'Opinion',
    data: 'Données',
    context: 'Contexte',
    auto: 'Auto',
  };
  return labels[role] || role;
}

export function getCoverageColor(coverage: number): string {
  if (coverage >= 0.8) return 'text-green-400';
  if (coverage >= 0.5) return 'text-yellow-400';
  return 'text-red-400';
}

export function getCoverageBgColor(coverage: number): string {
  if (coverage >= 0.8) return 'bg-green-500/20';
  if (coverage >= 0.5) return 'bg-yellow-500/20';
  return 'bg-red-500/20';
}

export function formatCoverage(coverage: number): string {
  return `${Math.round(coverage * 100)}%`;
}
