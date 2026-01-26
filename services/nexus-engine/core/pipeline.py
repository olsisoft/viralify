"""
NEXUS Main Pipeline
Neural Execution & Understanding Synthesis

Pipeline principal qui orchestre:
1. Décomposition cognitive
2. Orchestration multi-agents
3. Synthèse temporelle adaptative
4. Validation et packaging
"""

from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import logging
import time

from models.data_models import (
    NexusRequest, NexusResponse, ArchitectureDNA, CognitiveBlueprint,
    CodeSegment, ExecutionResult, TargetAudience, CodeVerbosity, ExecutionMode
)
from engines.cognitive_decomposition import CognitiveDecompositionAlgorithm
from engines.multi_agent_orchestrator import MultiAgentOrchestrator, AgentContext
from engines.temporal_synthesizer import TemporalCodeSynthesizer


logger = logging.getLogger(__name__)


@dataclass
class NexusConfig:
    """Configuration du pipeline NEXUS"""
    # Agents
    enable_reviewer: bool = True
    enable_executor: bool = True
    enable_narrator: bool = True
    max_feedback_iterations: int = 3
    
    # Sandbox
    sandbox_enabled: bool = True
    sandbox_timeout_seconds: int = 10
    
    # Optimisation
    cache_decomposition: bool = True
    parallel_generation: bool = False
    
    # Debug
    verbose: bool = True
    save_intermediate: bool = False
    output_dir: str = "./nexus_output"


@dataclass
class PipelineProgress:
    """Suivi de progression du pipeline"""
    stage: str
    current: int
    total: int
    message: str
    elapsed_seconds: float = 0.0
    
    @property
    def percent(self) -> float:
        return (self.current / self.total * 100) if self.total > 0 else 0


class NEXUSPipeline:
    """
    Pipeline principal NEXUS.
    
    Transforme une description de projet en code pédagogique
    synchronisé pour génération vidéo.
    
    Innovation clé: Combine décomposition cognitive unique,
    orchestration multi-agents, et adaptation temporelle.
    """
    
    def __init__(self, llm_provider, config: Optional[NexusConfig] = None):
        """
        Args:
            llm_provider: Provider LLM (interface agnostique)
            config: Configuration du pipeline
        """
        self.llm = llm_provider
        self.config = config or NexusConfig()
        
        # Initialiser les engines
        self.decomposer = CognitiveDecompositionAlgorithm(llm_provider)
        self.orchestrator = MultiAgentOrchestrator(
            llm_provider, 
            sandbox_enabled=self.config.sandbox_enabled
        )
        self.time_synthesizer = TemporalCodeSynthesizer()
        
        # Callbacks
        self._progress_callback: Optional[Callable[[PipelineProgress], None]] = None
        self._start_time: Optional[float] = None
    
    def set_progress_callback(self, callback: Callable[[PipelineProgress], None]):
        """Définit un callback pour le suivi de progression"""
        self._progress_callback = callback
    
    def _report_progress(self, stage: str, current: int, total: int, message: str):
        """Rapporte la progression"""
        elapsed = time.time() - self._start_time if self._start_time else 0
        
        progress = PipelineProgress(
            stage=stage,
            current=current,
            total=total,
            message=message,
            elapsed_seconds=elapsed,
        )
        
        if self.config.verbose:
            logger.info(f"[{stage}] {current}/{total} - {message}")
        
        if self._progress_callback:
            self._progress_callback(progress)
    
    def generate(self, request: NexusRequest) -> NexusResponse:
        """
        Point d'entrée principal: Génère le code pour une requête.
        
        Args:
            request: Requête de génération
            
        Returns:
            NexusResponse avec tout le code généré et synchronisé
        """
        self._start_time = time.time()
        logger.info(f"NEXUS Pipeline starting for: {request.project_description[:50]}...")
        
        # =========================================
        # STAGE 1: Cognitive Decomposition
        # =========================================
        self._report_progress("decomposition", 1, 5, "Analyzing project domain...")
        
        dna, blueprint = self.decomposer.decompose(request)
        
        logger.info(f"Decomposition complete: {len(dna.entities)} entities, "
                   f"{len(blueprint.all_steps)} cognitive steps")
        
        # =========================================
        # STAGE 2: Time Budget Planning
        # =========================================
        self._report_progress("time_planning", 2, 5, "Planning time allocation...")
        
        time_budget = self.time_synthesizer.create_time_budget(request, blueprint)
        
        logger.info(f"Time budget: {time_budget.total_seconds}s total, "
                   f"{time_budget.allocated_seconds}s allocated")
        
        # =========================================
        # STAGE 3: Multi-Agent Code Generation
        # =========================================
        self._report_progress("generation", 3, 5, "Generating code with multi-agent system...")
        
        # Créer le contexte pour les agents
        agent_context = AgentContext(
            request=request,
            dna=dna,
            blueprint=blueprint,
        )
        
        # Orchestrer la génération
        segments = self.orchestrator.orchestrate(agent_context)
        
        logger.info(f"Generated {len(segments)} code segments")
        
        # =========================================
        # STAGE 4: Temporal Adaptation
        # =========================================
        self._report_progress("adaptation", 4, 5, "Adapting code to time constraints...")
        
        # Adapter chaque segment au temps alloué
        step_id_to_slot = {slot.step_id: slot for slot in time_budget.slots}
        
        adapted_segments = []
        for segment in segments:
            slot = step_id_to_slot.get(segment.cognitive_step_id)
            if slot:
                segment = self.time_synthesizer.adapt_segment_to_time(
                    segment, slot, request
                )
            adapted_segments.append(segment)
        
        # Générer les versions progressives si demandé
        if request.show_evolution:
            evolved_segments = []
            for segment in adapted_segments:
                versions = self.time_synthesizer.generate_progressive_versions(segment, dna)
                evolved_segments.extend(versions)
            adapted_segments = evolved_segments
        
        # =========================================
        # STAGE 5: Packaging
        # =========================================
        self._report_progress("packaging", 5, 5, "Creating final package...")
        
        # Créer les métadonnées de synchronisation
        sync_metadata = self._create_sync_metadata(adapted_segments, dna, blueprint)
        
        # Construire la réponse
        response = NexusResponse(
            request_id=request.request_id,
            architecture_dna=dna,
            cognitive_blueprint=blueprint,
            code_segments=adapted_segments,
            execution_results=agent_context.execution_results,
            generation_time_ms=int((time.time() - self._start_time) * 1000),
            sync_metadata=sync_metadata,
        )
        
        elapsed = time.time() - self._start_time
        logger.info(f"NEXUS Pipeline complete in {elapsed:.1f}s: "
                   f"{len(adapted_segments)} segments, "
                   f"{response.total_lines_of_code} LOC")
        
        return response
    
    def _create_sync_metadata(self, segments: List[CodeSegment],
                               dna: ArchitectureDNA,
                               blueprint: CognitiveBlueprint) -> Dict[str, Any]:
        """
        Crée les métadonnées de synchronisation pour l'assembleur vidéo.
        """
        timeline = []
        current_time = 0
        
        for segment in sorted(segments, key=lambda s: s.display_order):
            entry = {
                "segment_id": segment.id,
                "start_time_seconds": current_time,
                "duration_seconds": segment.duration_seconds,
                "end_time_seconds": current_time + segment.duration_seconds,
                "filename": segment.filename,
                "component_type": segment.component_type.value,
                "narration_script": segment.narration_script,
                "key_concepts": segment.key_concepts,
                "display_mode": "code_editor",  # Pour l'assembleur
            }
            timeline.append(entry)
            current_time += segment.duration_seconds
        
        return {
            "project_name": dna.project_name,
            "language": dna.language,
            "framework": dna.framework,
            "total_duration_seconds": current_time,
            "segment_count": len(segments),
            "timeline": timeline,
            "cognitive_phases": {
                "analysis": len(blueprint.analysis_phase),
                "design": len(blueprint.design_phase),
                "implementation": len(blueprint.implementation_phase),
                "validation": len(blueprint.validation_phase),
            },
        }


# =============================================================================
# FACTORY & SHORTCUTS
# =============================================================================

def create_nexus_pipeline(
    provider: str = "groq",
    api_key: str = "",
    model: str = "",
    **kwargs
) -> NEXUSPipeline:
    """
    Factory pour créer un pipeline NEXUS.
    
    Args:
        provider: "groq", "openai", "anthropic", "ollama"
        api_key: Clé API
        model: Modèle (optionnel)
        
    Returns:
        NEXUSPipeline configuré
    """
    # Import du provider depuis MAESTRO ou local
    try:
        from providers.llm_provider import (
            LLMConfig, create_llm_provider, LLMProvider
        )
    except ImportError:
        # Fallback: créer un provider minimal
        raise ImportError("LLM provider not found. Ensure providers/llm_provider.py exists.")
    
    # Créer la config
    if provider == "groq":
        config = LLMConfig.for_groq(api_key, model or "llama-3.3-70b-versatile")
    elif provider == "openai":
        config = LLMConfig.for_openai(api_key, model or "gpt-4o")
    elif provider == "anthropic":
        config = LLMConfig.for_anthropic(api_key, model or "claude-sonnet-4-20250514")
    elif provider == "ollama":
        config = LLMConfig.for_ollama(model or "llama3.1")
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    llm = create_llm_provider(config)
    
    return NEXUSPipeline(llm, **kwargs)


def generate_code(
    project_description: str,
    lesson_context: str = "",
    skill_level: str = "intermediate",
    language: str = "python",
    audience: str = "developer",
    allocated_time: int = 300,
    provider: str = "groq",
    api_key: str = "",
    **kwargs
) -> NexusResponse:
    """
    Interface simplifiée pour générer du code.
    
    Args:
        project_description: Description du projet (ex: "une plateforme e-commerce")
        lesson_context: Contexte de la leçon
        skill_level: "beginner", "intermediate", "advanced", "expert"
        language: Langage de programmation
        audience: "developer", "architect", "student"
        allocated_time: Temps alloué en secondes
        provider: Provider LLM
        api_key: Clé API
        
    Returns:
        NexusResponse avec le code généré
    """
    # Mapper l'audience
    audience_map = {
        "developer": TargetAudience.DEVELOPER,
        "architect": TargetAudience.ARCHITECT,
        "student": TargetAudience.STUDENT,
        "lead": TargetAudience.TECHNICAL_LEAD,
    }
    
    request = NexusRequest(
        project_description=project_description,
        lesson_context=lesson_context,
        skill_level=skill_level,
        language=language,
        target_audience=audience_map.get(audience, TargetAudience.DEVELOPER),
        allocated_time_seconds=allocated_time,
    )
    
    pipeline = create_nexus_pipeline(provider=provider, api_key=api_key)
    
    return pipeline.generate(request)
