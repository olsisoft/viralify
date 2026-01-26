"""
MAESTRO Engine Service

Multi-level Adaptive Educational Structuring & Teaching Resource Orchestrator

5-Layer Pipeline:
1. Domain Discovery - Analyze domain structure and extract themes
2. Knowledge Graph - Build prerequisite graph between concepts
3. Difficulty Calibration - 4D difficulty vectors for each concept
4. Curriculum Sequencing - Optimal learning order with smooth progression
5. Content Generation - Generate lessons, quizzes, exercises
"""

import os
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid

from models.data_models import (
    CourseRequest,
    CoursePackage,
    ProgressionPath,
    Module,
)
from engines.domain_discovery import DomainDiscoveryEngine
from engines.knowledge_graph import KnowledgeGraphEngine
from engines.difficulty_calibrator import DifficultyCalibratorEngine
from engines.curriculum_sequencer import CurriculumSequencerEngine
from generators.content_generator import ContentGenerator


# ============================================================================
# Configuration
# ============================================================================

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ============================================================================
# Request/Response Models
# ============================================================================

class GenerateCourseRequest(BaseModel):
    """Request to generate a course using MAESTRO pipeline"""
    subject: str = Field(..., description="Course subject", min_length=3)
    progression_path: str = Field(
        default="beginner_to_intermediate",
        description="Progression path: beginner_to_intermediate, intermediate_to_advanced, advanced_to_expert, full_range"
    )
    total_duration_hours: float = Field(default=5.0, description="Target total duration in hours", ge=1.0, le=20.0)
    num_modules: int = Field(default=5, description="Target number of modules", ge=2, le=15)
    language: str = Field(default="en", description="Content language (en, fr, es, de, etc.)")
    include_quizzes: bool = Field(default=True, description="Include quiz questions")
    include_exercises: bool = Field(default=True, description="Include practical exercises")
    questions_per_lesson: int = Field(default=3, description="Quiz questions per lesson", ge=1, le=10)
    exercises_per_lesson: int = Field(default=1, description="Exercises per lesson", ge=0, le=3)

    class Config:
        json_schema_extra = {
            "example": {
                "subject": "Python Programming Fundamentals",
                "progression_path": "beginner_to_intermediate",
                "total_duration_hours": 5.0,
                "num_modules": 5,
                "language": "en",
                "include_quizzes": True,
                "include_exercises": True,
            }
        }


class CourseJobStatus(BaseModel):
    """Status of a course generation job"""
    job_id: str
    status: str  # queued, processing, completed, failed
    progress: float  # 0-100
    stage: str  # domain_discovery, knowledge_graph, difficulty_calibration, sequencing, content_generation
    message: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DomainAnalysisResponse(BaseModel):
    """Response from domain analysis"""
    overview: str
    core_themes: List[Dict[str, Any]]
    learning_objectives: List[str]
    prerequisite_knowledge: List[str]


class ConceptsResponse(BaseModel):
    """Response containing extracted concepts"""
    concepts: List[Dict[str, Any]]
    total_count: int
    skill_level_distribution: Dict[str, int]


# ============================================================================
# In-Memory Job Storage
# ============================================================================

_jobs: Dict[str, CourseJobStatus] = {}
_courses: Dict[str, CoursePackage] = {}


# ============================================================================
# MAESTRO Pipeline Orchestrator
# ============================================================================

class MaestroPipeline:
    """
    Orchestrates the 5-layer MAESTRO pipeline.
    """

    def __init__(self, model: str = OPENAI_MODEL):
        self.model = model
        self.domain_engine = DomainDiscoveryEngine(model=model)
        self.graph_engine = KnowledgeGraphEngine()
        self.calibrator = DifficultyCalibratorEngine(model=model)
        self.sequencer = CurriculumSequencerEngine()
        self.content_generator = ContentGenerator(model=model)

    async def generate_course(
        self,
        request: GenerateCourseRequest,
        job_id: str,
        update_progress: callable,
    ) -> CoursePackage:
        """
        Execute the full MAESTRO pipeline.

        Args:
            request: Course generation request
            job_id: Job ID for progress tracking
            update_progress: Callback to update job progress

        Returns:
            Complete CoursePackage
        """
        start_time = time.time()
        progression_path = ProgressionPath(request.progression_path)

        # Layer 1: Domain Discovery
        update_progress(job_id, 5, "domain_discovery", "Analyzing domain structure...")
        domain = await self.domain_engine.analyze_domain(
            subject=request.subject,
            progression_path=progression_path,
            language=request.language,
        )

        # Layer 1b: Extract concepts
        update_progress(job_id, 15, "domain_discovery", "Extracting concepts...")
        concepts = await self.domain_engine.extract_concepts(
            subject=request.subject,
            themes=domain.get("core_themes", []),
            progression_path=progression_path,
            language=request.language,
        )

        # Layer 2: Knowledge Graph
        update_progress(job_id, 25, "knowledge_graph", "Building knowledge graph...")
        self.graph_engine.build_graph(concepts)
        validation_issues = self.graph_engine.validate_prerequisites()
        if validation_issues:
            print(f"[MAESTRO] Knowledge graph issues: {validation_issues}", flush=True)

        # Layer 3: Difficulty Calibration
        update_progress(job_id, 35, "difficulty_calibration", "Calibrating difficulty...")
        calibrated_concepts = await self.calibrator.calibrate_concepts(concepts)

        # Layer 4: Curriculum Sequencing
        update_progress(job_id, 45, "sequencing", "Sequencing curriculum...")
        learning_path = self.sequencer.sequence_curriculum(
            concepts=calibrated_concepts,
            knowledge_graph=self.graph_engine,
            progression_path=progression_path,
            target_modules=request.num_modules,
        )

        # Layer 5: Content Generation
        update_progress(job_id, 55, "content_generation", "Generating lesson content...")
        total_concepts = len(calibrated_concepts)

        for i, module in enumerate(learning_path.modules):
            module_concepts = [c for c in calibrated_concepts if c.id in module.concept_ids]

            for j, concept in enumerate(module_concepts):
                progress = 55 + (40 * (i * len(module_concepts) + j) / total_concepts)
                update_progress(
                    job_id,
                    progress,
                    "content_generation",
                    f"Generating: {concept.name}..."
                )

                lesson = await self.content_generator.generate_lesson(
                    concept=concept,
                    language=request.language,
                )
                module.lessons.append(lesson)

        # Build final package
        update_progress(job_id, 95, "finalizing", "Building course package...")

        total_duration = sum(
            sum(l.estimated_duration_minutes for l in m.lessons)
            for m in learning_path.modules
        )

        course_package = CoursePackage(
            title=f"Course: {request.subject}",
            description=domain.get("overview", f"Comprehensive course on {request.subject}"),
            subject=request.subject,
            language=request.language,
            modules=learning_path.modules,
            concepts=calibrated_concepts,
            progression_path=progression_path,
            total_duration_minutes=total_duration,
            generation_time_seconds=time.time() - start_time,
        )

        update_progress(job_id, 100, "completed", "Course generation complete!")
        return course_package


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    print("[MAESTRO] Starting MAESTRO Engine service...", flush=True)
    yield
    print("[MAESTRO] Shutting down MAESTRO Engine service...", flush=True)


app = FastAPI(
    title="MAESTRO Engine",
    description="Multi-level Adaptive Educational Structuring & Teaching Resource Orchestrator",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = MaestroPipeline()


def update_job_progress(job_id: str, progress: float, stage: str, message: str):
    """Update job progress in storage"""
    if job_id in _jobs:
        _jobs[job_id].progress = progress
        _jobs[job_id].stage = stage
        _jobs[job_id].message = message
        if progress >= 100:
            _jobs[job_id].status = "completed"
            _jobs[job_id].completed_at = datetime.utcnow()


async def run_generation(job_id: str, request: GenerateCourseRequest):
    """Background task to run course generation"""
    try:
        _jobs[job_id].status = "processing"
        course = await pipeline.generate_course(
            request=request,
            job_id=job_id,
            update_progress=update_job_progress,
        )
        _courses[job_id] = course
        _jobs[job_id].result = {"course_id": course.id, "title": course.title}
    except Exception as e:
        print(f"[MAESTRO] Generation failed: {e}", flush=True)
        _jobs[job_id].status = "failed"
        _jobs[job_id].error = str(e)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "maestro-engine"}


@app.post("/api/v1/courses/generate", response_model=CourseJobStatus)
async def generate_course(
    request: GenerateCourseRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start course generation using MAESTRO pipeline.

    Returns a job ID to track progress.
    """
    job_id = str(uuid.uuid4())

    job = CourseJobStatus(
        job_id=job_id,
        status="queued",
        progress=0,
        stage="queued",
        message="Course generation queued",
        created_at=datetime.utcnow(),
    )
    _jobs[job_id] = job

    background_tasks.add_task(run_generation, job_id, request)

    return job


@app.get("/api/v1/courses/jobs/{job_id}", response_model=CourseJobStatus)
async def get_job_status(job_id: str):
    """Get the status of a course generation job"""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@app.get("/api/v1/courses/{course_id}")
async def get_course(course_id: str):
    """Get a generated course by ID"""
    # Search by job_id (course stored under job_id key)
    if course_id in _courses:
        return _courses[course_id].to_dict()

    # Search by course.id
    for course in _courses.values():
        if course.id == course_id:
            return course.to_dict()

    raise HTTPException(status_code=404, detail="Course not found")


@app.post("/api/v1/domain/analyze", response_model=DomainAnalysisResponse)
async def analyze_domain(
    subject: str,
    progression_path: str = "beginner_to_intermediate",
    language: str = "en",
):
    """
    Analyze a domain without generating a full course.

    Useful for previewing the domain structure.
    """
    path = ProgressionPath(progression_path)
    result = await pipeline.domain_engine.analyze_domain(
        subject=subject,
        progression_path=path,
        language=language,
    )
    return DomainAnalysisResponse(**result)


@app.post("/api/v1/concepts/extract", response_model=ConceptsResponse)
async def extract_concepts(
    subject: str,
    themes: List[Dict[str, Any]],
    progression_path: str = "beginner_to_intermediate",
    language: str = "en",
):
    """
    Extract concepts from themes.
    """
    path = ProgressionPath(progression_path)
    concepts = await pipeline.domain_engine.extract_concepts(
        subject=subject,
        themes=themes,
        progression_path=path,
        language=language,
    )

    # Calculate skill level distribution
    distribution = {}
    for concept in concepts:
        level = concept.skill_level.value
        distribution[level] = distribution.get(level, 0) + 1

    return ConceptsResponse(
        concepts=[c.to_dict() for c in concepts],
        total_count=len(concepts),
        skill_level_distribution=distribution,
    )


@app.get("/api/v1/config/progression-paths")
async def get_progression_paths():
    """Get available progression paths"""
    return {
        "paths": [
            {
                "id": p.value,
                "name": p.value.replace("_", " ").title(),
                "description": f"From {p.value.split('_to_')[0]} to {p.value.split('_to_')[-1] if '_to_' in p.value else 'expert'}"
            }
            for p in ProgressionPath
        ]
    }


@app.get("/api/v1/config/skill-levels")
async def get_skill_levels():
    """Get available skill levels"""
    from models.data_models import SkillLevel, SKILL_LEVEL_RANGES
    return {
        "levels": [
            {
                "id": level.value,
                "name": level.value.replace("_", " ").title(),
                "difficulty_range": SKILL_LEVEL_RANGES[level],
            }
            for level in SkillLevel
        ]
    }


@app.get("/api/v1/config/bloom-levels")
async def get_bloom_levels():
    """Get Bloom's taxonomy levels"""
    from models.data_models import BloomLevel, BLOOM_TO_COGNITIVE_LOAD
    return {
        "levels": [
            {
                "id": level.value,
                "name": level.value.title(),
                "cognitive_load": BLOOM_TO_COGNITIVE_LOAD[level],
            }
            for level in BloomLevel
        ]
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8008"))
    uvicorn.run(app, host="0.0.0.0", port=port)
