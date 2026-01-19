# Curriculum Enforcer Module

Enforces pedagogical structure on lesson content for consistent, effective learning experiences.

## Features

- **Templates**: Pre-built templates for Education, Enterprise, Bootcamp, Tutorial, Workshop, Certification
- **Validation**: Detects missing phases, wrong order, duration issues
- **Auto-fix**: Automatically restructures content to match template
- **Custom Templates**: Create your own curriculum templates

## Quick Start

```python
from curriculum_enforcer import (
    CurriculumEnforcerService,
    ContextType,
    LessonContent,
    EnforcementRequest
)

enforcer = CurriculumEnforcerService()

# Validate a lesson
validation = await enforcer.validate_only(
    content=lesson_content,
    context_type=ContextType.EDUCATION
)

print(f"Valid: {validation.is_valid}")
print(f"Score: {validation.score:.0%}")
print(f"Missing: {validation.missing_required_phases}")

# Enforce structure with auto-fix
result = await enforcer.enforce(
    EnforcementRequest(
        content=lesson_content,
        context_type=ContextType.ENTERPRISE,
        auto_fix=True
    )
)

if result.success:
    print(f"Changes: {result.changes_made}")
```

## Available Templates

### Education (Default)
**Flow**: Hook → Concept → Theory → Visualization → Code Demo → Recap

Best for: Online courses, tutorials, academic content

### Enterprise
**Flow**: Context → Concept → Use Case → ROI → Action Items

Best for: Corporate training, professional development

### Bootcamp
**Flow**: Concept → Code Demo → Exercise → Exercise → Challenge → Review

Best for: Intensive coding bootcamps, workshops

### Tutorial
**Flow**: Objectives → Prerequisites → Steps → Recap → Next Steps

Best for: Quick how-to guides, documentation

### Workshop
**Flow**: Context → Example → Exercise → Review → Exercise → Recap

Best for: Hands-on collaborative sessions

### Certification
**Flow**: Objectives → Theory → Examples → Practice → Quiz → Recap

Best for: Exam preparation, certification courses

## Lesson Phases

| Phase | Description | Common In |
|-------|-------------|-----------|
| `hook` | Engaging opener | Education |
| `concept` | Simple explanation | All |
| `theory` | Technical details | Education, Certification |
| `code_demo` | Live coding | Education, Bootcamp |
| `exercise` | Hands-on practice | Bootcamp, Workshop |
| `quiz` | Knowledge check | Certification |
| `recap` | Key takeaways | All |
| `use_case` | Real-world example | Enterprise |
| `roi` | Business value | Enterprise |
| `action_items` | Next steps | Enterprise |

## Integration with Course Generator

```python
# In course-generator/services/course_planner.py

from curriculum_enforcer import (
    CurriculumEnforcerService,
    ContextType,
    EnforcementRequest,
    LessonContent
)

class CoursePlanner:
    def __init__(self):
        self.curriculum_enforcer = CurriculumEnforcerService()

    async def plan_lesson(self, lesson_spec, context_type=ContextType.EDUCATION):
        # Generate initial lesson structure
        slides = await self._generate_slides(lesson_spec)

        # Enforce curriculum structure
        content = LessonContent(
            lesson_id=lesson_spec["id"],
            title=lesson_spec["title"],
            slides=slides
        )

        result = await self.curriculum_enforcer.enforce(
            EnforcementRequest(
                content=content,
                context_type=context_type,
                auto_fix=True
            )
        )

        if result.restructured_content:
            return result.restructured_content.slides
        return slides
```

## Custom Templates

```python
from curriculum_enforcer import (
    CurriculumTemplate,
    LessonTemplate,
    PhaseConfig,
    LessonPhase,
    ContextType,
    TemplateRegistry
)

# Create custom template
my_template = CurriculumTemplate(
    id="my_custom_template",
    name="My Custom Flow",
    context_type=ContextType.CUSTOM,
    description="My company's learning flow",
    default_lesson_template=LessonTemplate(
        name="Custom Lesson",
        description="Our way of teaching",
        total_duration_target_seconds=480,
        phases=[
            PhaseConfig(phase=LessonPhase.HOOK, required=True, order=0),
            PhaseConfig(phase=LessonPhase.USE_CASE, required=True, order=1),
            PhaseConfig(phase=LessonPhase.CODE_DEMO, required=True, order=2),
            PhaseConfig(phase=LessonPhase.ACTION_ITEMS, required=True, order=3),
        ]
    )
)

# Register it
TemplateRegistry.register(my_template)
```

## Configuration

```bash
# Required for AI-powered phase detection
export OPENAI_API_KEY=your_key_here
```

## Dependencies

```bash
pip install -r requirements.txt
```
