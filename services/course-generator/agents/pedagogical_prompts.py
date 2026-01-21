"""
Pedagogical Agent System Prompts

Contains all the prompts used by the pedagogical agent nodes.
"""

CONTEXT_ANALYSIS_PROMPT = """You are an expert pedagogical analyst. Analyze the following course topic and context to determine the best teaching approach.

TOPIC: {topic}
{description_section}
CATEGORY: {category}
TARGET AUDIENCE: {target_audience}
DIFFICULTY RANGE: {difficulty_start} to {difficulty_end}

Analyze and provide:
1. The likely learner persona (developer, architect, manager, student, hobbyist, etc.)
2. Topic complexity level (basic, intermediate, advanced, expert)
3. Whether this topic requires:
   - Code examples (true/false)
   - Diagrams and visual aids (true/false)
   - Hands-on exercises (true/false)
4. Key domain keywords for this topic

Respond in JSON format:
{{
    "detected_persona": "string",
    "topic_complexity": "basic|intermediate|advanced|expert",
    "requires_code": true/false,
    "requires_diagrams": true/false,
    "requires_hands_on": true/false,
    "domain_keywords": ["keyword1", "keyword2", ...],
    "reasoning": "Brief explanation of your analysis"
}}"""


PROFILE_ADAPTATION_PROMPT = """Based on the learner analysis, determine the optimal content mix for this course.

LEARNER PERSONA: {detected_persona}
TOPIC COMPLEXITY: {topic_complexity}
CATEGORY: {category}
REQUIRES CODE: {requires_code}
REQUIRES DIAGRAMS: {requires_diagrams}
REQUIRES HANDS-ON: {requires_hands_on}

Define content weights (0.0 to 1.0) for:
- code_weight: How much programming code to include
- diagram_weight: How many diagrams and visual aids
- demo_weight: Live demonstrations and walkthroughs
- theory_weight: Conceptual and theoretical content
- case_study_weight: Real-world examples and case studies

Also suggest the most relevant lesson elements from this list:
{available_elements}

Respond in JSON:
{{
    "content_preferences": {{
        "code_weight": 0.0-1.0,
        "diagram_weight": 0.0-1.0,
        "demo_weight": 0.0-1.0,
        "theory_weight": 0.0-1.0,
        "case_study_weight": 0.0-1.0
    }},
    "recommended_elements": ["element_id1", "element_id2", ...],
    "adaptation_notes": "Brief notes on content adaptation strategy"
}}"""


ELEMENT_SUGGESTION_PROMPT = """Suggest specific lesson elements for each lecture in this course outline.

COURSE TOPIC: {topic}
CATEGORY: {category}
CONTENT PREFERENCES:
- Code emphasis: {code_weight}
- Diagram emphasis: {diagram_weight}
- Demo emphasis: {demo_weight}

AVAILABLE ELEMENTS:
{available_elements}

COURSE STRUCTURE:
{outline_structure}

For each lecture, suggest 3-5 most relevant elements based on the lecture topic and overall content preferences.

Respond in JSON:
{{
    "element_mapping": {{
        "lecture_id_1": ["element1", "element2", ...],
        "lecture_id_2": ["element1", "element2", ...],
        ...
    }},
    "reasoning": "Brief explanation of element choices"
}}"""


QUIZ_PLANNING_PROMPT = """Plan quiz placement and content for this course.

QUIZ CONFIGURATION:
- Enabled: {quiz_enabled}
- Frequency: {quiz_frequency}
- Questions per quiz: {questions_per_quiz}

COURSE STRUCTURE:
{outline_structure}

LEARNING OBJECTIVES BY SECTION:
{section_objectives}

Plan quizzes that:
1. Reinforce key concepts from each section
2. Progress in difficulty matching the course difficulty curve
3. Cover all major learning objectives
4. Use appropriate question types for the content (multiple choice, true/false, code review, etc.)

Respond in JSON:
{{
    "quiz_placement": [
        {{
            "lecture_id": "string",
            "quiz_type": "section_review|lecture_check|final_assessment",
            "difficulty": "easy|medium|hard",
            "question_count": 5,
            "topics_covered": ["topic1", "topic2"],
            "question_types": ["multiple_choice", "true_false", ...]
        }},
        ...
    ],
    "total_quiz_count": N,
    "coverage_analysis": "Brief analysis of learning objective coverage"
}}"""


LANGUAGE_VALIDATION_PROMPT = """Validate that all content in this course outline is in the target language.

TARGET LANGUAGE: {target_language} ({language_name})

CONTENT TO VALIDATE:
{outline_content}

Check that:
1. All titles are in {language_name}
2. All descriptions are in {language_name}
3. All learning objectives are in {language_name}
4. Code comments (if any) are appropriate for the target audience

Respond in JSON:
{{
    "is_valid": true/false,
    "issues": [
        {{
            "location": "section/lecture identifier",
            "issue": "description of language issue",
            "suggested_fix": "corrected text"
        }},
        ...
    ],
    "overall_language_quality": "excellent|good|needs_improvement|poor"
}}"""


STRUCTURE_VALIDATION_PROMPT = """Validate the pedagogical quality of this course structure.

COURSE OUTLINE:
{outline_structure}

DIFFICULTY PROGRESSION: {difficulty_start} to {difficulty_end}
TOTAL DURATION: {total_duration} minutes
TARGET AUDIENCE: {target_audience}

Evaluate:
1. Logical progression of topics (score 0-100)
2. Appropriate difficulty curve (score 0-100)
3. Balanced content distribution (score 0-100)
4. Clear learning objectives (score 0-100)
5. Practical applicability (score 0-100)

Provide warnings for any issues and suggestions for improvement.

Respond in JSON:
{{
    "is_valid": true/false,
    "pedagogical_score": 0-100 (average of all scores),
    "scores": {{
        "logical_progression": 0-100,
        "difficulty_curve": 0-100,
        "content_balance": 0-100,
        "learning_objectives": 0-100,
        "practical_applicability": 0-100
    }},
    "warnings": ["warning1", "warning2", ...],
    "suggestions": ["suggestion1", "suggestion2", ...],
    "overall_assessment": "Brief assessment of pedagogical quality"
}}"""


FINALIZATION_PROMPT = """Finalize the course plan with all enhancements applied.

Review and confirm:
1. All adaptive elements are assigned to lectures
2. Quiz placement is correct
3. Language compliance is verified
4. Structure is pedagogically sound

Generate the final metadata summary.

Respond in JSON:
{{
    "ready_for_generation": true/false,
    "final_checks_passed": ["check1", "check2", ...],
    "metadata": {{
        "total_lectures": N,
        "total_quizzes": N,
        "estimated_duration_minutes": N,
        "content_mix": {{
            "code_heavy_lectures": N,
            "diagram_heavy_lectures": N,
            "theory_heavy_lectures": N
        }},
        "pedagogical_score": 0-100,
        "generation_timestamp": "ISO timestamp"
    }},
    "recommendations_for_generation": ["recommendation1", ...]
}}"""
