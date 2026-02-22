"""
Feedback Generator

Generates pedagogical feedback for learner submissions.
"""

from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from models.practice_models import Exercise, ExerciseAttempt
from models.assessment_models import (
    AssessmentResult,
    PedagogicalFeedback,
    FeedbackType,
    UnderstandingLevel,
)

try:
    from shared.llm_provider import get_llm_client, get_model_name

    _USE_SHARED_LLM = True
except ImportError:
    _USE_SHARED_LLM = False


class FeedbackGenerator:
    """
    Generates pedagogical feedback that:
    - Encourages the learner
    - Identifies specific issues without giving solutions
    - Provides actionable guidance
    - Uses Socratic questioning when appropriate
    """

    def __init__(self):
        self.llm = ChatOpenAI(model=get_model_name("quality") if _USE_SHARED_LLM else "gpt-4o", temperature=0.7)

    async def generate_feedback(
        self,
        exercise: Exercise,
        attempt: ExerciseAttempt,
        assessment: AssessmentResult,
        learner_history: Optional[List[ExerciseAttempt]] = None,
    ) -> List[PedagogicalFeedback]:
        """Generate comprehensive pedagogical feedback"""
        feedback_items = []

        # Always start with something positive
        positive_feedback = await self._generate_positive_feedback(exercise, attempt, assessment)
        if positive_feedback:
            feedback_items.append(positive_feedback)

        # Main feedback based on result
        if assessment.passed:
            success_feedback = await self._generate_success_feedback(exercise, attempt, assessment)
            feedback_items.extend(success_feedback)
        else:
            corrective_feedback = await self._generate_corrective_feedback(exercise, attempt, assessment)
            feedback_items.extend(corrective_feedback)

        # Add Socratic question if struggling
        if not assessment.passed and learner_history and len(learner_history) >= 2:
            socratic = await self._generate_socratic_question(exercise, attempt, assessment)
            if socratic:
                feedback_items.append(socratic)

        return feedback_items

    async def _generate_positive_feedback(
        self,
        exercise: Exercise,
        attempt: ExerciseAttempt,
        assessment: AssessmentResult,
    ) -> Optional[PedagogicalFeedback]:
        """Find something positive to highlight"""
        checks_passed = assessment.checks_passed
        code = attempt.submitted_code

        # Identify positives
        positives = []
        if checks_passed:
            positives.append(f"validé {len(checks_passed)} vérification(s)")
        if code and len(code.strip()) > 0:
            positives.append("structure de code présente")
        if "def " in code or "function" in code:
            positives.append("utilisation de fonctions")
        if "#" in code or "//" in code:
            positives.append("présence de commentaires")

        if not positives:
            return PedagogicalFeedback(
                feedback_type=FeedbackType.ENCOURAGEMENT,
                title="Tu es sur la bonne voie",
                message="Tu as fait le premier pas en soumettant du code. Continue !",
                priority=3,
            )

        return PedagogicalFeedback(
            feedback_type=FeedbackType.SUCCESS,
            title="Bon travail !",
            message=f"Points positifs : {', '.join(positives)}.",
            priority=2,
        )

    async def _generate_success_feedback(
        self,
        exercise: Exercise,
        attempt: ExerciseAttempt,
        assessment: AssessmentResult,
    ) -> List[PedagogicalFeedback]:
        """Generate feedback for successful completion"""
        feedback_items = []

        # Main success message
        prompt = f"""
L'apprenant a réussi l'exercice "{exercise.title}".
Score: {assessment.score}/{assessment.max_score}

Son code:
```
{attempt.submitted_code}
```

Génère un message de félicitations qui:
1. Reconnaît spécifiquement ce qu'il a bien fait
2. Mentionne un point d'amélioration optionnel (sans être critique)
3. Explique brièvement pourquoi cette solution fonctionne

Sois enthousiaste mais pas excessif. Maximum 3-4 phrases.
"""

        response = await self.llm.ainvoke(
            [SystemMessage(content="Tu es un mentor DevOps encourageant."), HumanMessage(content=prompt)]
        )

        feedback_items.append(
            PedagogicalFeedback(
                feedback_type=FeedbackType.SUCCESS,
                title="🎉 Exercice réussi !",
                message=response.content,
                priority=1,
            )
        )

        # Optionally suggest next steps
        if exercise.solution_explanation:
            feedback_items.append(
                PedagogicalFeedback(
                    feedback_type=FeedbackType.EXPLANATION,
                    title="Pour aller plus loin",
                    message=exercise.solution_explanation,
                    priority=3,
                )
            )

        return feedback_items

    async def _generate_corrective_feedback(
        self,
        exercise: Exercise,
        attempt: ExerciseAttempt,
        assessment: AssessmentResult,
    ) -> List[PedagogicalFeedback]:
        """Generate helpful feedback for failed attempts"""
        feedback_items = []

        checks_failed = assessment.checks_failed
        execution_errors = attempt.execution_errors

        prompt = f"""
L'apprenant a soumis du code pour l'exercice "{exercise.title}" mais n'a pas réussi.

Instructions de l'exercice:
{exercise.instructions[:500]}

Code soumis:
```
{attempt.submitted_code}
```

Erreurs d'exécution:
{execution_errors or "Aucune erreur d'exécution"}

Vérifications échouées: {", ".join(checks_failed) if checks_failed else "Non spécifié"}

Génère un feedback pédagogique qui:
1. N'est PAS décourageant
2. Identifie le problème PRINCIPAL (un seul)
3. Donne une DIRECTION sans donner la solution
4. Pose une question pour faire réfléchir

Format:
PROBLEME: [description courte]
DIRECTION: [guide sans solution]
QUESTION: [question socratique]

Réponds en français.
"""

        response = await self.llm.ainvoke(
            [SystemMessage(content="Tu es un mentor patient et pédagogue."), HumanMessage(content=prompt)]
        )

        # Parse response
        content = response.content
        feedback_items.append(
            PedagogicalFeedback(
                feedback_type=FeedbackType.CORRECTION,
                title="Quelques ajustements nécessaires",
                message=content,
                priority=1,
            )
        )

        # Add specific error feedback if execution failed
        if execution_errors:
            feedback_items.append(
                PedagogicalFeedback(
                    feedback_type=FeedbackType.WARNING,
                    title="Erreur d'exécution détectée",
                    message=f"```\n{execution_errors[:500]}\n```",
                    details="Lis attentivement le message d'erreur, il contient souvent la solution.",
                    priority=2,
                )
            )

        return feedback_items

    async def _generate_socratic_question(
        self,
        exercise: Exercise,
        attempt: ExerciseAttempt,
        assessment: AssessmentResult,
    ) -> Optional[PedagogicalFeedback]:
        """Generate a Socratic question to guide thinking"""
        prompt = f"""
L'apprenant échoue plusieurs fois sur l'exercice "{exercise.title}".

Instructions:
{exercise.instructions[:300]}

Dernier code:
```
{attempt.submitted_code}
```

Génère UNE question socratique qui:
1. Amène l'apprenant à réfléchir au problème fondamental
2. Ne donne pas la réponse
3. Est ouverte (pas oui/non)

Exemples de bonnes questions:
- "Que se passe-t-il si... ?"
- "Qu'est-ce que cette commande fait exactement ?"
- "Comment le système sait-il que... ?"

Réponds uniquement avec la question.
"""

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])

        return PedagogicalFeedback(
            feedback_type=FeedbackType.QUESTION,
            title="🤔 Réfléchissons ensemble",
            message=response.content.strip(),
            follow_up_question=response.content.strip(),
            priority=2,
        )

    async def generate_hint(
        self,
        exercise: Exercise,
        attempt: Optional[ExerciseAttempt],
        hint_level: int,
    ) -> PedagogicalFeedback:
        """Generate a hint at the specified level"""
        # First, try predefined hints
        if exercise.hints and hint_level <= len(exercise.hints):
            return PedagogicalFeedback(
                feedback_type=FeedbackType.HINT,
                title=f"💡 Indice {hint_level}",
                message=exercise.hints[hint_level - 1],
                priority=1,
            )

        # Generate contextual hint
        prompt = f"""
L'apprenant a besoin d'un indice de niveau {hint_level}/5 pour l'exercice "{exercise.title}".

Niveau 1 = très subtil (direction générale)
Niveau 2 = guide vers le concept
Niveau 3 = explique ce qu'il faut chercher
Niveau 4 = montre une partie de la solution
Niveau 5 = solution presque complète

Instructions:
{exercise.instructions[:400]}

Code actuel:
```
{attempt.submitted_code if attempt else "Pas encore de code"}
```

Génère un indice de niveau {hint_level}.
Sois progressif - ne donne pas trop d'information pour ce niveau.
"""

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])

        return PedagogicalFeedback(
            feedback_type=FeedbackType.HINT,
            title=f"💡 Indice {hint_level}",
            message=response.content,
            priority=1,
        )

    def assess_understanding_level(
        self,
        exercise: Exercise,
        attempts: List[ExerciseAttempt],
    ) -> UnderstandingLevel:
        """Assess the learner's understanding level based on attempts"""
        if not attempts:
            return UnderstandingLevel.NONE

        # Check if passed
        final_attempt = attempts[-1]
        if final_attempt.passed:
            # Assess quality of solution
            if final_attempt.score >= 90 and len(attempts) == 1:
                return UnderstandingLevel.EXPERT
            elif final_attempt.score >= 80 and len(attempts) <= 2:
                return UnderstandingLevel.PROFICIENT
            elif final_attempt.score >= 70:
                return UnderstandingLevel.FUNCTIONAL
            else:
                return UnderstandingLevel.DEVELOPING
        else:
            # Didn't pass
            if len(attempts) == 1:
                return UnderstandingLevel.DEVELOPING
            elif len(attempts) <= 3:
                return UnderstandingLevel.SURFACE
            else:
                return UnderstandingLevel.NONE
