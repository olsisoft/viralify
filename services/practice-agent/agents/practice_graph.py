"""
Practice Agent - LangGraph Implementation

Main agent that orchestrates the practice session using LangGraph.
"""

import asyncio
import json
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from models.practice_models import (
    Exercise,
    PracticeSession,
    ExerciseAttempt,
    Message,
    DifficultyLevel,
    ExerciseCategory,
)
from models.sandbox_models import SandboxResult, ExecutionResult
from models.assessment_models import (
    AssessmentResult,
    PedagogicalFeedback,
    FeedbackType,
    UnderstandingLevel,
)


class PracticeAgentState(TypedDict):
    """State for the practice agent graph"""
    # Messages (conversation history)
    messages: Annotated[list, add_messages]

    # Session info
    session_id: str
    user_id: str
    course_id: Optional[str]

    # Current exercise
    current_exercise: Optional[Dict[str, Any]]
    current_attempt: Optional[Dict[str, Any]]

    # Learner's input
    learner_code: Optional[str]
    learner_message: Optional[str]

    # Execution results
    sandbox_result: Optional[Dict[str, Any]]
    execution_output: Optional[str]

    # Assessment
    assessment: Optional[Dict[str, Any]]
    feedback: Optional[str]

    # Navigation
    intent: Optional[str]
    next_action: Optional[str]

    # Progress tracking
    exercises_completed: List[str]
    hints_used: int
    total_points: int

    # Settings
    difficulty_preference: str
    categories_focus: List[str]
    pair_programming_mode: bool


class PracticeAgent:
    """
    LangGraph-based Practice Agent

    Orchestrates practice sessions with:
    - Exercise selection based on course content and difficulty
    - Code execution in sandboxed environments
    - Pedagogical feedback generation
    - Socratic questioning
    - Progress tracking
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
        )
        self.fast_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
        )
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine"""
        graph = StateGraph(PracticeAgentState)

        # Add nodes
        graph.add_node("understand_intent", self._understand_intent)
        graph.add_node("select_exercise", self._select_exercise)
        graph.add_node("present_exercise", self._present_exercise)
        graph.add_node("execute_code", self._execute_code)
        graph.add_node("evaluate_solution", self._evaluate_solution)
        graph.add_node("provide_feedback", self._provide_feedback)
        graph.add_node("give_hint", self._give_hint)
        graph.add_node("explain_concept", self._explain_concept)
        graph.add_node("pair_program", self._pair_program)
        graph.add_node("chat_response", self._chat_response)
        graph.add_node("complete_exercise", self._complete_exercise)

        # Start edge
        graph.add_edge(START, "understand_intent")

        # Conditional edges from understand_intent
        graph.add_conditional_edges(
            "understand_intent",
            self._route_by_intent,
            {
                "start_exercise": "select_exercise",
                "submit_code": "execute_code",
                "need_hint": "give_hint",
                "need_explanation": "explain_concept",
                "pair_program": "pair_program",
                "general_chat": "chat_response",
                "skip_exercise": "select_exercise",
                "quit": END,
            }
        )

        # Exercise flow
        graph.add_edge("select_exercise", "present_exercise")
        graph.add_edge("present_exercise", END)

        # Code execution flow
        graph.add_edge("execute_code", "evaluate_solution")
        graph.add_conditional_edges(
            "evaluate_solution",
            self._check_if_passed,
            {
                "passed": "complete_exercise",
                "failed": "provide_feedback",
            }
        )

        graph.add_edge("provide_feedback", END)
        graph.add_edge("complete_exercise", END)

        # Help flow
        graph.add_edge("give_hint", END)
        graph.add_edge("explain_concept", END)
        graph.add_edge("pair_program", END)
        graph.add_edge("chat_response", END)

        return graph.compile(checkpointer=self.memory)

    async def _understand_intent(self, state: PracticeAgentState) -> Dict[str, Any]:
        """Understand what the learner wants to do"""
        learner_message = state.get("learner_message", "")
        current_exercise = state.get("current_exercise")
        learner_code = state.get("learner_code")

        # Build context for intent classification
        context = f"""
Current exercise: {current_exercise['title'] if current_exercise else 'None'}
Has submitted code: {bool(learner_code)}
Learner message: {learner_message}
"""

        prompt = f"""
Classify the learner's intent from their message. Choose ONE of:
- start_exercise: Wants to start a new exercise or the next one
- submit_code: Has submitted code to be evaluated
- need_hint: Asking for help/hint
- need_explanation: Wants explanation of a concept
- pair_program: Wants you to code with them
- general_chat: General question or conversation
- skip_exercise: Wants to skip current exercise
- quit: Wants to end the session

Context:
{context}

Respond with just the intent name, nothing else.
"""

        response = await self.fast_llm.ainvoke([
            SystemMessage(content="You are an intent classifier. Respond with only the intent name."),
            HumanMessage(content=prompt)
        ])

        intent = response.content.strip().lower()

        # Default to submit_code if code was provided
        if learner_code and not learner_message:
            intent = "submit_code"

        # If starting fresh and no message, start exercise
        if not current_exercise and not learner_message:
            intent = "start_exercise"

        return {"intent": intent}

    def _route_by_intent(self, state: PracticeAgentState) -> str:
        """Route to the appropriate node based on intent"""
        intent = state.get("intent", "general_chat")
        valid_intents = [
            "start_exercise", "submit_code", "need_hint",
            "need_explanation", "pair_program", "general_chat",
            "skip_exercise", "quit"
        ]
        if intent not in valid_intents:
            return "general_chat"
        return intent

    async def _select_exercise(self, state: PracticeAgentState) -> Dict[str, Any]:
        """Select the next appropriate exercise"""
        from services.exercise_service import ExerciseService

        exercise_service = ExerciseService()

        # Get learner's context
        completed = state.get("exercises_completed", [])
        difficulty = state.get("difficulty_preference", "beginner")
        categories = state.get("categories_focus", [])
        course_id = state.get("course_id")

        # Select next exercise
        exercise = await exercise_service.select_next_exercise(
            completed_exercises=completed,
            difficulty=difficulty,
            categories=categories,
            course_id=course_id,
        )

        return {
            "current_exercise": exercise.model_dump() if exercise else None,
            "current_attempt": None,
            "hints_used": 0,
        }

    async def _present_exercise(self, state: PracticeAgentState) -> Dict[str, Any]:
        """Present the exercise to the learner"""
        exercise = state.get("current_exercise")

        if not exercise:
            message = """
Je n'ai pas trouvé d'exercice correspondant à tes critères.
Veux-tu ajuster le niveau de difficulté ou les catégories ?
"""
        else:
            message = f"""
## 🎯 Exercice: {exercise['title']}

**Difficulté**: {exercise['difficulty'].capitalize()}
**Catégorie**: {exercise['category']}
**Points**: {exercise['points']} pts
**Durée estimée**: ~{exercise['estimated_minutes']} minutes

---

### Instructions

{exercise['instructions']}

---

"""
            if exercise.get('starter_code'):
                message += f"""
### Code de départ

```
{exercise['starter_code']}
```

---

"""

            message += """
💡 **Commandes disponibles**:
- Soumettre ton code pour évaluation
- Demander un indice si tu es bloqué
- Demander une explication sur un concept
- Demander de coder ensemble (pair programming)

Bonne chance ! 🚀
"""

        return {
            "messages": [AIMessage(content=message)],
            "feedback": message,
        }

    async def _execute_code(self, state: PracticeAgentState) -> Dict[str, Any]:
        """Execute the learner's code in a sandbox"""
        from services.sandbox_manager import SandboxManager

        exercise = state.get("current_exercise")
        learner_code = state.get("learner_code", "")

        if not exercise:
            return {
                "execution_output": "Aucun exercice en cours.",
                "sandbox_result": None,
            }

        sandbox_manager = SandboxManager()

        # Execute in appropriate sandbox
        result = await sandbox_manager.execute(
            sandbox_type=exercise.get("sandbox_type", "docker"),
            code=learner_code,
            exercise_config=exercise.get("sandbox_config", {}),
            timeout=exercise.get("timeout_seconds", 60),
        )

        return {
            "sandbox_result": result.model_dump() if result else None,
            "execution_output": result.execution.combined_output if result else "",
        }

    async def _evaluate_solution(self, state: PracticeAgentState) -> Dict[str, Any]:
        """Evaluate the learner's solution"""
        from services.assessment_service import AssessmentService

        exercise = state.get("current_exercise")
        sandbox_result = state.get("sandbox_result")
        learner_code = state.get("learner_code", "")

        assessment_service = AssessmentService()

        assessment = await assessment_service.evaluate(
            exercise=exercise,
            submitted_code=learner_code,
            execution_result=sandbox_result,
        )

        return {
            "assessment": assessment.model_dump() if assessment else None,
        }

    def _check_if_passed(self, state: PracticeAgentState) -> Literal["passed", "failed"]:
        """Check if the learner passed the exercise"""
        assessment = state.get("assessment", {})
        return "passed" if assessment.get("passed", False) else "failed"

    async def _provide_feedback(self, state: PracticeAgentState) -> Dict[str, Any]:
        """Provide pedagogical feedback on failed attempt"""
        assessment = state.get("assessment", {})
        exercise = state.get("current_exercise", {})
        learner_code = state.get("learner_code", "")
        execution_output = state.get("execution_output", "")

        # Generate helpful feedback
        prompt = f"""
Tu es un mentor DevOps bienveillant et pédagogue.

L'apprenant a soumis du code pour cet exercice:
**{exercise.get('title', 'Exercice')}**

Instructions de l'exercice:
{exercise.get('instructions', '')}

Code soumis:
```
{learner_code}
```

Sortie d'exécution:
```
{execution_output}
```

Résultat de l'évaluation:
- Passé: Non
- Score: {assessment.get('score', 0)}/{assessment.get('max_score', 100)}
- Vérifications échouées: {', '.join(assessment.get('checks_failed', []))}

Génère un feedback pédagogique:
1. Commence par quelque chose de positif (ce qu'ils ont bien fait)
2. Explique clairement le problème principal SANS donner la solution
3. Pose une question socratique pour les guider
4. Termine par un encouragement

Sois concis mais utile. Utilise le français.
"""

        response = await self.llm.ainvoke([
            SystemMessage(content="Tu es un mentor technique bienveillant."),
            HumanMessage(content=prompt)
        ])

        feedback_message = f"""
## 📝 Feedback

{response.content}

---

**Score actuel**: {assessment.get('score', 0)}/{assessment.get('max_score', 100)} points

Tu peux:
- Corriger ton code et le soumettre à nouveau
- Demander un indice
- Demander une explication
"""

        return {
            "messages": [AIMessage(content=feedback_message)],
            "feedback": feedback_message,
        }

    async def _complete_exercise(self, state: PracticeAgentState) -> Dict[str, Any]:
        """Handle successful exercise completion"""
        exercise = state.get("current_exercise", {})
        assessment = state.get("assessment", {})
        completed = state.get("exercises_completed", [])
        total_points = state.get("total_points", 0)

        # Update progress
        exercise_id = exercise.get("id", "")
        if exercise_id and exercise_id not in completed:
            completed.append(exercise_id)

        points_earned = assessment.get("score", 0)
        total_points += points_earned

        success_message = f"""
## 🎉 Bravo !

Tu as réussi l'exercice **{exercise.get('title', 'Exercice')}** !

**Score**: {assessment.get('score', 0)}/{assessment.get('max_score', 100)} points

### Ce que tu as appris:
{exercise.get('solution_explanation', 'Excellente maîtrise des concepts!')}

---

**Points totaux**: {total_points} pts
**Exercices complétés**: {len(completed)}

Prêt pour le prochain défi ? 💪
"""

        return {
            "messages": [AIMessage(content=success_message)],
            "feedback": success_message,
            "exercises_completed": completed,
            "total_points": total_points,
            "current_exercise": None,
        }

    async def _give_hint(self, state: PracticeAgentState) -> Dict[str, Any]:
        """Provide a progressive hint"""
        exercise = state.get("current_exercise", {})
        hints = exercise.get("hints", [])
        hints_used = state.get("hints_used", 0)
        learner_code = state.get("learner_code", "")

        if hints_used >= len(hints):
            # Generate a contextual hint using LLM
            prompt = f"""
L'apprenant est bloqué sur cet exercice et a déjà utilisé tous les indices prédéfinis.

Exercice: {exercise.get('title', '')}
Instructions: {exercise.get('instructions', '')}
Son code actuel:
```
{learner_code}
```

Génère un indice utile qui:
1. Ne donne PAS la solution
2. Guide vers la bonne direction
3. Pose une question pour faire réfléchir

Réponds en français.
"""
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            hint = response.content
        else:
            hint = hints[hints_used]

        hint_message = f"""
## 💡 Indice #{hints_used + 1}

{hint}

---

*{len(hints) - hints_used - 1} indices restants*
*-10 points par indice utilisé*
"""

        return {
            "messages": [AIMessage(content=hint_message)],
            "feedback": hint_message,
            "hints_used": hints_used + 1,
        }

    async def _explain_concept(self, state: PracticeAgentState) -> Dict[str, Any]:
        """Explain a concept related to the current exercise"""
        exercise = state.get("current_exercise", {})
        learner_message = state.get("learner_message", "")
        learner_code = state.get("learner_code", "")

        prompt = f"""
Tu es un formateur DevOps expert et pédagogue.

L'apprenant travaille sur: {exercise.get('title', 'un exercice')}
Catégorie: {exercise.get('category', 'DevOps')}

Il demande: {learner_message}

Son code actuel:
```
{learner_code}
```

Fournis une explication claire et pédagogique:
1. Explique le concept demandé
2. Donne un exemple simple
3. Relie à l'exercice en cours
4. Pose une question pour vérifier la compréhension

Réponds en français. Sois concis mais complet.
"""

        response = await self.llm.ainvoke([
            SystemMessage(content="Tu es un mentor technique expert et bienveillant."),
            HumanMessage(content=prompt)
        ])

        return {
            "messages": [AIMessage(content=response.content)],
            "feedback": response.content,
        }

    async def _pair_program(self, state: PracticeAgentState) -> Dict[str, Any]:
        """Enter pair programming mode"""
        exercise = state.get("current_exercise", {})
        learner_code = state.get("learner_code", "")

        prompt = f"""
Tu es un pair programmer DevOps. L'apprenant veut coder avec toi.

Exercice: {exercise.get('title', '')}
Instructions: {exercise.get('instructions', '')}

Code actuel de l'apprenant:
```
{learner_code or exercise.get('starter_code', '# Pas de code encore')}
```

En mode pair programming:
1. Propose de commencer par une étape spécifique
2. Montre UN SEUL petit bout de code
3. Explique ce que tu fais
4. Demande à l'apprenant de continuer

NE DONNE PAS la solution complète. Guide pas à pas.
Réponds en français.
"""

        response = await self.llm.ainvoke([
            SystemMessage(content="Tu es un pair programmer patient et pédagogue."),
            HumanMessage(content=prompt)
        ])

        pair_message = f"""
## 👥 Mode Pair Programming

{response.content}

---

Continue à m'envoyer ton code et je t'aiderai étape par étape !
"""

        return {
            "messages": [AIMessage(content=pair_message)],
            "feedback": pair_message,
            "pair_programming_mode": True,
        }

    async def _chat_response(self, state: PracticeAgentState) -> Dict[str, Any]:
        """Handle general conversation"""
        learner_message = state.get("learner_message", "")
        exercise = state.get("current_exercise")

        context = ""
        if exercise:
            context = f"L'apprenant travaille sur l'exercice '{exercise.get('title', '')}' en {exercise.get('category', 'DevOps')}."

        prompt = f"""
Tu es un assistant de formation DevOps sympathique et compétent.

Contexte: {context}
Message de l'apprenant: {learner_message}

Réponds de manière utile et engageante. Propose de l'aide si approprié.
Réponds en français.
"""

        response = await self.llm.ainvoke([
            SystemMessage(content="Tu es un assistant de formation DevOps sympathique."),
            HumanMessage(content=prompt)
        ])

        return {
            "messages": [AIMessage(content=response.content)],
            "feedback": response.content,
        }

    async def invoke(
        self,
        session_id: str,
        user_id: str,
        message: Optional[str] = None,
        code: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run the agent with the given input"""
        config = {"configurable": {"thread_id": session_id}}

        # Build initial state
        state = {
            "session_id": session_id,
            "user_id": user_id,
            "learner_message": message,
            "learner_code": code,
            "course_id": kwargs.get("course_id"),
            "difficulty_preference": kwargs.get("difficulty", "beginner"),
            "categories_focus": kwargs.get("categories", []),
            "exercises_completed": kwargs.get("exercises_completed", []),
            "hints_used": kwargs.get("hints_used", 0),
            "total_points": kwargs.get("total_points", 0),
            "current_exercise": kwargs.get("current_exercise"),
            "pair_programming_mode": kwargs.get("pair_programming_mode", False),
        }

        if message:
            state["messages"] = [HumanMessage(content=message)]

        # Run the graph
        result = await self.graph.ainvoke(state, config)

        return result


def create_practice_agent() -> PracticeAgent:
    """Factory function to create a practice agent"""
    return PracticeAgent()
