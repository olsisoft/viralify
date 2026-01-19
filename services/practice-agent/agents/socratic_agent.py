"""
Socratic Agent

An agent that uses the Socratic method to guide learning through questioning.
"""

from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from models.practice_models import Exercise, Message
from models.assessment_models import SocraticQuestion, LearningMoment


class SocraticAgent:
    """
    Uses the Socratic method to:
    - Guide learners to discover answers themselves
    - Probe understanding through questioning
    - Identify and address misconceptions
    - Deepen conceptual understanding
    """

    SOCRATIC_SYSTEM_PROMPT = """
Tu es un mentor Socratique pour l'enseignement DevOps.

Principes Socratiques:
1. JAMAIS donner la réponse directement
2. Poser des questions qui font réfléchir
3. Déconstruire les affirmations pour tester la compréhension
4. Guider vers la découverte par le questionnement
5. Accepter "je ne sais pas" comme point de départ

Types de questions à utiliser:
- Clarification: "Que veux-tu dire par...?"
- Hypothèse: "Que se passerait-il si...?"
- Raison: "Pourquoi penses-tu que...?"
- Preuve: "Comment sais-tu que...?"
- Implication: "Si c'est vrai, alors...?"
- Perspective: "Comment quelqu'un d'autre verrait-il...?"

Ton objectif: Amener l'apprenant à la compréhension par lui-même.
"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.8)
        self.conversation_history: List[Message] = []

    async def respond_to_stuck_learner(
        self,
        learner_code: str,
        exercise: Exercise,
        stuck_point: str,
        conversation_history: List[Message],
    ) -> Dict[str, Any]:
        """
        Instead of giving the solution, ask guiding questions.
        """
        # Build conversation context
        history_text = "\n".join([
            f"{m.role}: {m.content}" for m in conversation_history[-5:]
        ])

        prompt = f"""
L'apprenant est bloqué sur l'exercice "{exercise.title}".

Instructions de l'exercice:
{exercise.instructions}

Code actuel:
```
{learner_code}
```

Ce qui bloque l'apprenant: {stuck_point}

Historique récent:
{history_text}

En tant que mentor Socratique:
1. Identifie le gap conceptuel probable
2. Formule 1-2 questions qui guident vers la compréhension
3. Si approprié, demande à l'apprenant d'expliquer son raisonnement

Réponds avec:
- Une courte phrase d'encouragement
- Ta/tes question(s) Socratique(s)
- Optionnel: une demande de clarification

Ne donne JAMAIS la solution, même partiellement.
"""

        response = await self.llm.ainvoke([
            SystemMessage(content=self.SOCRATIC_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])

        return {
            "response": response.content,
            "response_type": "socratic_question",
            "should_wait_for_answer": True,
        }

    async def evaluate_learner_explanation(
        self,
        learner_explanation: str,
        concept: str,
        expected_understanding: str,
    ) -> Dict[str, Any]:
        """
        Evaluate if the learner truly understands or is just repeating.
        """
        prompt = f"""
L'apprenant explique sa compréhension du concept: "{concept}"

Son explication:
"{learner_explanation}"

Compréhension attendue:
{expected_understanding}

Évalue:
1. L'apprenant comprend-il vraiment ou répète-t-il?
2. Y a-t-il des misconceptions?
3. La compréhension est-elle superficielle ou profonde?

Formule une question de suivi pour:
- Approfondir si compréhension superficielle
- Corriger si misconception
- Challenger si compréhension correcte

Réponds en JSON:
{{
    "understanding_level": "none|surface|partial|good|excellent",
    "misconceptions": ["liste des misconceptions détectées"],
    "follow_up_question": "ta question de suivi",
    "encouragement": "message d'encouragement adapté"
}}
"""

        response = await self.llm.ainvoke([
            SystemMessage(content=self.SOCRATIC_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])

        try:
            import json
            return json.loads(response.content)
        except:
            return {
                "understanding_level": "partial",
                "misconceptions": [],
                "follow_up_question": response.content,
                "encouragement": "Continue à réfléchir!"
            }

    async def generate_socratic_dialogue(
        self,
        concept: str,
        learner_level: str,
        context: Optional[str] = None,
    ) -> List[SocraticQuestion]:
        """
        Generate a sequence of Socratic questions to teach a concept.
        """
        prompt = f"""
Génère une séquence de questions Socratiques pour enseigner le concept:
"{concept}"

Niveau de l'apprenant: {learner_level}
Contexte: {context or "Pas de contexte spécifique"}

Crée 4-5 questions progressives qui:
1. Commencent par ce que l'apprenant sait déjà
2. Construisent progressivement vers le concept
3. Permettent de découvrir le concept par soi-même
4. Se terminent par une question de vérification

Format JSON:
[
    {{
        "question": "La question",
        "purpose": "Pourquoi on pose cette question",
        "expected_insight": "Ce qu'on espère que l'apprenant réalise",
        "if_wrong": "Question de suivi si mauvaise réponse",
        "concept": "Sous-concept testé"
    }}
]
"""

        response = await self.llm.ainvoke([
            SystemMessage(content=self.SOCRATIC_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])

        try:
            import json
            questions_data = json.loads(response.content)
            return [
                SocraticQuestion(
                    question=q["question"],
                    purpose=q["purpose"],
                    expected_insight=q["expected_insight"],
                    follow_up_if_wrong=q.get("if_wrong", ""),
                    concept=q.get("concept", concept),
                )
                for q in questions_data
            ]
        except:
            return []

    async def identify_learning_moment(
        self,
        learner_action: str,
        exercise: Exercise,
        conversation_history: List[Message],
    ) -> Optional[LearningMoment]:
        """
        Identify teachable moments from learner interactions.
        """
        prompt = f"""
Analyse cette action de l'apprenant pour identifier un "moment d'apprentissage".

Exercice: {exercise.title}
Action de l'apprenant: {learner_action}

Types de moments d'apprentissage:
- ERREUR: Une erreur révélant une misconception
- QUESTION: Une question montrant curiosité ou confusion
- BREAKTHROUGH: Un déclic ou une réalisation
- PATTERN: Un pattern de code à améliorer

Si c'est un moment d'apprentissage, réponds en JSON:
{{
    "is_learning_moment": true,
    "type": "error|question|breakthrough|pattern",
    "concept": "Le concept à enseigner",
    "explanation": "Explication claire du concept",
    "example": "Exemple de code si applicable",
    "socratic_question": "Question pour approfondir"
}}

Si ce n'est pas un moment d'apprentissage:
{{
    "is_learning_moment": false
}}
"""

        response = await self.llm.ainvoke([
            SystemMessage(content="Tu identifies les moments d'apprentissage."),
            HumanMessage(content=prompt)
        ])

        try:
            import json
            data = json.loads(response.content)

            if not data.get("is_learning_moment"):
                return None

            return LearningMoment(
                trigger=learner_action,
                trigger_type=data.get("type", "error"),
                concept=data.get("concept", ""),
                explanation=data.get("explanation", ""),
                example_code=data.get("example"),
                socratic_questions=[
                    SocraticQuestion(
                        question=data.get("socratic_question", ""),
                        purpose="Approfondir la compréhension",
                        expected_insight="Réalisation du concept",
                        concept=data.get("concept", ""),
                    )
                ] if data.get("socratic_question") else [],
            )
        except:
            return None

    async def challenge_correct_answer(
        self,
        learner_answer: str,
        exercise: Exercise,
    ) -> str:
        """
        Even when the answer is correct, challenge to deepen understanding.
        """
        prompt = f"""
L'apprenant a correctement résolu l'exercice "{exercise.title}".

Sa solution:
```
{learner_answer}
```

Génère une question qui:
1. Reconnaît que la solution est correcte
2. Challenge l'apprenant à aller plus loin
3. Explore un cas limite ou une alternative

Exemples:
- "Excellent! Et si tu devais gérer 1000 fois plus de données?"
- "Parfait! Peux-tu expliquer pourquoi cette approche est meilleure que...?"
- "Bien joué! Comment modifierais-tu cela pour...?"

Réponds avec ta question de challenge.
"""

        response = await self.llm.ainvoke([
            SystemMessage(content=self.SOCRATIC_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])

        return response.content

    async def handle_i_dont_know(
        self,
        concept: str,
        exercise: Exercise,
    ) -> str:
        """
        Handle when learner says "I don't know" - this is a starting point.
        """
        prompt = f"""
L'apprenant a dit "je ne sais pas" concernant le concept "{concept}"
dans le contexte de l'exercice "{exercise.title}".

Dans la méthode Socratique, "je ne sais pas" est le point de départ idéal.

Réponds avec:
1. Validation que c'est OK de ne pas savoir
2. Une question très simple pour commencer
3. Un lien avec quelque chose que l'apprenant connaît probablement

La question doit être si simple que l'apprenant peut y répondre.
"""

        response = await self.llm.ainvoke([
            SystemMessage(content=self.SOCRATIC_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])

        return response.content
