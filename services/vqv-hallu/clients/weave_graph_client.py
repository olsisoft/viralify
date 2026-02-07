"""
VQV-HALLU WeaveGraph Client
Client pour l'intégration avec le service WeaveGraph de presentation-generator
"""

import asyncio
import logging
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
import re

import aiohttp


logger = logging.getLogger(__name__)


@dataclass
class ConceptMatch:
    """Représente un concept trouvé dans le texte"""
    name: str
    canonical_name: str
    confidence: float
    source: str  # "source" ou "transcription"
    aliases: List[str] = field(default_factory=list)


@dataclass
class ConceptIntegrityResult:
    """Résultat de la vérification d'intégrité des concepts"""
    score: float  # 0.0 - 1.0
    source_concepts: List[ConceptMatch]
    transcription_concepts: List[ConceptMatch]
    matched_concepts: List[str]  # Concepts présents dans les deux
    missing_concepts: List[str]  # Concepts source absents de la transcription
    extra_concepts: List[str]    # Concepts transcription absents de la source
    phonetic_matches: List[Tuple[str, str, float]]  # (source, transcription, similarity)
    boost: float  # Boost à appliquer au score sémantique (0.0 - 0.15)


class WeaveGraphClient:
    """
    Client pour interagir avec le service WeaveGraph.

    Permet de:
    - Récupérer les concepts d'un utilisateur
    - Vérifier l'intégrité des concepts entre source et transcription
    - Calculer un boost basé sur la correspondance des concepts
    """

    def __init__(
        self,
        base_url: str = "http://presentation-generator:8006",
        timeout: float = 10.0
    ):
        """
        Initialise le client WeaveGraph.

        Args:
            base_url: URL du service presentation-generator
            timeout: Timeout en secondes
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)

        # Cache des concepts par utilisateur
        self._concept_cache: Dict[str, Dict[str, ConceptMatch]] = {}
        self._cache_ttl = 300  # 5 minutes

        # Patterns pour extraction de termes techniques
        self._tech_patterns = [
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase
            r'\b[a-z]+_[a-z_]+\b',                 # snake_case
            r'\b[A-Z]{2,}\b',                      # ACRONYMES
            r'\b(?:API|SDK|CLI|GUI|SQL|NoSQL|HTTP|REST|gRPC|JWT|OAuth)\b',
        ]

    async def fetch_user_concepts(
        self,
        user_id: str,
        force_refresh: bool = False
    ) -> Dict[str, ConceptMatch]:
        """
        Récupère les concepts d'un utilisateur depuis WeaveGraph.

        Args:
            user_id: ID de l'utilisateur
            force_refresh: Forcer le rafraîchissement du cache

        Returns:
            Dict[canonical_name, ConceptMatch]
        """
        if not force_refresh and user_id in self._concept_cache:
            return self._concept_cache[user_id]

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                url = f"{self.base_url}/api/v1/weave-graph/concepts"
                params = {"user_id": user_id, "limit": 500}

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        concepts = {}

                        for concept_data in data.get("concepts", []):
                            match = ConceptMatch(
                                name=concept_data.get("name", ""),
                                canonical_name=concept_data.get("canonical_name", ""),
                                confidence=1.0,
                                source="weave_graph",
                                aliases=concept_data.get("aliases", [])
                            )
                            concepts[match.canonical_name.lower()] = match

                        self._concept_cache[user_id] = concepts
                        logger.debug(f"Loaded {len(concepts)} concepts for user {user_id}")
                        return concepts

                    else:
                        logger.warning(f"WeaveGraph returned {response.status}")
                        return {}

        except Exception as e:
            logger.error(f"Error fetching concepts from WeaveGraph: {e}")
            return {}

    def extract_key_terms(self, text: str) -> Set[str]:
        """
        Extrait les termes techniques clés d'un texte.

        Args:
            text: Texte à analyser

        Returns:
            Set de termes normalisés
        """
        terms = set()

        # Appliquer les patterns regex
        for pattern in self._tech_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                terms.add(match.lower().replace("_", " "))

        # Ajouter les mots de plus de 5 caractères (potentiels termes techniques)
        words = text.lower().split()
        for word in words:
            # Nettoyer la ponctuation
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) >= 6:
                terms.add(clean_word)

        return terms

    def compute_phonetic_similarity(self, word1: str, word2: str) -> float:
        """
        Calcule la similarité phonétique entre deux mots.

        Utilise une combinaison de:
        - Distance de Levenshtein normalisée
        - Correspondance de consonnes (structure phonétique)

        Args:
            word1: Premier mot
            word2: Deuxième mot

        Returns:
            Score de similarité (0.0 - 1.0)
        """
        import Levenshtein

        w1 = word1.lower()
        w2 = word2.lower()

        # Distance de Levenshtein normalisée
        max_len = max(len(w1), len(w2))
        if max_len == 0:
            return 0.0

        lev_sim = 1 - (Levenshtein.distance(w1, w2) / max_len)

        # Correspondance des consonnes
        consonants = "bcdfghjklmnpqrstvwxz"

        def get_consonants(word):
            return ''.join(c for c in word if c in consonants)

        c1 = get_consonants(w1)
        c2 = get_consonants(w2)

        if len(c1) == 0 or len(c2) == 0:
            consonant_sim = 0.0
        else:
            consonant_sim = 1 - (Levenshtein.distance(c1, c2) / max(len(c1), len(c2)))

        # Combinaison pondérée
        return (lev_sim * 0.6) + (consonant_sim * 0.4)

    async def check_concept_integrity(
        self,
        source_text: str,
        transcription_text: str,
        user_id: Optional[str] = None
    ) -> ConceptIntegrityResult:
        """
        Vérifie l'intégrité des concepts entre source et transcription.

        Args:
            source_text: Texte source original
            transcription_text: Transcription ASR
            user_id: ID utilisateur pour récupérer les concepts WeaveGraph

        Returns:
            ConceptIntegrityResult avec score et détails
        """
        # Extraire les termes des deux textes
        source_terms = self.extract_key_terms(source_text)
        transcript_terms = self.extract_key_terms(transcription_text)

        # Si user_id fourni, enrichir avec WeaveGraph
        weave_concepts = {}
        if user_id:
            weave_concepts = await self.fetch_user_concepts(user_id)

        # Créer les ConceptMatch pour source
        source_concepts = []
        for term in source_terms:
            # Vérifier si le terme est dans WeaveGraph
            if term in weave_concepts:
                source_concepts.append(weave_concepts[term])
            else:
                source_concepts.append(ConceptMatch(
                    name=term,
                    canonical_name=term,
                    confidence=0.8,
                    source="source"
                ))

        # Créer les ConceptMatch pour transcription
        transcript_concepts = []
        for term in transcript_terms:
            if term in weave_concepts:
                transcript_concepts.append(weave_concepts[term])
            else:
                transcript_concepts.append(ConceptMatch(
                    name=term,
                    canonical_name=term,
                    confidence=0.8,
                    source="transcription"
                ))

        # Trouver les correspondances
        matched = []
        missing = []
        phonetic_matches = []

        source_canonical = {c.canonical_name.lower() for c in source_concepts}
        transcript_canonical = {c.canonical_name.lower() for c in transcript_concepts}

        for src_concept in source_concepts:
            src_name = src_concept.canonical_name.lower()

            # Correspondance exacte
            if src_name in transcript_canonical:
                matched.append(src_name)
                continue

            # Vérifier les aliases
            alias_found = False
            for alias in src_concept.aliases:
                if alias.lower() in transcript_canonical:
                    matched.append(src_name)
                    alias_found = True
                    break

            if alias_found:
                continue

            # Vérifier la correspondance phonétique
            best_phonetic = None
            best_sim = 0.0

            for trans_name in transcript_canonical:
                sim = self.compute_phonetic_similarity(src_name, trans_name)
                if sim > 0.75 and sim > best_sim:  # Seuil de 75%
                    best_sim = sim
                    best_phonetic = trans_name

            if best_phonetic:
                phonetic_matches.append((src_name, best_phonetic, best_sim))
                matched.append(src_name)  # Considérer comme match
            else:
                missing.append(src_name)

        # Concepts extra (dans transcription mais pas dans source)
        extra = list(transcript_canonical - source_canonical - set(matched))

        # Calculer le score
        if not source_concepts:
            score = 1.0
        else:
            direct_match_score = len([m for m in matched if m not in [p[0] for p in phonetic_matches]]) / len(source_concepts)
            phonetic_match_score = sum(p[2] for p in phonetic_matches) / len(source_concepts) if phonetic_matches else 0
            score = direct_match_score + (phonetic_match_score * 0.5)  # Phonetic matches comptent pour 50%
            score = min(1.0, score)

        # Calculer le boost
        # Max boost: 15% si tous les concepts matchent parfaitement
        if score >= 0.9:
            boost = 0.15
        elif score >= 0.7:
            boost = 0.10
        elif score >= 0.5:
            boost = 0.05
        else:
            boost = 0.0

        # Bonus si WeaveGraph utilisé
        if weave_concepts and len(matched) > 0:
            weave_matched = sum(1 for m in matched if m in weave_concepts)
            if weave_matched > 0:
                boost += min(0.05, weave_matched * 0.01)  # Max +5% bonus WeaveGraph

        boost = min(0.15, boost)  # Cap at 15%

        return ConceptIntegrityResult(
            score=score,
            source_concepts=source_concepts,
            transcription_concepts=transcript_concepts,
            matched_concepts=matched,
            missing_concepts=missing,
            extra_concepts=extra,
            phonetic_matches=phonetic_matches,
            boost=boost
        )


# Factory function
def create_weave_graph_client(
    base_url: Optional[str] = None
) -> Optional[WeaveGraphClient]:
    """
    Crée un client WeaveGraph si l'URL est configurée.

    Args:
        base_url: URL du service (ou None pour désactiver)

    Returns:
        WeaveGraphClient ou None
    """
    if not base_url:
        return None

    return WeaveGraphClient(base_url=base_url)
