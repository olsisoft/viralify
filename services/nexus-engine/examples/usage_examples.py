#!/usr/bin/env python3
"""
NEXUS - Exemples d'utilisation
Neural Execution & Understanding Synthesis
"""

import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Importer NEXUS
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from nexus import (
    generate_code,
    NEXUSPipeline,
    NexusRequest,
    NexusConfig,
    TargetAudience,
    CodeVerbosity,
    ExecutionMode,
    create_nexus_pipeline,
)


# =============================================================================
# EXEMPLE 1: Génération simple
# =============================================================================
def exemple_simple():
    """
    Génération de code en une ligne.
    """
    print("\n" + "="*60)
    print("EXEMPLE 1: Génération simple de code")
    print("="*60)
    
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("⚠️ GROQ_API_KEY non définie")
        return None
    
    result = generate_code(
        project_description="une plateforme e-commerce simple",
        skill_level="intermediate",
        language="python",
        audience="developer",
        allocated_time=180,  # 3 minutes
        provider="groq",
        api_key=api_key,
    )
    
    print(f"\n✅ Génération terminée:")
    print(f"   • {len(result.code_segments)} segments de code")
    print(f"   • {result.total_lines_of_code} lignes de code")
    print(f"   • {result.total_duration_seconds}s de contenu")
    
    # Afficher les segments
    print("\n📄 Segments générés:")
    for segment in result.get_segments_ordered():
        print(f"\n   [{segment.filename}]")
        print(f"   {segment.explanation[:100]}...")
        print(f"   Concepts: {segment.key_concepts}")
    
    return result


# =============================================================================
# EXEMPLE 2: Configuration développeur
# =============================================================================
def exemple_developpeur():
    """
    Configuration pour un public développeur.
    Code production-ready avec gestion d'erreurs.
    """
    print("\n" + "="*60)
    print("EXEMPLE 2: Configuration pour développeur")
    print("="*60)
    
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("⚠️ GROQ_API_KEY non définie")
        return None
    
    request = NexusRequest(
        project_description="une API REST de gestion de produits avec CRUD complet",
        lesson_context="Formation développement backend Python",
        skill_level="advanced",
        language="python",
        target_audience=TargetAudience.DEVELOPER,
        verbosity=CodeVerbosity.PRODUCTION,
        execution_mode=ExecutionMode.SANDBOX,
        allocated_time_seconds=300,
        show_mistakes=True,
        include_tests=False,
    )
    
    pipeline = create_nexus_pipeline(provider="groq", api_key=api_key)
    
    def on_progress(p):
        print(f"   [{p.stage}] {p.percent:.0f}% - {p.message}")
    
    pipeline.set_progress_callback(on_progress)
    
    print("\n🚀 Génération en cours...")
    result = pipeline.generate(request)
    
    print(f"\n✅ Résultat:")
    print(f"   Framework choisi: {result.architecture_dna.framework}")
    print(f"   Patterns: {[p.value for p in result.architecture_dna.patterns]}")
    print(f"   Entités: {[e.name for e in result.architecture_dna.entities]}")
    
    return result


# =============================================================================
# EXEMPLE 3: Configuration architecte
# =============================================================================
def exemple_architecte():
    """
    Configuration pour un public architecte.
    Focus sur la structure et les patterns.
    """
    print("\n" + "="*60)
    print("EXEMPLE 3: Configuration pour architecte")
    print("="*60)
    
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("⚠️ GROQ_API_KEY non définie")
        return None
    
    request = NexusRequest(
        project_description="un système de gestion de commandes avec microservices",
        lesson_context="Architecture logicielle avancée",
        skill_level="expert",
        language="python",
        target_audience=TargetAudience.ARCHITECT,
        verbosity=CodeVerbosity.STANDARD,  # Moins verbeux, focus structure
        allocated_time_seconds=240,
        show_mistakes=False,
        show_evolution=False,
    )
    
    pipeline = create_nexus_pipeline(provider="groq", api_key=api_key)
    result = pipeline.generate(request)
    
    # Afficher le blueprint cognitif
    print(f"\n🧠 Blueprint cognitif:")
    print(f"   • Analyse: {len(result.cognitive_blueprint.analysis_phase)} étapes")
    print(f"   • Design: {len(result.cognitive_blueprint.design_phase)} étapes")
    print(f"   • Implémentation: {len(result.cognitive_blueprint.implementation_phase)} étapes")
    print(f"   • Validation: {len(result.cognitive_blueprint.validation_phase)} étapes")
    
    # Afficher les décisions architecturales
    print(f"\n🏗️ Architecture DNA:")
    print(f"   Layers: {result.architecture_dna.layers}")
    print(f"   Patterns: {[p.value for p in result.architecture_dna.patterns]}")
    
    return result


# =============================================================================
# EXEMPLE 4: Génération avec versions progressives
# =============================================================================
def exemple_evolution():
    """
    Génération avec versions progressives du code (v1 → v2 → v3).
    """
    print("\n" + "="*60)
    print("EXEMPLE 4: Versions progressives")
    print("="*60)
    
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("⚠️ GROQ_API_KEY non définie")
        return None
    
    request = NexusRequest(
        project_description="un système d'authentification avec JWT",
        lesson_context="Sécurité des applications web",
        skill_level="intermediate",
        language="python",
        target_audience=TargetAudience.STUDENT,
        verbosity=CodeVerbosity.VERBOSE,
        allocated_time_seconds=360,
        show_mistakes=True,
        show_evolution=True,  # Activer les versions progressives
    )
    
    pipeline = create_nexus_pipeline(provider="groq", api_key=api_key)
    result = pipeline.generate(request)
    
    print(f"\n📈 Évolution du code:")
    for segment in result.get_segments_ordered():
        if "_v1" in segment.id:
            print(f"\n   V1 (naïve): {segment.filename}")
        elif "_v2" in segment.id:
            print(f"   V2 (améliorée): {segment.filename}")
        elif "_v3" in segment.id:
            print(f"   V3 (finale): {segment.filename}")
    
    return result


# =============================================================================
# EXEMPLE 5: Export pour pipeline vidéo
# =============================================================================
def exemple_export_video():
    """
    Export des données pour intégration avec un pipeline vidéo.
    """
    print("\n" + "="*60)
    print("EXEMPLE 5: Export pour pipeline vidéo")
    print("="*60)
    
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("⚠️ GROQ_API_KEY non définie")
        # Créer un exemple statique pour la démo
        print("\n📹 Structure de sortie pour l'assembleur vidéo:\n")
        print(json.dumps({
            "timeline": [
                {
                    "segment_id": "seg_001",
                    "start_time_seconds": 0,
                    "duration_seconds": 30,
                    "filename": "models/product.py",
                    "narration_script": "Commençons par créer notre modèle Product...",
                    "key_concepts": ["model", "dataclass"],
                    "display_mode": "code_editor"
                },
                {
                    "segment_id": "seg_002",
                    "start_time_seconds": 30,
                    "duration_seconds": 45,
                    "filename": "repositories/product_repository.py",
                    "narration_script": "Maintenant, créons le repository...",
                    "key_concepts": ["repository", "CRUD"],
                    "display_mode": "code_editor"
                }
            ],
            "total_duration_seconds": 75
        }, indent=2))
        return None
    
    result = generate_code(
        project_description="un blog simple",
        allocated_time=120,
        provider="groq",
        api_key=api_key,
    )
    
    # Exporter pour l'assembleur
    sync_data = result.sync_metadata
    
    print(f"\n📹 Données pour l'assembleur vidéo:")
    print(f"   Durée totale: {sync_data['total_duration_seconds']}s")
    print(f"   Segments: {sync_data['segment_count']}")
    
    print(f"\n⏱️ Timeline:")
    for entry in sync_data["timeline"]:
        print(f"   {entry['start_time_seconds']}s - {entry['filename']} ({entry['duration_seconds']}s)")
    
    # Sauvegarder en JSON
    output_path = "nexus_video_export.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result.to_json())
    print(f"\n💾 Export complet sauvegardé: {output_path}")
    
    return result


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import sys
    
    examples = {
        "1": ("Génération simple", exemple_simple),
        "2": ("Config développeur", exemple_developpeur),
        "3": ("Config architecte", exemple_architecte),
        "4": ("Versions progressives", exemple_evolution),
        "5": ("Export vidéo", exemple_export_video),
    }
    
    print("\n" + "="*60)
    print("  NEXUS - Neural Execution & Understanding Synthesis")
    print("="*60)
    print("\nExemples disponibles:")
    
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nChoisir un exemple (1-5): ").strip()
    
    if choice in examples:
        name, func = examples[choice]
        print(f"\n🚀 Exécution: {name}")
        
        try:
            result = func()
            
            if result:
                save = input("\n💾 Sauvegarder le résultat JSON? (o/n): ").strip().lower()
                if save == 'o':
                    filename = f"nexus_output_{choice}.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(result.to_json())
                    print(f"✅ Sauvegardé dans {filename}")
                    
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Choix invalide")
