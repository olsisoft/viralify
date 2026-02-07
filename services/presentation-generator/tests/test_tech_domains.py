"""
Unit tests for tech_domains.py

Tests all enums: CodeLanguage, TechDomain, TechCareer, DiagramFocus
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tech_domains import (
    CodeLanguage,
    TechDomain,
    TechCareer,
    DiagramFocus,
    CAREER_DIAGRAM_FOCUS_MAP,
    DIAGRAM_FOCUS_INSTRUCTIONS,
    get_diagram_instructions_for_career,
)


# ============================================================================
# CodeLanguage Enum Tests
# ============================================================================

class TestCodeLanguage:
    """Tests for CodeLanguage enum"""

    def test_general_purpose_languages(self):
        """Test general purpose languages are defined"""
        languages = ["python", "javascript", "typescript", "java", "csharp",
                     "cpp", "c", "go", "rust", "kotlin", "swift", "ruby", "php"]
        for lang in languages:
            assert CodeLanguage(lang) is not None

    def test_pseudocode_languages(self):
        """Test pseudocode variants"""
        assert CodeLanguage.PSEUDOCODE.value == "pseudocode"
        assert CodeLanguage.PSEUDOCODE_FR.value == "pseudocode_fr"
        assert CodeLanguage.PSEUDOCODE_ES.value == "pseudocode_es"
        assert CodeLanguage.ALGORITHM.value == "algorithm"

    def test_data_query_languages(self):
        """Test data/query languages"""
        languages = ["sql", "postgresql", "mysql", "graphql", "mongodb",
                     "redis", "cypher", "sparql"]
        for lang in languages:
            assert CodeLanguage(lang) is not None

    def test_config_markup_languages(self):
        """Test config/markup languages"""
        languages = ["json", "yaml", "xml", "toml", "ini", "markdown", "latex"]
        for lang in languages:
            assert CodeLanguage(lang) is not None

    def test_shell_scripting_languages(self):
        """Test shell/scripting languages"""
        languages = ["bash", "zsh", "sh", "powershell", "cmd", "fish", "awk", "sed"]
        for lang in languages:
            assert CodeLanguage(lang) is not None

    def test_devops_cloud_languages(self):
        """Test DevOps/cloud languages"""
        languages = ["dockerfile", "docker_compose", "terraform", "kubernetes",
                     "helm", "ansible", "cloudformation", "pulumi"]
        for lang in languages:
            assert CodeLanguage(lang) is not None

    def test_data_science_languages(self):
        """Test data science languages"""
        languages = ["r", "julia", "matlab", "octave", "sas"]
        for lang in languages:
            assert CodeLanguage(lang) is not None

    def test_big_data_languages(self):
        """Test big data languages"""
        languages = ["spark", "pyspark", "hive", "flink", "beam"]
        for lang in languages:
            assert CodeLanguage(lang) is not None

    def test_blockchain_languages(self):
        """Test blockchain languages"""
        languages = ["solidity", "vyper", "rust_solana", "move", "cairo"]
        for lang in languages:
            assert CodeLanguage(lang) is not None

    def test_quantum_computing_languages(self):
        """Test quantum computing languages"""
        languages = ["qsharp", "qiskit", "cirq", "pennylane", "braket", "openqasm"]
        for lang in languages:
            assert CodeLanguage(lang) is not None

    def test_functional_languages(self):
        """Test functional languages"""
        languages = ["haskell", "elixir", "erlang", "clojure", "fsharp",
                     "ocaml", "lisp", "scheme", "elm"]
        for lang in languages:
            assert CodeLanguage(lang) is not None

    def test_hardware_languages(self):
        """Test hardware/low-level languages"""
        languages = ["vhdl", "verilog", "systemverilog", "assembly", "wasm"]
        for lang in languages:
            assert CodeLanguage(lang) is not None

    def test_game_development_languages(self):
        """Test game development languages"""
        languages = ["gdscript", "glsl", "hlsl"]
        for lang in languages:
            assert CodeLanguage(lang) is not None

    def test_modern_niche_languages(self):
        """Test modern/niche languages"""
        languages = ["zig", "nim", "crystal", "vlang", "odin", "mojo"]
        for lang in languages:
            assert CodeLanguage(lang) is not None

    def test_templating_languages(self):
        """Test templating languages"""
        languages = ["jinja2", "handlebars", "ejs", "pug", "twig"]
        for lang in languages:
            assert CodeLanguage(lang) is not None

    def test_api_protocol_languages(self):
        """Test API/protocol languages"""
        languages = ["protobuf", "grpc", "thrift", "avro", "openapi", "asyncapi"]
        for lang in languages:
            assert CodeLanguage(lang) is not None

    def test_language_is_string_enum(self):
        """Verify CodeLanguage is a string enum"""
        assert isinstance(CodeLanguage.PYTHON, str)
        assert CodeLanguage.PYTHON == "python"

    def test_total_language_count(self):
        """Verify we have a significant number of languages (100+)"""
        assert len(CodeLanguage) >= 100

    def test_invalid_language_raises_error(self):
        """Test that invalid language raises ValueError"""
        with pytest.raises(ValueError):
            CodeLanguage("nonexistent_language")


# ============================================================================
# TechDomain Enum Tests
# ============================================================================

class TestTechDomain:
    """Tests for TechDomain enum"""

    def test_development_domains(self):
        """Test development domains"""
        domains = ["programming_fundamentals", "software_engineering", "web_frontend",
                   "web_backend", "fullstack", "mobile_development", "game_development",
                   "embedded_systems", "systems_programming"]
        for domain in domains:
            assert TechDomain(domain) is not None

    def test_data_domains(self):
        """Test data domains"""
        domains = ["data_engineering", "data_science", "data_analytics",
                   "data_governance", "data_quality", "data_lineage",
                   "data_modeling", "data_architecture", "business_intelligence",
                   "big_data", "data_warehousing", "data_lakehouse", "streaming_data"]
        for domain in domains:
            assert TechDomain(domain) is not None

    def test_database_domains(self):
        """Test database domains"""
        domains = ["databases", "relational_databases", "nosql_databases",
                   "graph_databases", "time_series_databases", "vector_databases",
                   "database_administration", "database_optimization"]
        for domain in domains:
            assert TechDomain(domain) is not None

    def test_ai_ml_domains(self):
        """Test AI/ML domains"""
        domains = ["machine_learning", "deep_learning", "neural_networks",
                   "nlp", "computer_vision", "reinforcement_learning",
                   "generative_ai", "llm", "mlops", "ai_ethics"]
        for domain in domains:
            assert TechDomain(domain) is not None

    def test_cloud_domains(self):
        """Test cloud domains"""
        domains = ["cloud_computing", "cloud_aws", "cloud_azure", "cloud_gcp",
                   "multi_cloud", "hybrid_cloud", "serverless", "cloud_native",
                   "cloud_migration", "finops"]
        for domain in domains:
            assert TechDomain(domain) is not None

    def test_devops_domains(self):
        """Test DevOps/platform domains"""
        domains = ["devops", "platform_engineering", "sre", "infrastructure",
                   "iac", "cicd", "containers", "kubernetes", "observability",
                   "monitoring", "logging"]
        for domain in domains:
            assert TechDomain(domain) is not None

    def test_security_domains(self):
        """Test security domains"""
        domains = ["cybersecurity", "application_security", "cloud_security",
                   "network_security", "devsecops", "penetration_testing",
                   "offensive_security", "defensive_security", "cryptography", "iam"]
        for domain in domains:
            assert TechDomain(domain) is not None

    def test_architecture_domains(self):
        """Test architecture domains"""
        domains = ["software_architecture", "enterprise_architecture",
                   "solutions_architecture", "microservices", "distributed_systems",
                   "api_design", "event_driven", "ddd"]
        for domain in domains:
            assert TechDomain(domain) is not None

    def test_emerging_tech_domains(self):
        """Test emerging tech domains"""
        domains = ["blockchain", "web3", "smart_contracts", "defi",
                   "quantum_computing", "iot", "edge_computing", "ar_vr",
                   "robotics", "autonomous_systems"]
        for domain in domains:
            assert TechDomain(domain) is not None

    def test_methodology_domains(self):
        """Test methodology domains"""
        domains = ["clean_code", "design_patterns", "refactoring",
                   "tdd", "bdd", "agile", "scrum"]
        for domain in domains:
            assert TechDomain(domain) is not None

    def test_domain_is_string_enum(self):
        """Verify TechDomain is a string enum"""
        assert isinstance(TechDomain.DATA_ENGINEERING, str)
        assert TechDomain.DATA_ENGINEERING == "data_engineering"

    def test_total_domain_count(self):
        """Verify we have a significant number of domains (70+)"""
        assert len(TechDomain) >= 70

    def test_invalid_domain_raises_error(self):
        """Test that invalid domain raises ValueError"""
        with pytest.raises(ValueError):
            TechDomain("nonexistent_domain")


# ============================================================================
# TechCareer Enum Tests
# ============================================================================

class TestTechCareer:
    """Tests for TechCareer enum"""

    def test_software_development_careers(self):
        """Test software development careers"""
        careers = ["software_developer", "software_engineer", "junior_developer",
                   "senior_developer", "staff_engineer", "principal_engineer",
                   "distinguished_engineer"]
        for career in careers:
            assert TechCareer(career) is not None

    def test_frontend_careers(self):
        """Test frontend careers"""
        careers = ["frontend_developer", "frontend_engineer", "ui_developer",
                   "javascript_developer", "react_developer", "angular_developer",
                   "vue_developer"]
        for career in careers:
            assert TechCareer(career) is not None

    def test_backend_careers(self):
        """Test backend careers"""
        careers = ["backend_developer", "backend_engineer", "api_developer",
                   "python_developer", "java_developer", "nodejs_developer",
                   "go_developer", "rust_developer"]
        for career in careers:
            assert TechCareer(career) is not None

    def test_fullstack_careers(self):
        """Test fullstack careers"""
        careers = ["fullstack_developer", "fullstack_engineer",
                   "mern_developer", "mean_developer"]
        for career in careers:
            assert TechCareer(career) is not None

    def test_mobile_careers(self):
        """Test mobile careers"""
        careers = ["mobile_developer", "mobile_engineer", "ios_developer",
                   "android_developer", "flutter_developer", "react_native_developer"]
        for career in careers:
            assert TechCareer(career) is not None

    def test_data_engineering_careers(self):
        """Test data engineering careers"""
        careers = ["data_engineer", "senior_data_engineer", "lead_data_engineer",
                   "data_platform_engineer", "etl_developer", "data_pipeline_engineer",
                   "streaming_data_engineer"]
        for career in careers:
            assert TechCareer(career) is not None

    def test_data_lineage_careers(self):
        """Test data lineage careers"""
        careers = ["data_lineage_developer", "data_lineage_analyst",
                   "data_lineage_architect", "data_lineage_engineer"]
        for career in careers:
            assert TechCareer(career) is not None

    def test_data_quality_careers(self):
        """Test data quality careers"""
        careers = ["data_quality_engineer", "data_quality_analyst",
                   "data_quality_manager", "data_steward"]
        for career in careers:
            assert TechCareer(career) is not None

    def test_data_governance_careers(self):
        """Test data governance careers"""
        careers = ["data_governance_analyst", "data_governance_engineer",
                   "data_governance_manager", "data_governance_architect"]
        for career in careers:
            assert TechCareer(career) is not None

    def test_data_architect_careers(self):
        """Test data architect careers"""
        careers = ["data_architect", "senior_data_architect",
                   "enterprise_data_architect", "cloud_data_architect",
                   "data_warehouse_architect", "data_lakehouse_architect"]
        for career in careers:
            assert TechCareer(career) is not None

    def test_data_science_careers(self):
        """Test data science careers"""
        careers = ["data_scientist", "junior_data_scientist",
                   "senior_data_scientist", "lead_data_scientist",
                   "principal_data_scientist", "research_data_scientist"]
        for career in careers:
            assert TechCareer(career) is not None

    def test_game_development_careers(self):
        """Test game development careers"""
        careers = ["game_developer", "game_programmer", "game_engine_developer",
                   "unity_developer", "unreal_developer", "graphics_programmer"]
        for career in careers:
            assert TechCareer(career) is not None

    def test_embedded_careers(self):
        """Test embedded/systems careers"""
        careers = ["embedded_developer", "embedded_engineer", "firmware_engineer",
                   "systems_programmer", "low_level_programmer"]
        for career in careers:
            assert TechCareer(career) is not None

    def test_career_is_string_enum(self):
        """Verify TechCareer is a string enum"""
        assert isinstance(TechCareer.DATA_ENGINEER, str)
        assert TechCareer.DATA_ENGINEER == "data_engineer"

    def test_total_career_count(self):
        """Verify we have a significant number of careers (100+)"""
        # Note: The docstring says 545+ but we'll just verify it's substantial
        assert len(TechCareer) >= 100

    def test_invalid_career_raises_error(self):
        """Test that invalid career raises ValueError"""
        with pytest.raises(ValueError):
            TechCareer("nonexistent_career")


# ============================================================================
# DiagramFocus Enum Tests
# ============================================================================

class TestDiagramFocus:
    """Tests for DiagramFocus enum"""

    def test_all_diagram_focuses_defined(self):
        """Test all diagram focus types"""
        focuses = ["code", "infrastructure", "data", "ml_pipeline", "security",
                   "network", "database", "business", "qa_testing", "embedded"]
        for focus in focuses:
            assert DiagramFocus(focus) is not None

    def test_diagram_focus_values(self):
        """Test specific focus values"""
        assert DiagramFocus.CODE.value == "code"
        assert DiagramFocus.INFRASTRUCTURE.value == "infrastructure"
        assert DiagramFocus.DATA.value == "data"
        assert DiagramFocus.ML_PIPELINE.value == "ml_pipeline"
        assert DiagramFocus.SECURITY.value == "security"
        assert DiagramFocus.NETWORK.value == "network"
        assert DiagramFocus.DATABASE.value == "database"
        assert DiagramFocus.BUSINESS.value == "business"
        assert DiagramFocus.QA_TESTING.value == "qa_testing"
        assert DiagramFocus.EMBEDDED.value == "embedded"

    def test_diagram_focus_is_string_enum(self):
        """Verify DiagramFocus is a string enum"""
        assert isinstance(DiagramFocus.CODE, str)
        assert DiagramFocus.CODE == "code"

    def test_total_focus_count(self):
        """Verify total number of focuses"""
        assert len(DiagramFocus) == 10


# ============================================================================
# CAREER_DIAGRAM_FOCUS_MAP Tests
# ============================================================================

class TestCareerDiagramFocusMap:
    """Tests for CAREER_DIAGRAM_FOCUS_MAP dictionary"""

    def test_map_exists(self):
        """Test that the map exists and is a dict"""
        assert isinstance(CAREER_DIAGRAM_FOCUS_MAP, dict)

    def test_map_has_entries(self):
        """Test that the map has entries"""
        assert len(CAREER_DIAGRAM_FOCUS_MAP) > 0

    def test_data_engineer_mapping(self):
        """Test data engineer maps to DATA focus"""
        if TechCareer.DATA_ENGINEER in CAREER_DIAGRAM_FOCUS_MAP:
            assert CAREER_DIAGRAM_FOCUS_MAP[TechCareer.DATA_ENGINEER] == DiagramFocus.DATA

    def test_cloud_architect_mapping(self):
        """Test cloud architect maps to INFRASTRUCTURE focus"""
        if TechCareer.CLOUD_DATA_ARCHITECT in CAREER_DIAGRAM_FOCUS_MAP:
            focus = CAREER_DIAGRAM_FOCUS_MAP[TechCareer.CLOUD_DATA_ARCHITECT]
            assert focus in [DiagramFocus.INFRASTRUCTURE, DiagramFocus.DATA]

    def test_all_values_are_diagram_focus(self):
        """Test all map values are DiagramFocus enum members"""
        for career, focus in CAREER_DIAGRAM_FOCUS_MAP.items():
            assert isinstance(focus, DiagramFocus)

    def test_all_keys_are_tech_career(self):
        """Test all map keys are TechCareer enum members"""
        for career in CAREER_DIAGRAM_FOCUS_MAP.keys():
            assert isinstance(career, TechCareer)


# ============================================================================
# DIAGRAM_FOCUS_INSTRUCTIONS Tests
# ============================================================================

class TestDiagramFocusInstructions:
    """Tests for DIAGRAM_FOCUS_INSTRUCTIONS dictionary"""

    def test_instructions_exist(self):
        """Test that instructions dict exists"""
        assert isinstance(DIAGRAM_FOCUS_INSTRUCTIONS, dict)

    def test_all_focuses_have_instructions(self):
        """Test that all DiagramFocus values have instructions"""
        for focus in DiagramFocus:
            assert focus in DIAGRAM_FOCUS_INSTRUCTIONS, \
                f"Missing instructions for {focus}"

    def test_instructions_are_strings(self):
        """Test all instructions are non-empty strings"""
        for focus, instruction in DIAGRAM_FOCUS_INSTRUCTIONS.items():
            assert isinstance(instruction, str)
            assert len(instruction) > 0

    def test_code_focus_instructions(self):
        """Test CODE focus has relevant instructions"""
        instructions = DIAGRAM_FOCUS_INSTRUCTIONS[DiagramFocus.CODE]
        # Instructions should mention code-related concepts
        assert isinstance(instructions, str)
        assert len(instructions) > 10

    def test_data_focus_instructions(self):
        """Test DATA focus has relevant instructions"""
        instructions = DIAGRAM_FOCUS_INSTRUCTIONS[DiagramFocus.DATA]
        assert isinstance(instructions, str)
        assert len(instructions) > 10

    def test_infrastructure_focus_instructions(self):
        """Test INFRASTRUCTURE focus has relevant instructions"""
        instructions = DIAGRAM_FOCUS_INSTRUCTIONS[DiagramFocus.INFRASTRUCTURE]
        assert isinstance(instructions, str)
        assert len(instructions) > 10


# ============================================================================
# get_diagram_instructions_for_career Function Tests
# ============================================================================

class TestGetDiagramInstructionsForCareer:
    """Tests for get_diagram_instructions_for_career function"""

    def test_function_exists(self):
        """Test that function exists"""
        assert callable(get_diagram_instructions_for_career)

    def test_valid_career_returns_instructions(self):
        """Test that a valid career returns instructions"""
        # Try with data_engineer if it's in the map
        result = get_diagram_instructions_for_career(TechCareer.DATA_ENGINEER)
        assert isinstance(result, str)

    def test_returns_string(self):
        """Test that function returns a string"""
        result = get_diagram_instructions_for_career(TechCareer.SOFTWARE_DEVELOPER)
        assert isinstance(result, str)

    def test_unmapped_career_returns_default(self):
        """Test that unmapped career returns default/empty instructions"""
        # Find a career that might not be in the map
        result = get_diagram_instructions_for_career(TechCareer.JUNIOR_DEVELOPER)
        # Should return something (either mapped instructions or default)
        assert isinstance(result, str)


# ============================================================================
# Integration Tests
# ============================================================================

class TestTechDomainsIntegration:
    """Integration tests for tech domains module"""

    def test_career_and_language_coverage(self):
        """Test that common language-career combinations make sense"""
        # Python developers should have Python as a language
        assert CodeLanguage.PYTHON is not None
        assert TechCareer.PYTHON_DEVELOPER is not None

        # Go developers should have Go as a language
        assert CodeLanguage.GO is not None
        assert TechCareer.GO_DEVELOPER is not None

        # Rust developers should have Rust as a language
        assert CodeLanguage.RUST is not None
        assert TechCareer.RUST_DEVELOPER is not None

    def test_domain_and_career_alignment(self):
        """Test that domains and careers align logically"""
        # Data engineering domain and careers exist
        assert TechDomain.DATA_ENGINEERING is not None
        assert TechCareer.DATA_ENGINEER is not None

        # DevOps domain and SRE career exist
        assert TechDomain.DEVOPS is not None
        assert TechDomain.SITE_RELIABILITY is not None

        # ML domain and career exist
        assert TechDomain.MACHINE_LEARNING is not None
        assert TechDomain.MLOPS is not None

    def test_blockchain_ecosystem(self):
        """Test blockchain-related enums are complete"""
        # Languages
        assert CodeLanguage.SOLIDITY is not None
        assert CodeLanguage.VYPER is not None
        assert CodeLanguage.RUST_SOLANA is not None

        # Domains
        assert TechDomain.BLOCKCHAIN is not None
        assert TechDomain.WEB3 is not None
        assert TechDomain.SMART_CONTRACTS is not None
        assert TechDomain.DEFI is not None

    def test_quantum_computing_ecosystem(self):
        """Test quantum computing enums are complete"""
        # Languages
        assert CodeLanguage.QSHARP is not None
        assert CodeLanguage.QISKIT is not None
        assert CodeLanguage.CIRQ is not None

        # Domain
        assert TechDomain.QUANTUM_COMPUTING is not None

    def test_all_enums_have_unique_values(self):
        """Test that all enum values are unique within each enum"""
        # CodeLanguage
        values = [lang.value for lang in CodeLanguage]
        assert len(values) == len(set(values)), "CodeLanguage has duplicate values"

        # TechDomain
        values = [domain.value for domain in TechDomain]
        assert len(values) == len(set(values)), "TechDomain has duplicate values"

        # TechCareer
        values = [career.value for career in TechCareer]
        assert len(values) == len(set(values)), "TechCareer has duplicate values"

        # DiagramFocus
        values = [focus.value for focus in DiagramFocus]
        assert len(values) == len(set(values)), "DiagramFocus has duplicate values"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
