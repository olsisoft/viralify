"""
Tech Domains, Careers, and Languages - Complete 360° IT Ecosystem

This module provides comprehensive enums covering all IT domains,
career paths, and programming languages for contextual prompt generation.
"""
from enum import Enum
from typing import Dict, List, Optional


class CodeLanguage(str, Enum):
    """All supported programming languages and formats"""

    # === PSEUDO CODE & ALGORITHMS ===
    PSEUDOCODE = "pseudocode"
    PSEUDOCODE_FR = "pseudocode_fr"
    PSEUDOCODE_ES = "pseudocode_es"
    ALGORITHM = "algorithm"
    FLOWCHART_TEXT = "flowchart_text"

    # === GENERAL PURPOSE ===
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rust"
    KOTLIN = "kotlin"
    SWIFT = "swift"
    RUBY = "ruby"
    PHP = "php"
    SCALA = "scala"
    PERL = "perl"
    LUA = "lua"

    # === WEB ===
    HTML = "html"
    CSS = "css"
    SASS = "sass"
    SCSS = "scss"
    LESS = "less"

    # === DATA & QUERIES ===
    SQL = "sql"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    TSQL = "tsql"
    PLSQL = "plsql"
    GRAPHQL = "graphql"
    MONGODB = "mongodb"
    REDIS = "redis"
    CYPHER = "cypher"  # Neo4j
    SPARQL = "sparql"

    # === CONFIG & MARKUP ===
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    TOML = "toml"
    INI = "ini"
    MARKDOWN = "markdown"
    RST = "rst"  # ReStructuredText
    LATEX = "latex"

    # === SHELL & SCRIPTING ===
    BASH = "bash"
    ZSH = "zsh"
    SH = "sh"
    POWERSHELL = "powershell"
    CMD = "cmd"
    FISH = "fish"
    AWK = "awk"
    SED = "sed"

    # === DEVOPS & CLOUD ===
    DOCKERFILE = "dockerfile"
    DOCKER_COMPOSE = "docker_compose"
    TERRAFORM = "terraform"
    TERRAFORM_HCL = "hcl"
    KUBERNETES = "kubernetes"
    HELM = "helm"
    ANSIBLE = "ansible"
    PUPPET = "puppet"
    CHEF = "chef"
    SALTSTACK = "saltstack"
    CLOUDFORMATION = "cloudformation"
    PULUMI = "pulumi"
    BICEP = "bicep"  # Azure

    # === DATA SCIENCE & AI ===
    R = "r"
    JULIA = "julia"
    MATLAB = "matlab"
    OCTAVE = "octave"
    SAS = "sas"
    STATA = "stata"
    SPSS = "spss"

    # === BIG DATA & DISTRIBUTED ===
    SPARK = "spark"  # PySpark / Spark Scala
    PYSPARK = "pyspark"
    HADOOP = "hadoop"
    HIVE = "hive"
    PIG = "pig"
    FLINK = "flink"
    BEAM = "beam"  # Apache Beam

    # === BLOCKCHAIN ===
    SOLIDITY = "solidity"
    VYPER = "vyper"
    RUST_SOLANA = "rust_solana"
    MOVE = "move"  # Aptos/Sui
    CAIRO = "cairo"  # StarkNet

    # === HARDWARE & LOW LEVEL ===
    VHDL = "vhdl"
    VERILOG = "verilog"
    SYSTEMVERILOG = "systemverilog"
    ASSEMBLY = "assembly"
    ASSEMBLY_X86 = "assembly_x86"
    ASSEMBLY_ARM = "assembly_arm"
    ASSEMBLY_MIPS = "assembly_mips"
    WASM = "wasm"
    WAT = "wat"  # WebAssembly Text

    # === QUANTUM COMPUTING ===
    QSHARP = "qsharp"
    QISKIT = "qiskit"
    CIRQ = "cirq"
    PENNYLANE = "pennylane"
    BRAKET = "braket"  # AWS
    QUIL = "quil"  # Rigetti
    OPENQASM = "openqasm"

    # === FUNCTIONAL ===
    HASKELL = "haskell"
    ELIXIR = "elixir"
    ERLANG = "erlang"
    CLOJURE = "clojure"
    FSHARP = "fsharp"
    OCAML = "ocaml"
    LISP = "lisp"
    SCHEME = "scheme"
    RACKET = "racket"
    ELM = "elm"
    PURESCRIPT = "purescript"

    # === MOBILE ===
    DART = "dart"
    OBJECTIVE_C = "objectivec"
    REACT_NATIVE = "react_native"
    FLUTTER = "flutter"
    XAMARIN = "xamarin"

    # === GAME DEVELOPMENT ===
    GDSCRIPT = "gdscript"  # Godot
    UNREALSCRIPT = "unrealscript"
    UNITY_CSHARP = "unity_csharp"
    GLSL = "glsl"
    HLSL = "hlsl"

    # === SCIENTIFIC ===
    FORTRAN = "fortran"
    COBOL = "cobol"
    ADA = "ada"
    PASCAL = "pascal"

    # === MODERN/NICHE ===
    ZIG = "zig"
    NIM = "nim"
    CRYSTAL = "crystal"
    V = "vlang"
    ODIN = "odin"
    JAI = "jai"
    CARBON = "carbon"
    MOJO = "mojo"

    # === TEMPLATING ===
    JINJA2 = "jinja2"
    HANDLEBARS = "handlebars"
    EJS = "ejs"
    PUG = "pug"
    TWIG = "twig"

    # === API & PROTOCOLS ===
    PROTOBUF = "protobuf"
    GRPC = "grpc"
    THRIFT = "thrift"
    AVRO = "avro"
    OPENAPI = "openapi"
    ASYNCAPI = "asyncapi"


class TechDomain(str, Enum):
    """All IT/Tech domains for contextual expertise"""

    # === DEVELOPMENT ===
    PROGRAMMING_FUNDAMENTALS = "programming_fundamentals"
    SOFTWARE_ENGINEERING = "software_engineering"
    WEB_FRONTEND = "web_frontend"
    WEB_BACKEND = "web_backend"
    FULLSTACK = "fullstack"
    MOBILE_DEVELOPMENT = "mobile_development"
    DESKTOP_DEVELOPMENT = "desktop_development"
    GAME_DEVELOPMENT = "game_development"
    EMBEDDED_SYSTEMS = "embedded_systems"
    SYSTEMS_PROGRAMMING = "systems_programming"

    # === DATA ===
    DATA_ENGINEERING = "data_engineering"
    DATA_SCIENCE = "data_science"
    DATA_ANALYTICS = "data_analytics"
    DATA_GOVERNANCE = "data_governance"
    DATA_QUALITY = "data_quality"
    DATA_LINEAGE = "data_lineage"
    DATA_MODELING = "data_modeling"
    DATA_ARCHITECTURE = "data_architecture"
    DATA_INTEGRATION = "data_integration"
    DATA_CATALOG = "data_catalog"
    METADATA_MANAGEMENT = "metadata_management"
    MASTER_DATA_MANAGEMENT = "mdm"
    BUSINESS_INTELLIGENCE = "business_intelligence"
    ANALYTICS_ENGINEERING = "analytics_engineering"
    BIG_DATA = "big_data"
    DATA_WAREHOUSING = "data_warehousing"
    DATA_LAKEHOUSE = "data_lakehouse"
    STREAMING_DATA = "streaming_data"

    # === DATABASES ===
    DATABASES = "databases"
    RELATIONAL_DATABASES = "relational_databases"
    NOSQL_DATABASES = "nosql_databases"
    GRAPH_DATABASES = "graph_databases"
    TIME_SERIES_DATABASES = "time_series_databases"
    VECTOR_DATABASES = "vector_databases"
    DATABASE_ADMINISTRATION = "database_administration"
    DATABASE_OPTIMIZATION = "database_optimization"

    # === AI & ML ===
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    NEURAL_NETWORKS = "neural_networks"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GENERATIVE_AI = "generative_ai"
    LLM = "llm"
    MLOPS = "mlops"
    AI_ETHICS = "ai_ethics"
    AI_SAFETY = "ai_safety"
    RECOMMENDATION_SYSTEMS = "recommendation_systems"
    CONVERSATIONAL_AI = "conversational_ai"

    # === CLOUD ===
    CLOUD_COMPUTING = "cloud_computing"
    CLOUD_AWS = "cloud_aws"
    CLOUD_AZURE = "cloud_azure"
    CLOUD_GCP = "cloud_gcp"
    MULTI_CLOUD = "multi_cloud"
    HYBRID_CLOUD = "hybrid_cloud"
    SERVERLESS = "serverless"
    CLOUD_NATIVE = "cloud_native"
    CLOUD_MIGRATION = "cloud_migration"
    FINOPS = "finops"

    # === DEVOPS & PLATFORM ===
    DEVOPS = "devops"
    PLATFORM_ENGINEERING = "platform_engineering"
    SITE_RELIABILITY = "sre"
    INFRASTRUCTURE = "infrastructure"
    INFRASTRUCTURE_AS_CODE = "iac"
    CICD = "cicd"
    CONTAINERS = "containers"
    KUBERNETES = "kubernetes"
    OBSERVABILITY = "observability"
    MONITORING = "monitoring"
    LOGGING = "logging"

    # === SECURITY ===
    CYBERSECURITY = "cybersecurity"
    APPLICATION_SECURITY = "application_security"
    CLOUD_SECURITY = "cloud_security"
    NETWORK_SECURITY = "network_security"
    DEVSECOPS = "devsecops"
    PENETRATION_TESTING = "penetration_testing"
    OFFENSIVE_SECURITY = "offensive_security"
    DEFENSIVE_SECURITY = "defensive_security"
    THREAT_INTELLIGENCE = "threat_intelligence"
    INCIDENT_RESPONSE = "incident_response"
    DIGITAL_FORENSICS = "digital_forensics"
    CRYPTOGRAPHY = "cryptography"
    IAM = "iam"
    GRC = "grc"
    SECURE_CODING = "secure_coding"

    # === NETWORKING ===
    NETWORKING = "networking"
    NETWORK_ARCHITECTURE = "network_architecture"
    NETWORK_AUTOMATION = "network_automation"
    SDN = "sdn"
    SD_WAN = "sd_wan"
    WIRELESS = "wireless"

    # === SYSTEMS ===
    SYSTEM_ADMINISTRATION = "sysadmin"
    LINUX = "linux"
    WINDOWS_SERVER = "windows_server"
    UNIX = "unix"
    VIRTUALIZATION = "virtualization"
    STORAGE = "storage"

    # === ARCHITECTURE ===
    SOFTWARE_ARCHITECTURE = "software_architecture"
    ENTERPRISE_ARCHITECTURE = "enterprise_architecture"
    SOLUTIONS_ARCHITECTURE = "solutions_architecture"
    MICROSERVICES = "microservices"
    DISTRIBUTED_SYSTEMS = "distributed_systems"
    API_DESIGN = "api_design"
    EVENT_DRIVEN_ARCHITECTURE = "event_driven"
    DOMAIN_DRIVEN_DESIGN = "ddd"

    # === QA & TESTING ===
    SOFTWARE_TESTING = "software_testing"
    TEST_AUTOMATION = "test_automation"
    PERFORMANCE_TESTING = "performance_testing"
    SECURITY_TESTING = "security_testing"

    # === EMERGING TECH ===
    BLOCKCHAIN = "blockchain"
    WEB3 = "web3"
    SMART_CONTRACTS = "smart_contracts"
    DEFI = "defi"
    QUANTUM_COMPUTING = "quantum_computing"
    IOT = "iot"
    EDGE_COMPUTING = "edge_computing"
    AR_VR = "ar_vr"
    METAVERSE = "metaverse"
    ROBOTICS = "robotics"
    AUTONOMOUS_SYSTEMS = "autonomous_systems"

    # === METHODOLOGIES ===
    CLEAN_CODE = "clean_code"
    DESIGN_PATTERNS = "design_patterns"
    REFACTORING = "refactoring"
    TDD = "tdd"
    BDD = "bdd"
    AGILE = "agile"
    SCRUM = "scrum"


class TechCareer(str, Enum):
    """Complete 360° IT/Tech career mapping - 545+ roles"""

    # ═══════════════════════════════════════════════════════════════
    # SOFTWARE DEVELOPMENT
    # ═══════════════════════════════════════════════════════════════

    # --- General Development ---
    SOFTWARE_DEVELOPER = "software_developer"
    SOFTWARE_ENGINEER = "software_engineer"
    JUNIOR_DEVELOPER = "junior_developer"
    MID_LEVEL_DEVELOPER = "mid_level_developer"
    SENIOR_DEVELOPER = "senior_developer"
    STAFF_ENGINEER = "staff_engineer"
    PRINCIPAL_ENGINEER = "principal_engineer"
    DISTINGUISHED_ENGINEER = "distinguished_engineer"
    FELLOW_ENGINEER = "fellow_engineer"

    # --- Frontend ---
    FRONTEND_DEVELOPER = "frontend_developer"
    FRONTEND_ENGINEER = "frontend_engineer"
    UI_DEVELOPER = "ui_developer"
    UI_ENGINEER = "ui_engineer"
    JAVASCRIPT_DEVELOPER = "javascript_developer"
    REACT_DEVELOPER = "react_developer"
    ANGULAR_DEVELOPER = "angular_developer"
    VUE_DEVELOPER = "vue_developer"
    SVELTE_DEVELOPER = "svelte_developer"

    # --- Backend ---
    BACKEND_DEVELOPER = "backend_developer"
    BACKEND_ENGINEER = "backend_engineer"
    API_DEVELOPER = "api_developer"
    PYTHON_DEVELOPER = "python_developer"
    JAVA_DEVELOPER = "java_developer"
    NODEJS_DEVELOPER = "nodejs_developer"
    GO_DEVELOPER = "go_developer"
    RUST_DEVELOPER = "rust_developer"
    DOTNET_DEVELOPER = "dotnet_developer"
    PHP_DEVELOPER = "php_developer"
    RUBY_DEVELOPER = "ruby_developer"

    # --- Fullstack ---
    FULLSTACK_DEVELOPER = "fullstack_developer"
    FULLSTACK_ENGINEER = "fullstack_engineer"
    MERN_DEVELOPER = "mern_developer"
    MEAN_DEVELOPER = "mean_developer"

    # --- Mobile ---
    MOBILE_DEVELOPER = "mobile_developer"
    MOBILE_ENGINEER = "mobile_engineer"
    IOS_DEVELOPER = "ios_developer"
    ANDROID_DEVELOPER = "android_developer"
    FLUTTER_DEVELOPER = "flutter_developer"
    REACT_NATIVE_DEVELOPER = "react_native_developer"
    SWIFT_DEVELOPER = "swift_developer"
    KOTLIN_DEVELOPER = "kotlin_developer"

    # --- Desktop & Systems ---
    DESKTOP_DEVELOPER = "desktop_developer"
    SYSTEMS_PROGRAMMER = "systems_programmer"
    EMBEDDED_DEVELOPER = "embedded_developer"
    EMBEDDED_ENGINEER = "embedded_engineer"
    FIRMWARE_ENGINEER = "firmware_engineer"
    LOW_LEVEL_PROGRAMMER = "low_level_programmer"

    # --- Game Development ---
    GAME_DEVELOPER = "game_developer"
    GAME_PROGRAMMER = "game_programmer"
    GAME_ENGINE_DEVELOPER = "game_engine_developer"
    UNITY_DEVELOPER = "unity_developer"
    UNREAL_DEVELOPER = "unreal_developer"
    GRAPHICS_PROGRAMMER = "graphics_programmer"
    GAMEPLAY_PROGRAMMER = "gameplay_programmer"

    # ═══════════════════════════════════════════════════════════════
    # DATA - COMPLETE ECOSYSTEM
    # ═══════════════════════════════════════════════════════════════

    # --- Data Engineering ---
    DATA_ENGINEER = "data_engineer"
    SENIOR_DATA_ENGINEER = "senior_data_engineer"
    LEAD_DATA_ENGINEER = "lead_data_engineer"
    PRINCIPAL_DATA_ENGINEER = "principal_data_engineer"
    DATA_PLATFORM_ENGINEER = "data_platform_engineer"
    ETL_DEVELOPER = "etl_developer"
    ELT_DEVELOPER = "elt_developer"
    DATA_PIPELINE_ENGINEER = "data_pipeline_engineer"
    STREAMING_DATA_ENGINEER = "streaming_data_engineer"
    BATCH_DATA_ENGINEER = "batch_data_engineer"

    # --- Data Enablement ---
    DATA_ENABLER = "data_enabler"
    DATA_ENABLEMENT_LEAD = "data_enablement_lead"
    DATA_ENABLEMENT_MANAGER = "data_enablement_manager"
    DATA_DEMOCRATIZATION_LEAD = "data_democratization_lead"

    # --- Data Lineage ---
    DATA_LINEAGE_DEVELOPER = "data_lineage_developer"
    DATA_LINEAGE_ANALYST = "data_lineage_analyst"
    DATA_LINEAGE_ARCHITECT = "data_lineage_architect"
    DATA_LINEAGE_ENGINEER = "data_lineage_engineer"
    DATA_LINEAGE_SPECIALIST = "data_lineage_specialist"

    # --- Data Quality ---
    DATA_QUALITY_ENGINEER = "data_quality_engineer"
    DATA_QUALITY_ANALYST = "data_quality_analyst"
    DATA_QUALITY_MANAGER = "data_quality_manager"
    DATA_QUALITY_ARCHITECT = "data_quality_architect"
    DATA_STEWARD = "data_steward"
    DATA_CUSTODIAN = "data_custodian"

    # --- Data Governance ---
    DATA_GOVERNANCE_ANALYST = "data_governance_analyst"
    DATA_GOVERNANCE_ENGINEER = "data_governance_engineer"
    DATA_GOVERNANCE_MANAGER = "data_governance_manager"
    DATA_GOVERNANCE_ARCHITECT = "data_governance_architect"
    DATA_GOVERNANCE_LEAD = "data_governance_lead"
    DATA_GOVERNANCE_OFFICER = "data_governance_officer"
    CHIEF_DATA_GOVERNANCE_OFFICER = "chief_data_governance_officer"

    # --- Data Architecture ---
    DATA_ARCHITECT = "data_architect"
    SENIOR_DATA_ARCHITECT = "senior_data_architect"
    ENTERPRISE_DATA_ARCHITECT = "enterprise_data_architect"
    CLOUD_DATA_ARCHITECT = "cloud_data_architect"
    DATA_MODELING_ARCHITECT = "data_modeling_architect"
    DATA_WAREHOUSE_ARCHITECT = "data_warehouse_architect"
    DATA_LAKEHOUSE_ARCHITECT = "data_lakehouse_architect"

    # --- Data Modeling ---
    DATA_MODELER = "data_modeler"
    SENIOR_DATA_MODELER = "senior_data_modeler"
    DIMENSIONAL_MODELER = "dimensional_modeler"
    CONCEPTUAL_DATA_MODELER = "conceptual_data_modeler"
    LOGICAL_DATA_MODELER = "logical_data_modeler"
    PHYSICAL_DATA_MODELER = "physical_data_modeler"

    # --- Data Science ---
    DATA_SCIENTIST = "data_scientist"
    JUNIOR_DATA_SCIENTIST = "junior_data_scientist"
    SENIOR_DATA_SCIENTIST = "senior_data_scientist"
    LEAD_DATA_SCIENTIST = "lead_data_scientist"
    PRINCIPAL_DATA_SCIENTIST = "principal_data_scientist"
    STAFF_DATA_SCIENTIST = "staff_data_scientist"
    RESEARCH_DATA_SCIENTIST = "research_data_scientist"
    APPLIED_DATA_SCIENTIST = "applied_data_scientist"

    # --- Data Analytics ---
    DATA_ANALYST = "data_analyst"
    JUNIOR_DATA_ANALYST = "junior_data_analyst"
    SENIOR_DATA_ANALYST = "senior_data_analyst"
    LEAD_DATA_ANALYST = "lead_data_analyst"
    BUSINESS_DATA_ANALYST = "business_data_analyst"
    MARKETING_DATA_ANALYST = "marketing_data_analyst"
    FINANCIAL_DATA_ANALYST = "financial_data_analyst"
    PRODUCT_DATA_ANALYST = "product_data_analyst"
    OPERATIONS_DATA_ANALYST = "operations_data_analyst"
    HEALTHCARE_DATA_ANALYST = "healthcare_data_analyst"

    # --- Business Intelligence ---
    BI_DEVELOPER = "bi_developer"
    BI_ANALYST = "bi_analyst"
    BI_ENGINEER = "bi_engineer"
    BI_ARCHITECT = "bi_architect"
    BI_CONSULTANT = "bi_consultant"
    BI_MANAGER = "bi_manager"
    TABLEAU_DEVELOPER = "tableau_developer"
    POWER_BI_DEVELOPER = "power_bi_developer"
    LOOKER_DEVELOPER = "looker_developer"
    QLIK_DEVELOPER = "qlik_developer"

    # --- Analytics Engineering ---
    ANALYTICS_ENGINEER = "analytics_engineer"
    SENIOR_ANALYTICS_ENGINEER = "senior_analytics_engineer"
    LEAD_ANALYTICS_ENGINEER = "lead_analytics_engineer"
    DBT_DEVELOPER = "dbt_developer"

    # --- Data Operations ---
    DATAOPS_ENGINEER = "dataops_engineer"
    DATAOPS_ARCHITECT = "dataops_architect"
    DATAOPS_MANAGER = "dataops_manager"

    # --- Data Management ---
    DATA_MANAGER = "data_manager"
    DATA_MANAGEMENT_SPECIALIST = "data_management_specialist"
    MDM_SPECIALIST = "mdm_specialist"
    MDM_ARCHITECT = "mdm_architect"
    MDM_DEVELOPER = "mdm_developer"

    # --- Data Integration ---
    DATA_INTEGRATION_ENGINEER = "data_integration_engineer"
    DATA_INTEGRATION_ARCHITECT = "data_integration_architect"
    DATA_INTEGRATION_SPECIALIST = "data_integration_specialist"

    # --- Data Catalog ---
    DATA_CATALOG_ENGINEER = "data_catalog_engineer"
    DATA_CATALOG_ARCHITECT = "data_catalog_architect"
    DATA_CATALOG_ANALYST = "data_catalog_analyst"
    METADATA_ENGINEER = "metadata_engineer"
    METADATA_ANALYST = "metadata_analyst"
    METADATA_ARCHITECT = "metadata_architect"

    # --- Big Data ---
    BIG_DATA_ENGINEER = "big_data_engineer"
    BIG_DATA_ARCHITECT = "big_data_architect"
    BIG_DATA_DEVELOPER = "big_data_developer"
    HADOOP_DEVELOPER = "hadoop_developer"
    SPARK_DEVELOPER = "spark_developer"
    KAFKA_DEVELOPER = "kafka_developer"
    FLINK_DEVELOPER = "flink_developer"

    # --- Data Leadership ---
    HEAD_OF_DATA = "head_of_data"
    VP_DATA = "vp_data"
    DIRECTOR_DATA_ENGINEERING = "director_data_engineering"
    DIRECTOR_DATA_SCIENCE = "director_data_science"
    CHIEF_DATA_OFFICER = "chief_data_officer"
    CHIEF_ANALYTICS_OFFICER = "chief_analytics_officer"

    # ═══════════════════════════════════════════════════════════════
    # AI / MACHINE LEARNING
    # ═══════════════════════════════════════════════════════════════

    # --- Machine Learning Engineering ---
    ML_ENGINEER = "ml_engineer"
    JUNIOR_ML_ENGINEER = "junior_ml_engineer"
    SENIOR_ML_ENGINEER = "senior_ml_engineer"
    LEAD_ML_ENGINEER = "lead_ml_engineer"
    PRINCIPAL_ML_ENGINEER = "principal_ml_engineer"
    STAFF_ML_ENGINEER = "staff_ml_engineer"

    # --- ML Specializations ---
    ML_PLATFORM_ENGINEER = "ml_platform_engineer"
    ML_INFRASTRUCTURE_ENGINEER = "ml_infrastructure_engineer"
    ML_SYSTEMS_ENGINEER = "ml_systems_engineer"
    FEATURE_ENGINEER = "feature_engineer"
    FEATURE_STORE_ENGINEER = "feature_store_engineer"

    # --- MLOps ---
    MLOPS_ENGINEER = "mlops_engineer"
    SENIOR_MLOPS_ENGINEER = "senior_mlops_engineer"
    LEAD_MLOPS_ENGINEER = "lead_mlops_engineer"
    MLOPS_ARCHITECT = "mlops_architect"
    MLOPS_PLATFORM_ENGINEER = "mlops_platform_engineer"

    # --- Deep Learning ---
    DEEP_LEARNING_ENGINEER = "deep_learning_engineer"
    DEEP_LEARNING_RESEARCHER = "deep_learning_researcher"
    NEURAL_NETWORK_ENGINEER = "neural_network_engineer"

    # --- AI Research ---
    AI_RESEARCHER = "ai_researcher"
    AI_RESEARCH_SCIENTIST = "ai_research_scientist"
    AI_RESEARCH_ENGINEER = "ai_research_engineer"
    APPLIED_AI_RESEARCHER = "applied_ai_researcher"

    # --- AI Engineering ---
    AI_ENGINEER = "ai_engineer"
    AI_DEVELOPER = "ai_developer"
    AI_SOLUTIONS_ENGINEER = "ai_solutions_engineer"
    AI_PLATFORM_ENGINEER = "ai_platform_engineer"
    AI_INFRASTRUCTURE_ENGINEER = "ai_infrastructure_engineer"

    # --- NLP ---
    NLP_ENGINEER = "nlp_engineer"
    NLP_SCIENTIST = "nlp_scientist"
    NLP_RESEARCHER = "nlp_researcher"
    COMPUTATIONAL_LINGUIST = "computational_linguist"
    CONVERSATIONAL_AI_ENGINEER = "conversational_ai_engineer"
    CHATBOT_DEVELOPER = "chatbot_developer"

    # --- Computer Vision ---
    COMPUTER_VISION_ENGINEER = "computer_vision_engineer"
    COMPUTER_VISION_SCIENTIST = "computer_vision_scientist"
    COMPUTER_VISION_RESEARCHER = "computer_vision_researcher"
    IMAGE_PROCESSING_ENGINEER = "image_processing_engineer"
    VIDEO_ANALYTICS_ENGINEER = "video_analytics_engineer"

    # --- Generative AI ---
    GENERATIVE_AI_ENGINEER = "generative_ai_engineer"
    LLM_ENGINEER = "llm_engineer"
    PROMPT_ENGINEER = "prompt_engineer"
    AI_CONTENT_ENGINEER = "ai_content_engineer"
    DIFFUSION_MODEL_ENGINEER = "diffusion_model_engineer"

    # --- Recommendation Systems ---
    RECOMMENDATION_ENGINEER = "recommendation_engineer"
    PERSONALIZATION_ENGINEER = "personalization_engineer"
    SEARCH_RELEVANCE_ENGINEER = "search_relevance_engineer"

    # --- Reinforcement Learning ---
    RL_ENGINEER = "rl_engineer"
    RL_RESEARCHER = "rl_researcher"
    ROBOTICS_ML_ENGINEER = "robotics_ml_engineer"

    # --- AI Ethics & Safety ---
    AI_ETHICS_RESEARCHER = "ai_ethics_researcher"
    AI_SAFETY_ENGINEER = "ai_safety_engineer"
    RESPONSIBLE_AI_ENGINEER = "responsible_ai_engineer"
    AI_FAIRNESS_ENGINEER = "ai_fairness_engineer"
    AI_BIAS_ANALYST = "ai_bias_analyst"

    # --- AI Product ---
    AI_PRODUCT_MANAGER = "ai_product_manager"
    ML_PRODUCT_MANAGER = "ml_product_manager"
    AI_SOLUTIONS_ARCHITECT = "ai_solutions_architect"

    # --- AI Leadership ---
    HEAD_OF_AI = "head_of_ai"
    HEAD_OF_ML = "head_of_ml"
    VP_AI = "vp_ai"
    DIRECTOR_AI = "director_ai"
    DIRECTOR_ML = "director_ml"
    CHIEF_AI_OFFICER = "chief_ai_officer"

    # ═══════════════════════════════════════════════════════════════
    # DEVOPS / PLATFORM / SRE
    # ═══════════════════════════════════════════════════════════════

    # --- DevOps ---
    DEVOPS_ENGINEER = "devops_engineer"
    JUNIOR_DEVOPS_ENGINEER = "junior_devops_engineer"
    SENIOR_DEVOPS_ENGINEER = "senior_devops_engineer"
    LEAD_DEVOPS_ENGINEER = "lead_devops_engineer"
    PRINCIPAL_DEVOPS_ENGINEER = "principal_devops_engineer"
    STAFF_DEVOPS_ENGINEER = "staff_devops_engineer"
    DEVOPS_ARCHITECT = "devops_architect"
    DEVOPS_CONSULTANT = "devops_consultant"
    DEVOPS_MANAGER = "devops_manager"

    # --- Platform Engineering ---
    PLATFORM_ENGINEER = "platform_engineer"
    SENIOR_PLATFORM_ENGINEER = "senior_platform_engineer"
    LEAD_PLATFORM_ENGINEER = "lead_platform_engineer"
    PRINCIPAL_PLATFORM_ENGINEER = "principal_platform_engineer"
    PLATFORM_ARCHITECT = "platform_architect"
    INTERNAL_PLATFORM_ENGINEER = "internal_platform_engineer"
    DEVELOPER_PLATFORM_ENGINEER = "developer_platform_engineer"

    # --- Site Reliability Engineering ---
    SRE = "sre"
    JUNIOR_SRE = "junior_sre"
    SENIOR_SRE = "senior_sre"
    LEAD_SRE = "lead_sre"
    PRINCIPAL_SRE = "principal_sre"
    STAFF_SRE = "staff_sre"
    SRE_MANAGER = "sre_manager"
    SRE_ARCHITECT = "sre_architect"

    # --- Infrastructure ---
    INFRASTRUCTURE_ENGINEER = "infrastructure_engineer"
    INFRASTRUCTURE_DEVELOPER = "infrastructure_developer"
    INFRASTRUCTURE_ARCHITECT = "infrastructure_architect"
    IAC_ENGINEER = "iac_engineer"

    # --- Release Engineering ---
    RELEASE_ENGINEER = "release_engineer"
    RELEASE_MANAGER = "release_manager"
    BUILD_ENGINEER = "build_engineer"
    BUILD_RELEASE_ENGINEER = "build_release_engineer"

    # --- CI/CD ---
    CICD_ENGINEER = "cicd_engineer"
    PIPELINE_ENGINEER = "pipeline_engineer"
    AUTOMATION_ENGINEER = "automation_engineer"
    DEPLOYMENT_ENGINEER = "deployment_engineer"

    # --- Containers & Orchestration ---
    KUBERNETES_ENGINEER = "kubernetes_engineer"
    KUBERNETES_ADMINISTRATOR = "kubernetes_administrator"
    KUBERNETES_ARCHITECT = "kubernetes_architect"
    CONTAINER_ENGINEER = "container_engineer"
    DOCKER_ENGINEER = "docker_engineer"
    OPENSHIFT_ENGINEER = "openshift_engineer"

    # --- Monitoring & Observability ---
    OBSERVABILITY_ENGINEER = "observability_engineer"
    MONITORING_ENGINEER = "monitoring_engineer"
    LOGGING_ENGINEER = "logging_engineer"
    APM_ENGINEER = "apm_engineer"

    # --- Configuration Management ---
    CONFIGURATION_MANAGER = "configuration_manager"
    ANSIBLE_ENGINEER = "ansible_engineer"
    PUPPET_ENGINEER = "puppet_engineer"
    CHEF_ENGINEER = "chef_engineer"
    SALTSTACK_ENGINEER = "saltstack_engineer"

    # --- DevOps Leadership ---
    HEAD_OF_DEVOPS = "head_of_devops"
    HEAD_OF_PLATFORM = "head_of_platform"
    HEAD_OF_SRE = "head_of_sre"
    DIRECTOR_DEVOPS = "director_devops"
    DIRECTOR_PLATFORM = "director_platform"
    DIRECTOR_SRE = "director_sre"

    # ═══════════════════════════════════════════════════════════════
    # CLOUD
    # ═══════════════════════════════════════════════════════════════

    # --- General Cloud ---
    CLOUD_ENGINEER = "cloud_engineer"
    JUNIOR_CLOUD_ENGINEER = "junior_cloud_engineer"
    SENIOR_CLOUD_ENGINEER = "senior_cloud_engineer"
    LEAD_CLOUD_ENGINEER = "lead_cloud_engineer"
    PRINCIPAL_CLOUD_ENGINEER = "principal_cloud_engineer"
    CLOUD_ARCHITECT = "cloud_architect"
    SENIOR_CLOUD_ARCHITECT = "senior_cloud_architect"
    ENTERPRISE_CLOUD_ARCHITECT = "enterprise_cloud_architect"
    CLOUD_CONSULTANT = "cloud_consultant"
    CLOUD_SOLUTIONS_ARCHITECT = "cloud_solutions_architect"

    # --- AWS ---
    AWS_ENGINEER = "aws_engineer"
    AWS_DEVELOPER = "aws_developer"
    AWS_ARCHITECT = "aws_architect"
    AWS_SOLUTIONS_ARCHITECT = "aws_solutions_architect"
    AWS_DEVOPS_ENGINEER = "aws_devops_engineer"
    AWS_SYSOPS_ADMINISTRATOR = "aws_sysops_administrator"
    AWS_DATA_ENGINEER = "aws_data_engineer"
    AWS_ML_SPECIALIST = "aws_ml_specialist"
    AWS_SECURITY_SPECIALIST = "aws_security_specialist"
    AWS_NETWORKING_SPECIALIST = "aws_networking_specialist"

    # --- Azure ---
    AZURE_ENGINEER = "azure_engineer"
    AZURE_DEVELOPER = "azure_developer"
    AZURE_ARCHITECT = "azure_architect"
    AZURE_SOLUTIONS_ARCHITECT = "azure_solutions_architect"
    AZURE_DEVOPS_ENGINEER = "azure_devops_engineer"
    AZURE_ADMINISTRATOR = "azure_administrator"
    AZURE_DATA_ENGINEER = "azure_data_engineer"
    AZURE_AI_ENGINEER = "azure_ai_engineer"
    AZURE_SECURITY_ENGINEER = "azure_security_engineer"

    # --- GCP ---
    GCP_ENGINEER = "gcp_engineer"
    GCP_DEVELOPER = "gcp_developer"
    GCP_ARCHITECT = "gcp_architect"
    GCP_CLOUD_ARCHITECT = "gcp_cloud_architect"
    GCP_DEVOPS_ENGINEER = "gcp_devops_engineer"
    GCP_DATA_ENGINEER = "gcp_data_engineer"
    GCP_ML_ENGINEER = "gcp_ml_engineer"
    GCP_SECURITY_ENGINEER = "gcp_security_engineer"

    # --- Multi-Cloud ---
    MULTI_CLOUD_ARCHITECT = "multi_cloud_architect"
    MULTI_CLOUD_ENGINEER = "multi_cloud_engineer"
    HYBRID_CLOUD_ARCHITECT = "hybrid_cloud_architect"
    HYBRID_CLOUD_ENGINEER = "hybrid_cloud_engineer"

    # --- Serverless ---
    SERVERLESS_ENGINEER = "serverless_engineer"
    SERVERLESS_ARCHITECT = "serverless_architect"
    LAMBDA_DEVELOPER = "lambda_developer"
    FUNCTIONS_DEVELOPER = "functions_developer"

    # --- Cloud Native ---
    CLOUD_NATIVE_ENGINEER = "cloud_native_engineer"
    CLOUD_NATIVE_ARCHITECT = "cloud_native_architect"
    CLOUD_NATIVE_DEVELOPER = "cloud_native_developer"

    # --- Cloud FinOps ---
    FINOPS_ENGINEER = "finops_engineer"
    FINOPS_ANALYST = "finops_analyst"
    FINOPS_ARCHITECT = "finops_architect"
    CLOUD_COST_ANALYST = "cloud_cost_analyst"
    CLOUD_ECONOMIST = "cloud_economist"

    # --- Cloud Migration ---
    CLOUD_MIGRATION_ENGINEER = "cloud_migration_engineer"
    CLOUD_MIGRATION_ARCHITECT = "cloud_migration_architect"
    CLOUD_MIGRATION_SPECIALIST = "cloud_migration_specialist"

    # --- Cloud Leadership ---
    HEAD_OF_CLOUD = "head_of_cloud"
    VP_CLOUD = "vp_cloud"
    DIRECTOR_CLOUD = "director_cloud"
    CLOUD_PRACTICE_LEAD = "cloud_practice_lead"

    # ═══════════════════════════════════════════════════════════════
    # SECURITY / CYBERSECURITY
    # ═══════════════════════════════════════════════════════════════

    # --- Security Engineering ---
    SECURITY_ENGINEER = "security_engineer"
    JUNIOR_SECURITY_ENGINEER = "junior_security_engineer"
    SENIOR_SECURITY_ENGINEER = "senior_security_engineer"
    LEAD_SECURITY_ENGINEER = "lead_security_engineer"
    PRINCIPAL_SECURITY_ENGINEER = "principal_security_engineer"
    STAFF_SECURITY_ENGINEER = "staff_security_engineer"

    # --- Security Architecture ---
    SECURITY_ARCHITECT = "security_architect"
    SENIOR_SECURITY_ARCHITECT = "senior_security_architect"
    ENTERPRISE_SECURITY_ARCHITECT = "enterprise_security_architect"
    CLOUD_SECURITY_ARCHITECT = "cloud_security_architect"
    APPLICATION_SECURITY_ARCHITECT = "application_security_architect"

    # --- Application Security ---
    APPSEC_ENGINEER = "appsec_engineer"
    APPSEC_ANALYST = "appsec_analyst"
    SECURE_CODE_REVIEWER = "secure_code_reviewer"
    SAST_ENGINEER = "sast_engineer"
    DAST_ENGINEER = "dast_engineer"

    # --- DevSecOps ---
    DEVSECOPS_ENGINEER = "devsecops_engineer"
    SENIOR_DEVSECOPS_ENGINEER = "senior_devsecops_engineer"
    DEVSECOPS_ARCHITECT = "devsecops_architect"
    SECURITY_AUTOMATION_ENGINEER = "security_automation_engineer"

    # --- Penetration Testing ---
    PENETRATION_TESTER = "penetration_tester"
    JUNIOR_PENTESTER = "junior_pentester"
    SENIOR_PENTESTER = "senior_pentester"
    LEAD_PENTESTER = "lead_pentester"
    RED_TEAM_OPERATOR = "red_team_operator"
    RED_TEAM_LEAD = "red_team_lead"

    # --- Offensive Security ---
    OFFENSIVE_SECURITY_ENGINEER = "offensive_security_engineer"
    EXPLOIT_DEVELOPER = "exploit_developer"
    VULNERABILITY_RESEARCHER = "vulnerability_researcher"
    BUG_BOUNTY_HUNTER = "bug_bounty_hunter"

    # --- Defensive Security ---
    DEFENSIVE_SECURITY_ENGINEER = "defensive_security_engineer"
    BLUE_TEAM_ANALYST = "blue_team_analyst"
    BLUE_TEAM_ENGINEER = "blue_team_engineer"
    PURPLE_TEAM_ENGINEER = "purple_team_engineer"

    # --- Security Operations ---
    SOC_ANALYST = "soc_analyst"
    SOC_ANALYST_L1 = "soc_analyst_l1"
    SOC_ANALYST_L2 = "soc_analyst_l2"
    SOC_ANALYST_L3 = "soc_analyst_l3"
    SOC_ENGINEER = "soc_engineer"
    SOC_MANAGER = "soc_manager"

    # --- Incident Response ---
    INCIDENT_RESPONDER = "incident_responder"
    IR_ANALYST = "ir_analyst"
    IR_ENGINEER = "ir_engineer"
    IR_MANAGER = "ir_manager"

    # --- Threat Intelligence ---
    THREAT_INTEL_ANALYST = "threat_intel_analyst"
    THREAT_INTEL_ENGINEER = "threat_intel_engineer"
    THREAT_HUNTER = "threat_hunter"
    THREAT_RESEARCHER = "threat_researcher"

    # --- Digital Forensics ---
    FORENSICS_ANALYST = "forensics_analyst"
    FORENSICS_ENGINEER = "forensics_engineer"
    MALWARE_ANALYST = "malware_analyst"
    MALWARE_REVERSE_ENGINEER = "malware_reverse_engineer"

    # --- Cryptography ---
    CRYPTOGRAPHER = "cryptographer"
    CRYPTOGRAPHY_ENGINEER = "cryptography_engineer"
    PKI_ENGINEER = "pki_engineer"

    # --- Identity & Access ---
    IAM_ENGINEER = "iam_engineer"
    IAM_ARCHITECT = "iam_architect"
    IAM_ANALYST = "iam_analyst"
    IDENTITY_ENGINEER = "identity_engineer"
    ACCESS_MANAGEMENT_ENGINEER = "access_management_engineer"

    # --- Network Security ---
    NETWORK_SECURITY_ENGINEER = "network_security_engineer"
    FIREWALL_ENGINEER = "firewall_engineer"
    FIREWALL_ADMINISTRATOR = "firewall_administrator"

    # --- GRC ---
    GRC_ANALYST = "grc_analyst"
    GRC_ENGINEER = "grc_engineer"
    GRC_MANAGER = "grc_manager"
    COMPLIANCE_ANALYST = "compliance_analyst"
    COMPLIANCE_ENGINEER = "compliance_engineer"
    RISK_ANALYST = "risk_analyst"
    IT_AUDITOR = "it_auditor"

    # --- Security Leadership ---
    HEAD_OF_SECURITY = "head_of_security"
    VP_SECURITY = "vp_security"
    DIRECTOR_SECURITY = "director_security"
    CISO = "ciso"
    DEPUTY_CISO = "deputy_ciso"

    # ═══════════════════════════════════════════════════════════════
    # DATABASES
    # ═══════════════════════════════════════════════════════════════

    DBA = "dba"
    JUNIOR_DBA = "junior_dba"
    SENIOR_DBA = "senior_dba"
    LEAD_DBA = "lead_dba"
    PRINCIPAL_DBA = "principal_dba"
    POSTGRESQL_DBA = "postgresql_dba"
    MYSQL_DBA = "mysql_dba"
    ORACLE_DBA = "oracle_dba"
    SQL_SERVER_DBA = "sql_server_dba"
    MONGODB_DBA = "mongodb_dba"
    CASSANDRA_DBA = "cassandra_dba"
    REDIS_ENGINEER = "redis_engineer"
    ELASTICSEARCH_ENGINEER = "elasticsearch_engineer"
    DATABASE_DEVELOPER = "database_developer"
    SQL_DEVELOPER = "sql_developer"
    PLSQL_DEVELOPER = "plsql_developer"
    TSQL_DEVELOPER = "tsql_developer"
    DATABASE_ARCHITECT = "database_architect"
    DATABASE_SOLUTIONS_ARCHITECT = "database_solutions_architect"
    DBRE = "dbre"
    DATABASE_SRE = "database_sre"

    # ═══════════════════════════════════════════════════════════════
    # NETWORKING
    # ═══════════════════════════════════════════════════════════════

    NETWORK_ENGINEER = "network_engineer"
    JUNIOR_NETWORK_ENGINEER = "junior_network_engineer"
    SENIOR_NETWORK_ENGINEER = "senior_network_engineer"
    LEAD_NETWORK_ENGINEER = "lead_network_engineer"
    PRINCIPAL_NETWORK_ENGINEER = "principal_network_engineer"
    NETWORK_ARCHITECT = "network_architect"
    SENIOR_NETWORK_ARCHITECT = "senior_network_architect"
    ENTERPRISE_NETWORK_ARCHITECT = "enterprise_network_architect"
    NETWORK_ADMINISTRATOR = "network_administrator"
    WIRELESS_ENGINEER = "wireless_engineer"
    WAN_ENGINEER = "wan_engineer"
    LAN_ENGINEER = "lan_engineer"
    SD_WAN_ENGINEER = "sd_wan_engineer"
    VOIP_ENGINEER = "voip_engineer"
    UC_ENGINEER = "uc_engineer"
    LOAD_BALANCER_ENGINEER = "load_balancer_engineer"
    CDN_ENGINEER = "cdn_engineer"
    DNS_ENGINEER = "dns_engineer"
    NETWORK_AUTOMATION_ENGINEER = "network_automation_engineer"
    NETDEVOPS_ENGINEER = "netdevops_engineer"

    # ═══════════════════════════════════════════════════════════════
    # SYSTEMS ADMINISTRATION
    # ═══════════════════════════════════════════════════════════════

    SYSADMIN = "sysadmin"
    JUNIOR_SYSADMIN = "junior_sysadmin"
    SENIOR_SYSADMIN = "senior_sysadmin"
    LEAD_SYSADMIN = "lead_sysadmin"
    LINUX_ADMINISTRATOR = "linux_administrator"
    LINUX_ENGINEER = "linux_engineer"
    LINUX_SYSTEMS_ENGINEER = "linux_systems_engineer"
    WINDOWS_ADMINISTRATOR = "windows_administrator"
    WINDOWS_ENGINEER = "windows_engineer"
    WINDOWS_SYSTEMS_ENGINEER = "windows_systems_engineer"
    AD_ENGINEER = "ad_engineer"
    UNIX_ADMINISTRATOR = "unix_administrator"
    UNIX_ENGINEER = "unix_engineer"
    AIX_ADMINISTRATOR = "aix_administrator"
    SOLARIS_ADMINISTRATOR = "solaris_administrator"
    VIRTUALIZATION_ENGINEER = "virtualization_engineer"
    VMWARE_ENGINEER = "vmware_engineer"
    VMWARE_ADMINISTRATOR = "vmware_administrator"
    HYPERV_ADMINISTRATOR = "hyperv_administrator"
    STORAGE_ENGINEER = "storage_engineer"
    STORAGE_ADMINISTRATOR = "storage_administrator"
    STORAGE_ARCHITECT = "storage_architect"
    SAN_ENGINEER = "san_engineer"
    NAS_ENGINEER = "nas_engineer"
    BACKUP_ENGINEER = "backup_engineer"

    # ═══════════════════════════════════════════════════════════════
    # QA / TESTING
    # ═══════════════════════════════════════════════════════════════

    QA_ENGINEER = "qa_engineer"
    JUNIOR_QA_ENGINEER = "junior_qa_engineer"
    SENIOR_QA_ENGINEER = "senior_qa_engineer"
    LEAD_QA_ENGINEER = "lead_qa_engineer"
    PRINCIPAL_QA_ENGINEER = "principal_qa_engineer"
    TEST_ENGINEER = "test_engineer"
    SOFTWARE_TEST_ENGINEER = "software_test_engineer"
    SDET = "sdet"
    AUTOMATION_TEST_ENGINEER = "automation_test_engineer"
    TEST_AUTOMATION_ENGINEER = "test_automation_engineer"
    TEST_AUTOMATION_ARCHITECT = "test_automation_architect"
    SELENIUM_DEVELOPER = "selenium_developer"
    CYPRESS_DEVELOPER = "cypress_developer"
    PLAYWRIGHT_DEVELOPER = "playwright_developer"
    PERFORMANCE_TEST_ENGINEER = "performance_test_engineer"
    PERFORMANCE_ENGINEER = "performance_engineer"
    LOAD_TEST_ENGINEER = "load_test_engineer"
    MANUAL_TESTER = "manual_tester"
    MANUAL_QA = "manual_qa"
    MOBILE_QA_ENGINEER = "mobile_qa_engineer"
    API_TEST_ENGINEER = "api_test_engineer"
    SECURITY_TESTER = "security_tester"
    ACCESSIBILITY_TESTER = "accessibility_tester"
    USABILITY_TESTER = "usability_tester"
    LOCALIZATION_TESTER = "localization_tester"
    QA_MANAGER = "qa_manager"
    QA_LEAD = "qa_lead"
    TEST_MANAGER = "test_manager"
    HEAD_OF_QA = "head_of_qa"
    DIRECTOR_QA = "director_qa"
    VP_QA = "vp_qa"

    # ═══════════════════════════════════════════════════════════════
    # ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════

    SOFTWARE_ARCHITECT = "software_architect"
    SENIOR_SOFTWARE_ARCHITECT = "senior_software_architect"
    PRINCIPAL_ARCHITECT = "principal_architect"
    CHIEF_ARCHITECT = "chief_architect"
    ENTERPRISE_ARCHITECT = "enterprise_architect"
    SENIOR_ENTERPRISE_ARCHITECT = "senior_enterprise_architect"
    CHIEF_ENTERPRISE_ARCHITECT = "chief_enterprise_architect"
    TOGAF_ARCHITECT = "togaf_architect"
    SOLUTIONS_ARCHITECT = "solutions_architect"
    SENIOR_SOLUTIONS_ARCHITECT = "senior_solutions_architect"
    PRINCIPAL_SOLUTIONS_ARCHITECT = "principal_solutions_architect"
    DOMAIN_ARCHITECT = "domain_architect"
    TECHNICAL_ARCHITECT = "technical_architect"
    INTEGRATION_ARCHITECT = "integration_architect"
    API_ARCHITECT = "api_architect"
    MICROSERVICES_ARCHITECT = "microservices_architect"

    # ═══════════════════════════════════════════════════════════════
    # EMERGING TECH
    # ═══════════════════════════════════════════════════════════════

    # --- Blockchain ---
    BLOCKCHAIN_DEVELOPER = "blockchain_developer"
    BLOCKCHAIN_ENGINEER = "blockchain_engineer"
    BLOCKCHAIN_ARCHITECT = "blockchain_architect"
    SMART_CONTRACT_DEVELOPER = "smart_contract_developer"
    SOLIDITY_DEVELOPER = "solidity_developer"
    WEB3_DEVELOPER = "web3_developer"
    DEFI_DEVELOPER = "defi_developer"
    NFT_DEVELOPER = "nft_developer"
    CRYPTO_ENGINEER = "crypto_engineer"

    # --- Quantum Computing ---
    QUANTUM_SOFTWARE_ENGINEER = "quantum_software_engineer"
    QUANTUM_DEVELOPER = "quantum_developer"
    QUANTUM_ALGORITHM_DEVELOPER = "quantum_algorithm_developer"
    QUANTUM_RESEARCHER = "quantum_researcher"
    QUANTUM_APPLICATIONS_ENGINEER = "quantum_applications_engineer"
    QISKIT_DEVELOPER = "qiskit_developer"
    CIRQ_DEVELOPER = "cirq_developer"

    # --- IoT ---
    IOT_ENGINEER = "iot_engineer"
    IOT_DEVELOPER = "iot_developer"
    IOT_ARCHITECT = "iot_architect"
    IOT_SOLUTIONS_ARCHITECT = "iot_solutions_architect"
    EMBEDDED_IOT_ENGINEER = "embedded_iot_engineer"
    IOT_PLATFORM_ENGINEER = "iot_platform_engineer"

    # --- AR/VR/XR ---
    AR_DEVELOPER = "ar_developer"
    VR_DEVELOPER = "vr_developer"
    XR_DEVELOPER = "xr_developer"
    MIXED_REALITY_DEVELOPER = "mixed_reality_developer"
    METAVERSE_DEVELOPER = "metaverse_developer"

    # --- Edge Computing ---
    EDGE_COMPUTING_ENGINEER = "edge_computing_engineer"
    EDGE_ARCHITECT = "edge_architect"
    EDGE_ML_ENGINEER = "edge_ml_engineer"

    # --- Robotics ---
    ROBOTICS_ENGINEER = "robotics_engineer"
    ROBOTICS_SOFTWARE_ENGINEER = "robotics_software_engineer"
    ROS_DEVELOPER = "ros_developer"
    AUTONOMOUS_SYSTEMS_ENGINEER = "autonomous_systems_engineer"

    # ═══════════════════════════════════════════════════════════════
    # PRODUCT & TECHNICAL MANAGEMENT
    # ═══════════════════════════════════════════════════════════════

    TECHNICAL_PRODUCT_MANAGER = "technical_product_manager"
    PRODUCT_MANAGER = "product_manager"
    SENIOR_PRODUCT_MANAGER = "senior_product_manager"
    PRINCIPAL_PRODUCT_MANAGER = "principal_product_manager"
    GROUP_PRODUCT_MANAGER = "group_product_manager"
    DIRECTOR_PRODUCT = "director_product"
    VP_PRODUCT = "vp_product"
    CPO = "cpo"
    TPM = "tpm"
    SENIOR_TPM = "senior_tpm"
    PRINCIPAL_TPM = "principal_tpm"
    DIRECTOR_TPM = "director_tpm"
    ENGINEERING_MANAGER = "engineering_manager"
    SENIOR_ENGINEERING_MANAGER = "senior_engineering_manager"
    DIRECTOR_ENGINEERING = "director_engineering"
    SENIOR_DIRECTOR_ENGINEERING = "senior_director_engineering"
    VP_ENGINEERING = "vp_engineering"
    SVP_ENGINEERING = "svp_engineering"
    CTO = "cto"
    CIO = "cio"
    TECH_LEAD = "tech_lead"
    TEAM_LEAD = "team_lead"
    TECHNICAL_LEAD = "technical_lead"
    ENGINEERING_LEAD = "engineering_lead"

    # ═══════════════════════════════════════════════════════════════
    # SUPPORT & OPERATIONS
    # ═══════════════════════════════════════════════════════════════

    IT_SUPPORT_SPECIALIST = "it_support_specialist"
    IT_SUPPORT_ENGINEER = "it_support_engineer"
    HELP_DESK_TECHNICIAN = "help_desk_technician"
    DESKTOP_SUPPORT = "desktop_support"
    TECHNICAL_SUPPORT_ENGINEER = "technical_support_engineer"
    IT_OPERATIONS_ENGINEER = "it_operations_engineer"
    IT_OPERATIONS_MANAGER = "it_operations_manager"
    NOC_ENGINEER = "noc_engineer"
    NOC_ANALYST = "noc_analyst"
    TECHNICAL_WRITER = "technical_writer"
    DOCUMENTATION_ENGINEER = "documentation_engineer"
    API_DOCUMENTATION_WRITER = "api_documentation_writer"
    DEVELOPER_ADVOCATE = "developer_advocate"
    DEVELOPER_EVANGELIST = "developer_evangelist"
    DEVREL_ENGINEER = "devrel_engineer"
    COMMUNITY_MANAGER = "community_manager"
    SOLUTIONS_ENGINEER = "solutions_engineer"
    SALES_ENGINEER = "sales_engineer"
    PRE_SALES_ENGINEER = "pre_sales_engineer"
    TECHNICAL_CONSULTANT = "technical_consultant"
    IT_CONSULTANT = "it_consultant"


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_career_display_name(career: TechCareer) -> str:
    """Convert enum value to display name"""
    return career.value.replace("_", " ").title()


def get_language_display_name(lang: CodeLanguage) -> str:
    """Convert enum value to display name"""
    special_names = {
        "cpp": "C++",
        "csharp": "C#",
        "fsharp": "F#",
        "qsharp": "Q#",
        "nodejs": "Node.js",
        "plsql": "PL/SQL",
        "tsql": "T-SQL",
        "mongodb": "MongoDB",
        "postgresql": "PostgreSQL",
        "mysql": "MySQL",
        "graphql": "GraphQL",
        "openapi": "OpenAPI",
        "asyncapi": "AsyncAPI",
        "kubernetes": "Kubernetes",
        "dockerfile": "Dockerfile",
        "docker_compose": "Docker Compose",
        "terraform_hcl": "Terraform (HCL)",
        "wasm": "WebAssembly",
        "qiskit": "Qiskit",
        "cirq": "Cirq",
        "openqasm": "OpenQASM",
    }
    return special_names.get(lang.value, lang.value.replace("_", " ").title())


def get_domain_display_name(domain: TechDomain) -> str:
    """Convert enum value to display name"""
    special_names = {
        "sre": "Site Reliability Engineering",
        "mlops": "MLOps",
        "devsecops": "DevSecOps",
        "iac": "Infrastructure as Code",
        "cicd": "CI/CD",
        "iam": "Identity & Access Management",
        "grc": "Governance, Risk & Compliance",
        "llm": "Large Language Models",
        "nlp": "Natural Language Processing",
        "mdm": "Master Data Management",
        "sdn": "Software-Defined Networking",
        "sd_wan": "SD-WAN",
        "tdd": "Test-Driven Development",
        "bdd": "Behavior-Driven Development",
        "ddd": "Domain-Driven Design",
        "ar_vr": "AR/VR",
        "iot": "IoT",
        "defi": "DeFi",
        "web3": "Web3",
    }
    return special_names.get(domain.value, domain.value.replace("_", " ").title())


# Career to Domain mapping for context detection - COMPLETE 545+ MAPPINGS
CAREER_DOMAIN_MAP: Dict[TechCareer, List[TechDomain]] = {
    # ═══════════════════════════════════════════════════════════════
    # SOFTWARE DEVELOPMENT
    # ═══════════════════════════════════════════════════════════════

    # General Development
    TechCareer.SOFTWARE_DEVELOPER: [TechDomain.SOFTWARE_ENGINEERING, TechDomain.PROGRAMMING_FUNDAMENTALS],
    TechCareer.SOFTWARE_ENGINEER: [TechDomain.SOFTWARE_ENGINEERING, TechDomain.SOFTWARE_ARCHITECTURE],
    TechCareer.JUNIOR_DEVELOPER: [TechDomain.PROGRAMMING_FUNDAMENTALS, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.MID_LEVEL_DEVELOPER: [TechDomain.SOFTWARE_ENGINEERING, TechDomain.DESIGN_PATTERNS],
    TechCareer.SENIOR_DEVELOPER: [TechDomain.SOFTWARE_ENGINEERING, TechDomain.SOFTWARE_ARCHITECTURE],
    TechCareer.STAFF_ENGINEER: [TechDomain.SOFTWARE_ARCHITECTURE, TechDomain.DISTRIBUTED_SYSTEMS],
    TechCareer.PRINCIPAL_ENGINEER: [TechDomain.SOFTWARE_ARCHITECTURE, TechDomain.ENTERPRISE_ARCHITECTURE],
    TechCareer.DISTINGUISHED_ENGINEER: [TechDomain.ENTERPRISE_ARCHITECTURE, TechDomain.SOFTWARE_ARCHITECTURE],
    TechCareer.FELLOW_ENGINEER: [TechDomain.ENTERPRISE_ARCHITECTURE, TechDomain.SOFTWARE_ARCHITECTURE],

    # Frontend
    TechCareer.FRONTEND_DEVELOPER: [TechDomain.WEB_FRONTEND, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.FRONTEND_ENGINEER: [TechDomain.WEB_FRONTEND, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.UI_DEVELOPER: [TechDomain.WEB_FRONTEND, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.UI_ENGINEER: [TechDomain.WEB_FRONTEND, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.JAVASCRIPT_DEVELOPER: [TechDomain.WEB_FRONTEND, TechDomain.WEB_BACKEND],
    TechCareer.REACT_DEVELOPER: [TechDomain.WEB_FRONTEND, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.ANGULAR_DEVELOPER: [TechDomain.WEB_FRONTEND, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.VUE_DEVELOPER: [TechDomain.WEB_FRONTEND, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.SVELTE_DEVELOPER: [TechDomain.WEB_FRONTEND, TechDomain.SOFTWARE_ENGINEERING],

    # Backend
    TechCareer.BACKEND_DEVELOPER: [TechDomain.WEB_BACKEND, TechDomain.API_DESIGN],
    TechCareer.BACKEND_ENGINEER: [TechDomain.WEB_BACKEND, TechDomain.DISTRIBUTED_SYSTEMS],
    TechCareer.API_DEVELOPER: [TechDomain.API_DESIGN, TechDomain.WEB_BACKEND],
    TechCareer.PYTHON_DEVELOPER: [TechDomain.WEB_BACKEND, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.JAVA_DEVELOPER: [TechDomain.WEB_BACKEND, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.NODEJS_DEVELOPER: [TechDomain.WEB_BACKEND, TechDomain.WEB_FRONTEND],
    TechCareer.GO_DEVELOPER: [TechDomain.WEB_BACKEND, TechDomain.SYSTEMS_PROGRAMMING],
    TechCareer.RUST_DEVELOPER: [TechDomain.SYSTEMS_PROGRAMMING, TechDomain.WEB_BACKEND],
    TechCareer.DOTNET_DEVELOPER: [TechDomain.WEB_BACKEND, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.PHP_DEVELOPER: [TechDomain.WEB_BACKEND, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.RUBY_DEVELOPER: [TechDomain.WEB_BACKEND, TechDomain.SOFTWARE_ENGINEERING],

    # Fullstack
    TechCareer.FULLSTACK_DEVELOPER: [TechDomain.FULLSTACK, TechDomain.WEB_FRONTEND, TechDomain.WEB_BACKEND],
    TechCareer.FULLSTACK_ENGINEER: [TechDomain.FULLSTACK, TechDomain.SOFTWARE_ARCHITECTURE],
    TechCareer.MERN_DEVELOPER: [TechDomain.FULLSTACK, TechDomain.WEB_FRONTEND],
    TechCareer.MEAN_DEVELOPER: [TechDomain.FULLSTACK, TechDomain.WEB_FRONTEND],

    # Mobile
    TechCareer.MOBILE_DEVELOPER: [TechDomain.MOBILE_DEVELOPMENT, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.MOBILE_ENGINEER: [TechDomain.MOBILE_DEVELOPMENT, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.IOS_DEVELOPER: [TechDomain.MOBILE_DEVELOPMENT, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.ANDROID_DEVELOPER: [TechDomain.MOBILE_DEVELOPMENT, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.FLUTTER_DEVELOPER: [TechDomain.MOBILE_DEVELOPMENT, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.REACT_NATIVE_DEVELOPER: [TechDomain.MOBILE_DEVELOPMENT, TechDomain.WEB_FRONTEND],
    TechCareer.SWIFT_DEVELOPER: [TechDomain.MOBILE_DEVELOPMENT, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.KOTLIN_DEVELOPER: [TechDomain.MOBILE_DEVELOPMENT, TechDomain.SOFTWARE_ENGINEERING],

    # Desktop & Systems
    TechCareer.DESKTOP_DEVELOPER: [TechDomain.DESKTOP_DEVELOPMENT, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.SYSTEMS_PROGRAMMER: [TechDomain.SYSTEMS_PROGRAMMING, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.EMBEDDED_DEVELOPER: [TechDomain.EMBEDDED_SYSTEMS, TechDomain.SYSTEMS_PROGRAMMING],
    TechCareer.EMBEDDED_ENGINEER: [TechDomain.EMBEDDED_SYSTEMS, TechDomain.SYSTEMS_PROGRAMMING],
    TechCareer.FIRMWARE_ENGINEER: [TechDomain.EMBEDDED_SYSTEMS, TechDomain.SYSTEMS_PROGRAMMING],
    TechCareer.LOW_LEVEL_PROGRAMMER: [TechDomain.SYSTEMS_PROGRAMMING, TechDomain.EMBEDDED_SYSTEMS],

    # Game Development
    TechCareer.GAME_DEVELOPER: [TechDomain.GAME_DEVELOPMENT, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.GAME_PROGRAMMER: [TechDomain.GAME_DEVELOPMENT, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.GAME_ENGINE_DEVELOPER: [TechDomain.GAME_DEVELOPMENT, TechDomain.SYSTEMS_PROGRAMMING],
    TechCareer.UNITY_DEVELOPER: [TechDomain.GAME_DEVELOPMENT, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.UNREAL_DEVELOPER: [TechDomain.GAME_DEVELOPMENT, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.GRAPHICS_PROGRAMMER: [TechDomain.GAME_DEVELOPMENT, TechDomain.SYSTEMS_PROGRAMMING],
    TechCareer.GAMEPLAY_PROGRAMMER: [TechDomain.GAME_DEVELOPMENT, TechDomain.SOFTWARE_ENGINEERING],

    # ═══════════════════════════════════════════════════════════════
    # DATA - COMPLETE ECOSYSTEM
    # ═══════════════════════════════════════════════════════════════

    # Data Engineering
    TechCareer.DATA_ENGINEER: [TechDomain.DATA_ENGINEERING, TechDomain.BIG_DATA],
    TechCareer.SENIOR_DATA_ENGINEER: [TechDomain.DATA_ENGINEERING, TechDomain.DATA_ARCHITECTURE],
    TechCareer.LEAD_DATA_ENGINEER: [TechDomain.DATA_ENGINEERING, TechDomain.DATA_ARCHITECTURE],
    TechCareer.PRINCIPAL_DATA_ENGINEER: [TechDomain.DATA_ENGINEERING, TechDomain.DATA_ARCHITECTURE],
    TechCareer.DATA_PLATFORM_ENGINEER: [TechDomain.DATA_ENGINEERING, TechDomain.PLATFORM_ENGINEERING],
    TechCareer.ETL_DEVELOPER: [TechDomain.DATA_ENGINEERING, TechDomain.DATA_INTEGRATION],
    TechCareer.ELT_DEVELOPER: [TechDomain.DATA_ENGINEERING, TechDomain.DATA_INTEGRATION],
    TechCareer.DATA_PIPELINE_ENGINEER: [TechDomain.DATA_ENGINEERING, TechDomain.STREAMING_DATA],
    TechCareer.STREAMING_DATA_ENGINEER: [TechDomain.STREAMING_DATA, TechDomain.DATA_ENGINEERING],
    TechCareer.BATCH_DATA_ENGINEER: [TechDomain.DATA_ENGINEERING, TechDomain.BIG_DATA],

    # Data Enablement
    TechCareer.DATA_ENABLER: [TechDomain.DATA_GOVERNANCE, TechDomain.DATA_CATALOG],
    TechCareer.DATA_ENABLEMENT_LEAD: [TechDomain.DATA_GOVERNANCE, TechDomain.DATA_ARCHITECTURE],
    TechCareer.DATA_ENABLEMENT_MANAGER: [TechDomain.DATA_GOVERNANCE, TechDomain.DATA_CATALOG],
    TechCareer.DATA_DEMOCRATIZATION_LEAD: [TechDomain.DATA_GOVERNANCE, TechDomain.DATA_CATALOG],

    # Data Lineage
    TechCareer.DATA_LINEAGE_DEVELOPER: [TechDomain.DATA_LINEAGE, TechDomain.DATA_GOVERNANCE],
    TechCareer.DATA_LINEAGE_ANALYST: [TechDomain.DATA_LINEAGE, TechDomain.DATA_QUALITY],
    TechCareer.DATA_LINEAGE_ARCHITECT: [TechDomain.DATA_LINEAGE, TechDomain.DATA_ARCHITECTURE],
    TechCareer.DATA_LINEAGE_ENGINEER: [TechDomain.DATA_LINEAGE, TechDomain.DATA_ENGINEERING],
    TechCareer.DATA_LINEAGE_SPECIALIST: [TechDomain.DATA_LINEAGE, TechDomain.DATA_GOVERNANCE],

    # Data Quality
    TechCareer.DATA_QUALITY_ENGINEER: [TechDomain.DATA_QUALITY, TechDomain.DATA_ENGINEERING],
    TechCareer.DATA_QUALITY_ANALYST: [TechDomain.DATA_QUALITY, TechDomain.DATA_ANALYTICS],
    TechCareer.DATA_QUALITY_MANAGER: [TechDomain.DATA_QUALITY, TechDomain.DATA_GOVERNANCE],
    TechCareer.DATA_QUALITY_ARCHITECT: [TechDomain.DATA_QUALITY, TechDomain.DATA_ARCHITECTURE],
    TechCareer.DATA_STEWARD: [TechDomain.DATA_GOVERNANCE, TechDomain.DATA_QUALITY],
    TechCareer.DATA_CUSTODIAN: [TechDomain.DATA_GOVERNANCE, TechDomain.DATA_QUALITY],

    # Data Governance
    TechCareer.DATA_GOVERNANCE_ANALYST: [TechDomain.DATA_GOVERNANCE, TechDomain.DATA_QUALITY],
    TechCareer.DATA_GOVERNANCE_ENGINEER: [TechDomain.DATA_GOVERNANCE, TechDomain.DATA_CATALOG],
    TechCareer.DATA_GOVERNANCE_MANAGER: [TechDomain.DATA_GOVERNANCE, TechDomain.DATA_ARCHITECTURE],
    TechCareer.DATA_GOVERNANCE_ARCHITECT: [TechDomain.DATA_GOVERNANCE, TechDomain.DATA_ARCHITECTURE],
    TechCareer.DATA_GOVERNANCE_LEAD: [TechDomain.DATA_GOVERNANCE, TechDomain.DATA_QUALITY],
    TechCareer.DATA_GOVERNANCE_OFFICER: [TechDomain.DATA_GOVERNANCE, TechDomain.DATA_ARCHITECTURE],
    TechCareer.CHIEF_DATA_GOVERNANCE_OFFICER: [TechDomain.DATA_GOVERNANCE, TechDomain.DATA_ARCHITECTURE],

    # Data Architecture
    TechCareer.DATA_ARCHITECT: [TechDomain.DATA_ARCHITECTURE, TechDomain.DATA_MODELING],
    TechCareer.SENIOR_DATA_ARCHITECT: [TechDomain.DATA_ARCHITECTURE, TechDomain.ENTERPRISE_ARCHITECTURE],
    TechCareer.ENTERPRISE_DATA_ARCHITECT: [TechDomain.DATA_ARCHITECTURE, TechDomain.ENTERPRISE_ARCHITECTURE],
    TechCareer.CLOUD_DATA_ARCHITECT: [TechDomain.DATA_ARCHITECTURE, TechDomain.CLOUD_COMPUTING],
    TechCareer.DATA_MODELING_ARCHITECT: [TechDomain.DATA_MODELING, TechDomain.DATA_ARCHITECTURE],
    TechCareer.DATA_WAREHOUSE_ARCHITECT: [TechDomain.DATA_WAREHOUSING, TechDomain.DATA_ARCHITECTURE],
    TechCareer.DATA_LAKEHOUSE_ARCHITECT: [TechDomain.DATA_LAKEHOUSE, TechDomain.DATA_ARCHITECTURE],

    # Data Modeling
    TechCareer.DATA_MODELER: [TechDomain.DATA_MODELING, TechDomain.DATA_ARCHITECTURE],
    TechCareer.SENIOR_DATA_MODELER: [TechDomain.DATA_MODELING, TechDomain.DATA_ARCHITECTURE],
    TechCareer.DIMENSIONAL_MODELER: [TechDomain.DATA_MODELING, TechDomain.DATA_WAREHOUSING],
    TechCareer.CONCEPTUAL_DATA_MODELER: [TechDomain.DATA_MODELING, TechDomain.DATA_ARCHITECTURE],
    TechCareer.LOGICAL_DATA_MODELER: [TechDomain.DATA_MODELING, TechDomain.DATA_ARCHITECTURE],
    TechCareer.PHYSICAL_DATA_MODELER: [TechDomain.DATA_MODELING, TechDomain.DATABASES],

    # Data Science
    TechCareer.DATA_SCIENTIST: [TechDomain.DATA_SCIENCE, TechDomain.MACHINE_LEARNING],
    TechCareer.JUNIOR_DATA_SCIENTIST: [TechDomain.DATA_SCIENCE, TechDomain.DATA_ANALYTICS],
    TechCareer.SENIOR_DATA_SCIENTIST: [TechDomain.DATA_SCIENCE, TechDomain.MACHINE_LEARNING],
    TechCareer.LEAD_DATA_SCIENTIST: [TechDomain.DATA_SCIENCE, TechDomain.MACHINE_LEARNING],
    TechCareer.PRINCIPAL_DATA_SCIENTIST: [TechDomain.DATA_SCIENCE, TechDomain.DEEP_LEARNING],
    TechCareer.STAFF_DATA_SCIENTIST: [TechDomain.DATA_SCIENCE, TechDomain.MACHINE_LEARNING],
    TechCareer.RESEARCH_DATA_SCIENTIST: [TechDomain.DATA_SCIENCE, TechDomain.DEEP_LEARNING],
    TechCareer.APPLIED_DATA_SCIENTIST: [TechDomain.DATA_SCIENCE, TechDomain.MACHINE_LEARNING],

    # Data Analytics
    TechCareer.DATA_ANALYST: [TechDomain.DATA_ANALYTICS, TechDomain.BUSINESS_INTELLIGENCE],
    TechCareer.JUNIOR_DATA_ANALYST: [TechDomain.DATA_ANALYTICS, TechDomain.BUSINESS_INTELLIGENCE],
    TechCareer.SENIOR_DATA_ANALYST: [TechDomain.DATA_ANALYTICS, TechDomain.DATA_SCIENCE],
    TechCareer.LEAD_DATA_ANALYST: [TechDomain.DATA_ANALYTICS, TechDomain.BUSINESS_INTELLIGENCE],
    TechCareer.BUSINESS_DATA_ANALYST: [TechDomain.DATA_ANALYTICS, TechDomain.BUSINESS_INTELLIGENCE],
    TechCareer.MARKETING_DATA_ANALYST: [TechDomain.DATA_ANALYTICS, TechDomain.BUSINESS_INTELLIGENCE],
    TechCareer.FINANCIAL_DATA_ANALYST: [TechDomain.DATA_ANALYTICS, TechDomain.BUSINESS_INTELLIGENCE],
    TechCareer.PRODUCT_DATA_ANALYST: [TechDomain.DATA_ANALYTICS, TechDomain.BUSINESS_INTELLIGENCE],
    TechCareer.OPERATIONS_DATA_ANALYST: [TechDomain.DATA_ANALYTICS, TechDomain.BUSINESS_INTELLIGENCE],
    TechCareer.HEALTHCARE_DATA_ANALYST: [TechDomain.DATA_ANALYTICS, TechDomain.BUSINESS_INTELLIGENCE],

    # Business Intelligence
    TechCareer.BI_DEVELOPER: [TechDomain.BUSINESS_INTELLIGENCE, TechDomain.DATA_ANALYTICS],
    TechCareer.BI_ANALYST: [TechDomain.BUSINESS_INTELLIGENCE, TechDomain.DATA_ANALYTICS],
    TechCareer.BI_ENGINEER: [TechDomain.BUSINESS_INTELLIGENCE, TechDomain.DATA_ENGINEERING],
    TechCareer.BI_ARCHITECT: [TechDomain.BUSINESS_INTELLIGENCE, TechDomain.DATA_ARCHITECTURE],
    TechCareer.BI_CONSULTANT: [TechDomain.BUSINESS_INTELLIGENCE, TechDomain.DATA_ANALYTICS],
    TechCareer.BI_MANAGER: [TechDomain.BUSINESS_INTELLIGENCE, TechDomain.DATA_ANALYTICS],
    TechCareer.TABLEAU_DEVELOPER: [TechDomain.BUSINESS_INTELLIGENCE, TechDomain.DATA_ANALYTICS],
    TechCareer.POWER_BI_DEVELOPER: [TechDomain.BUSINESS_INTELLIGENCE, TechDomain.DATA_ANALYTICS],
    TechCareer.LOOKER_DEVELOPER: [TechDomain.BUSINESS_INTELLIGENCE, TechDomain.DATA_ANALYTICS],
    TechCareer.QLIK_DEVELOPER: [TechDomain.BUSINESS_INTELLIGENCE, TechDomain.DATA_ANALYTICS],

    # Analytics Engineering
    TechCareer.ANALYTICS_ENGINEER: [TechDomain.ANALYTICS_ENGINEERING, TechDomain.DATA_ENGINEERING],
    TechCareer.SENIOR_ANALYTICS_ENGINEER: [TechDomain.ANALYTICS_ENGINEERING, TechDomain.DATA_ARCHITECTURE],
    TechCareer.LEAD_ANALYTICS_ENGINEER: [TechDomain.ANALYTICS_ENGINEERING, TechDomain.DATA_ARCHITECTURE],
    TechCareer.DBT_DEVELOPER: [TechDomain.ANALYTICS_ENGINEERING, TechDomain.DATA_ENGINEERING],

    # Data Operations
    TechCareer.DATAOPS_ENGINEER: [TechDomain.DATA_ENGINEERING, TechDomain.DEVOPS],
    TechCareer.DATAOPS_ARCHITECT: [TechDomain.DATA_ARCHITECTURE, TechDomain.DEVOPS],
    TechCareer.DATAOPS_MANAGER: [TechDomain.DATA_ENGINEERING, TechDomain.DEVOPS],

    # Data Management
    TechCareer.DATA_MANAGER: [TechDomain.DATA_GOVERNANCE, TechDomain.DATA_ARCHITECTURE],
    TechCareer.DATA_MANAGEMENT_SPECIALIST: [TechDomain.DATA_GOVERNANCE, TechDomain.DATA_QUALITY],
    TechCareer.MDM_SPECIALIST: [TechDomain.MASTER_DATA_MANAGEMENT, TechDomain.DATA_GOVERNANCE],
    TechCareer.MDM_ARCHITECT: [TechDomain.MASTER_DATA_MANAGEMENT, TechDomain.DATA_ARCHITECTURE],
    TechCareer.MDM_DEVELOPER: [TechDomain.MASTER_DATA_MANAGEMENT, TechDomain.DATA_ENGINEERING],

    # Data Integration
    TechCareer.DATA_INTEGRATION_ENGINEER: [TechDomain.DATA_INTEGRATION, TechDomain.DATA_ENGINEERING],
    TechCareer.DATA_INTEGRATION_ARCHITECT: [TechDomain.DATA_INTEGRATION, TechDomain.DATA_ARCHITECTURE],
    TechCareer.DATA_INTEGRATION_SPECIALIST: [TechDomain.DATA_INTEGRATION, TechDomain.DATA_ENGINEERING],

    # Data Catalog
    TechCareer.DATA_CATALOG_ENGINEER: [TechDomain.DATA_CATALOG, TechDomain.DATA_GOVERNANCE],
    TechCareer.DATA_CATALOG_ARCHITECT: [TechDomain.DATA_CATALOG, TechDomain.DATA_ARCHITECTURE],
    TechCareer.DATA_CATALOG_ANALYST: [TechDomain.DATA_CATALOG, TechDomain.DATA_GOVERNANCE],
    TechCareer.METADATA_ENGINEER: [TechDomain.METADATA_MANAGEMENT, TechDomain.DATA_CATALOG],
    TechCareer.METADATA_ANALYST: [TechDomain.METADATA_MANAGEMENT, TechDomain.DATA_GOVERNANCE],
    TechCareer.METADATA_ARCHITECT: [TechDomain.METADATA_MANAGEMENT, TechDomain.DATA_ARCHITECTURE],

    # Big Data
    TechCareer.BIG_DATA_ENGINEER: [TechDomain.BIG_DATA, TechDomain.DATA_ENGINEERING],
    TechCareer.BIG_DATA_ARCHITECT: [TechDomain.BIG_DATA, TechDomain.DATA_ARCHITECTURE],
    TechCareer.BIG_DATA_DEVELOPER: [TechDomain.BIG_DATA, TechDomain.DATA_ENGINEERING],
    TechCareer.HADOOP_DEVELOPER: [TechDomain.BIG_DATA, TechDomain.DATA_ENGINEERING],
    TechCareer.SPARK_DEVELOPER: [TechDomain.BIG_DATA, TechDomain.STREAMING_DATA],
    TechCareer.KAFKA_DEVELOPER: [TechDomain.STREAMING_DATA, TechDomain.DATA_ENGINEERING],
    TechCareer.FLINK_DEVELOPER: [TechDomain.STREAMING_DATA, TechDomain.DATA_ENGINEERING],

    # Data Leadership
    TechCareer.HEAD_OF_DATA: [TechDomain.DATA_ARCHITECTURE, TechDomain.DATA_GOVERNANCE],
    TechCareer.VP_DATA: [TechDomain.DATA_ARCHITECTURE, TechDomain.DATA_GOVERNANCE],
    TechCareer.DIRECTOR_DATA_ENGINEERING: [TechDomain.DATA_ENGINEERING, TechDomain.DATA_ARCHITECTURE],
    TechCareer.DIRECTOR_DATA_SCIENCE: [TechDomain.DATA_SCIENCE, TechDomain.MACHINE_LEARNING],
    TechCareer.CHIEF_DATA_OFFICER: [TechDomain.DATA_GOVERNANCE, TechDomain.DATA_ARCHITECTURE],
    TechCareer.CHIEF_ANALYTICS_OFFICER: [TechDomain.DATA_ANALYTICS, TechDomain.BUSINESS_INTELLIGENCE],

    # ═══════════════════════════════════════════════════════════════
    # AI / MACHINE LEARNING
    # ═══════════════════════════════════════════════════════════════

    # Machine Learning Engineering
    TechCareer.ML_ENGINEER: [TechDomain.MACHINE_LEARNING, TechDomain.MLOPS],
    TechCareer.JUNIOR_ML_ENGINEER: [TechDomain.MACHINE_LEARNING, TechDomain.DATA_SCIENCE],
    TechCareer.SENIOR_ML_ENGINEER: [TechDomain.MACHINE_LEARNING, TechDomain.MLOPS],
    TechCareer.LEAD_ML_ENGINEER: [TechDomain.MACHINE_LEARNING, TechDomain.MLOPS],
    TechCareer.PRINCIPAL_ML_ENGINEER: [TechDomain.MACHINE_LEARNING, TechDomain.DEEP_LEARNING],
    TechCareer.STAFF_ML_ENGINEER: [TechDomain.MACHINE_LEARNING, TechDomain.MLOPS],

    # ML Specializations
    TechCareer.ML_PLATFORM_ENGINEER: [TechDomain.MLOPS, TechDomain.PLATFORM_ENGINEERING],
    TechCareer.ML_INFRASTRUCTURE_ENGINEER: [TechDomain.MLOPS, TechDomain.INFRASTRUCTURE],
    TechCareer.ML_SYSTEMS_ENGINEER: [TechDomain.MLOPS, TechDomain.DISTRIBUTED_SYSTEMS],
    TechCareer.FEATURE_ENGINEER: [TechDomain.MACHINE_LEARNING, TechDomain.DATA_ENGINEERING],
    TechCareer.FEATURE_STORE_ENGINEER: [TechDomain.MLOPS, TechDomain.DATA_ENGINEERING],

    # MLOps
    TechCareer.MLOPS_ENGINEER: [TechDomain.MLOPS, TechDomain.DEVOPS],
    TechCareer.SENIOR_MLOPS_ENGINEER: [TechDomain.MLOPS, TechDomain.PLATFORM_ENGINEERING],
    TechCareer.LEAD_MLOPS_ENGINEER: [TechDomain.MLOPS, TechDomain.PLATFORM_ENGINEERING],
    TechCareer.MLOPS_ARCHITECT: [TechDomain.MLOPS, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.MLOPS_PLATFORM_ENGINEER: [TechDomain.MLOPS, TechDomain.PLATFORM_ENGINEERING],

    # Deep Learning
    TechCareer.DEEP_LEARNING_ENGINEER: [TechDomain.DEEP_LEARNING, TechDomain.NEURAL_NETWORKS],
    TechCareer.DEEP_LEARNING_RESEARCHER: [TechDomain.DEEP_LEARNING, TechDomain.NEURAL_NETWORKS],
    TechCareer.NEURAL_NETWORK_ENGINEER: [TechDomain.NEURAL_NETWORKS, TechDomain.DEEP_LEARNING],

    # AI Research
    TechCareer.AI_RESEARCHER: [TechDomain.MACHINE_LEARNING, TechDomain.DEEP_LEARNING],
    TechCareer.AI_RESEARCH_SCIENTIST: [TechDomain.DEEP_LEARNING, TechDomain.NEURAL_NETWORKS],
    TechCareer.AI_RESEARCH_ENGINEER: [TechDomain.MACHINE_LEARNING, TechDomain.DEEP_LEARNING],
    TechCareer.APPLIED_AI_RESEARCHER: [TechDomain.MACHINE_LEARNING, TechDomain.GENERATIVE_AI],

    # AI Engineering
    TechCareer.AI_ENGINEER: [TechDomain.MACHINE_LEARNING, TechDomain.GENERATIVE_AI],
    TechCareer.AI_DEVELOPER: [TechDomain.MACHINE_LEARNING, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.AI_SOLUTIONS_ENGINEER: [TechDomain.MACHINE_LEARNING, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.AI_PLATFORM_ENGINEER: [TechDomain.MLOPS, TechDomain.PLATFORM_ENGINEERING],
    TechCareer.AI_INFRASTRUCTURE_ENGINEER: [TechDomain.MLOPS, TechDomain.INFRASTRUCTURE],

    # NLP
    TechCareer.NLP_ENGINEER: [TechDomain.NLP, TechDomain.MACHINE_LEARNING],
    TechCareer.NLP_SCIENTIST: [TechDomain.NLP, TechDomain.DEEP_LEARNING],
    TechCareer.NLP_RESEARCHER: [TechDomain.NLP, TechDomain.DEEP_LEARNING],
    TechCareer.COMPUTATIONAL_LINGUIST: [TechDomain.NLP, TechDomain.MACHINE_LEARNING],
    TechCareer.CONVERSATIONAL_AI_ENGINEER: [TechDomain.CONVERSATIONAL_AI, TechDomain.NLP],
    TechCareer.CHATBOT_DEVELOPER: [TechDomain.CONVERSATIONAL_AI, TechDomain.NLP],

    # Computer Vision
    TechCareer.COMPUTER_VISION_ENGINEER: [TechDomain.COMPUTER_VISION, TechDomain.DEEP_LEARNING],
    TechCareer.COMPUTER_VISION_SCIENTIST: [TechDomain.COMPUTER_VISION, TechDomain.DEEP_LEARNING],
    TechCareer.COMPUTER_VISION_RESEARCHER: [TechDomain.COMPUTER_VISION, TechDomain.NEURAL_NETWORKS],
    TechCareer.IMAGE_PROCESSING_ENGINEER: [TechDomain.COMPUTER_VISION, TechDomain.MACHINE_LEARNING],
    TechCareer.VIDEO_ANALYTICS_ENGINEER: [TechDomain.COMPUTER_VISION, TechDomain.DEEP_LEARNING],

    # Generative AI
    TechCareer.GENERATIVE_AI_ENGINEER: [TechDomain.GENERATIVE_AI, TechDomain.LLM],
    TechCareer.LLM_ENGINEER: [TechDomain.LLM, TechDomain.GENERATIVE_AI],
    TechCareer.PROMPT_ENGINEER: [TechDomain.LLM, TechDomain.GENERATIVE_AI],
    TechCareer.AI_CONTENT_ENGINEER: [TechDomain.GENERATIVE_AI, TechDomain.NLP],
    TechCareer.DIFFUSION_MODEL_ENGINEER: [TechDomain.GENERATIVE_AI, TechDomain.DEEP_LEARNING],

    # Recommendation Systems
    TechCareer.RECOMMENDATION_ENGINEER: [TechDomain.RECOMMENDATION_SYSTEMS, TechDomain.MACHINE_LEARNING],
    TechCareer.PERSONALIZATION_ENGINEER: [TechDomain.RECOMMENDATION_SYSTEMS, TechDomain.MACHINE_LEARNING],
    TechCareer.SEARCH_RELEVANCE_ENGINEER: [TechDomain.RECOMMENDATION_SYSTEMS, TechDomain.NLP],

    # Reinforcement Learning
    TechCareer.RL_ENGINEER: [TechDomain.REINFORCEMENT_LEARNING, TechDomain.DEEP_LEARNING],
    TechCareer.RL_RESEARCHER: [TechDomain.REINFORCEMENT_LEARNING, TechDomain.DEEP_LEARNING],
    TechCareer.ROBOTICS_ML_ENGINEER: [TechDomain.REINFORCEMENT_LEARNING, TechDomain.ROBOTICS],

    # AI Ethics & Safety
    TechCareer.AI_ETHICS_RESEARCHER: [TechDomain.AI_ETHICS, TechDomain.AI_SAFETY],
    TechCareer.AI_SAFETY_ENGINEER: [TechDomain.AI_SAFETY, TechDomain.MACHINE_LEARNING],
    TechCareer.RESPONSIBLE_AI_ENGINEER: [TechDomain.AI_ETHICS, TechDomain.MACHINE_LEARNING],
    TechCareer.AI_FAIRNESS_ENGINEER: [TechDomain.AI_ETHICS, TechDomain.MACHINE_LEARNING],
    TechCareer.AI_BIAS_ANALYST: [TechDomain.AI_ETHICS, TechDomain.DATA_ANALYTICS],

    # AI Product
    TechCareer.AI_PRODUCT_MANAGER: [TechDomain.MACHINE_LEARNING, TechDomain.GENERATIVE_AI],
    TechCareer.ML_PRODUCT_MANAGER: [TechDomain.MACHINE_LEARNING, TechDomain.MLOPS],
    TechCareer.AI_SOLUTIONS_ARCHITECT: [TechDomain.MACHINE_LEARNING, TechDomain.SOLUTIONS_ARCHITECTURE],

    # AI Leadership
    TechCareer.HEAD_OF_AI: [TechDomain.MACHINE_LEARNING, TechDomain.DEEP_LEARNING],
    TechCareer.HEAD_OF_ML: [TechDomain.MACHINE_LEARNING, TechDomain.MLOPS],
    TechCareer.VP_AI: [TechDomain.MACHINE_LEARNING, TechDomain.DEEP_LEARNING],
    TechCareer.DIRECTOR_AI: [TechDomain.MACHINE_LEARNING, TechDomain.DEEP_LEARNING],
    TechCareer.DIRECTOR_ML: [TechDomain.MACHINE_LEARNING, TechDomain.MLOPS],
    TechCareer.CHIEF_AI_OFFICER: [TechDomain.MACHINE_LEARNING, TechDomain.GENERATIVE_AI],

    # ═══════════════════════════════════════════════════════════════
    # DEVOPS / PLATFORM / SRE
    # ═══════════════════════════════════════════════════════════════

    # DevOps
    TechCareer.DEVOPS_ENGINEER: [TechDomain.DEVOPS, TechDomain.CICD],
    TechCareer.JUNIOR_DEVOPS_ENGINEER: [TechDomain.DEVOPS, TechDomain.CICD],
    TechCareer.SENIOR_DEVOPS_ENGINEER: [TechDomain.DEVOPS, TechDomain.CONTAINERS],
    TechCareer.LEAD_DEVOPS_ENGINEER: [TechDomain.DEVOPS, TechDomain.PLATFORM_ENGINEERING],
    TechCareer.PRINCIPAL_DEVOPS_ENGINEER: [TechDomain.DEVOPS, TechDomain.INFRASTRUCTURE_AS_CODE],
    TechCareer.STAFF_DEVOPS_ENGINEER: [TechDomain.DEVOPS, TechDomain.INFRASTRUCTURE_AS_CODE],
    TechCareer.DEVOPS_ARCHITECT: [TechDomain.DEVOPS, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.DEVOPS_CONSULTANT: [TechDomain.DEVOPS, TechDomain.CICD],
    TechCareer.DEVOPS_MANAGER: [TechDomain.DEVOPS, TechDomain.PLATFORM_ENGINEERING],

    # Platform Engineering
    TechCareer.PLATFORM_ENGINEER: [TechDomain.PLATFORM_ENGINEERING, TechDomain.KUBERNETES],
    TechCareer.SENIOR_PLATFORM_ENGINEER: [TechDomain.PLATFORM_ENGINEERING, TechDomain.KUBERNETES],
    TechCareer.LEAD_PLATFORM_ENGINEER: [TechDomain.PLATFORM_ENGINEERING, TechDomain.INFRASTRUCTURE_AS_CODE],
    TechCareer.PRINCIPAL_PLATFORM_ENGINEER: [TechDomain.PLATFORM_ENGINEERING, TechDomain.DISTRIBUTED_SYSTEMS],
    TechCareer.PLATFORM_ARCHITECT: [TechDomain.PLATFORM_ENGINEERING, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.INTERNAL_PLATFORM_ENGINEER: [TechDomain.PLATFORM_ENGINEERING, TechDomain.DEVOPS],
    TechCareer.DEVELOPER_PLATFORM_ENGINEER: [TechDomain.PLATFORM_ENGINEERING, TechDomain.DEVOPS],

    # Site Reliability Engineering
    TechCareer.SRE: [TechDomain.SITE_RELIABILITY, TechDomain.OBSERVABILITY],
    TechCareer.JUNIOR_SRE: [TechDomain.SITE_RELIABILITY, TechDomain.MONITORING],
    TechCareer.SENIOR_SRE: [TechDomain.SITE_RELIABILITY, TechDomain.OBSERVABILITY],
    TechCareer.LEAD_SRE: [TechDomain.SITE_RELIABILITY, TechDomain.DISTRIBUTED_SYSTEMS],
    TechCareer.PRINCIPAL_SRE: [TechDomain.SITE_RELIABILITY, TechDomain.DISTRIBUTED_SYSTEMS],
    TechCareer.STAFF_SRE: [TechDomain.SITE_RELIABILITY, TechDomain.OBSERVABILITY],
    TechCareer.SRE_MANAGER: [TechDomain.SITE_RELIABILITY, TechDomain.OBSERVABILITY],
    TechCareer.SRE_ARCHITECT: [TechDomain.SITE_RELIABILITY, TechDomain.SOLUTIONS_ARCHITECTURE],

    # Infrastructure
    TechCareer.INFRASTRUCTURE_ENGINEER: [TechDomain.INFRASTRUCTURE, TechDomain.INFRASTRUCTURE_AS_CODE],
    TechCareer.INFRASTRUCTURE_DEVELOPER: [TechDomain.INFRASTRUCTURE_AS_CODE, TechDomain.INFRASTRUCTURE],
    TechCareer.INFRASTRUCTURE_ARCHITECT: [TechDomain.INFRASTRUCTURE, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.IAC_ENGINEER: [TechDomain.INFRASTRUCTURE_AS_CODE, TechDomain.DEVOPS],

    # Release Engineering
    TechCareer.RELEASE_ENGINEER: [TechDomain.CICD, TechDomain.DEVOPS],
    TechCareer.RELEASE_MANAGER: [TechDomain.CICD, TechDomain.DEVOPS],
    TechCareer.BUILD_ENGINEER: [TechDomain.CICD, TechDomain.DEVOPS],
    TechCareer.BUILD_RELEASE_ENGINEER: [TechDomain.CICD, TechDomain.DEVOPS],

    # CI/CD
    TechCareer.CICD_ENGINEER: [TechDomain.CICD, TechDomain.DEVOPS],
    TechCareer.PIPELINE_ENGINEER: [TechDomain.CICD, TechDomain.DEVOPS],
    TechCareer.AUTOMATION_ENGINEER: [TechDomain.CICD, TechDomain.DEVOPS],
    TechCareer.DEPLOYMENT_ENGINEER: [TechDomain.CICD, TechDomain.DEVOPS],

    # Containers & Orchestration
    TechCareer.KUBERNETES_ENGINEER: [TechDomain.KUBERNETES, TechDomain.CONTAINERS],
    TechCareer.KUBERNETES_ADMINISTRATOR: [TechDomain.KUBERNETES, TechDomain.CONTAINERS],
    TechCareer.KUBERNETES_ARCHITECT: [TechDomain.KUBERNETES, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.CONTAINER_ENGINEER: [TechDomain.CONTAINERS, TechDomain.DEVOPS],
    TechCareer.DOCKER_ENGINEER: [TechDomain.CONTAINERS, TechDomain.DEVOPS],
    TechCareer.OPENSHIFT_ENGINEER: [TechDomain.KUBERNETES, TechDomain.CONTAINERS],

    # Monitoring & Observability
    TechCareer.OBSERVABILITY_ENGINEER: [TechDomain.OBSERVABILITY, TechDomain.MONITORING],
    TechCareer.MONITORING_ENGINEER: [TechDomain.MONITORING, TechDomain.OBSERVABILITY],
    TechCareer.LOGGING_ENGINEER: [TechDomain.LOGGING, TechDomain.OBSERVABILITY],
    TechCareer.APM_ENGINEER: [TechDomain.OBSERVABILITY, TechDomain.MONITORING],

    # Configuration Management
    TechCareer.CONFIGURATION_MANAGER: [TechDomain.INFRASTRUCTURE_AS_CODE, TechDomain.DEVOPS],
    TechCareer.ANSIBLE_ENGINEER: [TechDomain.INFRASTRUCTURE_AS_CODE, TechDomain.DEVOPS],
    TechCareer.PUPPET_ENGINEER: [TechDomain.INFRASTRUCTURE_AS_CODE, TechDomain.DEVOPS],
    TechCareer.CHEF_ENGINEER: [TechDomain.INFRASTRUCTURE_AS_CODE, TechDomain.DEVOPS],
    TechCareer.SALTSTACK_ENGINEER: [TechDomain.INFRASTRUCTURE_AS_CODE, TechDomain.DEVOPS],

    # DevOps Leadership
    TechCareer.HEAD_OF_DEVOPS: [TechDomain.DEVOPS, TechDomain.PLATFORM_ENGINEERING],
    TechCareer.HEAD_OF_PLATFORM: [TechDomain.PLATFORM_ENGINEERING, TechDomain.DEVOPS],
    TechCareer.HEAD_OF_SRE: [TechDomain.SITE_RELIABILITY, TechDomain.DEVOPS],
    TechCareer.DIRECTOR_DEVOPS: [TechDomain.DEVOPS, TechDomain.PLATFORM_ENGINEERING],
    TechCareer.DIRECTOR_PLATFORM: [TechDomain.PLATFORM_ENGINEERING, TechDomain.DEVOPS],
    TechCareer.DIRECTOR_SRE: [TechDomain.SITE_RELIABILITY, TechDomain.OBSERVABILITY],

    # ═══════════════════════════════════════════════════════════════
    # CLOUD
    # ═══════════════════════════════════════════════════════════════

    # General Cloud
    TechCareer.CLOUD_ENGINEER: [TechDomain.CLOUD_COMPUTING, TechDomain.INFRASTRUCTURE],
    TechCareer.JUNIOR_CLOUD_ENGINEER: [TechDomain.CLOUD_COMPUTING, TechDomain.INFRASTRUCTURE],
    TechCareer.SENIOR_CLOUD_ENGINEER: [TechDomain.CLOUD_COMPUTING, TechDomain.INFRASTRUCTURE_AS_CODE],
    TechCareer.LEAD_CLOUD_ENGINEER: [TechDomain.CLOUD_COMPUTING, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.PRINCIPAL_CLOUD_ENGINEER: [TechDomain.CLOUD_COMPUTING, TechDomain.ENTERPRISE_ARCHITECTURE],
    TechCareer.CLOUD_ARCHITECT: [TechDomain.CLOUD_COMPUTING, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.SENIOR_CLOUD_ARCHITECT: [TechDomain.CLOUD_COMPUTING, TechDomain.ENTERPRISE_ARCHITECTURE],
    TechCareer.ENTERPRISE_CLOUD_ARCHITECT: [TechDomain.CLOUD_COMPUTING, TechDomain.ENTERPRISE_ARCHITECTURE],
    TechCareer.CLOUD_CONSULTANT: [TechDomain.CLOUD_COMPUTING, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.CLOUD_SOLUTIONS_ARCHITECT: [TechDomain.CLOUD_COMPUTING, TechDomain.SOLUTIONS_ARCHITECTURE],

    # AWS
    TechCareer.AWS_ENGINEER: [TechDomain.CLOUD_AWS, TechDomain.CLOUD_COMPUTING],
    TechCareer.AWS_DEVELOPER: [TechDomain.CLOUD_AWS, TechDomain.SERVERLESS],
    TechCareer.AWS_ARCHITECT: [TechDomain.CLOUD_AWS, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.AWS_SOLUTIONS_ARCHITECT: [TechDomain.CLOUD_AWS, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.AWS_DEVOPS_ENGINEER: [TechDomain.CLOUD_AWS, TechDomain.DEVOPS],
    TechCareer.AWS_SYSOPS_ADMINISTRATOR: [TechDomain.CLOUD_AWS, TechDomain.SYSTEM_ADMINISTRATION],
    TechCareer.AWS_DATA_ENGINEER: [TechDomain.CLOUD_AWS, TechDomain.DATA_ENGINEERING],
    TechCareer.AWS_ML_SPECIALIST: [TechDomain.CLOUD_AWS, TechDomain.MACHINE_LEARNING],
    TechCareer.AWS_SECURITY_SPECIALIST: [TechDomain.CLOUD_AWS, TechDomain.CLOUD_SECURITY],
    TechCareer.AWS_NETWORKING_SPECIALIST: [TechDomain.CLOUD_AWS, TechDomain.NETWORKING],

    # Azure
    TechCareer.AZURE_ENGINEER: [TechDomain.CLOUD_AZURE, TechDomain.CLOUD_COMPUTING],
    TechCareer.AZURE_DEVELOPER: [TechDomain.CLOUD_AZURE, TechDomain.SERVERLESS],
    TechCareer.AZURE_ARCHITECT: [TechDomain.CLOUD_AZURE, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.AZURE_SOLUTIONS_ARCHITECT: [TechDomain.CLOUD_AZURE, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.AZURE_DEVOPS_ENGINEER: [TechDomain.CLOUD_AZURE, TechDomain.DEVOPS],
    TechCareer.AZURE_ADMINISTRATOR: [TechDomain.CLOUD_AZURE, TechDomain.SYSTEM_ADMINISTRATION],
    TechCareer.AZURE_DATA_ENGINEER: [TechDomain.CLOUD_AZURE, TechDomain.DATA_ENGINEERING],
    TechCareer.AZURE_AI_ENGINEER: [TechDomain.CLOUD_AZURE, TechDomain.MACHINE_LEARNING],
    TechCareer.AZURE_SECURITY_ENGINEER: [TechDomain.CLOUD_AZURE, TechDomain.CLOUD_SECURITY],

    # GCP
    TechCareer.GCP_ENGINEER: [TechDomain.CLOUD_GCP, TechDomain.CLOUD_COMPUTING],
    TechCareer.GCP_DEVELOPER: [TechDomain.CLOUD_GCP, TechDomain.SERVERLESS],
    TechCareer.GCP_ARCHITECT: [TechDomain.CLOUD_GCP, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.GCP_CLOUD_ARCHITECT: [TechDomain.CLOUD_GCP, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.GCP_DEVOPS_ENGINEER: [TechDomain.CLOUD_GCP, TechDomain.DEVOPS],
    TechCareer.GCP_DATA_ENGINEER: [TechDomain.CLOUD_GCP, TechDomain.DATA_ENGINEERING],
    TechCareer.GCP_ML_ENGINEER: [TechDomain.CLOUD_GCP, TechDomain.MACHINE_LEARNING],
    TechCareer.GCP_SECURITY_ENGINEER: [TechDomain.CLOUD_GCP, TechDomain.CLOUD_SECURITY],

    # Multi-Cloud
    TechCareer.MULTI_CLOUD_ARCHITECT: [TechDomain.MULTI_CLOUD, TechDomain.CLOUD_COMPUTING],
    TechCareer.MULTI_CLOUD_ENGINEER: [TechDomain.MULTI_CLOUD, TechDomain.CLOUD_COMPUTING],
    TechCareer.HYBRID_CLOUD_ARCHITECT: [TechDomain.HYBRID_CLOUD, TechDomain.CLOUD_COMPUTING],
    TechCareer.HYBRID_CLOUD_ENGINEER: [TechDomain.HYBRID_CLOUD, TechDomain.CLOUD_COMPUTING],

    # Serverless
    TechCareer.SERVERLESS_ENGINEER: [TechDomain.SERVERLESS, TechDomain.CLOUD_NATIVE],
    TechCareer.SERVERLESS_ARCHITECT: [TechDomain.SERVERLESS, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.LAMBDA_DEVELOPER: [TechDomain.SERVERLESS, TechDomain.CLOUD_AWS],
    TechCareer.FUNCTIONS_DEVELOPER: [TechDomain.SERVERLESS, TechDomain.CLOUD_COMPUTING],

    # Cloud Native
    TechCareer.CLOUD_NATIVE_ENGINEER: [TechDomain.CLOUD_NATIVE, TechDomain.KUBERNETES],
    TechCareer.CLOUD_NATIVE_ARCHITECT: [TechDomain.CLOUD_NATIVE, TechDomain.MICROSERVICES],
    TechCareer.CLOUD_NATIVE_DEVELOPER: [TechDomain.CLOUD_NATIVE, TechDomain.CONTAINERS],

    # Cloud FinOps
    TechCareer.FINOPS_ENGINEER: [TechDomain.FINOPS, TechDomain.CLOUD_COMPUTING],
    TechCareer.FINOPS_ANALYST: [TechDomain.FINOPS, TechDomain.CLOUD_COMPUTING],
    TechCareer.FINOPS_ARCHITECT: [TechDomain.FINOPS, TechDomain.CLOUD_COMPUTING],
    TechCareer.CLOUD_COST_ANALYST: [TechDomain.FINOPS, TechDomain.CLOUD_COMPUTING],
    TechCareer.CLOUD_ECONOMIST: [TechDomain.FINOPS, TechDomain.CLOUD_COMPUTING],

    # Cloud Migration
    TechCareer.CLOUD_MIGRATION_ENGINEER: [TechDomain.CLOUD_MIGRATION, TechDomain.CLOUD_COMPUTING],
    TechCareer.CLOUD_MIGRATION_ARCHITECT: [TechDomain.CLOUD_MIGRATION, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.CLOUD_MIGRATION_SPECIALIST: [TechDomain.CLOUD_MIGRATION, TechDomain.CLOUD_COMPUTING],

    # Cloud Leadership
    TechCareer.HEAD_OF_CLOUD: [TechDomain.CLOUD_COMPUTING, TechDomain.ENTERPRISE_ARCHITECTURE],
    TechCareer.VP_CLOUD: [TechDomain.CLOUD_COMPUTING, TechDomain.ENTERPRISE_ARCHITECTURE],
    TechCareer.DIRECTOR_CLOUD: [TechDomain.CLOUD_COMPUTING, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.CLOUD_PRACTICE_LEAD: [TechDomain.CLOUD_COMPUTING, TechDomain.SOLUTIONS_ARCHITECTURE],

    # ═══════════════════════════════════════════════════════════════
    # SECURITY / CYBERSECURITY
    # ═══════════════════════════════════════════════════════════════

    # Security Engineering
    TechCareer.SECURITY_ENGINEER: [TechDomain.CYBERSECURITY, TechDomain.APPLICATION_SECURITY],
    TechCareer.JUNIOR_SECURITY_ENGINEER: [TechDomain.CYBERSECURITY, TechDomain.NETWORK_SECURITY],
    TechCareer.SENIOR_SECURITY_ENGINEER: [TechDomain.CYBERSECURITY, TechDomain.APPLICATION_SECURITY],
    TechCareer.LEAD_SECURITY_ENGINEER: [TechDomain.CYBERSECURITY, TechDomain.CLOUD_SECURITY],
    TechCareer.PRINCIPAL_SECURITY_ENGINEER: [TechDomain.CYBERSECURITY, TechDomain.CRYPTOGRAPHY],
    TechCareer.STAFF_SECURITY_ENGINEER: [TechDomain.CYBERSECURITY, TechDomain.APPLICATION_SECURITY],

    # Security Architecture
    TechCareer.SECURITY_ARCHITECT: [TechDomain.CYBERSECURITY, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.SENIOR_SECURITY_ARCHITECT: [TechDomain.CYBERSECURITY, TechDomain.ENTERPRISE_ARCHITECTURE],
    TechCareer.ENTERPRISE_SECURITY_ARCHITECT: [TechDomain.CYBERSECURITY, TechDomain.ENTERPRISE_ARCHITECTURE],
    TechCareer.CLOUD_SECURITY_ARCHITECT: [TechDomain.CLOUD_SECURITY, TechDomain.CYBERSECURITY],
    TechCareer.APPLICATION_SECURITY_ARCHITECT: [TechDomain.APPLICATION_SECURITY, TechDomain.CYBERSECURITY],

    # Application Security
    TechCareer.APPSEC_ENGINEER: [TechDomain.APPLICATION_SECURITY, TechDomain.SECURE_CODING],
    TechCareer.APPSEC_ANALYST: [TechDomain.APPLICATION_SECURITY, TechDomain.SECURE_CODING],
    TechCareer.SECURE_CODE_REVIEWER: [TechDomain.SECURE_CODING, TechDomain.APPLICATION_SECURITY],
    TechCareer.SAST_ENGINEER: [TechDomain.APPLICATION_SECURITY, TechDomain.SECURE_CODING],
    TechCareer.DAST_ENGINEER: [TechDomain.APPLICATION_SECURITY, TechDomain.PENETRATION_TESTING],

    # DevSecOps
    TechCareer.DEVSECOPS_ENGINEER: [TechDomain.DEVSECOPS, TechDomain.DEVOPS],
    TechCareer.SENIOR_DEVSECOPS_ENGINEER: [TechDomain.DEVSECOPS, TechDomain.APPLICATION_SECURITY],
    TechCareer.DEVSECOPS_ARCHITECT: [TechDomain.DEVSECOPS, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.SECURITY_AUTOMATION_ENGINEER: [TechDomain.DEVSECOPS, TechDomain.CICD],

    # Penetration Testing
    TechCareer.PENETRATION_TESTER: [TechDomain.PENETRATION_TESTING, TechDomain.OFFENSIVE_SECURITY],
    TechCareer.JUNIOR_PENTESTER: [TechDomain.PENETRATION_TESTING, TechDomain.NETWORK_SECURITY],
    TechCareer.SENIOR_PENTESTER: [TechDomain.PENETRATION_TESTING, TechDomain.OFFENSIVE_SECURITY],
    TechCareer.LEAD_PENTESTER: [TechDomain.PENETRATION_TESTING, TechDomain.OFFENSIVE_SECURITY],
    TechCareer.RED_TEAM_OPERATOR: [TechDomain.OFFENSIVE_SECURITY, TechDomain.PENETRATION_TESTING],
    TechCareer.RED_TEAM_LEAD: [TechDomain.OFFENSIVE_SECURITY, TechDomain.PENETRATION_TESTING],

    # Offensive Security
    TechCareer.OFFENSIVE_SECURITY_ENGINEER: [TechDomain.OFFENSIVE_SECURITY, TechDomain.PENETRATION_TESTING],
    TechCareer.EXPLOIT_DEVELOPER: [TechDomain.OFFENSIVE_SECURITY, TechDomain.SYSTEMS_PROGRAMMING],
    TechCareer.VULNERABILITY_RESEARCHER: [TechDomain.OFFENSIVE_SECURITY, TechDomain.CYBERSECURITY],
    TechCareer.BUG_BOUNTY_HUNTER: [TechDomain.OFFENSIVE_SECURITY, TechDomain.PENETRATION_TESTING],

    # Defensive Security
    TechCareer.DEFENSIVE_SECURITY_ENGINEER: [TechDomain.DEFENSIVE_SECURITY, TechDomain.INCIDENT_RESPONSE],
    TechCareer.BLUE_TEAM_ANALYST: [TechDomain.DEFENSIVE_SECURITY, TechDomain.THREAT_INTELLIGENCE],
    TechCareer.BLUE_TEAM_ENGINEER: [TechDomain.DEFENSIVE_SECURITY, TechDomain.INCIDENT_RESPONSE],
    TechCareer.PURPLE_TEAM_ENGINEER: [TechDomain.DEFENSIVE_SECURITY, TechDomain.OFFENSIVE_SECURITY],

    # Security Operations
    TechCareer.SOC_ANALYST: [TechDomain.DEFENSIVE_SECURITY, TechDomain.INCIDENT_RESPONSE],
    TechCareer.SOC_ANALYST_L1: [TechDomain.DEFENSIVE_SECURITY, TechDomain.MONITORING],
    TechCareer.SOC_ANALYST_L2: [TechDomain.DEFENSIVE_SECURITY, TechDomain.THREAT_INTELLIGENCE],
    TechCareer.SOC_ANALYST_L3: [TechDomain.DEFENSIVE_SECURITY, TechDomain.INCIDENT_RESPONSE],
    TechCareer.SOC_ENGINEER: [TechDomain.DEFENSIVE_SECURITY, TechDomain.INCIDENT_RESPONSE],
    TechCareer.SOC_MANAGER: [TechDomain.DEFENSIVE_SECURITY, TechDomain.INCIDENT_RESPONSE],

    # Incident Response
    TechCareer.INCIDENT_RESPONDER: [TechDomain.INCIDENT_RESPONSE, TechDomain.DIGITAL_FORENSICS],
    TechCareer.IR_ANALYST: [TechDomain.INCIDENT_RESPONSE, TechDomain.THREAT_INTELLIGENCE],
    TechCareer.IR_ENGINEER: [TechDomain.INCIDENT_RESPONSE, TechDomain.DEFENSIVE_SECURITY],
    TechCareer.IR_MANAGER: [TechDomain.INCIDENT_RESPONSE, TechDomain.DEFENSIVE_SECURITY],

    # Threat Intelligence
    TechCareer.THREAT_INTEL_ANALYST: [TechDomain.THREAT_INTELLIGENCE, TechDomain.DEFENSIVE_SECURITY],
    TechCareer.THREAT_INTEL_ENGINEER: [TechDomain.THREAT_INTELLIGENCE, TechDomain.DEFENSIVE_SECURITY],
    TechCareer.THREAT_HUNTER: [TechDomain.THREAT_INTELLIGENCE, TechDomain.DEFENSIVE_SECURITY],
    TechCareer.THREAT_RESEARCHER: [TechDomain.THREAT_INTELLIGENCE, TechDomain.OFFENSIVE_SECURITY],

    # Digital Forensics
    TechCareer.FORENSICS_ANALYST: [TechDomain.DIGITAL_FORENSICS, TechDomain.INCIDENT_RESPONSE],
    TechCareer.FORENSICS_ENGINEER: [TechDomain.DIGITAL_FORENSICS, TechDomain.INCIDENT_RESPONSE],
    TechCareer.MALWARE_ANALYST: [TechDomain.DIGITAL_FORENSICS, TechDomain.OFFENSIVE_SECURITY],
    TechCareer.MALWARE_REVERSE_ENGINEER: [TechDomain.DIGITAL_FORENSICS, TechDomain.SYSTEMS_PROGRAMMING],

    # Cryptography
    TechCareer.CRYPTOGRAPHER: [TechDomain.CRYPTOGRAPHY, TechDomain.CYBERSECURITY],
    TechCareer.CRYPTOGRAPHY_ENGINEER: [TechDomain.CRYPTOGRAPHY, TechDomain.CYBERSECURITY],
    TechCareer.PKI_ENGINEER: [TechDomain.CRYPTOGRAPHY, TechDomain.IAM],

    # Identity & Access
    TechCareer.IAM_ENGINEER: [TechDomain.IAM, TechDomain.CYBERSECURITY],
    TechCareer.IAM_ARCHITECT: [TechDomain.IAM, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.IAM_ANALYST: [TechDomain.IAM, TechDomain.CYBERSECURITY],
    TechCareer.IDENTITY_ENGINEER: [TechDomain.IAM, TechDomain.CYBERSECURITY],
    TechCareer.ACCESS_MANAGEMENT_ENGINEER: [TechDomain.IAM, TechDomain.CYBERSECURITY],

    # Network Security
    TechCareer.NETWORK_SECURITY_ENGINEER: [TechDomain.NETWORK_SECURITY, TechDomain.NETWORKING],
    TechCareer.FIREWALL_ENGINEER: [TechDomain.NETWORK_SECURITY, TechDomain.NETWORKING],
    TechCareer.FIREWALL_ADMINISTRATOR: [TechDomain.NETWORK_SECURITY, TechDomain.NETWORKING],

    # GRC
    TechCareer.GRC_ANALYST: [TechDomain.GRC, TechDomain.CYBERSECURITY],
    TechCareer.GRC_ENGINEER: [TechDomain.GRC, TechDomain.CYBERSECURITY],
    TechCareer.GRC_MANAGER: [TechDomain.GRC, TechDomain.CYBERSECURITY],
    TechCareer.COMPLIANCE_ANALYST: [TechDomain.GRC, TechDomain.CYBERSECURITY],
    TechCareer.COMPLIANCE_ENGINEER: [TechDomain.GRC, TechDomain.CYBERSECURITY],
    TechCareer.RISK_ANALYST: [TechDomain.GRC, TechDomain.CYBERSECURITY],
    TechCareer.IT_AUDITOR: [TechDomain.GRC, TechDomain.CYBERSECURITY],

    # Security Leadership
    TechCareer.HEAD_OF_SECURITY: [TechDomain.CYBERSECURITY, TechDomain.GRC],
    TechCareer.VP_SECURITY: [TechDomain.CYBERSECURITY, TechDomain.GRC],
    TechCareer.DIRECTOR_SECURITY: [TechDomain.CYBERSECURITY, TechDomain.CLOUD_SECURITY],
    TechCareer.CISO: [TechDomain.CYBERSECURITY, TechDomain.GRC],
    TechCareer.DEPUTY_CISO: [TechDomain.CYBERSECURITY, TechDomain.GRC],

    # ═══════════════════════════════════════════════════════════════
    # DATABASES
    # ═══════════════════════════════════════════════════════════════

    TechCareer.DBA: [TechDomain.DATABASES, TechDomain.DATABASE_ADMINISTRATION],
    TechCareer.JUNIOR_DBA: [TechDomain.DATABASES, TechDomain.DATABASE_ADMINISTRATION],
    TechCareer.SENIOR_DBA: [TechDomain.DATABASES, TechDomain.DATABASE_OPTIMIZATION],
    TechCareer.LEAD_DBA: [TechDomain.DATABASES, TechDomain.DATABASE_OPTIMIZATION],
    TechCareer.PRINCIPAL_DBA: [TechDomain.DATABASES, TechDomain.DATABASE_OPTIMIZATION],
    TechCareer.POSTGRESQL_DBA: [TechDomain.RELATIONAL_DATABASES, TechDomain.DATABASE_ADMINISTRATION],
    TechCareer.MYSQL_DBA: [TechDomain.RELATIONAL_DATABASES, TechDomain.DATABASE_ADMINISTRATION],
    TechCareer.ORACLE_DBA: [TechDomain.RELATIONAL_DATABASES, TechDomain.DATABASE_ADMINISTRATION],
    TechCareer.SQL_SERVER_DBA: [TechDomain.RELATIONAL_DATABASES, TechDomain.DATABASE_ADMINISTRATION],
    TechCareer.MONGODB_DBA: [TechDomain.NOSQL_DATABASES, TechDomain.DATABASE_ADMINISTRATION],
    TechCareer.CASSANDRA_DBA: [TechDomain.NOSQL_DATABASES, TechDomain.DATABASE_ADMINISTRATION],
    TechCareer.REDIS_ENGINEER: [TechDomain.NOSQL_DATABASES, TechDomain.DATABASES],
    TechCareer.ELASTICSEARCH_ENGINEER: [TechDomain.NOSQL_DATABASES, TechDomain.DATABASES],
    TechCareer.DATABASE_DEVELOPER: [TechDomain.DATABASES, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.SQL_DEVELOPER: [TechDomain.RELATIONAL_DATABASES, TechDomain.DATABASES],
    TechCareer.PLSQL_DEVELOPER: [TechDomain.RELATIONAL_DATABASES, TechDomain.DATABASES],
    TechCareer.TSQL_DEVELOPER: [TechDomain.RELATIONAL_DATABASES, TechDomain.DATABASES],
    TechCareer.DATABASE_ARCHITECT: [TechDomain.DATABASES, TechDomain.DATA_ARCHITECTURE],
    TechCareer.DATABASE_SOLUTIONS_ARCHITECT: [TechDomain.DATABASES, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.DBRE: [TechDomain.DATABASES, TechDomain.SITE_RELIABILITY],
    TechCareer.DATABASE_SRE: [TechDomain.DATABASES, TechDomain.SITE_RELIABILITY],

    # ═══════════════════════════════════════════════════════════════
    # NETWORKING
    # ═══════════════════════════════════════════════════════════════

    TechCareer.NETWORK_ENGINEER: [TechDomain.NETWORKING, TechDomain.NETWORK_ARCHITECTURE],
    TechCareer.JUNIOR_NETWORK_ENGINEER: [TechDomain.NETWORKING, TechDomain.NETWORK_ARCHITECTURE],
    TechCareer.SENIOR_NETWORK_ENGINEER: [TechDomain.NETWORKING, TechDomain.NETWORK_ARCHITECTURE],
    TechCareer.LEAD_NETWORK_ENGINEER: [TechDomain.NETWORKING, TechDomain.NETWORK_ARCHITECTURE],
    TechCareer.PRINCIPAL_NETWORK_ENGINEER: [TechDomain.NETWORKING, TechDomain.SDN],
    TechCareer.NETWORK_ARCHITECT: [TechDomain.NETWORK_ARCHITECTURE, TechDomain.NETWORKING],
    TechCareer.SENIOR_NETWORK_ARCHITECT: [TechDomain.NETWORK_ARCHITECTURE, TechDomain.ENTERPRISE_ARCHITECTURE],
    TechCareer.ENTERPRISE_NETWORK_ARCHITECT: [TechDomain.NETWORK_ARCHITECTURE, TechDomain.ENTERPRISE_ARCHITECTURE],
    TechCareer.NETWORK_ADMINISTRATOR: [TechDomain.NETWORKING, TechDomain.SYSTEM_ADMINISTRATION],
    TechCareer.WIRELESS_ENGINEER: [TechDomain.WIRELESS, TechDomain.NETWORKING],
    TechCareer.WAN_ENGINEER: [TechDomain.NETWORKING, TechDomain.SD_WAN],
    TechCareer.LAN_ENGINEER: [TechDomain.NETWORKING, TechDomain.NETWORK_ARCHITECTURE],
    TechCareer.SD_WAN_ENGINEER: [TechDomain.SD_WAN, TechDomain.NETWORKING],
    TechCareer.VOIP_ENGINEER: [TechDomain.NETWORKING, TechDomain.WIRELESS],
    TechCareer.UC_ENGINEER: [TechDomain.NETWORKING, TechDomain.WIRELESS],
    TechCareer.LOAD_BALANCER_ENGINEER: [TechDomain.NETWORKING, TechDomain.INFRASTRUCTURE],
    TechCareer.CDN_ENGINEER: [TechDomain.NETWORKING, TechDomain.INFRASTRUCTURE],
    TechCareer.DNS_ENGINEER: [TechDomain.NETWORKING, TechDomain.INFRASTRUCTURE],
    TechCareer.NETWORK_AUTOMATION_ENGINEER: [TechDomain.NETWORK_AUTOMATION, TechDomain.DEVOPS],
    TechCareer.NETDEVOPS_ENGINEER: [TechDomain.NETWORK_AUTOMATION, TechDomain.DEVOPS],

    # ═══════════════════════════════════════════════════════════════
    # SYSTEMS ADMINISTRATION
    # ═══════════════════════════════════════════════════════════════

    TechCareer.SYSADMIN: [TechDomain.SYSTEM_ADMINISTRATION, TechDomain.LINUX],
    TechCareer.JUNIOR_SYSADMIN: [TechDomain.SYSTEM_ADMINISTRATION, TechDomain.LINUX],
    TechCareer.SENIOR_SYSADMIN: [TechDomain.SYSTEM_ADMINISTRATION, TechDomain.LINUX],
    TechCareer.LEAD_SYSADMIN: [TechDomain.SYSTEM_ADMINISTRATION, TechDomain.INFRASTRUCTURE],
    TechCareer.LINUX_ADMINISTRATOR: [TechDomain.LINUX, TechDomain.SYSTEM_ADMINISTRATION],
    TechCareer.LINUX_ENGINEER: [TechDomain.LINUX, TechDomain.SYSTEM_ADMINISTRATION],
    TechCareer.LINUX_SYSTEMS_ENGINEER: [TechDomain.LINUX, TechDomain.INFRASTRUCTURE],
    TechCareer.WINDOWS_ADMINISTRATOR: [TechDomain.WINDOWS_SERVER, TechDomain.SYSTEM_ADMINISTRATION],
    TechCareer.WINDOWS_ENGINEER: [TechDomain.WINDOWS_SERVER, TechDomain.SYSTEM_ADMINISTRATION],
    TechCareer.WINDOWS_SYSTEMS_ENGINEER: [TechDomain.WINDOWS_SERVER, TechDomain.INFRASTRUCTURE],
    TechCareer.AD_ENGINEER: [TechDomain.WINDOWS_SERVER, TechDomain.IAM],
    TechCareer.UNIX_ADMINISTRATOR: [TechDomain.UNIX, TechDomain.SYSTEM_ADMINISTRATION],
    TechCareer.UNIX_ENGINEER: [TechDomain.UNIX, TechDomain.SYSTEM_ADMINISTRATION],
    TechCareer.AIX_ADMINISTRATOR: [TechDomain.UNIX, TechDomain.SYSTEM_ADMINISTRATION],
    TechCareer.SOLARIS_ADMINISTRATOR: [TechDomain.UNIX, TechDomain.SYSTEM_ADMINISTRATION],
    TechCareer.VIRTUALIZATION_ENGINEER: [TechDomain.VIRTUALIZATION, TechDomain.INFRASTRUCTURE],
    TechCareer.VMWARE_ENGINEER: [TechDomain.VIRTUALIZATION, TechDomain.INFRASTRUCTURE],
    TechCareer.VMWARE_ADMINISTRATOR: [TechDomain.VIRTUALIZATION, TechDomain.SYSTEM_ADMINISTRATION],
    TechCareer.HYPERV_ADMINISTRATOR: [TechDomain.VIRTUALIZATION, TechDomain.WINDOWS_SERVER],
    TechCareer.STORAGE_ENGINEER: [TechDomain.STORAGE, TechDomain.INFRASTRUCTURE],
    TechCareer.STORAGE_ADMINISTRATOR: [TechDomain.STORAGE, TechDomain.SYSTEM_ADMINISTRATION],
    TechCareer.STORAGE_ARCHITECT: [TechDomain.STORAGE, TechDomain.INFRASTRUCTURE],
    TechCareer.SAN_ENGINEER: [TechDomain.STORAGE, TechDomain.INFRASTRUCTURE],
    TechCareer.NAS_ENGINEER: [TechDomain.STORAGE, TechDomain.INFRASTRUCTURE],
    TechCareer.BACKUP_ENGINEER: [TechDomain.STORAGE, TechDomain.INFRASTRUCTURE],

    # ═══════════════════════════════════════════════════════════════
    # QA / TESTING
    # ═══════════════════════════════════════════════════════════════

    TechCareer.QA_ENGINEER: [TechDomain.SOFTWARE_TESTING, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.JUNIOR_QA_ENGINEER: [TechDomain.SOFTWARE_TESTING, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.SENIOR_QA_ENGINEER: [TechDomain.SOFTWARE_TESTING, TechDomain.TEST_AUTOMATION],
    TechCareer.LEAD_QA_ENGINEER: [TechDomain.SOFTWARE_TESTING, TechDomain.TEST_AUTOMATION],
    TechCareer.PRINCIPAL_QA_ENGINEER: [TechDomain.SOFTWARE_TESTING, TechDomain.TEST_AUTOMATION],
    TechCareer.TEST_ENGINEER: [TechDomain.SOFTWARE_TESTING, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.SOFTWARE_TEST_ENGINEER: [TechDomain.SOFTWARE_TESTING, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.SDET: [TechDomain.SOFTWARE_TESTING, TechDomain.TEST_AUTOMATION],
    TechCareer.AUTOMATION_TEST_ENGINEER: [TechDomain.TEST_AUTOMATION, TechDomain.SOFTWARE_TESTING],
    TechCareer.TEST_AUTOMATION_ENGINEER: [TechDomain.TEST_AUTOMATION, TechDomain.SOFTWARE_TESTING],
    TechCareer.TEST_AUTOMATION_ARCHITECT: [TechDomain.TEST_AUTOMATION, TechDomain.SOFTWARE_ARCHITECTURE],
    TechCareer.SELENIUM_DEVELOPER: [TechDomain.TEST_AUTOMATION, TechDomain.SOFTWARE_TESTING],
    TechCareer.CYPRESS_DEVELOPER: [TechDomain.TEST_AUTOMATION, TechDomain.SOFTWARE_TESTING],
    TechCareer.PLAYWRIGHT_DEVELOPER: [TechDomain.TEST_AUTOMATION, TechDomain.SOFTWARE_TESTING],
    TechCareer.PERFORMANCE_TEST_ENGINEER: [TechDomain.PERFORMANCE_TESTING, TechDomain.SOFTWARE_TESTING],
    TechCareer.PERFORMANCE_ENGINEER: [TechDomain.PERFORMANCE_TESTING, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.LOAD_TEST_ENGINEER: [TechDomain.PERFORMANCE_TESTING, TechDomain.SOFTWARE_TESTING],
    TechCareer.MANUAL_TESTER: [TechDomain.SOFTWARE_TESTING, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.MANUAL_QA: [TechDomain.SOFTWARE_TESTING, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.MOBILE_QA_ENGINEER: [TechDomain.SOFTWARE_TESTING, TechDomain.MOBILE_DEVELOPMENT],
    TechCareer.API_TEST_ENGINEER: [TechDomain.SOFTWARE_TESTING, TechDomain.API_DESIGN],
    TechCareer.SECURITY_TESTER: [TechDomain.SECURITY_TESTING, TechDomain.PENETRATION_TESTING],
    TechCareer.ACCESSIBILITY_TESTER: [TechDomain.SOFTWARE_TESTING, TechDomain.WEB_FRONTEND],
    TechCareer.USABILITY_TESTER: [TechDomain.SOFTWARE_TESTING, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.LOCALIZATION_TESTER: [TechDomain.SOFTWARE_TESTING, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.QA_MANAGER: [TechDomain.SOFTWARE_TESTING, TechDomain.TEST_AUTOMATION],
    TechCareer.QA_LEAD: [TechDomain.SOFTWARE_TESTING, TechDomain.TEST_AUTOMATION],
    TechCareer.TEST_MANAGER: [TechDomain.SOFTWARE_TESTING, TechDomain.TEST_AUTOMATION],
    TechCareer.HEAD_OF_QA: [TechDomain.SOFTWARE_TESTING, TechDomain.TEST_AUTOMATION],
    TechCareer.DIRECTOR_QA: [TechDomain.SOFTWARE_TESTING, TechDomain.TEST_AUTOMATION],
    TechCareer.VP_QA: [TechDomain.SOFTWARE_TESTING, TechDomain.TEST_AUTOMATION],

    # ═══════════════════════════════════════════════════════════════
    # ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════

    TechCareer.SOFTWARE_ARCHITECT: [TechDomain.SOFTWARE_ARCHITECTURE, TechDomain.DESIGN_PATTERNS],
    TechCareer.SENIOR_SOFTWARE_ARCHITECT: [TechDomain.SOFTWARE_ARCHITECTURE, TechDomain.DISTRIBUTED_SYSTEMS],
    TechCareer.PRINCIPAL_ARCHITECT: [TechDomain.SOFTWARE_ARCHITECTURE, TechDomain.ENTERPRISE_ARCHITECTURE],
    TechCareer.CHIEF_ARCHITECT: [TechDomain.ENTERPRISE_ARCHITECTURE, TechDomain.SOFTWARE_ARCHITECTURE],
    TechCareer.ENTERPRISE_ARCHITECT: [TechDomain.ENTERPRISE_ARCHITECTURE, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.SENIOR_ENTERPRISE_ARCHITECT: [TechDomain.ENTERPRISE_ARCHITECTURE, TechDomain.SOFTWARE_ARCHITECTURE],
    TechCareer.CHIEF_ENTERPRISE_ARCHITECT: [TechDomain.ENTERPRISE_ARCHITECTURE, TechDomain.SOFTWARE_ARCHITECTURE],
    TechCareer.TOGAF_ARCHITECT: [TechDomain.ENTERPRISE_ARCHITECTURE, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.SOLUTIONS_ARCHITECT: [TechDomain.SOLUTIONS_ARCHITECTURE, TechDomain.SOFTWARE_ARCHITECTURE],
    TechCareer.SENIOR_SOLUTIONS_ARCHITECT: [TechDomain.SOLUTIONS_ARCHITECTURE, TechDomain.ENTERPRISE_ARCHITECTURE],
    TechCareer.PRINCIPAL_SOLUTIONS_ARCHITECT: [TechDomain.SOLUTIONS_ARCHITECTURE, TechDomain.ENTERPRISE_ARCHITECTURE],
    TechCareer.DOMAIN_ARCHITECT: [TechDomain.SOFTWARE_ARCHITECTURE, TechDomain.DOMAIN_DRIVEN_DESIGN],
    TechCareer.TECHNICAL_ARCHITECT: [TechDomain.SOFTWARE_ARCHITECTURE, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.INTEGRATION_ARCHITECT: [TechDomain.SOFTWARE_ARCHITECTURE, TechDomain.API_DESIGN],
    TechCareer.API_ARCHITECT: [TechDomain.API_DESIGN, TechDomain.SOFTWARE_ARCHITECTURE],
    TechCareer.MICROSERVICES_ARCHITECT: [TechDomain.MICROSERVICES, TechDomain.DISTRIBUTED_SYSTEMS],

    # ═══════════════════════════════════════════════════════════════
    # EMERGING TECH
    # ═══════════════════════════════════════════════════════════════

    # Blockchain
    TechCareer.BLOCKCHAIN_DEVELOPER: [TechDomain.BLOCKCHAIN, TechDomain.SMART_CONTRACTS],
    TechCareer.BLOCKCHAIN_ENGINEER: [TechDomain.BLOCKCHAIN, TechDomain.DISTRIBUTED_SYSTEMS],
    TechCareer.BLOCKCHAIN_ARCHITECT: [TechDomain.BLOCKCHAIN, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.SMART_CONTRACT_DEVELOPER: [TechDomain.SMART_CONTRACTS, TechDomain.BLOCKCHAIN],
    TechCareer.SOLIDITY_DEVELOPER: [TechDomain.SMART_CONTRACTS, TechDomain.BLOCKCHAIN],
    TechCareer.WEB3_DEVELOPER: [TechDomain.WEB3, TechDomain.BLOCKCHAIN],
    TechCareer.DEFI_DEVELOPER: [TechDomain.DEFI, TechDomain.BLOCKCHAIN],
    TechCareer.NFT_DEVELOPER: [TechDomain.WEB3, TechDomain.BLOCKCHAIN],
    TechCareer.CRYPTO_ENGINEER: [TechDomain.BLOCKCHAIN, TechDomain.CRYPTOGRAPHY],

    # Quantum Computing
    TechCareer.QUANTUM_SOFTWARE_ENGINEER: [TechDomain.QUANTUM_COMPUTING, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.QUANTUM_DEVELOPER: [TechDomain.QUANTUM_COMPUTING, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.QUANTUM_ALGORITHM_DEVELOPER: [TechDomain.QUANTUM_COMPUTING, TechDomain.MACHINE_LEARNING],
    TechCareer.QUANTUM_RESEARCHER: [TechDomain.QUANTUM_COMPUTING, TechDomain.DEEP_LEARNING],
    TechCareer.QUANTUM_APPLICATIONS_ENGINEER: [TechDomain.QUANTUM_COMPUTING, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.QISKIT_DEVELOPER: [TechDomain.QUANTUM_COMPUTING, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.CIRQ_DEVELOPER: [TechDomain.QUANTUM_COMPUTING, TechDomain.SOFTWARE_ENGINEERING],

    # IoT
    TechCareer.IOT_ENGINEER: [TechDomain.IOT, TechDomain.EMBEDDED_SYSTEMS],
    TechCareer.IOT_DEVELOPER: [TechDomain.IOT, TechDomain.EMBEDDED_SYSTEMS],
    TechCareer.IOT_ARCHITECT: [TechDomain.IOT, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.IOT_SOLUTIONS_ARCHITECT: [TechDomain.IOT, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.EMBEDDED_IOT_ENGINEER: [TechDomain.IOT, TechDomain.EMBEDDED_SYSTEMS],
    TechCareer.IOT_PLATFORM_ENGINEER: [TechDomain.IOT, TechDomain.PLATFORM_ENGINEERING],

    # AR/VR/XR
    TechCareer.AR_DEVELOPER: [TechDomain.AR_VR, TechDomain.GAME_DEVELOPMENT],
    TechCareer.VR_DEVELOPER: [TechDomain.AR_VR, TechDomain.GAME_DEVELOPMENT],
    TechCareer.XR_DEVELOPER: [TechDomain.AR_VR, TechDomain.GAME_DEVELOPMENT],
    TechCareer.MIXED_REALITY_DEVELOPER: [TechDomain.AR_VR, TechDomain.GAME_DEVELOPMENT],
    TechCareer.METAVERSE_DEVELOPER: [TechDomain.METAVERSE, TechDomain.AR_VR],

    # Edge Computing
    TechCareer.EDGE_COMPUTING_ENGINEER: [TechDomain.EDGE_COMPUTING, TechDomain.IOT],
    TechCareer.EDGE_ARCHITECT: [TechDomain.EDGE_COMPUTING, TechDomain.SOLUTIONS_ARCHITECTURE],
    TechCareer.EDGE_ML_ENGINEER: [TechDomain.EDGE_COMPUTING, TechDomain.MACHINE_LEARNING],

    # Robotics
    TechCareer.ROBOTICS_ENGINEER: [TechDomain.ROBOTICS, TechDomain.EMBEDDED_SYSTEMS],
    TechCareer.ROBOTICS_SOFTWARE_ENGINEER: [TechDomain.ROBOTICS, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.ROS_DEVELOPER: [TechDomain.ROBOTICS, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.AUTONOMOUS_SYSTEMS_ENGINEER: [TechDomain.AUTONOMOUS_SYSTEMS, TechDomain.ROBOTICS],

    # ═══════════════════════════════════════════════════════════════
    # PRODUCT & TECHNICAL MANAGEMENT
    # ═══════════════════════════════════════════════════════════════

    TechCareer.TECHNICAL_PRODUCT_MANAGER: [TechDomain.SOFTWARE_ENGINEERING, TechDomain.AGILE],
    TechCareer.PRODUCT_MANAGER: [TechDomain.SOFTWARE_ENGINEERING, TechDomain.AGILE],
    TechCareer.SENIOR_PRODUCT_MANAGER: [TechDomain.SOFTWARE_ENGINEERING, TechDomain.AGILE],
    TechCareer.PRINCIPAL_PRODUCT_MANAGER: [TechDomain.SOFTWARE_ARCHITECTURE, TechDomain.AGILE],
    TechCareer.GROUP_PRODUCT_MANAGER: [TechDomain.SOFTWARE_ARCHITECTURE, TechDomain.AGILE],
    TechCareer.DIRECTOR_PRODUCT: [TechDomain.SOFTWARE_ARCHITECTURE, TechDomain.ENTERPRISE_ARCHITECTURE],
    TechCareer.VP_PRODUCT: [TechDomain.ENTERPRISE_ARCHITECTURE, TechDomain.SOFTWARE_ARCHITECTURE],
    TechCareer.CPO: [TechDomain.ENTERPRISE_ARCHITECTURE, TechDomain.SOFTWARE_ARCHITECTURE],
    TechCareer.TPM: [TechDomain.SOFTWARE_ENGINEERING, TechDomain.AGILE],
    TechCareer.SENIOR_TPM: [TechDomain.SOFTWARE_ARCHITECTURE, TechDomain.AGILE],
    TechCareer.PRINCIPAL_TPM: [TechDomain.SOFTWARE_ARCHITECTURE, TechDomain.ENTERPRISE_ARCHITECTURE],
    TechCareer.DIRECTOR_TPM: [TechDomain.ENTERPRISE_ARCHITECTURE, TechDomain.SOFTWARE_ARCHITECTURE],
    TechCareer.ENGINEERING_MANAGER: [TechDomain.SOFTWARE_ENGINEERING, TechDomain.AGILE],
    TechCareer.SENIOR_ENGINEERING_MANAGER: [TechDomain.SOFTWARE_ARCHITECTURE, TechDomain.AGILE],
    TechCareer.DIRECTOR_ENGINEERING: [TechDomain.SOFTWARE_ARCHITECTURE, TechDomain.ENTERPRISE_ARCHITECTURE],
    TechCareer.SENIOR_DIRECTOR_ENGINEERING: [TechDomain.ENTERPRISE_ARCHITECTURE, TechDomain.SOFTWARE_ARCHITECTURE],
    TechCareer.VP_ENGINEERING: [TechDomain.ENTERPRISE_ARCHITECTURE, TechDomain.SOFTWARE_ARCHITECTURE],
    TechCareer.SVP_ENGINEERING: [TechDomain.ENTERPRISE_ARCHITECTURE, TechDomain.SOFTWARE_ARCHITECTURE],
    TechCareer.CTO: [TechDomain.ENTERPRISE_ARCHITECTURE, TechDomain.SOFTWARE_ARCHITECTURE],
    TechCareer.CIO: [TechDomain.ENTERPRISE_ARCHITECTURE, TechDomain.INFRASTRUCTURE],
    TechCareer.TECH_LEAD: [TechDomain.SOFTWARE_ENGINEERING, TechDomain.SOFTWARE_ARCHITECTURE],
    TechCareer.TEAM_LEAD: [TechDomain.SOFTWARE_ENGINEERING, TechDomain.AGILE],
    TechCareer.TECHNICAL_LEAD: [TechDomain.SOFTWARE_ENGINEERING, TechDomain.SOFTWARE_ARCHITECTURE],
    TechCareer.ENGINEERING_LEAD: [TechDomain.SOFTWARE_ENGINEERING, TechDomain.SOFTWARE_ARCHITECTURE],

    # ═══════════════════════════════════════════════════════════════
    # SUPPORT & OPERATIONS
    # ═══════════════════════════════════════════════════════════════

    TechCareer.IT_SUPPORT_SPECIALIST: [TechDomain.SYSTEM_ADMINISTRATION, TechDomain.NETWORKING],
    TechCareer.IT_SUPPORT_ENGINEER: [TechDomain.SYSTEM_ADMINISTRATION, TechDomain.NETWORKING],
    TechCareer.HELP_DESK_TECHNICIAN: [TechDomain.SYSTEM_ADMINISTRATION, TechDomain.NETWORKING],
    TechCareer.DESKTOP_SUPPORT: [TechDomain.SYSTEM_ADMINISTRATION, TechDomain.WINDOWS_SERVER],
    TechCareer.TECHNICAL_SUPPORT_ENGINEER: [TechDomain.SYSTEM_ADMINISTRATION, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.IT_OPERATIONS_ENGINEER: [TechDomain.INFRASTRUCTURE, TechDomain.MONITORING],
    TechCareer.IT_OPERATIONS_MANAGER: [TechDomain.INFRASTRUCTURE, TechDomain.MONITORING],
    TechCareer.NOC_ENGINEER: [TechDomain.MONITORING, TechDomain.NETWORKING],
    TechCareer.NOC_ANALYST: [TechDomain.MONITORING, TechDomain.NETWORKING],
    TechCareer.TECHNICAL_WRITER: [TechDomain.SOFTWARE_ENGINEERING, TechDomain.API_DESIGN],
    TechCareer.DOCUMENTATION_ENGINEER: [TechDomain.SOFTWARE_ENGINEERING, TechDomain.API_DESIGN],
    TechCareer.API_DOCUMENTATION_WRITER: [TechDomain.API_DESIGN, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.DEVELOPER_ADVOCATE: [TechDomain.SOFTWARE_ENGINEERING, TechDomain.API_DESIGN],
    TechCareer.DEVELOPER_EVANGELIST: [TechDomain.SOFTWARE_ENGINEERING, TechDomain.API_DESIGN],
    TechCareer.DEVREL_ENGINEER: [TechDomain.SOFTWARE_ENGINEERING, TechDomain.API_DESIGN],
    TechCareer.COMMUNITY_MANAGER: [TechDomain.SOFTWARE_ENGINEERING, TechDomain.API_DESIGN],
    TechCareer.SOLUTIONS_ENGINEER: [TechDomain.SOLUTIONS_ARCHITECTURE, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.SALES_ENGINEER: [TechDomain.SOLUTIONS_ARCHITECTURE, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.PRE_SALES_ENGINEER: [TechDomain.SOLUTIONS_ARCHITECTURE, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.TECHNICAL_CONSULTANT: [TechDomain.SOLUTIONS_ARCHITECTURE, TechDomain.SOFTWARE_ENGINEERING],
    TechCareer.IT_CONSULTANT: [TechDomain.SOLUTIONS_ARCHITECTURE, TechDomain.INFRASTRUCTURE],
}


# Domain to recommended languages
DOMAIN_LANGUAGE_MAP: Dict[TechDomain, List[CodeLanguage]] = {
    TechDomain.DATA_ENGINEERING: [CodeLanguage.PYTHON, CodeLanguage.SQL, CodeLanguage.SCALA, CodeLanguage.PYSPARK],
    TechDomain.DATA_SCIENCE: [CodeLanguage.PYTHON, CodeLanguage.R, CodeLanguage.SQL, CodeLanguage.JULIA],
    TechDomain.MACHINE_LEARNING: [CodeLanguage.PYTHON, CodeLanguage.R, CodeLanguage.JULIA],  # TensorFlow/PyTorch are Python libs
    TechDomain.WEB_FRONTEND: [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT, CodeLanguage.HTML, CodeLanguage.CSS],
    TechDomain.WEB_BACKEND: [CodeLanguage.PYTHON, CodeLanguage.JAVA, CodeLanguage.GO, CodeLanguage.JAVASCRIPT],
    TechDomain.DEVOPS: [CodeLanguage.BASH, CodeLanguage.PYTHON, CodeLanguage.YAML, CodeLanguage.TERRAFORM],
    TechDomain.KUBERNETES: [CodeLanguage.YAML, CodeLanguage.HELM, CodeLanguage.GO],
    TechDomain.CLOUD_AWS: [CodeLanguage.PYTHON, CodeLanguage.CLOUDFORMATION, CodeLanguage.TERRAFORM],
    TechDomain.CLOUD_AZURE: [CodeLanguage.PYTHON, CodeLanguage.BICEP, CodeLanguage.TERRAFORM, CodeLanguage.POWERSHELL],
    TechDomain.CLOUD_GCP: [CodeLanguage.PYTHON, CodeLanguage.TERRAFORM, CodeLanguage.GO],
    TechDomain.BLOCKCHAIN: [CodeLanguage.SOLIDITY, CodeLanguage.RUST, CodeLanguage.JAVASCRIPT],
    TechDomain.QUANTUM_COMPUTING: [CodeLanguage.PYTHON, CodeLanguage.QISKIT, CodeLanguage.CIRQ, CodeLanguage.QSHARP],
    TechDomain.CYBERSECURITY: [CodeLanguage.PYTHON, CodeLanguage.BASH, CodeLanguage.POWERSHELL, CodeLanguage.GO],
    TechDomain.DATABASES: [CodeLanguage.SQL, CodeLanguage.POSTGRESQL, CodeLanguage.MONGODB, CodeLanguage.REDIS],
    TechDomain.MOBILE_DEVELOPMENT: [CodeLanguage.KOTLIN, CodeLanguage.SWIFT, CodeLanguage.TYPESCRIPT, CodeLanguage.DART],
}
