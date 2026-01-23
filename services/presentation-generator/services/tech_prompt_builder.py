"""
Tech Prompt Builder - Contextual Prompt Generation for IT Education

Builds dynamic, high-quality prompts based on:
- User profile (career, experience level)
- Course context (domain, topic, keywords)
- Technical requirements (languages, tools)
- Quality standards (clean code, testable, professional)

This module ensures all generated content meets enterprise-grade standards.
"""

import os
from typing import Optional, List, Dict, Any
from enum import Enum

from models.tech_domains import (
    TechCareer,
    TechDomain,
    CodeLanguage,
    get_career_display_name,
    get_domain_display_name,
    get_language_display_name,
    CAREER_DOMAIN_MAP,
    DOMAIN_LANGUAGE_MAP,
)


class AudienceLevel(str, Enum):
    """Audience expertise levels"""
    ABSOLUTE_BEGINNER = "absolute_beginner"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class TechPromptBuilder:
    """
    Builds contextual prompts for code and diagram generation.

    Adapts tone, complexity, and examples based on:
    - Target audience level
    - Specific tech domain
    - Career path context
    - Required technologies
    """

    def __init__(self):
        # Code quality standards - always included
        self.code_standards = self._build_code_standards()
        self.diagram_standards = self._build_diagram_standards()

        # Domain-specific expertise contexts
        self.domain_contexts = self._build_domain_contexts()

        # Career-specific contexts (Option 2)
        self.career_contexts = self._build_career_contexts()

        # Level-appropriate teaching styles
        self.teaching_styles = self._build_teaching_styles()

    def build_code_prompt(
        self,
        topic: str,
        domain: Optional[TechDomain] = None,
        career: Optional[TechCareer] = None,
        audience_level: AudienceLevel = AudienceLevel.INTERMEDIATE,
        languages: Optional[List[CodeLanguage]] = None,
        tools: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        content_language: str = "en"
    ) -> str:
        """
        Build a comprehensive prompt for code generation.

        Args:
            topic: Main topic of the course/presentation
            domain: Tech domain (e.g., DATA_ENGINEERING, DEVOPS)
            career: Target career path
            audience_level: Expertise level of the audience
            languages: Programming languages to use
            tools: Specific tools/technologies to cover
            keywords: Important keywords to include
            content_language: Language for text content

        Returns:
            Complete system prompt for code generation
        """
        sections = []

        # 1. ROLE - Based on career and audience level
        role = self._build_role_section(domain, career, audience_level)
        sections.append(f"# ROLE\n{role}")

        # 2. CONTEXT - Based on topic and domain
        context = self._build_context_section(topic, domain, keywords)
        sections.append(f"# CONTEXT\n{context}")

        # 3. AUDIENCE - Teaching style adaptation
        audience = self._build_audience_section(audience_level)
        sections.append(f"# AUDIENCE\n{audience}")

        # 4. TECHNICAL REQUIREMENTS - Languages and tools
        tech_req = self._build_tech_requirements(languages, tools, domain)
        sections.append(f"# TECHNICAL REQUIREMENTS\n{tech_req}")

        # 5. CODE QUALITY STANDARDS - Always included
        sections.append(f"# CODE QUALITY STANDARDS (MANDATORY)\n{self.code_standards}")

        # 6. EXAMPLES - Language-specific good vs bad code
        examples = self._build_code_examples(languages, audience_level)
        if examples:
            sections.append(f"# CODE EXAMPLES\n{examples}")

        # 7. LANGUAGE - Content language requirements
        lang_req = self._build_language_requirements(content_language)
        sections.append(f"# CONTENT LANGUAGE\n{lang_req}")

        return "\n\n".join(sections)

    def build_diagram_prompt(
        self,
        description: str,
        domain: Optional[TechDomain] = None,
        diagram_type: str = "architecture",
        audience_level: AudienceLevel = AudienceLevel.INTERMEDIATE,
        style: str = "dark",
        content_language: str = "en"
    ) -> str:
        """
        Build a comprehensive prompt for diagram generation.

        Args:
            description: What the diagram should show
            domain: Tech domain for context
            diagram_type: Type of diagram
            audience_level: Expertise level (affects complexity)
            style: Visual style
            content_language: Language for labels

        Returns:
            Complete system prompt for diagram generation
        """
        sections = []

        # 1. ROLE
        role = f"""You are a Senior Solutions Architect and Technical Illustrator with expertise in:
- Enterprise architecture patterns
- System design and diagramming
- Visual communication of complex technical concepts
- {get_domain_display_name(domain) if domain else 'Multi-domain IT systems'}"""
        sections.append(f"# ROLE\n{role}")

        # 2. CONTEXT
        context = f"""Creating a {diagram_type} diagram for: {description}

Domain focus: {get_domain_display_name(domain) if domain else 'General IT'}
Audience level: {audience_level.value.replace('_', ' ').title()}"""
        sections.append(f"# CONTEXT\n{context}")

        # 3. DIAGRAM STANDARDS
        sections.append(f"# DIAGRAM QUALITY STANDARDS (MANDATORY)\n{self.diagram_standards}")

        # 4. COMPLEXITY GUIDELINES
        complexity = self._get_diagram_complexity(audience_level)
        sections.append(f"# COMPLEXITY GUIDELINES\n{complexity}")

        # 5. LANGUAGE
        lang_req = f"""All labels and text must be in {self._get_language_name(content_language)}.
Technical terms that are universally understood (API, REST, HTTP, etc.) can remain in English."""
        sections.append(f"# LABEL LANGUAGE\n{lang_req}")

        return "\n\n".join(sections)

    def _build_role_section(
        self,
        domain: Optional[TechDomain],
        career: Optional[TechCareer],
        audience_level: AudienceLevel
    ) -> str:
        """Build the role section of the prompt."""

        # Determine expertise level for the teacher
        teacher_level = self._get_teacher_level(audience_level)

        # Build domain expertise (auto-infer from career if not provided)
        if not domain and career:
            domain = self._infer_domain_from_career(career)

        domain_expertise = ""
        if domain:
            domain_expertise = f"with deep expertise in {get_domain_display_name(domain)}"

        # Build career context with specific guidance
        career_context = ""
        career_specific_guidance = ""
        if career:
            career_context = f"\nYour content targets {get_career_display_name(career)} professionals."

            # Add career-specific context if available
            if career in self.career_contexts:
                career_specific_guidance = f"\n\nCareer-specific focus:{self.career_contexts[career]}"

        # Teaching persona based on audience
        teaching_persona = self._get_teaching_persona(audience_level)

        return f"""You are a {teacher_level} Software Engineer and Technical Educator {domain_expertise}.

{teaching_persona}
{career_context}{career_specific_guidance}

Your code must be:
- Production-ready and enterprise-grade
- Following industry best practices
- Clear enough for the target audience to understand
- Well-documented with meaningful comments"""

    def _build_context_section(
        self,
        topic: str,
        domain: Optional[TechDomain],
        keywords: Optional[List[str]]
    ) -> str:
        """Build the context section."""

        context = f"Topic: {topic}\n"

        if domain:
            context += f"Domain: {get_domain_display_name(domain)}\n"

            # Add domain-specific context
            if domain in self.domain_contexts:
                context += f"\nDomain-specific considerations:\n{self.domain_contexts[domain]}"

        if keywords:
            context += f"\nKey concepts to cover: {', '.join(keywords)}"

        return context

    def _build_audience_section(self, audience_level: AudienceLevel) -> str:
        """Build audience-specific guidelines."""
        return self.teaching_styles.get(audience_level, self.teaching_styles[AudienceLevel.INTERMEDIATE])

    def _build_tech_requirements(
        self,
        languages: Optional[List[CodeLanguage]],
        tools: Optional[List[str]],
        domain: Optional[TechDomain]
    ) -> str:
        """Build technical requirements section."""

        requirements = []

        if languages:
            lang_names = [get_language_display_name(lang) for lang in languages]
            requirements.append(f"Programming languages: {', '.join(lang_names)}")
        elif domain and domain in DOMAIN_LANGUAGE_MAP:
            # Use domain default languages
            default_langs = DOMAIN_LANGUAGE_MAP[domain][:3]  # Top 3
            lang_names = [get_language_display_name(lang) for lang in default_langs]
            requirements.append(f"Recommended languages for this domain: {', '.join(lang_names)}")

        if tools:
            requirements.append(f"Tools/Technologies to cover: {', '.join(tools)}")

        if not requirements:
            return "Use appropriate languages and tools for the topic."

        return "\n".join(requirements)

    def _build_code_standards(self) -> str:
        """Build mandatory code quality standards."""
        return """ALL code must adhere to these standards:

1. NAMING CONVENTIONS:
   - Variables: descriptive, no single letters (except loop indices)
   - Functions: verb phrases (get_user, calculate_total, validate_input)
   - Classes: noun phrases in PascalCase
   - Constants: SCREAMING_SNAKE_CASE

2. STRUCTURE:
   - Functions: max 20 lines, single responsibility
   - Classes: max 200 lines, cohesive
   - Files: max 400 lines
   - Nesting: max 3 levels deep

3. TESTABILITY:
   - Pure functions where possible
   - Dependency injection for external services
   - No global state
   - Clear input/output contracts

4. DOCUMENTATION:
   - Docstrings for all public functions/methods
   - Include: purpose, args, returns, examples
   - Inline comments only for non-obvious logic

5. ERROR HANDLING:
   - Specific exceptions (not bare except)
   - Meaningful error messages
   - Fail fast, recover gracefully
   - Log errors appropriately

6. TYPE SAFETY:
   - Full type hints (Python 3.10+ style)
   - Generic types where appropriate
   - Optional types for nullable values

7. PATTERNS:
   - Apply appropriate design patterns
   - Factory for object creation
   - Strategy for interchangeable algorithms
   - Repository for data access

8. ANTI-PATTERNS TO AVOID:
   - Magic numbers/strings (use constants)
   - Deep nesting (use early returns)
   - God classes/functions
   - Copy-paste code (DRY principle)
   - Premature optimization

BAD CODE EXAMPLE (NEVER DO THIS):
```python
def p(d, x):
    r = []
    for i in d:
        if i > x:
            r.append(i*2)
    return r
```

GOOD CODE EXAMPLE (ALWAYS DO THIS):
```python
from typing import List

def filter_and_double_above_threshold(
    numbers: List[float],
    threshold: float
) -> List[float]:
    \"\"\"
    Filter numbers above threshold and double them.

    Args:
        numbers: List of numbers to process
        threshold: Minimum value to include (exclusive)

    Returns:
        List of doubled values for numbers above threshold

    Example:
        >>> filter_and_double_above_threshold([1, 5, 10], 4)
        [10, 20]
    \"\"\"
    return [num * 2 for num in numbers if num > threshold]
```"""

    def _build_diagram_standards(self) -> str:
        """Build mandatory diagram quality standards."""
        return """ALL diagrams must adhere to these standards:

1. PROFESSIONAL APPEARANCE:
   - Clean, enterprise-grade look
   - Consistent spacing and alignment
   - Professional color palette
   - High resolution output

2. READABILITY:
   - Maximum 12-15 nodes for clarity
   - Clear, concise labels (2-4 words)
   - Logical grouping with clusters
   - Left-to-right or top-to-bottom flow

3. CLARITY:
   - Each element serves a purpose
   - No decorative-only components
   - Clear connection meanings
   - Edge labels for non-obvious flows

4. CONSISTENCY:
   - Uniform icon style (same provider)
   - Consistent naming convention
   - Aligned elements where logical
   - Balanced visual weight

5. COMPLETENESS:
   - All connections shown
   - No orphan nodes
   - Entry and exit points clear
   - Legend if using custom symbols

BAD DIAGRAM (NEVER):
- Too many nodes (>15)
- Cryptic labels (A, B, C)
- Crossing lines everywhere
- Mixed icon styles
- No logical grouping

GOOD DIAGRAM (ALWAYS):
- Focused scope (5-12 nodes)
- Descriptive labels ("User API", "Order DB")
- Clean flow with minimal crossings
- Consistent provider icons
- Logical clusters (Frontend, Backend, Data)"""

    def _build_domain_contexts(self) -> Dict[TechDomain, str]:
        """Build domain-specific context snippets."""
        return {
            TechDomain.DATA_ENGINEERING: """
- Focus on data pipelines, ETL/ELT patterns
- Consider scalability and data quality
- Include error handling for data issues
- Show idempotent operations where relevant
- Consider batch vs streaming trade-offs""",

            TechDomain.MACHINE_LEARNING: """
- Include data preprocessing considerations
- Show model training/inference separation
- Consider reproducibility (random seeds, versioning)
- Include evaluation metrics
- Handle edge cases in predictions""",

            TechDomain.DEVOPS: """
- Focus on automation and repeatability
- Include error handling and rollback
- Consider security best practices
- Show logging and monitoring hooks
- Infrastructure as Code patterns""",

            TechDomain.CLOUD_AWS: """
- Use AWS SDK best practices
- Include IAM considerations
- Handle AWS-specific errors
- Consider cost optimization
- Show proper resource cleanup""",

            TechDomain.KUBERNETES: """
- Follow K8s manifest best practices
- Include resource limits
- Show health checks (liveness/readiness)
- Consider security contexts
- Include proper labels and annotations""",

            TechDomain.CYBERSECURITY: """
- Never show actual credentials or secrets
- Include input validation
- Demonstrate secure coding patterns
- Consider authentication/authorization
- Show proper error handling without info leakage""",

            TechDomain.BLOCKCHAIN: """
- Include gas optimization considerations
- Show security patterns (reentrancy guards)
- Consider upgrade patterns
- Include event emissions
- Show proper access control""",

            TechDomain.QUANTUM_COMPUTING: """
- Explain quantum concepts clearly
- Show circuit diagrams when relevant
- Include measurement considerations
- Explain superposition/entanglement
- Show classical-quantum interface""",
        }

    def _build_career_contexts(self) -> Dict[TechCareer, str]:
        """Build career-specific context snippets for 50+ key careers."""
        return {
            # ═══════════════════════════════════════════════════════════════
            # DATA ENGINEERING & DATA MANAGEMENT
            # ═══════════════════════════════════════════════════════════════

            TechCareer.DATA_ENGINEER: """
- Focus on ETL/ELT pipelines and data transformations
- Include data quality checks and validation
- Consider scalability and performance optimization
- Show idempotent and fault-tolerant patterns
- Tools: Airflow, dbt, Spark, Kafka, Snowflake""",

            TechCareer.DATA_LINEAGE_ARCHITECT: """
- Focus on metadata flow and data provenance tracking
- Include column-level and field-level lineage
- Consider impact analysis for schema changes
- Show integration with data catalogs
- Tools: OpenLineage, Marquez, DataHub, Atlan, Collibra""",

            TechCareer.DATA_LINEAGE_DEVELOPER: """
- Focus on implementing lineage extraction and tracking
- Include OpenLineage specification compliance
- Consider both technical and business lineage
- Show integration with orchestration tools
- Tools: OpenLineage SDK, Marquez, DataHub APIs""",

            TechCareer.DATA_LINEAGE_ANALYST: """
- Focus on analyzing data dependencies and flows
- Include impact assessment methodologies
- Consider root cause analysis patterns
- Show lineage visualization techniques
- Tools: DataHub UI, Collibra, Alation, Informatica""",

            TechCareer.DATA_ENABLER: """
- Focus on making data accessible and understandable
- Include self-service analytics patterns
- Consider data literacy training materials
- Show documentation best practices
- Tools: Data catalogs, BI tools, documentation platforms""",

            TechCareer.DATA_ENABLEMENT_LEAD: """
- Focus on data democratization strategies
- Include governance and self-service balance
- Consider organizational change management
- Show metrics for data adoption
- Tools: Data mesh patterns, catalog tools, training platforms""",

            TechCareer.DATA_QUALITY_ENGINEER: """
- Focus on data validation and profiling
- Include anomaly detection patterns
- Consider data contracts and SLAs
- Show monitoring and alerting strategies
- Tools: Great Expectations, dbt tests, Soda, Monte Carlo""",

            TechCareer.DATA_GOVERNANCE_ANALYST: """
- Focus on policy compliance and data classification
- Include privacy and security considerations
- Consider regulatory requirements (GDPR, CCPA)
- Show data lifecycle management
- Tools: Collibra, Alation, Informatica, custom policies""",

            TechCareer.DATA_STEWARD: """
- Focus on data ownership and accountability
- Include metadata curation practices
- Consider business glossary management
- Show data quality remediation processes
- Tools: Data catalog tools, governance platforms""",

            TechCareer.DATA_ARCHITECT: """
- Focus on data modeling and schema design
- Include data mesh and data fabric patterns
- Consider multi-cloud data strategies
- Show reference architectures
- Tools: ER modeling tools, cloud data services""",

            TechCareer.ANALYTICS_ENGINEER: """
- Focus on transformation and modeling layers
- Include dimensional modeling techniques
- Consider semantic layer design
- Show testing and documentation patterns
- Tools: dbt, LookML, Metrics Layer""",

            TechCareer.DATA_CATALOG_ENGINEER: """
- Focus on metadata ingestion and management
- Include automated discovery patterns
- Consider search and classification
- Show API integration strategies
- Tools: DataHub, Amundsen, Atlan, OpenMetadata""",

            TechCareer.METADATA_ARCHITECT: """
- Focus on enterprise metadata strategy
- Include metadata standards and taxonomies
- Consider metadata exchange formats
- Show metadata governance frameworks
- Tools: Apache Atlas, custom metadata stores""",

            TechCareer.BIG_DATA_ENGINEER: """
- Focus on distributed processing patterns
- Include partitioning and optimization strategies
- Consider cost optimization for large datasets
- Show batch and streaming architectures
- Tools: Spark, Hadoop, Flink, Presto, Trino""",

            TechCareer.STREAMING_DATA_ENGINEER: """
- Focus on real-time processing patterns
- Include exactly-once semantics
- Consider late data and watermarks
- Show stream-table duality
- Tools: Kafka, Flink, Spark Streaming, Kinesis""",

            # ═══════════════════════════════════════════════════════════════
            # MACHINE LEARNING & AI
            # ═══════════════════════════════════════════════════════════════

            TechCareer.ML_ENGINEER: """
- Focus on production ML system design
- Include feature engineering best practices
- Consider model serving and scaling
- Show experiment tracking patterns
- Tools: MLflow, Kubeflow, SageMaker, Vertex AI""",

            TechCareer.MLOPS_ENGINEER: """
- Focus on ML pipeline automation
- Include CI/CD for machine learning
- Consider model monitoring and drift detection
- Show feature store integration
- Tools: MLflow, Kubeflow, Feast, DVC, Weights & Biases""",

            TechCareer.DATA_SCIENTIST: """
- Focus on exploratory analysis and modeling
- Include statistical validation techniques
- Consider business impact measurement
- Show reproducibility best practices
- Tools: Jupyter, pandas, scikit-learn, statistical libraries""",

            TechCareer.DEEP_LEARNING_ENGINEER: """
- Focus on neural network architecture design
- Include training optimization techniques
- Consider distributed training patterns
- Show model interpretability approaches
- Tools: PyTorch, TensorFlow, JAX, Hugging Face""",

            TechCareer.NLP_ENGINEER: """
- Focus on text processing pipelines
- Include tokenization and embedding strategies
- Consider multilingual and domain-specific models
- Show evaluation metrics for NLP
- Tools: Hugging Face, spaCy, NLTK, LangChain""",

            TechCareer.LLM_ENGINEER: """
- Focus on LLM application development
- Include prompt engineering techniques
- Consider RAG and fine-tuning patterns
- Show evaluation and safety measures
- Tools: LangChain, LlamaIndex, OpenAI API, Anthropic""",

            TechCareer.PROMPT_ENGINEER: """
- Focus on effective prompt design
- Include chain-of-thought and few-shot patterns
- Consider prompt testing and optimization
- Show prompt template management
- Tools: LangChain, PromptFlow, various LLM APIs""",

            TechCareer.COMPUTER_VISION_ENGINEER: """
- Focus on image/video processing pipelines
- Include object detection and segmentation
- Consider real-time inference optimization
- Show data augmentation strategies
- Tools: OpenCV, PyTorch Vision, YOLO, detectron2""",

            TechCareer.RECOMMENDATION_ENGINEER: """
- Focus on recommendation system architectures
- Include collaborative and content-based filtering
- Consider A/B testing and experimentation
- Show cold start and diversity handling
- Tools: TensorFlow Recommenders, Surprise, custom systems""",

            TechCareer.FEATURE_STORE_ENGINEER: """
- Focus on feature management and serving
- Include online/offline feature consistency
- Consider feature versioning and lineage
- Show integration with ML pipelines
- Tools: Feast, Tecton, Hopsworks, SageMaker Feature Store""",

            # ═══════════════════════════════════════════════════════════════
            # DEVOPS / PLATFORM / SRE
            # ═══════════════════════════════════════════════════════════════

            TechCareer.DEVOPS_ENGINEER: """
- Focus on automation and CI/CD pipelines
- Include infrastructure as code patterns
- Consider GitOps and deployment strategies
- Show monitoring and alerting integration
- Tools: Jenkins, GitLab CI, GitHub Actions, ArgoCD""",

            TechCareer.PLATFORM_ENGINEER: """
- Focus on internal developer platform design
- Include self-service infrastructure patterns
- Consider golden paths and guardrails
- Show developer experience metrics
- Tools: Backstage, Kubernetes, Terraform, custom platforms""",

            TechCareer.SRE: """
- Focus on reliability and SLO/SLI/SLA
- Include incident management and postmortems
- Consider capacity planning and scaling
- Show toil reduction strategies
- Tools: Prometheus, Grafana, PagerDuty, custom tooling""",

            TechCareer.KUBERNETES_ENGINEER: """
- Focus on container orchestration best practices
- Include RBAC and security contexts
- Consider resource optimization and autoscaling
- Show networking and service mesh patterns
- Tools: kubectl, Helm, Kustomize, Istio, Linkerd""",

            TechCareer.INFRASTRUCTURE_ENGINEER: """
- Focus on infrastructure provisioning
- Include IaC best practices and modules
- Consider multi-environment management
- Show drift detection and remediation
- Tools: Terraform, Pulumi, CloudFormation, Ansible""",

            TechCareer.OBSERVABILITY_ENGINEER: """
- Focus on distributed tracing and metrics
- Include log aggregation strategies
- Consider correlation and root cause analysis
- Show dashboard and alert design
- Tools: OpenTelemetry, Jaeger, ELK, Datadog, New Relic""",

            TechCareer.CICD_ENGINEER: """
- Focus on pipeline design and optimization
- Include testing integration strategies
- Consider security scanning in pipelines
- Show artifact management
- Tools: Jenkins, GitLab CI, GitHub Actions, Tekton""",

            # ═══════════════════════════════════════════════════════════════
            # CLOUD
            # ═══════════════════════════════════════════════════════════════

            TechCareer.CLOUD_ARCHITECT: """
- Focus on cloud-native architecture patterns
- Include multi-region and DR strategies
- Consider cost optimization and FinOps
- Show migration and modernization paths
- Tools: AWS Well-Architected, Azure CAF, GCP frameworks""",

            TechCareer.AWS_SOLUTIONS_ARCHITECT: """
- Focus on AWS service selection and integration
- Include security and compliance patterns
- Consider cost and performance optimization
- Show reference architectures
- Tools: AWS CDK, CloudFormation, AWS services""",

            TechCareer.AZURE_SOLUTIONS_ARCHITECT: """
- Focus on Azure service integration
- Include hybrid cloud scenarios
- Consider Azure-specific best practices
- Show enterprise integration patterns
- Tools: Bicep, ARM, Azure services, Azure DevOps""",

            TechCareer.GCP_CLOUD_ARCHITECT: """
- Focus on GCP service selection
- Include BigQuery and data analytics patterns
- Consider GKE and serverless options
- Show Google-specific optimizations
- Tools: Terraform for GCP, gcloud, GCP services""",

            TechCareer.FINOPS_ENGINEER: """
- Focus on cloud cost visibility and optimization
- Include chargeback and showback models
- Consider reserved instances and savings plans
- Show cost allocation strategies
- Tools: CloudHealth, Kubecost, native cost tools""",

            TechCareer.SERVERLESS_ARCHITECT: """
- Focus on event-driven architectures
- Include cold start optimization
- Consider function composition patterns
- Show testing and debugging strategies
- Tools: AWS Lambda, Azure Functions, Cloud Functions""",

            # ═══════════════════════════════════════════════════════════════
            # SECURITY
            # ═══════════════════════════════════════════════════════════════

            TechCareer.SECURITY_ENGINEER: """
- Focus on secure development practices
- Include vulnerability management
- Consider threat modeling techniques
- Show security automation patterns
- Tools: SAST/DAST tools, security scanners""",

            TechCareer.DEVSECOPS_ENGINEER: """
- Focus on security in CI/CD pipelines
- Include shift-left security patterns
- Consider compliance as code
- Show security gates and policies
- Tools: Snyk, SonarQube, Trivy, OPA, Checkov""",

            TechCareer.PENETRATION_TESTER: """
- Focus on ethical hacking methodologies
- Include reconnaissance and exploitation
- Consider reporting and remediation
- Show tool usage and custom scripts
- Tools: Burp Suite, Metasploit, nmap, custom tools""",

            TechCareer.CLOUD_SECURITY_ARCHITECT: """
- Focus on cloud security posture
- Include IAM and least privilege
- Consider network security and encryption
- Show compliance frameworks mapping
- Tools: Cloud-native security tools, CSPM solutions""",

            TechCareer.IAM_ENGINEER: """
- Focus on identity and access management
- Include SSO and federation patterns
- Consider zero trust architecture
- Show RBAC and ABAC implementation
- Tools: Okta, Azure AD, AWS IAM, custom IAM""",

            TechCareer.THREAT_HUNTER: """
- Focus on proactive threat detection
- Include hypothesis-driven hunting
- Consider MITRE ATT&CK mapping
- Show detection engineering patterns
- Tools: SIEM, EDR, threat intel platforms""",

            # ═══════════════════════════════════════════════════════════════
            # DATABASES
            # ═══════════════════════════════════════════════════════════════

            TechCareer.DBA: """
- Focus on database administration
- Include backup and recovery strategies
- Consider performance tuning
- Show high availability patterns
- Tools: Database-specific tools, monitoring solutions""",

            TechCareer.DATABASE_ARCHITECT: """
- Focus on database design and modeling
- Include distributed database patterns
- Consider data consistency and CAP theorem
- Show migration strategies
- Tools: ER modeling tools, database platforms""",

            TechCareer.DBRE: """
- Focus on database reliability engineering
- Include SRE practices for databases
- Consider automated operations
- Show performance SLOs
- Tools: Database observability, automation tools""",

            # ═══════════════════════════════════════════════════════════════
            # SOFTWARE ARCHITECTURE
            # ═══════════════════════════════════════════════════════════════

            TechCareer.SOFTWARE_ARCHITECT: """
- Focus on system design and patterns
- Include non-functional requirements
- Consider trade-offs and decisions
- Show documentation practices (ADRs)
- Tools: Diagramming, modeling, documentation""",

            TechCareer.MICROSERVICES_ARCHITECT: """
- Focus on service decomposition
- Include inter-service communication
- Consider eventual consistency
- Show saga and CQRS patterns
- Tools: API gateways, service mesh, message brokers""",

            TechCareer.API_ARCHITECT: """
- Focus on API design and standards
- Include versioning and evolution
- Consider developer experience
- Show API governance patterns
- Tools: OpenAPI, GraphQL, API gateways""",

            TechCareer.ENTERPRISE_ARCHITECT: """
- Focus on organization-wide IT strategy
- Include business-IT alignment
- Consider technology roadmaps
- Show TOGAF and framework usage
- Tools: ArchiMate, enterprise architecture tools""",

            # ═══════════════════════════════════════════════════════════════
            # EMERGING TECH
            # ═══════════════════════════════════════════════════════════════

            TechCareer.BLOCKCHAIN_DEVELOPER: """
- Focus on smart contract development
- Include gas optimization techniques
- Consider security vulnerabilities
- Show testing and deployment patterns
- Tools: Hardhat, Foundry, Truffle, web3 libraries""",

            TechCareer.QUANTUM_SOFTWARE_ENGINEER: """
- Focus on quantum algorithm implementation
- Include circuit design patterns
- Consider hybrid classical-quantum
- Show noise and error mitigation
- Tools: Qiskit, Cirq, PennyLane, Q#""",

            TechCareer.IOT_ENGINEER: """
- Focus on embedded systems and connectivity
- Include edge computing patterns
- Consider power and bandwidth constraints
- Show device management strategies
- Tools: MQTT, IoT platforms, embedded SDKs""",

            TechCareer.ROBOTICS_SOFTWARE_ENGINEER: """
- Focus on ROS and robot control
- Include sensor integration
- Consider real-time constraints
- Show simulation and testing
- Tools: ROS/ROS2, Gazebo, robotics frameworks""",

            # ═══════════════════════════════════════════════════════════════
            # FRONTEND & FULLSTACK
            # ═══════════════════════════════════════════════════════════════

            TechCareer.FRONTEND_DEVELOPER: """
- Focus on UI component architecture
- Include state management patterns
- Consider performance optimization
- Show accessibility best practices
- Tools: React, Vue, Angular, testing libraries""",

            TechCareer.FULLSTACK_DEVELOPER: """
- Focus on end-to-end feature development
- Include API design and integration
- Consider full-stack testing strategies
- Show deployment patterns
- Tools: Full-stack frameworks, databases, cloud""",

            TechCareer.BACKEND_DEVELOPER: """
- Focus on API and service design
- Include database integration
- Consider caching and optimization
- Show security best practices
- Tools: Backend frameworks, databases, message queues""",
        }

    def _build_teaching_styles(self) -> Dict[AudienceLevel, str]:
        """Build audience-appropriate teaching styles."""
        return {
            AudienceLevel.ABSOLUTE_BEGINNER: """
AUDIENCE: Absolute beginners with no programming experience

Teaching approach:
- Explain EVERY concept, assume nothing
- Use real-world analogies extensively
- Break down into smallest possible steps
- Repeat key concepts multiple times
- Use extremely simple examples first
- Avoid jargon or define all terms
- Celebrate small victories
- Include "why" for everything""",

            AudienceLevel.BEGINNER: """
AUDIENCE: Beginners with basic programming knowledge

Teaching approach:
- Explain new concepts thoroughly
- Build on basic programming knowledge
- Use practical, relatable examples
- Introduce terminology gradually
- Show common mistakes and how to avoid them
- Provide step-by-step instructions
- Connect to concepts they likely know""",

            AudienceLevel.INTERMEDIATE: """
AUDIENCE: Intermediate developers with solid fundamentals

Teaching approach:
- Assume familiarity with basics
- Focus on best practices and patterns
- Explain the "why" behind decisions
- Show trade-offs between approaches
- Include real-world scenarios
- Mention edge cases and gotchas
- Reference related advanced topics""",

            AudienceLevel.ADVANCED: """
AUDIENCE: Advanced developers with significant experience

Teaching approach:
- Assume strong technical foundation
- Focus on optimization and edge cases
- Discuss architectural trade-offs
- Include performance considerations
- Show advanced patterns and techniques
- Reference industry standards
- Discuss scalability implications""",

            AudienceLevel.EXPERT: """
AUDIENCE: Expert developers and architects

Teaching approach:
- Peer-to-peer technical discussion
- Focus on cutting-edge techniques
- Deep dive into internals
- Discuss research and emerging patterns
- Challenge conventional approaches
- Include performance benchmarks
- Reference academic/industry papers""",
        }

    def _build_code_examples(
        self,
        languages: Optional[List[CodeLanguage]],
        audience_level: AudienceLevel
    ) -> str:
        """Build language-specific code examples."""

        if not languages:
            return ""

        examples = []

        for lang in languages[:2]:  # Max 2 examples
            example = self._get_language_example(lang, audience_level)
            if example:
                examples.append(example)

        return "\n\n".join(examples)

    def _get_language_example(
        self,
        lang: CodeLanguage,
        level: AudienceLevel
    ) -> Optional[str]:
        """Get a language-specific example."""

        # Python example
        if lang in [CodeLanguage.PYTHON, CodeLanguage.QISKIT]:
            return '''PYTHON CODE STYLE:
```python
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

class Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    """Represents a task in the system."""
    id: str
    title: str
    status: Status = Status.PENDING

    def complete(self) -> None:
        """Mark task as completed."""
        self.status = Status.COMPLETED

def get_pending_tasks(tasks: List[Task]) -> List[Task]:
    """
    Filter tasks to return only pending ones.

    Args:
        tasks: List of tasks to filter

    Returns:
        List of tasks with PENDING status
    """
    return [task for task in tasks if task.status == Status.PENDING]
```'''

        # TypeScript example
        if lang in [CodeLanguage.TYPESCRIPT, CodeLanguage.JAVASCRIPT]:
            return '''TYPESCRIPT CODE STYLE:
```typescript
interface User {
  id: string;
  email: string;
  createdAt: Date;
}

interface UserRepository {
  findById(id: string): Promise<User | null>;
  save(user: User): Promise<void>;
}

class UserService {
  constructor(private readonly repository: UserRepository) {}

  async getUserOrThrow(id: string): Promise<User> {
    const user = await this.repository.findById(id);

    if (!user) {
      throw new NotFoundError(`User ${id} not found`);
    }

    return user;
  }
}
```'''

        # Go example
        if lang == CodeLanguage.GO:
            return '''GO CODE STYLE:
```go
package user

import (
    "context"
    "errors"
)

// ErrUserNotFound is returned when a user cannot be found.
var ErrUserNotFound = errors.New("user not found")

// User represents a user in the system.
type User struct {
    ID    string
    Email string
}

// Repository defines the interface for user storage.
type Repository interface {
    FindByID(ctx context.Context, id string) (*User, error)
}

// Service handles user business logic.
type Service struct {
    repo Repository
}

// NewService creates a new user service.
func NewService(repo Repository) *Service {
    return &Service{repo: repo}
}

// GetUser retrieves a user by ID.
func (s *Service) GetUser(ctx context.Context, id string) (*User, error) {
    user, err := s.repo.FindByID(ctx, id)
    if err != nil {
        return nil, fmt.Errorf("failed to get user: %w", err)
    }
    return user, nil
}
```'''

        return None

    def _get_teacher_level(self, audience_level: AudienceLevel) -> str:
        """Get appropriate teacher seniority based on audience."""
        mapping = {
            AudienceLevel.ABSOLUTE_BEGINNER: "Patient Senior",
            AudienceLevel.BEGINNER: "Experienced Senior",
            AudienceLevel.INTERMEDIATE: "Staff",
            AudienceLevel.ADVANCED: "Principal",
            AudienceLevel.EXPERT: "Distinguished Fellow",
        }
        return mapping.get(audience_level, "Senior")

    def _get_teaching_persona(self, audience_level: AudienceLevel) -> str:
        """Get teaching persona based on audience level."""
        personas = {
            AudienceLevel.ABSOLUTE_BEGINNER:
                "You are patient, encouraging, and explain concepts as if teaching someone their very first program. "
                "Use analogies to everyday life. Celebrate progress.",

            AudienceLevel.BEGINNER:
                "You are supportive and thorough, building confidence while teaching proper foundations. "
                "Connect new concepts to what beginners likely already know.",

            AudienceLevel.INTERMEDIATE:
                "You are professional and practical, focusing on real-world applications and best practices. "
                "You challenge learners to think about edge cases and trade-offs.",

            AudienceLevel.ADVANCED:
                "You are technically rigorous, discussing advanced patterns and optimizations. "
                "You treat learners as capable developers ready for complex challenges.",

            AudienceLevel.EXPERT:
                "You engage as a technical peer, discussing cutting-edge techniques and research. "
                "You challenge assumptions and explore the boundaries of current best practices.",
        }
        return personas.get(audience_level, personas[AudienceLevel.INTERMEDIATE])

    def _get_diagram_complexity(self, audience_level: AudienceLevel) -> str:
        """Get diagram complexity guidelines based on audience."""
        guidelines = {
            AudienceLevel.ABSOLUTE_BEGINNER: """
- Maximum 5-7 components
- One concept per diagram
- Very simple flows
- Extensive labels
- No advanced patterns""",

            AudienceLevel.BEGINNER: """
- Maximum 8-10 components
- Simple groupings allowed
- Clear, linear flows
- Helpful annotations
- Basic patterns only""",

            AudienceLevel.INTERMEDIATE: """
- Maximum 10-12 components
- Logical clustering
- Multiple flows acceptable
- Technical labels
- Common patterns shown""",

            AudienceLevel.ADVANCED: """
- Maximum 12-15 components
- Complex groupings
- Multiple interaction patterns
- Technical detail
- Advanced patterns welcome""",

            AudienceLevel.EXPERT: """
- Complexity as needed (still readable)
- Full architectural detail
- All relevant components
- Expert-level patterns
- Cross-cutting concerns shown""",
        }
        return guidelines.get(audience_level, guidelines[AudienceLevel.INTERMEDIATE])

    def _build_language_requirements(self, content_language: str) -> str:
        """Build content language requirements."""
        lang_name = self._get_language_name(content_language)

        return f"""All text content must be in {lang_name}:
- Titles and headings
- Explanations and narration
- Comments in code (educational comments)
- Diagram labels (except universal tech terms)

Code syntax and keywords remain in their native language.
Variable names should be in English for code readability.
Technical terms universally known (API, REST, HTTP, JSON) can stay in English."""

    def _get_language_name(self, code: str) -> str:
        """Get full language name from code."""
        names = {
            "en": "English",
            "fr": "French (Français)",
            "es": "Spanish (Español)",
            "de": "German (Deutsch)",
            "pt": "Portuguese (Português)",
            "it": "Italian (Italiano)",
            "nl": "Dutch (Nederlands)",
            "pl": "Polish (Polski)",
            "ru": "Russian (Русский)",
            "zh": "Chinese (中文)",
            "ja": "Japanese (日本語)",
            "ko": "Korean (한국어)",
            "ar": "Arabic (العربية)",
        }
        return names.get(code.lower(), code)

    def _infer_domain_from_career(self, career: TechCareer) -> Optional[TechDomain]:
        """
        Auto-detect domain from career name (Option 3).
        Used as fallback when no explicit domain mapping exists.
        """
        # First check the explicit CAREER_DOMAIN_MAP
        if career in CAREER_DOMAIN_MAP:
            domains = CAREER_DOMAIN_MAP[career]
            return domains[0] if domains else None

        # Fallback: infer from career name
        career_name = career.value.lower()

        # Data-related careers
        if "lineage" in career_name:
            return TechDomain.DATA_LINEAGE
        if "data_quality" in career_name or "quality" in career_name and "data" in career_name:
            return TechDomain.DATA_QUALITY
        if "governance" in career_name:
            return TechDomain.DATA_GOVERNANCE
        if "catalog" in career_name:
            return TechDomain.DATA_CATALOG
        if "metadata" in career_name:
            return TechDomain.METADATA_MANAGEMENT
        if "enabler" in career_name or "enablement" in career_name:
            return TechDomain.DATA_GOVERNANCE
        if "steward" in career_name or "custodian" in career_name:
            return TechDomain.DATA_GOVERNANCE
        if "data_engineer" in career_name or "etl" in career_name or "elt" in career_name:
            return TechDomain.DATA_ENGINEERING
        if "data_scientist" in career_name:
            return TechDomain.DATA_SCIENCE
        if "data_analyst" in career_name:
            return TechDomain.DATA_ANALYTICS
        if "bi_" in career_name or "business_intelligence" in career_name:
            return TechDomain.BUSINESS_INTELLIGENCE
        if "analytics_engineer" in career_name or "dbt" in career_name:
            return TechDomain.ANALYTICS_ENGINEERING
        if "big_data" in career_name or "hadoop" in career_name or "spark" in career_name:
            return TechDomain.BIG_DATA
        if "streaming" in career_name or "kafka" in career_name or "flink" in career_name:
            return TechDomain.STREAMING_DATA

        # ML/AI careers
        if "mlops" in career_name:
            return TechDomain.MLOPS
        if "ml_" in career_name or "machine_learning" in career_name:
            return TechDomain.MACHINE_LEARNING
        if "deep_learning" in career_name or "neural" in career_name:
            return TechDomain.DEEP_LEARNING
        if "nlp" in career_name or "linguist" in career_name:
            return TechDomain.NLP
        if "computer_vision" in career_name or "image" in career_name or "video" in career_name:
            return TechDomain.COMPUTER_VISION
        if "llm" in career_name or "prompt" in career_name or "generative_ai" in career_name:
            return TechDomain.GENERATIVE_AI
        if "ai_" in career_name:
            return TechDomain.MACHINE_LEARNING
        if "recommendation" in career_name or "personalization" in career_name:
            return TechDomain.RECOMMENDATION_SYSTEMS

        # DevOps/Platform careers
        if "devops" in career_name:
            return TechDomain.DEVOPS
        if "sre" in career_name or "reliability" in career_name:
            return TechDomain.SITE_RELIABILITY
        if "platform" in career_name:
            return TechDomain.PLATFORM_ENGINEERING
        if "kubernetes" in career_name or "k8s" in career_name:
            return TechDomain.KUBERNETES
        if "container" in career_name or "docker" in career_name:
            return TechDomain.CONTAINERS
        if "cicd" in career_name or "pipeline" in career_name or "release" in career_name:
            return TechDomain.CICD
        if "infrastructure" in career_name or "iac" in career_name:
            return TechDomain.INFRASTRUCTURE_AS_CODE
        if "observability" in career_name or "monitoring" in career_name:
            return TechDomain.OBSERVABILITY

        # Cloud careers
        if "aws" in career_name:
            return TechDomain.CLOUD_AWS
        if "azure" in career_name:
            return TechDomain.CLOUD_AZURE
        if "gcp" in career_name:
            return TechDomain.CLOUD_GCP
        if "cloud" in career_name:
            return TechDomain.CLOUD_COMPUTING
        if "serverless" in career_name or "lambda" in career_name or "functions" in career_name:
            return TechDomain.SERVERLESS
        if "finops" in career_name or "cost" in career_name:
            return TechDomain.FINOPS

        # Security careers
        if "security" in career_name or "sec" in career_name:
            return TechDomain.CYBERSECURITY
        if "devsecops" in career_name:
            return TechDomain.DEVSECOPS
        if "pentester" in career_name or "pentest" in career_name or "offensive" in career_name:
            return TechDomain.PENETRATION_TESTING
        if "soc" in career_name or "blue_team" in career_name or "defensive" in career_name:
            return TechDomain.DEFENSIVE_SECURITY
        if "threat" in career_name:
            return TechDomain.THREAT_INTELLIGENCE
        if "forensics" in career_name or "malware" in career_name:
            return TechDomain.DIGITAL_FORENSICS
        if "iam" in career_name or "identity" in career_name or "access" in career_name:
            return TechDomain.IAM
        if "grc" in career_name or "compliance" in career_name or "audit" in career_name:
            return TechDomain.GRC
        if "cryptograph" in career_name or "pki" in career_name:
            return TechDomain.CRYPTOGRAPHY

        # Database careers
        if "dba" in career_name or "database" in career_name:
            return TechDomain.DATABASES
        if "sql" in career_name or "postgresql" in career_name or "mysql" in career_name:
            return TechDomain.RELATIONAL_DATABASES
        if "mongodb" in career_name or "cassandra" in career_name or "nosql" in career_name:
            return TechDomain.NOSQL_DATABASES
        if "redis" in career_name or "elasticsearch" in career_name:
            return TechDomain.NOSQL_DATABASES

        # Network careers
        if "network" in career_name:
            return TechDomain.NETWORKING
        if "wireless" in career_name or "wan" in career_name or "lan" in career_name:
            return TechDomain.NETWORKING

        # System administration
        if "sysadmin" in career_name or "administrator" in career_name:
            return TechDomain.SYSTEM_ADMINISTRATION
        if "linux" in career_name:
            return TechDomain.LINUX
        if "windows" in career_name:
            return TechDomain.WINDOWS_SERVER
        if "vmware" in career_name or "virtualization" in career_name:
            return TechDomain.VIRTUALIZATION
        if "storage" in career_name:
            return TechDomain.STORAGE

        # QA/Testing careers
        if "qa" in career_name or "test" in career_name or "sdet" in career_name:
            return TechDomain.SOFTWARE_TESTING
        if "automation" in career_name and ("test" in career_name or "qa" in career_name):
            return TechDomain.TEST_AUTOMATION
        if "performance" in career_name:
            return TechDomain.PERFORMANCE_TESTING

        # Architecture careers
        if "architect" in career_name:
            if "enterprise" in career_name or "togaf" in career_name:
                return TechDomain.ENTERPRISE_ARCHITECTURE
            if "solution" in career_name:
                return TechDomain.SOLUTIONS_ARCHITECTURE
            if "api" in career_name:
                return TechDomain.API_DESIGN
            if "microservice" in career_name:
                return TechDomain.MICROSERVICES
            return TechDomain.SOFTWARE_ARCHITECTURE

        # Development careers
        if "frontend" in career_name or "ui_" in career_name:
            return TechDomain.WEB_FRONTEND
        if "backend" in career_name or "api_developer" in career_name:
            return TechDomain.WEB_BACKEND
        if "fullstack" in career_name:
            return TechDomain.FULLSTACK
        if "mobile" in career_name or "ios" in career_name or "android" in career_name:
            return TechDomain.MOBILE_DEVELOPMENT
        if "game" in career_name:
            return TechDomain.GAME_DEVELOPMENT
        if "embedded" in career_name or "firmware" in career_name:
            return TechDomain.EMBEDDED_SYSTEMS
        if "systems_programmer" in career_name or "low_level" in career_name:
            return TechDomain.SYSTEMS_PROGRAMMING

        # Emerging tech
        if "blockchain" in career_name or "smart_contract" in career_name or "solidity" in career_name:
            return TechDomain.BLOCKCHAIN
        if "web3" in career_name or "defi" in career_name or "nft" in career_name:
            return TechDomain.WEB3
        if "quantum" in career_name or "qiskit" in career_name or "cirq" in career_name:
            return TechDomain.QUANTUM_COMPUTING
        if "iot" in career_name:
            return TechDomain.IOT
        if "robotics" in career_name or "ros" in career_name:
            return TechDomain.ROBOTICS
        if "ar" in career_name or "vr" in career_name or "xr" in career_name or "metaverse" in career_name:
            return TechDomain.AR_VR
        if "edge" in career_name:
            return TechDomain.EDGE_COMPUTING

        # Default to software engineering
        return TechDomain.SOFTWARE_ENGINEERING


# Singleton instance for easy import
prompt_builder = TechPromptBuilder()
