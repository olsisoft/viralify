"""
Unit tests for the code_pipeline module.

Tests cover:
- Models: CodeSpec, TechnologyContext, ExampleIO
- MaestroSpecExtractor: Language, purpose, and context detection
- SpecConstrainedCodeGenerator: Code generation validation
- ConsoleExecutor: Execution and validation
- CodePipeline: End-to-end orchestration
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Import models
from services.code_pipeline.models import (
    CodeLanguage,
    CodePurpose,
    TechnologyEcosystem,
    TechnologyContext,
    ExampleIO,
    CodeSpec,
    GeneratedCode,
    ConsoleExecution,
    CodeSlidePackage,
    CodeSpecRequest,
    CodeSpecResponse,
)

# Import components
from services.code_pipeline.spec_extractor import (
    MaestroSpecExtractor,
    get_spec_extractor,
)


class TestCodeLanguageEnum:
    """Tests for CodeLanguage enum"""

    def test_common_languages_exist(self):
        """Verify common programming languages are defined"""
        assert CodeLanguage.PYTHON == "python"
        assert CodeLanguage.JAVA == "java"
        assert CodeLanguage.JAVASCRIPT == "javascript"
        assert CodeLanguage.TYPESCRIPT == "typescript"
        assert CodeLanguage.GO == "go"
        assert CodeLanguage.RUST == "rust"

    def test_language_values_lowercase(self):
        """All language values should be lowercase"""
        for lang in CodeLanguage:
            assert lang.value == lang.value.lower()


class TestCodePurposeEnum:
    """Tests for CodePurpose enum"""

    def test_common_purposes_exist(self):
        """Verify common code purposes are defined"""
        assert CodePurpose.TRANSFORMER == "transformer"
        assert CodePurpose.VALIDATOR == "validator"
        assert CodePurpose.PROCESSOR == "processor"
        assert CodePurpose.ALGORITHM == "algorithm"
        assert CodePurpose.PATTERN_DEMO == "pattern_demo"


class TestTechnologyEcosystemEnum:
    """Tests for TechnologyEcosystem enum"""

    def test_messaging_ecosystems_exist(self):
        """Verify messaging systems are defined"""
        assert TechnologyEcosystem.KAFKA == "kafka"
        assert TechnologyEcosystem.RABBITMQ == "rabbitmq"
        assert TechnologyEcosystem.PULSAR == "pulsar"

    def test_esb_ecosystems_exist(self):
        """Verify ESB/Integration systems are defined"""
        assert TechnologyEcosystem.MULESOFT == "mulesoft"
        assert TechnologyEcosystem.APACHE_CAMEL == "apache_camel"
        assert TechnologyEcosystem.TALEND == "talend"

    def test_cloud_ecosystems_exist(self):
        """Verify cloud providers are defined"""
        assert TechnologyEcosystem.AWS == "aws"
        assert TechnologyEcosystem.GCP == "gcp"
        assert TechnologyEcosystem.AZURE == "azure"

    def test_standalone_exists(self):
        """Verify standalone option exists"""
        assert TechnologyEcosystem.STANDALONE == "standalone"


class TestTechnologyContext:
    """Tests for TechnologyContext dataclass"""

    def test_create_kafka_connect_context(self):
        """Test creating a Kafka Connect context"""
        context = TechnologyContext(
            ecosystem=TechnologyEcosystem.KAFKA,
            component="Kafka Connect",
            version="3.x",
            architecture_pattern="EIP",
            required_apis=["Transformation", "SourceConnector"],
            implicit_imports=["org.apache.kafka.connect.transforms.Transformation"],
            naming_conventions={"class": "PascalCase"},
            context_description="Dans le contexte de Kafka Connect, un transformer..."
        )

        assert context.ecosystem == TechnologyEcosystem.KAFKA
        assert context.component == "Kafka Connect"
        assert "Transformation" in context.required_apis

    def test_create_mulesoft_context(self):
        """Test creating a MuleSoft context"""
        context = TechnologyContext(
            ecosystem=TechnologyEcosystem.MULESOFT,
            component="Anypoint Platform",
            context_description="Dans MuleSoft, un transformer utilise DataWeave..."
        )

        assert context.ecosystem == TechnologyEcosystem.MULESOFT
        assert "MuleSoft" in context.context_description


class TestExampleIO:
    """Tests for ExampleIO dataclass"""

    def test_create_xml_to_json_example(self):
        """Test creating an XML to JSON example"""
        example = ExampleIO(
            input_value='<user><name>John</name></user>',
            input_description="XML user data",
            expected_output='{"name": "John"}',
            output_description="JSON user object",
            input_display="XML Input",
            output_display="JSON Output"
        )

        assert "<user>" in example.input_value
        assert "John" in example.expected_output


class TestCodeSpec:
    """Tests for CodeSpec dataclass"""

    def test_create_basic_spec(self):
        """Test creating a basic code spec"""
        spec = CodeSpec(
            spec_id="spec_001",
            concept_name="XML to JSON Transformer",
            language=CodeLanguage.JAVA,
            purpose=CodePurpose.TRANSFORMER,
            description="Transforms XML data to JSON format",
            input_type="XML string",
            output_type="JSON string",
            key_operations=["parse XML", "build JSON", "serialize"]
        )

        assert spec.spec_id == "spec_001"
        assert spec.language == CodeLanguage.JAVA
        assert spec.purpose == CodePurpose.TRANSFORMER
        assert len(spec.key_operations) == 3

    def test_create_spec_with_context(self):
        """Test creating a spec with technology context"""
        context = TechnologyContext(
            ecosystem=TechnologyEcosystem.KAFKA,
            component="Kafka Connect SMT",
            context_description="Single Message Transform for Kafka Connect"
        )

        spec = CodeSpec(
            spec_id="spec_002",
            concept_name="Kafka Connect SMT",
            language=CodeLanguage.JAVA,
            purpose=CodePurpose.TRANSFORMER,
            context=context,
            description="SMT for data transformation",
            input_type="ConnectRecord",
            output_type="ConnectRecord",
            key_operations=["apply transform"]
        )

        assert spec.context is not None
        assert spec.context.ecosystem == TechnologyEcosystem.KAFKA

    def test_spec_with_example_io(self):
        """Test creating a spec with example I/O"""
        example = ExampleIO(
            input_value="input data",
            input_description="Sample input",
            expected_output="output data",
            output_description="Expected output"
        )

        spec = CodeSpec(
            spec_id="spec_003",
            concept_name="Data Processor",
            language=CodeLanguage.PYTHON,
            purpose=CodePurpose.PROCESSOR,
            description="Processes data",
            input_type="str",
            output_type="str",
            key_operations=["process"],
            example_io=example
        )

        assert spec.example_io is not None
        assert spec.example_io.input_value == "input data"


class TestMaestroSpecExtractor:
    """Tests for MaestroSpecExtractor"""

    @pytest.fixture
    def extractor(self):
        """Create a spec extractor instance"""
        with patch.object(MaestroSpecExtractor, '_init_llm_client'):
            extractor = MaestroSpecExtractor()
            extractor.client = MagicMock()
            extractor.model = "gpt-4o"
            return extractor

    def test_detect_language_java_explicit(self, extractor):
        """Test detecting Java language from explicit mention"""
        result = extractor._detect_language("Nous allons développer en Java un transformer", None)
        assert result == CodeLanguage.JAVA

    def test_detect_language_python_explicit(self, extractor):
        """Test detecting Python language from explicit mention"""
        result = extractor._detect_language("Créons un script Python pour parser le XML", None)
        assert result == CodeLanguage.PYTHON

    def test_detect_language_javascript_explicit(self, extractor):
        """Test detecting JavaScript language"""
        result = extractor._detect_language("Let's write some JavaScript code", None)
        assert result == CodeLanguage.JAVASCRIPT

    def test_detect_language_with_preference(self, extractor):
        """Test detecting language with explicit preference"""
        result = extractor._detect_language("Write some code", "java")
        assert result == CodeLanguage.JAVA

    def test_detect_language_fallback_pseudocode(self, extractor):
        """Test fallback to pseudocode when no language detected"""
        result = extractor._detect_language("Write some generic code", None)
        assert result == CodeLanguage.PSEUDOCODE

    def test_detect_purpose_transformer(self, extractor):
        """Test detecting transformer purpose"""
        result = extractor._detect_purpose("transformer XML en JSON")
        assert result == CodePurpose.TRANSFORMER

    def test_detect_purpose_validator(self, extractor):
        """Test detecting validator purpose"""
        result = extractor._detect_purpose("valider les données d'entrée")
        assert result == CodePurpose.VALIDATOR

    def test_detect_purpose_parser(self, extractor):
        """Test detecting parser purpose"""
        result = extractor._detect_purpose("parser le fichier CSV")
        assert result == CodePurpose.PARSER

    def test_detect_context_kafka_connect(self, extractor):
        """Test detecting Kafka Connect context"""
        result = extractor._detect_context("Dans Kafka Connect, nous utilisons un SMT")
        assert result is not None
        assert result.ecosystem == TechnologyEcosystem.KAFKA
        assert "connect" in result.component.lower()

    def test_detect_context_kafka_streams(self, extractor):
        """Test detecting Kafka Streams context"""
        result = extractor._detect_context("Avec Kafka Streams, créons un KStream processor")
        assert result is not None
        assert result.ecosystem == TechnologyEcosystem.KAFKA
        assert "stream" in result.component.lower()

    def test_detect_context_mulesoft(self, extractor):
        """Test detecting MuleSoft context"""
        result = extractor._detect_context("Dans MuleSoft Anypoint, utilisons DataWeave")
        assert result is not None
        assert result.ecosystem == TechnologyEcosystem.MULESOFT

    def test_detect_context_spring(self, extractor):
        """Test detecting Spring context"""
        result = extractor._detect_context("Avec Spring Boot, créons un service REST")
        assert result is not None
        assert result.ecosystem == TechnologyEcosystem.SPRING

    def test_detect_context_standalone(self, extractor):
        """Test detecting standalone (no specific context)"""
        result = extractor._detect_context("Écrivons une fonction simple")
        assert result is None or result.ecosystem == TechnologyEcosystem.STANDALONE

    def test_detect_context_aws_lambda(self, extractor):
        """Test detecting AWS Lambda context"""
        result = extractor._detect_context("Déployons cette fonction sur AWS Lambda")
        assert result is not None
        assert result.ecosystem == TechnologyEcosystem.AWS


class TestGeneratedCode:
    """Tests for GeneratedCode dataclass"""

    def test_create_generated_code(self):
        """Test creating a generated code object"""
        code = GeneratedCode(
            spec_id="spec_001",
            language=CodeLanguage.JAVA,
            code="public class XmlToJson { ... }",
            highlighted_lines=[1, 5, 10],
            runnable=False,
            matches_spec=True
        )

        assert code.spec_id == "spec_001"
        assert code.language == CodeLanguage.JAVA
        assert "XmlToJson" in code.code
        assert len(code.highlighted_lines) == 3


class TestConsoleExecution:
    """Tests for ConsoleExecution dataclass"""

    def test_create_console_execution(self):
        """Test creating a console execution result"""
        execution = ConsoleExecution(
            spec_id="spec_001",
            input_shown='<user><name>John</name></user>',
            output_shown='{"name": "John"}',
            execution_time_ms=50.5,
            matches_expected=True,
            formatted_console="$ python transformer.py\nInput: <user>...\nOutput: {...}"
        )

        assert execution.matches_expected is True
        assert execution.execution_time_ms == 50.5


class TestCodeSlidePackage:
    """Tests for CodeSlidePackage dataclass"""

    def test_create_slide_package(self):
        """Test creating a complete slide package"""
        spec = CodeSpec(
            spec_id="spec_001",
            concept_name="XML to JSON",
            language=CodeLanguage.PYTHON,
            purpose=CodePurpose.TRANSFORMER,
            description="Transform XML to JSON",
            input_type="str",
            output_type="str",
            key_operations=["parse", "convert"]
        )

        generated_code = GeneratedCode(
            spec_id="spec_001",
            language=CodeLanguage.PYTHON,
            code="def transform(xml): ...",
            matches_spec=True
        )

        package = CodeSlidePackage(
            spec=spec,
            generated_code=generated_code,
            is_coherent=True,
            coherence_score=0.95
        )

        assert package.is_coherent is True
        assert package.coherence_score == 0.95


class TestCodePipelineIntegration:
    """Integration tests for the complete CodePipeline"""

    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """{
            "spec_id": "spec_001",
            "concept_name": "XML to JSON Transformer",
            "language": "java",
            "purpose": "transformer",
            "description": "Transforms XML data to JSON format",
            "input_type": "XML string",
            "output_type": "JSON string",
            "key_operations": ["parse XML", "build JSON"],
            "example_io": {
                "input_value": "<user><name>John</name></user>",
                "input_description": "XML user",
                "expected_output": "{\\"name\\": \\"John\\"}",
                "output_description": "JSON user"
            }
        }"""
        return mock_response


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
