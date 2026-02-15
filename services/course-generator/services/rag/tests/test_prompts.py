"""
Unit Tests for Prompt Builders

Tests BasePromptBuilder, DocumentSummaryPromptBuilder, and StructureExtractionPromptBuilder.
"""

import pytest
from ..prompts.base_prompt import BasePromptBuilder, PromptSection, PromptExample
from ..prompts.summary_prompts import DocumentSummaryPromptBuilder
from ..prompts.structure_prompts import StructureExtractionPromptBuilder


class TestPromptExample:
    """Tests for PromptExample dataclass."""

    def test_create_correct_example(self):
        """Test creating a correct example."""
        example = PromptExample(
            input="[PDF about Kafka]",
            output="This document covers Kafka architecture...",
            is_correct=True,
            explanation="Focuses on learning outcomes",
        )

        assert example.is_correct is True
        assert "Kafka" in example.input

    def test_create_incorrect_example(self):
        """Test creating an incorrect example."""
        example = PromptExample(
            input="[Same PDF]",
            output="This is a 50-page document...",
            is_correct=False,
            explanation="Focuses on metadata, not learning",
        )

        assert example.is_correct is False


class TestPromptSection:
    """Tests for PromptSection dataclass."""

    def test_create_section(self):
        """Test creating a prompt section."""
        section = PromptSection(
            title="CUSTOM SECTION",
            content="This is the section content.",
        )

        assert section.title == "CUSTOM SECTION"
        assert section.content == "This is the section content."


class TestDocumentSummaryPromptBuilder:
    """Tests for DocumentSummaryPromptBuilder."""

    @pytest.fixture
    def builder(self):
        """Create default builder."""
        return DocumentSummaryPromptBuilder()

    @pytest.fixture
    def french_builder(self):
        """Create French builder."""
        return DocumentSummaryPromptBuilder(
            document_type="pdf",
            content_language="fr",
        )

    def test_get_role(self, builder):
        """Test role definition."""
        role = builder.get_role()

        assert "Senior Content Analysis Agent" in role
        assert "AUTONOMOUSLY" in role
        assert "expert" in role.lower()

    def test_get_context(self, builder):
        """Test context includes Viralify."""
        context = builder.get_context()

        assert "Viralify" in context
        assert "course" in context.lower() or "Course" in context

    def test_get_context_language(self, french_builder):
        """Test context includes target language."""
        context = french_builder.get_context()

        assert "fr" in context

    def test_get_input_signals(self, builder):
        """Test input signals are defined."""
        signals = builder.get_input_signals()

        assert "Document content" in signals or "DOCUMENT" in signals
        assert len(signals) > 50

    def test_get_responsibilities(self, builder):
        """Test responsibilities are a non-empty list."""
        responsibilities = builder.get_responsibilities()

        assert isinstance(responsibilities, list)
        assert len(responsibilities) >= 5

        # Check for key responsibilities
        resp_text = " ".join(responsibilities).lower()
        assert "topic" in resp_text or "subject" in resp_text
        assert "summary" in resp_text or "language" in resp_text

    def test_get_decision_rules(self, builder):
        """Test decision rules contain hard constraints."""
        rules = builder.get_decision_rules()

        # Should have table format
        assert "|" in rules

        # Should have key constraints
        assert "2-3 sentence" in rules.lower() or "length" in rules.lower()

    def test_get_self_validation(self, builder):
        """Test self-validation is a checklist."""
        validation = builder.get_self_validation()

        assert isinstance(validation, list)
        assert len(validation) >= 3

    def test_get_examples(self, builder):
        """Test examples include correct and incorrect."""
        examples = builder.get_examples()

        assert isinstance(examples, list)
        assert len(examples) >= 2

        # Should have at least one correct and one incorrect
        has_correct = any(e.is_correct for e in examples)
        has_incorrect = any(not e.is_correct for e in examples)

        assert has_correct
        assert has_incorrect

    def test_get_output_contract(self, builder):
        """Test output contract is defined."""
        contract = builder.get_output_contract()

        assert len(contract) > 50
        assert "format" in contract.lower() or "output" in contract.lower()

    def test_build_complete_prompt(self, builder):
        """Test building complete prompt."""
        prompt = builder.build()

        # Should have all required sections
        assert "ROLE" in prompt
        assert "CONTEXT" in prompt
        assert "INPUT" in prompt
        assert "RESPONSIBILITIES" in prompt
        assert "DECISION" in prompt or "RULES" in prompt
        assert "VALIDATION" in prompt
        assert "EXAMPLES" in prompt or "✅" in prompt
        assert "OUTPUT" in prompt or "CONTRACT" in prompt

    def test_build_user_prompt(self, builder):
        """Test building user prompt with content."""
        content = "This is sample document content about Apache Kafka."
        user_prompt = builder.build_user_prompt(content)

        assert "document" in user_prompt.lower()
        assert content in user_prompt

    def test_build_user_prompt_includes_language(self, french_builder):
        """Test user prompt includes target language."""
        user_prompt = french_builder.build_user_prompt("Sample content")

        assert "fr" in user_prompt


class TestStructureExtractionPromptBuilder:
    """Tests for StructureExtractionPromptBuilder."""

    @pytest.fixture
    def builder(self):
        """Create default builder."""
        return StructureExtractionPromptBuilder()

    @pytest.fixture
    def youtube_builder(self):
        """Create YouTube-specific builder."""
        return StructureExtractionPromptBuilder(
            source_type="youtube",
            has_chapters=False,
            content_language="fr",
        )

    def test_get_role(self, builder):
        """Test role definition."""
        role = builder.get_role()

        assert "Document Structure Analyst" in role or "structure" in role.lower()
        assert "AUTONOMOUSLY" in role

    def test_get_context_youtube(self, youtube_builder):
        """Test context for YouTube source."""
        context = youtube_builder.get_context()

        assert "YouTube" in context or "video" in context.lower()

    def test_get_responsibilities(self, builder):
        """Test responsibilities include structure tasks."""
        responsibilities = builder.get_responsibilities()

        assert isinstance(responsibilities, list)
        assert len(responsibilities) >= 5

        resp_text = " ".join(responsibilities).lower()
        assert "section" in resp_text
        assert "hierarchy" in resp_text or "structure" in resp_text

    def test_get_decision_rules_section_count(self, builder):
        """Test decision rules include section count constraint."""
        rules = builder.get_decision_rules()

        assert "3-7" in rules or ("3" in rules and "7" in rules)

    def test_get_decision_rules_generic_titles_forbidden(self, builder):
        """Test that generic titles are forbidden."""
        rules = builder.get_decision_rules()

        assert "generic" in rules.lower() or "Introduction" in rules

    def test_get_examples_use_tree_notation(self, builder):
        """Test examples use tree notation."""
        examples = builder.get_examples()

        # At least one example should have tree notation
        all_outputs = " ".join(e.output for e in examples)
        assert "┌──" in all_outputs or "├──" in all_outputs

    def test_get_output_contract_tree_format(self, builder):
        """Test output contract specifies tree format."""
        contract = builder.get_output_contract()

        assert "tree" in contract.lower() or "┌──" in contract

    def test_get_additional_sections_youtube(self, youtube_builder):
        """Test additional sections for YouTube."""
        sections = youtube_builder.get_additional_sections()

        # YouTube should have transcript hints
        assert len(sections) > 0
        section_content = " ".join(s.content for s in sections)
        assert "transcript" in section_content.lower() or "transition" in section_content.lower()

    def test_get_additional_sections_document(self, builder):
        """Test additional sections for regular document."""
        sections = builder.get_additional_sections()

        # Regular document should have no extra sections
        assert len(sections) == 0

    def test_build_complete_prompt(self, builder):
        """Test building complete prompt."""
        prompt = builder.build()

        # Should have tree notation in examples
        assert "┌──" in prompt

        # Should mention section count
        assert "3" in prompt and "7" in prompt

    def test_build_user_prompt(self, builder):
        """Test building user prompt."""
        content = "# Chapter 1\nIntroduction to Kafka..."
        user_prompt = builder.build_user_prompt(content)

        assert content in user_prompt
        assert "CONTENT" in user_prompt or "content" in user_prompt.lower()


class TestPromptBuilderIntegration:
    """Integration tests for prompt builders."""

    def test_summary_prompt_length_reasonable(self):
        """Test that summary prompt isn't too long."""
        builder = DocumentSummaryPromptBuilder()
        prompt = builder.build()

        # Should be comprehensive but not excessive
        assert len(prompt) > 500  # Has content
        assert len(prompt) < 10000  # Not too long

    def test_structure_prompt_length_reasonable(self):
        """Test that structure prompt isn't too long."""
        builder = StructureExtractionPromptBuilder()
        prompt = builder.build()

        assert len(prompt) > 500
        assert len(prompt) < 10000

    def test_prompts_have_visual_separators(self):
        """Test that prompts use visual separators."""
        summary_builder = DocumentSummaryPromptBuilder()
        structure_builder = StructureExtractionPromptBuilder()

        summary_prompt = summary_builder.build()
        structure_prompt = structure_builder.build()

        # Should have visual separators (box drawing characters)
        assert "╔" in summary_prompt or "┌" in summary_prompt
        assert "╔" in structure_prompt or "┌" in structure_prompt

    def test_all_prompts_have_examples(self):
        """Test that all prompt builders have examples."""
        builders = [
            DocumentSummaryPromptBuilder(),
            StructureExtractionPromptBuilder(),
        ]

        for builder in builders:
            examples = builder.get_examples()
            assert len(examples) >= 2, f"{builder.__class__.__name__} needs examples"

    def test_all_prompts_have_validation(self):
        """Test that all prompt builders have validation."""
        builders = [
            DocumentSummaryPromptBuilder(),
            StructureExtractionPromptBuilder(),
        ]

        for builder in builders:
            validation = builder.get_self_validation()
            assert len(validation) >= 3, f"{builder.__class__.__name__} needs validation"
