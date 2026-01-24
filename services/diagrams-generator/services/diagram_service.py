"""
Main diagram generation service.

This service orchestrates diagram generation using the Python diagrams library.
It validates imports, executes code safely, and returns professional-quality diagrams.
"""

import os
import re
import uuid
import base64
from typing import Optional, Dict, Any
from pathlib import Path

from .import_validator import ImportValidator
from .code_executor import CodeExecutor
from ..models.diagram_models import (
    DiagramRequest,
    DiagramResponse,
    DiagramType,
    CloudProvider,
    ValidationResult,
)


class DiagramService:
    """Service for generating diagrams using the Python diagrams library."""

    def __init__(self):
        """Initialize the diagram service."""
        self.executor = CodeExecutor()
        self.output_dir = Path("/tmp/diagrams")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, request: DiagramRequest) -> DiagramResponse:
        """
        Generate a diagram from a request.

        Args:
            request: The diagram generation request

        Returns:
            DiagramResponse with the generated diagram or error
        """
        try:
            # If Python code is provided, validate and execute it
            if request.python_code:
                return self._generate_from_code(request.python_code)

            # Otherwise, generate code from description
            return self._generate_from_description(request)

        except Exception as e:
            return DiagramResponse(
                success=False,
                error=f"Diagram generation failed: {str(e)}"
            )

    def _generate_from_code(self, code: str) -> DiagramResponse:
        """
        Generate a diagram from provided Python code.

        Args:
            code: Python code for diagram generation

        Returns:
            DiagramResponse with the result
        """
        # Validate and fix imports
        fixed_code, errors, warnings = ImportValidator.fix_imports(code)

        validation = ValidationResult(
            is_valid=len(errors) == 0,
            corrected_code=fixed_code if warnings else None,
            errors=errors,
            warnings=warnings
        )

        # Log validation results
        if warnings:
            print(f"[DIAGRAMS] Fixed {len(warnings)} imports: {warnings}", flush=True)
        if errors:
            print(f"[DIAGRAMS] Import errors: {errors}", flush=True)

        # Validate code syntax
        is_valid, syntax_errors = self.executor.validate_code(fixed_code)
        if not is_valid:
            return DiagramResponse(
                success=False,
                error=f"Code validation failed: {'; '.join(syntax_errors)}",
                validation=validation
            )

        # Execute the code
        output_filename = f"diagram_{uuid.uuid4().hex[:8]}"
        success, output_path, error = self.executor.execute(fixed_code, output_filename)

        if success and output_path:
            # Read the image and encode as base64
            try:
                with open(output_path, 'rb') as f:
                    image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')

                return DiagramResponse(
                    success=True,
                    image_path=output_path,
                    image_base64=image_base64,
                    validation=validation,
                    metadata={
                        "filename": os.path.basename(output_path),
                        "size_bytes": len(image_data),
                        "imports_fixed": len(warnings),
                    }
                )
            except Exception as e:
                return DiagramResponse(
                    success=False,
                    error=f"Failed to read generated image: {str(e)}",
                    validation=validation
                )
        else:
            return DiagramResponse(
                success=False,
                error=error or "Unknown error during code execution",
                validation=validation
            )

    def _generate_from_description(self, request: DiagramRequest) -> DiagramResponse:
        """
        Generate a diagram from a text description.

        This is a placeholder for future LLM-based code generation.
        Currently returns an error as it requires external LLM.

        Args:
            request: The diagram generation request

        Returns:
            DiagramResponse with error indicating LLM is needed
        """
        return DiagramResponse(
            success=False,
            error="Code generation from description requires python_code parameter. "
                  "Use an LLM to generate the code first.",
            metadata={
                "suggestion": "Call an LLM API with the description to generate Python diagrams code, "
                              "then pass the generated code to this service."
            }
        )

    def validate_code(self, code: str) -> ValidationResult:
        """
        Validate diagram code without executing.

        Args:
            code: Python code to validate

        Returns:
            ValidationResult with validation status and fixes
        """
        fixed_code, errors, warnings = ImportValidator.fix_imports(code)

        # Also check syntax
        is_valid, syntax_errors = self.executor.validate_code(fixed_code)
        if not is_valid:
            errors.extend(syntax_errors)

        return ValidationResult(
            is_valid=len(errors) == 0,
            corrected_code=fixed_code if warnings or errors else None,
            errors=errors,
            warnings=warnings
        )

    def get_available_icons(self, provider: CloudProvider) -> Dict[str, list]:
        """
        Get available icons for a cloud provider.

        Args:
            provider: The cloud provider

        Returns:
            Dictionary mapping module paths to available icon names
        """
        provider_prefixes = {
            CloudProvider.AWS: "diagrams.aws",
            CloudProvider.AZURE: "diagrams.azure",
            CloudProvider.GCP: "diagrams.gcp",
            CloudProvider.KUBERNETES: "diagrams.k8s",
            CloudProvider.ONPREM: "diagrams.onprem",
            CloudProvider.GENERIC: "diagrams.generic",
        }

        prefix = provider_prefixes.get(provider, "diagrams")

        result = {}
        for module, icons in ImportValidator.VALID_IMPORTS.items():
            if module.startswith(prefix):
                result[module] = sorted(list(icons))

        return result

    def suggest_icons(self, description: str) -> list:
        """
        Suggest icons based on a description.

        Args:
            description: Text description of the diagram

        Returns:
            List of suggested (module, icon_name) tuples
        """
        return ImportValidator.get_suggested_imports(description)

    def detect_provider(self, description: str) -> CloudProvider:
        """
        Detect the cloud provider from a description.

        Args:
            description: Text description

        Returns:
            Detected CloudProvider
        """
        description_lower = description.lower()

        if any(kw in description_lower for kw in ['aws', 'amazon', 'ec2', 's3', 'lambda', 'dynamodb']):
            return CloudProvider.AWS
        elif any(kw in description_lower for kw in ['azure', 'microsoft', 'aks', 'cosmos']):
            return CloudProvider.AZURE
        elif any(kw in description_lower for kw in ['gcp', 'google', 'gke', 'bigquery', 'cloud run']):
            return CloudProvider.GCP
        elif any(kw in description_lower for kw in ['kubernetes', 'k8s', 'kubectl', 'helm']):
            return CloudProvider.KUBERNETES
        elif any(kw in description_lower for kw in ['docker', 'nginx', 'kafka', 'redis', 'postgres']):
            return CloudProvider.ONPREM

        return CloudProvider.GENERIC
