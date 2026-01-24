"""
Safe code executor for Python diagrams generation.

This module provides secure execution of diagram generation code
with proper sandboxing and error handling.
"""

import os
import sys
import tempfile
import subprocess
import traceback
from typing import Tuple, Optional
from pathlib import Path


class CodeExecutor:
    """Executes Python diagram code safely."""

    # Maximum execution time in seconds
    TIMEOUT = 60

    # Output directory for generated diagrams
    OUTPUT_DIR = Path("/tmp/diagrams")

    # Template for diagram generation code
    CODE_TEMPLATE = '''
import os
import sys

# Set output directory
os.chdir("{output_dir}")

# Suppress graphviz output
os.environ["DIAGRAMS_SILENCE_PROMPT"] = "1"

{code}
'''

    def __init__(self):
        """Initialize the executor."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        code: str,
        output_filename: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Execute diagram generation code.

        Args:
            code: Python code to execute
            output_filename: Optional output filename (without extension)

        Returns:
            Tuple of (success, output_path, error_message)
        """
        # Generate unique output filename if not provided
        if not output_filename:
            import uuid
            output_filename = f"diagram_{uuid.uuid4().hex[:8]}"

        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            dir=self.OUTPUT_DIR,
            delete=False
        ) as f:
            # Inject output directory and filename into code
            modified_code = self._inject_output_settings(code, output_filename)
            full_code = self.CODE_TEMPLATE.format(
                output_dir=str(self.OUTPUT_DIR),
                code=modified_code
            )
            f.write(full_code)
            temp_file = f.name

        try:
            # Execute the code in a subprocess
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT,
                cwd=str(self.OUTPUT_DIR),
                env={
                    **os.environ,
                    "DIAGRAMS_SILENCE_PROMPT": "1",
                }
            )

            # Check for errors
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                return False, None, f"Code execution failed: {error_msg}"

            # Look for generated PNG file
            expected_path = self.OUTPUT_DIR / f"{output_filename}.png"
            if expected_path.exists():
                return True, str(expected_path), None

            # Try to find any generated PNG
            png_files = list(self.OUTPUT_DIR.glob(f"*{output_filename}*.png"))
            if png_files:
                # Return the most recent one
                latest = max(png_files, key=lambda p: p.stat().st_mtime)
                return True, str(latest), None

            # Also check for files without the exact name
            recent_pngs = sorted(
                self.OUTPUT_DIR.glob("*.png"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if recent_pngs:
                # Return the most recently created PNG
                return True, str(recent_pngs[0]), None

            return False, None, "No diagram file was generated"

        except subprocess.TimeoutExpired:
            return False, None, f"Execution timed out after {self.TIMEOUT} seconds"
        except Exception as e:
            return False, None, f"Execution error: {str(e)}"
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

    def _inject_output_settings(self, code: str, output_filename: str) -> str:
        """
        Inject output settings into the diagram code.

        Modifies Diagram() calls to set the correct output filename.
        """
        import re

        # Pattern to match Diagram() constructor
        diagram_pattern = re.compile(
            r'(with\s+Diagram\s*\(\s*)(["\'])([^"\']+)\2',
            re.MULTILINE
        )

        def replace_diagram(match):
            prefix = match.group(1)
            quote = match.group(2)
            title = match.group(3)
            return f'{prefix}{quote}{title}{quote}, filename="{output_filename}"'

        # Check if filename is already set
        if 'filename=' not in code:
            code = diagram_pattern.sub(replace_diagram, code)

        return code

    def validate_code(self, code: str) -> Tuple[bool, list]:
        """
        Validate Python code syntax without executing.

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []

        try:
            compile(code, '<diagram>', 'exec')
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return False, errors

        # Check for dangerous operations
        dangerous_patterns = [
            'os.system',
            'subprocess',
            'eval(',
            'exec(',
            '__import__',
            'open(',
            'file(',
            'input(',
            'raw_input',
        ]

        for pattern in dangerous_patterns:
            if pattern in code:
                errors.append(f"Potentially dangerous operation: {pattern}")

        return len(errors) == 0, errors
