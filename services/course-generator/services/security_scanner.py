"""
Security Scanner Service

Validates documents for security threats before processing.
Implements multiple security checks to ensure uploaded files are safe.
"""
import hashlib
import io
import magic
import os
import re
import zipfile
from pathlib import Path
from typing import BinaryIO, Optional, Tuple

from models.document_models import (
    Document,
    DocumentType,
    SecurityScanResult,
    ALLOWED_MIME_TYPES,
    EXTENSION_TO_TYPE,
    MAX_FILE_SIZE,
    MAX_FILE_SIZE_BY_TYPE,
)


class SecurityScanner:
    """
    Security scanner for uploaded documents.

    Implements:
    - MIME type validation
    - Extension verification
    - File size limits
    - Macro detection (Office files)
    - Embedded object detection
    - Malicious pattern detection
    """

    # Suspicious patterns in text content
    SUSPICIOUS_PATTERNS = [
        # Script injection
        r'<script[^>]*>',
        r'javascript:',
        r'vbscript:',
        r'on\w+\s*=',
        # Executable patterns
        r'\.exe\b',
        r'\.dll\b',
        r'\.bat\b',
        r'\.cmd\b',
        r'\.ps1\b',
        r'\.vbs\b',
        # Macro patterns
        r'AutoOpen',
        r'AutoExec',
        r'Document_Open',
        r'Workbook_Open',
        # SQL injection patterns
        r';\s*DROP\s+TABLE',
        r';\s*DELETE\s+FROM',
        r'UNION\s+SELECT',
        # Shell commands
        r'\$\([^)]+\)',
        r'`[^`]+`',
        r'\|\s*bash',
        r'\|\s*sh\b',
    ]

    # Files that should never be in uploaded documents
    DANGEROUS_EXTENSIONS = {
        '.exe', '.dll', '.bat', '.cmd', '.ps1', '.vbs', '.js',
        '.jar', '.msi', '.scr', '.com', '.pif', '.hta', '.cpl',
    }

    def __init__(self):
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.SUSPICIOUS_PATTERNS
        ]

    async def scan_file(
        self,
        file_content: bytes,
        filename: str,
        declared_type: Optional[DocumentType] = None,
    ) -> SecurityScanResult:
        """
        Perform comprehensive security scan on uploaded file.

        Args:
            file_content: Raw file bytes
            filename: Original filename
            declared_type: Type declared by user (optional)

        Returns:
            SecurityScanResult with all findings
        """
        print(f"[SECURITY] Scanning file: {filename}", flush=True)

        result = SecurityScanResult(
            is_safe=True,
            threats_found=[],
            warnings=[],
        )

        try:
            # 1. Check file size
            self._check_file_size(file_content, filename, declared_type, result)

            # 2. Validate MIME type
            self._check_mime_type(file_content, filename, result)

            # 3. Check extension matches content
            self._check_extension_match(file_content, filename, result)

            # 4. Check for macros (Office files)
            await self._check_macros(file_content, filename, result)

            # 5. Check for embedded objects
            await self._check_embedded_objects(file_content, filename, result)

            # 6. Scan for suspicious patterns
            self._check_suspicious_patterns(file_content, filename, result)

            # 7. Check ZIP-based files for dangerous content
            await self._check_zip_contents(file_content, filename, result)

            # Determine final safety status
            result.is_safe = len(result.threats_found) == 0

            print(f"[SECURITY] Scan complete: {'SAFE' if result.is_safe else 'UNSAFE'}", flush=True)
            if result.threats_found:
                print(f"[SECURITY] Threats: {result.threats_found}", flush=True)
            if result.warnings:
                print(f"[SECURITY] Warnings: {result.warnings}", flush=True)

        except Exception as e:
            print(f"[SECURITY] Scan error: {e}", flush=True)
            result.is_safe = False
            result.threats_found.append(f"Scan error: {str(e)}")

        return result

    def _check_file_size(
        self,
        content: bytes,
        filename: str,
        declared_type: Optional[DocumentType],
        result: SecurityScanResult,
    ) -> None:
        """Check if file size is within limits"""
        file_size = len(content)
        result.scan_details["file_size"] = file_size

        # Get type-specific limit
        max_size = MAX_FILE_SIZE
        if declared_type and declared_type in MAX_FILE_SIZE_BY_TYPE:
            max_size = MAX_FILE_SIZE_BY_TYPE[declared_type]

        if file_size > max_size:
            result.file_size_ok = False
            result.threats_found.append(
                f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds limit ({max_size / 1024 / 1024:.1f}MB)"
            )
        elif file_size > max_size * 0.8:
            result.warnings.append(
                f"File size ({file_size / 1024 / 1024:.1f}MB) approaching limit"
            )

    def _check_mime_type(
        self,
        content: bytes,
        filename: str,
        result: SecurityScanResult,
    ) -> None:
        """Validate MIME type using libmagic"""
        try:
            detected_mime = magic.from_buffer(content, mime=True)
            result.scan_details["detected_mime"] = detected_mime

            if detected_mime not in ALLOWED_MIME_TYPES:
                result.mime_type_valid = False
                result.threats_found.append(
                    f"Unsupported or suspicious MIME type: {detected_mime}"
                )
        except Exception as e:
            result.warnings.append(f"Could not detect MIME type: {e}")

    def _check_extension_match(
        self,
        content: bytes,
        filename: str,
        result: SecurityScanResult,
    ) -> None:
        """Check if file extension matches actual content type"""
        ext = Path(filename).suffix.lower()
        result.scan_details["extension"] = ext

        if ext not in EXTENSION_TO_TYPE:
            result.extension_matches_content = False
            result.threats_found.append(f"Unsupported file extension: {ext}")
            return

        expected_type = EXTENSION_TO_TYPE[ext]

        try:
            detected_mime = magic.from_buffer(content, mime=True)
            actual_type = ALLOWED_MIME_TYPES.get(detected_mime)

            if actual_type and actual_type != expected_type:
                # Some flexibility for related types
                related_types = {
                    (DocumentType.DOC, DocumentType.DOCX),
                    (DocumentType.XLS, DocumentType.XLSX),
                    (DocumentType.PPT, DocumentType.PPTX),
                }
                if (expected_type, actual_type) not in related_types and \
                   (actual_type, expected_type) not in related_types:
                    result.extension_matches_content = False
                    result.threats_found.append(
                        f"Extension {ext} does not match content type {detected_mime}"
                    )
        except Exception:
            pass  # Already handled in MIME check

    async def _check_macros(
        self,
        content: bytes,
        filename: str,
        result: SecurityScanResult,
    ) -> None:
        """Check for VBA macros in Office files"""
        ext = Path(filename).suffix.lower()

        # Check modern Office formats (ZIP-based)
        if ext in ['.docx', '.xlsx', '.pptx']:
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for name in zf.namelist():
                        # Check for VBA project files
                        if 'vbaProject' in name.lower():
                            result.no_macros = False
                            result.threats_found.append(
                                "Document contains VBA macros (not allowed)"
                            )
                            return

                        # Check for macro-enabled content types
                        if '[Content_Types].xml' in name:
                            ct_content = zf.read(name).decode('utf-8', errors='ignore')
                            if 'macro' in ct_content.lower():
                                result.no_macros = False
                                result.threats_found.append(
                                    "Document contains macro-enabled content"
                                )
                                return
            except zipfile.BadZipFile:
                pass  # Not a valid ZIP, will be caught elsewhere

        # Check legacy Office formats
        elif ext in ['.doc', '.xls', '.ppt']:
            # Look for OLE compound document with VBA
            if b'_VBA_PROJECT' in content or b'VBA' in content:
                result.no_macros = False
                result.threats_found.append(
                    "Legacy Office document may contain macros (not allowed)"
                )

    async def _check_embedded_objects(
        self,
        content: bytes,
        filename: str,
        result: SecurityScanResult,
    ) -> None:
        """Check for embedded objects in documents"""
        ext = Path(filename).suffix.lower()

        if ext in ['.docx', '.xlsx', '.pptx']:
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for name in zf.namelist():
                        # Check for embedded OLE objects
                        if 'embeddings' in name.lower():
                            embedded_ext = Path(name).suffix.lower()
                            if embedded_ext in self.DANGEROUS_EXTENSIONS:
                                result.no_embedded_objects = False
                                result.threats_found.append(
                                    f"Document contains dangerous embedded file: {name}"
                                )
                            else:
                                result.warnings.append(
                                    f"Document contains embedded object: {name}"
                                )
            except zipfile.BadZipFile:
                pass

        # Check PDF for embedded files
        elif ext == '.pdf':
            if b'/EmbeddedFile' in content or b'/JavaScript' in content:
                result.warnings.append(
                    "PDF may contain embedded files or JavaScript"
                )
                # Check for specific dangerous patterns
                if b'/Launch' in content or b'/URI' in content:
                    result.warnings.append(
                        "PDF contains launch actions or URIs"
                    )

    def _check_suspicious_patterns(
        self,
        content: bytes,
        filename: str,
        result: SecurityScanResult,
    ) -> None:
        """Scan content for suspicious patterns"""
        try:
            # Try to decode as text
            text_content = content.decode('utf-8', errors='ignore')

            for pattern in self.compiled_patterns:
                matches = pattern.findall(text_content)
                if matches:
                    result.warnings.append(
                        f"Suspicious pattern detected: {pattern.pattern[:50]}"
                    )
        except Exception:
            pass  # Binary file, skip text pattern check

    async def _check_zip_contents(
        self,
        content: bytes,
        filename: str,
        result: SecurityScanResult,
    ) -> None:
        """Check contents of ZIP-based files"""
        ext = Path(filename).suffix.lower()

        # ZIP-based Office formats
        if ext in ['.docx', '.xlsx', '.pptx', '.zip']:
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for name in zf.namelist():
                        file_ext = Path(name).suffix.lower()

                        # Check for dangerous files
                        if file_ext in self.DANGEROUS_EXTENSIONS:
                            result.threats_found.append(
                                f"Archive contains dangerous file: {name}"
                            )

                        # Check for path traversal
                        if '..' in name or name.startswith('/'):
                            result.threats_found.append(
                                f"Archive contains path traversal attempt: {name}"
                            )

                        # Check compressed size vs uncompressed (zip bomb)
                        info = zf.getinfo(name)
                        if info.compress_size > 0:
                            ratio = info.file_size / info.compress_size
                            if ratio > 100:  # Compression ratio > 100:1
                                result.threats_found.append(
                                    f"Possible zip bomb detected: {name} (ratio: {ratio:.0f}:1)"
                                )

            except zipfile.BadZipFile:
                if ext in ['.docx', '.xlsx', '.pptx']:
                    result.threats_found.append(
                        "Office file is not a valid ZIP archive"
                    )

    def compute_file_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash of file content"""
        return hashlib.sha256(content).hexdigest()

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path injection"""
        # Remove path components
        filename = Path(filename).name

        # Remove null bytes and control characters
        filename = re.sub(r'[\x00-\x1f]', '', filename)

        # Replace dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

        # Limit length
        if len(filename) > 200:
            ext = Path(filename).suffix
            filename = filename[:200 - len(ext)] + ext

        return filename
