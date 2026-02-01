"""
SSVS-C - Code-Aware Synchronization Extension

Extends SSVS to handle code synchronization with:
- Code structure parsing (functions, classes, blocks)
- Element mention detection in voiceover
- Line-by-line reveal animation generation

APPROACH:
1. Parse code structure (functions, loops, variables)
2. Detect element mentions in narration
3. Generate reveal sequence with timing
4. Output FFmpeg filters for line-by-line reveal
"""

import ast
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set, Any
from enum import Enum

from pygments import lex
from pygments.lexers import get_lexer_by_name, TextLexer

from .ssvs_algorithm import VoiceSegment


# ==============================================================================
# DATA STRUCTURES FOR CODE
# ==============================================================================

class CodeElementType(Enum):
    """Types of code elements that can be detected and revealed"""
    LINE = "line"
    FUNCTION = "function"
    CLASS = "class"
    BLOCK = "block"           # if/for/while/with
    IMPORT = "import"
    VARIABLE = "variable"
    COMMENT = "comment"
    DECORATOR = "decorator"


@dataclass
class CodeElement:
    """A code element with line range and keywords"""
    id: str
    element_type: CodeElementType
    name: str
    start_line: int           # 1-indexed
    end_line: int             # Inclusive
    content: str
    keywords: List[str] = field(default_factory=list)
    importance: float = 1.0
    parent_id: Optional[str] = None

    def get_searchable_text(self) -> str:
        """Combine name and keywords for semantic matching"""
        parts = [self.name] + self.keywords
        return " ".join(filter(None, parts))


@dataclass
class CodeStructure:
    """Complete code structure representation"""
    language: str
    total_lines: int
    elements: List[CodeElement]
    raw_code: str

    def get_elements_by_type(self, elem_type: CodeElementType) -> List[CodeElement]:
        return [e for e in self.elements if e.element_type == elem_type]

    def get_element_by_id(self, element_id: str) -> Optional[CodeElement]:
        for elem in self.elements:
            if elem.id == element_id:
                return elem
        return None

    def infer_reveal_order(self) -> List[str]:
        """Infer logical reveal order: imports -> classes -> functions -> main"""
        order = []
        priority = [
            CodeElementType.IMPORT,
            CodeElementType.CLASS,
            CodeElementType.FUNCTION,
            CodeElementType.VARIABLE,
            CodeElementType.BLOCK,
            CodeElementType.LINE,
        ]
        for elem_type in priority:
            elements = sorted(
                self.get_elements_by_type(elem_type),
                key=lambda e: (e.start_line, -e.importance)
            )
            order.extend([e.id for e in elements])
        return order


@dataclass
class CodeRevealPoint:
    """A point in time when code lines should be revealed"""
    element_id: str
    start_line: int
    end_line: int
    reveal_time: float        # When to reveal
    hold_time: float          # Animation duration
    reveal_type: str = "fade" # "instant", "fade", "typewrite"
    confidence: float = 1.0

    def to_ffmpeg_instruction(self) -> Dict:
        return {
            "start_line": self.start_line,
            "end_line": self.end_line,
            "reveal_at": self.reveal_time,
            "duration": self.hold_time,
            "effect": self.reveal_type
        }


@dataclass
class CodeSyncResult:
    """Result of synchronizing code with voiceover"""
    code_id: str
    language: str
    total_lines: int
    reveal_sequence: List[CodeRevealPoint]
    element_mentions: Dict[str, List[Tuple[float, float]]]
    semantic_score: float
    coverage_score: float
    ffmpeg_filter: Optional[str] = None
    animation_timeline: Optional[Dict] = None

    def get_revealed_lines_at(self, time: float) -> Set[int]:
        """Get which lines should be visible at a given time"""
        revealed = set()
        for rp in self.reveal_sequence:
            if rp.reveal_time <= time:
                for line in range(rp.start_line, rp.end_line + 1):
                    revealed.add(line)
        return revealed


# ==============================================================================
# CODE PARSER
# ==============================================================================

class CodeParser:
    """Parse code structure for different languages"""

    @classmethod
    def parse(cls, code: str, language: str) -> CodeStructure:
        language = language.lower()
        if language in ("python", "py", "python3"):
            return PythonCodeParser().parse(code)
        else:
            return GenericCodeParser().parse(code, language)


class PythonCodeParser:
    """Parse Python code using ast module"""

    def parse(self, code: str) -> CodeStructure:
        lines = code.split('\n')
        elements = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    elem = self._parse_import(node, lines)
                    if elem:
                        elements.append(elem)

                elif isinstance(node, ast.ClassDef):
                    elem = self._parse_class(node, lines)
                    if elem:
                        elements.append(elem)

                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    elem = self._parse_function(node, lines)
                    if elem:
                        elements.append(elem)

                elif isinstance(node, (ast.For, ast.While)):
                    elem = self._parse_loop(node, lines)
                    if elem:
                        elements.append(elem)

                elif isinstance(node, ast.If):
                    elem = self._parse_conditional(node, lines)
                    if elem:
                        elements.append(elem)

                elif isinstance(node, ast.Assign):
                    elem = self._parse_assignment(node, lines)
                    if elem:
                        elements.append(elem)

        except SyntaxError:
            # Fallback to line-based parsing
            elements = self._parse_by_lines(lines)

        # Remove duplicates and sort
        seen_ids = set()
        unique_elements = []
        for elem in sorted(elements, key=lambda e: e.start_line):
            if elem.id not in seen_ids:
                seen_ids.add(elem.id)
                unique_elements.append(elem)

        return CodeStructure(
            language="python",
            total_lines=len(lines),
            elements=unique_elements,
            raw_code=code
        )

    def _parse_import(self, node, lines: List[str]) -> Optional[CodeElement]:
        start_line = node.lineno
        end_line = getattr(node, 'end_lineno', start_line) or start_line

        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            name = ", ".join(names)
        else:
            module = node.module or ""
            names = [alias.name for alias in node.names]
            name = f"{module}.{', '.join(names)}"

        content = '\n'.join(lines[start_line-1:end_line])

        return CodeElement(
            id=f"import_{start_line}",
            element_type=CodeElementType.IMPORT,
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=content,
            keywords=names + [module] if isinstance(node, ast.ImportFrom) else names,
            importance=0.8
        )

    def _parse_class(self, node: ast.ClassDef, lines: List[str]) -> CodeElement:
        start_line = node.lineno
        end_line = getattr(node, 'end_lineno', start_line) or start_line
        content = '\n'.join(lines[start_line-1:end_line])

        keywords = [node.name, node.name.lower()]
        # Add base class names
        for base in node.bases:
            if isinstance(base, ast.Name):
                keywords.append(base.id)

        return CodeElement(
            id=f"class_{node.name}_{start_line}",
            element_type=CodeElementType.CLASS,
            name=node.name,
            start_line=start_line,
            end_line=end_line,
            content=content,
            keywords=keywords,
            importance=1.0
        )

    def _parse_function(self, node, lines: List[str]) -> CodeElement:
        start_line = node.lineno
        end_line = getattr(node, 'end_lineno', start_line) or start_line
        content = '\n'.join(lines[start_line-1:end_line])

        keywords = [node.name, node.name.lower()]
        # Add argument names
        for arg in node.args.args:
            keywords.append(arg.arg)

        # Extract docstring keywords
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value
            keywords.extend(self._extract_keywords_from_text(docstring))

        return CodeElement(
            id=f"func_{node.name}_{start_line}",
            element_type=CodeElementType.FUNCTION,
            name=node.name,
            start_line=start_line,
            end_line=end_line,
            content=content,
            keywords=keywords,
            importance=1.0
        )

    def _parse_loop(self, node, lines: List[str]) -> CodeElement:
        start_line = node.lineno
        end_line = getattr(node, 'end_lineno', start_line) or start_line
        content = '\n'.join(lines[start_line-1:end_line])

        loop_type = "for" if isinstance(node, ast.For) else "while"
        keywords = [loop_type, "loop", "iterate", "boucle"]

        # Extract loop variable for for-loops
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            keywords.append(node.target.id)

        return CodeElement(
            id=f"loop_{loop_type}_{start_line}",
            element_type=CodeElementType.BLOCK,
            name=f"{loop_type} loop",
            start_line=start_line,
            end_line=end_line,
            content=content,
            keywords=keywords,
            importance=0.9
        )

    def _parse_conditional(self, node: ast.If, lines: List[str]) -> CodeElement:
        start_line = node.lineno
        end_line = getattr(node, 'end_lineno', start_line) or start_line
        content = '\n'.join(lines[start_line-1:end_line])

        return CodeElement(
            id=f"if_{start_line}",
            element_type=CodeElementType.BLOCK,
            name="if statement",
            start_line=start_line,
            end_line=end_line,
            content=content,
            keywords=["if", "condition", "check", "si", "condition"],
            importance=0.85
        )

    def _parse_assignment(self, node: ast.Assign, lines: List[str]) -> Optional[CodeElement]:
        start_line = node.lineno
        end_line = getattr(node, 'end_lineno', start_line) or start_line

        # Get variable names
        var_names = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_names.append(target.id)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        var_names.append(elt.id)

        if not var_names:
            return None

        content = '\n'.join(lines[start_line-1:end_line])
        name = ", ".join(var_names)

        return CodeElement(
            id=f"var_{name}_{start_line}",
            element_type=CodeElementType.VARIABLE,
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=content,
            keywords=var_names + [v.lower() for v in var_names],
            importance=0.7
        )

    def _parse_by_lines(self, lines: List[str]) -> List[CodeElement]:
        """Fallback: parse code line by line"""
        elements = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                elements.append(CodeElement(
                    id=f"line_{i}",
                    element_type=CodeElementType.LINE,
                    name=f"line {i}",
                    start_line=i,
                    end_line=i,
                    content=line,
                    keywords=[],
                    importance=0.5
                ))
        return elements

    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        words = re.findall(r'\b\w{3,}\b', text.lower())
        stopwords = {'the', 'and', 'for', 'this', 'that', 'with', 'from', 'are', 'was'}
        return [w for w in words if w not in stopwords][:5]


class GenericCodeParser:
    """Fallback parser using heuristics"""

    FUNCTION_PATTERNS = {
        "javascript": r'(?:async\s+)?function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(',
        "typescript": r'(?:async\s+)?function\s+(\w+)|(\w+)\s*[:=]\s*(?:async\s+)?\(',
        "java": r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(',
        "go": r'func\s+(\w+)',
        "rust": r'(?:pub\s+)?fn\s+(\w+)',
    }

    def parse(self, code: str, language: str = "generic") -> CodeStructure:
        lines = code.split('\n')
        elements = []

        # Get pattern for this language
        pattern = self.FUNCTION_PATTERNS.get(language.lower())

        if pattern:
            # Find functions using regex
            for i, line in enumerate(lines, 1):
                match = re.search(pattern, line)
                if match:
                    name = next((g for g in match.groups() if g), f"func_{i}")
                    elements.append(CodeElement(
                        id=f"func_{name}_{i}",
                        element_type=CodeElementType.FUNCTION,
                        name=name,
                        start_line=i,
                        end_line=i,  # Can't determine end without proper parsing
                        content=line,
                        keywords=[name, name.lower()],
                        importance=1.0
                    ))

        # Add lines as fallback elements
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped and not any(e.start_line == i for e in elements):
                elements.append(CodeElement(
                    id=f"line_{i}",
                    element_type=CodeElementType.LINE,
                    name=f"line {i}",
                    start_line=i,
                    end_line=i,
                    content=line,
                    keywords=[],
                    importance=0.5
                ))

        return CodeStructure(
            language=language,
            total_lines=len(lines),
            elements=elements,
            raw_code=code
        )


# ==============================================================================
# MENTION DETECTOR
# ==============================================================================

class CodeMentionDetector:
    """
    Detects when narrator mentions specific code elements.

    Patterns:
    1. Function mentions: "the function X", "we define X"
    2. Variable mentions: "the variable Y", "Y stores"
    3. Block references: "in this loop", "the if statement"
    4. Line references: "on line 5", "the first line"
    """

    CODE_PATTERNS = {
        "function": [
            r"function\s+(\w+)", r"method\s+(\w+)",
            r"define\s+(?:a\s+)?(?:function\s+)?(\w+)",
            r"call(?:ing)?\s+(\w+)", r"(\w+)\s+function",
            # French
            r"fonction\s+(\w+)", r"m[ée]thode\s+(\w+)",
            r"d[ée]finir\s+(\w+)", r"appeler\s+(\w+)",
        ],
        "variable": [
            r"variable\s+(\w+)", r"(\w+)\s+variable",
            r"store(?:s|d)?\s+in\s+(\w+)", r"(\w+)\s+stores?",
            r"assign\s+to\s+(\w+)", r"set\s+(\w+)",
            # French
            r"variable\s+(\w+)", r"stocker\s+dans\s+(\w+)",
        ],
        "loop": [
            r"(?:in\s+)?(?:this|the)\s+loop", r"for\s+(?:each|loop)",
            r"while\s+loop", r"iterate", r"iteration",
            # French
            r"(?:dans\s+)?(?:cette|la)\s+boucle", r"it[ée]rer",
        ],
        "conditional": [
            r"if\s+(?:statement|condition)", r"(?:this|the)\s+condition",
            r"check(?:ing)?\s+(?:if|whether)", r"else\s+(?:branch|clause)",
            # French
            r"(?:cette|la)\s+condition", r"v[ée]rifier\s+si",
        ],
        "class": [
            r"class\s+(\w+)", r"(\w+)\s+class",
            r"define\s+(?:a\s+)?class\s+(\w+)",
            # French
            r"classe\s+(\w+)", r"d[ée]finir\s+(?:la\s+)?classe\s+(\w+)",
        ],
        "line": [
            r"line\s+(\d+)", r"on\s+line\s+(\d+)",
            r"(?:the\s+)?first\s+line", r"(?:the\s+)?last\s+line",
            # French
            r"ligne\s+(\d+)", r"[àa]\s+la\s+ligne\s+(\d+)",
            r"(?:la\s+)?premi[èe]re\s+ligne", r"(?:la\s+)?derni[èe]re\s+ligne",
        ],
    }

    FLOW_MARKERS = {
        "next": ["next", "then", "after", "ensuite", "puis", "apr[èe]s"],
        "first": ["first", "start", "begin", "d'abord", "commencer", "d[ée]but"],
        "finally": ["finally", "lastly", "end", "enfin", "finalement", "fin"],
        "now": ["now", "here", "maintenant", "ici"],
    }

    def __init__(self, similarity_threshold: float = 0.45):
        self.similarity_threshold = similarity_threshold

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())

    def detect_element_mentions(
        self,
        segment: VoiceSegment,
        code_structure: CodeStructure
    ) -> List[Tuple[str, float]]:
        """
        Detect which code elements are mentioned in this segment.

        Returns:
            List of (element_id, confidence_score) sorted by confidence
        """
        mentions = []
        segment_text = self._normalize_text(segment.text)

        for element in code_structure.elements:
            confidence = 0.0

            # 1. Exact name match
            element_name = element.name.lower()
            if element_name and len(element_name) > 2 and element_name in segment_text:
                confidence = max(confidence, 0.95)

            # 2. Keyword match
            for keyword in element.keywords:
                kw_lower = keyword.lower()
                if len(kw_lower) > 2 and kw_lower in segment_text:
                    confidence = max(confidence, 0.75)
                    break

            # 3. Pattern-based matching
            pattern_match = self._match_patterns(segment_text, element)
            if pattern_match > 0:
                confidence = max(confidence, pattern_match)

            # 4. Element type mention
            type_confidence = self._check_type_mention(segment_text, element.element_type)
            if type_confidence > 0:
                confidence = max(confidence, type_confidence * 0.6)

            if confidence > 0.3:
                mentions.append((element.id, confidence))

        # Sort by confidence descending
        mentions.sort(key=lambda x: -x[1])
        return mentions

    def _match_patterns(self, text: str, element: CodeElement) -> float:
        """Check if text matches patterns for this element type"""
        type_patterns = {
            CodeElementType.FUNCTION: self.CODE_PATTERNS["function"],
            CodeElementType.CLASS: self.CODE_PATTERNS["class"],
            CodeElementType.VARIABLE: self.CODE_PATTERNS["variable"],
            CodeElementType.BLOCK: self.CODE_PATTERNS["loop"] + self.CODE_PATTERNS["conditional"],
        }

        patterns = type_patterns.get(element.element_type, [])

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Check if matched group equals element name
                groups = [g for g in match.groups() if g]
                if groups:
                    for group in groups:
                        if group.lower() == element.name.lower():
                            return 0.9
                    return 0.6
                return 0.5

        return 0

    def _check_type_mention(self, text: str, element_type: CodeElementType) -> float:
        """Check if the element type is mentioned generically"""
        type_keywords = {
            CodeElementType.FUNCTION: ["function", "method", "def", "fonction", "methode"],
            CodeElementType.CLASS: ["class", "classe"],
            CodeElementType.VARIABLE: ["variable", "value", "valeur"],
            CodeElementType.BLOCK: ["loop", "if", "condition", "boucle"],
            CodeElementType.IMPORT: ["import", "library", "module", "biblioth"],
        }

        keywords = type_keywords.get(element_type, [])
        for kw in keywords:
            if kw in text:
                return 0.7

        return 0

    def detect_flow_reference(self, segment: VoiceSegment) -> Optional[str]:
        """Detect flow markers like 'next', 'first', 'now'"""
        text_lower = segment.text.lower()

        for flow_type, markers in self.FLOW_MARKERS.items():
            for marker in markers:
                if re.search(marker, text_lower):
                    return flow_type

        return None

    def detect_line_reference(self, segment: VoiceSegment) -> Optional[int]:
        """Detect explicit line number references"""
        for pattern in self.CODE_PATTERNS["line"]:
            match = re.search(pattern, segment.text, re.IGNORECASE)
            if match and match.groups():
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    pass

        # Check for "first line" / "last line"
        text_lower = segment.text.lower()
        if "first line" in text_lower or "premiere ligne" in text_lower:
            return 1
        if "last line" in text_lower or "derniere ligne" in text_lower:
            return -1  # Special marker for last line

        return None


# ==============================================================================
# CODE SYNCHRONIZER (MAIN ALGORITHM)
# ==============================================================================

class CodeAwareSynchronizer:
    """
    SSVS-C: Code-Aware Synchronization Algorithm

    Algorithm:
    1. Parse code structure (functions, blocks, variables)
    2. Detect element mentions in each voice segment
    3. Generate reveal sequence with timing
    4. Optimize reveal points for smooth animation
    """

    def __init__(self):
        self.code_parser = CodeParser()
        self.mention_detector = CodeMentionDetector()
        self.min_reveal_duration = 0.5
        self.default_reveal_type = "fade"

    def synchronize(
        self,
        code: str,
        language: str,
        segments: List[VoiceSegment]
    ) -> CodeSyncResult:
        """
        Synchronize code revelation with voice narration.

        Args:
            code: Source code to reveal
            language: Programming language
            segments: Voice segments

        Returns:
            CodeSyncResult with reveal sequence
        """
        if not code or not segments:
            return self._empty_result(code, language)

        total_lines = len(code.split('\n'))
        print(f"[SSVS-C] Synchronizing {language} code ({total_lines} lines) "
              f"with {len(segments)} segments", flush=True)

        # Step 1: Parse code structure
        code_structure = self.code_parser.parse(code, language)

        # Step 2: Detect mentions per segment
        segment_mentions: Dict[int, List[Tuple[str, float]]] = {}
        for segment in segments:
            mentions = self.mention_detector.detect_element_mentions(segment, code_structure)
            segment_mentions[segment.id] = mentions

        # Step 3: Generate reveal sequence
        reveal_sequence = self._generate_reveal_sequence(
            code_structure, segments, segment_mentions
        )

        # Step 4: Ensure complete coverage
        reveal_sequence = self._ensure_coverage(
            reveal_sequence, code_structure, segments
        )

        # Step 5: Optimize timing
        reveal_sequence = self._optimize_reveal_timing(reveal_sequence)

        # Compute metrics
        element_mentions = self._compute_element_mentions(segment_mentions)
        coverage_score = self._compute_coverage(reveal_sequence, code_structure)
        semantic_score = self._compute_semantic_score(segment_mentions)

        print(f"[SSVS-C] Sync complete. Semantic: {semantic_score:.3f}, "
              f"Coverage: {coverage_score:.1%}, Reveal points: {len(reveal_sequence)}", flush=True)

        return CodeSyncResult(
            code_id=f"code_{hash(code) % 10000}",
            language=language,
            total_lines=code_structure.total_lines,
            reveal_sequence=reveal_sequence,
            element_mentions=element_mentions,
            semantic_score=semantic_score,
            coverage_score=coverage_score
        )

    def _generate_reveal_sequence(
        self,
        code_structure: CodeStructure,
        segments: List[VoiceSegment],
        segment_mentions: Dict[int, List[Tuple[str, float]]]
    ) -> List[CodeRevealPoint]:
        """Generate reveal points based on detected mentions"""
        reveal_sequence = []
        revealed_lines: Set[int] = set()
        reading_order = code_structure.infer_reveal_order()
        reading_order_idx = 0

        for segment in segments:
            mentions = segment_mentions.get(segment.id, [])

            if mentions:
                # Reveal mentioned elements
                for elem_id, confidence in mentions[:2]:  # Top 2 mentions
                    element = code_structure.get_element_by_id(elem_id)
                    if element and not self._lines_revealed(
                        element.start_line, element.end_line, revealed_lines
                    ):
                        reveal_point = CodeRevealPoint(
                            element_id=elem_id,
                            start_line=element.start_line,
                            end_line=element.end_line,
                            reveal_time=segment.start_time,
                            hold_time=self._calculate_hold_time(element),
                            reveal_type=self.default_reveal_type,
                            confidence=confidence
                        )
                        reveal_sequence.append(reveal_point)
                        self._mark_revealed(
                            element.start_line, element.end_line, revealed_lines
                        )
            else:
                # No explicit mentions - check for flow markers
                flow = self.mention_detector.detect_flow_reference(segment)
                if flow in ("next", "now", "first"):
                    # Reveal next element in reading order
                    while reading_order_idx < len(reading_order):
                        next_elem_id = reading_order[reading_order_idx]
                        reading_order_idx += 1
                        element = code_structure.get_element_by_id(next_elem_id)
                        if element and not self._lines_revealed(
                            element.start_line, element.end_line, revealed_lines
                        ):
                            reveal_point = CodeRevealPoint(
                                element_id=next_elem_id,
                                start_line=element.start_line,
                                end_line=element.end_line,
                                reveal_time=segment.start_time,
                                hold_time=self._calculate_hold_time(element),
                                reveal_type=self.default_reveal_type,
                                confidence=0.6
                            )
                            reveal_sequence.append(reveal_point)
                            self._mark_revealed(
                                element.start_line, element.end_line, revealed_lines
                            )
                            break

        return reveal_sequence

    def _ensure_coverage(
        self,
        reveal_sequence: List[CodeRevealPoint],
        code_structure: CodeStructure,
        segments: List[VoiceSegment]
    ) -> List[CodeRevealPoint]:
        """Ensure all lines are eventually revealed"""
        if not segments:
            return reveal_sequence

        revealed_lines: Set[int] = set()
        for rp in reveal_sequence:
            for line in range(rp.start_line, rp.end_line + 1):
                revealed_lines.add(line)

        # Find unrevealed lines
        unrevealed = []
        for line in range(1, code_structure.total_lines + 1):
            if line not in revealed_lines:
                unrevealed.append(line)

        if unrevealed:
            # Reveal remaining lines at the end
            total_duration = segments[-1].end_time
            reveal_time = max(0, total_duration - 2.0)  # 2 seconds before end

            # Group consecutive lines
            groups = self._group_consecutive_lines(unrevealed)
            time_per_group = 1.5 / max(len(groups), 1)

            for i, (start, end) in enumerate(groups):
                reveal_sequence.append(CodeRevealPoint(
                    element_id=f"remaining_{start}_{end}",
                    start_line=start,
                    end_line=end,
                    reveal_time=reveal_time + i * time_per_group,
                    hold_time=0.3,
                    reveal_type="fade",
                    confidence=0.4
                ))

        return reveal_sequence

    def _group_consecutive_lines(self, lines: List[int]) -> List[Tuple[int, int]]:
        """Group consecutive line numbers"""
        if not lines:
            return []

        groups = []
        start = lines[0]
        end = lines[0]

        for line in lines[1:]:
            if line == end + 1:
                end = line
            else:
                groups.append((start, end))
                start = line
                end = line

        groups.append((start, end))
        return groups

    def _optimize_reveal_timing(
        self,
        reveal_sequence: List[CodeRevealPoint]
    ) -> List[CodeRevealPoint]:
        """Optimize reveal timing for smooth animation"""
        if not reveal_sequence:
            return reveal_sequence

        # Sort by reveal time
        reveal_sequence.sort(key=lambda rp: rp.reveal_time)

        # Ensure minimum gap between reveals
        min_gap = 0.3
        for i in range(1, len(reveal_sequence)):
            prev = reveal_sequence[i - 1]
            curr = reveal_sequence[i]
            if curr.reveal_time < prev.reveal_time + min_gap:
                curr.reveal_time = prev.reveal_time + min_gap

        return reveal_sequence

    def _lines_revealed(self, start: int, end: int, revealed: Set[int]) -> bool:
        """Check if all lines in range are already revealed"""
        for line in range(start, end + 1):
            if line not in revealed:
                return False
        return True

    def _mark_revealed(self, start: int, end: int, revealed: Set[int]):
        """Mark lines as revealed"""
        for line in range(start, end + 1):
            revealed.add(line)

    def _calculate_hold_time(self, element: CodeElement) -> float:
        """Calculate how long a reveal animation should take"""
        lines = element.end_line - element.start_line + 1
        return max(self.min_reveal_duration, lines * 0.1)

    def _compute_element_mentions(
        self,
        segment_mentions: Dict[int, List[Tuple[str, float]]]
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Compute when each element was mentioned"""
        result: Dict[str, List[Tuple[float, float]]] = {}
        for mentions in segment_mentions.values():
            for elem_id, _ in mentions:
                if elem_id not in result:
                    result[elem_id] = []
        return result

    def _compute_coverage(
        self,
        reveal_sequence: List[CodeRevealPoint],
        code_structure: CodeStructure
    ) -> float:
        """Compute percentage of lines covered by reveals"""
        revealed = set()
        for rp in reveal_sequence:
            for line in range(rp.start_line, rp.end_line + 1):
                revealed.add(line)

        total = code_structure.total_lines
        return len(revealed) / total if total > 0 else 0

    def _compute_semantic_score(
        self,
        segment_mentions: Dict[int, List[Tuple[str, float]]]
    ) -> float:
        """Compute average confidence of detected mentions"""
        scores = []
        for mentions in segment_mentions.values():
            for _, confidence in mentions:
                scores.append(confidence)
        return sum(scores) / len(scores) if scores else 0

    def _empty_result(self, code: str, language: str) -> CodeSyncResult:
        """Return empty result for edge cases"""
        return CodeSyncResult(
            code_id="empty",
            language=language,
            total_lines=len(code.split('\n')) if code else 0,
            reveal_sequence=[],
            element_mentions={},
            semantic_score=0.0,
            coverage_score=0.0
        )


# ==============================================================================
# ANIMATION GENERATOR
# ==============================================================================

class CodeRevealAnimationGenerator:
    """Generate FFmpeg filters for line-by-line code reveal"""

    def __init__(self, video_width: int = 1920, video_height: int = 1080):
        self.width = video_width
        self.height = video_height
        # Code area boundaries (matching TypingAnimatorService)
        self.margin_x = 100
        self.margin_y = 100
        self.code_start_x = self.margin_x + 60  # After line numbers

    def generate_drawbox_filter(
        self,
        sync_result: CodeSyncResult,
        line_height: int = 32,
        code_start_y: int = 150,
        background_color: str = "#1e1e2e"
    ) -> str:
        """
        Generate FFmpeg drawbox filter to mask unrevealed lines.

        Strategy:
        - Each line has a mask that disappears at its reveal_time
        - Uses time-based enable expressions
        """
        total_lines = sync_result.total_lines
        reveals = sorted(sync_result.reveal_sequence, key=lambda r: r.reveal_time)

        if not reveals:
            return ""

        # Build line -> reveal_time mapping
        revealed_at: Dict[int, float] = {}
        for reveal in reveals:
            for line in range(reveal.start_line, reveal.end_line + 1):
                if line not in revealed_at:
                    revealed_at[line] = reveal.reveal_time

        # Default: reveal at end for any remaining lines
        total_duration = reveals[-1].reveal_time + reveals[-1].hold_time + 1.0
        for line in range(1, total_lines + 1):
            if line not in revealed_at:
                revealed_at[line] = total_duration

        # Generate mask filters
        filters = []
        code_width = self.width - self.code_start_x - self.margin_x

        for line_num, reveal_time in sorted(revealed_at.items()):
            y = code_start_y + (line_num - 1) * line_height

            # Mask this line UNTIL reveal_time
            filter_str = (
                f"drawbox=x={self.code_start_x}:y={y}:"
                f"w={code_width}:h={line_height}:"
                f"c={background_color.replace('#', '0x')}@1:t=fill:"
                f"enable='lt(t,{reveal_time:.2f})'"
            )
            filters.append(filter_str)

        return ",".join(filters)

    def generate_json_timeline(self, sync_result: CodeSyncResult) -> Dict:
        """Generate JSON timeline for custom renderers"""
        timeline = {
            "code_id": sync_result.code_id,
            "language": sync_result.language,
            "total_lines": sync_result.total_lines,
            "keyframes": []
        }

        for reveal in sync_result.reveal_sequence:
            keyframe = {
                "time": reveal.reveal_time,
                "duration": reveal.hold_time,
                "lines": {
                    "start": reveal.start_line,
                    "end": reveal.end_line
                },
                "effect": reveal.reveal_type,
                "confidence": reveal.confidence,
                "element_id": reveal.element_id
            }
            timeline["keyframes"].append(keyframe)

        return timeline
