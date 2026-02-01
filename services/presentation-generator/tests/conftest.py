"""
Pytest configuration for code_pipeline tests.

This file mocks external dependencies before any tests are collected.
"""

import sys
from unittest.mock import MagicMock

# Create a proper mock module class that supports nested attribute access
class MockModule(MagicMock):
    """A mock module that supports from X import Y syntax"""
    def __init__(self, name='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__name__ = name
        self.__file__ = f'/mock/{name}.py'
        self.__path__ = [f'/mock/{name}']
        self.__package__ = name

# Mock external modules that may not be installed in test environment
MOCK_MODULES = [
    'viralify_diagrams',
    'viralify_diagrams.core',
    'viralify_diagrams.core.diagram',
    'viralify_diagrams.core.theme',
    'viralify_diagrams.layouts',
    'viralify_diagrams.layouts.base',
    'viralify_diagrams.layouts.horizontal',
    'viralify_diagrams.layouts.vertical',
    'viralify_diagrams.layouts.grid',
    'viralify_diagrams.layouts.radial',
    'viralify_diagrams.exporters',
    'viralify_diagrams.exporters.svg',
    'viralify_diagrams.exporters.png',
    'viralify_diagrams.narration',
    'viralify_diagrams.narration.narrator',
    'shared',
    'shared.llm_provider',
    'shared.training_logger',
    'cairosvg',
    'pygraphviz',
    'diagrams',
    'diagrams.aws',
    'diagrams.aws.compute',
    'diagrams.aws.database',
    'diagrams.aws.network',
    'diagrams.aws.storage',
    'diagrams.aws.integration',
    'diagrams.azure',
    'diagrams.gcp',
    'diagrams.k8s',
    'diagrams.onprem',
    'diagrams.programming',
    'diagrams.saas',
    'diagrams.generic',
]

for mod_name in MOCK_MODULES:
    if mod_name not in sys.modules:
        mock = MockModule(mod_name)
        # Create common attributes that might be imported
        mock.NodeShape = MagicMock()
        mock.EdgeStyle = MagicMock()
        mock.EdgeDirection = MagicMock()
        mock.Diagram = MagicMock()
        mock.Node = MagicMock()
        mock.Edge = MagicMock()
        mock.Theme = MagicMock()
        mock.ThemeManager = MagicMock()
        mock.HorizontalLayout = MagicMock()
        mock.VerticalLayout = MagicMock()
        mock.GridLayout = MagicMock()
        mock.RadialLayout = MagicMock()
        mock.SVGExporter = MagicMock()
        mock.PNGExporter = MagicMock()
        mock.DiagramNarrator = MagicMock()
        mock.get_llm_client = MagicMock()
        mock.get_model_name = MagicMock(return_value="gpt-4o")
        mock.log_training_example = MagicMock()
        mock.TaskType = MagicMock()
        sys.modules[mod_name] = mock

# Mock OpenAI if not installed
try:
    import openai
except ImportError:
    sys.modules['openai'] = MockModule('openai')

# Common pytest fixtures
import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
