"""
Conftest for rag_enforcement tests.

Handles the namespace conflict between presentation-generator's models/
package and rag_enforcement's models.py module by pre-loading all
rag_enforcement modules with correct import resolution.
"""

import sys
import os
import importlib.util

_rag_enforcement_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "services",
    "rag_enforcement",
)

# Load rag_enforcement/models.py under a unique module name so test files
# can import from it without conflicting with presentation-generator/models/
_models_spec = importlib.util.spec_from_file_location(
    "rag_enforcement_models",
    os.path.join(_rag_enforcement_path, "models.py"),
)
rag_models = importlib.util.module_from_spec(_models_spec)
sys.modules["rag_enforcement_models"] = rag_models
_models_spec.loader.exec_module(rag_models)

# Temporarily replace the cached 'models' in sys.modules with our rag_enforcement
# models so that the source files' fallback `from models import X` resolves correctly.
sys.path.insert(0, _rag_enforcement_path)
_saved_models = {}
for key in list(sys.modules):
    if key == "models" or key.startswith("models."):
        _saved_models[key] = sys.modules.pop(key)
sys.modules["models"] = rag_models

# Pre-import source modules — they'll find the right 'models' and cache properly
import citation_validator  # noqa: E402
import sentence_verifier  # noqa: E402
import rag_enforcer  # noqa: E402

# Restore the original models package so other tests are unaffected
for key, mod in _saved_models.items():
    sys.modules[key] = mod
if not _saved_models:
    # No original models was cached; remove rag_enforcement's models
    sys.modules.pop("models", None)
if _rag_enforcement_path in sys.path:
    sys.path.remove(_rag_enforcement_path)
