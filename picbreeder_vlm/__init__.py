"""picbreeder_vlm: In Search of the Ingredients of Open-Endedness.

Reimplementing Picbreeder with Large Vision-Language Models.
See README.md for an overview and the subpackage docstrings for a map.
"""
# Keep the repository root importable so the auxiliary top-level packages that
# live beside this one -- tools/, archive_animations/, fer/ and the
# pickle-compat shims (neat_components.py, config.py, ...) -- resolve whenever
# picbreeder_vlm is imported, regardless of the current working directory.
import sys as _sys
from ._paths import REPO_ROOT as _REPO_ROOT

_repo_root = str(_REPO_ROOT)
if _repo_root not in _sys.path:
    _sys.path.append(_repo_root)
