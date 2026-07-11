"""picbreeder_vlm: In Search of the Ingredients of Open-Endedness.

Reimplementing Picbreeder with Large Vision-Language Models.
See README.md for an overview and the subpackage docstrings for a map.
"""
# Keep the repository root importable so the auxiliary top-level packages that
# live beside this one -- tools/, archive_animations/ and the pickle-compat
# shims (neat_components.py, config.py, ...) -- resolve whenever picbreeder_vlm
# is imported, regardless of the current working directory. The vendored fer
# package now lives under third-party/, so put that on the path too (see
# _paths.ensure_fer_importable) to keep ``import fer.src.X`` working.
import sys as _sys
from ._paths import REPO_ROOT as _REPO_ROOT, THIRD_PARTY_ROOT as _THIRD_PARTY_ROOT

for _p in (str(_REPO_ROOT), str(_THIRD_PARTY_ROOT)):
    if _p not in _sys.path:
        _sys.path.append(_p)
