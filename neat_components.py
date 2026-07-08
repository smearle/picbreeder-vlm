"""Backward-compatibility shim for pickled data.

``neat_components`` moved to ``picbreeder_vlm.core.neat_components`` during the 2026 repo restructure.
Pickled genomes / checkpoints store the *original* module path, so this thin
shim keeps ``import neat_components`` working (e.g. ``pickle.load`` of an archive .pkl).
New code should import from ``picbreeder_vlm.core.neat_components`` directly.
"""
from picbreeder_vlm.core.neat_components import *  # noqa: F401,F403
from picbreeder_vlm.core import neat_components as _m  # noqa: F401
import sys as _sys
_sys.modules[__name__].__dict__.update({k: v for k, v in _m.__dict__.items() if not k.startswith('__')})
