"""Backward-compatibility shim for pickled data.

``config`` moved to ``picbreeder_vlm.core.config`` during the 2026 repo restructure.
Pickled genomes / checkpoints store the *original* module path, so this thin
shim keeps ``import config`` working (e.g. ``pickle.load`` of an archive .pkl).
New code should import from ``picbreeder_vlm.core.config`` directly.
"""
from picbreeder_vlm.core.config import *  # noqa: F401,F403
from picbreeder_vlm.core import config as _m  # noqa: F401
import sys as _sys
_sys.modules[__name__].__dict__.update({k: v for k, v in _m.__dict__.items() if not k.startswith('__')})
