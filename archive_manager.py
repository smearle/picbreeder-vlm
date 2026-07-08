"""Backward-compatibility shim for pickled data.

``archive_manager`` moved to ``picbreeder_vlm.core.archive_manager`` during the 2026 repo restructure.
Pickled genomes / checkpoints store the *original* module path, so this thin
shim keeps ``import archive_manager`` working (e.g. ``pickle.load`` of an archive .pkl).
New code should import from ``picbreeder_vlm.core.archive_manager`` directly.
"""
from picbreeder_vlm.core.archive_manager import *  # noqa: F401,F403
from picbreeder_vlm.core import archive_manager as _m  # noqa: F401
import sys as _sys
_sys.modules[__name__].__dict__.update({k: v for k, v in _m.__dict__.items() if not k.startswith('__')})
