"""Backward-compatibility shim for pickled data.

``picbreeder_reproduction`` moved to ``picbreeder_vlm.core.picbreeder_reproduction`` during the 2026 repo restructure.
Pickled genomes / checkpoints store the *original* module path, so this thin
shim keeps ``import picbreeder_reproduction`` working (e.g. ``pickle.load`` of an archive .pkl).
New code should import from ``picbreeder_vlm.core.picbreeder_reproduction`` directly.
"""
from picbreeder_vlm.core.picbreeder_reproduction import *  # noqa: F401,F403
from picbreeder_vlm.core import picbreeder_reproduction as _m  # noqa: F401
import sys as _sys
_sys.modules[__name__].__dict__.update({k: v for k, v in _m.__dict__.items() if not k.startswith('__')})
