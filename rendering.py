"""Backward-compatibility shim.

``rendering`` moved to ``picbreeder_vlm.core.rendering`` during the 2026 repo
restructure. The previous packaging exposed ``rendering`` as a top-level module,
so this thin shim preserves ``import rendering`` for any external notebooks/tools.
New code should import from ``picbreeder_vlm.core.rendering`` directly.
"""
from picbreeder_vlm.core.rendering import *  # noqa: F401,F403
from picbreeder_vlm.core import rendering as _m  # noqa: F401
import sys as _sys
_sys.modules[__name__].__dict__.update(
    {k: v for k, v in _m.__dict__.items() if not k.startswith("__")}
)
