"""Shared filesystem anchors.

REPO_ROOT is the repository root, i.e. the directory that holds this package,
the committed data (under data/, e.g. data/noun_lists/, data/human_baseline/) and the tools/.
Modules that used to compute ``Path(__file__).parent`` (when they lived at the
repo root) should import REPO_ROOT from here instead.

NEAT_CONFIG_PATH is the CPPN/NEAT preset every entry point evolves against. It
ships inside the package, next to the ``core.picture2d`` renderer that consumes
it, so it resolves the same whether the repo is on sys.path or pip-installed.
Nothing swaps it out; treat it as code, not data.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

NEAT_CONFIG_PATH = Path(__file__).resolve().parent / "core" / "interactive_config_color"

# Vendored third-party code (see third-party/README.md). FER_ROOT is the trimmed
# akarshkumar0101/fer snapshot; build and analysis scripts read the human
# Picbreeder archive out of it. Anchor those reads on FER_ROOT rather than
# hard-coding "fer/..." or "third-party/fer/..." so a future move stays one edit.
THIRD_PARTY_ROOT = REPO_ROOT / "third-party"
FER_ROOT = THIRD_PARTY_ROOT / "fer"


def ensure_fer_importable():
    """Make ``import fer.src.X`` resolve the vendored fer package.

    fer lives under ``third-party/`` rather than at the repo root, so the
    directory that must be on sys.path for ``import fer`` to work is
    THIRD_PARTY_ROOT. Call this before importing anything from ``fer.src``.
    """
    import sys

    p = str(THIRD_PARTY_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
