"""Shared filesystem anchors.

REPO_ROOT is the repository root, i.e. the directory that holds this package,
the data files (imagenet*.txt, noun_lists/, picture2d/, ...) and the tools/.
Modules that used to compute ``Path(__file__).parent`` (when they lived at the
repo root) should import REPO_ROOT from here instead.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
