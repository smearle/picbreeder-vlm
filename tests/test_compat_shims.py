"""Top-level shims that keep archived pickles loadable.

Genomes and population checkpoints were pickled when these modules lived at the
repo root, so their class references say ``neat_components.PicbreederGenome``.
Unpickling imports that exact path. If a shim stops resolving -- or resolves to a
*different* class object than the package one -- every archived .pkl on disk and
on the Hub becomes unreadable, and the failure only shows up at load time.
"""
import importlib
import pickle
import random

import neat
import pytest

from picbreeder_vlm._paths import NEAT_CONFIG_PATH
from picbreeder_vlm.core.neat_components import build_neat_config, seed_initial_population

SHIMS = {
    "neat_components": "picbreeder_vlm.core.neat_components",
    "config": "picbreeder_vlm.core.config",
    "picbreeder_reproduction": "picbreeder_vlm.core.picbreeder_reproduction",
    "archive_manager": "picbreeder_vlm.core.archive_manager",
    "rendering": "picbreeder_vlm.core.rendering",
}


@pytest.mark.parametrize("shim,target", sorted(SHIMS.items()))
def test_shim_is_importable(shim, target):
    assert importlib.import_module(shim) is not None
    assert importlib.import_module(target) is not None


def test_setup_py_declares_every_shim():
    from pathlib import Path
    setup_py = (Path(__file__).resolve().parents[1] / "setup.py").read_text()
    for shim in SHIMS:
        assert f'"{shim}"' in setup_py, f"{shim} missing from _COMPAT_SHIMS"


def test_genome_class_is_shared_not_copied():
    """`from x import *` would rebind names; a duplicate class object would make
    isinstance() checks and pickle round-trips disagree across the two import paths."""
    shimmed = importlib.import_module("neat_components")
    packaged = importlib.import_module("picbreeder_vlm.core.neat_components")
    assert shimmed.PicbreederGenome is packaged.PicbreederGenome
    assert shimmed.InteractiveStagnation is packaged.InteractiveStagnation


def test_genome_pickled_under_the_legacy_module_path_still_loads():
    """Simulates an archived .pkl: force the class's recorded module back to the
    pre-restructure top-level name, pickle it, and load it again."""
    cfg = build_neat_config(NEAT_CONFIG_PATH, 3, 5, True, False, False)
    random.seed(0)
    pop = neat.Population(cfg)
    seed_initial_population(pop, cfg.genome_config)
    genome = next(iter(pop.population.values()))

    packaged = importlib.import_module("picbreeder_vlm.core.neat_components")
    original = packaged.PicbreederGenome.__module__
    try:
        packaged.PicbreederGenome.__module__ = "neat_components"
        blob = pickle.dumps(genome)
    finally:
        packaged.PicbreederGenome.__module__ = original

    assert b"neat_components" in blob
    restored = pickle.loads(blob)
    assert isinstance(restored, packaged.PicbreederGenome)
    assert len(restored.connections) == len(genome.connections)
    assert len(restored.nodes) == len(genome.nodes)
