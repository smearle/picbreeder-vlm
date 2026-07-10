"""The CPPN/NEAT preset: where it lives, that it loads, and that it still renders.

`interactive_config_color` is effectively source -- nothing swaps it out -- so it
sits inside the package next to the picture2d renderer. These tests pin the
properties that moving it could plausibly have broken.
"""
import subprocess
import sys
from pathlib import Path

import neat
import pytest

from picbreeder_vlm._paths import NEAT_CONFIG_PATH
from picbreeder_vlm.core.neat_components import (
    build_neat_config,
    seed_initial_population,
    sync_population_output_activations,
)
from picbreeder_vlm.core.rendering import render_genome_image


def test_preset_ships_inside_the_package():
    assert NEAT_CONFIG_PATH.is_file()
    package_root = Path(__import__("picbreeder_vlm").__file__).resolve().parent
    assert package_root in NEAT_CONFIG_PATH.resolve().parents, (
        "the preset must live under picbreeder_vlm/ or a wheel install loses it"
    )


def test_preset_resolves_from_any_working_directory(tmp_path):
    """It used to be reached via CWD-relative 'data/neat/...' strings, which only
    worked when you happened to launch from the repo root."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "from picbreeder_vlm._paths import NEAT_CONFIG_PATH; "
         "assert NEAT_CONFIG_PATH.is_file(); print('ok')"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_preset_is_declared_as_package_data():
    """A non-editable install copies only what package_data names. The preset has no
    suffix, so no wildcard would catch it -- guard the explicit entry."""
    setup_py = (Path(__file__).resolve().parents[1] / "setup.py").read_text()
    assert "interactive_config_color" in setup_py


@pytest.fixture(scope="module")
def neat_config():
    return build_neat_config(
        NEAT_CONFIG_PATH,
        rows=3,
        cols=5,
        enable_output_activations=True,
        enable_input_activations=False,
        enable_crossover=False,
    )


def test_preset_defines_a_picbreeder_cppn(neat_config):
    # Four inputs (x, y, d, bias) and three outputs (brightness + two colour channels)
    # are what picture2d.py and the colour/structure mutation split both assume.
    assert neat_config.genome_config.num_inputs == 4
    assert neat_config.genome_config.num_outputs == 3
    assert neat_config.pop_size == 15  # rows * cols


def test_build_neat_config_sizes_population_to_the_grid():
    cfg = build_neat_config(NEAT_CONFIG_PATH, 4, 6, True, False, False)
    assert cfg.pop_size == 24


def test_the_evolution_core_imports_without_the_heavy_stack():
    """core.neat_components must stay reachable without torch / vLLM / google-genai.
    build_neat_config used to live in agents.collaborative_multi_agent, so merely
    constructing a NEAT config dragged in the whole VLM stack -- and CI with it."""
    probe = (
        "import sys;"
        "import picbreeder_vlm.core.neat_components;"
        "heavy = {'torch', 'vllm', 'google.genai', 'transformers', 'open_clip'};"
        "hit = heavy & set(sys.modules);"
        "assert not hit, hit; print('ok')"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _seeded_population(neat_config, seed):
    import random
    random.seed(seed)
    pop = neat.Population(neat_config)
    seed_initial_population(pop, neat_config.genome_config)
    sync_population_output_activations(pop, neat_config.genome_config)
    return pop


def test_initial_population_shares_one_seeded_topology(neat_config):
    """seed_initial_population imposes a fixed starting structure on every genome;
    the population differs only in weights. Genomes that skipped it (or a preset
    whose num_inputs/num_hidden drifted) would show up as a ragged topology."""
    pop = _seeded_population(neat_config, 0)
    assert len(pop.population) == neat_config.pop_size

    shapes = {(len(g.nodes), len(g.connections)) for g in pop.population.values()}
    assert len(shapes) == 1, f"population topology is not uniform: {shapes}"

    gc = neat_config.genome_config
    genome = next(iter(pop.population.values()))
    sources = {src for src, _ in genome.connections}
    sinks = {dst for _, dst in genome.connections}
    assert set(gc.input_keys) <= sources, "every CPPN input must drive something"
    assert set(gc.output_keys) <= sinks, "every output channel must be reachable"


def test_genome_renders_to_grayscale_and_color(neat_config):
    pop = _seeded_population(neat_config, 0)
    genome = next(iter(pop.population.values()))
    gray, color = render_genome_image(genome, neat_config, 16, 16)
    assert (gray.size, gray.mode) == ((16, 16), "L")
    assert (color.size, color.mode) == ((16, 16), "RGB")


def test_rendering_is_deterministic(neat_config):
    """Same genome, same pixels: the archive's images are regenerated from genomes,
    so a drift here silently invalidates every published image."""
    genome = next(iter(_seeded_population(neat_config, 0).population.values()))
    first = render_genome_image(genome, neat_config, 16, 16)
    second = render_genome_image(genome, neat_config, 16, 16)
    assert first[0].tobytes() == second[0].tobytes()
    assert first[1].tobytes() == second[1].tobytes()
