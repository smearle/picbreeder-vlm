"""Genome diagrams are optional; their absence must be loud, not silent.

graphviz is imported softly so that `pip install -r requirements-test.txt` (and any
environment without the system Graphviz binaries) can still evolve CPPNs and render
archive images. There is no fallback diagram renderer -- render_genome_diagram just
returns None -- so it has to say so.
"""
import random
import warnings

import neat
import pytest

from picbreeder_vlm._paths import NEAT_CONFIG_PATH
from picbreeder_vlm.core import rendering
from picbreeder_vlm.core.neat_components import build_neat_config, seed_initial_population


@pytest.fixture
def genome_and_config():
    cfg = build_neat_config(NEAT_CONFIG_PATH, 3, 5, True, False, False)
    random.seed(0)
    pop = neat.Population(cfg)
    seed_initial_population(pop, cfg.genome_config)
    return next(iter(pop.population.values())), cfg


@pytest.fixture
def graphviz_missing(monkeypatch):
    monkeypatch.setattr(rendering, "graphviz", None)
    monkeypatch.setattr(
        rendering, "_GRAPHVIZ_IMPORT_ERROR", ImportError("No module named 'graphviz'")
    )
    rendering._warn_graphviz_missing.cache_clear()
    yield
    rendering._warn_graphviz_missing.cache_clear()


def test_missing_graphviz_warns_and_returns_none(graphviz_missing, genome_and_config, tmp_path):
    genome, cfg = genome_and_config
    with pytest.warns(RuntimeWarning, match="no genome diagrams will be written"):
        result = rendering.render_genome_diagram(genome, cfg, output_stem=tmp_path / "g")
    assert result is None
    assert not list(tmp_path.iterdir()), "no diagram file should be left behind"


def test_the_warning_names_the_remedy(graphviz_missing, genome_and_config, tmp_path):
    genome, cfg = genome_and_config
    with pytest.warns(RuntimeWarning) as caught:
        rendering.render_genome_diagram(genome, cfg, output_stem=tmp_path / "g")
    message = str(caught[0].message)
    assert "pip install graphviz" in message
    # ...and reassures the reader that nothing else broke.
    assert "unaffected" in message


def test_the_warning_fires_once_per_process(graphviz_missing, genome_and_config, tmp_path):
    """A run with render_genome_diagrams=true calls this once per genome per
    generation; warning every time would bury the log."""
    genome, cfg = genome_and_config
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in range(20):
            rendering.render_genome_diagram(genome, cfg, output_stem=tmp_path / f"g{i}")
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime) == 1, f"expected one warning, got {len(runtime)}"


def test_diagram_helper_degrades_to_an_empty_list(graphviz_missing, genome_and_config, tmp_path):
    """save_neat_genome_diagrams is what the agent loop actually calls."""
    from picbreeder_vlm.core.artifacts import save_neat_genome_diagrams

    genome, cfg = genome_and_config
    with pytest.warns(RuntimeWarning):
        paths = save_neat_genome_diagrams([(1, genome)], cfg, tmp_path, generation=0)
    assert paths == []


def test_image_rendering_never_needs_graphviz(graphviz_missing, genome_and_config):
    """The load-bearing path: archive images must render with graphviz absent."""
    genome, cfg = genome_and_config
    gray, color = rendering.render_genome_image(genome, cfg, 16, 16)
    assert gray.size == (16, 16) and color.size == (16, 16)
