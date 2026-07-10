"""The eval-metric names, and the artifact names deliberately frozen beneath them.

The analysis modules import torch/open_clip at module scope, so we check names
statically rather than importing. That is also what CI can afford.
"""
import ast
from dataclasses import fields
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "picbreeder_vlm" / "analysis"


def _toplevel_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def test_the_paper_named_modules_exist():
    assert (ANALYSIS / "compute_semantic_recall.py").is_file()
    assert (ANALYSIS / "compute_visual_coverage.py").is_file()
    # Semantic Coverage has no module of its own: it falls out of the captioning pass.
    assert (ANALYSIS / "caption_and_embed_archive.py").is_file()
    assert not (ANALYSIS / "compute_noun_coverage.py").exists()


@pytest.mark.parametrize("module,symbol", [
    ("compute_semantic_recall.py", "SemanticRecallConfig"),
    ("compute_semantic_recall.py", "render_semantic_recall_grid"),
    ("compute_visual_coverage.py", "VisualCoverageConfig"),
])
def test_paper_named_symbols_are_defined(module, symbol):
    assert symbol in _toplevel_names(ANALYSIS / module)


def test_sweep_exposes_the_paper_named_eval_flag():
    from picbreeder_vlm.experiments.sweep_configs import SweepConfig
    names = {f.name for f in fields(SweepConfig)}
    assert "eval_semantic_recall" in names
    assert "eval_noun_coverage" not in names


def test_no_module_still_imports_the_old_names():
    stale = ["compute_noun_coverage", "NounSimilarityConfig",
             "PairwiseDistanceConfig", "render_noun_similarity_grid"]
    offenders = []
    for py in REPO.glob("picbreeder_vlm/**/*.py"):
        text = py.read_text()
        offenders += [f"{py.relative_to(REPO)}:{name}" for name in stale if name in text]
    assert not offenders, offenders


def test_on_disk_artifact_names_stay_frozen():
    """Renaming these would orphan every completed run, locally and on the Hub.
    They intentionally do NOT match the paper's vocabulary."""
    recall = (ANALYSIS / "compute_semantic_recall.py").read_text()
    assert "noun_similarity_metrics.json" in recall
    assert "mean_max_similarity" in recall

    coverage = (ANALYSIS / "compute_visual_coverage.py").read_text()
    assert "embedding_mean_pairwise_distance_over_time" in coverage
    assert "k_covering_radii" in coverage


def test_coverage_and_diversity_are_reported_under_distinct_keys():
    """`best_seeds` once filed mean-pairwise-distance (a diversity stat) under
    'visual_coverage', while real coverage (the k-covering radius) hid under
    'visual_k_covering_k*'. Reading a cross_eval summary meant guessing which."""
    sweep = (REPO / "picbreeder_vlm" / "experiments" / "sweep.py").read_text()
    assert '"visual_diversity"' in sweep
    assert '"semantic_diversity"' in sweep
    assert 'f"visual_coverage_k{k}"' in sweep
    assert 'f"semantic_coverage_k{k}"' in sweep
    assert '"visual_k_covering_k' not in sweep
