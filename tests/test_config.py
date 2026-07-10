"""PicbreederConfig: preset resolution, run-directory naming, and worker round-trips.

The experiment name is the on-disk identity of a run. Sweeps, the HF archive, and
every tool that globs `sweep_logs/sweep/<name>` depend on it, so the goldens below
are transcribed from real published run directories. Changing one means orphaning
data, not just renaming a folder.
"""
from dataclasses import fields
from pathlib import Path

import pytest

from picbreeder_vlm._paths import NEAT_CONFIG_PATH
from picbreeder_vlm.core.config import (
    PATH_FIELDS,
    PicbreederConfig,
    _deserialize_config_for_worker,
    _serialize_config_for_worker,
    ensure_valid_config,
    resolve_neat_config_path,
)


# --------------------------------------------------------------------------
# NEAT preset resolution (incl. the resume shim for runs that predate the move)
# --------------------------------------------------------------------------

def test_unset_path_resolves_to_the_packaged_preset():
    assert resolve_neat_config_path(PicbreederConfig()) == NEAT_CONFIG_PATH


@pytest.mark.parametrize("legacy", [
    "picture2d/interactive_config_color",     # pre-2026-restructure
    "data/neat/interactive_config_color",     # post-restructure, pre-move-into-core
])
def test_stale_recorded_paths_fall_back_to_the_packaged_preset(legacy):
    """Runs write neat_config_path into agents_metadata.json. Resuming one started
    before the preset moved must not explode on a path that no longer exists."""
    cfg = PicbreederConfig(neat_config_path=Path(legacy))
    assert resolve_neat_config_path(cfg) == NEAT_CONFIG_PATH


def test_an_explicit_existing_path_is_honoured(tmp_path):
    custom = tmp_path / "interactive_config_color"
    custom.write_text(NEAT_CONFIG_PATH.read_text())
    cfg = PicbreederConfig(neat_config_path=custom)
    assert resolve_neat_config_path(cfg) == custom


def test_an_unrecognised_missing_path_is_not_silently_swallowed():
    """Only the preset's own filename gets the fallback. A user pointing at their
    own config and typo-ing the path should see the failure, not evolve against ours."""
    cfg = PicbreederConfig(neat_config_path=Path("/nonexistent/my_own_config"))
    resolved = resolve_neat_config_path(cfg)
    assert resolved != NEAT_CONFIG_PATH
    with pytest.raises(FileNotFoundError):
        ensure_valid_config(cfg, original_cwd=Path.cwd())


# --------------------------------------------------------------------------
# Experiment-directory naming
# --------------------------------------------------------------------------

def _name_for(tmp_path, **overrides) -> str:
    cfg = PicbreederConfig(log_dir=str(tmp_path), **overrides)
    return Path(ensure_valid_config(cfg, original_cwd=tmp_path).experiment_dir).name


def test_experiment_name_matches_a_published_run(tmp_path):
    # From the paper's archive: sweep_logs/sweep/th10_ag20_model-gemini-2.5-pro_...
    assert _name_for(
        tmp_path, chat_history_turns=10, agent_generations=20,
        model="gemini-2.5-pro", scheme="toggle", seed=6,
    ) == "th10_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s6"


def test_remote_model_ids_use_their_short_alias(tmp_path):
    # remote:Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 -> qwen3-vl-30b-fp8, or the dir name
    # gains slashes and colons and stops being a path.
    name = _name_for(
        tmp_path, chat_history_turns=10, agent_generations=20,
        model="remote:Qwen/Qwen3-VL-30B-A3B-Instruct-FP8", scheme="toggle", seed=3,
    )
    assert name == (
        "th10_ag20_model-qwen3-vl-30b-fp8_tb-1_scheme-toggle"
        "_nopersonalities_fixed-sesh_s3"
    )
    assert "/" not in name and ":" not in name


def test_unknown_model_ids_are_made_path_safe(tmp_path):
    name = _name_for(tmp_path, model="vendor/Some_Model:v2", scheme="toggle", seed=0)
    assert "model-vendor-Some_Model-v2" in name


def test_default_valued_knobs_stay_out_of_the_name(tmp_path):
    """Only non-default settings earn a suffix -- that is why the paper's runs have
    short names. A knob that always printed would rename every historical run."""
    name = _name_for(tmp_path, scheme="toggle", seed=0)
    for absent in ("_goal-", "_ts", "_baseline-", "_randp", "_temp", "_traits",
                   "_colornudge", "_norationale", "_include-branch-img"):
        assert absent not in name, f"{absent!r} leaked into a default-config name"


@pytest.mark.parametrize("overrides,fragment", [
    ({"n_personality_traits": 8}, "_traits8"),
    ({"generate_personalities": True}, "_personalities"),
    ({"color_nudge": True}, "_colornudge"),
    ({"temperature": "random"}, "_temp-random"),
    ({"selection_baseline": "random"}, "_baseline-random"),
    ({"rand_select_prob": 0.25}, "_randp0.25"),
    ({"request_rationale": False}, "_norationale"),
    ({"thumb_size": 256}, "_ts256"),
])
def test_non_default_knobs_are_recorded_in_the_name(tmp_path, overrides, fragment):
    assert fragment in _name_for(tmp_path, scheme="toggle", seed=0, **overrides)


def test_fixed_session_lengths_pins_agent_generations(tmp_path):
    cfg = PicbreederConfig(log_dir=str(tmp_path), fixed_session_lengths=True,
                           agent_generations=7)
    assert ensure_valid_config(cfg, original_cwd=tmp_path).agent_generations == 20


def test_test_mode_caps_the_run(tmp_path):
    cfg = PicbreederConfig(log_dir=str(tmp_path), test_mode=True,
                           fixed_session_lengths=False,
                           agent_generations=50, num_agents=200)
    out = ensure_valid_config(cfg, original_cwd=tmp_path)
    assert (out.agent_generations, out.num_agents) == (3, 2)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("overrides", [
    {"rows": 0},
    {"thumb_size": 4},
    {"agent_generations": 0, "fixed_session_lengths": False},
    {"num_agents": 0},
    {"num_proc": 0},
    {"select_k": 0},
    {"selection_baseline": "not-a-baseline"},
    {"rand_select_prob": 1.5},
    {"rand_select_mode": "sometimes"},
    {"always_include_branched_image": True, "always_include_archive_sample": True},
    {"chat_history_turns": -1, "always_include_branched_image": True},
])
def test_invalid_configs_are_rejected(tmp_path, overrides):
    cfg = PicbreederConfig(log_dir=str(tmp_path), **overrides)
    with pytest.raises(ValueError):
        ensure_valid_config(cfg, original_cwd=tmp_path)


def test_warm_start_structure_is_gone():
    """It gated dead code paths and was removed; a stray re-add would silently
    resurrect the colour-zeroing branches in AgentRunner."""
    assert "warm_start_structure" not in {f.name for f in fields(PicbreederConfig)}


# --------------------------------------------------------------------------
# Worker round-trip (agents are dispatched to subprocesses as plain dicts)
# --------------------------------------------------------------------------

def test_path_fields_tracks_the_actual_path_typed_fields():
    """PATH_FIELDS drove `config_path` long after that field became `neat_config_path`,
    so workers silently received a str. Pin it to the dataclass instead of a memory."""
    declared = {f.name for f in fields(PicbreederConfig) if "Path" in str(f.type)}
    assert PATH_FIELDS == declared


def test_config_survives_the_worker_round_trip(tmp_path):
    cfg = ensure_valid_config(
        PicbreederConfig(log_dir=str(tmp_path), model="gemini-2.5-pro",
                         scheme="toggle", seed=6, chat_history_turns=10),
        original_cwd=tmp_path,
    )
    restored = _deserialize_config_for_worker(_serialize_config_for_worker(cfg))

    for f in fields(PicbreederConfig):
        if f.name == "hydra":
            continue
        assert getattr(restored, f.name) == getattr(cfg, f.name), f.name
    assert isinstance(restored.experiment_dir, Path)
    assert isinstance(restored.neat_config_path, Path)
