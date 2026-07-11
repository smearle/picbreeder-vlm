"""End-to-end smoke tests for the evolution loop using the zero-cost `mock` model.

These exercise the full agent / archive / rendering plumbing (no API key, GPU, or
network) two ways:

  * a random dry-run from scratch -- agents start from an empty archive, evolve,
    and publish; the resulting archive must be on disk and self-consistent;
  * a reload/continue -- a second run pointed at the same directory picks the
    existing archive back up (its metadata round-trips) and does not lose entries.

Kept serial (`num_proc=1`) and tiny so they run in CI without torch/vLLM.
"""
import json
from pathlib import Path

from picbreeder_vlm.core.config import PicbreederConfig, ensure_valid_config
from picbreeder_vlm.agents.collaborative_multi_agent import run


def _mock_config(experiment_dir: Path, num_agents: int = 1, generations: int = 2, resume: bool = False):
    cfg = PicbreederConfig(
        model="mock",
        num_agents=num_agents,
        agent_generations=generations,
        num_proc=1,          # serial, in-process
        rows=2, cols=2,      # tiny grid -> fast render; keep these smoke tests cheap for CI
        seed=0,
        resume=resume,
    )
    cfg.experiment_dir = experiment_dir
    # ensure_valid_config resolves the NEAT preset path + absolutises experiment_dir
    # and returns the validated config (it does not mutate in place).
    return ensure_valid_config(cfg, original_cwd=Path.cwd())


def _archive_entries(experiment_dir: Path):
    meta = Path(experiment_dir) / "archive" / "archive_metadata.json"
    if not meta.exists():
        return []
    return json.loads(meta.read_text(encoding="utf-8")).get("entries", [])


def _assert_self_consistent(experiment_dir: Path):
    """Every published entry must have its genome pickle and rendered image."""
    archive = Path(experiment_dir) / "archive"
    for entry in _archive_entries(experiment_dir):
        eid = entry["id"]
        assert (archive / "genomes" / f"{eid}.pkl").exists(), f"missing genome for {eid}"
        assert (archive / "images" / f"{eid}.png").exists(), f"missing image for {eid}"


def test_mock_dry_run_from_scratch(tmp_path):
    exp = tmp_path / "run"
    cfg = _mock_config(exp, num_agents=1, generations=3)
    run(cfg)

    # The archive and its metadata exist, and the agent is recorded as completed.
    assert (exp / "archive" / "archive_metadata.json").exists()
    agents_meta = json.loads((exp / "agents_metadata.json").read_text(encoding="utf-8"))
    assert agents_meta.get("agents"), "no agent recorded in agents_metadata.json"

    # The mock publishes deterministically, so the archive grew and is self-consistent.
    entries = _archive_entries(exp)
    assert len(entries) >= 1, "expected at least one published entry from the mock run"
    _assert_self_consistent(exp)


def test_mock_reload_continues_archive(tmp_path):
    exp = tmp_path / "run"

    # First run: build an archive from scratch.
    run(_mock_config(exp, num_agents=1, generations=3))
    before = _archive_entries(exp)
    assert before, "first run produced no entries to reload"
    before_ids = {e["id"] for e in before}
    before_next_id = json.loads(
        (exp / "archive" / "archive_metadata.json").read_text(encoding="utf-8")
    )["next_id"]

    # Second run on the SAME directory with resume=true: the existing archive and
    # agent metadata are reloaded rather than clobbered, and new agents extend it.
    run(_mock_config(exp, num_agents=2, generations=3, resume=True))

    after = _archive_entries(exp)
    after_ids = {e["id"] for e in after}
    # The reload preserved the original entries (nothing lost) ...
    assert before_ids <= after_ids, "reload dropped previously-published entries"
    # ... and any new entries got fresh, non-colliding ids past the reloaded next_id.
    for eid in after_ids - before_ids:
        assert int(eid.split("_")[-1]) >= before_next_id
    _assert_self_consistent(exp)


def test_mock_continue_without_agents_metadata(tmp_path):
    """A run reconstructed from the HF dataset (tools/pull_run_from_hf.py) ships the
    archive but *no* agents_metadata.json -- we deliberately don't publish the
    orchestrator ledger. Continuing such a run must degrade gracefully: the
    orchestrator starts a fresh ledger, keeps the existing archive, and extends it."""
    exp = tmp_path / "run"

    # Build an archive, then simulate an HF-pulled run by removing the ledger.
    run(_mock_config(exp, num_agents=1, generations=3))
    before = _archive_entries(exp)
    assert before, "first run produced no entries"
    before_ids = {e["id"] for e in before}
    meta_file = exp / "agents_metadata.json"
    assert meta_file.exists()
    meta_file.unlink()

    # Continue: no crash, a fresh ledger is written, and the archive is extended
    # rather than clobbered.
    run(_mock_config(exp, num_agents=1, generations=3, resume=True))

    assert meta_file.exists(), "continue did not recreate agents_metadata.json"
    after_ids = {e["id"] for e in _archive_entries(exp)}
    assert before_ids <= after_ids, "continue dropped previously-published entries"
    _assert_self_consistent(exp)
