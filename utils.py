from pathlib import Path
import random
from typing import Any, Dict, List, Optional

import numpy


def apply_random_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy
    except ImportError:
        return
    numpy.random.seed(seed)


def _rebase_submitit_path(raw_path: Path, experiment_prefix: Optional[Path]) -> Optional[Path]:
    if not (experiment_prefix and raw_path.is_absolute()):
        return None
    parts = raw_path.parts
    try:
        submitit_idx = parts.index("submitit_sweeps")
    except ValueError:
        return None
    suffix_parts = parts[submitit_idx + 1 :]
    if not suffix_parts:
        return None
    # Note that the agent_dir is the first suffix part, but already included in experiment_prefix, so we skip it.
    rebased = (experiment_prefix / Path(*suffix_parts[1:])).resolve()
    return rebased if rebased.exists() else None


def _resolve_image_path(
    entry: Dict[str, Any],
    archive_dir: Path,
    experiment_prefix: Optional[Path],
) -> Optional[Path]:
    raw_value = entry.get("image_path")
    if not raw_value:
        return None
    raw_path = Path(raw_value)

    candidates: List[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
        rebased = _rebase_submitit_path(raw_path, experiment_prefix)
        print(rebased)
        if rebased:
            candidates.append(rebased)
    else:
        candidates.append((archive_dir / raw_path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _resolve_source_experiment_dir(
    entry: Dict[str, Any],
    *,
    experiment_dir: Path,
    experiment_prefix: Optional[Path],
) -> Optional[Path]:
    raw_value = entry.get("source_experiment")
    agent_id = entry.get("agent_id")
    candidates: List[Path] = []

    if raw_value:
        raw_path = Path(raw_value)
        if raw_path.is_absolute():
            candidates.append(raw_path)
            rebased = _rebase_submitit_path(raw_path, experiment_prefix)
            if rebased:
                candidates.append(rebased)
        else:
            candidates.append((experiment_dir / raw_path).resolve())

    if agent_id:
        candidates.append((experiment_dir / "agents" / str(agent_id)).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None
