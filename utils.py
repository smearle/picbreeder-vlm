import json
import os
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Optional

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
        if rebased:
            candidates.append(rebased)
        archive_suffix = relative_suffix_after_dir(raw_path)
        if archive_suffix is not None:
            candidates.append((archive_dir / archive_suffix).resolve())
    else:
        candidates.append((archive_dir / raw_path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def relative_suffix_after_dir(path: Path, dir_name: str = "archive") -> Optional[Path]:
    parts = path.parts
    try:
        idx = parts.index(dir_name)
    except ValueError:
        return None
    suffix_parts = parts[idx + 1 :]
    if not suffix_parts:
        return None
    return Path(*suffix_parts)


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


def _ensure_absolute(path: Path, base: Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (base / expanded).resolve()


def _ensure_int_list(values: Iterable[Any]) -> List[int]:
    result: List[int] = []
    if isinstance(values, int):
        values = [values]
    for value in values:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        result.append(idx)
    return result


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_human_archive_images(archive_dir: Path) -> List[Path]:
    """Load images from the directory, sorted by numeric filename."""
    if not archive_dir.exists():
        raise FileNotFoundError(f"Archive directory not found: {archive_dir}")
    
    images = []
    for p in archive_dir.glob("*.png"):
        try:
            num = int(p.stem)
            images.append((num, p))
        except ValueError:
            continue
            
    images.sort(key=lambda x: x[0])
    return [p for _, p in images]

