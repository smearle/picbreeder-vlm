#!/usr/bin/env python3
"""Summarize agent runtimes, restarts, and publications for a collaborative run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import hydra
from hydra.utils import get_original_cwd

from config import CollaborativeConfig, ensure_valid_config

LOG_DIR_CANDIDATES: List[str] = ["logs", "logs_collaborative"]


@dataclass
class AgentRunSummary:
    agent_id: str
    start_timestamp: Optional[str]
    end_timestamp: Optional[str]
    duration_seconds: Optional[float]
    generations_recorded: Optional[int]
    quit_flag: Optional[bool]
    restart_requested: bool
    restart_count: int
    publication_count: int

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _find_log_file(agent_dir: Path, filename: str) -> Optional[Path]:
    for log_dir in LOG_DIR_CANDIDATES:
        candidate = agent_dir / log_dir / filename
        if candidate.exists():
            return candidate
    return None


def _summarize_selection_history(selection_path: Path) -> Dict[str, Optional[Any]]:
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    last_generation: Optional[int] = None
    quit_flag: Optional[bool] = None

    for record in _iter_jsonl(selection_path):
        ts = _parse_iso_timestamp(record.get("timestamp"))
        if ts:
            start = ts if start is None or ts < start else start
            end = ts if end is None or ts > end else end
        generation = record.get("generation")
        try:
            generation_int = int(generation)
        except (TypeError, ValueError):
            generation_int = None
        if generation_int is not None:
            last_generation = generation_int if last_generation is None else max(last_generation, generation_int)
        quit_value = record.get("quit")
        if isinstance(quit_value, bool):
            quit_flag = quit_value

    duration_seconds: Optional[float] = None
    if start and end:
        duration_seconds = (end - start).total_seconds()

    return {
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "duration_seconds": duration_seconds,
        "last_generation": last_generation,
        "quit_flag": quit_flag,
    }


def _count_jsonl_records(path: Optional[Path]) -> int:
    if path is None or not path.exists():
        return 0
    return sum(1 for _ in _iter_jsonl(path))


def summarize_agent(agent_dir: Path) -> Optional[AgentRunSummary]:
    selection_path = _find_log_file(agent_dir, "selection_history.jsonl")
    if selection_path is None or not selection_path.exists():
        return None

    selection_summary = _summarize_selection_history(selection_path)
    publication_path = _find_log_file(agent_dir, "publication_history.jsonl")
    restart_path = _find_log_file(agent_dir, "restart_history.jsonl")
    restart_count = _count_jsonl_records(restart_path)

    return AgentRunSummary(
        agent_id=agent_dir.name,
        start_timestamp=selection_summary["start"],
        end_timestamp=selection_summary["end"],
        duration_seconds=selection_summary["duration_seconds"],
        generations_recorded=selection_summary["last_generation"],
        quit_flag=selection_summary["quit_flag"],
        restart_requested=restart_count > 0,
        restart_count=restart_count,
        publication_count=_count_jsonl_records(publication_path),
    )


def _iter_agent_dirs(experiment_dir: Path) -> Iterable[Path]:
    agents_dir = experiment_dir / "agents"
    if not agents_dir.exists():
        return []
    return (
        path
        for path in sorted(agents_dir.iterdir())
        if path.is_dir() and path.name.startswith("agent_")
    )


@hydra.main(version_base="1.3", config_path=None, config_name="collaborative_base")
def main(cfg: CollaborativeConfig) -> None:
    original_cwd = Path(get_original_cwd())
    cfg = ensure_valid_config(cfg, original_cwd=original_cwd)
    experiment_dir = Path(cfg.experiment_dir)
    summaries: List[AgentRunSummary] = []

    for agent_dir in _iter_agent_dirs(experiment_dir):
        summary = summarize_agent(agent_dir)
        if summary is not None:
            summaries.append(summary)

    output_path = experiment_dir / "agent_run_summary.jsonl"
    with output_path.open("w", encoding="utf-8") as fp:
        for summary in summaries:
            fp.write(summary.to_json())
            fp.write("\n")

    print(f"Summarized {len(summaries)} agents in {experiment_dir}")
    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
