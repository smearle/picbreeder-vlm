#!/usr/bin/env python3
"""Summarise color/grayscale usage in collaborative_multi_agent toggle runs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_METADATA_PATTERN = re.compile(r"gen_(\d+)(?:_view_(\d+))?_selection\.json$", re.IGNORECASE)
_PROMPT_COLOR_PATTERN = re.compile(r"color\s*=\s*(ON|OFF)", re.IGNORECASE)
_MODE_ALIASES = {
    "structure": "structure_only",
    "color": "color_only",
    "colour": "color_only",
    "all_channels": "all",
    "both": "all",
}


def _parse_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return None


def _normalize_mode(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    lowered = value.strip().lower()
    lowered = _MODE_ALIASES.get(lowered, lowered)
    if lowered in {"structure_only", "color_only", "all"}:
        return lowered
    return None


@dataclass
class ViewEvent:
    generation: int
    view_index: int
    metadata_path: Path
    color_requested: Optional[bool]
    mutation_mode: Optional[str]
    color_toggle_only: bool
    selected_count: int
    prompt_color: Optional[bool] = None
    resolved_color: Optional[bool] = None
    source: str = "metadata"

    @property
    def is_selection(self) -> bool:
        return not self.color_toggle_only


def _parse_prompt_color(prompt: Optional[str]) -> Optional[bool]:
    if not prompt:
        return None
    match = _PROMPT_COLOR_PATTERN.search(prompt)
    if not match:
        return None
    return match.group(1).upper() == "ON"


def _iter_agent_metadata(agent_dir: Path) -> List[ViewEvent]:
    metadata_dir = agent_dir / "queries" / "metadata"
    if not metadata_dir.exists():
        return []

    events: List[ViewEvent] = []
    for path in sorted(metadata_dir.glob("**/gen_*_selection.json")):
        if "errors" in path.parts:
            continue
        match = _METADATA_PATTERN.search(path.name)
        if not match:
            continue
        with path.open(encoding="utf-8") as fp:
            try:
                payload = json.load(fp)
            except json.JSONDecodeError:
                continue
        generation = int(payload.get("generation", match.group(1)))
        view_index_raw = payload.get("view_index")
        view_index = int(view_index_raw) if view_index_raw is not None else match.group(2)
        view_index = int(view_index) if view_index is not None else 0
        color_value = _parse_bool(payload.get("color"))
        mutation_mode = _normalize_mode(payload.get("mutation_mode"))
        color_toggle_only = bool(payload.get("color_toggle_only", False))
        selected = payload.get("selected") or []
        prompt_color = _parse_prompt_color(payload.get("prompt"))
        events.append(
            ViewEvent(
                generation=generation,
                view_index=view_index,
                metadata_path=path,
                color_requested=color_value,
                mutation_mode=mutation_mode,
                color_toggle_only=color_toggle_only,
                selected_count=len(selected),
                prompt_color=prompt_color,
            )
        )
    events.sort(key=lambda ev: (ev.generation, ev.view_index, str(ev.metadata_path)))
    return events


def _load_selection_history(agent_dir: Path) -> List[Dict[str, object]]:
    for candidate in (
        agent_dir / "logs_collaborative" / "selection_history.jsonl",
        agent_dir / "logs" / "selection_history.jsonl",
    ):
        if candidate.exists():
            path = candidate
            break
    else:
        return []

    records: List[Dict[str, object]] = []
    with path.open() as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload["_source_path"] = path
            records.append(payload)
    return records


def _resolve_metadata_path(agent_dir: Path, raw_path: Optional[str]) -> Optional[Path]:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute():
        relative_candidate = (agent_dir / candidate).resolve()
        if relative_candidate.exists():
            return relative_candidate
    else:
        parts = candidate.parts
        if agent_dir.name in parts:
            idx = parts.index(agent_dir.name)
            suffix = Path(*parts[idx + 1 :])
            rebuilt = (agent_dir / suffix).resolve()
            if rebuilt.exists():
                return rebuilt
    name_candidate = (agent_dir / candidate.name).resolve()
    if name_candidate.exists():
        return name_candidate
    return candidate


def _align_selection_records(
    agent_dir: Path,
    events: List[ViewEvent],
    history_records: Iterable[Dict[str, object]],
) -> None:
    if not events:
        return
    path_lookup: Dict[Path, ViewEvent] = {}
    name_lookup: Dict[str, ViewEvent] = {}
    for event in events:
        if not event.is_selection:
            continue
        resolved = event.metadata_path.resolve()
        path_lookup[resolved] = event
        name_lookup[event.metadata_path.name] = event

    for record in history_records:
        raw_path = record.get("metadata_path")
        meta_path = _resolve_metadata_path(agent_dir, raw_path)
        if meta_path is None:
            event = None
        else:
            event = path_lookup.get(meta_path.resolve())
        if event is None and raw_path:
            raw_name = Path(str(raw_path)).name
            event = name_lookup.get(raw_name)
        if event is None:
            continue
        color_value = record.get("color")
        normalized_color = _parse_bool(color_value)
        if normalized_color is not None:
            event.resolved_color = normalized_color
        mode_value = record.get("mutation_mode")
        normalized_mode = _normalize_mode(mode_value)
        if normalized_mode:
            event.mutation_mode = normalized_mode
        event.source = "selection_history"


def _resolve_event_colors(
    events: List[ViewEvent],
    initial_color: Optional[bool],
) -> Tuple[List[ViewEvent], Counter]:
    toggles = Counter()
    if not events:
        return events, toggles

    color_state = initial_color
    if color_state is None:
        for event in events:
            if event.prompt_color is not None:
                color_state = event.prompt_color
                break
            if event.resolved_color is not None:
                color_state = event.resolved_color
                break
            if event.color_requested is not None and not event.color_toggle_only:
                color_state = event.color_requested
                break
    if color_state is None:
        color_state = False

    for event in events:
        if event.prompt_color is not None:
            color_state = event.prompt_color
        prev_state = color_state
        if (
            event.color_toggle_only
            and event.color_requested is not None
            and prev_state is not None
            and event.color_requested != prev_state
        ):
            toggles[(prev_state, event.color_requested)] += 1
            color_state = event.color_requested
        if event.resolved_color is None:
            event.resolved_color = color_state
        else:
            color_state = event.resolved_color
    return events, toggles


@dataclass
class AgentReport:
    agent_id: str
    events: List[ViewEvent]
    toggle_counts: Counter
    selection_events: List[ViewEvent] = field(default_factory=list)

    def summary(self) -> Dict[str, object]:
        color_counts = Counter(
            ev.resolved_color for ev in self.selection_events if ev.resolved_color is not None
        )
        color_generations = [
            ev.generation for ev in self.selection_events if ev.resolved_color
        ]
        mode_counts = Counter(
            ev.mutation_mode
            for ev in self.selection_events
            if ev.resolved_color and ev.mutation_mode
        )
        return {
            "agent_id": self.agent_id,
            "generations": len(self.selection_events),
            "final_view_counts": {
                "color": color_counts.get(True, 0),
                "grayscale": color_counts.get(False, 0),
                "unknown": color_counts.get(None, 0),
            },
            "color_mutation_modes": dict(mode_counts),
            "color_generations": color_generations,
            "toggle_counts": {
                "gray_to_color": self.toggle_counts.get((False, True), 0),
                "color_to_gray": self.toggle_counts.get((True, False), 0),
            },
        }


def analyse_agent(agent_dir: Path, initial_color: Optional[bool]) -> AgentReport:
    agent_id = agent_dir.name
    events = _iter_agent_metadata(agent_dir)
    history = _load_selection_history(agent_dir)
    _align_selection_records(agent_dir, events, history)
    events, toggle_counts = _resolve_event_colors(events, initial_color)
    selection_events = [ev for ev in events if ev.is_selection]
    return AgentReport(agent_id=agent_id, events=events, toggle_counts=toggle_counts, selection_events=selection_events)


def _format_agent_report(report: AgentReport) -> str:
    summary = report.summary()
    lines = [
        f"{summary['agent_id']}: generations={summary['generations']}, "
        f"color={summary['final_view_counts']['color']}, "
        f"grayscale={summary['final_view_counts']['grayscale']}, "
        f"unknown={summary['final_view_counts']['unknown']}",
    ]
    if summary["color_mutation_modes"]:
        mode_chunks = ", ".join(f"{mode}={count}" for mode, count in sorted(summary["color_mutation_modes"].items()))
        lines.append(f"  color mutation modes: {mode_chunks}")
    if summary["color_generations"]:
        gens = ", ".join(str(gen) for gen in summary["color_generations"])
        lines.append(f"  color generations: {gens}")
    toggles = summary["toggle_counts"]
    if toggles["gray_to_color"] or toggles["color_to_gray"]:
        lines.append(
            f"  toggles: gray→color={toggles['gray_to_color']}, color→gray={toggles['color_to_gray']}"
        )
    if not report.selection_events:
        lines.append("  (no selection events found)")
    else:
        lines.append("  timeline:")
        for ev in report.selection_events:
            mode = ev.mutation_mode or "unknown"
            view = "color" if ev.resolved_color else ("grayscale" if ev.resolved_color is False else "unknown")
            lines.append(f"    gen {ev.generation:03d} view {ev.view_index:02d}: {view}, mode={mode}")
    return "\n".join(lines)


def _load_archive_color_map(run_dir: Path) -> Dict[str, bool]:
    archive_path = run_dir / "archive" / "archive_metadata.json"
    if not archive_path.exists():
        return {}
    try:
        payload = json.loads(archive_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    entries = payload.get("entries", [])
    color_map: Dict[str, bool] = {}
    for entry in entries:
        entry_id = entry.get("id")
        if not entry_id:
            continue
        color_map[entry_id] = bool(entry.get("color_enabled", False))
    return color_map


def _infer_initial_color(agent_dir: Path, archive_colors: Dict[str, bool]) -> Optional[bool]:
    for logs_dir in ("logs", "logs_collaborative"):
        branch_path = agent_dir / logs_dir / "branching_selection.json"
        if not branch_path.exists():
            continue
        try:
            payload = json.loads(branch_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        choice = (payload.get("choice") or "").lower()
        if choice != "branch":
            return False
        entry_ids = payload.get("selected_entry_ids") or []
        for entry_id in entry_ids:
            if entry_id in archive_colors:
                return bool(archive_colors[entry_id])
        return None
    return False


def analyse_run(run_dir: Path) -> Tuple[List[AgentReport], Dict[str, object]]:
    agents_dir = run_dir / "agents"
    if not agents_dir.exists():
        raise FileNotFoundError(f"{run_dir} does not contain an 'agents' directory")

    archive_colors = _load_archive_color_map(run_dir)
    reports: List[AgentReport] = []
    aggregate_modes = Counter()
    aggregate_color_counts = Counter()
    aggregate_toggles = Counter()
    total_generations = 0

    for agent_dir in sorted(p for p in agents_dir.iterdir() if p.is_dir()):
        initial_color = _infer_initial_color(agent_dir, archive_colors)
        report = analyse_agent(agent_dir, initial_color)
        reports.append(report)
        summary = report.summary()
        total_generations += summary["generations"]
        aggregate_color_counts.update(
            {
                True: summary["final_view_counts"]["color"],
                False: summary["final_view_counts"]["grayscale"],
                None: summary["final_view_counts"]["unknown"],
            }
        )
        aggregate_modes.update(summary["color_mutation_modes"])
        aggregate_toggles[(False, True)] += summary["toggle_counts"]["gray_to_color"]
        aggregate_toggles[(True, False)] += summary["toggle_counts"]["color_to_gray"]

    run_summary = {
        "agents": len(reports),
        "generations": total_generations,
        "color_views": aggregate_color_counts.get(True, 0),
        "grayscale_views": aggregate_color_counts.get(False, 0),
        "unknown_views": aggregate_color_counts.get(None, 0),
        "color_mutation_modes": dict(aggregate_modes),
        "toggles": {
            "gray_to_color": aggregate_toggles.get((False, True), 0),
            "color_to_gray": aggregate_toggles.get((True, False), 0),
        },
    }
    return reports, run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse color toggle usage from collaborative_multi_agent logs.",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to the collaborative_multi_agent_* run directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    reports, run_summary = analyse_run(run_dir)
    if args.json:
        payload = {
            "run": str(run_dir),
            "agents": [report.summary() for report in reports],
            "summary": run_summary,
        }
        print(json.dumps(payload, indent=2))
        return

    print(f"Run: {run_dir}")
    print(
        f"Agents: {run_summary['agents']}, generations={run_summary['generations']}, "
        f"color_views={run_summary['color_views']}, grayscale_views={run_summary['grayscale_views']}, "
        f"unknown={run_summary['unknown_views']}"
    )
    if run_summary["color_mutation_modes"]:
        mode_summary = ", ".join(
            f"{mode}={count}" for mode, count in sorted(run_summary["color_mutation_modes"].items())
        )
        print(f"Color mutation modes: {mode_summary}")
    if run_summary["toggles"]["gray_to_color"] or run_summary["toggles"]["color_to_gray"]:
        print(
            f"Toggles: gray→color={run_summary['toggles']['gray_to_color']}, "
            f"color→gray={run_summary['toggles']['color_to_gray']}"
        )
    print("")
    for report in reports:
        print(_format_agent_report(report))
        print("")


if __name__ == "__main__":
    main()
