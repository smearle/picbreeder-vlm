#!/usr/bin/env python3
"""Summarize embedding metrics across experiments and seeds.

This script searches a sweep directory for ``embedding_metrics.json`` files
produced by ``embed_and_visualize.py``. It writes two tables:

1. Per-experiment metrics with one row per run.
2. Aggregated metrics averaged across seeds (with standard deviations).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import re

__all__ = [
    "render_tables",
    "DEFAULT_METRICS_FILENAME",
    "aggregate_by_group_key",
    "format_latex_table",
]


DEFAULT_METRICS_FILENAME = "embedding_metrics.json"
PER_EXPERIMENT_CSV = "embedding_metrics_table.csv"
BY_SEED_CSV = "embedding_metrics_by_seed.csv"
BY_SEED_LATEX = "embedding_metrics_by_seed.tex"


TAG_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_tag(tag: str) -> str:
    cleaned = TAG_SANITIZE_PATTERN.sub("_", str(tag)).strip("_")
    return cleaned or "tag"


def _with_tag(filename: str, tag: Optional[str]) -> str:
    if not tag:
        return filename
    safe = _sanitize_tag(tag)
    path = Path(filename)
    return f"{path.stem}_{safe}{path.suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render tables of embedding metrics collected across experiments."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing experiment runs (searched recursively for embedding_metrics.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where CSV summaries are written (defaults to the root).",
    )
    parser.add_argument(
        "--metrics-name",
        default=DEFAULT_METRICS_FILENAME,
        help=f"Filename to search for within experiments (default: {DEFAULT_METRICS_FILENAME}).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal precision when printing float values (default: 4).",
    )
    return parser.parse_args()


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return not math.isnan(value)
    return False


def flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested metric dictionaries into a single-level mapping."""

    flat: Dict[str, Any] = {}

    def _sanitize(key: Any) -> str:
        return str(key).strip().replace(" ", "_")

    def _flatten(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                new_prefix = f"{prefix}_{_sanitize(sub_key)}" if prefix else _sanitize(sub_key)
                _flatten(new_prefix, sub_value)
        else:
            flat[prefix] = value

    for key, value in metrics.items():
        base = _sanitize(key)
        if isinstance(value, dict):
            _flatten(base, value)
        else:
            flat[base] = value
    return flat


SEED_PATTERN = re.compile(r"seed_(\d+)")


def extract_seed(experiment_name: str) -> int | None:
    match = SEED_PATTERN.search(experiment_name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def normalize_seed_component(name: str) -> str:
    """Remove the seed token from a path component while keeping other identifiers."""
    match = SEED_PATTERN.search(name)
    if not match:
        return name
    start, end = match.span()
    prefix = name[:start].rstrip("_-")
    suffix = name[end:].lstrip("_-")
    pieces = [piece for piece in (prefix, suffix) if piece]
    return "_".join(pieces) if pieces else "seedless"


def derive_group_key(relative_dir: Path) -> str:
    parts = list(relative_dir.parts)
    if not parts:
        return "seedless"
    normalized_parts = parts[:-1]
    normalized_parts.append(normalize_seed_component(parts[-1]))
    filtered = [part for part in normalized_parts if part and part != "."]
    return "/".join(filtered) if filtered else "seedless"


def collect_metrics(
    root: Path,
    metrics_name: str,
    *,
    experiment_dirs: Optional[Sequence[Path]] = None,
    agent_limits: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if experiment_dirs is None:
        search_paths = sorted(root.rglob(metrics_name))
    else:
        unique_dirs: List[Path] = []
        seen = set()
        for directory in experiment_dirs:
            resolved = Path(directory).expanduser().resolve()
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            unique_dirs.append(resolved)
        search_paths = [directory / metrics_name for directory in unique_dirs]

    for metrics_path in search_paths:
        if not metrics_path.exists():
            print(f"[warn] Metrics file missing: {metrics_path}", file=sys.stderr)
            continue
        try:
            raw_data = json.loads(metrics_path.read_text())
        except json.JSONDecodeError as exc:
            print(f"[warn] Failed to parse {metrics_path}: {exc}", file=sys.stderr)
            continue

        metrics_data: Dict[str, Any] = {}
        if isinstance(raw_data, list):
            # Handle trajectory format (list of dicts)
            if raw_data:
                target_entry = None
                
                # Try to find entry matching agent limit
                if agent_limits:
                    exp_path_str = str(metrics_path.parent.expanduser().resolve())
                    limit = agent_limits.get(exp_path_str)
                    if limit is not None:
                        for entry in raw_data:
                            if isinstance(entry, dict) and entry.get("index") == limit:
                                target_entry = entry
                                break
                
                # Fallback to last entry if no specific limit found or matched
                if target_entry is None:
                    target_entry = raw_data[-1]

                if isinstance(target_entry, dict):
                    # For embedding_mean_pairwise_distance_over_time*.json
                    if "mean_pairwise_distance" in target_entry:
                        metrics_data["mean_pairwise_distance"] = target_entry["mean_pairwise_distance"]
                    metrics_data.update(target_entry)
        elif isinstance(raw_data, dict):
            metrics_data = raw_data
        
        if not metrics_data:
             print(f"[warn] No valid metrics data found in {metrics_path}", file=sys.stderr)
             continue

        flattened = flatten_metrics(metrics_data)
        experiment_dir = metrics_path.parent
        experiment_name = experiment_dir.name

        try:
            relative_dir = experiment_dir.relative_to(root)
        except ValueError:
            relative_dir = experiment_dir
        if relative_dir == Path("."):
            relative_dir = Path(experiment_name)

        seed = extract_seed(experiment_name)
        group_key = derive_group_key(relative_dir)

        record: Dict[str, Any] = {
            "experiment": experiment_name,
            "seed": seed,
            "group_id": group_key,
            "relative_path": str(relative_dir),
        }
        record.update(flattened)
        records.append(record)

    return records


def determine_columns(records: Sequence[Dict[str, Any]]) -> List[str]:
    base_columns = ["experiment", "seed", "group_id", "relative_path"]
    metric_columns = sorted(
        {key for record in records for key in record.keys() if key not in base_columns}
    )
    return base_columns + metric_columns


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            row_copy = {key: row.get(key, "") for key in columns}
            writer.writerow(row_copy)


def format_table(
    columns: Sequence[str],
    rows: Sequence[Dict[str, Any]],
    *,
    precision: int,
) -> str:
    if not rows:
        return "(no data)"

    def _format(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.{precision}f}"
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        if value is None:
            return ""
        return str(value)

    widths = []
    for col in columns:
        column_values = [_format(row.get(col, "")) for row in rows]
        width = max(len(col), *(len(val) for val in column_values)) if column_values else len(col)
        widths.append(width)

    header = " | ".join(col.ljust(width) for col, width in zip(columns, widths))
    separator = "-+-".join("-" * width for width in widths)

    lines = [header, separator]
    for row in rows:
        formatted = [_format(row.get(col, "")) for col in columns]
        padded = [value.ljust(width) for value, width in zip(formatted, widths)]
        lines.append(" | ".join(padded))
    return "\n".join(lines)


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def format_latex_table(
    columns: Sequence[str],
    rows: Sequence[Dict[str, Any]],
    *,
    precision: int,
    caption: str = "Embedding metrics aggregated across seeds",
    label: str = "tab:embedding_metrics",
) -> str:
    """Format a table as a LaTeX tabular environment."""
    if not rows:
        return "% No data"

    def _format(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.{precision}f}"
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        if value is None:
            return ""
        return _escape_latex(str(value))

    # Build column specification
    col_spec = "l" * len(columns)
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{_escape_latex(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Header row
    header_cells = [_escape_latex(col) for col in columns]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for row in rows:
        cells = [_format(row.get(col, "")) for col in columns]
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def aggregate_by_seed(
    records: Sequence[Dict[str, Any]],
    numeric_columns: Sequence[str],
) -> List[Dict[str, Any]]:
    """Aggregate records by group_id, computing mean/std for numeric columns."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["group_id"], []).append(record)

    summaries: List[Dict[str, Any]] = []

    for group_id, group_records in grouped.items():
        summary: Dict[str, Any] = {
            "experiment_group": group_id,
            "seed_count": len(group_records),
        }

        seeds = sorted(
            str(record["seed"]) for record in group_records if record.get("seed") is not None
        )
        summary["seeds"] = ",".join(seeds) if seeds else ""

        for column in numeric_columns:
            values = [
                record[column]
                for record in group_records
                if column in record and _is_number(record[column])
            ]
            if not values:
                continue
            mean_value = statistics.mean(values)
            std_value = statistics.pstdev(values) if len(values) > 1 else 0.0
            summary[f"{column}_mean"] = mean_value
            summary[f"{column}_std"] = std_value

        summaries.append(summary)

    summaries.sort(key=lambda item: item["experiment_group"])
    return summaries


def aggregate_by_group_key(
    records: Sequence[Dict[str, Any]],
    numeric_columns: Sequence[str],
    *,
    group_key_field: str = "group_key",
) -> List[Dict[str, Any]]:
    """Aggregate records by an explicit group key (e.g., from config fields).

    This is the preferred aggregation method when group keys are derived from
    configuration fields rather than directory names.
    """
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for record in records:
        key = record.get(group_key_field)
        if key is None:
            key = record.get("group_id", "default")
        # Convert tuple keys to string for consistent handling
        key_str = str(key) if not isinstance(key, str) else key
        grouped.setdefault(key_str, []).append(record)

    summaries: List[Dict[str, Any]] = []

    for group_key_str, group_records in grouped.items():
        summary: Dict[str, Any] = {
            "experiment_group": group_key_str,
            "seed_count": len(group_records),
        }

        seeds = sorted(
            str(record["seed"]) for record in group_records if record.get("seed") is not None
        )
        summary["seeds"] = ",".join(seeds) if seeds else ""

        for column in numeric_columns:
            values = [
                record[column]
                for record in group_records
                if column in record and _is_number(record[column])
            ]
            if not values:
                continue
            mean_value = statistics.mean(values)
            std_value = statistics.stdev(values) if len(values) > 1 else 0.0
            summary[f"{column}_mean"] = mean_value
            summary[f"{column}_std"] = std_value

        summaries.append(summary)

    summaries.sort(key=lambda item: item["experiment_group"])
    return summaries


def render_tables(
    root: Path,
    *,
    output_dir: Optional[Path] = None,
    metrics_name: str = DEFAULT_METRICS_FILENAME,
    precision: int = 4,
    experiment_dirs: Optional[Sequence[Path]] = None,
    filename_tag: Optional[str] = None,
    group_labels: Optional[Dict[str, str]] = None,
    agent_limits: Optional[Dict[str, int]] = None,
) -> Tuple[str, str, Path, Path, Path]:
    """Render per-experiment and aggregated metrics tables.

    Args:
        root: Base directory for experiment search.
        output_dir: Destination for output files (defaults to root).
        metrics_name: Filename to search for within experiments.
        precision: Decimal precision for float formatting.
        experiment_dirs: Explicit list of experiment directories to include.
        filename_tag: Tag appended to output filenames.
        group_labels: Mapping from experiment paths to group labels for aggregation.
            If provided, experiments with the same label are aggregated together.
        agent_limits: Mapping from experiment path (str) to agent limit (int).
            Used to extract metrics at a specific step from trajectory files.

    Returns:
        Tuple of (per_experiment_table, aggregated_table, per_csv_path, agg_csv_path, latex_path).
    """
    records = collect_metrics(
        root,
        metrics_name,
        experiment_dirs=experiment_dirs,
        agent_limits=agent_limits,
    )
    if not records:
        raise ValueError("No embedding metrics found.")

    # Apply explicit group labels if provided
    if group_labels:
        for record in records:
            exp_path = record.get("relative_path", "")
            # Try to match by experiment name or full path
            label = group_labels.get(record["experiment"]) or group_labels.get(exp_path)
            if label:
                record["group_id"] = label

    columns = determine_columns(records)

    numeric_columns: List[str] = []
    for column in columns:
        if column in {"experiment", "seed", "group_id", "relative_path"}:
            continue
        if any(_is_number(record.get(column)) for record in records):
            numeric_columns.append(column)

    destination = output_dir.expanduser().resolve() if output_dir else root
    destination.mkdir(parents=True, exist_ok=True)

    per_experiment_csv_path = destination / _with_tag(PER_EXPERIMENT_CSV, filename_tag)
    write_csv(per_experiment_csv_path, columns, records)
    per_experiment_table = format_table(columns, records, precision=precision)

    aggregated_records = aggregate_by_seed(records, numeric_columns)
    agg_columns = ["experiment_group", "seed_count", "seeds"]
    extra_metric_keys = sorted(
        {key for summary in aggregated_records for key in summary.keys()} - set(agg_columns)
    )
    agg_columns.extend(extra_metric_keys)

    by_seed_csv_path = destination / _with_tag(BY_SEED_CSV, filename_tag)
    write_csv(by_seed_csv_path, agg_columns, aggregated_records)
    aggregated_table = format_table(agg_columns, aggregated_records, precision=precision)

    # Write LaTeX version
    latex_path = destination / _with_tag(BY_SEED_LATEX, filename_tag)
    latex_table = format_latex_table(
        agg_columns,
        aggregated_records,
        precision=precision,
        caption=f"Embedding metrics aggregated across seeds ({filename_tag or 'sweep'})",
        label=f"tab:embedding_metrics_{_sanitize_tag(filename_tag or 'sweep')}",
    )
    latex_path.write_text(latex_table, encoding="utf-8")

    return per_experiment_table, aggregated_table, per_experiment_csv_path, by_seed_csv_path, latex_path


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        print(f"Root directory not found: {root}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else None

    try:
        per_table, agg_table, per_csv, agg_csv, latex_path = render_tables(
            root,
            output_dir=output_dir,
            metrics_name=args.metrics_name,
            precision=args.precision,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("Per-experiment metrics:\n")
    # print(per_table)
    print(f"\nWrote CSV: {per_csv}")

    print("\nAggregated across seeds:\n")
    # print(agg_table)
    print(f"\nWrote CSV: {agg_csv}")
    print(f"Wrote LaTeX: {latex_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
