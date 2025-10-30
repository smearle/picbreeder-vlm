#!/usr/bin/env python3
"""Render a phylogenetic view of archived Picbreeder images."""

import argparse
import json
from datetime import datetime
from pathlib import Path
import html
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from graphviz import Digraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct and render the phylogenetic tree for an experiment archive.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the collaborative experiment directory (containing the archive folder).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (defaults to archive/archive_phylogeny.<format>).",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Graphviz output format.",
    )
    parser.add_argument(
        "--mode",
        choices=("annotated", "images"),
        default="annotated",
        help=(
            "Rendering style: 'annotated' shows thumbnails plus text, 'images' renders thumbnails only."
        ),
    )
    return parser.parse_args()


def _load_archive_entries(archive_dir: Path) -> List[Dict[str, Any]]:
    metadata_path = archive_dir / "archive_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Archive metadata not found at {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    entries = list(metadata.get("entries", []))

    def sort_key(entry: Dict[str, Any]) -> Tuple[int, datetime]:
        timestamp = entry.get("added_at")
        if isinstance(timestamp, str):
            try:
                return (0, datetime.fromisoformat(timestamp))
            except ValueError:
                pass
        return (1, datetime.min)

    entries.sort(key=sort_key)
    return entries


def _choose_color(agent_id: str, palette: Sequence[str], assigned: Dict[str, str]) -> str:
    if agent_id in assigned:
        return assigned[agent_id]
    color = palette[len(assigned) % len(palette)]
    assigned[agent_id] = color
    return color


def _resolve_image_path(entry: Dict[str, Any], archive_dir: Path) -> Optional[Path]:
    raw_path = entry.get("image_path")
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = (archive_dir / path).resolve()
    return path if path.exists() else None


def _truncate_text(value: str, max_len: int = 40) -> str:
    value = value.strip()
    if max_len > 0 and len(value) > max_len:
        return value[: max_len - 1] + "…"
    return value


def _build_node_attributes(
    *,
    entry: Dict[str, Any],
    image_path: Optional[Path],
    fill_color: str,
    is_root: bool,
    mode: str,
) -> Dict[str, str]:
    entry_id = str(entry.get("id"))
    attrs: Dict[str, str] = {}

    if mode == "images" and image_path is not None:
        attrs.update(
            {
                "shape": "box",
                "image": str(image_path),
                "label": "",
                "imagescale": "true",
                "fixedsize": "false",
                "color": fill_color,
                "style": "rounded",
            }
        )
    else:
        title = _truncate_text(str(entry.get("title") or "untitled"))
        agent_id = str(entry.get("agent_id", "unknown"))
        generation = entry.get("generation")
        agent_line = f"{agent_id} | gen {generation}"

        if image_path is not None:
            image_tag = f'<TR><TD><IMG SRC="{html.escape(str(image_path))}" SCALE="TRUE"/></TD></TR>'
        else:
            image_tag = ""

        label = (
            "<"
            "<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='2' BGCOLOR='white'>"
            f"{image_tag}"
            # f"<TR><TD ALIGN='LEFT'><FONT POINT-SIZE='10'><B>{html.escape(entry_id)}</B></FONT></TD></TR>"
            f"<TR><TD ALIGN='LEFT'><FONT POINT-SIZE='10'><B>{html.escape(title)}</B></FONT></TD></TR>"
            f"<TR><TD ALIGN='LEFT'><FONT POINT-SIZE='9'>{html.escape(agent_line)}</FONT></TD></TR>"
            "</TABLE>"
            ">"
        )

        attrs.update(
            {
                "shape": "plaintext",
                "label": label,
                "color": fill_color,
                "style": "filled",
                "fillcolor": "#f7f7f7",
            }
        )

    if is_root:
        attrs["penwidth"] = "2"
        attrs["peripheries"] = "2"

    return attrs


def _collect_parent_links(
    entry: Dict[str, Any],
    *,
    known_entry_ids: Set[str],
    genome_to_entry_id: Dict[int, str],
) -> Set[str]:
    parents: Set[str] = set()

    # Direct links recorded in the archive entry metadata.
    for value in entry.get("source_entry_ids", []) or []:
        if value is None:
            continue
        parent_id = str(value)
        if parent_id in known_entry_ids and parent_id != entry.get("id"):
            parents.add(parent_id)

    # Fallback: infer parent candidates via recorded genome keys.
    for raw_parent_key in entry.get("parent_genome_keys", []) or []:
        try:
            parent_key = int(raw_parent_key)
        except (TypeError, ValueError):
            continue
        parent_id = genome_to_entry_id.get(parent_key)
        if parent_id and parent_id in known_entry_ids and parent_id != entry.get("id"):
            parents.add(parent_id)

    return parents


def build_phylogeny_graph(
    entries: Iterable[Dict[str, Any]],
    *,
    archive_dir: Path,
    output_format: str,
    mode: str,
) -> Digraph:
    graph = Digraph("archive_phylogeny", format=output_format)
    graph.attr(rankdir="LR", bgcolor="white", nodesep="0.6", ranksep="1")
    graph.attr("node", fontname="Helvetica", fontsize="10", color="#333333")
    graph.attr("edge", color="#666666")

    palette = (
        "#8dd3c7",
        "#80b1d3",
        "#bebada",
        "#fb8072",
        "#b3de69",
        "#fccde5",
        "#d9d9d9",
        "#bc80bd",
        "#ccebc5",
        "#ffed6f",
    )
    assigned_colors: Dict[str, str] = {}

    known_ids: Set[str] = set()
    prepared_entries: List[Dict[str, Any]] = []
    genome_to_entry_id: Dict[int, str] = {}
    for entry in entries:
        entry_id = str(entry.get("id")) if entry.get("id") is not None else None
        if not entry_id:
            continue
        known_ids.add(entry_id)
        prepared_entries.append(entry)
        try:
            genome_key = int(entry.get("genome_key")) if entry.get("genome_key") is not None else None
        except (TypeError, ValueError):
            genome_key = None
        if genome_key is not None and genome_key not in genome_to_entry_id:
            genome_to_entry_id[genome_key] = entry_id

    edges: Set[Tuple[str, str]] = set()
    root_ids: Set[str] = set()

    for entry in prepared_entries:
        entry_id = str(entry.get("id"))
        agent_id = str(entry.get("agent_id", "unknown"))
        parent_ids = _collect_parent_links(
            entry,
            known_entry_ids=known_ids,
            genome_to_entry_id=genome_to_entry_id,
        )

        if parent_ids:
            for parent in parent_ids:
                edges.add((parent, entry_id))
        else:
            root_ids.add(entry_id)

        fill_color = _choose_color(agent_id, palette, assigned_colors)
        image_path = _resolve_image_path(entry, archive_dir)
        attrs = _build_node_attributes(
            entry=entry,
            image_path=image_path,
            fill_color=fill_color,
            is_root=entry_id in root_ids,
            mode=mode,
        )
        graph.node(entry_id, **attrs)

    for parent_id, child_id in sorted(edges):
        graph.edge(parent_id, child_id)

    return graph


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    archive_dir = experiment_dir / "archive"
    entries = _load_archive_entries(archive_dir)
    if not entries:
        raise SystemExit("Archive metadata contains no entries to visualise.")

    graph = build_phylogeny_graph(
        entries,
        archive_dir=archive_dir,
        output_format=args.format,
        mode=args.mode,
    )

    if args.output is None:
        output_path = archive_dir / f"archive_phylogeny.{args.format}"
    else:
        output_path = args.output
    output_path = output_path.resolve()
    if output_path.suffix.lower() != f".{args.format}":
        output_path = output_path.with_suffix(f".{args.format}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rendered_path = Path(
        graph.render(
            filename=output_path.stem,
            directory=str(output_path.parent),
            cleanup=True,
        )
    )

    # graphviz.render adds the requested suffix automatically; ensure final path matches expectation.
    final_path = rendered_path.with_suffix(f".{args.format}")
    if final_path != output_path and final_path.exists():
        final_path.rename(output_path)
        final_path = output_path

    print(f"Phylogenetic tree saved to {final_path}")


if __name__ == "__main__":
    main()
