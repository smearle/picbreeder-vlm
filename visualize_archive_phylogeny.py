#!/usr/bin/env python3
"""Render a phylogenetic view of archived Picbreeder images."""

import argparse
import json
from collections import defaultdict
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
    parser.add_argument(
        "--scope",
        choices=("archive", "full"),
        default="archive",
        help="Select 'archive' to show only archive entries; 'full' includes all recorded genomes and archive entries.",
    )
    return parser.parse_args()

PALETTE: Tuple[str, ...] = (
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


def _init_graph(output_format: str) -> Digraph:
    graph = Digraph("archive_phylogeny", format=output_format)
    graph.attr(rankdir="LR", bgcolor="white", nodesep="0.6", ranksep="1")
    graph.attr("node", fontname="Helvetica", fontsize="10", color="#333333")
    graph.attr("edge", color="#666666")
    return graph


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


def _load_selected_entry_ids(branch_path: Path) -> List[str]:
    if not branch_path.exists():
        return []
    try:
        data = json.loads(branch_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    selected = data.get("selected_entry_ids") or []
    result: List[str] = []
    for value in selected:
        if value is None:
            continue
        result.append(str(value))
    return result


def _resolve_generation_image(agent_dir: Path, generation: int, image_index: int) -> Optional[Path]:
    images_dir = agent_dir / "images" / f"gen_{generation:03d}"
    image_path = images_dir / f"idx_{image_index:02d}.png"
    return image_path.resolve() if image_path.exists() else None


def _build_archive_node_attributes(
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


def _build_archive_graph(
    entries: Iterable[Dict[str, Any]],
    *,
    experiment_dir: Path,
    archive_dir: Path,
    output_format: str,
    mode: str,
) -> Digraph:
    graph = _init_graph(output_format)
    assigned_colors: Dict[str, str] = {}
    entries_by_id: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        entry_id_raw = entry.get("id")
        if entry_id_raw is None:
            continue
        entry_id = str(entry_id_raw)
        entries_by_id[entry_id] = entry

    edges: Set[Tuple[str, str]] = set()

    for entry_id, entry in entries_by_id.items():
        parent_ids: Set[str] = set()
        source_experiment = entry.get("source_experiment")
        if source_experiment:
            agent_dir = Path(source_experiment)
            if not agent_dir.is_absolute():
                agent_dir = (experiment_dir / agent_dir).resolve()
            branch_path = agent_dir / "logs" / "branching_selection.json"
            for parent in _load_selected_entry_ids(branch_path):
                if parent != entry_id and parent in entries_by_id:
                    parent_ids.add(parent)
        if not parent_ids:
            for value in entry.get("source_entry_ids", []) or []:
                parent = str(value)
                if parent != entry_id and parent in entries_by_id:
                    parent_ids.add(parent)
        for parent in parent_ids:
            edges.add((parent, entry_id))

    incoming = defaultdict(int)
    for _, dst in edges:
        incoming[dst] += 1
    root_ids = {entry_id for entry_id in entries_by_id if incoming.get(entry_id, 0) == 0}

    for entry_id, entry in entries_by_id.items():
        agent_id = str(entry.get("agent_id", "unknown"))
        fill_color = _choose_color(agent_id, PALETTE, assigned_colors)
        image_path = _resolve_image_path(entry, archive_dir)
        attrs = _build_archive_node_attributes(
            entry=entry,
            image_path=image_path,
            fill_color=fill_color,
            is_root=entry_id in root_ids,
            mode=mode,
        )
        graph.node(entry_id, **attrs)

    for parent, child in sorted(edges):
        graph.edge(parent, child)

    return graph


def _lighten_color(hex_color: str, amount: float = 0.35) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return hex_color
    try:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
    except ValueError:
        return hex_color
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_genome_node_attributes(
    *,
    node_id: str,
    record: Dict[str, Any],
    agent_color: str,
    is_root: bool,
    image_path: Optional[Path],
    mode: str,
) -> Dict[str, str]:
    generation = record.get("generation")
    image_index = record.get("image_index")
    genome_key = record.get("genome_key")
    info_bits: List[str] = []
    if generation is not None:
        info_bits.append(f"g{generation}")
    if image_index is not None:
        info_bits.append(f"i{image_index}")
    if genome_key is not None:
        info_bits.append(f"k{genome_key}")
    caption = " ".join(str(bit) for bit in info_bits) if info_bits else ""

    if mode == "images" and image_path is not None:
        attrs = {
            "shape": "box",
            "image": str(image_path),
            "label": "",
            "imagescale": "true",
            "fixedsize": "false",
            "color": agent_color,
            "style": "rounded",
        }
        if caption:
            attrs["xlabel"] = caption
            attrs["labelloc"] = "b"
        return attrs

    image_tag = (
        f'<TR><TD><IMG SRC="{html.escape(str(image_path))}" SCALE="TRUE"/></TD></TR>'
        if image_path is not None
        else ""
    )
    caption_tag = (
        f"<TR><TD ALIGN='LEFT'><FONT POINT-SIZE='9'>{html.escape(caption)}</FONT></TD></TR>"
        if caption
        else ""
    )
    label = (
        "<"
        "<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='2' BGCOLOR='white'>"
        f"{image_tag}"
        f"<TR><TD ALIGN='LEFT'><FONT POINT-SIZE='10'><B>{html.escape(node_id)}</B></FONT></TD></TR>"
        f"{caption_tag}"
        "</TABLE>"
        ">"
    )

    attrs = {
        "shape": "plaintext",
        "label": label,
        "style": "filled",
        "fillcolor": _lighten_color(agent_color, 0.65),
        "color": agent_color,
    }
    if is_root:
        attrs["penwidth"] = "2"
    return attrs


#FIXME: Something here is broken
def _build_full_phylogeny_graph(
    entries: Iterable[Dict[str, Any]],
    *,
    experiment_dir: Path,
    archive_dir: Path,
    output_format: str,
    mode: str,
) -> Digraph:
    graph = _init_graph(output_format)
    assigned_colors: Dict[str, str] = {}
    entries_by_id: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        entry_id_raw = entry.get("id")
        if entry_id_raw is None:
            continue
        entry_id = str(entry_id_raw)
        entries_by_id[entry_id] = entry

    nodes: Dict[str, Dict[str, Any]] = {}
    node_agent: Dict[str, str] = {}
    for entry_id, entry in entries_by_id.items():
        agent_id = str(entry.get("agent_id", "unknown"))
        nodes[entry_id] = {"type": "archive", "record": entry}
        node_agent[entry_id] = agent_id

    edges: Set[Tuple[str, str]] = set()
    node_index_map: Dict[Tuple[str, int, int], str] = {}

    agents_dir = experiment_dir / "agents"
    if agents_dir.exists():
        for agent_dir in sorted(agents_dir.iterdir())[:1]:
            if not agent_dir.is_dir():
                continue
            logs_path = agent_dir / "logs" / "lineage.jsonl"
            if not logs_path.exists():
                continue
            latest_node_by_key: Dict[Tuple[str, int], str] = {}
            with logs_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    agent_id = str(record.get("agent_id") or agent_dir.name)
                    genome_key = record.get("genome_key")
                    generation = record.get("generation")
                    image_index = record.get("image_index")
                    if genome_key is None or generation is None or image_index is None:
                        continue
                    try:
                        genome_key_int = int(genome_key)
                        generation_int = int(generation)
                        image_index_int = int(image_index)
                    except (TypeError, ValueError):
                        continue
                    node_id = f"{agent_id}#g{generation_int}#i{image_index_int}#k{genome_key_int}"
                    image_path = _resolve_generation_image(agent_dir, generation_int, image_index_int)
                    nodes[node_id] = {
                        "type": "genome",
                        "record": record,
                        "image_path": image_path,
                    }
                    node_agent[node_id] = agent_id
                    node_index_map[(agent_id, generation_int, image_index_int)] = node_id

                    for parent_raw in record.get("parent_genome_keys") or []:
                        try:
                            parent_key = int(parent_raw)
                        except (TypeError, ValueError):
                            continue
                        parent_node_id = latest_node_by_key.get((agent_id, parent_key))
                        if parent_node_id:
                            edges.add((parent_node_id, node_id))

                    for source_raw in record.get("source_entry_ids") or []:
                        if source_raw is None:
                            continue
                        source_id = str(source_raw)
                        if source_id in entries_by_id:
                            edges.add((source_id, node_id))

                    latest_node_by_key[(agent_id, genome_key_int)] = node_id

    for entry_id, entry in entries_by_id.items():
        agent_id = str(entry.get("agent_id", "unknown"))
        generation = entry.get("generation")
        image_index = entry.get("image_index")
        try:
            generation_int = int(generation)
            image_index_int = int(image_index)
        except (TypeError, ValueError):
            continue
        node_id = node_index_map.get((agent_id, generation_int, image_index_int))
        if node_id:
            edges.add((node_id, entry_id))

    incoming = defaultdict(int)
    for _, dst in edges:
        incoming[dst] += 1
    for node_id in nodes:
        incoming.setdefault(node_id, 0)
    root_ids = {node_id for node_id, count in incoming.items() if count == 0}

    for node_id, meta in nodes.items():
        agent_id = node_agent.get(node_id, "unknown")
        color = _choose_color(agent_id, PALETTE, assigned_colors)
        if meta["type"] == "archive":
            entry = meta["record"]
            image_path = _resolve_image_path(entry, archive_dir)
            attrs = _build_archive_node_attributes(
                entry=entry,
                image_path=image_path,
                fill_color=color,
                is_root=node_id in root_ids,
                mode=mode,
            )
        else:
            record = meta["record"]
            attrs = _build_genome_node_attributes(
                node_id=node_id,
                record=record,
                agent_color=color,
                is_root=node_id in root_ids,
                image_path=meta.get("image_path"),
                mode=mode,
            )
        graph.node(node_id, **attrs)

    for parent, child in sorted(edges):
        graph.edge(parent, child)

    return graph


def build_phylogeny_graph(
    entries: Iterable[Dict[str, Any]],
    *,
    experiment_dir: Path,
    archive_dir: Path,
    output_format: str,
    mode: str,
    scope: str,
) -> Digraph:
    if scope == "full":
        return _build_full_phylogeny_graph(
            entries,
            experiment_dir=experiment_dir,
            archive_dir=archive_dir,
            output_format=output_format,
            mode=mode,
        )
    return _build_archive_graph(
        entries,
        experiment_dir=experiment_dir,
        archive_dir=archive_dir,
        output_format=output_format,
        mode=mode,
    )


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    archive_dir = experiment_dir / "archive"
    entries = _load_archive_entries(archive_dir)
    if not entries:
        raise SystemExit("Archive metadata contains no entries to visualise.")

    graph = build_phylogeny_graph(
        entries,
        experiment_dir=experiment_dir,
        archive_dir=archive_dir,
        output_format=args.format,
        mode=args.mode,
        scope=args.scope,
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
