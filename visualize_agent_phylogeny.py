#!/usr/bin/env python3
"""Render a phylogenetic view of every image produced during a single agent session."""

import argparse
import json
import html
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from graphviz import Digraph


PALETTE: Tuple[str, ...] = (
    "#fb8072",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
    "#8dd3c7",
    "#80b1d3",
    "#bebada",
)


@dataclass
class LineageEntry:
    genome_key: str
    agent_id: str
    generation: int
    image_index: int
    parent_keys: List[str] = field(default_factory=list)
    image_path: Optional[Path] = None
    order: int = 0


@dataclass
class PublicationRecord:
    genome_key: str
    title: str
    generation: Optional[int]
    image_index: Optional[int]
    is_final: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct and render the phylogenetic tree for a single agent session.",
    )
    parser.add_argument(
        "--agent-dir",
        type=Path,
        required=True,
        help="Path to the agent directory (containing images/ and logs/lineage.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (defaults to <agent-dir>/agent_phylogeny.<format>).",
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
        help="Rendering style: 'annotated' shows thumbnails plus text, 'images' renders thumbnails only.",
    )
    return parser.parse_args()


def _init_graph(output_format: str) -> Digraph:
    graph = Digraph("agent_phylogeny", format=output_format)
    graph.attr(rankdir="TB", bgcolor="white", nodesep="0.12", ranksep="1.1")
    graph.attr("node", fontname="Helvetica", fontsize="10", color="#333333")
    graph.attr("edge", color="#666666")
    return graph


def _resolve_generation_image(agent_dir: Path, generation: int, image_index: int) -> Optional[Path]:
    if generation < 0 or image_index < 0:
        return None
    images_dir = agent_dir / "images" / f"gen_{generation:03d}"
    image_path = images_dir / f"idx_{image_index:02d}.png"
    return image_path.resolve() if image_path.exists() else None


def _load_lineage(agent_dir: Path) -> List[LineageEntry]:
    lineage_path = agent_dir / "logs" / "lineage.jsonl"
    if not lineage_path.exists():
        raise FileNotFoundError(f"Lineage log not found at {lineage_path}")

    entries_by_key: Dict[str, LineageEntry] = {}
    order: List[str] = []
    with lineage_path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {lineage_path} line {line_num}: {exc}") from exc

            genome_key = record.get("genome_key")
            if genome_key is None:
                continue
            key = str(genome_key)

            generation = int(record.get("generation", -1))
            image_index = int(record.get("image_index", -1))
            parent_keys = [
                str(parent)
                for parent in (record.get("parent_genome_keys") or [])
                if parent is not None
            ]
            agent_id = str(record.get("agent_id") or agent_dir.name)
            image_path = _resolve_generation_image(agent_dir, generation, image_index)

            existing = entries_by_key.get(key)
            if existing is None:
                entry = LineageEntry(
                    genome_key=key,
                    agent_id=agent_id,
                    generation=generation,
                    image_index=image_index,
                    parent_keys=parent_keys,
                    image_path=image_path,
                    order=len(order),
                )
                entries_by_key[key] = entry
                order.append(key)
            else:
                if not existing.parent_keys and parent_keys:
                    existing.parent_keys = parent_keys
                if existing.image_path is None and image_path is not None:
                    existing.image_path = image_path
                if existing.generation < 0 <= generation:
                    existing.generation = generation
                if existing.image_index < 0 <= image_index:
                    existing.image_index = image_index

    ordered_entries = [entries_by_key[key] for key in order]
    ordered_entries.sort(key=lambda entry: (entry.generation if entry.generation >= 0 else float("inf"), entry.order))
    return ordered_entries


def _load_publications(agent_dir: Path) -> Dict[str, PublicationRecord]:
    publication_path = agent_dir / "logs" / "publication_history.jsonl"
    if not publication_path.exists():
        return {}

    raw_records: List[PublicationRecord] = []
    with publication_path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {publication_path} line {line_num}: {exc}") from exc

            genome_key = record.get("genome_key")
            if genome_key is None:
                continue
            title = str(record.get("title") or "").strip()
            raw_records.append(
                PublicationRecord(
                    genome_key=str(genome_key),
                    title=title or "untitled",
                    generation=record.get("generation"),
                    image_index=record.get("index"),
                    is_final=False,
                )
            )

    if raw_records:
        raw_records[-1].is_final = True
    return {record.genome_key: record for record in raw_records}


def _truncate_text(value: str, max_len: int = 40) -> str:
    value = value.strip()
    if max_len > 0 and len(value) > max_len:
        return value[: max_len - 1] + "…"
    return value


def _build_node_attributes(
    entry: LineageEntry,
    fill_color: str,
    publication: Optional[PublicationRecord],
    mode: str,
) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    node_label = f"genome {entry.genome_key}"
    meta_parts: List[str] = []
    if entry.generation >= 0:
        meta_parts.append(f"gen {entry.generation}")
    if entry.image_index >= 0:
        meta_parts.append(f"idx {entry.image_index}")
    meta_line = ", ".join(meta_parts) if meta_parts else "untracked"
    outline_style: List[str] = []

    if mode == "images" and entry.image_path is not None:
        attrs.update(
            {
                "shape": "box",
                "image": str(entry.image_path),
                "label": "",
                "imagescale": "true",
                "fixedsize": "false",
                "style": "rounded",
            }
        )
        outline_style.append("rounded")
    else:
        image_tag = ""
        if entry.image_path is not None:
            image_tag = f'<TR><TD><IMG SRC="{html.escape(str(entry.image_path))}" SCALE="TRUE"/></TD></TR>'
        publication_row = ""
        attrs["fillcolor"] = "#f7f7f7"
        if publication is not None:
            title = html.escape(_truncate_text(publication.title or "untitled"))
            pub_prefix = "Final" if publication.is_final else "Draft"
            publication_row = (
                f"<TR><TD ALIGN='LEFT'><FONT POINT-SIZE='9'><I>{pub_prefix}: {title}</I></FONT></TD></TR>"
            )
            attrs["fillcolor"] = fill_color
        label = (
            "<"
            "<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='2' BGCOLOR='white'>"
            f"{image_tag}"
            f"<TR><TD ALIGN='LEFT'><FONT POINT-SIZE='10'><B>{html.escape(_truncate_text(node_label))}</B></FONT></TD></TR>"
            f"<TR><TD ALIGN='LEFT'><FONT POINT-SIZE='9'>{html.escape(meta_line)}</FONT></TD></TR>"
            f"{publication_row}"
            "</TABLE>"
            ">"
        )
        attrs.update(
            {
                "shape": "plaintext",
                "label": label,
            }
        )
        outline_style.append("filled")

    penwidth = "1.0"
    if publication is not None:
        penwidth = "2.0"
        if not publication.is_final:
            outline_style.append("dashed")

    attrs["color"] = fill_color
    attrs["style"] = ",".join(outline_style) if outline_style else "solid"
    attrs["penwidth"] = penwidth
    return attrs


def build_agent_phylogeny_graph(
    entries: Iterable[LineageEntry],
    output_format: str,
    mode: str,
    publications: Dict[str, PublicationRecord],
) -> Digraph:
    graph = _init_graph(output_format)
    entries_list = list(entries)
    entries_by_id = {entry.genome_key: entry for entry in entries_list}
    edges: List[Tuple[str, str]] = []
    incoming: Dict[str, int] = {}
    generation_nodes: Dict[int, List[str]] = defaultdict(list)

    for entry in entries_list:
        for parent in entry.parent_keys:
            if parent == entry.genome_key:
                continue
            if parent not in entries_by_id:
                continue
            edges.append((parent, entry.genome_key))
            incoming[entry.genome_key] = incoming.get(entry.genome_key, 0) + 1
        if entry.generation >= 0:
            generation_nodes[entry.generation].append(entry.genome_key)

    for entry in entries_list:
        publication = publications.get(entry.genome_key)
        fill_color = PALETTE[0]

        attrs = _build_node_attributes(
            entry,
            fill_color=fill_color,
            publication=publication,
            mode=mode,
        )
        graph.node(entry.genome_key, **attrs)

    for parent, child in edges:
        graph.edge(parent, child)

    return graph


def main() -> None:
    args = parse_args()
    agent_dir = args.agent_dir.resolve()
    entries = _load_lineage(agent_dir)
    publications = _load_publications(agent_dir)
    if not entries:
        raise SystemExit("Lineage log contains no entries to visualise.")

    graph = build_agent_phylogeny_graph(
        entries,
        output_format=args.format,
        mode=args.mode,
        publications=publications,
    )

    if args.output is None:
        output_path = agent_dir / f"agent_phylogeny.{args.format}"
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

    final_path = rendered_path.with_suffix(f".{args.format}")
    if final_path != output_path and final_path.exists():
        final_path.rename(output_path)
        final_path = output_path

    print(f"Agent phylogenetic tree saved to {final_path}")


if __name__ == "__main__":
    main()
