#!/usr/bin/env python3
"""Reconstruct and render the global Picbreeder phylogeny using Graphviz."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from graphviz import Digraph
from tqdm.auto import tqdm

from fer.src.lineage_utils import get_lineage_genomes, get_pid_lineage

VALID_FORMATS: Tuple[str, ...] = ("pdf", "png", "svg")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan every Picbreeder lineage, detect overlaps, and plot the merged tree via Graphviz."
        ),
    )
    parser.add_argument(
        "--pb-dir",
        type=Path,
        default=Path("fer/spaghetti/pbRender/genomeAll"),
        help="Directory that contains pid subdirectories (each with a main.zip).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/lineages/full_phylogeny"),
        help="Output path without suffix (default: figures/lineages/full_phylogeny).",
    )
    parser.add_argument(
        "--format",
        choices=VALID_FORMATS,
        default="pdf",
        help="Graphviz render format (default: pdf).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of pid folders to scan (helps with smoke tests).",
    )
    parser.add_argument(
        "--min-edge-weight",
        type=int,
        default=1,
        help="Hide edges that appear in fewer than N distinct lineages (default: 1).",
    )
    parser.add_argument(
        "--rankdir",
        choices=("LR", "TB"),
        default="LR",
        help="Graph orientation passed to Graphviz (default: LR).",
    )
    return parser.parse_args()


def _discover_pids(pb_dir: Path) -> List[str]:
    if not pb_dir.is_dir():
        raise FileNotFoundError(f"Picbreeder directory does not exist: {pb_dir}")
    pids: List[str] = []
    for entry in sorted(pb_dir.iterdir()):
        if entry.is_dir() and (entry / "main.zip").exists():
            pids.append(entry.name)
    if not pids:
        raise SystemExit(f"No pid directories found in {pb_dir}")
    return pids


def _collect_lineages(pb_dir: Path, pids: Iterable[str]) -> Dict[str, List[str]]:
    lineages: Dict[str, List[str]] = {}
    for pid in tqdm(pids, desc="Collecting lineages"):
        try:
            lineage = get_lineage_genomes(pb_dir, pid)
        except Exception as exc:  # pragma: no cover - defensive logging
            tqdm.write(f"[WARN] Failed to parse lineage for pid {pid}: {exc}")
            continue
        if len(lineage) <= 1:
            tqdm.write(f"[WARN] Lineage for pid {pid} contains no parent; keeping singleton node.")
        lineages[pid] = lineage
    if not lineages:
        raise SystemExit("No valid lineages were recovered; nothing to render.")
    return lineages


def _build_parent_maps(
    lineages: List[str]
) -> Tuple[Dict[str, str], Dict[str, Set[str]], Dict[Tuple[str, str], int]]:
    parent_for: Dict[str, str] = {}
    children_for: Dict[str, Set[str]] = defaultdict(set)
    edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for lineage in lineages:
        for parent, child in zip(lineage, lineage[1:]):
            if child not in parent_for:
                parent_for[child] = parent
            children_for[parent].add(child)
            edge_counts[(parent, child)] += 1
    return parent_for, children_for, edge_counts


def _compute_roots(nodes: Set[str], parent_for: Mapping[str, str]) -> List[str]:
    roots = sorted(node for node in nodes if node not in parent_for)
    if not roots:
        roots = sorted(nodes)
    return roots


def _compute_depths(nodes: Iterable[str], parent_for: Mapping[str, str]) -> Dict[str, int]:
    depths: Dict[str, int] = {}

    def resolve(pid: str) -> int:
        if pid in depths:
            return depths[pid]
        parent = parent_for.get(pid)
        if parent is None or parent == pid:
            depths[pid] = 0
        else:
            depths[pid] = resolve(parent) + 1
        return depths[pid]

    for node in nodes:
        resolve(node)
    return depths


def _compute_descendants(children_for: Mapping[str, Set[str]], roots: Sequence[str]) -> Dict[str, int]:
    descendants: Dict[str, int] = {}

    def count(node: str) -> int:
        if node in descendants:
            return descendants[node]
        total = 0
        for child in children_for.get(node, set()):
            total += 1 + count(child)
        descendants[node] = total
        return total

    for root in roots:
        count(root)
    return descendants


def _assign_root_colors(roots: Sequence[str]) -> Dict[str, str]:
    color_map: Dict[str, str] = {}
    for idx, root in enumerate(roots):
        color_map[root] = PALETTE[idx % len(PALETTE)]
    return color_map


def _lighten_color(hex_color: str, factor: float = 0.5) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return hex_color
    try:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
    except ValueError:
        return hex_color
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _resolve_root(pid: str, parent_for: Mapping[str, str]) -> str:
    current = pid
    seen: Set[str] = set()
    while True:
        parent = parent_for.get(current)
        if parent is None or parent in seen:
            return current
        seen.add(current)
        current = parent


def _edge_penwidth(weight: int) -> str:
    if weight <= 1:
        return "1.0"
    return f"{1.0 + min(3.0, math.log(weight + 1, 2)):0.2f}"


def _init_graph(output_format: str, rankdir: str) -> Digraph:
    graph = Digraph("picbreeder_phylogeny", format=output_format)
    graph.attr(rankdir=rankdir, bgcolor="white", nodesep="0.1", ranksep="0.7")
    graph.attr("node", fontname="Helvetica", fontsize="9", shape="box", style="rounded")
    graph.attr("edge", color="#4a4a4a")
    return graph


def build_phylogeny_graph(
    lineages: List[str],
    output_format: str,
    rankdir: str,
    min_edge_weight: int,
) -> Digraph:
    nodes: Set[str] = set()
    for lineage in lineages:
        nodes.update(lineage)
    parent_for, children_for, edge_counts = _build_parent_maps(lineages)
    roots = _compute_roots(nodes, parent_for)
    depths = _compute_depths(nodes, parent_for)
    descendants = _compute_descendants(children_for, roots)
    root_colors = _assign_root_colors(roots)

    graph = _init_graph(output_format, rankdir)

    resolved_root_cache: Dict[str, str] = {}
    for node in sorted(nodes, key=lambda item: (depths.get(item, 0), item)):
        if node in resolved_root_cache:
            root = resolved_root_cache[node]
        else:
            root = _resolve_root(node, parent_for)
            resolved_root_cache[node] = root
        color = root_colors.get(root, "#dddddd")
        fill = _lighten_color(color, factor=0.5)
        depth = depths.get(node, 0)
        desc = descendants.get(node, 0)
        label = f"{node}\\ndepth={depth}\\ndesc={desc}"
        attrs = {
            "label": label,
            "style": "rounded,filled",
            "fillcolor": fill,
            "color": color,
            "penwidth": "2" if node in roots else "1",
        }
        graph.node(node, **attrs)

    for (parent, child), weight in sorted(edge_counts.items()):
        if weight < min_edge_weight:
            continue
        edge_attrs = {
            "penwidth": _edge_penwidth(weight),
        }
        if weight > 1:
            edge_attrs["label"] = str(weight)
        graph.edge(parent, child, **edge_attrs)

    return graph


def main() -> None:
    args = parse_args()
    pb_dir = args.pb_dir.expanduser().resolve()
    output_base = args.output.expanduser().resolve()
    if output_base.suffix:
        output_base = output_base.with_suffix("")

    pids = _discover_pids(pb_dir)
    if args.limit is not None:
        pids = pids[: args.limit]
    lineages = _collect_lineages(pb_dir, pids)

    graph = build_phylogeny_graph(
        lineages=lineages,
        output_format=args.format,
        rankdir=args.rankdir,
        min_edge_weight=max(1, args.min_edge_weight),
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    rendered = Path(
        graph.render(
            filename=output_base.name,
            directory=str(output_base.parent),
            cleanup=True,
        )
    )
    final_path = rendered.with_suffix(f".{args.format}")
    print(f"Saved Picbreeder phylogeny to {final_path}")


if __name__ == "__main__":
    main()
