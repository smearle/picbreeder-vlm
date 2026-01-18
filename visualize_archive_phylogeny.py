#!/usr/bin/env python3
"""Render a phylogenetic view of archived Picbreeder images."""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import html
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import hydra
from graphviz import Digraph
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd

import tree_metrics
from config import PicbreederConfig, ensure_valid_config
from utils import _ensure_absolute, _resolve_image_path, _resolve_source_experiment_dir


VALID_FORMATS: Tuple[str, ...] = ("png", "pdf", "svg")
VALID_MODES: Tuple[str, ...] = ("annotated", "images")
VALID_SCOPES: Tuple[str, ...] = ("archive", "full")


@dataclass
class ArchivePhylogenyConfig(PicbreederConfig):
    output: Optional[Path] = None
    format: str = "pdf"
    mode: str = "annotated"
    scope: str = "archive"
    archive_limit: Optional[int] = None
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            help=HelpConf(
                app_name="visualize_archive_phylogeny",
                header=(
                    "Hydra entry point for archive phylogeny visualization.\n"
                    "\n"
                    "Key options:\n"
                    "  experiment_dir          Override to point directly at an existing run.\n"
                    "  goal / scheme / seed    Used with ensure_valid_config to infer experiment path.\n"
                    "  mode / format           Control rendering style and Graphviz output format.\n"
                    "  archive_limit           Trim the number of archive entries considered.\n"
                ),
                footer="Override with +option=value (e.g. format=svg mode=images).",
            )
        )
    )


ConfigStore.instance().store(name="archive_phylogeny_base", node=ArchivePhylogenyConfig)

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


def _validate_visualization_options(cfg: ArchivePhylogenyConfig) -> None:
    if cfg.format not in VALID_FORMATS:
        raise ValueError(f"format must be one of {VALID_FORMATS}")
    if cfg.mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}")
    if cfg.scope not in VALID_SCOPES:
        raise ValueError(f"scope must be one of {VALID_SCOPES}")
    if cfg.archive_limit is not None and cfg.archive_limit <= 0:
        raise ValueError("archive_limit must be a positive integer when provided")


def _init_graph(output_format: str) -> Digraph:
    graph = Digraph("archive_phylogeny", format=output_format)
    graph.attr(rankdir="TB", bgcolor="white", nodesep="0.1", ranksep="1")
    graph.attr("node", fontname="Helvetica", fontsize="10", color="#333333")
    graph.attr("edge", color="#666666")
    return graph


def _choose_color(agent_id: str, palette: Sequence[str], assigned: Dict[str, str]) -> str:
    if agent_id in assigned:
        return assigned[agent_id]
    color = palette[len(assigned) % len(palette)]
    assigned[agent_id] = color
    return color


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
    bg_color: str,
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
            f"<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='2' BGCOLOR='{bg_color}'>"
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
                "fillcolor": bg_color,
            }
        )

    if is_root:
        attrs["penwidth"] = "2"
        attrs["peripheries"] = "2"

    return attrs


def _compute_leaf_agent_lineages(
    *,
    entries_by_id: Dict[str, Dict[str, Any]],
    entry_order: Dict[str, int],
    edges: Set[Tuple[str, str]],
) -> Dict[str, List[str]]:
    parent_map: Dict[str, List[str]] = defaultdict(list)
    child_map: Dict[str, List[str]] = defaultdict(list)
    for parent, child in edges:
        parent_map[child].append(parent)
        child_map[parent].append(child)

    leaf_ids: List[str] = []
    for entry_id in entries_by_id:
        if not child_map.get(entry_id):
            leaf_ids.append(entry_id)

    def _choose_primary_parent(child_id: str) -> Optional[str]:
        candidates = parent_map.get(child_id)
        if not candidates:
            return None
        return min(candidates, key=lambda candidate: entry_order.get(candidate, float("inf")))

    leaf_agent_lineages: Dict[str, List[str]] = {}
    for leaf_id in leaf_ids:
        lineage: List[str] = []
        current_id: Optional[str] = leaf_id
        visited: Set[str] = set()
        while current_id and current_id not in visited and current_id in entries_by_id:
            visited.add(current_id)
            lineage.append(current_id)
            current_id = _choose_primary_parent(current_id)

        lineage.reverse()
        agent_ids: List[str] = []
        for entry_id in lineage:
            entry = entries_by_id.get(entry_id)
            agent_id = str(entry.get("agent_id", "unknown")) if entry is not None else "unknown"
            agent_ids.append(agent_id)
        if agent_ids:
            leaf_agent_lineages[leaf_id] = agent_ids

    return leaf_agent_lineages


def _compute_tree_balance_metrics(
    entries_by_id: Dict[str, Dict[str, Any]],
    entry_order: Dict[str, int],
    edges: Set[Tuple[str, str]],
) -> Dict[str, Any]:
    # 1. Build Primary Tree
    parent_map: Dict[str, List[str]] = defaultdict(list)
    for parent, child in edges:
        parent_map[child].append(parent)
    
    primary_children: Dict[str, List[str]] = defaultdict(list)
    primary_parents: Dict[str, str] = {}
    
    for child, parents in parent_map.items():
        # Choose primary parent (min index)
        primary = min(parents, key=lambda p: entry_order.get(p, float("inf")))
        primary_parents[child] = primary
        primary_children[primary].append(child)
        
    # Find roots
    all_ids = set(entries_by_id.keys())
    roots = [nid for nid in all_ids if nid not in primary_parents]
    
    return tree_metrics.compute_tree_metrics(roots, primary_children)


def _build_archive_graph(
    entries: Iterable[Dict[str, Any]],
    *,
    experiment_dir: Path,
    archive_dir: Path,
    output_format: str,
    mode: str,
    experiment_prefix: Optional[Path],
) -> Tuple[Digraph, Dict[str, List[str]], Dict[str, Any]]:
    entries = list(entries)
    graph = _init_graph(output_format)
    assigned_colors: Dict[str, str] = {}
    entries_by_id: Dict[str, Dict[str, Any]] = {}
    entry_order: Dict[str, int] = {}
    for index, entry in enumerate(entries):
        entry_id_raw = entry.get("id")
        if entry_id_raw is None:
            continue
        entry_id = str(entry_id_raw)
        entries_by_id[entry_id] = entry
        entry_order[entry_id] = index

    edges: Set[Tuple[str, str]] = set()

    for entry_id, entry in entries_by_id.items():
        parent_ids: Set[str] = set()
        agent_dir = _resolve_source_experiment_dir(
            entry,
            experiment_dir=experiment_dir,
            experiment_prefix=experiment_prefix,
        )
        if agent_dir:
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
    parent_map = defaultdict(list)
    for src, dst in edges:
        incoming[dst] += 1
        parent_map[dst].append(src)
        
    root_ids = {entry_id for entry_id in entries_by_id if incoming.get(entry_id, 0) == 0}

    # Color propagation
    primary_children = defaultdict(list)
    roots = []
    
    # Reconstruct primary tree for coloring
    for entry_id in entries_by_id:
        parents = parent_map.get(entry_id, [])
        if not parents:
            roots.append(entry_id)
        else:
            primary = min(parents, key=lambda p: entry_order.get(p, float("inf")))
            primary_children[primary].append(entry_id)
            
    roots.sort(key=lambda x: entry_order.get(x, 0))
    node_colors = {}
    queue = list(roots)
    
    for i, root in enumerate(roots):
        node_colors[root] = PALETTE[i % len(PALETTE)]
        
    idx = 0
    while idx < len(queue):
        curr = queue[idx]
        idx += 1
        
        color = node_colors.get(curr, "#dddddd")
        children = primary_children.get(curr, [])
        children.sort(key=lambda x: entry_order.get(x, 0))
        
        for child in children:
            if child not in node_colors:
                node_colors[child] = color
                queue.append(child)

    for entry_id, entry in entries_by_id.items():
        # agent_id = str(entry.get("agent_id", "unknown"))
        # fill_color = _choose_color(agent_id, PALETTE, assigned_colors)
        
        base_color = node_colors.get(entry_id, "#dddddd")
        bg_color = _lighten_color(base_color, 0.5)
        
        image_path = _resolve_image_path(entry, archive_dir, experiment_prefix)
        attrs = _build_archive_node_attributes(
            entry=entry,
            image_path=image_path,
            fill_color=base_color,
            bg_color=bg_color,
            is_root=entry_id in root_ids,
            mode=mode,
        )
        graph.node(entry_id, **attrs)

    for parent, child in sorted(edges):
        graph.edge(parent, child)

    leaf_agent_lineages = _compute_leaf_agent_lineages(
        entries_by_id=entries_by_id,
        entry_order=entry_order,
        edges=edges,
    )

    metrics = _compute_tree_balance_metrics(
        entries_by_id=entries_by_id,
        entry_order=entry_order,
        edges=edges,
    )

    return graph, leaf_agent_lineages, metrics


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


def build_phylogeny_graph(
    entries: Iterable[Dict[str, Any]],
    *,
    experiment_dir: Path,
    archive_dir: Path,
    output_format: str,
    mode: str,
    scope: str,
    experiment_prefix: Optional[Path],
) -> Tuple[Digraph, Dict[str, List[str]], Dict[str, Any]]:
    return _build_archive_graph(
        entries,
        experiment_dir=experiment_dir,
        archive_dir=archive_dir,
        output_format=output_format,
        mode=mode,
        experiment_prefix=experiment_prefix,
    )


def _resolve_output_path(
    cfg: ArchivePhylogenyConfig,
    archive_dir: Path,
    original_cwd: Path,
) -> Path:
    if cfg.output is None:
        output_path = archive_dir / f"archive_phylogeny.{cfg.format}"
    else:
        output_path = _ensure_absolute(Path(cfg.output), original_cwd)
    if output_path.suffix.lower() != f".{cfg.format}":
        output_path = output_path.with_suffix(f".{cfg.format}")
    return output_path.resolve()


@hydra.main(version_base="1.3", config_path=None, config_name="archive_phylogeny_base")
def main(cfg: ArchivePhylogenyConfig) -> None:
    original_cwd = Path(get_original_cwd())
    validated_cfg = ensure_valid_config(cfg, original_cwd=original_cwd)
    _validate_visualization_options(validated_cfg)

    experiment_dir = Path(validated_cfg.experiment_dir).resolve()
    archive_dir = experiment_dir / "archive"
    entries = _load_archive_entries(archive_dir)
    if validated_cfg.archive_limit is not None:
        entries = entries[: validated_cfg.archive_limit]
    if not entries:
        raise SystemExit("Archive metadata contains no entries to visualise.")

    graph, leaf_agent_lineages, metrics = build_phylogeny_graph(
        entries,
        experiment_dir=experiment_dir,
        archive_dir=archive_dir,
        output_format=validated_cfg.format,
        mode=validated_cfg.mode,
        scope=validated_cfg.scope,
        experiment_prefix=experiment_dir,
    )

    output_path = _resolve_output_path(validated_cfg, archive_dir, original_cwd)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rendered_path = Path(
        graph.render(
            filename=output_path.stem,
            directory=str(output_path.parent),
            cleanup=True,
        )
    )

    # graphviz.render adds the requested suffix automatically; ensure final path matches expectation.
    final_path = rendered_path.with_suffix(f".{validated_cfg.format}")
    if final_path != output_path and final_path.exists():
        final_path.rename(output_path)
        final_path = output_path

    print(f"Phylogenetic tree saved to {final_path}")

    leaf_lineage_path = archive_dir / "leaf_agent_lineages.json"
    leaf_lineage_path.write_text(json.dumps(leaf_agent_lineages, indent=2), encoding="utf-8")
    print(f"Saved leaf agent lineages to {leaf_lineage_path}")

    metrics_path = archive_dir / "phylogeny_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved phylogeny metrics to {metrics_path}")


if __name__ == "__main__":
    main()
