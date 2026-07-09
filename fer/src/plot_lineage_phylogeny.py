#!/usr/bin/env python3
"""
Reconstruct and render a simplified Picbreeder phylogeny using only the final image of each lineage.
This is much faster than the full phylogeny as it avoids reading intermediate genomes.
"""

from __future__ import annotations

import argparse
import html
import math
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import traceback
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image
from graphviz import Digraph
from tqdm.auto import tqdm

# Add root to path to allow importing tree_metrics
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from picbreeder_vlm.analysis.tree_metrics import compute_tree_metrics

from fer.src.save_lineage_figures import do_forward_pass, load_pbcppn
from fer.src.picbreeder_util import load_zip_xml_as_dict

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

@dataclass
class LineageNode:
    pid: str
    parent_pid: Optional[str]
    final_genome: Dict[str, Any]
    final_genome_key: Tuple[str, str]
    generation_count: int
    thumbnail_path: Optional[Path] = None

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

def _extract_genome_key(genome: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    identifier = genome.get("identifier") if isinstance(genome, Mapping) else None
    if not isinstance(identifier, Mapping):
        return None
    branch = identifier.get("@branch")
    gid = identifier.get("@id")
    if branch is None or gid is None:
        return None
    return str(branch), str(gid)

def _render_thumbnail_array(genome: Mapping[str, Any], size: int) -> Optional[np.ndarray]:
    if size <= 0:
        return None
    try:
        cppn = load_pbcppn(genome)
        forward = do_forward_pass(cppn, res=size)
        rgb = np.asarray(forward["rgb"], dtype=np.float32)
        rgb = np.clip(rgb, 0.0, 1.0)
        return (rgb * 255).astype(np.uint8)
    except Exception:
        return None

def _ensure_thumbnail(
    node: LineageNode,
    thumb_dir: Optional[Path],
    thumb_size: int,
) -> Optional[Path]:
    if thumb_dir is None or thumb_size <= 0:
        return None
    
    filename = f"{node.pid}_final.png"
    thumb_path = (thumb_dir / filename).resolve()
    
    if node.thumbnail_path is not None and node.thumbnail_path.exists():
        return node.thumbnail_path
    
    if thumb_path.exists():
        node.thumbnail_path = thumb_path
        return thumb_path
        
    thumb_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        rgb = _render_thumbnail_array(node.final_genome, thumb_size)
        if rgb is None:
            return None
        image = Image.fromarray(rgb, "RGB")
        image.save(thumb_path)
        node.thumbnail_path = thumb_path
        return thumb_path
    except Exception as exc:
        tqdm.write(f"[WARN] Failed to render thumbnail for pid {node.pid}: {exc}")
        return None

def _load_lineage_info(pb_dir: Path, pid: str) -> Optional[LineageNode]:
    """
    Reads main.zip to find parent PID and location of final genome.
    Then reads the final genome.
    """
    main_zip = pb_dir / pid / "main.zip"
    if not main_zip.exists():
        print(f"[WARN] main.zip not found for pid {pid}")
        return None
        
    try:
        main_data = load_zip_xml_as_dict(str(main_zip))
    except Exception as exc:
        tqdm.write(f"[WARN] Failed to read main.zip for {pid}: {exc}")
        return None

    genome_meta = main_data.get("genome", {})
    series = genome_meta.get("series", {})
    # Fallback: sometimes info is directly in genome, not under series
    if not series and "generation" in genome_meta:
        series = genome_meta
    
    # Parent PID
    parent_pid = None
    branch_from = series.get("branchFrom")
    if isinstance(branch_from, dict):
        parent_pid = branch_from.get("@branch")
    
    # Find last generation
    generations = series.get("generation")
    if not generations:
        print(f"[WARN] No generations found for pid {pid}")
        return None
    generations = _ensure_list(generations)
    
    if not generations:
        print(f"[WARN] No generations found for pid {pid} after ensuring list")
        return None
        
    last_gen_meta = max(generations, key=lambda g: _safe_int(g.get("@number", -1)))
    storage_id = last_gen_meta.get("@storage")
    last_gen_num = _safe_int(last_gen_meta.get("@number"))
    
    if storage_id is None:
        print(f"[WARN] No storage ID found for last generation of pid {pid}")
        return None
        
    # Read storage zip
    storage_zip = pb_dir / pid / f"{storage_id}.zip"
    if not storage_zip.exists():
        # Fallback: maybe the storage id is an index into a file list?
        # But usually in PB it matches the filename.
        # Let's try to match blindly if 1.zip etc exists.
        print(f"[WARN] Storage zip {storage_id}.zip not found for pid {pid}")
        return None
        
    try:
        storage_data = load_zip_xml_as_dict(str(storage_zip))
    except Exception:
        traceback.print_exc()
        return None
        
    genome_data = storage_data.get("genome", {})
    if "storage" not in genome_data:
        storage_gens = genome_data["generation"]
    else:
        storage_gens = genome_data["storage"]["generation"]
    storage_gens = _ensure_list(storage_gens)
    
    # Find the specific generation in storage
    # It might be a list or a single dict
    target_gen = None
    for g in storage_gens:
        if _safe_int(g.get("@number")) == last_gen_num:
            target_gen = g
            break
            
    if target_gen is None:
        # Fallback to the last one in the file if we can't match number?
        # Or maybe the file only contains what we need.
        if storage_gens:
            target_gen = storage_gens[-1]
            
    if target_gen is None:
        breakpoint()
        print(f"[WARN] Target generation {last_gen_num} not found in storage for pid {pid}")
        return None
        
    # Extract genome
    # It can be 'population' (list of genomes) or 'genome' (single)
    final_genome = None
    if "population" in target_gen:
        # If population, pick the last one or best?
        # Usually user-guided implies one "chosen" one.
        # We'll take the last one in the list.
        pop = target_gen["population"]
        if isinstance(pop, list):
            final_genome = pop[-1]
        elif isinstance(pop, dict):
             # check if it has nodes/links
             if "nodes" in pop:
                 final_genome = pop
             else:
                 # maybe it's a wrapper?
                 pass
    elif "genome" in target_gen:
        g_cand = target_gen["genome"]
        if isinstance(g_cand, list):
            final_genome = g_cand[-1]
        else:
            final_genome = g_cand
            
    if final_genome is None:
        # Check if target_gen itself is a genome (sometimes happens?)
        if "nodes" in target_gen:
            final_genome = target_gen
            
    if final_genome is None:
        print(f"[WARN] Final genome not found for pid {pid} in generation {last_gen_num}")
        return None
        
    key = _extract_genome_key(final_genome)
    if key is None:
        # Fallback key
        key = (pid, f"gen{last_gen_num}")
        
    return LineageNode(
        pid=pid,
        parent_pid=parent_pid,
        final_genome=final_genome,
        final_genome_key=key,
        generation_count=last_gen_num,
    )

def _build_node_label(
    node: LineageNode,
    thumbnail_path: Optional[Path],
    fill_color: str,
) -> str:
    image_row = ""
    if thumbnail_path is not None:
        image_row = (
            f"<TR><TD><IMG SRC=\"{html.escape(str(thumbnail_path))}\" SCALE='TRUE'/></TD></TR>"
        )
    
    key_str = f"{node.final_genome_key[0]}:{node.final_genome_key[1]}"
    
    label = (
        "<"
        f"<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='2' BGCOLOR='{fill_color}'>"
        f"{image_row}"
        f"<TR><TD ALIGN='LEFT'><FONT POINT-SIZE='10'><B>PID {html.escape(node.pid)}</B></FONT></TD></TR>"
        f"<TR><TD ALIGN='LEFT'><FONT POINT-SIZE='9'>Final: {html.escape(key_str)}</FONT></TD></TR>"
        f"<TR><TD ALIGN='LEFT'><FONT POINT-SIZE='9'>Gens: {node.generation_count}</FONT></TD></TR>"
        "</TABLE>"
        ">"
    )
    return label

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

def _compute_tree_balance_metrics(nodes: Dict[str, LineageNode]) -> Dict[str, Any]:
    if compute_tree_metrics is None:
        return {}
        
    # Build tree structure
    children: Dict[str, List[str]] = defaultdict(list)
    parents: Dict[str, str] = {}
    
    for pid, node in nodes.items():
        if node.parent_pid and node.parent_pid in nodes:
            parents[pid] = node.parent_pid
            children[node.parent_pid].append(pid)
            
    # Find roots
    roots = [pid for pid in nodes if pid not in parents]
    
    return compute_tree_metrics(roots, children)

def build_graph(
    nodes: Dict[str, LineageNode],
    output_format: str,
    rankdir: str,
    thumb_dir: Optional[Path],
    thumb_size: int,
) -> Digraph:
    graph = Digraph("picbreeder_lineage_phylogeny", format=output_format)
    graph.attr(rankdir=rankdir, bgcolor="white", nodesep="0.2", ranksep="0.8")
    graph.attr("node", fontname="Helvetica", fontsize="9", shape="box", style="rounded")
    graph.attr("edge", color="#4a4a4a")
    
    # Assign colors to roots
    roots = [pid for pid, node in nodes.items() if not node.parent_pid or node.parent_pid not in nodes]
    root_colors = {pid: PALETTE[i % len(PALETTE)] for i, pid in enumerate(sorted(roots))}
    
    # Propagate colors
    node_colors = {}
    queue = list(roots)
    for r in roots:
        node_colors[r] = root_colors[r]
        
    # Simple BFS for coloring
    # Build children map
    children = defaultdict(list)
    for pid, node in nodes.items():
        if node.parent_pid and node.parent_pid in nodes:
            children[node.parent_pid].append(pid)
            
    processed = set(roots)
    while queue:
        curr = queue.pop(0)
        color = node_colors.get(curr, "#dddddd")
        for child in children[curr]:
            if child not in processed:
                node_colors[child] = color
                processed.add(child)
                queue.append(child)

    for pid in sorted(nodes.keys()):
        node = nodes[pid]
        color = node_colors.get(pid, "#dddddd")
        fill = _lighten_color(color, 0.5)
        
        thumb_path = _ensure_thumbnail(node, thumb_dir, thumb_size)
        label = _build_node_label(node, thumb_path, fill)
        
        attrs = {
            "label": label,
            "shape": "plaintext",
            "color": color,
        }
        graph.node(pid, **attrs)
        
        if node.parent_pid and node.parent_pid in nodes:
            graph.edge(node.parent_pid, pid)
            
    return graph

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a simplified Picbreeder lineage phylogeny (final images only)."
    )
    parser.add_argument(
        "--pb-dir",
        type=Path,
        default=Path("fer/spaghetti/pbRender/genomeAll"),
        help="Directory that contains pid subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("human_lineages/lineages/lineage_phylogeny"),
        help="Output path without suffix.",
    )
    parser.add_argument(
        "--format",
        choices=VALID_FORMATS,
        default="pdf",
        help="Graphviz render format.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of PIDs to scan.",
    )
    parser.add_argument(
        "--rankdir",
        choices=("LR", "TB"),
        default="TB",
        help="Graph orientation.",
    )
    parser.add_argument(
        "--thumb-size",
        type=int,
        default=64,
        help="Thumbnail size.",
    )
    parser.add_argument(
        "--thumb-dir",
        type=Path,
        default=None,
        help="Optional thumbnail cache directory.",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[500, 750, 1000, 1500, 2000, 10000],
        help="Checkpoints (number of lineages) to compute metrics for.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip rendering the graph to file.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    pb_dir = args.pb_dir.expanduser().resolve()
    output_base = args.output.expanduser().resolve()
    
    if args.thumb_size > 0:
        if args.thumb_dir is None:
            thumb_dir = output_base.parent / f"{output_base.name}_thumbs"
        else:
            thumb_dir = args.thumb_dir
        thumb_dir = thumb_dir.expanduser().resolve()
    else:
        thumb_dir = None
        
    if not pb_dir.is_dir():
        raise FileNotFoundError(f"Picbreeder directory not found: {pb_dir}")
        
    pids = sorted(
        pid.name
        for pid in pb_dir.iterdir()
        if pid.is_dir() and (pid / "main.zip").exists()
    )
    
    if args.limit:
        pids = pids[:args.limit]
        
    nodes: Dict[str, LineageNode] = {}
    print(f"Scanning {len(pids)} lineages...")
    
    for pid in tqdm(pids):
        node = _load_lineage_info(pb_dir, pid)
        if node:
            nodes[pid] = node
            
    print(f"Found {len(nodes)} valid lineages.")
    
    # Sort PIDs numerically if possible for consistent checkpointing
    sorted_pids = sorted(nodes.keys(), key=int)

    all_metrics = {}
    for limit in args.checkpoints:
        if limit > len(sorted_pids):
            # If we have fewer lineages than the checkpoint, we can skip or just take max.
            # But usually checkpoints are strictly defined. We'll skip if not enough data.
            # Or should we just take what we have if it's the last one? 
            # Let's strictly skip to avoid misleading '10000' label for 500 nodes.
            if limit > len(sorted_pids) + 100: # allow some slack? No, strict is better.
                continue
        
        subset_pids = sorted_pids[:limit]
        subset_nodes = {pid: nodes[pid] for pid in subset_pids}
        
        # When taking a subset, some nodes might point to parents NOT in the subset.
        # _compute_tree_balance_metrics handles "parent not in nodes" by treating as root.
        # This is correct for a "snapshot" of the archive.
        
        m = _compute_tree_balance_metrics(subset_nodes)
        all_metrics[str(limit)] = m
        print(f"Computed metrics for first {len(subset_nodes)} lineages.")

        if not args.no_render:
            # Render partial tree
            print(f"Rendering partial tree for {limit} lineages...")
            graph = build_graph(
               subset_nodes,
               args.format,
               args.rankdir,
               thumb_dir,
               args.thumb_size
            )
            
            checkpoint_name = f"{output_base.name}_ckpt{limit}"
            
            rendered = Path(
               graph.render(
                   filename=checkpoint_name,
                   directory=str(output_base.parent),
                   cleanup=True,
               )
            )
            print(f"Saved partial lineage phylogeny to {rendered}")

    metrics_path = output_base.parent / f"{output_base.name}_metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"Saved phylogeny metrics to {metrics_path}")

    if not args.no_render:
        graph = build_graph(
            nodes,
            args.format,
            args.rankdir,
            thumb_dir,
            args.thumb_size
        )
        
        output_base.parent.mkdir(parents=True, exist_ok=True)
        rendered = Path(
            graph.render(
                filename=output_base.name,
                directory=str(output_base.parent),
                cleanup=True,
            )
        )
        print(f"Saved lineage phylogeny to {rendered}")

if __name__ == "__main__":
    main()
