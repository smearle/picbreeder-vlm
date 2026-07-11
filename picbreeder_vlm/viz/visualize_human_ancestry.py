#!/usr/bin/env python3
"""
Visualize the ancestry of a specific image/lineage from the human Picbreeder archive.
"""

import argparse
import sys
import html
from pathlib import Path
from typing import Dict, Any, Set, List, Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass
import numpy as np
from PIL import Image

try:
    from graphviz import Digraph
except ImportError:
    print("Error: graphviz is not installed. Please install it with `pip install graphviz`.")
    sys.exit(1)

# Add current directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from picbreeder_vlm._paths import FER_ROOT, ensure_fer_importable
ensure_fer_importable()
try:
    from fer.src.picbreeder_util import load_zip_xml_as_dict
    from fer.src.save_lineage_figures import do_forward_pass, load_pbcppn
except ImportError as e:
    print(f"Error importing helper modules: {e}")
    print("Ensure you are running this script from the project root.")
    sys.exit(1)

@dataclass
class HumanLineageNode:
    pid: str
    parent_pid: Optional[str]
    title: str
    generation_count: int
    age: int
    final_genome: Dict[str, Any]
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

def extract_pid_from_path(path: Path) -> Optional[str]:
    """
    Extracts PID from a path like .../genomeAll/12345/something or .../genomeAll/12345
    """
    # If path points to a file, take parent
    if path.is_file():
        path = path.parent
    
    # Check if the name is a number (PID)
    if path.name.isdigit():
        return path.name
    
    # Walk up until we find a digit directory that has main.zip
    current = path
    while len(current.parts) > 1:
        if current.name.isdigit() and (current / "main.zip").exists():
            return current.name
        current = current.parent
        
    return None

def find_pb_root(start_path: Path) -> Path:
    """
    Finds the root directory containing PIDs (e.g. genomeAll).
    """
    path = start_path.resolve()
    if path.is_file():
        path = path.parent
        
    # If current is PID, parent is root
    if path.name.isdigit() and (path / "main.zip").exists():
        return path.parent
        
    # Walk up
    current = path
    while len(current.parts) > 1:
        if (current / "1000").exists(): # Check for a known folder existence or just heuristics
             # Better: check if it contains numbered folders
             for child in current.iterdir():
                 if child.is_dir() and child.name.isdigit():
                     return current
        current = current.parent
        
    raise FileNotFoundError("Could not locate Picbreeder archive root (genomeAll).")

def load_lineage_node(pb_dir: Path, pid: str) -> Optional[HumanLineageNode]:
    """
    Loads lineage info for a single PID.
    Reused logic from plot_lineage_phylogeny.py but simplified.
    """
    lineage_dir = pb_dir / pid
    main_zip = lineage_dir / "main.zip"
    if not main_zip.exists():
        return None
        
    try:
        main_data = load_zip_xml_as_dict(str(main_zip))
    except Exception:
        return None

    genome_meta = main_data.get("genome", {})
    series = genome_meta.get("series", {})
    # Fallback: sometimes info is directly in genome, not under series
    if not series and "generation" in genome_meta:
        series = genome_meta

    header = genome_meta.get("header", {})
    title = header.get("name", "Untitled")
    
    # Parent PID
    parent_pid = None
    branch_from = series.get("branchFrom")
    if isinstance(branch_from, dict):
        parent_pid = branch_from.get("@branch")
    
    # Get final genome for rendering
    # Logic adapted from plot_lineage_phylogeny.py
    generations = series.get("generation")
    if not generations:
        return None
    generations = _ensure_list(generations)
    if not generations:
        return None
        
    last_gen_meta = max(generations, key=lambda g: _safe_int(g.get("@number", -1)))
    storage_id = last_gen_meta.get("@storage")
    last_gen_num = _safe_int(last_gen_meta.get("@number"))
    
    final_genome = None
    
    if storage_id:
        storage_zip = lineage_dir / f"{storage_id}.zip"
        if storage_zip.exists():
            try:
                storage_data = load_zip_xml_as_dict(str(storage_zip))
                storage_gens = storage_data.get("genome", {}).get("storage", {}).get("generation")
                storage_gens = _ensure_list(storage_gens)
                
                target_gen = None
                for g in storage_gens:
                    if _safe_int(g.get("@number")) == last_gen_num:
                        target_gen = g
                        break
                if target_gen is None and storage_gens:
                    target_gen = storage_gens[-1]
                    
                if target_gen:
                    if "population" in target_gen:
                        pop = target_gen["population"]
                        if isinstance(pop, list):
                            final_genome = pop[-1]
                        elif isinstance(pop, dict) and "nodes" in pop:
                             final_genome = pop
                    elif "genome" in target_gen:
                        g_cand = target_gen["genome"]
                        if isinstance(g_cand, list):
                            final_genome = g_cand[-1]
                        else:
                            final_genome = g_cand
                    elif "nodes" in target_gen:
                        final_genome = target_gen
            except Exception:
                pass
                
    age = 0
    if final_genome:
        age = _safe_int(final_genome.get("@age", 0))

    return HumanLineageNode(
        pid=pid,
        parent_pid=str(parent_pid) if parent_pid else None,
        title=title,
        generation_count=last_gen_num,
        age=age,
        final_genome=final_genome,
    )

def render_thumbnail(node: HumanLineageNode, thumb_dir: Path, pre_rendered_dir: Optional[Path] = None, size: int = 128) -> Optional[Path]:
    pre_rendered_thumb_path = pre_rendered_dir / f"{node.pid}.png"
    if not node.final_genome and not (pre_rendered_dir and (pre_rendered_dir / f"{node.pid}.png").exists()):
        print(f"Warning: No genome to render thumbnail for PID {node.pid} at {pre_rendered_thumb_path}")
        return None
        
    # Check pre-rendered first
    if pre_rendered_dir:
        pre_path = pre_rendered_dir / f"{node.pid}.png"
        if pre_path.exists():
            return pre_path

    thumb_path = thumb_dir / f"{node.pid}.png"
    if thumb_path.exists():
        return thumb_path
        
    if not node.final_genome:
        return None

    try:
        cppn = load_pbcppn(node.final_genome)
        forward = do_forward_pass(cppn, res=size)
        rgb = np.asarray(forward["rgb"], dtype=np.float32)
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        
        image = Image.fromarray(rgb_uint8, "RGB")
        image.save(thumb_path)
        return thumb_path
    except Exception as e:
        print(f"Warning: Failed to render {node.pid}: {e}")
        return None

def trace_ancestry(start_pid: str, pb_dir: Path) -> Dict[str, HumanLineageNode]:
    nodes = {}
    current_pid = start_pid
    
    print(f"Tracing ancestry for PID {start_pid}...")
    
    while current_pid:
        if current_pid in nodes:
            break
            
        node = load_lineage_node(pb_dir, current_pid)
        if not node:
            print(f"Warning: Could not load info for PID {current_pid}. Creating placeholder.")
            nodes[current_pid] = HumanLineageNode(
                pid=current_pid,
                parent_pid=None,
                title="Unknown",
                generation_count=0,
                age=0,
                final_genome={}
            )
            break
            
        nodes[current_pid] = node
        
        if not node.parent_pid:
            break
             
        current_pid = node.parent_pid
        
    return nodes

def render_graph(
    nodes: Dict[str, HumanLineageNode],
    target_pid: str,
    output_path: Path,
    thumb_dir: Path,
    pre_rendered_dir: Optional[Path] = None,
    format: str = "pdf"
):
    dot = Digraph("human_ancestry", format=format)
    # Use neato layout engine for manual positioning
    dot.attr(layout="neato")
    dot.attr(bgcolor="white")
    dot.attr("node", fontname="Helvetica", fontsize="10", color="#333333")
    
    thumb_dir.mkdir(parents=True, exist_ok=True)
    
    # Reconstruct chain order: Root -> Target
    chain = []
    curr = target_pid
    while curr and curr in nodes:
        chain.append(curr)
        curr = nodes[curr].parent_pid
    chain.reverse()
    
    # Layout configuration
    COLS = 99
    X_SPACING = 2.0  # inches
    Y_SPACING = 2.5  # inches
    
    for i, pid in enumerate(chain):
        node = nodes[pid]
        is_target = (pid == target_pid)
        
        # Calculate snake position
        row = i // COLS
        col_idx = i % COLS
        
        if row % 2 == 0:
            # Odd rows go right-to-left
            col = col_idx
        else:
            # Even rows go left-to-right
            col = COLS - 1 - col_idx
            
        # Pos is in inches (approx), origin at bottom-left usually, but neato can handle negative
        # To make it flow down, we use negative Y
        x = col * X_SPACING
        y = row * Y_SPACING
        pos_str = f"{x},{y}!"
        
        base_color = "white"
            
        thumb_path = render_thumbnail(node, thumb_dir, pre_rendered_dir=pre_rendered_dir, size=128)
        
        image_tag = ""
        if thumb_path:
            image_tag = f'<TR><TD><IMG SRC="{html.escape(str(thumb_path))}" SCALE="TRUE"/></TD></TR>'
            
        gen_info = f"Gens: {node.generation_count}"
            
        label = (
            "<"
            f"<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='2' BGCOLOR='{base_color}'>"
            f"{image_tag}"
            f"<TR><TD ALIGN='LEFT'><FONT POINT-SIZE='10'><B>PID {pid}</B></FONT></TD></TR>"
            f"<TR><TD ALIGN='LEFT'><FONT POINT-SIZE='9'>{html.escape(gen_info)}</FONT></TD></TR>"
            "</TABLE>"
            ">"
        )
        
        attrs = {
            "shape": "plaintext",
            "label": label,
            "style": "filled",
            "fillcolor": base_color,
            "pos": pos_str,
        }
        
        if is_target:
            attrs["penwidth"] = "3"
            attrs["color"] = "red"
            
        dot.node(pid, **attrs)
        
        if node.parent_pid and node.parent_pid in nodes:
            dot.edge(node.parent_pid, pid)
            
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = dot.render(filename=output_path.stem, directory=str(output_path.parent), cleanup=True)
    print(f"Ancestry visualization saved to: {rendered}")

def main():
    parser = argparse.ArgumentParser(description="Render ancestry of a specific image/lineage from the human Picbreeder archive.")
    parser.add_argument("input_path", type=str, help="Path to an image file or directory within the archive, or a PID if --root is provided.")
    parser.add_argument("--root", type=str, default=None, help="Path to the genomeAll directory (optional if input_path implies it).")
    parser.add_argument("--output", type=str, default=None, help="Output path for the rendering.")
    parser.add_argument("--format", type=str, default="png", choices=["pdf", "png", "svg"], help="Output format")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    # Determine PID and Root
    if args.root:
        pb_dir = Path(args.root).resolve()
        # If input is just a number, treat as PID
        if input_path.name.isdigit() and len(input_path.parts) == 1:
            pid = input_path.name
        else:
            # Maybe full path provided
            pid = extract_pid_from_path(input_path)
    else:
        # Try to infer everything from input_path
        input_abs = input_path.resolve()
        pid = extract_pid_from_path(input_abs)
        try:
            pb_dir = find_pb_root(input_abs)
        except FileNotFoundError:
            # If input_path is just a number, we can't infer root
             print("Error: Could not infer archive root from input path. Please provide --root.")
             sys.exit(1)
             
    if not pid:
        print(f"Error: Could not determine PID from {input_path}")
        sys.exit(1)
        
    print(f"Target PID: {pid}")
    print(f"Archive Root: {pb_dir}")
    
    if not (pb_dir / pid).exists():
        print(f"Error: PID {pid} not found in {pb_dir}")
        sys.exit(1)
        
    nodes = trace_ancestry(pid, pb_dir)
    print(f"Found {len(nodes)} ancestors.")
    
    print("\nLineage Ages:")
    # Print from root to target
    chain = []
    curr = pid
    while curr and curr in nodes:
        chain.append(curr)
        curr = nodes[curr].parent_pid
        
    for p in reversed(chain):
        n = nodes[p]
        age_str = str(n.age) if n.final_genome else "?"
        print(f"PID {p}: age {age_str}")
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path.cwd() / f"human_ancestry_{pid}.{args.format}"
        
    thumb_dir = pb_dir.parent / "thumbs_cache" # Reuse a cache dir
    
    # Check for pre-rendered images in standard location
    pre_rendered_dir = FER_ROOT / "src/archive_res-128/images"
    if not pre_rendered_dir.exists():
        pre_rendered_dir = None
        print(f"Notice: pre-rendered directory {FER_ROOT / 'src/archive_res-128/images'} not found. Will render on the fly.")
    else:
        print(f"Using pre-rendered images from {pre_rendered_dir}")

    render_graph(nodes, pid, output_path, thumb_dir, pre_rendered_dir=pre_rendered_dir, format=args.format)

if __name__ == "__main__":
    main()
