#!/usr/bin/env python3
"""
Visualize the ancestry of a specific image from the archive.

Usage:
    python visualize_image_ancestry.py /path/to/archive/images/entry_123.png
"""

import argparse
import json
import sys
import html
from pathlib import Path
from typing import Dict, Any, Set, List, Optional, Tuple
from collections import deque, defaultdict

try:
    from graphviz import Digraph
except ImportError:
    print("Error: graphviz is not installed. Please install it with `pip install graphviz`.")
    sys.exit(1)

# Add current directory to sys.path to ensure imports work
sys.path.append(str(Path(__file__).parent.resolve()))

from utils import _resolve_source_experiment_dir, _resolve_image_path

def find_archive_root(image_path: Path) -> Path:
    """Find the directory containing archive_metadata.json by searching upwards."""
    current = image_path.parent
    # Stop at root or if we hit a permission error (handled by loop condition effectively)
    while len(current.parts) > 1: 
        if (current / "archive_metadata.json").exists():
            return current
        if current.parent == current: # Root
            break
        current = current.parent
    raise FileNotFoundError("Could not find archive_metadata.json in any parent directory of the provided image path.")

def load_archive_entries(archive_dir: Path) -> Dict[str, Dict[str, Any]]:
    metadata_path = archive_dir / "archive_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    entries_by_id = {}
    for entry in data.get("entries", []):
        entry_id = str(entry.get("id"))
        entries_by_id[entry_id] = entry
    return entries_by_id

def identify_target_entry(target_image_path: Path, entries_by_id: Dict[str, Dict[str, Any]], archive_dir: Path) -> Optional[str]:
    """Find the entry ID that corresponds to the given image path."""
    target_absolute = target_image_path.resolve()
    
    # First pass: check for exact path match
    for entry_id, entry in entries_by_id.items():
        # Using _resolve_image_path from utils if possible, or manual resolution
        # entry['image_path'] is usually relative to archive root or absolute
        
        # We simulate what _resolve_image_path does or just construct it
        # In archive_manager, image_path is stored. 
        # Usually it's like "images/entry_X.png" relative to archive_dir
        
        entry_img_path = Path(entry.get("image_path"))
        if not entry_img_path.is_absolute():
            entry_img_path = archive_dir / entry_img_path
        
        if entry_img_path.resolve() == target_absolute:
            return entry_id

    # Second pass: check by filename (if unique)
    target_name = target_image_path.name
    candidates = []
    for entry_id, entry in entries_by_id.items():
        entry_img_path = Path(entry.get("image_path"))
        if entry_img_path.name == target_name:
            candidates.append(entry_id)
            
    if len(candidates) == 1:
        return candidates[0]
    
    return None

def _load_selected_entry_ids(branch_path: Path) -> List[str]:
    """Load parent IDs from branching_selection.json."""
    if not branch_path.exists():
        return []
    try:
        data = json.loads(branch_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    selected = data.get("selected_entry_ids") or []
    result: List[str] = []
    for value in selected:
        if value is not None:
            result.append(str(value))
    return result

def get_parents(entry_id: str, entries_by_id: Dict[str, Dict[str, Any]], experiment_dir: Path) -> Set[str]:
    """Identify parents of a given entry."""
    entry = entries_by_id.get(entry_id)
    if not entry:
        return set()
        
    parent_ids: Set[str] = set()
    
    # Attempt to resolve agent directory to check branching selection
    # experiment_dir is the parent of archive_dir
    
    agent_dir = _resolve_source_experiment_dir(
        entry,
        experiment_dir=experiment_dir,
        experiment_prefix=experiment_dir, 
    )
    
    if agent_dir:
        branch_path = agent_dir / "logs" / "branching_selection.json"
        for parent in _load_selected_entry_ids(branch_path):
            if parent != entry_id and parent in entries_by_id:
                parent_ids.add(parent)
                
    # Fallback/Addition: check source_entry_ids in metadata
    if not parent_ids: # Logic in original script implies this is fallback, or maybe additive? 
        # Original: if not parent_ids: check source_entry_ids
        # But let's check source_entry_ids anyway if branching logic failed or was empty?
        # Actually, looking at visualize_archive_phylogeny.py:
        # if not parent_ids: for value in entry.get("source_entry_ids")...
        # So it prefers branching_selection if available and non-empty.
        
        for value in entry.get("source_entry_ids", []) or []:
            parent = str(value)
            if parent != entry_id and parent in entries_by_id:
                parent_ids.add(parent)
                
    return parent_ids

def build_ancestry_subgraph(target_id: str, entries_by_id: Dict[str, Dict[str, Any]], experiment_dir: Path) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    """Trace back ancestry from target_id to roots."""
    nodes = set()
    edges = set()
    queue = deque([target_id])
    
    while queue:
        current_id = queue.popleft()
        if current_id in nodes:
            continue
        nodes.add(current_id)
        
        parents = get_parents(current_id, entries_by_id, experiment_dir)
        for parent in parents:
            edges.add((parent, current_id))
            if parent not in nodes:
                queue.append(parent)
                
    return nodes, edges

def truncate_text(value: str, max_len: int = 40) -> str:
    value = value.strip()
    if max_len > 0 and len(value) > max_len:
        return value[: max_len - 1] + "…"
    return value

def render_ancestry(
    nodes: Set[str], 
    edges: Set[Tuple[str, str]], 
    target_id: str, 
    entries_by_id: Dict[str, Any], 
    archive_dir: Path,
    output_path: Path,
    format: str = "pdf"
):
    dot = Digraph("ancestry", format=format)
    # Use neato for absolute positioning
    dot.attr(layout="neato")
    dot.attr(bgcolor="white")
    dot.attr("node", fontname="Helvetica", fontsize="10", color="#333333")
    
    # Trace back to roots to establish order
    # Build adjacency
    parent_map = {}
    for parent, child in edges:
        parent_map[child] = parent
        
    chain = []
    curr = target_id
    while curr in nodes:
        chain.append(curr)
        if curr in parent_map:
            curr = parent_map[curr]
        else:
            break
    chain.reverse()
    
    # Layout configuration (snake)
    COLS = 99
    X_SPACING = 2.0  # inches
    Y_SPACING = 2.5  # inches
    
    # Sort nodes to ensure deterministic output
    for i, entry_id in enumerate(chain):
        entry = entries_by_id[entry_id]
        
        # Determine styling
        is_target = (entry_id == target_id)
        
        # Snake position calculation
        row = i // COLS
        col_idx = i % COLS
        
        if row % 2 == 1:
            col = COLS - 1 - col_idx
        else:
            col = col_idx
            
        x = col * X_SPACING
        y = -row * Y_SPACING
        pos_str = f"{x},{y}!"
        
        # Colors
        bg_color = "white"
        
        title = truncate_text(str(entry.get("title") or "untitled"))
        agent_id = str(entry.get("agent_id", "unknown"))
        generation = int(entry.get("generation") or 0) # + 1 # zero-based to one-based ?
        agent_line = f"{agent_id} | gen {generation}"
        
        image_filename = Path(entry.get("image_path")).name
        image_path = archive_dir / "images" / image_filename
        print(image_path)
            
        image_tag = ""
        if image_path.exists():
            image_tag = f'<TR><TD><IMG SRC="{html.escape(str(image_path))}" SCALE="TRUE"/></TD></TR>'
            
        label = (
            "<"
            f"<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='2' BGCOLOR='{bg_color}'>"
            f"{image_tag}"
            f"<TR><TD ALIGN='LEFT'><FONT POINT-SIZE='10'><B>{html.escape(title)}</B></FONT></TD></TR>"
            f"<TR><TD ALIGN='LEFT'><FONT POINT-SIZE='9'>{html.escape(agent_line)}</FONT></TD></TR>"
            "</TABLE>"
            ">"
        )
        
        attrs = {
            "shape": "plaintext",
            "label": label,
            "style": "filled",
            "fillcolor": bg_color,
            "pos": pos_str,
        }
        
        if is_target:
            attrs["penwidth"] = "3"
            attrs["color"] = "red"
            
        dot.node(entry_id, **attrs)
        
    for parent, child in sorted(list(edges)):
        dot.edge(parent, child)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = dot.render(filename=output_path.stem, directory=str(output_path.parent), cleanup=True)
    print(f"Ancestry visualization saved to: {rendered}")


def main():
    parser = argparse.ArgumentParser(description="Render ancestry of a specific image in the archive.")
    parser.add_argument("image_path", type=str, help="Path to the image file in the archive")
    parser.add_argument("--output", type=str, default=None, help="Output path for the rendering (default: ancestry_<id>.pdf)")
    parser.add_argument("--format", type=str, default="png", choices=["pdf", "png", "svg"], help="Output format")
    
    args = parser.parse_args()
    
    image_path = Path(args.image_path).resolve()
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
        
    try:
        archive_dir = find_archive_root(image_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    print(f"Archive root found at: {archive_dir}")
    experiment_dir = archive_dir.parent
    
    entries_by_id = load_archive_entries(archive_dir)
    target_id = identify_target_entry(image_path, entries_by_id, archive_dir)
    
    if not target_id:
        print(f"Error: Could not find an archive entry matching image {image_path}")
        sys.exit(1)
        
    print(f"Found entry ID: {target_id}")
    
    nodes, edges = build_ancestry_subgraph(target_id, entries_by_id, experiment_dir)
    print(f"Ancestry graph has {len(nodes)} nodes and {len(edges)} edges.")
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path.cwd() / f"ancestry_{target_id}.{args.format}"
        
    render_ancestry(nodes, edges, target_id, entries_by_id, archive_dir, output_path, args.format)

if __name__ == "__main__":
    main()
