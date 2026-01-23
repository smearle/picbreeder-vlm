#!/usr/bin/env python3
"""
Inspect lineages with 0 generations.
Check if they are duplicates of their parents.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm.auto import tqdm

# Add root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from fer.src.picbreeder_util import load_zip_xml_as_dict

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

def _get_lineage_info(pb_dir: Path, pid: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Returns (generation_count, parent_pid)
    """
    main_zip = pb_dir / pid / "main.zip"
    if not main_zip.exists():
        return None, None
        
    try:
        main_data = load_zip_xml_as_dict(str(main_zip))
    except Exception:
        return None, None

    genome_meta = main_data.get("genome", {})
    series = genome_meta.get("series", {})
    
    # Parent PID
    parent_pid = None
    branch_from = series.get("branchFrom")
    if isinstance(branch_from, dict):
        parent_pid = branch_from.get("@branch")
    
    # Find last generation
    generations = series.get("generation")
    if not generations:
        return 0, parent_pid # If no generation tag, assume 0? Or maybe 1 if it's the start? 
                             # Actually usually there is at least one generation entry.
                             # If missing, it's weird.
    
    generations = _ensure_list(generations)
    
    if not generations:
        return 0, parent_pid
        
    last_gen_meta = max(generations, key=lambda g: _safe_int(g.get("@number", -1)))
    last_gen_num = _safe_int(last_gen_meta.get("@number"))
    
    return last_gen_num, parent_pid

def main() -> None:
    pb_dir = Path("fer/spaghetti/pbRender/genomeAll").expanduser().resolve()
    # image_dir = Path("fer/src/archive_res-128").expanduser().resolve()
    # Use res-224 for better quality if available, else 128
    image_dir = Path("fer/src/archive_res-224").expanduser().resolve()
    if not image_dir.exists():
        image_dir = Path("fer/src/archive_res-128").expanduser().resolve()
        
    output_dir = Path("figures/zero_gen_examples").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not pb_dir.is_dir():
        print(f"Error: {pb_dir} not found")
        return

    pids = sorted(
        pid.name
        for pid in pb_dir.iterdir()
        if pid.is_dir() and (pid / "main.zip").exists()
    )
    
    print(f"Scanning {len(pids)} lineages...")
    
    zero_gen_pids = []
    zero_gen_with_parent = []
    
    for pid in tqdm(pids):
        gen_count, parent_pid = _get_lineage_info(pb_dir, pid)
        if gen_count == 0:
            zero_gen_pids.append(pid)
            if parent_pid:
                zero_gen_with_parent.append((pid, parent_pid))
                
    print(f"Found {len(zero_gen_pids)} lineages with 0 generations.")
    print(f"Of these, {len(zero_gen_with_parent)} have a parent (are branches).")
    print(f"The remaining {len(zero_gen_pids) - len(zero_gen_with_parent)} are likely roots (random starts).")
    
    # Save examples
    # We want to show: Parent Image -> Child Image (should be identical)
    
    examples_saved = 0
    max_examples = 20
    
    print(f"\nSaving up to {max_examples} examples to {output_dir}...")
    
    for child_pid, parent_pid in zero_gen_with_parent:
        if examples_saved >= max_examples:
            break
            
        child_img = image_dir / f"{child_pid}.png"
        parent_img = image_dir / f"{parent_pid}.png"
        
        if child_img.exists() and parent_img.exists():
            # Copy to output
            shutil.copy(child_img, output_dir / f"{child_pid}_child_gen0.png")
            shutil.copy(parent_img, output_dir / f"{child_pid}_parent_{parent_pid}.png")
            examples_saved += 1
            print(f"Saved pair: Child {child_pid} (Gen 0) <- Parent {parent_pid}")
        else:
            # print(f"Missing images for pair {child_pid} <- {parent_pid}")
            pass
            
    print(f"\nSaved {examples_saved} pairs.")

if __name__ == "__main__":
    main()
