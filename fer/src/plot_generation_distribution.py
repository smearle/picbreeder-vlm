#!/usr/bin/env python3
"""
Plot the distribution of number of generations between publications in the human archive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# Add root to path to allow importing modules
# This assumes the script is located at fer/src/plot_generation_distribution.py
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

def _get_generation_count(pb_dir: Path, pid: str) -> Optional[int]:
    """
    Reads main.zip to find the last generation number.
    """
    main_zip = pb_dir / pid / "main.zip"
    if not main_zip.exists():
        return None
        
    try:
        main_data = load_zip_xml_as_dict(str(main_zip))
    except Exception as exc:
        # tqdm.write(f"[WARN] Failed to read main.zip for {pid}: {exc}")
        return None

    genome_meta = main_data.get("genome", {})
    series = genome_meta.get("series", {})
    
    # Find last generation
    generations = series.get("generation")
    if not generations:
        return None
    generations = _ensure_list(generations)
    
    if not generations:
        return None
        
    last_gen_meta = max(generations, key=lambda g: _safe_int(g.get("@number", -1)))
    last_gen_num = _safe_int(last_gen_meta.get("@number"))
    
    # User analysis suggests generation "0" is actually the 1st generation of evolution.
    # We apply a +1 offset to all counts.
    return last_gen_num + 1

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot distribution of generations between publications."
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
        default=Path("human_lineages/human_generations_hist.png"),
        help="Output path for the plot.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of PIDs to scan.",
    )
    parser.add_argument(
        "--max-gen",
        type=int,
        default=100,
        help="Max generation to plot in histogram (for better visualization).",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    pb_dir = args.pb_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    
    if not pb_dir.is_dir():
        raise FileNotFoundError(f"Picbreeder directory not found: {pb_dir}")
        
    pids = sorted(
        pid.name
        for pid in pb_dir.iterdir()
        if pid.is_dir() and (pid / "main.zip").exists()
    )
    
    if args.limit:
        pids = pids[:args.limit]
        
    print(f"Scanning {len(pids)} lineages...")
    
    generation_counts = []
    
    for pid in tqdm(pids):
        count = _get_generation_count(pb_dir, pid)
        if count is not None:
            generation_counts.append(count)
            
    print(f"Found {len(generation_counts)} valid generation counts.")
    
    if not generation_counts:
        print("No data found. Exiting.")
        return

    # Basic stats
    min_gen = min(generation_counts)
    max_gen = max(generation_counts)
    mean_gen = np.mean(generation_counts)
    median_gen = np.median(generation_counts)
    
    count_le_20 = sum(1 for c in generation_counts if c <= 20)
    pct_le_20 = (count_le_20 / len(generation_counts)) * 100

    print(f"Min generations: {min_gen}")
    print(f"Max generations: {max_gen}")
    print(f"Mean generations: {mean_gen:.2f}")
    print(f"Median generations: {median_gen:.2f}")
    print(f"Percentage <= 20 generations: {pct_le_20:.2f}%")
    
    # Generate LaTeX table
    tex_output_path = output_path.parent / "human_generations_stats.tex"
    tex_content = f"""\\begin{{table}}[ht]
    \\centering
    \\caption{{Statistics of generations between publications in the human archive.}}
    \\begin{{tabular}}{{lc}}
        \\toprule
        \\textbf{{Statistic}} & \\textbf{{Value}} \\\\
        \\midrule
        Mean Generations & {mean_gen:.2f} \\\\
        Median Generations & {median_gen:.2f} \\\\
        Min Generations & {min_gen} \\\\
        Max Generations & {max_gen} \\\\
        \\% $\\le$ 20 Generations & {pct_le_20:.1f}\\% \\\\
        \\bottomrule
    \\end{{tabular}}
    \\label{{tab:human_gen_stats}}
\\end{{table}}
"""
    with open(tex_output_path, "w") as f:
        f.write(tex_content)
    print(f"Saved LaTeX stats table to {tex_output_path}")
    
    # Save raw data
    data_path = output_path.with_suffix(".json")
    with open(data_path, "w") as f:
        json.dump(generation_counts, f)
    print(f"Saved raw data to {data_path}")

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Filter for visualization if needed (or just plot log scale)
    # Let's do a standard histogram with a cutoff for outliers if requested, 
    # but also log scale option?
    
    # Plot 1: Full distribution (up to sensible max or all)
    # If max is huge, maybe cap it.
    
    filtered_counts = [c for c in generation_counts if c <= args.max_gen]
    outliers = len(generation_counts) - len(filtered_counts)
    
    plt.hist(filtered_counts, bins=range(0, args.max_gen + 2), edgecolor='black', alpha=0.7)
    plt.axvline(x=20, color='red', linestyle=':', linewidth=2, label='20 Generations')
    plt.legend()

    plt.title(f"Distribution of Generations Between Publications\n(n={len(generation_counts)}, showing <= {args.max_gen}, {outliers} outliers)")
    plt.xlabel("Number of Generations")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.5)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    
    # Plot 2: Log scale / Full range
    plt.figure(figsize=(10, 6))
    plt.hist(generation_counts, bins=50, edgecolor='black', alpha=0.7, log=True)
    plt.axvline(x=20, color='red', linestyle=':', linewidth=2, label='20 Generations')
    plt.legend()
    
    plt.title(f"Distribution of Generations Between Publications (Log Scale)\n(n={len(generation_counts)})")
    plt.xlabel("Number of Generations")
    plt.ylabel("Count (Log)")
    plt.grid(axis='y', alpha=0.5)
    
    log_output_path = output_path.parent / f"{output_path.stem}_log{output_path.suffix}"
    plt.savefig(log_output_path)
    print(f"Saved log scale plot to {log_output_path}")

if __name__ == "__main__":
    main()
