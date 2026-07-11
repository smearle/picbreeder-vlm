#!/usr/bin/env python3
"""
Count grayscale-only vs color Picbreeder genomes across lineages.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

from tqdm.auto import tqdm

from lineage_utils import _ensure_list, get_lineage_genomes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count grayscale-only (ink-based) vs color genomes across Picbreeder lineages.",
    )
    parser.add_argument(
        "--pb-dir",
        default=Path("../spaghetti/pbrender/genomeAll"),
        type=Path,
        help="Directory containing pid subdirectories with Picbreeder archives.",
    )
    parser.add_argument(
        "--per-lineage",
        action="store_true",
        help="Print one line per pid detailing whether the final genome is grayscale or color.",
    )
    return parser.parse_args()


def classify_genome(genome: Dict) -> str:
    """
    Return 'grayscale' when the genome uses the legacy ink node, otherwise 'color'.
    """
    node_container = genome.get("nodes", {})
    node_entries = _ensure_list(node_container.get("node"))
    labels = {node.get("@label", "") for node in node_entries}
    return "grayscale" if "ink" in labels else "color"


def summarize_counts(counter: Counter) -> str:
    total = sum(counter.values())
    if total == 0:
        return "total=0"
    grayscale = counter.get("grayscale", 0)
    color = counter.get("color", 0)
    return (
        f"total={total} | grayscale={grayscale} ({grayscale / total:.1%}) | "
        f"color={color} ({color / total:.1%})"
    )


def main():
    args = parse_args()
    pb_dir = args.pb_dir.expanduser().resolve()
    if not pb_dir.is_dir():
        raise SystemExit(f"Picbreeder directory does not exist: {pb_dir}")

    pids = sorted(
        pid.name
        for pid in pb_dir.iterdir()
        if pid.is_dir() and (pid / "main.zip").exists()
    )
    if not pids:
        raise SystemExit(f"No pid directories found in {pb_dir}")

    stats = {
        "intermediate": Counter(),
        "final": Counter(),
    }
    per_lineage_rows: List[str] = []

    for pid in tqdm(pids, desc="Scanning lineages"):
        try:
            genomes = get_lineage_genomes(pb_dir, pid)
        except Exception as exc:
            print(f"Warning: failed to load lineage for pid {pid}: {exc}")
            continue
        if not genomes:
            if args.per_lineage:
                per_lineage_rows.append(f"{pid},0,none,0,0")
            continue

        intermediate = genomes[:-1]
        final_genome = genomes[-1]

        intermediate_classes = [classify_genome(genome) for genome in intermediate]
        for label in intermediate_classes:
            stats["intermediate"][label] += 1

        final_class = classify_genome(final_genome)
        stats["final"][final_class] += 1

        if args.per_lineage:
            grayscale_intermediate = sum(1 for label in intermediate_classes if label == "grayscale")
            color_intermediate = len(intermediate_classes) - grayscale_intermediate
            row = f"{pid},{len(genomes)},{final_class},{grayscale_intermediate},{color_intermediate}"
            print(row)
            per_lineage_rows.append(row)

    print("=== Intermediary genomes ===")
    print(summarize_counts(stats["intermediate"]))
    print("\n=== Final genomes ===")
    print(summarize_counts(stats["final"]))

    if args.per_lineage and per_lineage_rows:
        print("\npid,total_genomes,final_class,intermediate_grayscale,intermediate_color")
        for row in per_lineage_rows:
            print(row)

    # Save this data
    output_path = pb_dir / "lineage_color_mode_counts.csv"
    with output_path.open("w", encoding="utf-8") as f:
        f.write("pid,total_genomes,final_class,intermediate_grayscale,intermediate_color\n")
        for row in per_lineage_rows:
            f.write(f"{row}\n")
    print(f"\nSaved detailed per-lineage data to: {output_path}")


if __name__ == "__main__":
    main()
