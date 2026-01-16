#!/usr/bin/env python3
"""
Generate lineage figures for every Picbreeder pid in a directory.

This script mirrors the plotting logic from `picbreeder_genomes_lineages.ipynb`
but makes it easy to batch-export the figures as image files.
"""

from __future__ import annotations

import os

# Enforce CPU usage for JAX to avoid GPU OOM/contention and because individual CPPNs are small/variable
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import argparse
import math
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Dict, List, Set, Tuple

import jax.numpy as jnp
import matplotlib
import numpy as np
from tqdm.auto import tqdm

matplotlib.use("Agg")  # noqa: E402 - set backend before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402

from fer.src.color import hsv2rgb
from fer.src.constants import activation_fn_map
from fer.src.lineage_utils import _ensure_list, get_lineage_genomes


def load_pbcppn(genome: Dict) -> Dict:
    nodes_ = _ensure_list(genome["nodes"]["node"])
    links_ = _ensure_list(genome["links"]["link"])
    nodes, links = [], []
    for node in nodes_:
        node_id = f"{node['marking']['@branch']}_{node['marking']['@id']}"
        nodes.append(
            {
                "label": node.get("@label", ""),
                "id": node_id,
                "activation": node["activation"]["#text"][:-3],
            }
        )
    for link in links_:
        link_dict = {
            "id": int(link["marking"]["@id"]),
            "source": f"{link['source']['@branch']}_{link['source']['@id']}",
            "target": f"{link['target']['@branch']}_{link['target']['@id']}",
            "weight": float(link["weight"]["#text"]),
        }
        links.append(link_dict)

    labels = [node["label"] for node in nodes]
    if "ink" in labels:
        node_v = nodes[labels.index("ink")]
        node_v["label"] = "brightness"
        nodes.append({"label": "hue", "id": 1_000_000, "activation": "identity"})
        nodes.append({"label": "saturation", "id": 1_000_001, "activation": "identity"})
        links.append({"id": 1_000_002, "source": node_v["id"], "target": 1_000_000, "weight": 0.0})
        links.append({"id": 1_000_003, "source": node_v["id"], "target": 1_000_001, "weight": 0.0})

    def special(label):
        return [node["id"] for node in nodes if node["label"] == label][0]

    special_nodes = {
        "x": special("x"),
        "y": special("y"),
        "d": special("d"),
        "bias": special("bias"),
        "h": special("hue"),
        "s": special("saturation"),
        "v": special("brightness"),
    }
    return {"nodes": nodes, "links": links, "special_nodes": special_nodes}


def _format_node_label(node_id: str, labels: Dict[str, str]) -> str:
    label = labels.get(node_id, "")
    if label:
        return f"{label} ({node_id})"
    return node_id


def do_forward_pass(nn: Dict, res: int = 64):
    x = y = jnp.linspace(-1.0, 1.0, res)
    x, y = jnp.meshgrid(x, y)
    d = jnp.sqrt(x ** 2 + y ** 2) * 1.4
    b = jnp.ones_like(x)

    node2activation = {n["id"]: n["activation"] for n in nn["nodes"]}
    node2in_links = {
        n["id"]: [(l["source"], l["weight"]) for l in nn["links"] if l["target"] == n["id"]]
        for n in nn["nodes"]
    }
    node_labels = {n["id"]: n.get("label", "") or "" for n in nn["nodes"]}
    recurrent_edges: Set[Tuple[str, str]] = set()
    recurrent_cycles: List[List[str]] = []

    node_x = nn["special_nodes"]["x"]
    node_y = nn["special_nodes"]["y"]
    node_d = nn["special_nodes"]["d"]
    node_b = nn["special_nodes"]["bias"]
    node_h = nn["special_nodes"]["h"]
    node_s = nn["special_nodes"]["s"]
    node_v = nn["special_nodes"]["v"]

    node2val = {node_x: x, node_y: y, node_d: d, node_b: b}

    def get_value(node, path=None):
        if node in node2val:
            return node2val[node]
        path = path or []
        if node in path:
            idx = path.index(node)
            cycle = path[idx:] + [node]
            recurrent_cycles.append(cycle)
            if path:
                recurrent_edges.add((path[-1], node))
            for src, dst in zip(cycle, cycle[1:]):
                recurrent_edges.add((src, dst))
            return jnp.zeros_like(x)
        val = jnp.zeros_like(x)
        for node_src, weight in node2in_links[node]:
            val = val + weight * get_value(node_src, path=path + [node])
        node2val[node] = activation_fn_map[node2activation[node]](val)
        return node2val[node]

    for node in (node_h, node_s, node_v):
        get_value(node)

    h, s, v = node2val[node_h], node2val[node_s], node2val[node_v]
    r, g, b = hsv2rgb((h + 1) % 1, s.clip(0, 1), jnp.abs(v).clip(0, 1))
    rgb = jnp.stack([r, g, b], axis=-1)
    return {
        "rgb": rgb,
        "recurrent_edges": recurrent_edges,
        "recurrent_cycles": recurrent_cycles,
        "node_labels": node_labels,
    }


def render_lineage_figure(genomes: List[Dict], max_genomes: int, grid_cols: int, res: int):
    if not genomes:
        return None, 0
    if max_genomes is None or max_genomes < 0:
        take = len(genomes)
    else:
        take = min(len(genomes), max_genomes)
    if take == 0:
        return None, 0
    subset = genomes[-take:]

    grid_cols = max(1, grid_cols)
    rows = max(1, math.ceil(take / grid_cols))
    figsize = (grid_cols, rows)
    fig, axes = plt.subplots(rows, grid_cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    for ax, genome in zip(axes, subset):
        nn = load_pbcppn(genome)
        forward = do_forward_pass(nn, res=res)
        rgb = np.asarray(forward["rgb"])
        ax.imshow(np.clip(rgb, 0, 1))
        ax.set_title(str(genome.get("@age", "?")), fontsize=6)
        if forward["recurrent_edges"]:
            formatted = ", ".join(
                f"{_format_node_label(src, forward['node_labels'])} -> {_format_node_label(dst, forward['node_labels'])}"
                for src, dst in sorted(forward["recurrent_edges"])
            )
            tqdm.write(
                f"[INFO] Recurrent edges detected (trimmed to zero) for genome age {genome.get('@age', '?')}: {formatted}"
            )
            if forward["recurrent_cycles"]:
                for cycle in forward["recurrent_cycles"]:
                    cycle_fmt = " -> ".join(
                        _format_node_label(node_id, forward["node_labels"]) for node_id in cycle
                    )
                    tqdm.write(f"        cycle: {cycle_fmt}")
        else:
            pass
            # print("No recurrent edges detected.")

    plt.tight_layout(pad=0.3)
    return fig, take


def render_final_image(genomes: List[Dict], res: int = 200):
    if not genomes:
        return None
    final_genome = genomes[-1]
    nn = load_pbcppn(final_genome)
    forward = do_forward_pass(nn, res=res)
    if forward["recurrent_edges"]:
        formatted = ", ".join(
            f"{_format_node_label(src, forward['node_labels'])} -> {_format_node_label(dst, forward['node_labels'])}"
            for src, dst in sorted(forward["recurrent_edges"])
        )
        tqdm.write(
            f"[INFO] Recurrent edges detected (trimmed to zero) for final genome age {final_genome.get('@age', '?')}: {formatted}"
        )
        if forward["recurrent_cycles"]:
            for cycle in forward["recurrent_cycles"]:
                cycle_fmt = " -> ".join(
                    _format_node_label(node_id, forward["node_labels"]) for node_id in cycle
                )
                tqdm.write(f"        cycle: {cycle_fmt}")
    else:
        pass
        # print("No recurrent edges detected.")
    rgb = np.asarray(forward["rgb"])
    return np.clip(rgb, 0, 1)


def process_one_pid(pid, pb_dir, output_dir, archive_dir, args):
    if args.archive_final:
        out_path = archive_dir / f"{pid}.png"
    else:
        out_path = output_dir / f"{pid}.{args.format}"

    if out_path.exists():
        return None

    try:
        genomes = get_lineage_genomes(pb_dir, pid)
    except Exception as exc:
        return f"[WARN] Failed to gather genomes for pid {pid}: {exc}"

    if args.archive_final:
        rgb = render_final_image(genomes, res=args.res)
        if rgb is None:
            return f"[INFO] No genomes found for pid {pid}; skipping."
        plt.imsave(out_path, rgb)
        return f"[OK] Archived final genome for pid {pid} -> {out_path}"
    else:
        fig, count = render_lineage_figure(
            genomes,
            max_genomes=args.max_genomes,
            grid_cols=args.grid_size,
            res=args.res,
        )
        if fig is None:
            return f"[INFO] No genomes found for pid {pid}; skipping."
        fig.savefig(out_path, format=args.format, bbox_inches="tight", dpi=100)
        plt.close(fig)
        return f"[OK] Saved {count} genomes for pid {pid} -> {out_path}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save lineage figures (adaptive grids) for every pid in a Picbreeder directory.",
    )
    parser.add_argument(
        "--pb-dir",
        default=Path("../spaghetti/pbRender/genomeAll"),
        type=Path,
        help="Directory that contains pid subdirectories (each with Picbreeder zip files).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lineages"),
        help="Where to write the generated figures (default: figures/lineages).",
    )
    parser.add_argument(
        "--max-genomes",
        type=int,
        default=-1,
        help="Maximum number of genomes per figure (-1 renders all genomes).",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=20,
        help="Number of columns in the grid (rows adapt automatically).",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=128,
        help="Resolution for rendering each genome (default: 200).",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        help="matplotlib format/extension for the saved figures (default: pdf).",
    )
    parser.add_argument(
        "--archive-final",
        action="store_true",
        help="If set, skip lineage grids and save each pid's final genome as a PNG.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("archive"),
        help="Directory for PNGs when --archive-final is set (default: ./archive).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of pid folders to process (helps with smoke tests).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pb_dir = args.pb_dir.expanduser().resolve()
    ouput_dir = Path(str(args.output_dir) + f"res-{args.res}")
    output_dir = ouput_dir.expanduser().resolve()
    archive_dir = Path(str(args.archive_dir) + f"_res-{args.res}")
    archive_dir = archive_dir.expanduser().resolve()
    if args.archive_final:
        archive_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    if not pb_dir.is_dir():
        raise SystemExit(f"Picbreeder directory does not exist: {pb_dir}")

    pids = sorted(
        pid.name
        for pid in pb_dir.iterdir()
        if pid.is_dir() and (pid / "main.zip").exists()
    )

    if args.limit is not None:
        pids = pids[: args.limit]
    if not pids:
        raise SystemExit(f"No pid directories found in {pb_dir}")

    # Multiprocessing setup
    num_workers = multiprocessing.cpu_count()
    if args.limit and args.limit < num_workers:
        num_workers = args.limit

    func = partial(
        process_one_pid,
        pb_dir=pb_dir,
        output_dir=output_dir,
        archive_dir=archive_dir,
        args=args,
    )

    print(f"Rendering lineages using {num_workers} workers...")
    with multiprocessing.Pool(num_workers) as pool:
        for msg in tqdm(pool.imap_unordered(func, pids), total=len(pids), desc="Rendering lineages"):
            if msg:
                tqdm.write(msg)


if __name__ == "__main__":
    main()
