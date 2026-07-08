#!/usr/bin/env python3
"""Render Picbreeder lineages using the NEAT-Python CPPN pipeline.

This script mirrors :mod:`save_lineage_figures.py`, but instead of replaying
legacy CPPNs with a JAX-based interpreter it converts every legacy genome into a
``PicbreederGenome`` and renders it with the same picture-rendering code used by
our modern NEAT workflows. The output therefore matches the activations,
clamping, and post-processing performed inside :mod:`collaborative_multi_agent`
(and other NEAT-driven tooling).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import traceback
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")  # set backend before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from fer.src.lineage_utils import get_lineage_genomes  # noqa: E402
from fer.src.save_lineage_figures import load_pbcppn  # noqa: E402
from picbreeder_vlm.core.rendering import render_genome_image  # noqa: E402
from tools.render_legacy_genome import (  # noqa: E402
    LegacyGenome,
    LegacyLink,
    LegacyNode,
    _build_neat_config,
    _infer_scheme,
    _legacy_to_picbreeder_genome,
)

import neat  # noqa: E402
from PIL import Image  # noqa: E402

from picture2d.common import (  # noqa: E402
    _canvas_coords,
    hsb_to_rgb,
)

ConfigCache = Dict[str, neat.Config]


def _ensure_config(scheme: str, cache: ConfigCache) -> neat.Config:
    scheme_normalized = scheme.lower()
    if scheme_normalized not in cache:
        cache[scheme_normalized] = _build_neat_config()
    return cache[scheme_normalized]


def _format_neat_node(node_key: int, labels: Dict[int, str]) -> str:
    label = labels.get(int(node_key), "")
    if label:
        return f"{label} [{node_key}]"
    return str(node_key)


def _format_connection(edge: Tuple[int, int], labels: Dict[int, str]) -> str:
    src, dst = edge
    return f"{_format_neat_node(src, labels)} -> {_format_neat_node(dst, labels)}"


def _log_recurrent_details(
    legacy: LegacyGenome,
    genome: neat.DefaultGenome,
    recurrent_edges: Set[Tuple[int, int]],
    disabled_edges: List[Tuple[int, int]],
    context: str,
) -> None:
    if not recurrent_edges and not disabled_edges:
        return

    label_map: Dict[int, str] = getattr(genome, "_legacy_node_label_map", {})
    connection_lookup: Dict[Tuple[int, int], LegacyLink] = getattr(
        genome,
        "_legacy_connection_lookup",
        {},
    )

    def describe(edge: Tuple[int, int]) -> str:
        text = _format_connection(edge, label_map)
        link = connection_lookup.get(edge)
        if link is not None:
            text = f"{text} [legacy={link.key}, w={link.weight:+.4f}]"
        return text

    disabled_set = set(disabled_edges)
    if disabled_edges:
        formatted = ", ".join(describe(edge) for edge in disabled_edges)
        tqdm.write(
            f"[INFO] Trimmed {len(disabled_edges)} recurrent connections in {context} genome "
            f"{legacy.identifier} (age {legacy.age}): {formatted}"
        )
    remaining = sorted(edge for edge in recurrent_edges if edge not in disabled_set)
    if remaining:
        formatted = ", ".join(describe(edge) for edge in remaining)
        tqdm.write(
            f"[INFO] Remaining recurrent connections in {context} genome {legacy.identifier} "
            f"(age {legacy.age}): {formatted}"
        )


def _convert_legacy_genome(legacy: LegacyGenome, cache: ConfigCache) -> Tuple[LegacyGenome, neat.Config, neat.DefaultGenome]:
    scheme = _infer_scheme(legacy)
    config = _ensure_config(scheme, cache)
    genome = _legacy_to_picbreeder_genome(legacy, config)
    return legacy, config, genome


def _split_marking(raw_id: str) -> Tuple[str, str, str]:
    text = str(raw_id)
    if "_" in text:
        branch, local = text.split("_", 1)
        return f"{branch}:{local}", branch, local
    return text, "", text


def _infer_node_type_and_affinity(label: str) -> Tuple[str, str]:
    label_lower = label.lower()
    if label_lower in {"x", "y", "d", "bias"}:
        return "in", "grey"
    if label_lower in {"hue", "saturation"}:
        return "out", "color"
    if label_lower in {"brightness", "value", "v", "lightness", "ink"}:
        return "out", "grey"
    return "hidden", "grey"


def _dict_to_legacy_genome(genome_dict: Dict, *, pid: str, index: int) -> Optional[LegacyGenome]:
    try:
        cppn = load_pbcppn(genome_dict)
    except Exception as exc:  # pragma: no cover - defensive logging
        tqdm.write(f"[WARN] Failed to decode Picbreeder genome for pid {pid} idx {index}: {exc}")
        return None

    nodes: Dict[str, LegacyNode] = {}
    for node in cppn["nodes"]:
        label = node.get("label", "") or ""
        activation = node.get("activation", "") or "identity"
        key_combined, branch, local = _split_marking(node.get("id", ""))
        node_type, affinity = _infer_node_type_and_affinity(label)
        legacy_node = LegacyNode(
            key=key_combined,
            branch=branch,
            local_id=local,
            type=node_type,
            label=label,
            activation=activation,
            affinity=affinity,
            bias=0.0,
        )
        nodes[key_combined] = legacy_node

    links: List[LegacyLink] = []
    for link in cppn["links"]:
        weight = float(link.get("weight", 0.0))
        key_combined, branch, local = _split_marking(link.get("id", ""))
        source_key, _, _ = _split_marking(link.get("source", ""))
        target_key, _, _ = _split_marking(link.get("target", ""))
        links.append(
            LegacyLink(
                key=key_combined,
                branch=branch,
                local_id=local,
                source_key=source_key,
                target_key=target_key,
                weight=weight,
            )
        )

    try:
        age = int(genome_dict.get("@age", 0))
    except (TypeError, ValueError):
        age = 0
    phenotype = str(genome_dict.get("@phenotype", "structure"))
    identifier = str(genome_dict.get("@id", f"{pid}:{age}:{index}"))

    outputs_present = [node for node in nodes.values() if node.type == "out"]
    if not outputs_present:
        for alias in ("brightness", "hue", "saturation"):
            key_combined, branch, local = _split_marking(f"synthetic_{alias}_{index}")
            node_type, affinity = _infer_node_type_and_affinity(alias)
            nodes[key_combined] = LegacyNode(
                key=key_combined,
                branch=branch,
                local_id=local,
                type=node_type,
                label=alias,
                activation="identity",
                affinity=affinity,
                bias=0.0,
            )

    legacy = LegacyGenome(
        identifier=identifier,
        age=age,
        phenotype=phenotype,
        parents=tuple(),
        nodes=nodes,
        links=tuple(links),
    )
    return legacy


def _detect_cyclic_nodes(genome: neat.DefaultGenome) -> Tuple[Set[int], Set[Tuple[int, int]]]:
    connections = [conn.key for conn in genome.connections.values() if conn.enabled]
    if not connections:
        return set(), set()

    edges: Dict[int, Set[int]] = {}
    nodes: Set[int] = set()
    self_loops: Set[int] = set()
    for src, dst in connections:
        nodes.add(src)
        nodes.add(dst)
        edges.setdefault(src, set()).add(dst)
        edges.setdefault(dst, set())
        if src == dst:
            self_loops.add(src)

    index_counter = [0]
    stack: List[int] = []
    on_stack: Set[int] = set()
    indices: Dict[int, int] = {}
    lowlinks: Dict[int, int] = {}
    cyclic_nodes: Set[int] = set()
    node_to_component: Dict[int, int] = {}
    cyclic_components: List[Set[int]] = []

    def strongconnect(node: int) -> None:
        indices[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack.add(node)

        for neighbour in edges.get(node, ()):  # type: ignore[arg-type]
            if neighbour not in indices:
                strongconnect(neighbour)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbour])
            elif neighbour in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbour])

        if lowlinks[node] == indices[node]:
            component: List[int] = []
            while True:
                element = stack.pop()
                on_stack.remove(element)
                component.append(element)
                if element == node:
                    break
            component_set = set(component)
            has_self_loop = any(member in self_loops for member in component_set)
            if len(component_set) > 1 or has_self_loop:
                component_index = len(cyclic_components)
                cyclic_components.append(component_set)
                for member in component_set:
                    node_to_component[member] = component_index
                cyclic_nodes.update(component_set)

    for node in nodes:
        if node not in indices:
            strongconnect(node)

    cyclic_edges: Set[Tuple[int, int]] = set()
    if cyclic_components:
        for src, dst in connections:
            comp_idx = node_to_component.get(src)
            if comp_idx is not None and comp_idx == node_to_component.get(dst):
                cyclic_edges.add((src, dst))

    return cyclic_nodes, cyclic_edges


def _find_recurrent_connection_keys(
    genome: neat.DefaultGenome,
    config: neat.Config,
) -> Set[Tuple[int, int]]:
    """Replicate the legacy DFS trimming: mark edges that point to an ancestor."""

    incoming: Dict[int, List[Any]] = {}
    for conn in genome.connections.values():
        if not conn.enabled:
            continue
        src, dst = conn.key
        incoming.setdefault(dst, []).append(conn)

    visited: Set[int] = set()
    recurrent_edges: Set[Tuple[int, int]] = set()

    def explore(node: int, path: Tuple[int, ...]) -> None:
        if node in path:
            return
        if node in visited:
            return

        next_path = path + (node,)
        parents = incoming.get(node, ())
        for conn in parents:
            src = conn.key[0]
            if src in next_path:
                recurrent_edges.add(conn.key)
                continue
            explore(src, next_path)

        visited.add(node)

    for output_key in config.genome_config.output_keys:
        explore(output_key, tuple())

    return recurrent_edges


def _disable_cyclic_connections(
    genome: neat.DefaultGenome,
    cyclic_edges: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    disabled: List[Tuple[int, int]] = []
    if not cyclic_edges:
        return disabled
    for conn in genome.connections.values():
        if conn.enabled and conn.key in cyclic_edges:
            conn.enabled = False
            disabled.append(conn.key)
    return disabled


def _prepare_render_genome(
    legacy: LegacyGenome,
    cache: ConfigCache,
    cycle_strategy: str,
) -> Tuple[
    neat.Config,
    neat.DefaultGenome,
    Set[int],
    Set[Tuple[int, int]],
    List[Tuple[int, int]],
]:
    _, config, genome = _convert_legacy_genome(legacy, cache)
    cyclic_nodes, _ = _detect_cyclic_nodes(genome)
    recurrent_edges = _find_recurrent_connection_keys(genome, config)
    disabled_edges: List[Tuple[int, int]] = []
    if recurrent_edges and cycle_strategy == "trim":
        disabled_edges = _disable_cyclic_connections(genome, recurrent_edges)
    return config, genome, cyclic_nodes, recurrent_edges, disabled_edges


def _render_color_image(
    legacy: LegacyGenome,
    cache: ConfigCache,
    size: int,
    cycle_strategy: str,
):
    config, genome, cyclic_nodes, recurrent_edges, disabled_edges = _prepare_render_genome(
        legacy,
        cache,
        cycle_strategy,
    )

    if cycle_strategy == "recurrent" and cyclic_nodes:
        color_image = _render_color_image_recurrent(genome, config, size)
        return color_image, genome, cyclic_nodes, recurrent_edges, disabled_edges

    _, color_image = render_genome_image(genome, config, size, size)
    return color_image, genome, cyclic_nodes, recurrent_edges, disabled_edges


def _render_color_image_recurrent(genome: neat.DefaultGenome, config: neat.Config, size: int) -> Image.Image:
    net = neat.nn.RecurrentNetwork.create(genome, config)
    color_rows: List[List[Tuple[int, int, int]]] = []
    for coord_row in _canvas_coords(size, size):
        row: List[Tuple[int, int, int]] = []
        for coords in coord_row:
            inputs = list(coords)
            transformer = getattr(genome, "transform_inputs", None)
            if transformer is not None:
                inputs = transformer(inputs)
            net.reset()
            outputs = net.activate(inputs)
            output_transformer = getattr(genome, "transform_outputs", None)
            if output_transformer is not None:
                outputs = output_transformer(outputs)
            rgb = hsb_to_rgb(outputs[:3])
            row.append(rgb)
        color_rows.append(row)
    array = np.array(color_rows, dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def _render_lineage_figure(
    genomes: List[LegacyGenome],
    cache: ConfigCache,
    max_genomes: int,
    grid_cols: int,
    size: int,
    cycle_strategy: str,
):
    if not genomes:
        return None, 0
    take = len(genomes) if max_genomes < 0 else min(len(genomes), max_genomes)
    if take <= 0:
        return None, 0
    subset = genomes[-take:]

    cols = max(1, grid_cols)
    rows = max(1, math.ceil(take / cols))
    figsize = (cols, rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    for ax, legacy in zip(axes, subset):
        try:
            color_image, genome, cyclic_nodes, recurrent_edges, disabled_edges = _render_color_image(
                legacy,
                cache,
                size,
                cycle_strategy,
            )
            rgb = np.asarray(color_image)
            ax.imshow(rgb)
            if recurrent_edges or disabled_edges:
                _log_recurrent_details(
                    legacy,
                    genome,
                    recurrent_edges,
                    disabled_edges,
                    context="lineage",
                )
            elif cyclic_nodes:
                tqdm.write(
                    f"[INFO] Cyclic nodes detected in lineage genome {legacy.identifier} (age {legacy.age})"
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            ax.text(0.5, 0.5, "render err", ha="center", va="center", color="red", fontsize=6)
            traceback.print_exc()
            tqdm.write(
                f"[WARN] Rendering failed for genome {legacy.identifier} (age {legacy.age}): {exc}"
            )
        ax.set_title(str(legacy.age), fontsize=6)

    plt.tight_layout(pad=0.3)
    return fig, take


def _render_final_image(
    genomes: List[LegacyGenome],
    cache: ConfigCache,
    size: int,
    cycle_strategy: str,
):
    if not genomes:
        return None
    final_genome = genomes[-1]
    try:
        color_image, genome, cyclic_nodes, recurrent_edges, disabled_edges = _render_color_image(
            final_genome,
            cache,
            size,
            cycle_strategy,
        )
        if recurrent_edges or disabled_edges:
            _log_recurrent_details(
                final_genome,
                genome,
                recurrent_edges,
                disabled_edges,
                context="final",
            )
        elif cyclic_nodes:
            tqdm.write(
                f"[INFO] Cyclic nodes detected in final genome {final_genome.identifier}"
            )
        return color_image
    except Exception as exc:  # pragma: no cover - defensive logging
        tqdm.write(
            f"[WARN] Final image rendering failed for genome {final_genome.identifier}: {exc}"
        )
        return None


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Save lineage figures using NEAT-Python rendering for every Picbreeder pid.",
    )
    parser.add_argument(
        "--pb-dir",
        default=Path("../spaghetti/pbRender/genomeAll"),
        type=Path,
        help="Directory containing pid subdirectories (each with Picbreeder zip files).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/lineages_neat"),
        help="Directory to write generated lineage figures (default: figures/lineages_neat).",
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
        default=200,
        help="Render resolution for each genome (width/height in pixels).",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        help="Matplotlib format/extension for the saved figures (default: pdf).",
    )
    parser.add_argument(
        "--archive-final",
        action="store_true",
        help="If set, skip lineage grids and save each pid's final genome as a PNG.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("archive_neat"),
        help="Directory for PNGs when --archive-final is set (default: ./archive_neat).",
    )
    parser.add_argument(
        "--cycle-strategy",
        choices=("trim", "recurrent"),
        default="trim",
        help=(
            "How to handle cyclic CPPNs: 'trim' disables connections touching cycles to "
            "mimic legacy JAX behaviour, 'recurrent' renders with a recurrent NEAT network."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    pb_dir = args.pb_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    archive_dir = args.archive_dir.expanduser().resolve()

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

    if not pids:
        raise SystemExit(f"No pid directories found in {pb_dir}")

    config_cache: ConfigCache = {}

    for pid in tqdm(pids, desc="Rendering lineages (NEAT)"):
        try:
            raw_genomes = get_lineage_genomes(pb_dir, pid)
        except Exception as exc:  # pragma: no cover - defensive logging aid
            tqdm.write(f"[WARN] Failed to gather genomes for pid {pid}: {exc}")
            continue

        genomes: List[LegacyGenome] = []
        for idx, genome_dict in enumerate(raw_genomes):
            legacy = _dict_to_legacy_genome(genome_dict, pid=pid, index=idx)
            if legacy is not None:
                genomes.append(legacy)
        genomes.sort(key=lambda g: g.age)

        if not genomes:
            tqdm.write(f"[INFO] No genomes found for pid {pid}; skipping.")
            continue

        if args.archive_final:
            color_image = _render_final_image(
                genomes,
                config_cache,
                args.res,
                args.cycle_strategy,
            )
            if color_image is None:
                tqdm.write(f"[WARN] Could not render final genome for pid {pid}; skipping.")
                continue
            out_path = archive_dir / f"{pid}.png"
            color_image.save(out_path, format="PNG")
            tqdm.write(f"[OK] Archived final genome for pid {pid} -> {out_path}")
            continue

        figure, count = _render_lineage_figure(
            genomes,
            config_cache,
            max_genomes=args.max_genomes,
            grid_cols=args.grid_size,
            size=args.res,
            cycle_strategy=args.cycle_strategy,
        )
        if figure is None:
            tqdm.write(f"[INFO] No genomes rendered for pid {pid}; skipping.")
            continue
        out_path = output_dir / f"{pid}.{args.format}"
        figure.savefig(out_path, format=args.format, bbox_inches="tight", dpi=100)
        plt.close(figure)
        tqdm.write(f"[OK] Saved {count} genomes for pid {pid} -> {out_path}")


if __name__ == "__main__":
    main()
