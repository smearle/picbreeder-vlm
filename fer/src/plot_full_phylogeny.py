#!/usr/bin/env python3
"""Reconstruct and render the global Picbreeder phylogeny using Graphviz."""

from __future__ import annotations

import argparse
import html
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image
from graphviz import Digraph
from tqdm.auto import tqdm

from fer.src.lineage_utils import (
    get_lineage_genomes,
    iter_generation_archives,
    recursive_parse_all_genomes,
)
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

GenomeKey = Tuple[str, str]


@dataclass
class IndexedGenome:
    key: GenomeKey
    pid: str
    genome: Dict[str, Any]
    age: int
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


def _extract_genome_key(genome: Mapping[str, Any]) -> Optional[GenomeKey]:
    identifier = genome.get("identifier") if isinstance(genome, Mapping) else None
    if not isinstance(identifier, Mapping):
        return None
    branch = identifier.get("@branch")
    gid = identifier.get("@id")
    if branch is None or gid is None:
        return None
    return str(branch), str(gid)


def _extract_parent_keys(genome: Mapping[str, Any]) -> List[GenomeKey]:
    parents_block = genome.get("parents") if isinstance(genome, Mapping) else None
    if not isinstance(parents_block, Mapping):
        return []
    identifiers = _ensure_list(parents_block.get("identifier"))
    parent_keys: List[GenomeKey] = []
    for identifier in identifiers:
        if not isinstance(identifier, Mapping):
            continue
        branch = identifier.get("@branch")
        gid = identifier.get("@id")
        if branch is None or gid is None:
            continue
        parent_keys.append((str(branch), str(gid)))
    return parent_keys


def _format_node_id(key: GenomeKey) -> str:
    branch, local_id = key
    return f"{branch}_{local_id}"


def _format_node_display(key: GenomeKey) -> str:
    branch, local_id = key
    return f"{branch}:{local_id}"


def _index_genomes(lineages: Mapping[str, Sequence[Dict[str, Any]]]) -> Dict[GenomeKey, IndexedGenome]:
    indexed: Dict[GenomeKey, IndexedGenome] = {}
    for pid, genomes in lineages.items():
        for genome in genomes:
            key = _extract_genome_key(genome)
            if key is None or key in indexed:
                continue
            indexed[key] = IndexedGenome(
                key=key,
                pid=pid,
                genome=genome,
                age=_safe_int(genome.get("@age"), default=0),
            )
    if not indexed:
        raise SystemExit("No genomes with identifiers were recovered; nothing to render.")
    return indexed


def _thumbnail_filename(key: GenomeKey) -> str:
    branch, local_id = key
    return f"{branch}_{local_id}.png"


def _render_thumbnail_array(genome: Mapping[str, Any], size: int) -> Optional[np.ndarray]:
    if size <= 0:
        return None
    cppn = load_pbcppn(genome)
    forward = do_forward_pass(cppn, res=size)
    rgb = np.asarray(forward["rgb"], dtype=np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def _ensure_thumbnail(
    record: IndexedGenome,
    thumb_dir: Optional[Path],
    thumb_size: int,
) -> Optional[Path]:
    if thumb_dir is None or thumb_size <= 0:
        return None
    if record.thumbnail_path is not None and record.thumbnail_path.exists():
        return record.thumbnail_path
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = (thumb_dir / _thumbnail_filename(record.key)).resolve()
    if thumb_path.exists():
        record.thumbnail_path = thumb_path
        return thumb_path
    try:
        rgb = _render_thumbnail_array(record.genome, thumb_size)
    except Exception as exc:  # pragma: no cover - diagnostic aid
        tqdm.write(
            f"[WARN] Failed to render thumbnail for genome {record.key} (pid {record.pid}): {exc}"
        )
        return None
    if rgb is None:
        return None
    image = Image.fromarray(rgb, "RGB")
    image.save(thumb_path)
    record.thumbnail_path = thumb_path
    return thumb_path


def _build_node_label(
    record: IndexedGenome,
    depth: int,
    desc: int,
    thumbnail_path: Optional[Path],
    fill_color: str,
) -> str:
    meta_parts = [
        f"pid {html.escape(record.pid)}",
        f"age {record.age}",
        f"depth {depth}",
        f"desc {desc}",
    ]
    meta_line = " &#8226; ".join(meta_parts)
    image_row = ""
    if thumbnail_path is not None:
        image_row = (
            f"<TR><TD><IMG SRC=\"{html.escape(str(thumbnail_path))}\" SCALE='TRUE'/></TD></TR>"
        )
    label = (
        "<"
        f"<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='2' BGCOLOR='{fill_color}'>"
        f"{image_row}"
        f"<TR><TD ALIGN='LEFT'><FONT POINT-SIZE='10'><B>{html.escape(_format_node_display(record.key))}</B></FONT></TD></TR>"
        f"<TR><TD ALIGN='LEFT'><FONT POINT-SIZE='9'>{meta_line}</FONT></TD></TR>"
        "</TABLE>"
        ">"
    )
    return label


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
        default=Path("human_lineages/lineages/full_phylogeny"),
        help="Output path without suffix (default: human_lineages/lineages/full_phylogeny).",
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
    parser.add_argument(
        "--thumb-size",
        type=int,
        default=64,
        help="Pixel size for rendered genome thumbnails embedded in the tree (set to 0 to disable).",
    )
    parser.add_argument(
        "--thumb-dir",
        type=Path,
        default=None,
        help="Optional cache directory for thumbnails (defaults to <output>_thumbs alongside the final graph).",
    )
    return parser.parse_args()


def _discover_pids(pb_dir: Path) -> List[str]:
    if not pb_dir.is_dir():
        raise FileNotFoundError(f"Picbreeder directory does not exist: {pb_dir}")
    pids = sorted(
        pid.name
        for pid in pb_dir.iterdir()
        if pid.is_dir() and (pid / "main.zip").exists()
    )
    if not pids:
        raise SystemExit(f"No pid directories found in {pb_dir}")
    return pids


def _collect_lineages(pb_dir: Path, pids: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
    lineages: Dict[str, List[Dict[str, Any]]] = {}
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


def _load_genomes_single_pass(
    pb_dir: Path,
    pids: Iterable[str],
) -> Dict[GenomeKey, IndexedGenome]:
    """
    Load genomes by walking each pid directory exactly once and dedupe by branch/id.
    """
    indexed: Dict[GenomeKey, IndexedGenome] = {}
    for pid in tqdm(pids, desc="Loading genomes"):
        for archive in iter_generation_archives(pb_dir, pid):
            try:
                root = load_zip_xml_as_dict(str(archive))
            except Exception as exc:  # pragma: no cover - defensive logging
                tqdm.write(f"[WARN] Failed to parse archive {archive} for pid {pid}: {exc}")
                continue
            generation = (
                root.get("genome", {})
                .get("storage", {})
                .get("generation")
            )
            # genomes = recursive_parse_all_genomes(generation)
            genomes = get_lineage_genomes(pb_dir, pid)
            for genome in genomes:
                key = _extract_genome_key(genome)
                if key is None or key in indexed:
                    continue
                indexed[key] = IndexedGenome(
                    key=key,
                    pid=pid,
                    genome=genome,
                    age=_safe_int(genome.get("@age"), default=0),
                )
    if not indexed:
        raise SystemExit("No genomes with identifiers were recovered; nothing to render.")
    return indexed


def _normalize_limit(limit: Optional[int]) -> Optional[int]:
    if limit is None or limit < 0:
        return None
    return limit


def _parse_checkpoint_limit(raw: str) -> Optional[int]:
    if raw == "None":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _find_best_checkpoint(
    output_base: Path, requested_limit: Optional[int]
) -> Tuple[Optional[Path], Optional[int]]:
    prefix = f"{output_base.name}.limit-"
    suffix = "_indexed_genomes.npz"
    best_path: Optional[Path] = None
    best_limit: Optional[int] = None

    def _score(limit_val: Optional[int]) -> float:
        return float("inf") if limit_val is None else float(limit_val)

    for path in output_base.parent.glob(f"{output_base.name}.limit-*{suffix}"):
        name = path.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        raw_limit = name[len(prefix) : -len(suffix)]
        limit_val = _parse_checkpoint_limit(raw_limit)
        if limit_val is None and raw_limit != "None":
            continue
        if requested_limit is not None:
            if limit_val is None or limit_val > requested_limit:
                continue
        if best_path is None or _score(limit_val) > _score(best_limit):
            best_path = path
            best_limit = limit_val
    return best_path, best_limit


def _build_parent_maps(
    genomes: Mapping[GenomeKey, IndexedGenome]
) -> Tuple[
    Dict[GenomeKey, GenomeKey],
    Dict[GenomeKey, Set[GenomeKey]],
    Dict[Tuple[GenomeKey, GenomeKey], int],
]:
    pid_age_index: Dict[str, Dict[int, List[GenomeKey]]] = defaultdict(lambda: defaultdict(list))
    for key, record in genomes.items():
        pid_age_index[record.pid][record.age].append(key)

    parent_for: Dict[GenomeKey, GenomeKey] = {}
    children_for: Dict[GenomeKey, Set[GenomeKey]] = defaultdict(set)
    edge_counts: Dict[Tuple[GenomeKey, GenomeKey], int] = defaultdict(int)
    for key, record in genomes.items():
        parent_keys = _extract_parent_keys(record.genome)
        if len(parent_keys) == 0 and record.age > 0:
            # Fill missing parent by assuming previous-age genome from same lineage.
            fallback_parents = pid_age_index.get(record.pid, {}).get(record.age - 1, [])
            if fallback_parents:
                parent_keys = sorted(fallback_parents)
        for parent in parent_keys:
            if parent not in genomes:
                continue
            children_for[parent].add(key)
            edge_counts[(parent, key)] += 1
            if key not in parent_for:
                parent_for[key] = parent
    return parent_for, children_for, edge_counts



def _compute_roots(nodes: Set[GenomeKey], parent_for: Mapping[GenomeKey, GenomeKey]) -> List[GenomeKey]:
    roots = sorted((node for node in nodes if node not in parent_for))
    if not roots:
        roots = sorted(nodes)
    return roots


def _compute_depths(
    nodes: Iterable[GenomeKey],
    parent_for: Mapping[GenomeKey, GenomeKey],
) -> Dict[GenomeKey, int]:
    depths: Dict[GenomeKey, int] = {}

    def resolve(pid: GenomeKey) -> int:
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


def _compute_descendants(
    children_for: Mapping[GenomeKey, Set[GenomeKey]],
    roots: Sequence[GenomeKey],
) -> Dict[GenomeKey, int]:
    descendants: Dict[GenomeKey, int] = {}

    def count(node: GenomeKey) -> int:
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


def _assign_root_colors(roots: Sequence[GenomeKey]) -> Dict[GenomeKey, str]:
    color_map: Dict[GenomeKey, str] = {}
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


def _resolve_root(pid: GenomeKey, parent_for: Mapping[GenomeKey, GenomeKey]) -> GenomeKey:
    current = pid
    seen: Set[GenomeKey] = set()
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
    genomes: Mapping[GenomeKey, IndexedGenome],
    output_format: str,
    rankdir: str,
    min_edge_weight: int,
    thumb_dir: Optional[Path],
    thumb_size: int,
) -> Digraph:
    nodes: Set[GenomeKey] = set(genomes.keys())
    parent_for, children_for, edge_counts = _build_parent_maps(genomes)
    roots = _compute_roots(nodes, parent_for)
    depths = _compute_depths(nodes, parent_for)
    descendants = _compute_descendants(children_for, roots)
    root_colors = _assign_root_colors(roots)

    graph = _init_graph(output_format, rankdir)

    resolved_root_cache: Dict[GenomeKey, GenomeKey] = {}
    def _sort_key(item: GenomeKey) -> Tuple[int, str, str]:
        depth = depths.get(item, 0)
        return depth, item[0], item[1]

    for node in sorted(nodes, key=_sort_key):
        record = genomes[node]
        if node in resolved_root_cache:
            root = resolved_root_cache[node]
        else:
            root = _resolve_root(node, parent_for)
            resolved_root_cache[node] = root
        color = root_colors.get(root, "#dddddd")
        fill = _lighten_color(color, factor=0.5)
        depth = depths.get(node, 0)
        desc = descendants.get(node, 0)
        thumbnail_path = _ensure_thumbnail(record, thumb_dir, thumb_size)
        label = _build_node_label(
            record,
            depth=depth,
            desc=desc,
            thumbnail_path=thumbnail_path,
            fill_color=fill,
        )
        attrs = {
            "label": label,
            "shape": "plaintext",
            "color": color,
            "penwidth": "2" if node in roots else "1",
        }
        graph.node(_format_node_id(node), **attrs)

    for (parent, child), weight in sorted(edge_counts.items()):
        if weight < min_edge_weight:
            continue
        edge_attrs = {
            "penwidth": _edge_penwidth(weight),
        }
        if weight > 1:
            edge_attrs["label"] = str(weight)
        graph.edge(_format_node_id(parent), _format_node_id(child), **edge_attrs)

    return graph


def main() -> None:
    args = parse_args()
    pb_dir = args.pb_dir.expanduser().resolve()
    output_base = args.output.expanduser().resolve()
    if output_base.suffix:
        output_base = output_base.with_suffix("")

    thumb_dir: Optional[Path]
    if args.thumb_size > 0:
        if args.thumb_dir is None:
            thumb_dir = output_base.parent / f"{output_base.name}_thumbs"
        else:
            thumb_dir = args.thumb_dir
        thumb_dir = thumb_dir.expanduser().resolve()
    else:
        thumb_dir = None

    pids = _discover_pids(pb_dir)

    requested_limit = _normalize_limit(args.limit)
    target_pids = pids if requested_limit is None else pids[: requested_limit]

    checkpoint_path, checkpoint_limit = _find_best_checkpoint(output_base, requested_limit)
    indexed_genomes: Dict[GenomeKey, IndexedGenome]
    if checkpoint_path is not None:
        print(f"Loading previously indexed genomes from {checkpoint_path}")
        indexed_genomes = np.load(checkpoint_path, allow_pickle=True)["indexed_genomes"].item()
    else:
        indexed_genomes = {}
        checkpoint_limit = 0

    start_idx = len(target_pids) if checkpoint_limit is None else checkpoint_limit or 0
    start_idx = min(start_idx, len(target_pids))
    remaining_pids = target_pids[start_idx:]

    if remaining_pids:
        print("Indexing genomes from Picbreeder lineages (this may take a while)...")
        new_genomes = _load_genomes_single_pass(pb_dir, remaining_pids)
        for key, record in new_genomes.items():
            if key not in indexed_genomes:
                indexed_genomes[key] = record

    limit_label = str(requested_limit)
    index_genomes_pickle_path = output_base.with_suffix(f".limit-{limit_label}_indexed_genomes.npz")
    if remaining_pids or not index_genomes_pickle_path.exists():
        np.savez_compressed(
            index_genomes_pickle_path,
            indexed_genomes=indexed_genomes,
        )

    graph = build_phylogeny_graph(
        genomes=indexed_genomes,
        output_format=args.format,
        rankdir=args.rankdir,
        min_edge_weight=max(1, args.min_edge_weight),
        thumb_dir=thumb_dir,
        thumb_size=args.thumb_size,
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
