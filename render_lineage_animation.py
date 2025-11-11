#!/usr/bin/env python3
"""Render smooth interpolation animations for every link in a publication lineage.

Note that child genomes will never be absent any nodes/edges present in the parent (because of the way Picbreeder
strictly complexifies CPPNs).
"""

from __future__ import annotations

import argparse
import collections
import copy
import gzip
import json
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import re

import neat
from PIL import Image, ImageDraw, ImageFont

from rendering import render_genome_image, render_genome_diagram


PublicationRecord = Dict[str, object]


def _identity_activation(value: float) -> float:
    return value


def _create_blended_activation(
    fn_a: Callable[[float], float],
    fn_b: Callable[[float], float],
    fraction: float,
) -> Callable[[float], float]:
    def _blend(z: float) -> float:
        return (1.0 - fraction) * fn_a(z) + fraction * fn_b(z)

    return _blend


def _load_caption_font(size: int = 8) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _add_caption(image: Image.Image, text: str, font: ImageFont.ImageFont) -> Image.Image:
    caption = (text or "").strip()
    if not caption:
        return image
    draw = ImageDraw.Draw(image)
    try:
        text_bbox = draw.textbbox((0, 0), caption, font=font)
    except AttributeError:
        text_width, text_height = draw.textsize(caption, font=font)
        text_bbox = (0, 0, text_width, text_height)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    padding_y = 12
    caption_height = text_height + (padding_y * 2)
    result = Image.new("RGB", (image.width, image.height + caption_height), color=(0, 0, 0))
    result.paste(image, (0, 0))
    caption_x = max(0, (image.width - text_width) // 2)
    caption_y = image.height + padding_y - text_bbox[1]
    ImageDraw.Draw(result).text((caption_x, caption_y), caption, font=font, fill=(255, 255, 255))
    return result


@dataclass(frozen=True)
class ConnectionStats:
    start_weight: float
    end_weight: float
    start_enabled: bool
    end_enabled: bool


@dataclass(frozen=True)
class OutputActivationStats:
    start_enabled: bool
    end_enabled: bool
    start_names: Tuple[str, ...]
    end_names: Tuple[str, ...]
    start_funcs: Tuple[Callable[[float], float], ...]
    end_funcs: Tuple[Callable[[float], float], ...]


@dataclass
class LoadedGenome:
    config: neat.Config
    genome: neat.DefaultGenome
    generation: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--agent-dir",
        type=Path,
        default=None,
        help="Path to a single agent directory (containing images/, logs/, populations/).",
    )
    target_group.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Path to an experiment directory whose agents/ subdirectory will be processed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for rendered frames and animation (default: <agent-dir>/animations/<archive-id>).",
    )
    parser.add_argument(
        "--publication-index",
        type=int,
        default=-1,
        help="Publication entry to animate (0-based, defaults to the latest entry when negative).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=90,
        help="Number of interpolation steps (frames) between the parent and published genome.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Width/height in pixels for every rendered frame.",
    )
    parser.add_argument(
        "--variant",
        choices=("auto", "color", "gray"),
        default="auto",
        help="Color mode for rendered frames. 'auto' follows the publication's color flag.",
    )
    parser.add_argument(
        "--frame-duration",
        type=int,
        default=20,
        help="Frame duration in milliseconds when composing the GIF animation.",
    )
    parser.add_argument(
        "--save-genome-diagrams",
        action="store_true",
        default=False,
        help="If set, save genome diagram images for each interpolation step.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-render animations even if a lineage_story GIF already exists.",
    )
    return parser.parse_args()


def load_publications(agent_dir: Path) -> List[PublicationRecord]:
    publication_path = agent_dir / "logs" / "publication_history.jsonl"
    if not publication_path.exists():
        raise FileNotFoundError(f"No publication history found under {publication_path}")
    records: List[PublicationRecord] = []
    with publication_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"{publication_path} is empty; nothing to animate.")
    return records


def load_lineage_generations(agent_dir: Path) -> Dict[int, List[int]]:
    lineage_path = agent_dir / "logs" / "lineage.jsonl"
    if not lineage_path.exists():
        raise FileNotFoundError(f"Lineage log missing under {lineage_path}")
    generations: Dict[int, List[int]] = collections.defaultdict(list)
    with lineage_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            key = record.get("genome_key")
            generation = record.get("generation")
            if key is None or generation is None:
                continue
            try:
                genome_key = int(key)
                gen_index = int(generation)
            except (TypeError, ValueError):
                continue
            generations[genome_key].append(gen_index)
    return generations


def discover_checkpoints(populations_dir: Path) -> Dict[int, Path]:
    checkpoints: Dict[int, Path] = {}
    for path in populations_dir.glob("gen_*_checkpoint.pkl.gz"):
        stem = path.stem
        generation = int(stem.split("_")[1])
        checkpoints[generation] = path
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {populations_dir}")
    return checkpoints


def _load_checkpoint(path: Path) -> Tuple[neat.Config, Dict[int, neat.DefaultGenome]]:
    with gzip.open(path, "rb") as handle:
        _, config, population, _, _ = pickle.load(handle)
    return config, population


def resolve_genome(
    genome_key: int,
    candidate_generations: Sequence[int],
    checkpoints: Dict[int, Path],
    checkpoint_cache: Dict[int, Tuple[neat.Config, Dict[int, neat.DefaultGenome]]],
) -> Tuple[neat.Config, neat.DefaultGenome, int]:
    tried = []
    generations = sorted(candidate_generations, reverse=True)
    if not generations:
        generations = sorted(checkpoints.keys(), reverse=True)
    for generation in generations:
        checkpoint_path = checkpoints.get(generation)
        tried.append(generation)
        if generation not in checkpoint_cache:
            checkpoint_cache[generation] = _load_checkpoint(checkpoint_path)
        config, population = checkpoint_cache[generation]
        genome = population.get(genome_key)
        if genome is not None:
            return config, copy.deepcopy(genome), generation
    tried_desc = ", ".join(str(gen) for gen in tried) or "none"
    raise RuntimeError(
        f"Unable to locate genome {genome_key} in the available checkpoints. Generations tried: {tried_desc}"
    )


def fetch_genome(
    genome_key: int,
    lineage_generations: Dict[int, List[int]],
    checkpoints: Dict[int, Path],
    checkpoint_cache: Dict[int, Tuple[neat.Config, Dict[int, neat.DefaultGenome]]],
    loaded_cache: Dict[int, LoadedGenome],
) -> LoadedGenome:
    if genome_key in loaded_cache:
        return loaded_cache[genome_key]
    config, genome, generation = resolve_genome(
        genome_key,
        lineage_generations.get(genome_key),
        checkpoints,
        checkpoint_cache,
    )
    record = LoadedGenome(config=config, genome=genome, generation=generation)
    loaded_cache[genome_key] = record
    return record


def _set_output_activation_state(
    genome: neat.DefaultGenome,
    enabled: bool,
    names: Sequence[str],
    funcs: Sequence[Callable[[float], float]],
) -> None:
    genome._output_activations_enabled = bool(enabled)  # type: ignore[attr-defined]
    if not enabled:
        for attribute in ("_output_activation_names", "_output_activation_funcs"):
            if hasattr(genome, attribute):
                delattr(genome, attribute)
        return
    genome._output_activation_names = list(names)  # type: ignore[attr-defined]
    genome._output_activation_funcs = list(funcs)  # type: ignore[attr-defined]


def build_superset(
    parent: neat.DefaultGenome,
    child: neat.DefaultGenome,
) -> Tuple[
    neat.DefaultGenome,
    Dict[int, Tuple[str, str]],
    Dict[Tuple[int, int], ConnectionStats],
    Optional[OutputActivationStats],
]:
    superset = copy.deepcopy(child)
    node_pairs: Dict[int, Tuple[str, str]] = {}
    for node_key, node_gene in superset.nodes.items():
        parent_node = parent.nodes.get(node_key)
        published_node = child.nodes.get(node_key)
        parent_activation = getattr(parent_node, "activation", getattr(published_node, "activation", "identity"))
        published_activation = getattr(published_node, "activation", parent_activation)
        node_pairs[node_key] = (parent_activation, published_activation)
        if parent_node is not None:
            node_gene.activation = parent_activation
    connection_pairs: Dict[Tuple[int, int], ConnectionStats] = {}
    for key, conn in superset.connections.items():
        parent_conn = parent.connections.get(key)
        published_conn = child.connections.get(key)
        if parent_conn is not None:
            conn.weight = parent_conn.weight
            conn.enabled = parent_conn.enabled
            start_weight = parent_conn.weight
            start_enabled = parent_conn.enabled
        else:
            conn.weight = 0.0
            conn.enabled = False
            start_weight = 0.0
            start_enabled = False
        if published_conn is None:
            end_weight = start_weight
            end_enabled = start_enabled
        else:
            end_weight = published_conn.weight
            end_enabled = published_conn.enabled
        connection_pairs[key] = ConnectionStats(
            start_weight=start_weight,
            end_weight=end_weight,
            start_enabled=start_enabled,
            end_enabled=end_enabled,
        )
    start_funcs_raw = tuple(getattr(parent, "_output_activation_funcs", ()))
    end_funcs_raw = tuple(getattr(child, "_output_activation_funcs", ()))
    count = max(len(start_funcs_raw), len(end_funcs_raw))
    start_enabled = bool(getattr(parent, "_output_activations_enabled", False))
    end_enabled = bool(getattr(child, "_output_activations_enabled", False))
    if count == 0:
        output_stats = None
        return superset, node_pairs, connection_pairs, output_stats

    def _normalize_name(raw: Optional[str], prefix: str, idx: int) -> str:
        base = (raw or "").strip()
        if not base:
            base = f"{prefix}_{idx:02d}"
        return _sanitize_label(base)

    def _normalize_names(raw_names: Sequence[str], prefix: str) -> Tuple[str, ...]:
        normalized: List[str] = []
        for idx in range(count):
            value = raw_names[idx] if idx < len(raw_names) else None
            normalized.append(_normalize_name(value, prefix, idx))
        return tuple(normalized)

    def _normalize_funcs(raw_funcs: Sequence[Callable[[float], float]]) -> Tuple[Callable[[float], float], ...]:
        funcs: List[Callable[[float], float]] = []
        for idx in range(count):
            candidate = raw_funcs[idx] if idx < len(raw_funcs) else None
            if not callable(candidate):
                candidate = _identity_activation
            funcs.append(candidate)  # type: ignore[arg-type]
        return tuple(funcs)

    start_names = _normalize_names(getattr(parent, "_output_activation_names", ()), "start")
    end_names = _normalize_names(getattr(child, "_output_activation_names", ()), "end")
    start_funcs = _normalize_funcs(start_funcs_raw)
    end_funcs = _normalize_funcs(end_funcs_raw)
    output_stats: Optional[OutputActivationStats] = None
    output_stats = OutputActivationStats(
        start_enabled=start_enabled,
        end_enabled=end_enabled,
        start_names=start_names,
        end_names=end_names,
        start_funcs=start_funcs,
        end_funcs=end_funcs,
    )
    _set_output_activation_state(
        superset,
        output_stats.start_enabled,
        output_stats.start_names,
        output_stats.start_funcs,
    )
    return superset, node_pairs, connection_pairs, output_stats


def _sanitize_label(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def _format_agent_caption(agent_dir: Path) -> str:
    name = agent_dir.name
    match = re.search(r"(\d+)", name)
    if match:
        try:
            number = int(match.group(1))
            return f"Agent {number}"
        except ValueError:
            pass
    return f"Agent {name}" if name else "Agent"


def assign_interpolated_values(
    genome: neat.DefaultGenome,
    node_pairs: Dict[int, Tuple[str, str]],
    connection_pairs: Dict[Tuple[int, int], ConnectionStats],
    config: neat.Config,
    step_index: int,
    total_steps: int,
    step_fraction: float,
    activation_cache: Dict[Tuple[str, str, int], str],
    output_activation_stats: Optional[OutputActivationStats] = None,
) -> None:
    defs = config.genome_config.activation_defs
    for key, stats in connection_pairs.items():
        conn = genome.connections[key]
        if stats.start_enabled == stats.end_enabled:
            conn.enabled = stats.start_enabled
            conn.weight = (1.0 - step_fraction) * stats.start_weight + step_fraction * stats.end_weight
        elif stats.start_enabled and not stats.end_enabled:
            conn.enabled = True
            conn.weight = stats.start_weight * (1.0 - step_fraction)
        elif not stats.start_enabled and stats.end_enabled:
            conn.enabled = True
            conn.weight = stats.end_weight * step_fraction
    for node_key, (start_act, end_act) in node_pairs.items():
        node_gene = genome.nodes[node_key]
        if start_act == end_act:
            node_gene.activation = start_act
            continue
        cache_key = (start_act, end_act, step_index)
        if cache_key in activation_cache:
            node_gene.activation = activation_cache[cache_key]
            continue
        start_fn = defs.get(start_act)
        end_fn = defs.get(end_act)
        if start_fn is None or end_fn is None:
            raise RuntimeError(f"Unknown activation functions ({start_act}, {end_act}) encountered.")
        name = f"blend_{_sanitize_label(start_act)}_{_sanitize_label(end_act)}_{step_index:03d}"
        defs.add(name, _create_blended_activation(start_fn, end_fn, step_fraction))
        activation_cache[cache_key] = name
        node_gene.activation = name
    if output_activation_stats is not None:
        total_outputs = len(output_activation_stats.start_funcs)
        if total_outputs == 0:
            _set_output_activation_state(genome, False, (), ())
            return
        last_step_index = total_steps - 1
        if output_activation_stats.start_enabled == output_activation_stats.end_enabled:
            current_enabled = output_activation_stats.start_enabled
        else:
            if step_index == 0:
                current_enabled = output_activation_stats.start_enabled
            elif step_index == last_step_index:
                current_enabled = output_activation_stats.end_enabled
            else:
                current_enabled = True
        if not current_enabled:
            _set_output_activation_state(genome, False, (), ())
            return
        names: List[str] = []
        funcs: List[Callable[[float], float]] = []
        for idx in range(total_outputs):
            start_name = output_activation_stats.start_names[idx]
            end_name = output_activation_stats.end_names[idx]
            start_func = output_activation_stats.start_funcs[idx]
            end_func = output_activation_stats.end_funcs[idx]
            if step_index == 0:
                names.append(start_name)
                funcs.append(start_func)
                continue
            if step_index == last_step_index:
                names.append(end_name)
                funcs.append(end_func)
                continue
            blend_name = (
                f"blend_output_{idx:02d}_"
                f"{_sanitize_label(start_name)}_"
                f"{_sanitize_label(end_name)}_{step_index:03d}"
            )
            names.append(blend_name)
            funcs.append(_create_blended_activation(start_func, end_func, step_fraction))
        _set_output_activation_state(genome, True, names, funcs)


def pick_variant(
    gray_image: Image.Image,
    color_image: Image.Image,
    variant_mode: str,
    color_enabled_flag: bool,
) -> Image.Image:
    if variant_mode == "color":
        return color_image
    if variant_mode == "gray":
        return gray_image
    return color_image if color_enabled_flag else gray_image


def render_frames(
    base_genome: neat.DefaultGenome,
    node_pairs: Dict[int, Tuple[str, str]],
    connection_pairs: Dict[Tuple[int, int], ConnectionStats],
    config: neat.Config,
    steps: int,
    width: int,
    height: int,
    variant_mode: str,
    color_enabled: bool,
    artifact_dir: Optional[Path] = None,
    save_genome_diagrams: bool = False,
    output_activation_stats: Optional[OutputActivationStats] = None,
) -> List[Image.Image]:
    if steps < 2:
        raise ValueError("Interpolation requires at least two steps.")
    frames: List[Image.Image] = []
    activation_cache: Dict[Tuple[str, str, int], str] = {}
    for idx in range(steps):
        fraction = idx / (steps - 1)
        working_genome = copy.deepcopy(base_genome)
        assign_interpolated_values(
            working_genome,
            node_pairs,
            connection_pairs,
            config,
            idx,
            steps,
            fraction,
            activation_cache,
            output_activation_stats,
        )
        gray_image, color_image = render_genome_image(working_genome, config, width, height)
        if save_genome_diagrams:
            if artifact_dir is None:
                raise ValueError("artifact_dir must be provided when save_genome_diagrams is enabled.")
            artifact_dir.mkdir(parents=True, exist_ok=True)
            diagram_path = artifact_dir / f"genome_diagram_{idx:03d}.png"
            render_genome_diagram(working_genome, config, output_stem=diagram_path)
        frame = pick_variant(gray_image, color_image, variant_mode, color_enabled).convert("RGB")
        frames.append(frame)
    return frames


def _load_gif_frames(path: Path) -> List[Image.Image]:
    frames: List[Image.Image] = []
    try:
        with Image.open(path) as gif:
            total_frames = getattr(gif, "n_frames", 1)
            for frame_index in range(total_frames):
                gif.seek(frame_index)
                frame = gif.convert("RGB").copy()
                frames.append(frame)
    except Exception as exc:
        print(f"[warn] Unable to read GIF frames from {path}: {exc}")
    return frames


def _load_transition_frames_from_disk(
    transition_dir: Path,
    steps: int,
) -> Optional[Tuple[List[Image.Image], List[Path]]]:
    frame_paths = [transition_dir / f"frame_{idx:03d}.png" for idx in range(steps)]
    for path in frame_paths:
        if not path.exists():
            return None
    frames: List[Image.Image] = []
    for path in frame_paths:
        try:
            with Image.open(path) as image_obj:
                frames.append(image_obj.convert("RGB").copy())
        except Exception as exc:
            print(f"[warn] Failed to load cached frame {path}: {exc}")
            for frame in frames:
                frame.close()
            return None
    return frames, frame_paths


def stitch_leaf_lineage_stories(
    experiment_dir: Path,
    agent_summaries: Dict[str, Dict[str, object]],
    frame_duration_ms: int,
) -> None:
    archive_dir = experiment_dir / "archive"
    lineage_manifest = archive_dir / "leaf_agent_lineages.json"
    if not lineage_manifest.exists():
        return
    try:
        leaf_data = json.loads(lineage_manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] Failed to load leaf agent lineages from {lineage_manifest}: {exc}")
        return
    if not isinstance(leaf_data, dict):
        print(f"[warn] Unexpected data format in {lineage_manifest}; expected object mapping.")
        return

    output_dir = archive_dir / "leaf_lineage_stories"
    output_dir.mkdir(parents=True, exist_ok=True)

    for leaf_id, agent_sequence in leaf_data.items():
        if not isinstance(agent_sequence, list) or not agent_sequence:
            continue
        combined_frames: List[Image.Image] = []
        component_frame_dirs: List[str] = []
        component_gifs: List[str] = []
        for raw_agent_id in agent_sequence:
            agent_id = str(raw_agent_id)
            summary = agent_summaries.get(agent_id)
            if summary is None:
                print(f"[warn] Missing lineage summary for agent {agent_id}; skipping for leaf {leaf_id}.")
                continue

            frames_dir_raw = summary.get("combined_frames_dir")
            frames_dir: Optional[Path] = None
            frame_paths: List[Path] = []
            if frames_dir_raw:
                frames_dir = Path(str(frames_dir_raw)).expanduser()
                if not frames_dir.is_absolute():
                    agent_dir_raw = summary.get("agent_dir")
                    if agent_dir_raw:
                        frames_dir = Path(str(agent_dir_raw)).expanduser() / frames_dir
                    else:
                        frames_dir = (experiment_dir / "agents" / agent_id / frames_dir)
                if frames_dir.exists():
                    frame_paths = sorted(frames_dir.glob("combined_frame_*.png"))

            if frame_paths:
                for frame_path in frame_paths:
                    try:
                        with Image.open(frame_path) as image_obj:
                            combined_frames.append(image_obj.convert("RGB").copy())
                    except Exception as exc:
                        print(f"[warn] Failed to load frame {frame_path} for agent {agent_id}: {exc}")
                        continue
                component_frame_dirs.append(str(frames_dir.resolve()))
                continue

            combined_path_raw = summary.get("combined_animation")
            if not combined_path_raw:
                continue
            combined_path = Path(str(combined_path_raw)).expanduser()
            if not combined_path.is_absolute():
                agent_dir_raw = summary.get("agent_dir")
                if agent_dir_raw:
                    combined_path = Path(str(agent_dir_raw)).expanduser() / combined_path
                else:
                    combined_path = (experiment_dir / "agents" / agent_id / combined_path)
            if not combined_path.exists():
                print(f"[warn] Lineage GIF for agent {agent_id} not found at {combined_path}; skipping.")
                continue
            frames = _load_gif_frames(combined_path)
            if not frames:
                continue
            for frame in frames:
                combined_frames.append(frame)
            component_gifs.append(str(combined_path))

        if not combined_frames:
            continue

        output_path = output_dir / f"{leaf_id}_lineage_story.gif"
        first_frame = combined_frames[0]
        remaining_frames = combined_frames[1:]
        first_frame.save(
            output_path,
            save_all=True,
            append_images=remaining_frames,
            loop=0,
            format="GIF",
            duration=frame_duration_ms,
        )
        for frame in combined_frames:
            frame.close()

        metadata = {
            "leaf_entry_id": str(leaf_id),
            "agents": [str(agent) for agent in agent_sequence],
            "component_frame_dirs": component_frame_dirs,
            "component_gifs": component_gifs,
            "output_path": str(output_path),
        }
        metadata_path = output_dir / f"{leaf_id}_lineage_story.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Stitched lineage story for {leaf_id} at {output_path}")


def ensure_transition_keys(publication: PublicationRecord) -> List[int]:
    genome_key = int(publication["genome_key"])
    ancestor_keys = [int(key) for key in publication.get("ancestor_genome_keys", [])]
    parent_keys = [int(key) for key in (publication.get("parent_genome_keys") or [])]
    if not parent_keys:
        raise RuntimeError(f"Publication {genome_key} is missing parent information.")
    parent_key = parent_keys[0]
    if not ancestor_keys or ancestor_keys[-1] != parent_key:
        ancestor_keys.append(parent_key)
    chain = ancestor_keys + [genome_key]
    if len(chain) < 2:
        raise RuntimeError(f"Insufficient lineage length for genome {genome_key}; expected at least one parent.")
    return chain


def render_transition(
    transition_index: int,
    parent: LoadedGenome,
    child: LoadedGenome,
    variant_mode: str,
    color_flag: bool,
    steps: int,
    image_size: int,
    frame_duration: int,
    transition_dir: Path,
    save_genome_diagrams: bool = False,
    persist_all_frames: bool = False,
    overwrite: bool = False,
) -> Tuple[Dict[str, object], List[Image.Image]]:
    transition_dir.mkdir(parents=True, exist_ok=True)
    desired_color_enabled = color_flag if variant_mode == "auto" else (variant_mode == "color")
    metadata_path = transition_dir / "transition_metadata.json"
    cached_metadata: Optional[Dict[str, object]] = None
    if persist_all_frames and not overwrite and metadata_path.exists():
        try:
            cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] Failed to parse existing transition metadata in {transition_dir}: {exc}")
            cached_metadata = None

    if persist_all_frames and not overwrite and cached_metadata is not None:
        metadata_matches = (
            cached_metadata.get("transition_index") == transition_index
            and cached_metadata.get("parent_genome_key") == parent.genome.key
            and cached_metadata.get("child_genome_key") == child.genome.key
            and cached_metadata.get("steps") == steps
            and cached_metadata.get("image_size") == image_size
            and cached_metadata.get("variant") == variant_mode
            and cached_metadata.get("color_enabled") == desired_color_enabled
            and cached_metadata.get("frame_duration_ms") == frame_duration
        )
        cached_frames: Optional[Tuple[List[Image.Image], List[Path]]] = None
        if metadata_matches:
            cached_frames = _load_transition_frames_from_disk(transition_dir, steps)
        if cached_frames is not None:
            frames_loaded, frame_paths = cached_frames
            metadata = dict(cached_metadata)
            metadata["persisted_all_frames"] = True
            metadata["frames_reused"] = True
            metadata["frame_paths"] = [str(path) for path in frame_paths]
            metadata["color_enabled"] = desired_color_enabled
            metadata.setdefault("output_dir", str(transition_dir))
            metadata["transition_index"] = transition_index
            metadata["parent_genome_key"] = parent.genome.key
            metadata["child_genome_key"] = child.genome.key
            metadata["parent_generation"] = parent.generation
            metadata["child_generation"] = child.generation
            metadata["steps"] = steps
            metadata["image_size"] = image_size
            metadata["variant"] = variant_mode
            metadata.pop("animation", None)
            metadata.setdefault("start_frame", str(frame_paths[0]))
            metadata.setdefault("end_frame", str(frame_paths[-1]))
            metadata.setdefault("frame_duration_ms", frame_duration)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return metadata, frames_loaded

    output_activation_stats: Optional[OutputActivationStats]
    superset_genome, node_pairs, connection_pairs, output_activation_stats = build_superset(parent.genome, child.genome)
    diagram_dir = transition_dir / "genome_diagrams" if save_genome_diagrams else None
    frames = render_frames(
        superset_genome,
        node_pairs,
        connection_pairs,
        child.config,
        steps,
        image_size,
        image_size,
        variant_mode,
        desired_color_enabled,
        diagram_dir,
        save_genome_diagrams=save_genome_diagrams,
        output_activation_stats=output_activation_stats,
    )
    frame_paths: List[Path] = []
    start_path = transition_dir / "frame_000.png"
    end_path = transition_dir / f"frame_{steps - 1:03d}.png"
    if persist_all_frames:
        for frame_idx, frame in enumerate(frames):
            frame_path = transition_dir / f"frame_{frame_idx:03d}.png"
            frame.save(frame_path)
            frame_paths.append(frame_path)
        if frame_paths:
            start_path = frame_paths[0]
            end_path = frame_paths[-1]
    else:
        frames[0].save(start_path)
        frames[-1].save(end_path)
    metadata = {
        "transition_index": transition_index,
        "output_dir": str(transition_dir),
        "parent_genome_key": parent.genome.key,
        "child_genome_key": child.genome.key,
        "parent_generation": parent.generation,
        "child_generation": child.generation,
        "steps": steps,
        "image_size": image_size,
        "variant": variant_mode,
    "color_enabled": desired_color_enabled,
        "frame_duration_ms": frame_duration,
        "start_frame": str(start_path),
        "end_frame": str(end_path),
        "persisted_all_frames": persist_all_frames,
        "frames_reused": False,
    }
    if frame_paths:
        metadata["frame_paths"] = [str(path) for path in frame_paths]
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata, frames


def render_agent_lineage(
    agent_dir: Path,
    args: argparse.Namespace,
    output_override: Optional[Path] = None,
    append_archive_to_override: bool = False,
) -> Path:
    agent_dir = agent_dir.expanduser().resolve()
    if not agent_dir.exists():
        raise FileNotFoundError(f"Agent directory {agent_dir} does not exist.")
    populations_dir = agent_dir / "populations"
    checkpoints = discover_checkpoints(populations_dir)
    lineage_generations = load_lineage_generations(agent_dir)
    publications = load_publications(agent_dir)

    pub_index = args.publication_index
    if pub_index < 0:
        pub_index = len(publications) + pub_index
    if not (0 <= pub_index < len(publications)):
        raise IndexError(
            f"Publication index {args.publication_index} is out of range (found {len(publications)} entries)."
        )
    publication = publications[pub_index]
    genome_key = int(publication["genome_key"])
    chain_keys = ensure_transition_keys(publication)

    checkpoint_cache: Dict[int, Tuple[neat.Config, Dict[int, neat.DefaultGenome]]] = {}
    loaded_cache: Dict[int, LoadedGenome] = {}
    variant_mode = args.variant
    color_flag = bool(publication.get("color_enabled", False))

    archive_entry_id = str(publication.get("archive_entry_id") or f"genome_{genome_key}")
    base_output = output_override.expanduser().resolve() if output_override is not None else None
    if base_output is not None:
        output_dir = base_output / archive_entry_id if append_archive_to_override else base_output
    else:
        output_dir = agent_dir / "animations" / archive_entry_id
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_caption = _format_agent_caption(agent_dir)
    combined_output_path = output_dir / f"{archive_entry_id}_lineage_story.gif"
    summary_path = output_dir / "animation_summary.json"
    if not args.overwrite and combined_output_path.exists():
        print(f"[skip] {combined_output_path} exists; use --overwrite to regenerate.")
        return summary_path if summary_path.exists() else combined_output_path

    publication_info: Dict[int, Tuple[PublicationRecord, int]] = {}
    for record_index, record in enumerate(publications[: pub_index + 1]):
        try:
            key = int(record["genome_key"])
        except (KeyError, TypeError, ValueError):
            continue
        publication_info[key] = (record, record_index)

    transition_metadata: List[Dict[str, object]] = []
    combined_frames: List[Image.Image] = []
    combined_frames_dir = output_dir / "combined_frames"
    if combined_frames_dir.exists() and args.overwrite:
        for existing_frame in sorted(combined_frames_dir.glob("*.png")):
            try:
                existing_frame.unlink()
            except OSError:
                continue
    caption_font: Optional[ImageFont.ImageFont] = None
    for idx in range(1, len(chain_keys)):
        parent_key = chain_keys[idx - 1]
        child_key = chain_keys[idx]
        parent = fetch_genome(
            parent_key,
            lineage_generations,
            checkpoints,
            checkpoint_cache,
            loaded_cache,
        )
        child = fetch_genome(
            child_key,
            lineage_generations,
            checkpoints,
            checkpoint_cache,
            loaded_cache,
        )
        transition_dir = output_dir / f"{idx:02d}_{parent_key:06d}_to_{child_key:06d}"
        os.makedirs(transition_dir, exist_ok=True)
        parent_image_path = transition_dir / "parent_genome.png"
        child_image_path = transition_dir / "child_genome.png"
        if args.overwrite or not parent_image_path.exists() or not child_image_path.exists():
            gray_parent_image, color_parent_image = render_genome_image(
                parent.genome,
                parent.config,
                args.image_size,
                args.image_size,
            )
            gray_child_image, color_child_image = render_genome_image(
                child.genome,
                child.config,
                args.image_size,
                args.image_size,
            )
            if color_flag:
                color_parent_image.save(parent_image_path)
                color_child_image.save(child_image_path)
            else:
                gray_parent_image.save(parent_image_path)
                gray_child_image.save(child_image_path)
            for image_obj in (
                gray_parent_image,
                color_parent_image,
                gray_child_image,
                color_child_image,
            ):
                try:
                    image_obj.close()
                except Exception:
                    continue
        metadata, frames = render_transition(
            idx,
            parent,
            child,
            variant_mode,
            color_flag,
            args.steps,
            args.image_size,
            args.frame_duration,
            transition_dir,
            save_genome_diagrams=args.save_genome_diagrams,
            persist_all_frames=True,
            overwrite=args.overwrite,
        )
        transition_metadata.append(metadata)
        caption_text = ""
        publication_entry = publication_info.get(child.genome.key)
        if publication_entry is not None:
            record, _ = publication_entry
            title_value = str(record.get("title", "")).strip()
            if title_value:
                caption_text = title_value
        if not caption_text:
            caption_text = f"Generation {child.generation}"
        if caption_font is None:
            caption_font = _load_caption_font()
        for frame in frames:
            frame_rgb = frame.convert("RGB")
            full_caption = f"{agent_caption}\n{caption_text}" if caption_text else agent_caption
            annotated_frame = _add_caption(frame_rgb, full_caption, caption_font)
            combined_frames.append(annotated_frame)
            frame.close()
            frame_rgb.close()
        if metadata.get("frames_reused"):
            print(f"[transition {idx:02d}] Reused cached frames in {transition_dir}")
        else:
            print(f"[transition {idx:02d}] Rendered frames in {transition_dir}")

    if args.save_genome_diagrams:
        immediate_parent = fetch_genome(
            chain_keys[-2],
            lineage_generations,
            checkpoints,
            checkpoint_cache,
            loaded_cache,
        )
        published = fetch_genome(
            chain_keys[-1],
            lineage_generations,
            checkpoints,
            checkpoint_cache,
            loaded_cache,
        )
        parent_diag = output_dir / "parent_genome_diagram.png"
        render_genome_diagram(immediate_parent.genome, immediate_parent.config, output_stem=parent_diag)
        published_diag = output_dir / "published_genome_diagram.png"
        render_genome_diagram(published.genome, published.config, output_stem=published_diag)

    combined_generated = False
    if combined_frames:
        combined_frames_dir.mkdir(parents=True, exist_ok=True)
        for frame_index, frame in enumerate(combined_frames):
            frame_path = combined_frames_dir / f"combined_frame_{frame_index:04d}.png"
            frame.save(frame_path)

        frame_duration_ms = max(1, int(args.frame_duration))
        per_frame_durations = [frame_duration_ms] * len(combined_frames)

        combined_frames[0].save(
            combined_output_path,
            save_all=True,
            append_images=combined_frames[1:],
            loop=0,
            format="GIF",
            duration=
            per_frame_durations
        )
        combined_generated = True
        print(f"Saved combined lineage animation to {combined_output_path}")
        for frame in combined_frames:
            frame.close()
        combined_frames.clear()

    summary = {
        "agent_dir": str(agent_dir),
        "publication_index": pub_index,
        "archive_entry_id": archive_entry_id,
        "published_genome_key": genome_key,
        "lineage_keys": chain_keys,
        "steps_per_transition": args.steps,
        "image_size": args.image_size,
        "variant": args.variant,
        "color_flag": color_flag,
        "frame_duration_ms": args.frame_duration,
        "transitions": transition_metadata,
        "combined_animation": str(combined_output_path) if combined_generated else None,
        "combined_frames_dir": str(combined_frames_dir.resolve()) if combined_generated else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote lineage summary to {summary_path}")
    return summary_path


def main() -> None:
    args = parse_args()
    output_root = args.output_dir.expanduser().resolve() if args.output_dir else None
    if args.agent_dir is not None:
        render_agent_lineage(args.agent_dir, args, output_override=output_root, append_archive_to_override=False)
        return

    experiment_dir = args.experiment_dir.expanduser().resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory {experiment_dir} does not exist.")

    agents_root = experiment_dir / "agents"
    search_root = agents_root if agents_root.is_dir() else experiment_dir
    candidate_agents = [
        path
        for path in sorted(search_root.iterdir())
        if path.is_dir() and (path / "logs" / "publication_history.jsonl").exists()
    ]
    if not candidate_agents:
        raise RuntimeError(f"No agent directories with publication logs found under {search_root}")

    failures: List[Tuple[Path, Exception]] = []
    agent_summaries: Dict[str, Dict[str, object]] = {}
    for agent_dir in candidate_agents:
        output_override = (output_root / agent_dir.name) if output_root is not None else None
        print(f"=== Rendering {agent_dir} ===")
        try:
            summary_path = render_agent_lineage(
                agent_dir,
                args,
                output_override=output_override,
                append_archive_to_override=output_root is not None,
            )
        except Exception as exc:
            failures.append((agent_dir, exc))
            continue

        try:
            summary_data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] Unable to load lineage summary from {summary_path}: {exc}")
            continue

        agent_id = agent_dir.name
        agent_record: Dict[str, object] = {
            "summary_path": str(summary_path),
            "agent_dir": str(agent_dir),
            "combined_animation": summary_data.get("combined_animation"),
            "archive_entry_id": summary_data.get("archive_entry_id"),
            "combined_frames_dir": summary_data.get("combined_frames_dir"),
        }
        agent_summaries[agent_id] = agent_record

    stitch_leaf_lineage_stories(experiment_dir, agent_summaries, args.frame_duration)

    if failures:
        failed_list = "; ".join(f"{path}: {exc}" for path, exc in failures)
        raise RuntimeError(f"Failed to render some agents: {failed_list}")


if __name__ == "__main__":
    main()
