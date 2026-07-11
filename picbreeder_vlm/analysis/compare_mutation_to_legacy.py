#!/usr/bin/env python3
"""Contrast Picbreeder reproduction against legacy offspring snapshots.

For every legacy Picbreeder lineage (``pid`` directory) this script:
    1. Loads the final published genome from ``rep.zip``.
    2. Finds the recorded parent genome inside the lineage history.
    3. Converts both parent and child into ``PicbreederGenome`` instances.
    4. Generates N offspring from the parent using the modern NEAT mutation
       pipeline in ``picbreeder_reproduction.py``.
    5. Measures which simulated offspring is closest to the legacy child.
    6. Sweeps mutation strength from 0.0 to 1.0 (step 0.1) with N samples per level.
    7. Optionally renders parent/child/mutant images and compares image-space similarity.

Results are printed to stdout and can optionally be written to JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from picbreeder_vlm._paths import FER_ROOT, ensure_fer_importable
ensure_fer_importable()
from fer.src.lineage_utils import recursive_parse_all_genomes  # type: ignore[import]
from fer.src.picbreeder_util import load_zip_xml_as_dict  # type: ignore[import]
from picbreeder_vlm.core.picbreeder_reproduction import PicbreederReproduction
from fer.src.save_lineage_figures_neatpython import _dict_to_legacy_genome  # type: ignore[import]
from tools.render_legacy_genome import (  # type: ignore[import]
    _build_neat_config,
    _infer_scheme,
    _legacy_to_picbreeder_genome,
)
from picbreeder_vlm.core.rendering import render_genome_image


@dataclass
class LineageResult:
    pid: str
    child_key: str
    parent_key: str
    scheme: str
    parent_distance: float
    best_distance: float
    best_sample_index: Optional[int]
    distances: List[float]
    mutation_strengths: List[float]
    image_log_dir: Optional[str]
    best_image_sample_index: Optional[int]
    best_image_mse: Optional[float]
    best_image_similarity: Optional[float]
    best_genome_image_mse: Optional[float]
    best_genome_image_similarity: Optional[float]
    best_genome_mutation_strength: Optional[float]
    best_image_mutation_strength: Optional[float]
    notes: List[str]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        # Make floats JSON friendly, keep consistent formatting downstream.
        payload["parent_distance"] = float(self.parent_distance)
        payload["best_distance"] = float(self.best_distance)
        payload["distances"] = [float(d) for d in self.distances]
        payload["mutation_strengths"] = [float(s) for s in self.mutation_strengths]
        if self.best_image_sample_index is not None:
            payload["best_image_sample_index"] = int(self.best_image_sample_index)
        if self.best_image_mse is not None:
            payload["best_image_mse"] = float(self.best_image_mse)
        if self.best_image_similarity is not None:
            payload["best_image_similarity"] = float(self.best_image_similarity)
        if self.best_genome_image_mse is not None:
            payload["best_genome_image_mse"] = float(self.best_genome_image_mse)
        if self.best_genome_image_similarity is not None:
            payload["best_genome_image_similarity"] = float(self.best_genome_image_similarity)
        if self.best_genome_mutation_strength is not None:
            payload["best_genome_mutation_strength"] = float(self.best_genome_mutation_strength)
        if self.best_image_mutation_strength is not None:
            payload["best_image_mutation_strength"] = float(self.best_image_mutation_strength)
        return payload


def identifier_to_key(identifier: Dict[str, object] | None, default_branch: str) -> Optional[str]:
    if not isinstance(identifier, dict):
        return None
    local = identifier.get("@id")
    if local is None:
        return None
    branch = str(identifier.get("@branch", "") or default_branch)
    return f"{branch}:{local}" if branch else str(local)


def extract_parent_keys(parents_block: Dict[str, object] | None, default_branch: str) -> List[str]:
    if not isinstance(parents_block, dict):
        return []
    identifiers = parents_block.get("identifier")
    if identifiers is None:
        return []
    if isinstance(identifiers, list):
        raw_items: Iterable[Dict[str, object]] = identifiers
    else:
        raw_items = [identifiers]  # type: ignore[list-item]
    keys: List[str] = []
    for entry in raw_items:
        key = identifier_to_key(entry, default_branch)
        if key is not None:
            keys.append(key)
    return keys


def split_key(key: str) -> Tuple[str, str]:
    if ":" not in key:
        raise ValueError(f"Malformed genome key '{key}'")
    branch, local = key.split(":", 1)
    return branch, local


class GenomeRepository:
    """Lazy loader for legacy Picbreeder genomes grouped by pid."""

    def __init__(self, pb_dir: Path) -> None:
        self.pb_dir = pb_dir
        self._raw_cache: Dict[str, Dict[str, Tuple[Dict[str, object], int, str]]] = {}
        self._legacy_cache: Dict[str, object] = {}

    def _ensure_pid(self, pid: str) -> None:
        pid = str(pid)
        if pid in self._raw_cache:
            return
        mapping: Dict[str, Tuple[Dict[str, object], int, str]] = {}
        pid_dir = self.pb_dir / pid
        if not pid_dir.is_dir():
            self._raw_cache[pid] = mapping
            return
        index = 0
        for zip_path in sorted(pid_dir.glob("*.zip")):
            if zip_path.name in {"main.zip", "rep.zip"}:
                continue
            try:
                payload = load_zip_xml_as_dict(str(zip_path))
            except FileNotFoundError:
                continue
            genome_block = payload.get("genome")
            if not isinstance(genome_block, dict):
                continue

            generation_blocks: List[object] = []
            maybe_block: Optional[Dict[str, object]] = genome_block
            while isinstance(maybe_block, dict):
                if "generation" in maybe_block:
                    generation_blocks.append(maybe_block["generation"])
                maybe_block = maybe_block.get("storage")  # type: ignore[assignment]

            for generation in generation_blocks:
                for genome_dict in recursive_parse_all_genomes(generation):
                    if not isinstance(genome_dict, dict):
                        continue
                    key = identifier_to_key(genome_dict.get("identifier"), pid)
                    if key is None or key in mapping:
                        continue
                    mapping[key] = (genome_dict, index, pid)
                    index += 1
        self._raw_cache[pid] = mapping

    def get_raw(self, key: str) -> Optional[Tuple[Dict[str, object], int, str]]:
        branch, _ = split_key(key)
        self._ensure_pid(branch)
        record = self._raw_cache.get(branch, {}).get(key)
        if record is not None:
            return record
        return None

    def get_legacy(self, key: str) -> Optional[object]:
        cached = self._legacy_cache.get(key)
        if cached is not None:
            return cached
        raw = self.get_raw(key)
        if raw is None:
            return None
        genome_dict, index, pid = raw
        legacy = _dict_to_legacy_genome(genome_dict, pid=pid, index=index)
        self._legacy_cache[key] = legacy
        return legacy


def genome_distance(expected, candidate, config) -> float:
    missing_node_penalty = 5.0
    extra_node_penalty = 5.0
    missing_conn_penalty = 3.0
    extra_conn_penalty = 3.0
    toggle_penalty = 1.0
    bias_scale = 0.5
    weight_scale = 1.0

    distance = 0.0
    expected_nodes = expected.nodes
    candidate_nodes = candidate.nodes
    input_keys = set(config.genome_config.input_keys)

    for key, exp_node in expected_nodes.items():
        cand_node = candidate_nodes.get(key)
        if cand_node is None:
            distance += missing_node_penalty
            continue
        if key not in input_keys:
            distance += bias_scale * abs(exp_node.bias - cand_node.bias)
        if exp_node.activation != cand_node.activation:
            distance += toggle_penalty
    for key in candidate_nodes:
        if key not in expected_nodes:
            distance += extra_node_penalty

    expected_conns = expected.connections
    candidate_conns = candidate.connections
    for key, exp_conn in expected_conns.items():
        cand_conn = candidate_conns.get(key)
        if cand_conn is None:
            distance += missing_conn_penalty
            continue
        distance += weight_scale * abs(exp_conn.weight - cand_conn.weight)
        if exp_conn.enabled != cand_conn.enabled:
            distance += toggle_penalty
    for key in candidate_conns:
        if key not in expected_conns:
            distance += extra_conn_penalty

    return distance


def compute_base_seed(global_seed: int, pid: str) -> int:
    try:
        pid_value = int(pid)
    except ValueError:
        pid_value = sum(ord(ch) for ch in pid)
    return (global_seed * 1_000_003 + pid_value) & 0xFFFFFFFF


def apply_mutation_strength(config, strength: float) -> None:
    strength = float(strength)
    setattr(config, "picbreeder_mutation_strength", strength)
    if hasattr(config, "genome_config"):
        setattr(config.genome_config, "picbreeder_mutation_strength", strength)


def analyze_pid(
    pid: str,
    repo: GenomeRepository,
    samples: int,
    seed: int,
    image_dir: Optional[Path] = None,
    image_resolution: int = 256,
) -> Tuple[Optional[LineageResult], List[str]]:
    notes: List[str] = []
    pid_dir = repo.pb_dir / pid
    rep_path = pid_dir / "rep.zip"
    if not rep_path.exists():
        notes.append("missing rep.zip")
        return None, notes
    payload = load_zip_xml_as_dict(str(rep_path))
    child_dict = payload.get("genome")
    if not isinstance(child_dict, dict):
        notes.append("rep.zip did not contain a genome block")
        return None, notes

    child_key = identifier_to_key(child_dict.get("identifier"), pid) or f"{pid}:final"
    legacy_child = _dict_to_legacy_genome(child_dict, pid=pid, index=0)

    parent_candidates = extract_parent_keys(child_dict.get("parents"), pid)
    if not parent_candidates:
        notes.append("final genome has no recorded parent")
        return None, notes
    parent_key = parent_candidates[0]
    if len(parent_candidates) > 1:
        notes.append(f"multiple parents recorded; using {parent_key}")
    legacy_parent = repo.get_legacy(parent_key)
    if legacy_parent is None:
        notes.append(f"parent {parent_key} not found in lineage data")
        return None, notes

    scheme = _infer_scheme(legacy_child)
    config = _build_neat_config()

    parent_genome = _legacy_to_picbreeder_genome(legacy_parent, config)
    child_genome = _legacy_to_picbreeder_genome(legacy_child, config)

    parent_distance = genome_distance(child_genome, parent_genome, config)

    mutation_repeats = max(0, int(getattr(config.reproduction_config, "mutation_repeats", 0)))
    base_seed = compute_base_seed(seed, pid)

    image_output_dir: Optional[Path] = None
    parent_image_path: Optional[Path] = None
    child_image_path: Optional[Path] = None
    best_genome_path: Optional[Path] = None
    best_genome_image_mse: Optional[float] = None
    best_genome_image_similarity: Optional[float] = None
    best_genome_mutation_strength: Optional[float] = None
    best_image_mse_value: float = math.inf
    best_image_similarity_value: Optional[float] = None
    best_image_index: Optional[int] = None
    best_image_path: Optional[Path] = None
    best_image_mutation_strength: Optional[float] = None

    _, child_color_img = render_genome_image(child_genome, config, image_resolution, image_resolution)
    child_color_array = np.asarray(child_color_img, dtype=np.float32) / 255.0

    if image_dir is not None:
        image_output_dir = image_dir / pid
        image_output_dir.mkdir(parents=True, exist_ok=True)
        _, parent_color_img = render_genome_image(parent_genome, config, image_resolution, image_resolution)
        parent_image_path = image_output_dir / "0_parent.png"
        child_image_path = image_output_dir / "1_child.png"
        parent_color_img.save(parent_image_path)
        child_color_img.save(child_image_path)

    original_strength = getattr(config.genome_config, "picbreeder_mutation_strength", 0.5)
    strength_values = [round(i / 10.0, 2) for i in range(0, 11)]
    samples_per_strength = max(1, samples)

    distances: List[float] = []
    mutation_strength_log: List[float] = []
    best_distance = math.inf
    best_index: Optional[int] = None

    sample_index_global = 0
    for strength in strength_values:
        apply_mutation_strength(config, strength)
        for _ in range(samples_per_strength):
            random.seed(base_seed + sample_index_global)
            clone_key = 1_000_000 + sample_index_global
            clone = PicbreederReproduction._clone_genome(parent_genome, clone_key)
            clone.mutate(config.genome_config)
            for _ in range(mutation_repeats):
                clone.mutate(config.genome_config)
            distance = genome_distance(child_genome, clone, config)
            distances.append(distance)
            mutation_strength_log.append(strength)

            _, clone_color_img = render_genome_image(clone, config, image_resolution, image_resolution)
            clone_color_array_current = np.asarray(clone_color_img, dtype=np.float32) / 255.0

            mutant_path: Optional[Path] = None
            if image_output_dir is not None:
                mutant_path = image_output_dir / f"3_mutant_{sample_index_global:04d}.png"
                clone_color_img.save(mutant_path)

            diff = child_color_array - clone_color_array_current
            image_mse = float(np.mean(diff * diff))
            image_similarity = float(max(0.0, min(1.0, 1.0 - image_mse)))

            if distance < best_distance:
                best_distance = distance
                best_index = sample_index_global
                best_genome_path = mutant_path
                best_genome_image_mse = image_mse
                best_genome_image_similarity = image_similarity
                best_genome_mutation_strength = strength

            if image_mse < best_image_mse_value:
                best_image_mse_value = image_mse
                best_image_similarity_value = image_similarity
                best_image_index = sample_index_global
                best_image_path = mutant_path
                best_image_mutation_strength = strength

            sample_index_global += 1

            if math.isclose(distance, 0.0, abs_tol=1e-9):
                break

        if math.isclose(best_distance, 0.0, abs_tol=1e-9):
            break

    apply_mutation_strength(config, original_strength)

    if math.isinf(best_image_mse_value):
        best_image_mse = None
        best_image_similarity = None
    else:
        best_image_mse = float(best_image_mse_value)
        best_image_similarity = best_image_similarity_value

    stats_payload = {
        "pid": pid,
        "parent_key": parent_key,
        "child_key": child_key,
        "scheme": scheme,
        "samples_requested": int(len(distances)),
        "samples_per_strength": int(samples_per_strength),
        "best_sample_index": int(best_index) if best_index is not None else None,
        "parent_distance": float(parent_distance),
        "best_distance": float(best_distance) if math.isfinite(best_distance) else None,
        "best_genome_image_mse": float(best_genome_image_mse) if best_genome_image_mse is not None else None,
        "best_genome_image_similarity": float(best_genome_image_similarity) if best_genome_image_similarity is not None else None,
        "best_image_sample_index": int(best_image_index) if best_image_index is not None else None,
        "best_image_mse": float(best_image_mse) if best_image_mse is not None else None,
        "best_image_similarity": float(best_image_similarity) if best_image_similarity is not None else None,
        "best_genome_mutation_strength": float(best_genome_mutation_strength) if best_genome_mutation_strength is not None else None,
        "best_image_mutation_strength": float(best_image_mutation_strength) if best_image_mutation_strength is not None else None,
        "mutation_strengths": [float(s) for s in mutation_strength_log],
        "strength_values": [float(v) for v in strength_values],
        "distances": [float(d) for d in distances],
    }

    best_genome_label_path: Optional[Path] = None
    best_genome_copy_path: Optional[Path] = None
    best_image_label_path: Optional[Path] = None
    best_image_copy_path: Optional[Path] = None

    if image_output_dir is not None:
        stats_payload["image_log_dir"] = str(image_output_dir)
        if parent_image_path is not None:
            stats_payload["parent_image"] = str(parent_image_path)
        if child_image_path is not None:
            stats_payload["child_image"] = str(child_image_path)

        if best_genome_path is not None:
            best_genome_label_path = image_output_dir / "2_best_genome.png"
            if best_genome_path != best_genome_label_path:
                shutil.copyfile(best_genome_path, best_genome_label_path)
            else:
                best_genome_label_path = best_genome_path

            legacy_label_path = image_output_dir / "2_best_mutant.png"
            if legacy_label_path != best_genome_label_path:
                shutil.copyfile(best_genome_label_path, legacy_label_path)

            best_genome_copy_path = image_output_dir / "best_mutant_genome.png"
            if best_genome_copy_path != best_genome_label_path:
                shutil.copyfile(best_genome_label_path, best_genome_copy_path)

            best_mutant_alias_path = image_output_dir / "best_mutant.png"
            if best_mutant_alias_path != best_genome_label_path:
                shutil.copyfile(best_genome_label_path, best_mutant_alias_path)

            stats_payload["best_mutant_genome_image"] = str(best_genome_label_path)
            stats_payload["best_mutant_genome_copy"] = str(best_genome_copy_path)
            stats_payload["legacy_best_mutant_image"] = str(legacy_label_path)
            stats_payload["best_mutant_alias"] = str(best_mutant_alias_path)

        if best_image_path is not None:
            best_image_label_path = image_output_dir / "2_best_image.png"
            if best_image_path != best_image_label_path:
                shutil.copyfile(best_image_path, best_image_label_path)
            else:
                best_image_label_path = best_image_path

            best_image_copy_path = image_output_dir / "best_mutant_image.png"
            if best_image_copy_path != best_image_label_path:
                shutil.copyfile(best_image_label_path, best_image_copy_path)

            stats_payload["best_mutant_image"] = str(best_image_label_path)
            stats_payload["best_mutant_image_copy"] = str(best_image_copy_path)

        stats_path = image_output_dir / "stats.json"
        stats_path.write_text(json.dumps(stats_payload, indent=2, sort_keys=True))

        notes.append(f"images saved to {image_output_dir}")
        if best_genome_label_path is not None:
            notes.append(f"best genome-space mutant image: {best_genome_label_path}")
        if best_image_label_path is not None:
            notes.append(f"best image-space mutant image: {best_image_label_path}")
        notes.append(f"stats saved to {stats_path}")

    if best_genome_mutation_strength is not None:
        notes.append(f"best genome strength: {best_genome_mutation_strength:.2f}")
    if best_image_mutation_strength is not None:
        notes.append(f"best image strength: {best_image_mutation_strength:.2f}")

    return (
        LineageResult(
            pid=pid,
            child_key=child_key,
            parent_key=parent_key,
            scheme=scheme,
            parent_distance=parent_distance,
            best_distance=best_distance if math.isfinite(best_distance) else float("inf"),
            best_sample_index=best_index,
            distances=distances,
            mutation_strengths=mutation_strength_log,
            image_log_dir=str(image_output_dir) if image_output_dir is not None else None,
            best_image_sample_index=best_image_index,
            best_image_mse=best_image_mse,
            best_image_similarity=best_image_similarity,
            best_genome_image_mse=best_genome_image_mse,
            best_genome_image_similarity=best_genome_image_similarity,
            best_genome_mutation_strength=best_genome_mutation_strength,
            best_image_mutation_strength=best_image_mutation_strength,
            notes=notes,
        ),
        notes,
    )


def iter_pids(pb_dir: Path) -> Iterable[str]:
    for entry in sorted(pb_dir.iterdir()):
        if entry.is_dir():
            yield entry.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare modern mutation against legacy Picbreeder offspring")
    parser.add_argument("--pb-dir", type=Path, default=FER_ROOT / "spaghetti/pbRender/genomeAll", help="Directory containing legacy pid folders")
    parser.add_argument("--pids", nargs="*", default=None, help="Optional subset of pid directories to process")
    parser.add_argument("--samples", type=int, default=200, help="Number of offspring samples per mutation strength (default: 100)")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for reproducibility")
    parser.add_argument("--image-dir", type=Path, default=None, help="Directory to store rendered parent/child/mutant images")
    parser.add_argument("--image-res", type=int, default=256, help="Resolution (width=height) for rendered images (default: 256)")
    parser.add_argument("--output", type=Path, default='mutation_similarities.json', help="Write detailed JSON report to this path")
    args = parser.parse_args()

    pb_dir = args.pb_dir.expanduser().resolve()
    if not pb_dir.is_dir():
        raise SystemExit(f"Picbreeder directory not found: {pb_dir}")

    image_dir = args.image_dir.expanduser().resolve() if args.image_dir else None
    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)

    selected_pids = args.pids if args.pids else list(iter_pids(pb_dir))
    repo = GenomeRepository(pb_dir)

    results: List[LineageResult] = []
    skipped: List[Tuple[str, List[str]]] = []
    for pid in selected_pids:
        result, notes = analyze_pid(
            pid,
            repo,
            samples=max(1, args.samples),
            seed=args.seed,
            image_dir=image_dir,
            image_resolution=max(1, args.image_res),
        )
        if result is None:
            skipped.append((pid, notes))
            if notes:
                print(f"pid {pid}: skipped ({'; '.join(notes)})")
            continue
        results.append(result)
        summary = "match" if math.isclose(result.best_distance, 0.0, abs_tol=1e-9) else "gap"
        sample_text = f"sample {result.best_sample_index}" if result.best_sample_index is not None else "n/a"
        image_text = (
            f"img sim {result.best_image_similarity:.3f}" if result.best_image_similarity is not None else "img sim n/a"
        )
        genome_strength_text = (
            f"{result.best_genome_mutation_strength:.2f}" if result.best_genome_mutation_strength is not None else "n/a"
        )
        image_strength_text = (
            f"{result.best_image_mutation_strength:.2f}" if result.best_image_mutation_strength is not None else "n/a"
        )
        strength_text = f"strengths g={genome_strength_text} i={image_strength_text}"
        print(
            f"pid {pid}: parent {result.parent_key} -> child {result.child_key} | "
            f"best {result.best_distance:.3f} ({sample_text}) | parent diff {result.parent_distance:.3f} | "
            f"{summary} | {image_text} | {strength_text}"
        )
        if result.notes:
            for note in result.notes:
                print(f"  note: {note}")

    if skipped:
        print(f"Skipped {len(skipped)} pid(s) due to missing data.")

    total = len(results)
    if total:
        exact = sum(1 for entry in results if math.isclose(entry.best_distance, 0.0, abs_tol=1e-9))
        avg_best = sum(entry.best_distance for entry in results) / total
        print(f"Exact matches: {exact}/{total} ({exact / total:.2%})")
        print(f"Mean best distance: {avg_best:.3f}")

    if args.output:
        report = {
            "pb_dir": str(pb_dir),
            "samples": args.samples,
            "seed": args.seed,
            "results": [entry.to_dict() for entry in results],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"Wrote JSON report -> {args.output}")


if __name__ == "__main__":
    main()
