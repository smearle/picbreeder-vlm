#!/usr/bin/env python3
"""Niche-based evolutionary strategy for Picbreeder CPPNs guided by CLIP noun similarities.

This variant maintains a single elite genome for every noun in nounlist.txt (one niche per noun).
Each offspring is embedded in CLIP space; if it scores higher for any noun than that noun's
current elite, it replaces the elite and the replacement image is saved (when --save-images is set).
Runs can resume from previously saved state in the hyperparameter-derived run directory.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import random
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

from hydra import main as hydra_main
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
import neat
import numpy as np
import open_clip
from PIL import Image
import torch
from tqdm import tqdm

from model_loader import load_model_by_name

from neat_components import (
    InteractiveStagnation,
    PicbreederGenome,
    apply_picbreeder_config_defaults,
    seed_initial_population,
    sync_population_node_indexer,
    sync_population_output_activations,
    _initialize_color_bootstrap,
    _seed_picbreeder_genome,
)
from picbreeder_reproduction import PicbreederReproduction
from picture2d.common import eval_genome_as_grayscale_and_color
from rendering import create_numbered_grid

from clip_noun_niche_config import ClipNounNicheConfig
from clip_noun_niche_shared import (
    build_run_name,
    resolve_path,
    compress_run_images,
    decompress_run_images,
)
from utils import resolve_nounlist


STATE_FILENAME = "state.pkl"
cs = ConfigStore.instance()
cs.store(name="clip_noun_niche", node=ClipNounNicheConfig)


@dataclass
class ClipComponents:
    model: torch.nn.Module
    preprocess: Any  # Transform callable from open_clip
    text_embeddings: torch.Tensor
    device: torch.device


@dataclass
class NicheElite:
    genome: PicbreederGenome
    score: float




def ensure_output_dir(base: Path, run_name: str, save_images: bool) -> Path:
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if save_images:
        (run_dir / "images" / "grids").mkdir(parents=True, exist_ok=True)
        (run_dir / "images" / "niches").mkdir(parents=True, exist_ok=True)
        (run_dir / "elites").mkdir(parents=True, exist_ok=True)
    return run_dir


def build_config(cfg_path: Path, pop_size: int, mutation_strength: float, crossover_strength: float) -> neat.Config:
    config = neat.Config(
        PicbreederGenome,
        PicbreederReproduction,
        neat.DefaultSpeciesSet,
        InteractiveStagnation,
        str(cfg_path),
    )
    apply_picbreeder_config_defaults(config, enable_output_activations=True, enable_input_activations=False, enable_crossover=(crossover_strength > 0))
    config.pop_size = pop_size
    config.picbreeder_mutation_strength = mutation_strength
    config.genome_config.picbreeder_mutation_strength = mutation_strength
    return config


def initialize_population(config: neat.Config) -> Tuple[List[PicbreederGenome], int]:
    population = neat.Population(config)
    sync_population_output_activations(population, config.genome_config)
    seed_initial_population(population, config.genome_config)
    sync_population_node_indexer(population)
    genomes = list(population.population.values())
    next_key = max(genome.key for genome in genomes) + 1 if genomes else 1
    return genomes, next_key


def set_mutation_mode(config: neat.Config, mode: str) -> None:
    mode = mode.lower().strip()
    config.picbreeder_mutation_mode = mode
    config.genome_config.picbreeder_mutation_mode = mode


def _new_random_genome(
    config: neat.Config,
    key: int,
    activation_choices: Sequence[str],
) -> PicbreederGenome:
    genome: PicbreederGenome = config.genome_type(key)  # type: ignore[call-arg]
    genome.configure_new(config.genome_config)
    _seed_picbreeder_genome(genome, config.genome_config, activation_choices)
    dummy_population = type("DummyPop", (), {"population": {key: genome}})
    _initialize_color_bootstrap(dummy_population, config.genome_config)
    genome.fitness = None
    return genome


def _render_worker(args: Tuple[PicbreederGenome, neat.Config, int]) -> np.ndarray:
    genome, config, render_size = args
    _, color_image = eval_genome_as_grayscale_and_color(genome, config, render_size, render_size)
    return np.array(color_image, dtype=np.uint8)


def render_population(
    genomes: Sequence[PicbreederGenome],
    config: neat.Config,
    render_size: int,
    image_paths: List[Path] = None,
    num_proc: int = 1,
) -> List[Image.Image]:
    images: List[Image.Image] = []
    if num_proc > 1:
        with ProcessPoolExecutor(max_workers=num_proc) as executor:
            args_list = [(genome, config, render_size) for genome in genomes]
            results = list(
                tqdm(
                    executor.map(_render_worker, args_list),
                    total=len(genomes),
                    desc="Rendering genomes (MP)",
                )
            )
            images = [Image.fromarray(arr, mode="RGB") for arr in results]
    else:
        for i, genome in tqdm(enumerate(genomes), desc="Rendering genomes"):
            _, color_image = eval_genome_as_grayscale_and_color(
                genome, config, render_size, render_size
            )
            arr = np.array(color_image, dtype=np.uint8)
            image = Image.fromarray(arr, mode="RGB")
            images.append(image)

    if image_paths is not None:
        for i, image in enumerate(images):
            image_path = image_paths[i]
            image_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving image for genome {i} to {image_path}")
            image.save(image_path, format="PNG")
    return images


def embed_images(
    model: torch.nn.Module,
    preprocess,
    device: torch.device,
    images: Sequence[Image.Image],
    batch_size: int,
    output_dim: int,
) -> torch.Tensor:
    if not images:
        return torch.zeros((0, output_dim), device=device)

    tensors = [preprocess(img.convert("RGB")) for img in images]
    batches: List[torch.Tensor] = []
    for idx in range(0, len(tensors), batch_size):
        chunk = torch.stack(tensors[idx : idx + batch_size]).to(device)
        with torch.no_grad():
            emb = model.encode_image(chunk)
            emb = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-6)
        batches.append(emb)
    return torch.cat(batches, dim=0)


def load_nouns(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Noun list not found at {path}")
    nouns: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        noun = line.strip().replace("_", " ")
        if noun:
            nouns.append(noun)
    if not nouns:
        raise ValueError(f"No nouns loaded from {path}")
    return nouns


def _embed_texts_in_batches(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    texts: Sequence[str],
    batch_size: int = 256,
) -> torch.Tensor:
    embeddings: List[torch.Tensor] = []
    with torch.no_grad():
        print(f"Embedding {len(texts)} texts in batches of {batch_size}...")
        for idx in tqdm(range(0, len(texts), batch_size)):
            tokens = tokenizer(list(texts[idx : idx + batch_size])).to(device)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-6)
            embeddings.append(emb)
    return torch.cat(embeddings, dim=0)


def load_clip_components(
    model_name: str,
    pretrained: str,
    device_override: str | None,
    nouns: Sequence[str],
) -> ClipComponents:
    device = torch.device(device_override) if device_override else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, tokenizer = load_model_by_name(model_name, pretrained, device)
    
    text_embs = _embed_texts_in_batches(model, tokenizer, device, nouns)
    return ClipComponents(model=model, preprocess=preprocess, text_embeddings=text_embs, device=device)


def sanitize_noun(noun: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", noun.strip())
    return cleaned or "noun"


def save_niche_image(run_dir: Path, noun: str, generation: int, mode: str, score: float, image: Image.Image, save_history: bool = True) -> Path | None:
    noun_slug = sanitize_noun(noun)
    
    # Ensure output directories exist
    (run_dir / "elites").mkdir(parents=True, exist_ok=True)
    
    # Always update the current elite image for this noun
    elite_path = run_dir / "elites" / f"{noun_slug}.png"
    image.save(elite_path, format="PNG")
    
    path = None
    if save_history:
        (run_dir / "images" / "niches").mkdir(parents=True, exist_ok=True)
        filename = f"gen_{generation:04d}_mode-{mode}_{noun_slug}_score_{score:.4f}.png"
        path = run_dir / "images" / "niches" / filename
        image.save(path, format="PNG")
    
    return path


def evaluate_and_update_niches(
    genomes: Sequence[PicbreederGenome],
    config: neat.Config,
    clip: ClipComponents,
    nouns: Sequence[str],
    render_size: int,
    batch_size: int,
    best_scores: List[float],
    niche_elites: List[NicheElite | None],
    run_dir: Path,
    generation: int,
    phase: str,
    save_images: bool,
    num_proc: int = 1,
) -> Tuple[int, List[Image.Image], List[str]]:
    images = render_population(genomes, config, render_size, num_proc=num_proc)
    embedding_dim = int(clip.text_embeddings.shape[1])
    image_embeddings = embed_images(clip.model, clip.preprocess, clip.device, images, batch_size, embedding_dim)
    if image_embeddings.numel() == 0:
        return 0, images, []

    sims = image_embeddings @ clip.text_embeddings.T
    replacements = 0
    replaced_nouns: List[str] = []
    for idx, genome in enumerate(genomes):
        sim_values = sims[idx].tolist()
        image = images[idx]
        genome.fitness = max(sim_values) if sim_values else float("-inf")
        for noun_idx, sim in enumerate(sim_values):
            if sim > best_scores[noun_idx]:
                best_scores[noun_idx] = float(sim)
                niche_elites[noun_idx] = NicheElite(genome=genome, score=float(sim))
                save_niche_image(run_dir, nouns[noun_idx], generation, phase, float(sim), image, save_history=save_images)
                replacements += 1
                replaced_nouns.append(nouns[noun_idx])
    return replacements, images, replaced_nouns


def qd_score(best_scores: Sequence[float]) -> float:
    finite_scores = [score for score in best_scores if math.isfinite(score)]
    return float(sum(finite_scores)) if finite_scores else float("-inf")


def allocate_key(next_key_value: int) -> Tuple[int, int]:
    return next_key_value, next_key_value + 1


def make_args_signature(
    cfg: ClipNounNicheConfig,
    nouns: Sequence[str],
    nounlist_path: Path,
    config_path: Path,
    output_root: Path,
) -> dict:
    return {
        "stage_length": cfg.stage_length,
        "render_size": cfg.render_size,
        "batch_size": cfg.batch_size,
        "mutation_strength": cfg.mutation_strength,
        "new_random_prob": cfg.new_random_prob,
        "crossover_strength": cfg.crossover_strength,
        "nounlist": str(nounlist_path),
        "config": str(config_path),
        "output_dir": str(output_root),
        "clip_model": cfg.clip_model,
        "clip_pretrained": cfg.clip_pretrained,
        "seed": cfg.seed,
        "nouns": list(nouns),
    }


def save_state(
    path: Path,
    generation: int,
    best_scores: List[float],
    niche_elites: List[NicheElite | None],
    next_key_value: int,
    args_signature: dict,
) -> None:
    torch_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    payload = {
        "generation": generation,
        "best_scores": best_scores,
        "niche_elites": niche_elites,
        "next_key_value": next_key_value,
        "random_state": random.getstate(),
        "numpy_state": np.random.get_state(),
        "torch_state": torch.random.get_rng_state(),
        "torch_cuda_state": torch_cuda_state,
        "args_signature": args_signature,
    }
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)


def load_state(path: Path, expected_signature: dict) -> Tuple[int, List[float], List[NicheElite | None], int]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    saved_signature = payload.get("args_signature")
    random.setstate(payload["random_state"])
    np.random.set_state(payload["numpy_state"])
    torch.random.set_rng_state(payload["torch_state"])
    torch_cuda_state = payload.get("torch_cuda_state")
    if torch_cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch_cuda_state)
    return (
        int(payload["generation"]),
        list(payload["best_scores"]),
        list(payload["niche_elites"]),
        int(payload["next_key_value"]),
    )


def _run_es_unsafe(cfg: ClipNounNicheConfig, original_cwd: Path) -> None:
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    nounlist_path = resolve_nounlist(cfg.nounlist, Path(original_cwd))
    config_path = resolve_path(cfg.config, Path(original_cwd))
    output_root = resolve_path(cfg.output_dir, Path(original_cwd))

    if not config_path.exists():
        raise FileNotFoundError(f"NEAT config not found at {config_path}")

    nouns = load_nouns(nounlist_path)
    run_name = build_run_name(cfg)
    run_dir = ensure_output_dir(output_root, run_name, cfg.save_images)
    if cfg.save_offspring_grids and not cfg.save_images:
        print("Warning: --save-offspring-grids was provided but --save-images is disabled; no images will be written.")
    metrics_path = run_dir / "metrics.jsonl"
    state_path = run_dir / STATE_FILENAME
    args_signature = make_args_signature(cfg, nouns, nounlist_path, config_path, output_root)

    config = build_config(config_path, cfg.batch_size, cfg.mutation_strength, cfg.crossover_strength)
    print(f"Starting CLIP noun-niche ES run: {run_name}")
    clip = load_clip_components(cfg.clip_model, cfg.clip_pretrained, cfg.device, nouns)
    activation_choices = [opt for opt in getattr(config.genome_config, "activation_options", [])]

    if state_path.exists():
        loaded_gen, best_scores, niche_elites, next_key_value = load_state(state_path, args_signature)
        start_generation = loaded_gen + 1
        # Restore node indexer to avoid collisions on resumed mutations.
        dummy_population = type("DummyPop", (), {"population": {elite.genome.key: elite.genome for elite in niche_elites if elite is not None}, "config": config})
        sync_population_node_indexer(dummy_population)  # type: ignore[arg-type]
        print(f"Resuming from generation {loaded_gen} in {run_dir}")
        print(f"Parents available: {len([e for e in niche_elites if e is not None])} | next key: {next_key_value}")
    else:
        genomes, next_key_value = initialize_population(config)
        best_scores = [float("-inf")] * len(nouns)
        niche_elites = [None for _ in nouns]

        init_replacements, _, init_replaced_nouns = evaluate_and_update_niches(
            genomes,
            config,
            clip,
            nouns,
            cfg.render_size,
            cfg.batch_size,
            best_scores,
            niche_elites,
            run_dir,
            generation=0,
            phase="init",
            save_images=cfg.save_images,
            num_proc=cfg.num_proc,
        )

        init_payload = {
            "generation": 0,
            "phase": "init",
            "replacements": init_replacements,
            "filled_niches": int(sum(1 for elite in niche_elites if elite is not None)),
            "mean_best_score": float(np.mean(best_scores)),
            "std_best_score": float(np.std(best_scores)),
            "max_best_score": float(np.max(best_scores)),
            "qd_score": qd_score(best_scores),
            "nouns": len(nouns),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(init_payload) + "\n")
        save_state(state_path, 0, best_scores, niche_elites, next_key_value, args_signature)
        replaced_summary = sorted(set(init_replaced_nouns))
        replaced_preview = ", ".join(replaced_summary[:5])
        extra = f" (+{len(replaced_summary) - 5} more)" if len(replaced_summary) > 5 else ""
        print(
            f"[Gen 000 | init] replacements={init_replacements} "
            f"filled_niches={init_payload['filled_niches']} "
            f"mean_best={init_payload['mean_best_score']:.4f} max_best={init_payload['max_best_score']:.4f} "
            f"qd={init_payload['qd_score']:.4f}"
            f"{' | niches: ' + replaced_preview + extra if replaced_preview else ''}"
        )
        start_generation = 1

    if start_generation > cfg.generations:
        print(f"Target generations already reached (start={start_generation}, target={cfg.generations}). Nothing to do.")
        return

    for generation in range(start_generation, cfg.generations + 1):
        phase = "structure_only" if ((generation - 1) // cfg.stage_length) % 2 == 0 else "color_only"
        set_mutation_mode(config, phase)

        parents = [elite.genome for elite in niche_elites if elite is not None]
        if not parents:
            raise RuntimeError("No parents available; niche elites list is empty.")

        children: List[PicbreederGenome] = []
        for _ in range(cfg.batch_size):
            key, next_key_value = allocate_key(next_key_value)
            if random.random() < cfg.new_random_prob:
                child = _new_random_genome(config, key, activation_choices)
            else:
                # If crossover is enabled, use it with probability crossover_strength (if we have enough parents).
                if cfg.crossover_strength > 0 and len(parents) > 1 and random.random() < cfg.crossover_strength:
                    parent1 = random.choice(parents)
                    parent2 = random.choice(parents)
                    child = config.genome_type(key)
                    child.configure_crossover(parent1, parent2, config.genome_config)
                else:
                    parent = random.choice(parents)
                    child = PicbreederReproduction._clone_genome(parent, key)  # type: ignore[attr-defined]
                child.mutate(config.genome_config)
            children.append(child)

        if cfg.save_images and cfg.save_offspring_grids:
            child_images = render_population(children, config, cfg.render_size, num_proc=cfg.num_proc)
            total = len(child_images)
            if total > 0:
                cols = max(1, int(math.ceil(math.sqrt(total))))
                rows = int(math.ceil(total / cols))
                grid = create_numbered_grid(child_images, rows, cols, cfg.render_size)
                grid_path = run_dir / "images" / "grids" / f"gen_{generation:04d}_mode-{phase}_offspring.png"
                grid.save(grid_path, format="PNG")

        replacements, _, replaced_nouns = evaluate_and_update_niches(
            children,
            config,
            clip,
            nouns,
            cfg.render_size,
            cfg.batch_size,
            best_scores,
            niche_elites,
            run_dir,
            generation=generation,
            phase=phase,
            save_images=cfg.save_images,
            num_proc=cfg.num_proc,
        )

        payload = {
            "generation": generation,
            "phase": phase,
            "replacements": replacements,
            "filled_niches": int(sum(1 for elite in niche_elites if elite is not None)),
            "mean_best_score": float(np.mean(best_scores)),
            "std_best_score": float(np.std(best_scores)),
            "max_best_score": float(np.max(best_scores)),
            "qd_score": qd_score(best_scores),
            "nouns": len(nouns),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        save_state(state_path, generation, best_scores, niche_elites, next_key_value, args_signature)

        replaced_summary = sorted(set(replaced_nouns))
        replaced_preview = ", ".join(replaced_summary[:5])
        extra = f" (+{len(replaced_summary) - 5} more)" if len(replaced_summary) > 5 else ""
        print(
            f"[Gen {generation:03d} | {phase}] replacements={replacements} "
            f"filled_niches={payload['filled_niches']} "
            f"mean_best={payload['mean_best_score']:.4f} max_best={payload['max_best_score']:.4f} "
            f"qd={payload['qd_score']:.4f}"
            f"{' | niches: ' + replaced_preview + extra if replaced_preview else ''}"
        )

    print(
        f"Run complete. Metrics/state in {run_dir}. "
        f"Images saved: {'yes' if cfg.save_images else 'no'}"
    )


def _validate_cfg(cfg: ClipNounNicheConfig) -> ClipNounNicheConfig:
    if cfg.generations <= 0:
        raise ValueError("generations must be positive.")
    if cfg.stage_length <= 0:
        raise ValueError("stage_length must be positive.")
    if cfg.new_random_prob < 0 or cfg.new_random_prob > 1:
        raise ValueError("new_random_prob must be between 0 and 1.")
    return cfg


def run_es(cfg: ClipNounNicheConfig, original_cwd: Path) -> None:
    # Determine run directory to handle zip compression/decompression
    output_root = resolve_path(cfg.output_dir, Path(original_cwd))
    run_name = build_run_name(cfg)
    run_dir = output_root / run_name

    # Decompress existing images if we are resuming
    decompress_run_images(run_dir, remove_zip=True)
    try:
        _run_es_unsafe(cfg, original_cwd)
    finally:
        # Compress images on exit (success, failure, or interruption)
        compress_run_images(run_dir)


@hydra_main(version_base="1.3", config_path=None, config_name="clip_noun_niche")
def main(cfg: ClipNounNicheConfig) -> None:
    validated = _validate_cfg(cfg)
    original_cwd = Path(get_original_cwd())
    run_es(validated, original_cwd)


if __name__ == "__main__":
    main()
