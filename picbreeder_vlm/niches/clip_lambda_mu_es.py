#!/usr/bin/env python3
"""(lambda + mu) evolutionary strategy for Picbreeder CPPNs guided by CLIP text similarity.

This script mutates Picbreeder CPPNs using the existing mutation operators and scores them
with an OpenCLIP embedding against a target word (default: "apple"). It alternates between
structure-only and color-only mutation phases, spending N generations in each phase.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import neat
import numpy as np
from PIL import Image

try:
    import torch
    import open_clip
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError("This script requires `torch` and `open_clip` to be installed.") from exc

from picbreeder_vlm.core.neat_components import (
    InteractiveStagnation,
    PicbreederGenome,
    apply_picbreeder_config_defaults,
    seed_initial_population,
    sync_population_node_indexer,
    sync_population_output_activations,
    _initialize_color_bootstrap,
    _seed_picbreeder_genome,
)
from picbreeder_vlm.core.picbreeder_reproduction import PicbreederReproduction
from picture2d.common import eval_genome_as_grayscale_and_color
from picbreeder_vlm.core.rendering import create_numbered_grid


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "picture2d" / "interactive_config_color"


@dataclass
class ClipComponents:
    model: torch.nn.Module
    preprocess: Any  # Transform callable from open_clip
    text_embedding: torch.Tensor
    device: torch.device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a (lambda + mu) ES on Picbreeder genomes toward a CLIP text target.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", type=str, default="apple", help="Target word or phrase.")
    parser.add_argument("--generations", type=int, default=1_000, help="Number of ES generations.")
    parser.add_argument("--mu", type=int, default=12, help="Number of parents retained each generation.")
    parser.add_argument("--lambda-offspring", type=int, dest="lambda_offspring", default=36, help="Children produced each generation.")
    parser.add_argument("--stage-length", type=int, default=5, help="Generations to spend in each structure/color phase.")
    parser.add_argument("--render-size", type=int, default=224, help="Rendered image size (square).")
    parser.add_argument("--batch-size", type=int, default=8, help="CLIP image batch size.")
    parser.add_argument("--mutation-strength", type=float, default=0.5, help="Picbreeder mutation strength (0-1).")
    parser.add_argument(
        "--new-random-prob",
        type=float,
        default=0.05,
        help="Probability that an offspring is a fresh random CPPN instead of a mutation.",
    )
    parser.add_argument(
        "--save-offspring-grids",
        action="store_true",
        default=False,
        help="If set, save a numbered grid image of all offspring each generation.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to NEAT configuration file.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "clip_lambda_mu_es", help="Directory to write renders and logs.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override (cpu / cuda / mps).")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    parser.add_argument("--clip-model", type=str, default="ViT-H-14", help="OpenCLIP model name.")
    parser.add_argument("--clip-pretrained", type=str, default="laion2b_s32b_b79k", help="OpenCLIP pretrained tag.")
    return parser.parse_args()


def ensure_output_dir(base: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"run_{timestamp}"
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    (run_dir / "images" / "grids").mkdir(parents=True, exist_ok=True)
    return run_dir


def build_config(cfg_path: Path, pop_size: int, mutation_strength: float) -> neat.Config:
    config = neat.Config(
        PicbreederGenome,
        PicbreederReproduction,
        neat.DefaultSpeciesSet,
        InteractiveStagnation,
        str(cfg_path),
    )
    apply_picbreeder_config_defaults(config, enable_output_activations=True, enable_input_activations=False, enable_crossover=False)
    config.pop_size = pop_size
    config.picbreeder_mutation_strength = mutation_strength
    config.genome_config.picbreeder_mutation_strength = mutation_strength
    return config


def initialize_population(config: neat.Config) -> Tuple[List[PicbreederGenome], Iterable[int]]:
    population = neat.Population(config)
    sync_population_output_activations(population, config.genome_config)
    seed_initial_population(population, config.genome_config)
    sync_population_node_indexer(population)
    genomes = list(population.population.values())
    next_key = count(start=(max(genome.key for genome in genomes) + 1 if genomes else 1))
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


def render_population(genomes: Sequence[PicbreederGenome], config: neat.Config, render_size: int) -> List[Image.Image]:
    images: List[Image.Image] = []
    for genome in genomes:
        _, color_image = eval_genome_as_grayscale_and_color(genome, config, render_size, render_size)
        arr = np.array(color_image, dtype=np.uint8)
        images.append(Image.fromarray(arr, mode="RGB"))
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


def score_population(
    genomes: Sequence[PicbreederGenome],
    config: neat.Config,
    clip: ClipComponents,
    render_size: int,
    batch_size: int,
) -> Tuple[List[float], List[Image.Image]]:
    images = render_population(genomes, config, render_size)
    embedding_dim = int(clip.text_embedding.shape[1])
    image_embeddings = embed_images(clip.model, clip.preprocess, clip.device, images, batch_size, embedding_dim)
    if image_embeddings.numel() == 0:
        scores = [float("-inf")] * len(genomes)
    else:
        sims = image_embeddings @ clip.text_embedding.T
        scores = sims.squeeze(1).cpu().tolist()

    for genome, score in zip(genomes, scores):
        genome.fitness = float(score)
    return scores, images


def select_top(genomes: Sequence[PicbreederGenome], k: int) -> List[PicbreederGenome]:
    sorted_genomes = sorted(genomes, key=lambda g: (g.fitness if g.fitness is not None else -math.inf), reverse=True)
    return list(sorted_genomes[:k])


def load_clip_components(
    model_name: str,
    pretrained: str,
    device_override: str | None,
    target: str,
) -> ClipComponents:
    device = torch.device(device_override) if device_override else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device)
    model.eval()
    with torch.no_grad():
        tokens = tokenizer([target]).to(device)
        text_emb = model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=1, keepdim=True).clamp(min=1e-6)
    return ClipComponents(model=model, preprocess=preprocess, text_embedding=text_emb, device=device)


def save_best_image(run_dir: Path, generation: int, mode: str, score: float, image: Image.Image) -> Path:
    filename = f"gen_{generation:04d}_mode-{mode}_score_{score:.4f}.png"
    path = run_dir / "images" / filename
    image.save(path, format="PNG")
    return path


def run_es(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"NEAT config not found at {config_path}")

    config = build_config(config_path, args.mu, args.mutation_strength)
    genomes, key_gen = initialize_population(config)
    clip = load_clip_components(args.clip_model, args.clip_pretrained, args.device, args.target)
    run_dir = ensure_output_dir(args.output_dir.resolve())
    metrics_path = run_dir / "metrics.jsonl"
    activation_choices = [opt for opt in getattr(config.genome_config, "activation_options", [])]

    # Evaluate initial parents.
    scores, images = score_population(genomes, config, clip, args.render_size, args.batch_size)
    best_idx = int(np.argmax(scores)) if scores else -1
    if best_idx >= 0:
        save_best_image(run_dir, 0, "init", scores[best_idx], images[best_idx])

    init_payload = {
        "generation": 0,
        "phase": "init",
        "best_score": float(scores[best_idx]) if best_idx >= 0 else float("-inf"),
        "mean_score": float(np.mean(scores)) if scores else float("nan"),
        "max_score": float(max(scores)) if scores else float("-inf"),
        "mu": args.mu,
        "lambda": args.lambda_offspring,
        "target": args.target,
    }
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(init_payload) + "\n")
    print(
        f"[Gen 000 | init] best={init_payload['best_score']:.4f} "
        f"mean={init_payload['mean_score']:.4f} max={init_payload['max_score']:.4f}"
    )

    for generation in range(1, args.generations + 1):
        phase = "structure_only" if ((generation - 1) // args.stage_length) % 2 == 0 else "color_only"
        set_mutation_mode(config, phase)

        parents = select_top(genomes, args.mu)
        children: List[PicbreederGenome] = []
        for _ in range(args.lambda_offspring):
            parent = random.choice(parents)
            if random.random() < args.new_random_prob:
                child = _new_random_genome(config, next(key_gen), activation_choices)
            else:
                child = PicbreederReproduction._clone_genome(parent, next(key_gen))  # type: ignore[attr-defined]
                child.mutate(config.genome_config)
            children.append(child)

        if args.save_offspring_grids:
            child_images = render_population(children, config, args.render_size)
            total = len(child_images)
            if total > 0:
                cols = max(1, int(math.ceil(math.sqrt(total))))
                rows = int(math.ceil(total / cols))
                grid = create_numbered_grid(child_images, rows, cols, args.render_size)
                grid_path = run_dir / "images" / "grids" / f"gen_{generation:04d}_mode-{phase}_offspring.png"
                grid.save(grid_path, format="PNG")

        genomes = parents + children
        scores, images = score_population(genomes, config, clip, args.render_size, args.batch_size)
        genomes = select_top(genomes, args.mu)

        best_idx = int(np.argmax(scores)) if scores else -1
        best_score = scores[best_idx] if best_idx >= 0 else float("-inf")
        best_image = images[best_idx] if best_idx >= 0 else None
        if best_image is not None:
            save_best_image(run_dir, generation, phase, float(best_score), best_image)

        payload = {
            "generation": generation,
            "phase": phase,
            "best_score": float(best_score),
            "mean_score": float(np.mean(scores)) if scores else float("nan"),
            "max_score": float(max(scores)) if scores else float("-inf"),
            "mu": args.mu,
            "lambda": args.lambda_offspring,
            "target": args.target,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

        print(
            f"[Gen {generation:03d} | {phase}] best={best_score:.4f} "
            f"mean={payload['mean_score']:.4f} max={payload['max_score']:.4f}"
        )

    print(f"Run complete. Images and metrics written to {run_dir}")


def main() -> None:
    args = parse_args()
    if args.mu <= 0 or args.lambda_offspring <= 0:
        raise ValueError("--mu and --lambda-offspring must be positive.")
    if args.generations <= 0:
        raise ValueError("--generations must be positive.")
    if args.stage_length <= 0:
        raise ValueError("--stage-length must be positive.")
    if args.new_random_prob < 0 or args.new_random_prob > 1:
        raise ValueError("--new-random-prob must be between 0 and 1.")
    run_es(args)


if __name__ == "__main__":
    main()
