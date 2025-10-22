import argparse
import functools
import json
import os
import random
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from statistics import mean, pstdev

import sys

try:
    import graphviz  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    graphviz = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent
NEAT_VENDOR_PATH = REPO_ROOT / "neat-python"
if NEAT_VENDOR_PATH.exists():
    sys.path.insert(0, str(NEAT_VENDOR_PATH))

import neat
from neat.checkpoint import Checkpointer
from neat.population import CompleteExtinctionException
from neurogram_backend import DEFAULT_MODULE_PATH, run_neurogram
from neat_components import (
    CHECKPOINT_SUFFIX,
    GenerationCheckpointer,
    InteractiveStagnation,
    PicbreederGenome,
    apply_picbreeder_config_defaults,
    seed_initial_population,
    sync_population_output_activations,
)
from artifacts import build_generation_state, save_neat_genome_diagrams, save_neat_population
from rendering import create_numbered_grid, decode_image
import im_query  # type: ignore
from render_experiment_results import (  # type: ignore
    process_experiment as _process_selection_gif,
    render_population_structure_plots_from_experiment as _render_population_plots,
)
from picbreeder_reproduction import PicbreederReproduction

DEFAULT_PROMPT = (
    "You are playing with an online platform which evolves small neural networks called Compositional Pattern Producing Networks (CPPNs) to generate images. "
    # "You are searching in an open-ended fashion for visually interesting images. "
    "Your goal is to evolve images that resemble familiar real-world objects. "
    "At each generation, you are shown a grid of numbered images produced by different CPPNs. "
    "Pick one or several images by their numeric labels--the corresponding CPPNs will be used as the parents in the next generation. "
    "The point is not to pick the most interesting images relative to the others in the grid, necessarily, but rather to pick images that you want to mutate and evolve further. "
    "Respond with JSON only: {{\"selected\": [indices], \"rationale\": \"brief explanation\"}}. "
)

DEFAULT_BASELINE_SELECTION_LIMIT = 1
SELECTION_BASELINES = ("none", "random", "max-depth", "max-nodes")


def apply_select_limit_to_prompt(prompt: str, select_k: Optional[int]) -> str:
    if select_k is None:
        return prompt
    needle = "Pick one or several images"
    if f"{needle} (up to" in prompt:
        return prompt
    if needle in prompt:
        return prompt.replace(needle, f"{needle} (up to {select_k})", 1)
    return prompt


def cap_select_k_for_engine(engine: str, select_k: Optional[int]) -> Optional[int]:
    if select_k is None:
        return None
    if engine == "neurogram":
        return min(select_k, 4)
    return select_k


def dump_initial_populations(
    count: int,
    output_dir: Path,
    config_path: Path,
    rows: int,
    cols: int,
    thumb_size: int,
    scheme: str,
    palette: str,
    enable_output_activations: bool,
) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(count):
        config = neat.Config(
            PicbreederGenome,
            PicbreederReproduction,
            neat.DefaultSpeciesSet,
            InteractiveStagnation,
            str(config_path),
        )
        apply_picbreeder_config_defaults(config, enable_output_activations=enable_output_activations)
        config.pop_size = rows * cols

        population = neat.Population(config)
        sync_population_output_activations(population, config.genome_config)
        seed_initial_population(population, config.genome_config)

        genomes = sorted(population.population.items(), key=lambda item: item[0])
        state, png_cache = build_generation_state(genomes, config, 0, rows, cols, thumb_size, scheme, palette)

        population_dir = output_dir / f"population_{index:03d}"
        # persist_neat_population(state, population_dir, 0, png_cache)

        grid_image = create_numbered_grid(state)
        grid_image.save(os.path.join(output_dir, f"pop-{index:03d}_grid.png"), format="PNG")

        save_neat_genome_diagrams(genomes, config, population_dir, 0)


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("empty response")

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end >= brace_start:
        candidate = cleaned[brace_start : brace_end + 1]
    else:
        candidate = cleaned

    return json.loads(candidate)



def select_parents_from_grid(
    state: Dict[str, Any],
    prompt_template: str,
    query_dir: Path,
    select_k: Optional[int] = None,
) -> Dict[str, Any]:
    if im_query is None:
        raise RuntimeError("im_query module is not available. Install required dependencies to perform image queries.")

    generation = int(state["generation"])
    grid_image = create_numbered_grid(state)

    query_dir.mkdir(parents=True, exist_ok=True)
    base_grid_path = query_dir / f"gen_{generation:03d}_grid.png"
    grid_image.save(base_grid_path, format="PNG")

    buffer = BytesIO()
    grid_image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    prompt = prompt_template.format(
        rows=state["rows"],
        cols=state["cols"],
        total=len(state["images"]),
    )

    response = im_query.query_im(image_bytes, prompt=prompt, mime_type="image/png")
    response_text = getattr(response, "text", "") or ""
    parsed = extract_json_object(response_text)

    raw_selected = parsed.get("selected")
    if not isinstance(raw_selected, list):
        raise ValueError("Gemini response missing 'selected' list.")

    max_index = int(state["rows"]) * int(state["cols"]) - 1
    cleaned: List[int] = []
    for value in raw_selected:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= idx <= max_index and idx not in cleaned:
            cleaned.append(idx)

    if not cleaned:
        raise ValueError("Gemini response did not contain any valid indices.")

    if select_k is not None:
        cleaned = cleaned[:select_k]

    selection_image = create_numbered_grid(state, selected=cleaned)
    selection_path = query_dir / f"gen_{generation:03d}_selection.png"
    selection_image.save(selection_path, format="PNG")

    metadata = {
        "selected": cleaned,
        "raw_selected": raw_selected,
        "rationale": parsed.get("rationale") or parsed.get("reason", ""),
        "response_text": response_text,
        "grid_path": str(base_grid_path),
        "selection_path": str(selection_path),
        "select_k": select_k,
    }
    meta_path = selection_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata


def summarize_genome_structure(genome: neat.DefaultGenome, genome_config: neat.genome.DefaultGenomeConfig) -> Dict[str, Any]:
    input_keys = set(getattr(genome_config, "input_keys", ()))
    output_keys = set(getattr(genome_config, "output_keys", ()))

    enabled_connections = [
        conn.key for conn in genome.connections.values() if getattr(conn, "enabled", False)
    ]

    depth_layers = []
    if enabled_connections:
        depth_layers = neat.graphs.feed_forward_layers(
            list(input_keys),
            list(output_keys),
            enabled_connections,
        )

    depth = len(depth_layers) if depth_layers else 0

    required_nodes: set = set()
    if enabled_connections:
        required_nodes = neat.graphs.required_for_output(
            list(input_keys),
            list(output_keys),
            enabled_connections,
        )

    active_hidden_nodes = len(required_nodes - output_keys)
    hidden_node_count = sum(1 for key in genome.nodes if key not in input_keys and key not in output_keys)

    return {
        "depth": depth,
        "active_hidden_nodes": active_hidden_nodes,
        "hidden_node_count": hidden_node_count,
        "enabled_connection_count": len(enabled_connections),
    }


def _summarize_numeric_collection(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "avg": 0.0, "std": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "avg": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
    }


def compute_population_structure_stats(
    genomes: List[Tuple[int, neat.DefaultGenome]],
    genome_config: neat.genome.DefaultGenomeConfig,
) -> Dict[str, Dict[str, float]]:
    node_counts: List[int] = []
    depths: List[int] = []

    for _, genome in genomes:
        structure = summarize_genome_structure(genome, genome_config)
        node_counts.append(len(genome.nodes))
        depths.append(structure["depth"])

    return {
        "node_count": _summarize_numeric_collection([float(value) for value in node_counts]),
        "depth": _summarize_numeric_collection([float(value) for value in depths]),
    }


class AutomatedNeatEvolver:
    def __init__(
        self,
        population: neat.Population,
        rows: int,
        cols: int,
        thumb_size: int,
        scheme: str,
        palette: str,
        prompt: str,
        experiment_dir: Path,
        population_dir: Path,
        query_dir: Path,
        selection_baseline: str = "none",
        select_k: Optional[int] = None,
    ) -> None:
        self.population = population
        self.rows = rows
        self.cols = cols
        self.thumb_size = thumb_size
        self.scheme = scheme
        self.palette = palette
        self.prompt = prompt
        self.experiment_dir = experiment_dir
        self.population_dir = population_dir
        self.query_dir = query_dir
        self.population_size = rows * cols
        self.selection_baseline = selection_baseline
        self.select_k = select_k
        self._diagram_warning_emitted = False
        self.metrics_dir = experiment_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.metrics_dir / "population_structure.jsonl"
        self._metrics_history: List[Dict[str, Any]] = []
        self._recorded_generations: set[int] = set()
        if self.metrics_path.exists():
            with self.metrics_path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        generation_value = int(entry.get("generation"))
                    except (ValueError, TypeError, json.JSONDecodeError):
                        continue
                    self._recorded_generations.add(generation_value)
                    self._metrics_history.append(entry)

    def evaluate_generation(self, genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config,
                            render_diagrams: bool = False) -> None:
        generation = int(self.population.generation)
        if len(genomes) != self.population_size:
            raise ValueError(
                f"Expected {self.population_size} genomes, received {len(genomes)}."
            )

        print(f"\n--- Generation {generation} ---")
        state, cache = build_generation_state(
            genomes,
            config,
            generation,
            self.rows,
            self.cols,
            self.thumb_size,
            self.scheme,
            self.palette,
        )

        state_path = save_neat_population(state, self.population_dir, generation, cache)
        if render_diagrams:
            diagram_paths = save_neat_genome_diagrams(genomes, config, self.population_dir, generation)
            if diagram_paths:
                diagram_dir = diagram_paths[0].parent
                try:
                    relative_diagram_dir = diagram_dir.relative_to(self.experiment_dir)
                except ValueError:
                    relative_diagram_dir = diagram_dir
                print(f"Genome diagrams saved to {relative_diagram_dir}")
            elif graphviz is None and not self._diagram_warning_emitted:
                print("Graphviz not available; skipping genome diagram export.")
                self._diagram_warning_emitted = True

        if self.selection_baseline == "none":
            selection_meta = select_parents_from_grid(state, self.prompt, self.query_dir, self.select_k)
        else:
            selection_meta = self._select_parents_baseline(generation, genomes, config, state)
        selected = selection_meta["selected"]
        rationale = selection_meta.get("rationale") or "(no rationale)"

        for idx, (_, genome) in enumerate(genomes):
            genome.fitness = 1.0 if idx in selected else 0.0

        print(f"Selected indices: {selected}")
        print(f"Rationale: {rationale}")
        # print(f"Snapshot saved to {state_path}")
        print(f"Grid image saved to {selection_meta.get('grid_path')}")
        metrics_entry = self._record_population_metrics(generation, genomes, config, selected)
        node_stats = metrics_entry["node_count"]
        depth_stats = metrics_entry["depth"]
        print(
            "Node count stats - avg: "
            f"{node_stats['avg']:.2f}, min: {node_stats['min']:.0f}, "
            f"max: {node_stats['max']:.0f}, std: {node_stats['std']:.2f}"
        )
        print(
            "Depth stats - avg: "
            f"{depth_stats['avg']:.2f}, min: {depth_stats['min']:.0f}, "
            f"max: {depth_stats['max']:.0f}, std: {depth_stats['std']:.2f}"
        )
        print(f"Parents selected this generation: {metrics_entry['parents_selected']}")

    def _select_parents_baseline(
        self,
        generation: int,
        genomes: List[Tuple[int, neat.DefaultGenome]],
        config: neat.Config,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        genome_config = config.genome_config
        metrics: List[Dict[str, Any]] = []
        for idx, (genome_id, genome) in enumerate(genomes):
            summary = summarize_genome_structure(genome, genome_config)
            summary.update(
                {
                    "index": idx,
                    "genome_id": genome_id,
                }
            )
            metrics.append(summary)

        max_selection = self.select_k if self.select_k is not None else DEFAULT_BASELINE_SELECTION_LIMIT
        selection_count = max(1, min(max_selection, len(genomes)))
        selected_indices: List[int]
        rationale: str

        if self.selection_baseline == "random":
            selected_indices = random.sample(range(len(genomes)), k=selection_count)
            rationale = f"Random baseline selected {selected_indices}."
        else:
            if self.selection_baseline == "max-depth":
                scoring_key = "depth"
                baseline_label = "maximum depth"
            else:
                scoring_key = "hidden_node_count"
                baseline_label = "maximum hidden-node count"

            max_score = max(metric[scoring_key] for metric in metrics)
            top_candidates = [metric for metric in metrics if metric[scoring_key] == max_score]
            random.shuffle(top_candidates)
            selected_entries = top_candidates[:selection_count]
            if len(selected_entries) < selection_count:
                remaining = [metric for metric in metrics if metric[scoring_key] < max_score]
                random.shuffle(remaining)
                selected_entries.extend(remaining[: selection_count - len(selected_entries)])
            selected_indices = [entry["index"] for entry in selected_entries]
            rationale = (
                f"Baseline favoring {baseline_label} selected {selected_indices} "
                f"(score={max_score})."
            )

        metadata: Dict[str, Any] = {
            "selected": selected_indices,
            "rationale": rationale,
            "baseline": self.selection_baseline,
            "metrics": metrics,
            "select_k": self.select_k,
            "selection_count": selection_count,
        }

        return self._write_baseline_artifacts(generation, state, metadata)

    def _write_baseline_artifacts(
        self,
        generation: int,
        state: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.query_dir.mkdir(parents=True, exist_ok=True)
        selected = metadata.get("selected", [])
        grid_image = create_numbered_grid(state)
        selection_image = create_numbered_grid(state, selected=selected)

        grid_path = self.query_dir / f"gen_{generation:03d}_grid.png"
        selection_path = self.query_dir / f"gen_{generation:03d}_selection.png"
        grid_image.save(grid_path, format="PNG")
        selection_image.save(selection_path, format="PNG")

        enriched = dict(metadata)
        enriched.update(
            {
                "grid_path": str(grid_path),
                "selection_path": str(selection_path),
                "generation": generation,
            }
        )

        meta_path = selection_path.with_suffix(".json")
        meta_path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
        return enriched

    def _record_population_metrics(
        self,
        generation: int,
        genomes: List[Tuple[int, neat.DefaultGenome]],
        config: neat.Config,
        selected_indices: List[int],
    ) -> Dict[str, Any]:
        stats = compute_population_structure_stats(genomes, config.genome_config)
        entry: Dict[str, Any] = {
            "generation": generation,
            "node_count": stats["node_count"],
            "depth": stats["depth"],
            "parents_selected": len(selected_indices),
        }

        if generation in self._recorded_generations:
            updated = False
            for idx, existing in enumerate(self._metrics_history):
                if int(existing.get("generation", -1)) == generation:
                    self._metrics_history[idx] = entry
                    updated = True
                    break
            if updated:
                with self.metrics_path.open("w", encoding="utf-8") as fp:
                    for record in sorted(self._metrics_history, key=lambda item: item["generation"]):
                        fp.write(json.dumps(record))
                        fp.write("\n")
        else:
            with self.metrics_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(entry))
                fp.write("\n")
            self._metrics_history.append(entry)
            self._recorded_generations.add(generation)

        return entry


def ensure_gemini_key() -> None:
    if im_query is None:
        raise ImportError("im_query module is not available. Install required dependencies.")
    if not getattr(im_query, "api_key", None):
        raise EnvironmentError("Environment variable GEMINI_API_KEY is not set.")


def build_experiment_dir(args: argparse.Namespace) -> Path:
    slug_parts = [
        f"g{args.generations}",
        f"ht{args.chat_history_turns}",
        f"r{args.rows}",
        f"c{args.cols}",
        f"ts{args.thumb_size}",
        f"eng-{args.engine}",
    ]
    if args.engine == "neat":
        slug_parts.append(f"pal-{args.color_palette}")
        slug_parts.append(args.scheme)
        if getattr(args, "output_activations", False):
            slug_parts.append("outact")
    if args.select_k is not None:
        slug_parts.append(f"sk{args.select_k}")
    if args.selection_baseline != "none":
        slug_parts.append(f"baseline-{args.selection_baseline}")
    slug = "_".join(slug_parts)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = Path("logs") / f"{slug}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def write_run_metadata(experiment_dir: Path, args: argparse.Namespace) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": args.rows,
        "cols": args.cols,
        "thumb_size": args.thumb_size,
        "generations_planned": args.generations,
        "prompt": args.prompt,
        "chat_history_turns": args.chat_history_turns,
        "engine": args.engine,
        "selection_baseline": args.selection_baseline,
        "select_k": args.select_k,
    }
    if args.engine == "neat":
        metadata["scheme"] = args.scheme
        metadata["config_path"] = str(args.config_path) if args.config_path else None
        metadata["color_palette"] = args.color_palette
        metadata["output_activations"] = bool(getattr(args, "output_activations", False))
    else:
        metadata["module_path"] = str(args.module_path) if args.module_path else None
    metadata_path = experiment_dir / "run_config.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def persist_initial_prompt(experiment_dir: Path, prompt: str) -> Path:
    prompt_path = experiment_dir / "initial_prompt.txt"
    if not prompt_path.exists():
        prompt_path.write_text(prompt, encoding="utf-8")
    return prompt_path


def load_run_metadata(experiment_dir: Path) -> Dict[str, Any]:
    metadata_path = experiment_dir / "run_config.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_neat_checkpoint(population_dir: Path, generation: Optional[int]) -> Path:
    if generation is not None:
        path = population_dir / f"gen_{generation:03d}{CHECKPOINT_SUFFIX}"
        if not path.exists():
            raise FileNotFoundError(
                f"Requested checkpoint for generation {generation} not found at {path}"
            )
        return path

    candidates = sorted(population_dir.glob(f"gen_*{CHECKPOINT_SUFFIX}"))
    if not candidates:
        raise FileNotFoundError(
            f"No saved checkpoints found in '{population_dir}'."
        )
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate Picbreeder-style evolution with either NEAT-Python or the legacy Neurogram backend."
    )
    parser.add_argument(
        "--engine",
        choices=("neat", "neurogram"),
        default="neat",
        help="Evolution engine to use ('neat' for NEAT-Python, 'neurogram' for the JavaScript runtime).",
    )
    parser.add_argument("--generations", type=int, default=200, help="Number of generations to evolve.")
    parser.add_argument("--rows", type=int, default=4, help="Number of rows in the grid (Picbreeder uses 4).")
    parser.add_argument("--cols", type=int, default=5, help="Number of columns in the grid (Picbreeder uses 5).")
    parser.add_argument(
        "--thumb-size",
        type=int,
        default=200,
        help="Thumbnail size for rendered genomes (Picbreeder buttons are 200x200).",
    )
    parser.add_argument(
        "--scheme",
        choices=("color", "gray"),
        default="gray",
        help="Image rendering scheme to use.",
    )
    parser.add_argument(
        "--color-palette",
        choices=("hsb", "sigmoid"),
        default="hsb",
        help="Color palette to use for rendering ('hsb' matches CPPNArtEvolution, 'sigmoid' uses the older sigmoid-based palette).",
    )
    parser.add_argument(
        "--output-activations",
        action="store_true",
        help="Enable CPPN output activation functions (disabled by default).",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Path to the NEAT configuration file (defaults based on scheme).",
    )
    parser.add_argument(
        "--module-path",
        type=Path,
        default=None,
        help="Path to the neurogram_standalone.js module (legacy backend).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Custom prompt template for Gemini. Available tokens: {rows}, {cols}, {total}.",
    )
    parser.add_argument(
        "--select-k",
        type=int,
        default=None,
        help="Maximum number of parents to select each generation (defaults to unlimited).",
    )
    parser.add_argument(
        "--selection-baseline",
        choices=SELECTION_BASELINES,
        default="none",
        help="Baseline selection strategy to use instead of querying Gemini.",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Existing experiment directory to resume.",
    )
    parser.add_argument(
        "--resume-generation",
        type=int,
        default=None,
        help="Checkpoint generation to resume from (defaults to the latest available).",
    )
    parser.add_argument(
        "--chat-history-turns",
        type=int,
        default=0,
        help="Past turns from the conversation to include when querying Gemini.",
    )
    parser.add_argument(
        "--dump-initial-populations",
        type=int,
        default=0,
        help="Number of initial populations to generate and persist instead of running evolution.",
    )
    parser.add_argument(
        "--dump-output-dir",
        type=str,
        default=None,
        help="Directory for dumped initial populations (defaults to a subdirectory of the experiment dir).",
    )
    parser.add_argument(
        "--render-diagrams",
        action="store_true",
        help="Render genome topology diagrams using graphviz (NEAT engine only).",
    )
    parser.add_argument(
        "--gif-output-name",
        default="grid_with_selection.gif",
        help="Filename for the generated selection GIF.",
    )
    parser.add_argument(
        "--gif-duration",
        type=int,
        default=500,
        help="Duration for each selection GIF frame in milliseconds.",
    )
    parser.add_argument(
        "--gif-frame-mode",
        choices=("grid", "first-selection"),
        default="grid",
        help="Frame strategy to use when assembling the selection GIF.",
    )

    args = parser.parse_args()
    if args.resume_dir is None:
        if args.engine == "neat":
            if args.config_path is None:
                base = REPO_ROOT / "picture2d"
                config_name = "interactive_config_color" if args.scheme == "color" else "interactive_config_gray"
                args.config_path = (base / config_name).resolve()
            else:
                args.config_path = args.config_path.resolve()
        else:
            if args.module_path is None:
                args.module_path = DEFAULT_MODULE_PATH.resolve()
            else:
                args.module_path = args.module_path.resolve()
    else:
        if args.config_path is not None:
            args.config_path = args.config_path.resolve()
        if args.module_path is not None:
            args.module_path = args.module_path.resolve()

    if args.dump_output_dir is not None:
        args.dump_output_dir = Path(args.dump_output_dir).resolve()

    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.generations < 1:
        raise ValueError("generations must be at least 1")

    if args.select_k is not None:
        if args.select_k < 1:
            raise ValueError("select-k must be at least 1 when provided.")
        args.select_k = cap_select_k_for_engine(args.engine, args.select_k)

    if args.selection_baseline not in SELECTION_BASELINES:
        raise ValueError(f"Unknown selection baseline '{args.selection_baseline}'.")

    if args.selection_baseline != "none" and args.engine != "neat":
        raise ValueError("Selection baselines other than 'none' are only supported with the NEAT engine.")

    if args.resume_dir:
        if not args.resume_dir.exists():
            raise ValueError(f"resume directory '{args.resume_dir}' does not exist")
        if args.resume_generation is not None and args.resume_generation < 1:
            raise ValueError("resume-generation must be a positive integer")
        if args.config_path is not None and not args.config_path.exists():
            raise ValueError(f"config file not found at {args.config_path}")
        if args.module_path is not None and not args.module_path.exists():
            raise ValueError(f"module not found at {args.module_path}")
    else:
        if args.rows < 1 or args.cols < 1:
            raise ValueError("rows and cols must be positive integers")
        if args.thumb_size < 8:
            raise ValueError("thumb-size must be at least 8")
        if args.engine == "neat":
            if args.config_path is None or not args.config_path.exists():
                raise ValueError(f"config file not found at {args.config_path}")
        else:
            if args.module_path is None or not args.module_path.exists():
                raise ValueError(f"module not found at {args.module_path}")

    if args.chat_history_turns < 0:
        raise ValueError("chat-history-turns must be non-negative")

    if getattr(args, "gif_duration", 0) <= 0:
        raise ValueError("gif-duration must be a positive integer")


def run_neat(args: argparse.Namespace, experiment_dir: Path, population_dir: Path, query_dir: Path) -> None:
    enable_output_activations = args.output_activations 
    if args.resume_dir:
        checkpoint_path = resolve_neat_checkpoint(population_dir, args.resume_generation)
        population = Checkpointer.restore_checkpoint(str(checkpoint_path))
        apply_picbreeder_config_defaults(
            population.config,
            enable_output_activations=enable_output_activations,
        )
        sync_population_output_activations(population, population.config.genome_config)
        print(
            f"Resuming from generation {population.generation}: "
            f"{checkpoint_path.relative_to(experiment_dir)}"
        )
    else:
        config = neat.Config(
            PicbreederGenome,
            PicbreederReproduction,
            neat.DefaultSpeciesSet,
            InteractiveStagnation,
            str(args.config_path),
        )
        apply_picbreeder_config_defaults(
            config,
            enable_output_activations=enable_output_activations,
        )
        config.pop_size = args.rows * args.cols
        population = neat.Population(config)
        sync_population_output_activations(population, config.genome_config)
        seed_initial_population(population, config.genome_config)
        write_run_metadata(experiment_dir, args)
        print(f"Starting new NEAT experiment in {experiment_dir}")

    population.add_reporter(GenerationCheckpointer(population_dir))

    evolver = AutomatedNeatEvolver(
        population,
        args.rows,
        args.cols,
        args.thumb_size,
        args.scheme,
        args.color_palette,
        args.prompt,
        experiment_dir,
        population_dir,
        query_dir,
        selection_baseline=args.selection_baseline,
        select_k=args.select_k,
    )

    evolver.evaluate_generation = functools.partial(
        evolver.evaluate_generation, render_diagrams=args.render_diagrams)
    try:
        population.run(evolver.evaluate_generation, args.generations)
    except CompleteExtinctionException as exc:
        raise SystemExit("Population went extinct; evolution cannot continue.") from exc

    print(f"\nRun complete. Next generation index: {population.generation}")
    try:
        plot_path = _render_population_plots(experiment_dir)
    except FileNotFoundError:
        print("Population metrics not found; skipping structure plot rendering.")
    except RuntimeError as exc:
        print(f"Skipping population structure plot due to error: {exc}")
    else:
        try:
            relative_plot = plot_path.relative_to(experiment_dir)
        except ValueError:
            relative_plot = plot_path
        print(f"Population structure plot saved to {relative_plot}")


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
        if args.dump_initial_populations <= 0 and args.selection_baseline == "none":
            ensure_gemini_key()
    except Exception as exc:
        raise SystemExit(f"Argument error: {exc}") from exc

    if args.dump_initial_populations > 0:
        config_path = args.config_path
        if config_path is None:
            raise SystemExit("Configuration path is required to dump initial populations.")
        output_root = args.dump_output_dir or (REPO_ROOT / "initial_populations")
        timestamp = datetime.now().strftime("initial_%Y%m%d-%H%M%S")
        output_dir = (output_root / timestamp) if isinstance(output_root, Path) else Path(output_root) / timestamp
        dump_initial_populations(
            args.dump_initial_populations,
            output_dir,
            config_path,
            args.rows,
            args.cols,
            args.thumb_size,
            args.scheme,
            args.color_palette,
            bool(args.output_activations),
        )
        print(f"Initial populations saved to {output_dir}")
        return

    if args.resume_dir:
        experiment_dir = args.resume_dir.resolve()
    else:
        experiment_dir = build_experiment_dir(args)

    population_dir = experiment_dir / "populations"
    query_dir = experiment_dir / "queries"
    population_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_dir:
        metadata = load_run_metadata(experiment_dir)
        metadata_engine = metadata.get("engine")
        if not metadata_engine:
            metadata_engine = "neat" if metadata.get("config_path") else "neurogram"
        if args.engine and args.engine != metadata_engine:
            raise SystemExit(
                f"Resume directory was created with engine '{metadata_engine}', "
                f"but '--engine' was set to '{args.engine}'."
            )
        args.engine = metadata_engine
        args.rows = int(metadata["rows"])
        args.cols = int(metadata["cols"])
        args.thumb_size = int(metadata["thumb_size"])
        stored_prompt = metadata.get("prompt")
        if stored_prompt:
            args.prompt = stored_prompt

        stored_select_k = metadata.get("select_k", args.select_k)
        if stored_select_k is not None:
            stored_select_k = int(stored_select_k)
            if stored_select_k < 1:
                raise SystemExit("Resume directory select-k must be at least 1.")
        if stored_select_k != args.select_k:
            print(
                f"Resume directory select-k '{stored_select_k}' overrides CLI value '{args.select_k}'."
            )
        args.select_k = cap_select_k_for_engine(args.engine, stored_select_k)

        stored_baseline = metadata.get("selection_baseline", "none")
        if stored_baseline != args.selection_baseline:
            print(
                f"Resume directory selection baseline '{stored_baseline}' overrides CLI value '{args.selection_baseline}'."
            )
        args.selection_baseline = stored_baseline

        if args.engine == "neat":
            args.scheme = metadata.get("scheme", args.scheme)
            args.color_palette = metadata.get("color_palette", args.color_palette)
            stored_output_activations = metadata.get("output_activations", args.output_activations)
            if isinstance(stored_output_activations, str):
                stored_output_activations = stored_output_activations.lower() == "on"
            stored_output_activations = bool(stored_output_activations)
            if bool(args.output_activations) != stored_output_activations:
                previous = "on" if args.output_activations else "off"
                override = "on" if stored_output_activations else "off"
                print(
                    f"Resume directory output-activations '{override}' overrides CLI value '{previous}'."
                )
            args.output_activations = stored_output_activations
            stored_config = metadata.get("config_path")
            if stored_config:
                args.config_path = Path(stored_config).resolve()
        else:
            stored_module = metadata.get("module_path")
            if stored_module:
                args.module_path = Path(stored_module).resolve()

    args.prompt = apply_select_limit_to_prompt(args.prompt, args.select_k)
    persist_initial_prompt(experiment_dir, args.prompt)

    if args.engine == "neat":
        run_neat(args, experiment_dir, population_dir, query_dir)
    else:
        run_neurogram(
            args,
            experiment_dir,
            population_dir,
            query_dir,
            select_parents_from_grid=select_parents_from_grid,
            decode_image_fn=decode_image,
            write_run_metadata=write_run_metadata,
        )

    _process_selection_gif(
        experiment_dir,
        args.gif_output_name,
        args.gif_duration,
        args.gif_frame_mode,
        render_structure_plot=False,
    )


if __name__ == "__main__":
    main()
