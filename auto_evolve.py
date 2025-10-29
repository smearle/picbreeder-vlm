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
import configparser
import sys

try:
    import graphviz  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    graphviz = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent

# Optional imports for SigLIP; only needed if selection-baseline is 'siglip'
try:
    import torch  # type: ignore
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    pipeline = None  # type: ignore

import neat
from neat.checkpoint import Checkpointer
from neat.reproduction import DefaultReproduction
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
# import im_query  # type: ignore
from render_experiment_results import (  # type: ignore
    process_experiment as _process_selection_gif,
    render_population_structure_plots_from_experiment as _render_population_plots,
)
from picbreeder_reproduction import PicbreederReproduction
from experiment_cli import (
    SELECTION_BASELINES,
    add_experiment_cli_arguments,
    build_experiment_slug,
    cap_select_k_for_engine,
)

DEFAULT_BASELINE_SELECTION_LIMIT = 1
_CHAT_SESSION: Optional[Any] = None
_CHAT_SESSION_MAX_TURNS: Optional[int] = None


GOAL_PROMPT = (
    "Your goal is to evolve images that resemble familiar real-world objects."
    # "Your goal is to evolve images that resemble familiar real-world objects. We want the object to be colored, but try to move away from the high-frequency rainbow artefact. "
    # "Your goal is to evolve an image that looks like a fish."
    # "Your goal is to generate a lizard."
)


DEFAULT_SYSTEM_INSTRUCTION = (
    "You are playing with an online platform which evolves small neural networks called Compositional Pattern Producing Networks (CPPNs) to generate images. "
    f"{GOAL_PROMPT} "
    "At each generation, you are shown a grid of numbered images produced by different CPPNs. "
    "{selection_prompt}"
    "The point is not to pick the most interesting images relative to the others in the grid, necessarily, but rather to pick images that you want to mutate and evolve further. "
    # "(Also, for debugging, please tell me how many previous grids you see in the chat history, and describe the overall evolution progress so far.) "
    "Respond with JSON only: {{\"selected\": [indices], \"rationale\": \"brief explanation\"}}. "
)


DEFAULT_PROMPT = "Grid at generation {generation}:"



# ================================
# Siglip stuff
# ================================
_SIGLIP_PIPELINE = None
def _get_siglip_pipeline(model_name: str = "google/siglip-base-patch16-224",
                         device: Optional[str|int] = None):
    global _SIGLIP_PIPELINE
    if _SIGLIP_PIPELINE is not None:
        return _SIGLIP_PIPELINE
    if pipeline is None:
        raise ImportError("'transformers' is required for SigLIP selection.")
    init_kwargs: Dict[str, Any] = {"task": "zero-shot-image-classification", "model": model_name}
    if device is not None and str(device).lower() != "cpu":
        try:
            init_kwargs["device"] = int(device)
            if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():  # type: ignore[attr-defined]
                init_kwargs["dtype"] = torch.bfloat16  # type: ignore[attr-defined]
        except Exception:
            pass
    _SIGLIP_PIPELINE = pipeline(**init_kwargs)
    return _SIGLIP_PIPELINE

def select_parents_by_text_similarity(  state: Dict[str, Any],
                                        label: str,
                                        select_k: int,
                                        *,
                                        model_name: str = "google/siglip-base-patch16-224",
                                        device: Optional[str|int] = None,
                                        query_dir: Optional[Path] = None,
                                        ) -> Dict[str, Any]:
    generation = int(state["generation"])
    images: List = [decode_image(entry) for entry in state["images"]]
    clf = _get_siglip_pipeline(model_name=model_name, device=device)

    # Batch infer; pipeline returns list per image: [{label, score}, ...]
    outputs: List[List[Dict[str, Any]]] = clf(images, candidate_labels=[label])

    scored: List[tuple[int, float]] = []
    for idx, out in enumerate(outputs):
        # With a single candidate label, out[0]["score"] is the score for `label`.
        score = float(out[0]["score"]) if out and out[0].get("label") == label else 0.0
        scored.append((idx, score))

    scored.sort(key=lambda t: t[1], reverse=True)
    selected = [idx for idx, _ in scored[:max(1, select_k)]]

    # Optional: write artifacts (grid and selection overlays) consistent with existing outputs.
    if query_dir is not None:
        query_dir.mkdir(parents=True, exist_ok=True)
        grid = create_numbered_grid(state)
        grid.save(query_dir / f"gen_{generation:03d}_grid.png", format="PNG")
        sel = create_numbered_grid(state, selected=selected)
        sel.save(query_dir / f"gen_{generation:03d}_selection.png", format="PNG")
        meta = {
            "selected": selected,
            "rationale": f"Top-{select_k} by SigLIP score for '{label}'",
            "scores": [{"index": i, "score": s} for i, s in scored],
        }
        meta_dir = query_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / f"gen_{generation:03d}_selection.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "selected": selected,
        "rationale": f"Top-{select_k} by SigLIP score for '{label}'",
        "scores": [{"index": i, "score": s} for i, s in scored],
    }


def gen_selection_prompt(select_k: Optional[int]) -> str:
    if select_k is None:
        return "Pick one or several images by their numeric labels--the corresponding CPPNs will be used as the parents of the next generation. "
    if select_k == 1:
        return "Pick one image by its numeric label--the corresponding CPPN will be used as the parent of the next generation. "

    return f"Pick up to {select_k} images by their numeric labels--the corresponding CPPNs will be used as the parents of the next generation. "


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
        # save_neat_population(state, population_dir, 0, png_cache)

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


def _session_max_turns(chat_history_turns: Optional[int]) -> Optional[int]:
    if chat_history_turns is None or chat_history_turns < 0:
        return None
    return chat_history_turns


def _ensure_chat_session(chat_history_turns: Optional[int]) -> Any:
    global _CHAT_SESSION, _CHAT_SESSION_MAX_TURNS
    max_turns = _session_max_turns(chat_history_turns)
    if _CHAT_SESSION is None or _CHAT_SESSION_MAX_TURNS != max_turns:
        _CHAT_SESSION = im_query.create_chat_session(max_turns=max_turns)
        _CHAT_SESSION_MAX_TURNS = max_turns
    return _CHAT_SESSION


def _query_with_history(
    image_bytes: bytes,
    prompt: str,
    *,
    system_instruction: Optional[str],
    chat_history_turns: Optional[int],
) -> Any:
    session: im_query.ImageChatSession = _ensure_chat_session(chat_history_turns)
    return session.send(
        image_bytes,
        prompt,
        history_turns=chat_history_turns,
        mime_type="image/png",
        system_instruction=system_instruction,
    )


def select_parents_from_grid(
    state: Dict[str, Any],
    prompt_template: str,
    query_dir: Path,
    select_k: Optional[int] = None,
    system_instruction: Optional[str] = None,
    chat_history_turns: Optional[int] = 0,
) -> Dict[str, Any]:
    generation = int(state["generation"])
    grid_image = create_numbered_grid(state)

    query_dir.mkdir(parents=True, exist_ok=True)
    base_grid_path = query_dir / f"gen_{generation:03d}_grid.png"
    grid_image.save(base_grid_path, format="PNG")

    buffer = BytesIO()
    grid_image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    total_images = len(state["images"])
    max_index = max(total_images - 1, 0)
    prompt = prompt_template.format(generation=generation)

    response = _query_with_history(
        image_bytes,
        prompt=prompt,
        system_instruction=system_instruction,
        chat_history_turns=chat_history_turns,
    )
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
        "chat_history_turns": chat_history_turns,
    }
    metadata_dir = query_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    meta_path = metadata_dir / f"gen_{generation:03d}_selection.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["metadata_path"] = str(meta_path)

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
        system_instruction: Optional[str],
        experiment_dir: Path,
        population_dir: Path,
        query_dir: Path,
        selection_baseline: str = "none",
        select_k: Optional[int] = None,
        chat_history_turns: Optional[int] = 0,
        target_label: Optional[str] = None,
        siglip_model: str = "google/siglip-base-patch16-224",
        siglip_device: Optional[str] = None,
    ) -> None:
        self.population = population
        self.rows = rows
        self.cols = cols
        self.thumb_size = thumb_size
        self.scheme = scheme
        self.palette = palette
        self.prompt = prompt
        self.system_instruction = system_instruction
        self.experiment_dir = experiment_dir
        self.population_dir = population_dir
        self.query_dir = query_dir
        self.population_size = rows * cols
        self.selection_baseline = selection_baseline
        self.select_k = select_k
        self.chat_history_turns = chat_history_turns
        self._diagram_warning_emitted = False
        self.target_label = target_label
        self.siglip_model = siglip_model
        self.siglip_device = siglip_device
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

        if self.selection_baseline == "siglip":
            selection_meta = select_parents_by_text_similarity(
                state,
                label=self.target_label or "",
                select_k=self.select_k or 1,
                model_name=self.siglip_model,
                device=self.siglip_device,
                query_dir=self.query_dir,
            )
        elif self.selection_baseline == "none":
            selection_meta = select_parents_from_grid(
                state,
                self.prompt,
                self.query_dir,
                self.select_k,
                self.system_instruction,
                self.chat_history_turns,
            )
        else:
            selection_meta = self._select_parents_baseline(generation, genomes, config, state)
        selected = selection_meta["selected"]
        rationale = selection_meta.get("rationale") or "(no rationale)"

        # Assign fitness: for SigLIP set score for every genome; otherwise binary.
        if self.selection_baseline == "siglip":
            score_map: Dict[int, float] = {int(s["index"]): float(s["score"]) for s in selection_meta.get("scores", [])}
            for idx, (_, genome) in enumerate(genomes):
                genome.fitness = float(score_map.get(idx, 0.0))
        else:
            for idx, (_, genome) in enumerate(genomes):
                genome.fitness = 1.0 if idx in selected else 0.0

        print(f"Selected indices: {selected}")
        print(f"Rationale: {rationale}")    
        # print(f"Snapshot saved to {state_path}")
        print(f"Selection image saved to {selection_meta.get('selection_path')}")
        if self.selection_baseline == "siglip":
            try:
                score_map = {int(s["index"]): float(s["score"]) for s in selection_meta.get("scores", [])}
                all_scores: List[float] = [float(score_map.get(i, 0.0)) for i in range(len(genomes))]
                if all_scores:
                    avg = mean(all_scores)
                    mn = min(all_scores)
                    mx = max(all_scores)
                    st = pstdev(all_scores) if len(all_scores) > 1 else 0.0
                    print(f"SigLIP fitness - avg: {avg:.4f}, min: {mn:.4f}, max: {mx:.4f}, std: {st:.4f}")
                top_n = self.select_k or 5
                top_pairs = sorted(score_map.items(), key=lambda t: t[1], reverse=True)[: max(1, top_n)]
                if top_pairs:
                    summary = ", ".join(f"{idx}:{score:.4f}" for idx, score in top_pairs)
                    print(f"Top fitness: {summary}")
            except Exception:
                pass
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

        metadata_dir = self.query_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        meta_path = metadata_dir / f"gen_{generation:03d}_selection.json"
        meta_path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
        enriched["metadata_path"] = str(meta_path)
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
    slug = build_experiment_slug(args)
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
        "system_instruction": args.system_instruction,
        "chat_history_turns": args.chat_history_turns,
        "engine": args.engine,
        "selection_baseline": args.selection_baseline,
        "select_k": args.select_k,
    }
    if args.engine == "neat":
        metadata["scheme"] = args.scheme
        metadata["config_path"] = str(args.config_path) if args.config_path else None
        metadata["color_palette"] = args.color_palette
        metadata["output_activations"] = args.output_activations
    else:
        metadata["module_path"] = str(args.module_path) if args.module_path else None
    metadata_path = experiment_dir / "run_config.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_initial_prompt(experiment_dir: Path, prompt: str) -> Path:
    prompt_path = experiment_dir / "initial_prompt.txt"
    if not prompt_path.exists():
        prompt_path.write_text(prompt, encoding="utf-8")
    return prompt_path


def save_initial_system_instruction(experiment_dir: Path, instruction: str) -> Path:
    instruction_path = experiment_dir / "initial_system_instruction.txt"
    if not instruction_path.exists():
        instruction_path.write_text(instruction, encoding="utf-8")
    return instruction_path


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
    add_experiment_cli_arguments(parser)
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
    if args.selection_baseline == "siglip":
        if not getattr(args, "target_label", None):
            raise ValueError("--target-label is required when using selection baseline 'siglip'.")

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

    if args.chat_history_turns < -1:
        raise ValueError("chat-history-turns must be at least -1 (-1 for unlimited)")

    if args.gif_duration <= 0:
        raise ValueError("gif-duration must be a positive integer")


def _ensure_default_reproduction_section(config_path: Path, experiment_dir: Path, elitism: Optional[int]) -> Path:
    """Ensure the NEAT config has a [DefaultReproduction] section and optional elitism override.
    If we add the section or change values, write a derived config into the experiment directory and return its path.
    Otherwise return the original path.
    """
    parser = configparser.ConfigParser()
    parser.read(config_path)
    changed = False
    if not parser.has_section("DefaultReproduction"):
        parser.add_section("DefaultReproduction")
        changed = True

    # Ensure survival_threshold has some sensible default if absent
    if not parser.has_option("DefaultReproduction", "survival_threshold"):
        parser.set("DefaultReproduction", "survival_threshold", "0.2")
        changed = True

    # Apply elitism if provided; otherwise ensure at least present
    if elitism is not None:
        current = parser.get("DefaultReproduction", "elitism", fallback=None)
        if current is None or current != str(int(elitism)):
            parser.set("DefaultReproduction", "elitism", str(int(elitism)))
            changed = True
    else:
        if not parser.has_option("DefaultReproduction", "elitism"):
            parser.set("DefaultReproduction", "elitism", "0")
            changed = True

    if not changed:
        return config_path

    derived = experiment_dir / "config_with_default_reproduction.cfg"
    with derived.open("w", encoding="utf-8") as fh:
        parser.write(fh)
    return derived


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
        # Use default reproduction when 'siglip' selection baseline is active; otherwise Picbreeder reproduction.
        reproduction_cls = DefaultReproduction if args.selection_baseline == "siglip" else PicbreederReproduction
        effective_config_path = args.config_path
        if reproduction_cls is DefaultReproduction and effective_config_path is not None:
            effective_config_path = _ensure_default_reproduction_section(
                effective_config_path, experiment_dir, getattr(args, "elitism", None)
            )
            args.config_path = effective_config_path
        config = neat.Config(
            PicbreederGenome,
            reproduction_cls,
            neat.DefaultSpeciesSet,
            InteractiveStagnation,
            str(effective_config_path),
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
        args.system_instruction,
        experiment_dir,
        population_dir,
        query_dir,
        selection_baseline=args.selection_baseline,
        select_k=args.select_k,
        chat_history_turns=args.chat_history_turns,
        target_label=getattr(args, "target_label", None),
        siglip_model=getattr(args, "siglip_model", "google/siglip-base-patch16-224"),
        siglip_device=getattr(args, "siglip_device", None),
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

    args.prompt = DEFAULT_PROMPT
    args.system_instruction = DEFAULT_SYSTEM_INSTRUCTION.format(
        selection_prompt=gen_selection_prompt(args.select_k)
    )
    # save_initial_prompt(experiment_dir, args.prompt)
    save_initial_system_instruction(experiment_dir, args.system_instruction)

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
        stored_chat_history = metadata.get("chat_history_turns")
        if stored_chat_history is not None:
            stored_chat_history = int(stored_chat_history)
            if stored_chat_history != args.chat_history_turns:
                print(
                    f"Resume directory chat-history-turns '{stored_chat_history}' overrides CLI value '{args.chat_history_turns}'."
                )
            args.chat_history_turns = stored_chat_history
        stored_prompt = metadata.get("prompt")
        if stored_prompt:
            args.prompt = stored_prompt

        stored_system_instruction = metadata.get("system_instruction")
        if stored_system_instruction:
            args.system_instruction = stored_system_instruction

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
