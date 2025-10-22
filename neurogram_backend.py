import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PIL import Image


DEFAULT_MODULE_PATH = Path(__file__).resolve().parent / "neurogram_standalone.js"


def to_python(value: Any) -> Any:
    return value.valueOf() if hasattr(value, "valueOf") else value


def resolve_neurogram_resume_snapshot(population_dir: Path, generation: int | None) -> Path:
    if generation is not None:
        path = population_dir / f"gen_{generation:03d}_state.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Requested snapshot for generation {generation} not found at {path}"
            )
        return path

    candidates = sorted(population_dir.glob("gen_*_state.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No saved state snapshots found in '{population_dir}'."
        )
    return candidates[-1]


def _save_neurogram_population(
    neurogram: Any,
    state: Dict[str, Any],
    output_dir: Path,
    generation: int,
    decode_image_fn: Callable[[Dict[str, Any]], Image.Image],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in state["images"]:
        idx = int(entry["index"])
        image = decode_image_fn(entry)
        image_path = output_dir / f"gen_{generation:03d}_idx_{idx:02d}.png"
        image.save(image_path, format="PNG")
        meta = {
            "generation": generation,
            "index": idx,
            "row": int(entry["row"]),
            "col": int(entry["col"]),
            "width": int(entry["width"]),
            "height": int(entry["height"]),
        }
        meta_path = image_path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    snapshot = to_python(neurogram.exportState({"includeImages": False}))
    state_path = output_dir / f"gen_{generation:03d}_state.json"
    state_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return state_path


def run_neurogram(
    args: argparse.Namespace,
    experiment_dir: Path,
    population_dir: Path,
    query_dir: Path,
    *,
    select_parents_from_grid: Callable[[Dict[str, Any], str, Path, Optional[int], Optional[str]], Dict[str, Any]],
    decode_image_fn: Callable[[Dict[str, Any]], Image.Image],
    write_run_metadata: Callable[[Path, argparse.Namespace], None],
) -> None:
    try:
        import javascript  # type: ignore
    except ImportError as exc:
        raise SystemExit("The javascript bridge (py_mini_racer) is required for the Neurogram backend.") from exc

    module_path = args.module_path or DEFAULT_MODULE_PATH
    neurogram = javascript.require(str(module_path))

    if args.resume_dir:
        snapshot_path = resolve_neurogram_resume_snapshot(population_dir, args.resume_generation)
        snapshot_data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        neurogram.loadState(snapshot_data)
        state = to_python(neurogram.getState())
        args.rows = snapshot_data["rows"]
        args.cols = snapshot_data["cols"]
        args.thumb_size = snapshot_data["thumbSize"]
        print(
            f"Resuming from generation {state['generation']}: "
            f"{snapshot_path.relative_to(experiment_dir)}"
        )
    else:
        init_state = neurogram.initPop(
            {
                "rows": args.rows,
                "cols": args.cols,
                "thumbSize": args.thumb_size,
            }
        )
        state = to_python(init_state)
        write_run_metadata(experiment_dir, args)
        print(f"Starting new Neurogram experiment in {experiment_dir}")

    for _ in range(args.generations):
        generation = int(state["generation"])
        print(f"\n--- Generation {generation} ---")

        state_path = _save_neurogram_population(
            neurogram,
            state,
            population_dir,
            generation,
            decode_image_fn,
        )
        selection_meta = select_parents_from_grid(
            state,
            args.prompt,
            query_dir,
            args.select_k,
            args.system_instruction,
            args.chat_history_turns,
        )
        selected = selection_meta["selected"]
        rationale = selection_meta.get("rationale") or "(no rationale)"

        try:
            relative_state = state_path.relative_to(experiment_dir)
        except ValueError:
            relative_state = state_path

        print(f"Selected indices: {selected}")
        print(f"Rationale: {rationale}")
        print(f"Snapshot saved to {relative_state}")

        state = to_python(neurogram.evolve(selected))

    print(f"\nRun complete. Next generation index: {state['generation']}")
