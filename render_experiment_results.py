#!/usr/bin/env python3
"""
Render visual artifacts for completed experiments.

For each experiment directory matching the provided pattern (e.g. logs/g200_*),
this script assembles GIFs summarizing generation selections and saves plots that
describe how the CPPN population evolved structurally across generations.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from metrics_visualization import render_first_selection_grid, render_population_structure_plots_from_experiment
from experiment_cli import add_experiment_cli_arguments, build_experiment_slug, cap_select_k_for_engine


CAPTION_HEIGHT = 30
CAPTION_FILL = (255, 255, 255)
CAPTION_BACKGROUND = (0, 0, 0)
DEFAULT_FONT = ImageFont.load_default(size=20)
COMPRESSED_MAX_WIDTH = 640


@dataclass
class GifFrame:
    """Container describing a single frame in the output GIF."""

    generation: int
    frame_kind: str
    image_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_cli_arguments(parser)
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Root directory that contains experiment subdirectories.",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Optional glob pattern to override slug-based discovery (e.g., 'g200_*').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for how many experiment directories to process.",
    )
    parser.add_argument(
        "--output-name",
        dest="gif_output_name",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--duration",
        dest="gif_duration",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--frame-mode",
        dest="gif_frame_mode",
        choices=("grid", "first-selection"),
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def gather_experiments(logs_dir: Path, pattern: str) -> Iterable[Path]:
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")
    candidates = [p for p in logs_dir.glob(pattern) if p.is_dir()]
    return sorted(candidates, key=lambda path: _experiment_timestamp_key(path.name), reverse=True)


def _experiment_timestamp_key(slug: str) -> Tuple[datetime, str]:
    _, timestamp_text = slug.rsplit("_", 1)
    timestamp = datetime.strptime(timestamp_text, "%Y%m%d-%H%M%S")
    return timestamp, slug


def load_first_selection(selection_json: Path) -> Optional[int]:
    with selection_json.open("r") as fp:
        data = json.load(fp)
    selected = data.get("selected") or []
    if not isinstance(selected, list) or not selected:
        return None
    first = selected[0]
    if not isinstance(first, int):
        raise ValueError(f"Expected integer selection in {selection_json}, got {first!r}")
    return first


def parse_generation_index(generation_prefix: str) -> int:
    try:
        _, number = generation_prefix.split("_", 1)
        return int(number)
    except (ValueError, IndexError):
        raise ValueError(f"Unable to parse generation number from prefix: {generation_prefix}")


def _resolve_recorded_path(path_value: Optional[str], base_dir: Path) -> Optional[Path]:
    if not path_value:
        return None
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def collect_grid_frames(queries_dir: Path, selection_files: List[Path]) -> List[GifFrame]:
    frames: List[GifFrame] = []
    for selection_file in selection_files:
        generation_prefix = selection_file.stem.replace("_selection", "")
        generation_index = parse_generation_index(generation_prefix)
        try:
            payload = json.loads(selection_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}

        recorded_grid = _resolve_recorded_path(payload.get("grid_path"), queries_dir)
        recorded_selection = _resolve_recorded_path(payload.get("selection_path"), queries_dir)

        grid_path = recorded_grid or (queries_dir / f"{generation_prefix}_grid.png")
        selection_path = recorded_selection or (queries_dir / f"{generation_prefix}_selection.png")

        if grid_path.exists():
            frames.append(GifFrame(generation_index, "grid", grid_path))
        else:
            print(f"[WARN] Missing grid image: {grid_path}", file=sys.stderr)

        if selection_path.exists():
            frames.append(GifFrame(generation_index, "selection", selection_path))
        else:
            print(f"[WARN] Missing selection image: {selection_path}", file=sys.stderr)
    return frames


def find_first_selection_image(
    populations_dir: Path,
    generation_prefix: str,
    selected_index: int,
) -> Optional[Path]:
    idx_formats = [f"{selected_index:02d}", f"{selected_index:03d}", str(selected_index)]
    for idx_str in idx_formats:
        candidate = populations_dir / f"{generation_prefix}_idx_{idx_str}.png"
        if candidate.exists():
            return candidate
    return None


def collect_first_selection_frames(
    experiment_dir: Path,
    selection_files: List[Path],
) -> List[GifFrame]:
    populations_dir = experiment_dir / "populations"
    if not populations_dir.exists():
        raise FileNotFoundError(f"Missing populations directory in {experiment_dir}")

    frames: List[GifFrame] = []
    for selection_file in selection_files:
        selected_index = load_first_selection(selection_file)
        if selected_index is None:
            continue
        generation_prefix = selection_file.stem.replace("_selection", "")
        generation_index = parse_generation_index(generation_prefix)
        image_path = find_first_selection_image(populations_dir, generation_prefix, selected_index)
        if image_path is None:
            print(
                f"[WARN] Could not locate image for {generation_prefix} index {selected_index} in {experiment_dir}",
                file=sys.stderr,
            )
            continue
        frames.append(GifFrame(generation_index, "first selection", image_path))
    return frames


def collect_frames(experiment_dir: Path, frame_mode: str) -> List[GifFrame]:
    queries_dir = experiment_dir / "queries"
    if not queries_dir.exists():
        raise FileNotFoundError(f"Missing queries directory in {experiment_dir}")

    metadata_dir = queries_dir / "metadata"
    if metadata_dir.exists():
        selection_files = sorted(metadata_dir.glob("gen_*_selection.json"))
    else:
        selection_files = []

    if not selection_files:
        selection_files = sorted(queries_dir.glob("gen_*_selection.json"))

    if frame_mode == "grid":
        return collect_grid_frames(queries_dir, selection_files)

    if frame_mode == "first-selection":
        return collect_first_selection_frames(experiment_dir, selection_files)

    raise ValueError(f"Unsupported frame mode: {frame_mode}")


def compress_gif_with_ffmpeg(input_path: Path, output_path: Path, source_size: Tuple[int, int]) -> bool:
    palette_path = output_path.with_suffix(".palette.png")
    width, _ = source_size
    needs_scaling = COMPRESSED_MAX_WIDTH is not None and width > COMPRESSED_MAX_WIDTH

    palette_filter = "palettegen=stats_mode=diff"
    paletteuse_filter = (
        "[0:v][1:v]paletteuse="
        "dither=bayer:bayer_scale=5:diff_mode=rectangle"
    )

    if needs_scaling:
        palette_filter = (
            f"scale={COMPRESSED_MAX_WIDTH}:-2:"
            "flags=lanczos,"
            f"{palette_filter}"
        )
        paletteuse_filter = (
            f"[0:v]scale={COMPRESSED_MAX_WIDTH}:-2:flags=lanczos[s0];"
            "[s0][1:v]paletteuse="
            "dither=bayer:bayer_scale=5:diff_mode=rectangle"
        )

    output_path.unlink(missing_ok=True)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-vf",
                palette_filter,
                str(palette_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-i",
                str(palette_path),
                "-lavfi",
                paletteuse_filter,
                "-gifflags",
                "+transdiff",
                "-loop",
                "0",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("[WARN] ffmpeg not found; skipping GIF compression.", file=sys.stderr)
        return False
    except subprocess.CalledProcessError as exc:
        stderr_text = exc.stderr or ""
        trimmed_error = stderr_text.strip().splitlines()
        if trimmed_error:
            print(f"[WARN] ffmpeg failed: {trimmed_error[-1]}", file=sys.stderr)
        else:
            print("[WARN] ffmpeg failed during GIF compression.", file=sys.stderr)
        return False
    finally:
        palette_path.unlink(missing_ok=True)

    return output_path.exists()


def annotate_with_caption(image: Image.Image, frame: GifFrame) -> Image.Image:
    caption_text = f"Generation {frame.generation}"

    width = image.width
    canvas = Image.new(
        "RGB",
        (width, image.height + CAPTION_HEIGHT),
        color=CAPTION_BACKGROUND,
    )
    canvas.paste(image, (0, CAPTION_HEIGHT))

    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), caption_text, font=DEFAULT_FONT)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_x = 10
    text_y = max((CAPTION_HEIGHT - text_height) / 2, 0)
    draw.text((text_x, text_y), caption_text, fill=CAPTION_FILL, font=DEFAULT_FONT)
    return canvas


def _run_ffmpeg_palette_pipeline(
    input_pattern: str,
    output_path: Path,
    fps: float,
) -> None:
    palette_path = output_path.with_suffix(".palette.png")
    fps_str = f"{fps:.6f}".rstrip("0").rstrip(".")
    if not fps_str:
        fps_str = "1"

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                fps_str,
                "-i",
                input_pattern,
                "-vf",
                "palettegen=stats_mode=diff",
                str(palette_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                fps_str,
                "-i",
                input_pattern,
                "-i",
                str(palette_path),
                "-lavfi",
                "[0:v][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                "-gifflags",
                "+transdiff",
                "-loop",
                "0",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        palette_path.unlink(missing_ok=True)


def assemble_gif(frames: List[GifFrame], output_path: Path, duration: int) -> Tuple[int, int]:
    if not frames:
        raise ValueError("No frames available to build GIF.")

    annotated_frames: List[Image.Image] = []
    first_size: Optional[Tuple[int, int]] = None
    fps = 1000.0 / duration if duration > 0 else 1.0

    with tempfile.TemporaryDirectory(prefix="frames_", dir=output_path.parent) as tmpdir:
        tmpdir_path = Path(tmpdir)
        for index, frame in enumerate(frames):
            with Image.open(frame.image_path) as img:
                annotated = annotate_with_caption(img.convert("RGB"), frame)
            annotated_frames.append(annotated)
            if first_size is None:
                first_size = annotated.size
            frame_path = tmpdir_path / f"frame_{index:05d}.png"
            annotated.save(frame_path, format="PNG")

        input_pattern = str(tmpdir_path / "frame_%05d.png")
        try:
            _run_ffmpeg_palette_pipeline(input_pattern, output_path, fps)
        except FileNotFoundError:
            print("[WARN] ffmpeg not found; falling back to Pillow GIF writer.", file=sys.stderr)
        except subprocess.CalledProcessError as exc:
            stderr_text = (exc.stderr or "").strip().splitlines()
            if stderr_text:
                print(f"[WARN] ffmpeg failed: {stderr_text[-1]}", file=sys.stderr)
            else:
                print("[WARN] ffmpeg failed during GIF rendering; falling back to Pillow.", file=sys.stderr)
        else:
            return first_size if first_size is not None else annotated_frames[0].size

    annotated_frames[0].save(
        output_path,
        save_all=True,
        append_images=annotated_frames[1:],
        duration=duration,
        loop=0,
    )
    return first_size if first_size is not None else annotated_frames[0].size


def process_experiment(
    experiment_dir: Path,
    output_name: str,
    duration: int,
    frame_mode: str,
    render_structure_plot: bool = True,
) -> None:
    print(f"[INFO] Processing {experiment_dir}")
    if render_structure_plot:
        try:
            plot_path = render_population_structure_plots_from_experiment(experiment_dir)
        except FileNotFoundError:
            print(
                f"[INFO] No population metrics found in {experiment_dir}; skipping structure plot.",
                file=sys.stderr,
            )
        except RuntimeError as exc:
            print(
                f"[WARN] Unable to render population structure plot for {experiment_dir}: {exc}",
                file=sys.stderr,
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(
                f"[WARN] Unexpected error rendering population metrics plot for {experiment_dir}: {exc}",
                file=sys.stderr,
            )
        else:
            print(f"[INFO] Wrote population structure plot to {plot_path}")

    frames = collect_frames(experiment_dir, frame_mode=frame_mode)
    if not frames:
        print(f"[INFO] Skipping {experiment_dir} (no frames).")
        return

    render_first_selection_grid(experiment_dir)

    output_path = experiment_dir / output_name
    size = assemble_gif(frames, output_path, duration)
    print(f"[INFO] Wrote GIF to {output_path}")

    # add _compressed to filename
    compressed_output = output_path.with_name(
        output_path.stem + "_compressed" + output_path.suffix
    )
    if compress_gif_with_ffmpeg(output_path, compressed_output, size):
        print(f"[INFO] Wrote compressed GIF to {compressed_output}")
    else:
        compressed_output.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.select_k is not None:
        if args.select_k < 1:
            raise SystemExit("select-k must be at least 1 when provided.")
        args.select_k = cap_select_k_for_engine(args.engine, args.select_k)

    slug = build_experiment_slug(args)
    pattern = args.pattern or f"{slug}_*"

    try:
        experiments = list(gather_experiments(args.logs_dir, pattern))
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    if args.pattern is None:
        experiments = [exp for exp in experiments if exp.name.startswith(f"{slug}_")]

    if args.limit is not None:
        experiments = experiments[: args.limit]

    if not experiments:
        if args.pattern is None:
            message = f"No experiments matched slug '{slug}' in {args.logs_dir}"
        else:
            message = f"No experiments matched pattern '{pattern}' in {args.logs_dir}"
        print(f"[WARN] {message}", file=sys.stderr)
        return

    if args.pattern is None:
        print(f"[INFO] Rendering experiments for slug '{slug}' (pattern '{pattern}')")
    else:
        print(f"[INFO] Rendering experiments matching pattern '{pattern}'")
    for experiment_dir in experiments:
        try:
            process_experiment(
                experiment_dir,
                args.gif_output_name,
                args.gif_duration,
                args.gif_frame_mode,
            )
        except Exception as exc:
            print(f"[ERROR] Failed to process {experiment_dir}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
