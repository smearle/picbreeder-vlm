import argparse
from pathlib import Path
from typing import Optional, Sequence


SELECTION_BASELINES: Sequence[str] = ("none", "random", "max-depth", "max-nodes", "siglip")


def add_experiment_cli_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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
        "--elitism",
        type=int,
        default=None,
        help="Elites per species preserved unchanged (DefaultReproduction; used with 'siglip').",
    )
    parser.add_argument(
        "--target-label",
        type=str,
        default=None,
        help="Target text label used by text–image similarity selection (e.g., when using 'siglip').",
    )
    parser.add_argument(
        "--siglip-model",
        type=str,
        default="google/siglip-base-patch16-224",
        help="Hugging Face model id for SigLIP zero-shot classification.",
    )
    parser.add_argument(
        "--siglip-device",
        default=None,
        help="Device for SigLIP pipeline (e.g., 0 for CUDA device 0, or 'cpu').",
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
        help="Number of initial populations to generate and save instead of running evolution.",
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
    return parser


def cap_select_k_for_engine(engine: str, select_k: Optional[int]) -> Optional[int]:
    if select_k is None:
        return None
    if engine == "neurogram":
        return min(select_k, 4)
    return select_k


def build_experiment_slug(args: argparse.Namespace) -> str:
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
        if args.output_activations:
            slug_parts.append("outact")
    if args.select_k is not None:
        slug_parts.append(f"sk{args.select_k}")
    if args.selection_baseline != "none":
        slug_parts.append(f"baseline-{args.selection_baseline}")
    return "_".join(slug_parts)
