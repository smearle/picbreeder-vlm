from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import neat
import pygame

from artifacts import build_generation_state, save_neat_population
from neat_components import (
    InteractiveStagnation,
    PicbreederGenome,
    apply_picbreeder_config_defaults,
    seed_initial_population,
    sync_population_output_activations,
)
from picbreeder_reproduction import PicbreederReproduction
from rendering import create_numbered_grid, render_genome_image


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MARGIN = 12
INFO_BAR_HEIGHT = 80
STATUS_DURATION_MS = 4000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively evolve CPPN-based images using NEAT-Python with Picbreeder defaults.",
    )
    parser.add_argument("--rows", type=int, default=4, help="Number of rows in the genome grid.")
    parser.add_argument("--cols", type=int, default=5, help="Number of columns in the genome grid.")
    parser.add_argument("--thumb-size", type=int, default=200, help="Thumbnail size for each genome preview.")
    parser.add_argument(
        "--scheme",
        choices=("color", "gray"),
        default="gray",
        help="Rendering scheme for the CPPN outputs.",
    )
    parser.add_argument(
        "--color-palette",
        choices=("hsb", "sigmoid"),
        default="hsb",
        help="Palette to use when rendering color outputs.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Path to the NEAT configuration file. Defaults to the picture2d presets.",
    )
    parser.add_argument(
        "--output-activations",
        action="store_true",
        help="Enable CPPN output activation functions during mutation.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=REPO_ROOT / "interactive_runs",
        help="Directory where snapshots and renders should be stored.",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        default=600,
        help="Width/height of saved high-resolution renders (square).",
    )
    parser.add_argument(
        "--select-k",
        type=int,
        default=None,
        help="Maximum number of parents you may pick each generation (defaults to unlimited).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Upper bound on UI refresh rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to initialize Python's RNG.",
    )
    return parser.parse_args()


def resolve_config_path(config_path: Optional[Path], scheme: str) -> Path:
    if config_path is not None:
        return config_path.resolve()
    base = REPO_ROOT / "picture2d"
    fallback = "interactive_config_color" if scheme == "color" else "interactive_config_gray"
    return (base / fallback).resolve()


class PygameGridUI:
    def __init__(self, rows: int, cols: int, thumb: int, fps: int, title: str = "Picbreeder Interactive Evolution"):
        pygame.init()
        self.rows = rows
        self.cols = cols
        self.thumb = thumb
        self.margin = DEFAULT_MARGIN
        self.info_height = INFO_BAR_HEIGHT
        self.fps = max(5, fps)
        self.clock = pygame.time.Clock()
        self.status_message: str = ""
        self.status_timestamp = 0

        grid_width = (cols * thumb) + ((cols + 1) * self.margin)
        grid_height = (rows * thumb) + ((rows + 1) * self.margin)
        self.window_size = (grid_width, grid_height + self.info_height)
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont(None, 24)

        self._last_selected: Optional[Tuple[int, ...]] = None
        self._grid_surface: Optional[pygame.Surface] = None
        self._grid_state: Optional[Dict[str, object]] = None

    def close(self) -> None:
        pygame.quit()

    def set_status(self, message: str) -> None:
        self.status_message = message
        self.status_timestamp = pygame.time.get_ticks()

    def select_parents(
        self,
        state: Dict[str, object],
        generation: int,
        on_high_res: Callable[[int], Optional[Path]],
        select_limit: Optional[int],
    ) -> List[int]:
        selected: Set[int] = set()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        raise SystemExit
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        if select_limit is None or len(selected) <= select_limit:
                            return sorted(selected)
                        self.set_status(f"Selection limit is {select_limit}. Deselect something first.")
                    if event.key == pygame.K_BACKSPACE:
                        selected.clear()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    idx = self._index_for_position(event.pos)
                    if idx is None:
                        continue
                    if event.button == 1:
                        if idx in selected:
                            selected.remove(idx)
                        else:
                            if select_limit is not None and len(selected) >= select_limit:
                                self.set_status(f"Selection limit is {select_limit}.")
                            else:
                                selected.add(idx)
                    elif event.button == 3:
                        path = on_high_res(idx)
                        if path is not None:
                            self.set_status(f"Saved render to {path.name}")

            self._render(state, generation, selected, select_limit)
            self.clock.tick(self.fps)

        return sorted(selected)

    def _render(
        self,
        state: Dict[str, object],
        generation: int,
        selected: Set[int],
        select_limit: Optional[int],
    ) -> None:
        surface = self._grid_surface_for(state, selected)
        self.screen.fill((10, 10, 16))
        self.screen.blit(surface, (0, self.info_height))
        self._draw_overlay(generation, len(selected), select_limit)
        pygame.display.flip()

    def _draw_overlay(self, generation: int, selected_count: int, select_limit: Optional[int]) -> None:
        lines = [
            f"Generation {generation}  |  Selected: {selected_count}"
            + (f"/{select_limit}" if select_limit is not None else ""),
            "Left click: toggle selection    Right click: save high-res render",
            "Press Enter/Space to evolve next generation. Backspace clears selections.",
            "Press Esc or Q to quit.",
        ]
        now = pygame.time.get_ticks()
        if self.status_message and (now - self.status_timestamp) < STATUS_DURATION_MS:
            lines.append(f"Status: {self.status_message}")
        elif self.status_message:
            self.status_message = ""

        y = 10
        for line in lines:
            text_surface = self.font.render(line, True, (240, 240, 240))
            self.screen.blit(text_surface, (12, y))
            y += text_surface.get_height() + 4

    def _grid_surface_for(self, state: Dict[str, object], selected: Iterable[int]) -> pygame.Surface:
        selected_tuple = tuple(sorted(int(idx) for idx in selected))
        if self._grid_state is state and self._last_selected == selected_tuple and self._grid_surface is not None:
            return self._grid_surface

        image = create_numbered_grid(state, selected=selected_tuple)
        data = image.tobytes()
        surface = pygame.image.frombuffer(data, image.size, image.mode).convert()
        self._grid_surface = surface
        self._last_selected = selected_tuple
        self._grid_state = state
        return surface

    def _index_for_position(self, position: Tuple[int, int]) -> Optional[int]:
        x, y = position
        y -= self.info_height
        if y < 0:
            return None

        margin = self.margin
        thumb = self.thumb

        col = (x - margin) // (thumb + margin)
        row = (y - margin) // (thumb + margin)

        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return None

        in_x = (x - margin) % (thumb + margin)
        in_y = (y - margin) % (thumb + margin)

        if in_x >= thumb or in_y >= thumb:
            return None

        return int(row * self.cols + col)


class HumanDrivenEvolver:
    def __init__(
        self,
        population: neat.Population,
        ui: PygameGridUI,
        output_dir: Path,
        rows: int,
        cols: int,
        thumb_size: int,
        scheme: str,
        palette: str,
        render_size: int,
        select_limit: Optional[int],
    ) -> None:
        self.population = population
        self.ui = ui
        self.output_dir = output_dir
        self.rows = rows
        self.cols = cols
        self.thumb_size = thumb_size
        self.scheme = scheme
        self.palette = palette
        self.render_size = max(thumb_size, render_size)
        self.select_limit = select_limit
        self.snapshot_dir = output_dir / "snapshots"
        self.render_dir = output_dir / "renders"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.render_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_generation(self, genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config) -> None:
        expected = self.rows * self.cols
        if len(genomes) != expected:
            raise ValueError(f"Expected {expected} genomes, received {len(genomes)}.")

        generation = int(self.population.generation)
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
        save_neat_population(state, self.snapshot_dir, generation, cache)

        def render_high_res(index: int) -> Optional[Path]:
            if index < 0 or index >= len(genomes):
                return None
            genome_id, genome = genomes[index]
            image = render_genome_image(
                genome,
                config,
                self.render_size,
                self.render_size,
                self.scheme,
                self.palette,
            )
            filename = f"gen_{generation:03d}_idx_{index:02d}_id_{genome_id}.png"
            path = self.render_dir / filename
            image.save(path, format="PNG")
            return path

        selected = self.ui.select_parents(state, generation, render_high_res, self.select_limit)

        if not selected:
            print("No parents selected; best genome will be preserved automatically.")
        else:
            print(f"Selected indices: {selected}")

        for idx, (_, genome) in enumerate(genomes):
            genome.fitness = 1.0 if idx in selected else 0.0


def build_population(args: argparse.Namespace) -> neat.Population:
    config_path = resolve_config_path(args.config_path, args.scheme)
    if not config_path.exists():
        raise FileNotFoundError(f"NEAT configuration not found at {config_path}")
    if args.seed is not None:
        random.seed(args.seed)
    config = neat.Config(
        PicbreederGenome,
        PicbreederReproduction,
        neat.DefaultSpeciesSet,
        InteractiveStagnation,
        str(config_path),
    )
    apply_picbreeder_config_defaults(config, enable_output_activations=args.output_activations)
    config.pop_size = args.rows * args.cols
    population = neat.Population(config)
    sync_population_output_activations(population, config.genome_config)
    seed_initial_population(population, config.genome_config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    return population


def ensure_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("interactive_%Y%m%d-%H%M%S")
    if base_dir.exists() and not base_dir.is_dir():
        raise ValueError(f"Cannot use '{base_dir}' as a directory; path exists and is not a directory.")
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    args = parse_args()
    population = build_population(args)
    output_dir = ensure_output_dir(args.save_dir.resolve())
    ui = PygameGridUI(args.rows, args.cols, args.thumb_size, args.fps)
    evolver = HumanDrivenEvolver(
        population,
        ui,
        output_dir,
        args.rows,
        args.cols,
        args.thumb_size,
        args.scheme,
        args.color_palette,
        args.render_size,
        args.select_k,
    )

    try:
        while True:
            population.run(evolver.evaluate_generation, 1)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        ui.close()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pygame.quit()
        sys.exit()
