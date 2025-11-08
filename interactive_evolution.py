from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

import neat
import numpy as np
from PIL import Image
import pygame

from artifacts import build_generation_state, save_neat_genome_diagrams, save_neat_population
from neat_components import (
    InteractiveStagnation,
    PicbreederGenome,
    apply_picbreeder_config_defaults,
    seed_initial_population,
    sync_population_output_activations,
)
from picbreeder_reproduction import PicbreederReproduction
from picture2d.common import eval_color_image, eval_genome_as_grayscale_and_color
from rendering import create_numbered_grid, render_genome_image


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MARGIN = 12
INFO_BAR_HEIGHT = 120
STATUS_DURATION_MS = 4000


class MutationToggleControls(NamedTuple):
    get_mode: Callable[[], str]
    cycle_mode: Callable[[], str]
    set_mode: Callable[[str], str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively evolve CPPN-based images using NEAT-Python with Picbreeder defaults.",
    )
    parser.add_argument("--rows", type=int, default=4, help="Number of rows in the genome grid.")
    parser.add_argument("--cols", type=int, default=5, help="Number of columns in the genome grid.")
    parser.add_argument("--thumb-size", type=int, default=200, help="Thumbnail size for each genome preview.")
    parser.add_argument(
        "--scheme",
        choices=("color", "gray", "toggle"),
        default="gray",
        help="Rendering scheme for the CPPN outputs.",
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
    parser.add_argument(
        "--save-genome-diagrams",
        action="store_true",
        help="Export network topology diagrams for every genome each generation (requires graphviz).",
    )
    parser.add_argument(
        "--save-gray-renders",
        action="store_true",
        help="Also export grayscale renders for every genome each generation.",
    )
    return parser.parse_args()


def resolve_config_path(config_path: Optional[Path], scheme: str) -> Path:
    if config_path is not None:
        return config_path.resolve()
    base = REPO_ROOT / "picture2d"
    fallback = "interactive_config_color" if (scheme == "color" or scheme == "toggle") else "interactive_config_gray"
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
        self._grid_base_size = (grid_width, grid_height)
        base_window_height = grid_height + self.info_height
        self.window_size, initial_scale = self._initial_window_size((grid_width, base_window_height))
        self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont(None, 24)
        self._grid_draw_size: Tuple[int, int] = self._grid_base_size
        self._grid_offset: Tuple[int, int] = (0, self.info_height)
        self._display_scale = 1.0
        self._update_scale_for_window(self.window_size, scale_override=initial_scale)

        self._last_selected: Optional[Tuple[int, ...]] = None
        self._grid_surface: Optional[pygame.Surface] = None
        self._grid_state: Optional[Dict[str, object]] = None
        self._mutation_controls: Optional[MutationToggleControls] = None
        self._mutation_mode_label: Optional[str] = None

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
        mutation_controls: Optional[MutationToggleControls] = None,
    ) -> List[int]:
        selected: Set[int] = set()
        running = True
        self._mutation_controls = mutation_controls
        if mutation_controls is not None:
            self._mutation_mode_label = mutation_controls.get_mode()
        else:
            self._mutation_mode_label = None
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise SystemExit
                    if event.type == pygame.VIDEORESIZE:
                        self.window_size = (event.w, event.h)
                        self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
                        self._update_scale_for_window(self.window_size)
                        continue
                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            raise SystemExit
                        if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                            if select_limit is None or len(selected) <= select_limit:
                                return sorted(selected)
                            self.set_status(f"Selection limit is {select_limit}. Deselect something first.")
                        if event.key == pygame.K_BACKSPACE:
                            selected.clear()
                        if self._mutation_controls is not None:
                            if event.key == pygame.K_t:
                                label = self._mutation_controls.cycle_mode()
                                self._mutation_mode_label = label
                                self.set_status(f"Mutation mode: {label}")
                            else:
                                key_to_mode = {
                                    pygame.K_1: "all",
                                    pygame.K_2: "color_only",
                                    pygame.K_3: "structure_only",
                                }
                                target = key_to_mode.get(event.key)
                                if target is not None:
                                    label = self._mutation_controls.set_mode(target)
                                    self._mutation_mode_label = label
                                    self.set_status(f"Mutation mode: {label}")
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
        finally:
            self._mutation_controls = None

    def _render(
        self,
        state: Dict[str, object],
        generation: int,
        selected: Set[int],
        select_limit: Optional[int],
    ) -> None:
        surface = self._grid_surface_for(state, selected)
        draw_surface = surface
        if self._grid_draw_size != surface.get_size():
            draw_surface = pygame.transform.smoothscale(surface, self._grid_draw_size)
        self.screen.fill((10, 10, 16))
        offset_x, offset_y = self._grid_offset
        self.screen.blit(draw_surface, (offset_x, offset_y))
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
        if self._mutation_mode_label:
            lines.append(
                f"Mutation mode: {self._mutation_mode_label}  (T cycle, 1=All, 2=Color, 3=Structure)"
            )
        now = pygame.time.get_ticks()
        if self.status_message and (now - self.status_timestamp) < STATUS_DURATION_MS:
            # lines.append(f"Status: {self.status_message}")
            pass
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
        offset_x, offset_y = self._grid_offset
        x -= offset_x
        y -= offset_y
        if x < 0 or y < 0:
            return None

        scale = self._display_scale if self._display_scale > 0 else 1.0
        x = x / scale
        y = y / scale

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

    def _initial_window_size(self, base_window: Tuple[int, int]) -> Tuple[Tuple[int, int], float]:
        info = pygame.display.Info()
        screen_width = max(640, getattr(info, "current_w", base_window[0]))
        screen_height = max(480, getattr(info, "current_h", base_window[1]))
        grid_width, grid_height = self._grid_base_size
        max_grid_height = max(1, screen_height - self.info_height)
        limit_scale = min(
            1.0,
            screen_width / float(grid_width),
            max_grid_height / float(grid_height),
        )
        preferred_scale = 0.9
        scale = min(preferred_scale, limit_scale)
        scale = max(0.4, scale)
        width = int(grid_width * scale)
        height = int(grid_height * scale) + self.info_height
        return (width, height), scale

    def _update_scale_for_window(
        self,
        window_size: Tuple[int, int],
        *,
        scale_override: Optional[float] = None,
    ) -> None:
        width, height = window_size
        grid_width, grid_height = self._grid_base_size
        available_height = max(1, height - self.info_height)
        if scale_override is not None:
            scale = scale_override
        else:
            scale = min(width / float(grid_width), available_height / float(grid_height))
        scale = max(0.3, min(scale, 2.5))
        draw_width = max(1, int(grid_width * scale))
        draw_height = max(1, int(grid_height * scale))
        offset_x = max(0, (width - draw_width) // 2)
        offset_y = max(0, (available_height - draw_height) // 2)
        self._display_scale = scale
        self._grid_draw_size = (draw_width, draw_height)
        self._grid_offset = (offset_x, self.info_height + offset_y)


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
        render_size: int,
        select_limit: Optional[int],
        save_genome_diagrams: bool,
        save_gray_renders: bool,
    ) -> None:
        self.population = population
        self.ui = ui
        self.output_dir = output_dir
        self.rows = rows
        self.cols = cols
        self.thumb_size = thumb_size
        self.scheme = scheme
        self.render_size = max(thumb_size, render_size)
        self.select_limit = select_limit
        self.save_genome_diagrams = save_genome_diagrams
        self.save_gray_renders = save_gray_renders
        self.snapshot_dir = output_dir / "snapshots"
        self.render_dir = output_dir / "renders"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.render_dir.mkdir(parents=True, exist_ok=True)
        if self.save_gray_renders:
            self.gray_render_dir = output_dir / "renders_gray"
            self.gray_render_dir.mkdir(parents=True, exist_ok=True)
            self._gray_notice_emitted = False
        else:
            self.gray_render_dir = None
            self._gray_notice_emitted = True
        self._diagram_warning_emitted = False
        self._mutation_modes: Tuple[str, ...] = ("all", "color_only", "structure_only")
        initial_mode = getattr(population.config.genome_config, "picbreeder_mutation_mode", "all")
        self._mutation_mode = self._normalize_mutation_mode(initial_mode)
        self._apply_mutation_mode(population.config, self._mutation_mode)

    def _normalize_mutation_mode(self, mode: str) -> str:
        candidate = str(mode).lower()
        if candidate in self._mutation_modes:
            return candidate
        if candidate in ("color", "coloronly", "color-only"):
            return "color_only"
        if candidate in ("structure", "structureonly", "structure-only"):
            return "structure_only"
        return "all"

    def _describe_mutation_mode(self, mode: Optional[str] = None) -> str:
        current = self._normalize_mutation_mode(mode if mode is not None else self._mutation_mode)
        descriptions = {
            "all": "All channels (structure + color)",
            "color_only": "Color channels only (Hue/Sat)",
            "structure_only": "Structure channel only (Brightness)",
        }
        return descriptions.get(current, current)

    def _apply_mutation_mode(self, config: neat.Config, mode: str) -> None:
        normalized = self._normalize_mutation_mode(mode)
        setattr(config, "picbreeder_mutation_mode", normalized)
        setattr(config.genome_config, "picbreeder_mutation_mode", normalized)
        setattr(self.population.config, "picbreeder_mutation_mode", normalized)
        setattr(self.population.config.genome_config, "picbreeder_mutation_mode", normalized)

    def _set_mutation_mode(self, mode: str, config: neat.Config) -> str:
        normalized = self._normalize_mutation_mode(mode)
        if normalized != self._mutation_mode:
            self._mutation_mode = normalized
        self._apply_mutation_mode(config, self._mutation_mode)
        return self._describe_mutation_mode(self._mutation_mode)

    def _cycle_mutation_mode(self, config: neat.Config) -> str:
        current_index = self._mutation_modes.index(self._mutation_mode)
        next_index = (current_index + 1) % len(self._mutation_modes)
        return self._set_mutation_mode(self._mutation_modes[next_index], config)

    def _sync_mutation_mode(self, config: neat.Config) -> None:
        configured = getattr(config.genome_config, "picbreeder_mutation_mode", self._mutation_mode)
        normalized = self._normalize_mutation_mode(configured)
        if normalized != self._mutation_mode:
            self._mutation_mode = normalized
        self._apply_mutation_mode(config, self._mutation_mode)

    def _mutation_controls_for(self, config: neat.Config) -> Optional[MutationToggleControls]:
        if self.scheme != "toggle":
            return None
        return MutationToggleControls(
            get_mode=lambda: self._describe_mutation_mode(),
            cycle_mode=lambda: self._cycle_mutation_mode(config),
            set_mode=lambda target: self._set_mutation_mode(target, config),
        )

    def _save_gray_renders(
        self,
        genomes: List[Tuple[int, neat.DefaultGenome]],
        config: neat.Config,
        generation: int,
    ) -> List[Path]:
        if not self.save_gray_renders or self.gray_render_dir is None:
            return []
        artifacts: List[Path] = []
        for index, (genome_id, genome) in enumerate(genomes):
            image = render_genome_image(
                genome,
                config,
                self.render_size,
                self.render_size,
                self.scheme,
            )
            gray_image_data, color_image_data = eval_genome_as_grayscale_and_color(genome, config, self.render_size, self.render_size)
            gray_image = Image.new(mode="L", size=(self.render_size, self.render_size))
            flat_pixels: List[Any] = [int(pixel * 255) for row in gray_image_data for pixel in row]
            gray_image.putdata(flat_pixels)
            gray_filename = f"gen_{generation:03d}_idx_{index:02d}_id_{genome_id}_gray.png"
            gray_image.save(self.gray_render_dir / gray_filename, format="PNG")
            color_filename = f"gen_{generation:03d}_idx_{index:02d}_id_{genome_id}_color.png"
            color_path = self.gray_render_dir / color_filename
            image.save(color_path, format="PNG")
            gray_path = self.gray_render_dir / gray_filename
            artifacts.append(gray_path)
        return artifacts

    def evaluate_generation(self, genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config) -> None:
        expected = self.rows * self.cols
        if len(genomes) != expected:
            raise ValueError(f"Expected {expected} genomes, received {len(genomes)}.")

        self._sync_mutation_mode(config)
        mutation_controls = self._mutation_controls_for(config)

        generation = int(self.population.generation)
        print(f"\n--- Generation {generation} ---")

        if self.save_genome_diagrams:
            diagram_paths = save_neat_genome_diagrams(genomes, config, self.output_dir, generation)
            diagram_dir = diagram_paths[0].parent
            print(f"Genome diagrams saved to {diagram_dir}")

        if self.save_gray_renders:
            gray_paths = self._save_gray_renders(genomes, config, generation)
            if gray_paths and not self._gray_notice_emitted:
                print(f"Grayscale renders saved to {self.gray_render_dir}")
                self._gray_notice_emitted = True

        states, caches = build_generation_state(
            genomes,
            config,
            generation,
            self.rows,
            self.cols,
            self.thumb_size,
            variant="both"
        )
        state = states['color']
        cache = caches['color']
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
            )
            filename = f"gen_{generation:03d}_idx_{index:02d}_id_{genome_id}.png"
            path = self.render_dir / filename
            image.save(path, format="PNG")
            return path

        selected = self.ui.select_parents(
            state,
            generation,
            render_high_res,
            self.select_limit,
            mutation_controls=mutation_controls,
        )

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
        args.render_size,
        args.select_k,
        args.save_genome_diagrams,
        args.save_gray_renders,
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
