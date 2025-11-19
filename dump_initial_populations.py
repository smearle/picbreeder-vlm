
from pathlib import Path

import hydra
import neat

from artifacts import render_genome_images, save_neat_genome_diagrams
from neat_components import InteractiveStagnation, PicbreederGenome, apply_picbreeder_config_defaults, seed_initial_population, sync_population_output_activations
from picbreeder_reproduction import PicbreederReproduction
from rendering import create_numbered_grid



def dump_initial_populations(
    count: int,
    output_dir: Path,
    config_path: Path,
    rows: int,
    cols: int,
    thumb_size: int,
    enable_output_activations: bool,
    enable_input_activations: bool,
) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    supports_color_outputs = None
    for index in range(count):
        config = neat.Config(
            PicbreederGenome,
            PicbreederReproduction,
            neat.DefaultSpeciesSet,
            InteractiveStagnation,
            str(config_path),
        )
        apply_picbreeder_config_defaults(
            config,
            enable_output_activations=enable_output_activations,
            enable_input_activations=enable_input_activations,
        )
        config.pop_size = rows * cols

        population = neat.Population(config)
        sync_population_output_activations(population, config.genome_config)
        seed_initial_population(population, config.genome_config)

        if supports_color_outputs is None:
            supports_color_outputs = len(getattr(config.genome_config, "output_keys", ())) >= 3

        genomes = sorted(population.population.items(), key=lambda item: item[0])

        population_dir = output_dir / f"population_{index:03d}"
        # save_neat_population(state, population_dir, 0, png_cache)

        states, _ = render_genome_images(
            genomes,
            config,
            0,
            rows,
            cols,
            thumb_size,
            variant="both",
        )
        for k, state in states.items():
            grid_image = create_numbered_grid(state)
            grid_path = output_dir / f"pop-{index:03d}_{k}.png"
            grid_image.save(grid_path, format="PNG")

        save_neat_genome_diagrams(genomes, config, population_dir, 0)


@hydra.main(version_base="1.3", config_path=None, config_name="collaborative_base")
def main():
    dump_initial_populations(
        count=10,
        output_dir=Path("initial_populations"),
        config_path=Path("configs/interactive_config_color.ini"),
        rows=3,
        cols=5,
        thumb_size=200,
        enable_output_activations=True,
        enable_input_activations=False,
    )

if __name__ == "__main__":
    main()
