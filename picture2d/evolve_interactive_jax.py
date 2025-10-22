"""
This is an example that amounts to an offline picbreeder.org without any nice features. :)

Left-click on thumbnails to pick images to breed for next generation, right-click to
render a high-resolution version of an image.  Genomes and images chosen for breeding
and rendering are saved to disk.

This example also demonstrates how to customize species stagnation.
"""
import argparse
import os
import pickle
import random
from multiprocessing import Pool
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pygame

import neat
from common import (
    _canvas_coords,
    _to_byte,
    clamp,
    eval_color_image,
    eval_gray_image,
    eval_mono_image,
    hsb_to_rgb,
)
from tensorneat.common import State
from tensorneat.genome import DefaultGenome
from tensorneat.genome.operations import DefaultMutation


class InteractiveStagnation(object):
    """
    This class is used as a drop-in replacement for the default species stagnation scheme.

    A species is only marked as stagnant if the user has not selected one of its output images
    within the last `max_stagnation` generations.
    """

    def __init__(self, config, reporters):
        self.max_stagnation = int(config.get('max_stagnation'))
        self.reporters = reporters

    @classmethod
    def parse_config(cls, param_dict):
        config = {'max_stagnation': 15}
        config.update(param_dict)

        return config

    @classmethod
    def write_config(cls, f, config):
        f.write('max_stagnation       = {}\n'.format(config['max_stagnation']))

    def update(self, species_set, generation):
        result = []
        for s in species_set.species.values():
            # If any member of the species is selected (i.e., has a fitness above zero),
            # mark the species as improved.
            for m in s.members.values():
                if m.fitness > 0:
                    s.last_improved = generation
                    break

            stagnant_time = generation - s.last_improved
            is_stagnant = stagnant_time >= self.max_stagnation
            result.append((s.key, s, is_stagnant))

        return result


def seed_initial_population(population, genome_config, passes=8, node_prob=0.5, conn_prob=0.5):
    """Roughly mirror Neurogram's richer starting genomes."""
    for genome in population.population.values():
        for _ in range(passes):
            if random.random() < node_prob:
                genome.mutate_add_node(genome_config)
            if random.random() < conn_prob:
                genome.mutate_add_connection(genome_config)


class TensorNeatGenomeGrid:
    """TensorNEAT-backed population that mirrors Neurogram's interactive evolve loop."""

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        num_inputs: int = 3,
        num_outputs: int = 3,
        max_nodes: int = 128,
        max_conns: int = 256,
        weight_precision: int = 4,
        seed: Optional[int] = None,
        mutation: Optional[DefaultMutation] = None,
    ) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive integers.")

        self.rows = rows
        self.cols = cols
        self.size = rows * cols
        self.generation = 0
        self._weight_precision = weight_precision

        if seed is None:
            seed = random.randrange(2**31)
        self._seed = seed
        self._py_rng = random.Random(seed)

        if mutation is None:
            mutation = DefaultMutation(
                conn_add=0.5,
                conn_delete=0.0,
                node_add=0.5,
                node_delete=0.0,
            )

        self.genome = DefaultGenome(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            max_nodes=max_nodes,
            max_conns=max_conns,
            mutation=mutation,
        )

        self.state = State(randkey=jax.random.PRNGKey(seed))
        self.state = self.genome.setup(self.state)

        pop_nodes, pop_conns = self._initialise_population()
        self.pop_nodes = pop_nodes
        self.pop_conns = self._round_population_weights(pop_conns)

    def _initialise_population(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        keys = jax.random.split(self.state.randkey, self.size + 1)
        node_list = []
        conn_list = []
        for i in range(self.size):
            nodes, conns = self.genome.initialize(self.state, keys[i])
            node_list.append(nodes)
            conn_list.append(conns)
        # reserve the last key for future mutations
        self.state = self.state.update(randkey=keys[-1])
        return jnp.stack(node_list), jnp.stack(conn_list)

    # --- convenience helpers -------------------------------------------------
    def flatten_index(self, row: int, col: int) -> int:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError("row/col out of bounds.")
        return row * self.cols + col

    def coordinates(self, index: int) -> Tuple[int, int]:
        if not (0 <= index < self.size):
            raise IndexError("index out of bounds.")
        return divmod(index, self.cols)

    def get_genome(self, index: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if not (0 <= index < self.size):
            raise IndexError("index out of bounds.")
        return self.pop_nodes[index], self.pop_conns[index]

    def iter_indices(self) -> Iterable[int]:
        return range(self.size)

    # --- evolution -----------------------------------------------------------
    def evolve(self, selected_indices: Sequence[int]) -> None:
        """Replicate Neurogram's evolve loop using TensorNEAT operators."""
        unique = sorted(set(selected_indices))
        if not unique:
            raise ValueError("At least one genome must be selected before evolving.")
        for idx in unique:
            if idx < 0 or idx >= self.size:
                raise IndexError(f"selection index {idx} outside population.")

        preserved = set(unique)
        offspring_count = self.size - len(preserved)
        if offspring_count == 0:
            # Still round weights to maintain consistent presentation.
            self.pop_conns = self._round_population_weights(self.pop_conns)
            return

        # Prepare random keys for crossover/mutation.
        key_total = offspring_count * 2 + 1
        keys = jax.random.split(self.state.randkey, key_total)
        key_iter = iter(keys[:-1])
        new_randkey = keys[-1]

        parent_nodes = self.pop_nodes
        parent_conns = self.pop_conns
        updated_nodes = parent_nodes
        updated_conns = parent_conns

        max_node_key = int(float(jnp.nanmax(parent_nodes[:, :, 0])))
        next_node_key = max_node_key + 1
        new_node_keys = jnp.arange(self.size) + next_node_key

        if "historical_marker" in self.genome.conn_gene.fixed_attrs:
            marker_index = self.genome.conn_gene.fixed_attrs.index("historical_marker")
            max_conn_marker = int(float(jnp.nanmax(parent_conns[:, :, marker_index])))
            marker_start = max_conn_marker + 1
            new_conn_keys = (
                jnp.arange(self.size * 3, dtype=jnp.int32).reshape(self.size, 3)
                + marker_start
            )
        else:
            new_conn_keys = jnp.zeros((self.size, 3), dtype=jnp.int32)

        selection_pool = unique

        # Apply rounding to preserved individuals.
        for idx in preserved:
            rounded = self._round_connection_weights(parent_conns[idx])
            updated_conns = updated_conns.at[idx].set(rounded)

        # Breed new offspring for the remaining slots.
        for slot in range(self.size):
            if slot in preserved:
                continue

            mom_idx = self._py_rng.choice(selection_pool)
            dad_idx = self._py_rng.choice(selection_pool)
            cross_key = next(key_iter)
            mut_key = next(key_iter)

            mom_nodes, mom_conns = parent_nodes[mom_idx], parent_conns[mom_idx]
            dad_nodes, dad_conns = parent_nodes[dad_idx], parent_conns[dad_idx]

            if mom_idx == dad_idx:
                child_nodes, child_conns = mom_nodes, mom_conns
            else:
                child_nodes, child_conns = self.genome.execute_crossover(
                    self.state, cross_key, mom_nodes, mom_conns, dad_nodes, dad_conns
                )

            child_nodes, child_conns = self.genome.execute_mutation(
                self.state,
                mut_key,
                child_nodes,
                child_conns,
                new_node_keys[slot],
                new_conn_keys[slot],
            )

            rounded_child_conns = self._round_connection_weights(child_conns)
            updated_nodes = updated_nodes.at[slot].set(child_nodes)
            updated_conns = updated_conns.at[slot].set(rounded_child_conns)

        self.pop_nodes = updated_nodes
        self.pop_conns = updated_conns
        self.state = self.state.update(randkey=new_randkey)
        self.generation += 1

    # --- utility -------------------------------------------------------------
    def _round_connection_weights(self, conns: jnp.ndarray) -> jnp.ndarray:
        weight_idx = len(self.genome.conn_gene.fixed_attrs)
        weights = conns[:, weight_idx]
        factor = float(10**self._weight_precision)
        rounded = jnp.round(weights * factor) / factor
        keep = jnp.isnan(weights)
        cleaned = jnp.where(keep, weights, rounded)
        return conns.at[:, weight_idx].set(cleaned)

    def _round_population_weights(self, conns: jnp.ndarray) -> jnp.ndarray:
        result = conns
        for idx in range(self.size):
            result = result.at[idx].set(self._round_connection_weights(result[idx]))
        return result


class TensorNeatImageRenderer:
    """Render TensorNEAT genomes using the existing Picbreeder colour pipeline."""

    def __init__(self, grid: TensorNeatGenomeGrid) -> None:
        self.grid = grid
        self._coords_cache: Dict[Tuple[int, int], jnp.ndarray] = {}
        self._forward_cache: Dict[int, Tuple[int, Callable[[jnp.ndarray], jnp.ndarray]]] = {}

    def clear_cache(self) -> None:
        self._forward_cache.clear()

    def _coords(self, width: int, height: int) -> jnp.ndarray:
        key = (width, height)
        if key not in self._coords_cache:
            coords: List[Tuple[float, float, float]] = []
            for row in _canvas_coords(width, height):
                coords.extend(row)
            self._coords_cache[key] = jnp.asarray(coords, dtype=jnp.float32)
        return self._coords_cache[key]

    def _resolve_forward(self, index: int):
        cache = self._forward_cache.get(index)
        if cache and cache[0] == self.grid.generation:
            return cache[1]

        nodes, conns = self.grid.get_genome(index)
        transformed = self.grid.genome.transform(self.grid.state, nodes, conns)

        def single_forward(inputs):
            return self.grid.genome.forward(self.grid.state, transformed, inputs)

        batch_forward = jax.jit(jax.vmap(single_forward))
        self._forward_cache[index] = (self.grid.generation, batch_forward)
        return batch_forward

    def render(self, index: int, width: int, height: int, scheme: str) -> List[List]:
        coords = self._coords(width, height)
        forward = self._resolve_forward(index)
        outputs = forward(coords)
        outputs_np = np.asarray(jax.device_get(outputs))
        num_outputs = outputs_np.shape[-1]
        outputs_np = outputs_np.reshape(height, width, num_outputs)

        image: List[List] = []
        for r in range(height):
            row: List = []
            for c in range(width):
                pixel = outputs_np[r, c]
                channel = float(pixel[0])
                if scheme == 'color':
                    channels = [float(pixel[i]) for i in range(min(3, num_outputs))]
                    while len(channels) < 3:
                        channels.append(0.0)
                    row.append(hsb_to_rgb(channels[:3]))
                elif scheme == 'gray':
                    brightness = abs(clamp(channel, -1.0, 1.0))
                    row.append(_to_byte(brightness))
                else:  # mono
                    brightness = abs(clamp(channel, -1.0, 1.0))
                    row.append(255 if brightness > 0.5 else 0)
            image.append(row)
        return image


def make_surface_from_image_data(image_data, width, height, scheme):
    """Create a pygame surface from numerical image data."""
    if scheme == 'color':
        image = pygame.Surface((width, height))
    else:
        image = pygame.Surface((width, height), depth=8)
        palette = tuple([(i, i, i) for i in range(256)])
        image.set_palette(palette)

    for r, row in enumerate(image_data):
        for c, color in enumerate(row):
            image.set_at((r, c), color)

    return image


class PictureBreeder(object):
    def __init__(self, thumb_width, thumb_height, full_width, full_height,
                 num_cols, num_rows, scheme, num_workers):
        """
        :param thumb_width: Width of preview image
        :param thumb_height: Height of preview image
        :param full_width: Width of full rendered image
        :param full_height: Height of full rendered image
        :param num_cols: Number of thumbnails per row
        :param num_rows: Number of thumbnail rows
        :param scheme: Image type to generate: mono, gray, or color
        """
        self.generation = 0
        self.thumb_width = thumb_width
        self.thumb_height = thumb_height
        self.full_width = full_width
        self.full_height = full_height

        assert scheme in ('mono', 'gray', 'color')
        self.scheme = scheme

        self.num_cols = int(num_cols)
        self.num_rows = int(num_rows)
        self.window_width = 16 + self.num_cols * (self.thumb_width + 4)
        self.window_height = 16 + self.num_rows * (self.thumb_height + 4)

        self.num_workers = num_workers

    def make_image_from_data(self, image_data, width, height):
        return make_surface_from_image_data(image_data, width, height, self.scheme)

    def make_thumbnails(self, genomes, config):
        img_func = eval_mono_image
        if self.scheme == 'gray':
            img_func = eval_gray_image
        elif self.scheme == 'color':
            img_func = eval_color_image

        with Pool(self.num_workers) as pool:
            jobs = []
            for genome_id, genome in genomes:
                jobs.append(pool.apply_async(img_func, (genome, config, self.thumb_width, self.thumb_height)))

            thumbnails = []

            # Sequential version for debugging
            for genome_id, genome in genomes:
                image_data = img_func(genome, config, self.thumb_width, self.thumb_height)
                thumbnails.append(self.make_image_from_data(image_data, self.thumb_width, self.thumb_height))

            # for j in jobs:
            #     # TODO: This code currently generates the image data using the multiprocessing
            #     # pool, and then does the actual image construction here because pygame complained
            #     # about not being initialized if the pool workers tried to construct an image.
            #     # Presumably there is some way to fix this, but for now this seems fast enough
            #     # for the purposes of a demo.
            #     image_data = j.get()

            #     thumbnails.append(self.make_image_from_data(image_data, self.thumb_width, self.thumb_height))

        return thumbnails

    def make_high_resolution(self, genome, config):
        genome_id, genome = genome

        # Make sure the output directory exists.
        if not os.path.isdir('rendered'):
            os.mkdir('rendered')

        if self.scheme == 'gray':
            image_data = eval_gray_image(genome, config, self.full_width, self.full_height)
        elif self.scheme == 'color':
            image_data = eval_color_image(genome, config, self.full_width, self.full_height)
        else:
            image_data = eval_mono_image(genome, config, self.full_width, self.full_height)

        image = self.make_image_from_data(image_data, self.full_width, self.full_height)
        pygame.image.save(image, "rendered/rendered-{}-{}.png".format(os.getpid(), genome_id))

        with open("rendered/genome-{}-{}.bin".format(os.getpid(), genome_id), "wb") as f:
            pickle.dump(genome, f, 2)

    def eval_fitness(self, genomes, config):
        selected = []
        rects = []
        for n, (genome_id, genome) in enumerate(genomes):
            selected.append(False)
            row, col = divmod(n, self.num_cols)
            rects.append(pygame.Rect(4 + (self.thumb_width + 4) * col,
                                     4 + (self.thumb_height + 4) * row,
                                     self.thumb_width, self.thumb_height))

        pygame.init()
        screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Interactive NEAT-python generation {0}".format(self.generation))

        buttons = self.make_thumbnails(genomes, config)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    running = False
                    break

                if event.type == pygame.MOUSEBUTTONDOWN:
                    clicked_button = -1
                    for n, button in enumerate(buttons):
                        if rects[n].collidepoint(pygame.mouse.get_pos()):
                            clicked_button = n
                            break

                    if event.button == 1:
                        selected[clicked_button] = not selected[clicked_button]
                    else:
                        self.make_high_resolution(genomes[clicked_button], config)

            if running:
                screen.fill((128, 128, 192))
                for n, button in enumerate(buttons):
                    screen.blit(button, rects[n])
                    if selected[n]:
                        pygame.draw.rect(screen, (255, 0, 0), rects[n], 3)
                pygame.display.flip()

        for n, (genome_id, genome) in enumerate(genomes):
            if selected[n]:
                genome.fitness = 1.0
                pygame.image.save(buttons[n], "image-{}.{}.png".format(os.getpid(), genome_id))
                with open("genome-{}-{}.bin".format(os.getpid(), genome_id), "wb") as f:
                    pickle.dump(genome, f, 2)
            else:
                genome.fitness = 0.0


class TensorNeatPictureBreeder(PictureBreeder):
    """Interactive breeder that uses TensorNEAT genomes with the Neurogram evolve loop."""

    def __init__(
        self,
        thumb_width,
        thumb_height,
        full_width,
        full_height,
        num_cols,
        num_rows,
        scheme,
        *,
        seed: Optional[int] = None,
        mutation: Optional[DefaultMutation] = None,
    ):
        super(TensorNeatPictureBreeder, self).__init__(
            thumb_width,
            thumb_height,
            full_width,
            full_height,
            num_cols,
            num_rows,
            scheme,
            num_workers=0,
        )

        num_outputs = 3 if scheme == 'color' else 1
        self.tensor_grid = TensorNeatGenomeGrid(
            rows=self.num_rows,
            cols=self.num_cols,
            num_inputs=3,
            num_outputs=num_outputs,
            seed=seed,
            mutation=mutation,
        )
        self.tensor_renderer = TensorNeatImageRenderer(self.tensor_grid)

    # ---- rendering helpers --------------------------------------------------
    def make_thumbnails(self, *_args, **_kwargs) -> List[pygame.Surface]:  # type: ignore[override]
        thumbnails: List[pygame.Surface] = []
        for idx in self.tensor_grid.iter_indices():
            image_data = self.tensor_renderer.render(
                idx, self.thumb_width, self.thumb_height, self.scheme
            )
            thumbnails.append(
                self.make_image_from_data(image_data, self.thumb_width, self.thumb_height)
            )
        return thumbnails

    def make_high_resolution(self, index: int, *_args) -> None:  # type: ignore[override]
        if not os.path.isdir('rendered'):
            os.mkdir('rendered')

        image_data = self.tensor_renderer.render(
            index, self.full_width, self.full_height, self.scheme
        )
        surface = make_surface_from_image_data(
            image_data, self.full_width, self.full_height, self.scheme
        )
        pygame.image.save(
            surface, f"rendered/rendered-{os.getpid()}-{index}.png"
        )

        nodes, conns = self.tensor_grid.get_genome(index)
        payload = (
            np.asarray(jax.device_get(nodes)),
            np.asarray(jax.device_get(conns)),
        )
        with open(f"rendered/genome-{os.getpid()}-{index}.bin", "wb") as f:
            pickle.dump(payload, f, 2)

    # ---- interactive loop ---------------------------------------------------
    def evaluate_generation(self) -> bool:
        indices = list(self.tensor_grid.iter_indices())
        if not indices:
            return False

        selected = [False] * len(indices)
        rects = []
        for n, idx in enumerate(indices):
            row, col = divmod(n, self.num_cols)
            rects.append(
                pygame.Rect(
                    4 + (self.thumb_width + 4) * col,
                    4 + (self.thumb_height + 4) * row,
                    self.thumb_width,
                    self.thumb_height,
                )
            )

        pygame.init()
        screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption(
            "Interactive TensorNEAT generation {}".format(self.tensor_grid.generation + 1)
        )

        buttons = self.make_thumbnails()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    running = False
                    break

                if event.type == pygame.MOUSEBUTTONDOWN:
                    clicked_button = -1
                    for i, rect in enumerate(rects):
                        if rect.collidepoint(pygame.mouse.get_pos()):
                            clicked_button = i
                            break

                    if clicked_button == -1:
                        continue

                    if event.button == 1:
                        selected[clicked_button] = not selected[clicked_button]
                    else:
                        self.make_high_resolution(indices[clicked_button])

            if running:
                screen.fill((128, 128, 192))
                for n, button in enumerate(buttons):
                    screen.blit(button, rects[n])
                    if selected[n]:
                        pygame.draw.rect(screen, (255, 0, 0), rects[n], 3)
                pygame.display.flip()

        chosen = [indices[i] for i, flag in enumerate(selected) if flag]
        if chosen:
            for idx, surface in [
                (indices[i], buttons[i]) for i, flag in enumerate(selected) if flag
            ]:
                pygame.image.save(surface, f"image-{os.getpid()}.{idx}.png")
                nodes, conns = self.tensor_grid.get_genome(idx)
                payload = (
                    np.asarray(jax.device_get(nodes)),
                    np.asarray(jax.device_get(conns)),
                )
                with open(f"genome-{os.getpid()}-{idx}.bin", "wb") as f:
                    pickle.dump(payload, f, 2)

            self.tensor_grid.evolve(chosen)
            self.tensor_renderer.clear_cache()
            self.generation += 1

        return True

def run():
    # 128x128 thumbnails, 1500x1500 renders, 5x5 grid, color images, 4 worker processes.
    pb = PictureBreeder(128, 128, 1500, 1500, 5, 5, 'color', 4)

    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    if pb.scheme == 'color':
        config_path = os.path.join(local_dir, 'interactive_config_color')
    else:
        config_path = os.path.join(local_dir, 'interactive_config_gray')

    # Note that we provide the custom stagnation class to the Config constructor.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, InteractiveStagnation,
                         config_path)

    config.pop_size = pb.num_cols * pb.num_rows
    pop = neat.Population(config)

    seed_initial_population(pop, config.genome_config)

    # Add a stdout reporter to show progress in the terminal.
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    while 1:
        pb.generation = pop.generation + 1
        pop.run(pb.eval_fitness, 1)


def run_tensorneat():
    pb = TensorNeatPictureBreeder(128, 128, 1500, 1500, 5, 5, 'color')

    keep_running = True
    while keep_running:
        keep_running = pb.evaluate_generation()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactive Picbreeder clone.")
    parser.add_argument(
        '--engine',
        choices=('neat', 'tensorneat'),
        default='neat',
        help="Choose the evolutionary backend. "
             "'neat' uses neat-python, 'tensorneat' mirrors Neurogram's loop via TensorNEAT.",
    )
    args = parser.parse_args()

    if args.engine == 'tensorneat':
        run_tensorneat()
    else:
        run()
