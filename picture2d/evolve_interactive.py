"""
This is an example that amounts to an offline picbreeder.org without any nice features. :)

Left-click on thumbnails to pick images to breed for next generation, right-click to
render a high-resolution version of an image.  Genomes and images chosen for breeding
and rendering are saved to disk.

This example also demonstrates how to customize species stagnation.
"""
import os
import pickle
import random
from multiprocessing import Pool
from typing import Iterable, List, Sequence, Tuple

import pygame

import neat
from common import (
    eval_color_image,
    eval_gray_image,
    eval_mono_image,
)


class NeurogramGenome(neat.DefaultGenome):
    """
    DefaultGenome variant that bootstraps the same base structure used by neurogram.js.
    Every genome starts with a single hidden 'aggregator' node fed by the three inputs
    and projecting to the three outputs with unit weights.
    """

    _ACTIVATION_CHOICES = ("sigmoid", "tanh", "relu", "gauss", "sin", "cos", "identity")

    def _ensure_aggregator_scaffold(self, config):
        aggregator_key = getattr(config, "neurogram_aggregator_key", None)
        if aggregator_key is None:
            aggregator_key = max(config.output_keys) + 1
            config.neurogram_aggregator_key = aggregator_key

        if aggregator_key not in self.nodes:
            aggregator_node = self.create_node(config, aggregator_key)
            aggregator_node.activation = "identity"
            aggregator_node.aggregation = "sum"
            aggregator_node.bias = 0.0
            aggregator_node.response = 1.0
            self.nodes[aggregator_key] = aggregator_node

        for output_key in config.output_keys:
            key = (aggregator_key, output_key)
            if key not in self.connections:
                self.add_connection(config, aggregator_key, output_key, 1.0, True)

        for input_key in config.input_keys:
            key = (input_key, aggregator_key)
            if key not in self.connections:
                self.add_connection(config, input_key, aggregator_key, random.gauss(0.0, 1.0), True)

        return aggregator_key

    def configure_new(self, config) -> None:
        # Reset any state inherited from potential reuse.
        self.connections = {}
        self.nodes = {}
        self.fitness = None

        # Recreate the output node genes exactly as DefaultGenome would.
        for node_key in config.output_keys:
            node = self.create_node(config, node_key)
            node.activation = "identity"
            node.aggregation = "sum"
            node.bias = 0.0
            node.response = 1.0
            self.nodes[node_key] = node

        self._ensure_aggregator_scaffold(config)
        self.round_weights()

    def configure_crossover(self, parent1, parent2, config):
        super().configure_crossover(parent1, parent2, config)
        self._ensure_aggregator_scaffold(config)
        self.round_weights()

    def round_weights(self):
        for connection in self.connections.values():
            connection.weight = round(connection.weight, 4)

    def mutate_weights(self, mutation_rate=0.2, mutation_size=0.5):
        for connection in self.connections.values():
            if random.random() < mutation_rate:
                connection.weight += random.gauss(0.0, mutation_size)

    def add_random_node(self, config):
        enabled_connections = [cg for cg in self.connections.values() if cg.enabled]
        if not enabled_connections:
            return

        conn_to_split = random.choice(enabled_connections)
        conn_to_split.enabled = False
        in_node, out_node = conn_to_split.key

        new_node_id = config.get_new_node_key(self.nodes)
        new_node = self.create_node(config, new_node_id)
        new_node.activation = random.choice(self._ACTIVATION_CHOICES)
        new_node.aggregation = "sum"
        new_node.bias = 0.0
        new_node.response = 1.0
        self.nodes[new_node_id] = new_node

        self.add_connection(config, in_node, new_node_id, 1.0, True)
        self.add_connection(config, new_node_id, out_node, conn_to_split.weight, True)

    def add_random_connection(self, config):
        nodes_in_use = set()
        for (start, end), connection in self.connections.items():
            if connection.enabled:
                nodes_in_use.add(start)
                nodes_in_use.add(end)

        if not nodes_in_use:
            return

        potential_sources = [n for n in nodes_in_use if n not in config.output_keys]
        potential_targets = [n for n in nodes_in_use if n >= 0]

        if not potential_sources or not potential_targets:
            return

        from_node = random.choice(potential_sources)
        to_node = random.choice(potential_targets)

        if from_node == to_node or to_node in config.input_keys:
            return

        key = (from_node, to_node)
        if key in self.connections:
            self.connections[key].enabled = True
            return

        self.add_connection(config, from_node, to_node, random.gauss(0.0, 1.0), True)

    def mutate(self, config):
        self.mutate_weights()
        if random.random() < 0.5:
            self.add_random_node(config)
        if random.random() < 0.5:
            self.add_random_connection(config)
        self.round_weights()


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
                genome.add_random_node(genome_config)
            if random.random() < conn_prob:
                genome.add_random_connection(genome_config)
        if hasattr(genome, "round_weights"):
            genome.round_weights()


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

    def _update_caption(self):
        pygame.display.set_caption(f"Interactive NEAT-python generation {self.generation}")

    def _build_thumbnails(self, genomes, config):
        thumbnails = self.make_thumbnails(genomes, config)
        self._update_caption()
        return thumbnails

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
                if hasattr(genome, "round_weights"):
                    genome.round_weights()
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

        if hasattr(genome, "round_weights"):
            genome.round_weights()

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
        self._update_caption()

        buttons = self._build_thumbnails(genomes, config)

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
                pygame.image.save(buttons[n], "log/image-{}.{}.png".format(os.getpid(), genome_id))
                with open("log/genome-{}-{}.bin".format(os.getpid(), genome_id), "wb") as f:
                    pickle.dump(genome, f, 2)
            else:
                genome.fitness = 0.0


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
    config = neat.Config(NeurogramGenome, neat.DefaultReproduction,
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


if __name__ == '__main__':
    run()
