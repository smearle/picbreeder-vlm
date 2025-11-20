from __future__ import annotations

import random
from itertools import count
from typing import Dict, List, Tuple

from neat.config import ConfigParameter, DefaultClassConfig


class PicbreederReproduction(DefaultClassConfig):
    """Reproduction strategy that mirrors CPPNArtEvolution's selective breeding."""

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(
            param_dict,
            [
                ConfigParameter("mating", bool, False),
                ConfigParameter("crossover_rate", float, 0.0),
                ConfigParameter("mutation_repeats", int, 0),
            ],
        )

    def __init__(self, config, reporters, stagnation):
        # pylint: disable=super-init-not-called
        self.reproduction_config = config
        self.reporters = reporters
        self.stagnation = stagnation
        self.genome_indexer = count(1)
        self.ancestors: Dict[int, Tuple[int, int]] = {}

    def create_new(self, genome_type, genome_config, num_genomes):
        new_genomes = {}
        for _ in range(num_genomes):
            key = next(self.genome_indexer)
            genome = genome_type(key)
            genome.configure_new(genome_config)
            new_genomes[key] = genome
            self.ancestors[key] = tuple()
        return new_genomes

    def reproduce(self, config, species, pop_size, generation):
        # Keep stagnation bookkeeping consistent with NEAT-python expectations.
        self.stagnation.update(species, generation)

        selected, all_members = self._collect_members(species)
        if not selected:
            # Fallback: keep the single best genome so evolution can continue even if
            # the assistant failed to select anything.
            selected = [
                max(
                    all_members,
                    key=lambda item: item[2].fitness if item[2].fitness is not None else float("-inf"),
                )
            ]

        new_population: Dict[int, object] = {}

        # Reset species membership – speciation will rebuild assignments later.
        preserved_species = {}
        for sid, s in species.species.items():
            s.members = {}
            preserved_species[sid] = s
        species.species = preserved_species

        # Carry over selected genomes unchanged.
        for species_id, genome_id, genome in selected:
            new_population[genome_id] = genome
            species.species[species_id].members[genome_id] = genome
            self.ancestors.setdefault(genome_id, tuple())

        parents = [entry[2] for entry in selected]
        parent_ids = [entry[1] for entry in selected]
        parent_species = [entry[0] for entry in selected]

        mating_enabled = bool(self.reproduction_config.mating)
        crossover_rate = float(self.reproduction_config.crossover_rate)
        mutation_repeats = max(0, int(self.reproduction_config.mutation_repeats))

        while len(new_population) < pop_size:
            parent_index = random.randrange(len(parents))
            parent1 = parents[parent_index]
            parent1_id = parent_ids[parent_index]
            parent1_species = parent_species[parent_index]

            gid = next(self.genome_indexer)

            if mating_enabled and len(parents) > 1 and random.random() < crossover_rate:
                mate_index = parent_index
                while mate_index == parent_index:
                    mate_index = random.randrange(len(parents))
                parent2 = parents[mate_index]
                parent2_id = parent_ids[mate_index]

                child = self._picbreeder_crossover(config, parent1, parent2, gid)
                child.mutate(config.genome_config)
                for _ in range(mutation_repeats):
                    child.mutate(config.genome_config)
                ancestor_record = (parent1_id, parent2_id)
            else:
                child = self._clone_genome(parent1, gid)
                child.mutate(config.genome_config)
                for _ in range(mutation_repeats):
                    child.mutate(config.genome_config)
                ancestor_record = (parent1_id, -1)

            child.fitness = None
            new_population[child.key] = child
            species.species[parent1_species].members[child.key] = child
            self.ancestors[child.key] = ancestor_record

        return new_population

    def _picbreeder_crossover(self, config, parent1, parent2, new_key):
        child = self._clone_genome(parent2, new_key)

        parent1_affinity = getattr(parent1, "get_node_affinity", None)
        child_set_affinity = getattr(child, "set_node_affinity", None)

        for key, donor_node in parent1.nodes.items():
            recipient = child.nodes.get(key)
            if recipient is None:
                child.nodes[key] = donor_node.copy()
                if parent1_affinity and child_set_affinity:
                    try:
                        affinity_value = parent1_affinity(key)
                    except Exception:  # pragma: no cover - defensive
                        affinity_value = None
                    if affinity_value is not None:
                        child_set_affinity(key, affinity_value)
            else:
                if random.random() < 0.5:
                    recipient.activation = donor_node.activation

        for key, donor_conn in parent1.connections.items():
            recipient_conn = child.connections.get(key)
            if recipient_conn is None:
                child.connections[key] = donor_conn.copy()
            else:
                recipient_conn.weight = (recipient_conn.weight + donor_conn.weight) / 2.0

        return child

    @staticmethod
    def _clone_genome(parent, new_key):
        clone = parent.__class__(new_key)
        clone.nodes = {k: node.copy() for k, node in parent.nodes.items()}
        clone.connections = {k: conn.copy() for k, conn in parent.connections.items()}
        clone.fitness = None

        if not hasattr(parent, "_node_affinities"):
            raise RuntimeError(
                "Cannot clone Picbreeder genome: parent is missing _node_affinities. "
                "Ensure genomes are initialized via PicbreederGenome.configure_new."
            )
        clone._node_affinities = {int(k): str(v) for k, v in parent._node_affinities.items()}

        if hasattr(parent, "_input_activation_names"):
            names = list(getattr(parent, "_input_activation_names"))
            setter = getattr(clone, "_set_input_activation_names", None)
            setter(names)

        if hasattr(parent, "_output_activations_enabled"):
            clone._output_activations_enabled = getattr(parent, "_output_activations_enabled")

        if hasattr(parent, "_output_activation_names"):
            output_names = list(getattr(parent, "_output_activation_names"))
            setter = getattr(clone, "_set_output_activation_names", None)
            setter(output_names)

        else:
            clear_fn = getattr(clone, "_clear_output_activations", None)
            clear_fn()

        return clone

    @staticmethod
    def _collect_members(species) -> Tuple[List[Tuple[int, int, object]], List[Tuple[int, int, object]]]:
        selected: List[Tuple[int, int, object]] = []
        all_members: List[Tuple[int, int, object]] = []
        for sid, s in species.species.items():
            for gid, genome in s.members.items():
                entry = (sid, gid, genome)
                all_members.append(entry)
                if genome.fitness is not None and genome.fitness > 0.0:
                    selected.append(entry)
        return selected, all_members
