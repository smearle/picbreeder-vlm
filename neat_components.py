import gzip
import math
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import neat
from neat.reporting import BaseReporter

CHECKPOINT_SUFFIX = "_checkpoint.pkl.gz"


def _identity(z: float) -> float:
    return z


def _sigmoid_full(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    value = 1.0 / (1.0 + math.exp(-z))
    return (value * 2.0) - 1.0


def _gauss_full(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    return (math.exp(-(z * z)) * 2.0) - 1.0


def _cosine(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    return math.cos(z)


def _sine(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    return math.sin(z)


def _full_sawtooth(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    return 2.0 * (z - math.floor(z + 0.5))


def _square_wave(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    sine = math.sin(2.0 * math.pi * z)
    if sine > 0.0:
        return 1.0
    if sine < 0.0:
        return -1.0
    return 0.0


def apply_picbreeder_config_defaults(config: neat.Config, enable_output_activations: bool = False) -> None:
    """Match NEAT settings to the defaults used by CPPNArtEvolution."""

    genome_config = config.genome_config
    config.picbreeder_enable_output_activations = bool(enable_output_activations)
    genome_config.picbreeder_enable_output_activations = bool(enable_output_activations)

    # Picbreeder exposes a curated function set.
    genome_config.activation_defs.add("sigmoid_full", _sigmoid_full)
    genome_config.activation_defs.add("gauss_full", _gauss_full)
    genome_config.activation_defs.add("cos", _cosine)
    genome_config.activation_defs.add("sin", _sine)
    genome_config.activation_defs.add("sawtooth_full", _full_sawtooth)
    genome_config.activation_defs.add("square", _square_wave)
    genome_config.activation_options = [
        "identity",
        "sin",
        "sigmoid_full",
        "gauss_full",

        # "cos",
        # "sawtooth_full",
        # "square",
    ]
    genome_config.activation_default = "random"
    genome_config.aggregation_default = "sum"
    genome_config.aggregation_options = ["sum"]
    genome_config.aggregation_mutate_rate = 0.0

    # Biases and responses remain fixed in Picbreeder CPPNs.
    genome_config.bias_init_mean = 0.0
    genome_config.bias_init_stdev = 0.0
    genome_config.bias_mutate_rate = 0.0
    genome_config.bias_replace_rate = 0.0
    genome_config.bias_mutate_power = 0.0
    genome_config.bias_min_value = 0.0
    genome_config.bias_max_value = 0.0

    genome_config.response_init_mean = 1.0
    genome_config.response_init_stdev = 0.0
    genome_config.response_mutate_rate = 0.0
    genome_config.response_replace_rate = 0.0
    genome_config.response_mutate_power = 0.0
    genome_config.response_min_value = 1.0
    genome_config.response_max_value = 1.0

    # Weight behaviour mirrors AllWeightMutation in TWEANNGenotype.
    genome_config.weight_init_mean = 0.0
    # Match RandomNumbers.fullSmallRand() range [-1, 1] used by Java TWEANNs.
    genome_config.weight_init_stdev = 0.5
    genome_config.weight_init_type = "uniform"
    genome_config.weight_mutate_rate = 0.05
    genome_config.weight_mutate_power = 2.0
    genome_config.weight_replace_rate = 0.0
    genome_config.enabled_mutate_rate = 0.0

    # Match the CPPN starting scaffold: no hidden nodes and full direct connectivity.
    genome_config.num_hidden = 0
    genome_config.initial_connection = "full_direct"
    genome_config.connection_fraction = None

    # Store Picbreeder-specific mutation settings for the custom genome class.
    genome_config.picbreeder_activation_rate = 0.3
    genome_config.picbreeder_node_add_prob = 0.2
    genome_config.picbreeder_conn_add_prob = 0.4
    genome_config.picbreeder_weight_mutate_rate = 0.05
    genome_config.picbreeder_weight_sigma = 1.0

    reproduction_config = config.reproduction_config
    if hasattr(reproduction_config, "mating"):
        reproduction_config.mating = True
    if hasattr(reproduction_config, "crossover_rate"):
        reproduction_config.crossover_rate = 0.3
    if hasattr(reproduction_config, "mutation_repeats"):
        reproduction_config.mutation_repeats = 0


class PicbreederGenome(neat.DefaultGenome):
    _ACTIVATION_CHOICES = (
        "identity",
        "sin",
        "sigmoid_full",
        "gauss_full",

        # "cos",
        # "sawtooth_full",
        # "square",
    )

    _INPUT_ACTIVATION_IMPL = {
        "identity": _identity,
        "sin": _sine,
        "cos": _cosine,
        "sigmoid_full": _sigmoid_full,
        "gauss_full": _gauss_full,
        "sawtooth_full": _full_sawtooth,
        "square": _square_wave,
    }

    def configure_new(self, config) -> None:
        super().configure_new(config)
        self.fitness = None
        self._initialize_input_activations(config)
        enabled = getattr(config, "picbreeder_enable_output_activations", False)
        if enabled:
            self._output_activations_enabled = True
            self._initialize_output_activations(config)
        else:
            self._output_activations_enabled = False
            self._clear_output_activations()

    def configure_crossover(self, parent1, parent2, config) -> None:
        super().configure_crossover(parent1, parent2, config)
        self.fitness = None
        self._inherit_input_activations(parent1, parent2, config)
        enabled = getattr(config, "picbreeder_enable_output_activations", False)
        if enabled:
            self._output_activations_enabled = True
            self._inherit_output_activations(parent1, parent2, config)
        else:
            self._output_activations_enabled = False
            self._clear_output_activations()

    def mutate(self, config) -> None:
        enabled = getattr(config, "picbreeder_enable_output_activations", False)
        self._output_activations_enabled = enabled
        if enabled and not hasattr(self, "_output_activation_names"):
            self._initialize_output_activations(config)
        elif not enabled:
            self._clear_output_activations()

        self._mutate_activation(config)

        if random.random() < getattr(config, "picbreeder_node_add_prob", 0.0):
            self._mutate_add_node(config)

        if random.random() < getattr(config, "picbreeder_conn_add_prob", 0.0):
            self._mutate_add_connection(config)

        self._mutate_weights(config)
        self._mutate_input_activations(config)
        if enabled:
            self._mutate_output_activations(config)

    def transform_inputs(self, inputs: Sequence[float]) -> List[float]:
        funcs = getattr(self, "_input_activation_funcs", None)
        if not funcs:
            return list(inputs)
        return [func(value) for func, value in zip(funcs, inputs)]

    def transform_outputs(self, outputs: Sequence[float]) -> List[float]:
        if not getattr(self, "_output_activations_enabled", False):
            return list(outputs)
        funcs = getattr(self, "_output_activation_funcs", None)
        if not funcs:
            return list(outputs)
        values = list(outputs)
        result: List[float] = []
        for idx, func in enumerate(funcs):
            if idx < len(values):
                result.append(func(values[idx]))
        if len(values) > len(result):
            result.extend(values[len(result):])
        return result

    def _initialize_input_activations(self, config) -> None:
        options = tuple(getattr(config, "activation_options", self._ACTIVATION_CHOICES))
        names = [random.choice(options) for _ in config.input_keys]
        self._set_input_activation_names(names)

    def _inherit_input_activations(self, parent1, parent2, config) -> None:
        options = tuple(getattr(config, "activation_options", self._ACTIVATION_CHOICES))
        parent1_names = getattr(parent1, "_input_activation_names", ())
        parent2_names = getattr(parent2, "_input_activation_names", ())
        names: List[str] = []
        for idx, _ in enumerate(config.input_keys):
            candidates = []
            if idx < len(parent1_names):
                candidates.append(parent1_names[idx])
            if idx < len(parent2_names):
                candidates.append(parent2_names[idx])
            if not candidates:
                names.append(random.choice(options))
            else:
                names.append(random.choice(candidates))
        self._set_input_activation_names(names)

    def _set_input_activation_names(self, names: Sequence[str]) -> None:
        self._input_activation_names = list(names)
        funcs: List = []
        for name in self._input_activation_names:
            func = self._INPUT_ACTIVATION_IMPL.get(name)
            if func is None:
                raise KeyError(f"Unknown input activation {name!r}")
            funcs.append(func)
        self._input_activation_funcs = funcs

    def _mutate_input_activations(self, config) -> None:
        if not hasattr(self, "_input_activation_names"):
            self._initialize_input_activations(config)
        rate = getattr(config, "picbreeder_activation_rate", 0.0)
        if rate <= 0.0:
            return
        options = tuple(getattr(config, "activation_options", self._ACTIVATION_CHOICES))
        changed = False
        for idx in range(len(self._input_activation_names)):
            if random.random() < rate:
                self._input_activation_names[idx] = random.choice(options)
                changed = True
        if changed:
            self._set_input_activation_names(self._input_activation_names)

    def _initialize_output_activations(self, config) -> None:
        options = tuple(getattr(config, "activation_options", self._ACTIVATION_CHOICES))
        names = [random.choice(options) for _ in config.output_keys]
        self._set_output_activation_names(names)

    def _inherit_output_activations(self, parent1, parent2, config) -> None:
        options = tuple(getattr(config, "activation_options", self._ACTIVATION_CHOICES))
        parent1_names = getattr(parent1, "_output_activation_names", ())
        parent2_names = getattr(parent2, "_output_activation_names", ())
        names: List[str] = []
        for idx, _ in enumerate(config.output_keys):
            candidates = []
            if idx < len(parent1_names):
                candidates.append(parent1_names[idx])
            if idx < len(parent2_names):
                candidates.append(parent2_names[idx])
            if not candidates:
                names.append(random.choice(options))
            else:
                names.append(random.choice(candidates))
        self._set_output_activation_names(names)

    def _set_output_activation_names(self, names: Sequence[str]) -> None:
        self._output_activation_names = list(names)
        funcs: List = []
        for name in self._output_activation_names:
            func = self._INPUT_ACTIVATION_IMPL.get(name)
            if func is None:
                raise KeyError(f"Unknown output activation {name!r}")
            funcs.append(func)
        self._output_activation_funcs = funcs

    def _mutate_output_activations(self, config) -> None:
        if not getattr(self, "_output_activations_enabled", False):
            return
        if not hasattr(self, "_output_activation_names"):
            self._initialize_output_activations(config)
        rate = getattr(config, "picbreeder_activation_rate", 0.0)
        if rate <= 0.0:
            return
        options = tuple(getattr(config, "activation_options", self._ACTIVATION_CHOICES))
        changed = False
        for idx in range(len(self._output_activation_names)):
            if random.random() < rate:
                self._output_activation_names[idx] = random.choice(options)
                changed = True
        if changed:
            self._set_output_activation_names(self._output_activation_names)

    def _clear_output_activations(self) -> None:
        for attribute in ("_output_activation_names", "_output_activation_funcs"):
            if hasattr(self, attribute):
                delattr(self, attribute)

    def sync_output_activations(self, config) -> None:
        enabled = getattr(config, "picbreeder_enable_output_activations", False)
        self._output_activations_enabled = enabled
        if not enabled:
            self._clear_output_activations()
            return
        names = getattr(self, "_output_activation_names", None)
        if not names or len(names) != len(config.output_keys):
            self._initialize_output_activations(config)
        else:
            self._set_output_activation_names(names)

    def _mutate_activation(self, config) -> None:
        rate = getattr(config, "picbreeder_activation_rate", 0.0)
        if rate <= 0.0 or random.random() >= rate:
            return

        candidate_keys = [key for key in self.nodes if key not in config.input_keys]
        if not candidate_keys:
            return

        key = random.choice(candidate_keys)
        node = self.nodes[key]

        options = tuple(getattr(config, "activation_options", self._ACTIVATION_CHOICES))
        node.activation = random.choice(options)

    def _mutate_add_node(self, config) -> None:
        enabled_connections = [cg for cg in self.connections.values() if cg.enabled]
        if not enabled_connections:
            return

        conn_to_split = random.choice(enabled_connections)
        conn_to_split.enabled = False
        input_key, output_key = conn_to_split.key

        new_node_id = config.get_new_node_key(self.nodes)
        new_node = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = new_node

        self.add_connection(config, input_key, new_node_id, self._random_weight(), True)
        self.add_connection(config, new_node_id, output_key, self._random_weight(), True)

    def _mutate_add_connection(self, config) -> None:
        before = set(self.connections)
        super().mutate_add_connection(config)
        added = set(self.connections) - before
        for key in added:
            self.connections[key].weight = self._random_weight()

    def _mutate_weights(self, config) -> None:
        rate = getattr(config, "picbreeder_weight_mutate_rate", 0.0)
        sigma = getattr(config, "picbreeder_weight_sigma", 1.0)
        if rate <= 0.0:
            return

        for connection in self.connections.values():
            if not connection.enabled:
                continue
            if random.random() < rate:
                connection.weight += random.gauss(0.0, sigma)

    def _random_weight(self) -> float:
        return random.uniform(-1.0, 1.0)


class InteractiveStagnation:
    def __init__(self, config, reporters):
        self.max_stagnation = int(config.get("max_stagnation"))
        self.reporters = reporters

    @classmethod
    def parse_config(cls, param_dict):
        config = {"max_stagnation": 15}
        config.update(param_dict)
        return config

    @classmethod
    def write_config(cls, handle, config):
        handle.write(f"max_stagnation       = {config['max_stagnation']}\n")

    def update(self, species_set, generation):
        result = []
        for species in species_set.species.values():
            for member in species.members.values():
                if member.fitness and member.fitness > 0:
                    species.last_improved = generation
                    break
            stagnant_time = generation - species.last_improved
            is_stagnant = stagnant_time >= self.max_stagnation
            result.append((species.key, species, is_stagnant))
        return result


class GenerationCheckpointer(BaseReporter):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_generation: Optional[int] = None

    def start_generation(self, generation: int) -> None:
        self.current_generation = generation

    def end_generation(self, config: neat.Config, population: Dict[int, neat.DefaultGenome], species_set) -> None:
        if self.current_generation is None:
            return
        next_generation = self.current_generation + 1
        filename = self.output_dir / f"gen_{next_generation:03d}{CHECKPOINT_SUFFIX}"
        with gzip.open(filename, "wb", compresslevel=5) as handle:
            data = (next_generation, config, population, species_set, random.getstate())
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def seed_initial_population(population, genome_config) -> None:
    activation_choices = [opt for opt in getattr(genome_config, "activation_options", [])]
    for genome in population.population.values():
        if activation_choices:
            for node in genome.nodes.values():
                node.activation = random.choice(activation_choices)


def sync_population_output_activations(population, genome_config) -> None:
    """Ensure genomes in a population honour the configured output activation toggle."""
    for genome in population.population.values():
        sync_fn = getattr(genome, "sync_output_activations", None)
        if sync_fn is not None:
            sync_fn(genome_config)
