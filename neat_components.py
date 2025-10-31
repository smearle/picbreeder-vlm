import gzip
import math
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

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
    # Add symmetric deletion to control bloat.
    genome_config.picbreeder_node_del_prob = 0.00
    genome_config.picbreeder_conn_del_prob = 0.00
    # Prune unused (disconnected) nodes/edges after mutation.
    genome_config.picbreeder_prune_unused = True
    genome_config.picbreeder_weight_mutate_rate = 0.05
    genome_config.picbreeder_weight_sigma = 1.0

    # Mutation scoping defaults (channel-masked mutation)
    # Modes: "all" (default), "color_only" (limit to H/S), "structure_only" (limit to B)
    genome_config.picbreeder_mutation_mode = "all"
    # Policies: "strict" (disallow touching entangled nodes/edges), "soft" (currently same as strict)
    genome_config.picbreeder_mask_policy = "strict"

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

        # Deletion to counter growth
        if random.random() < getattr(config, "picbreeder_conn_del_prob", 0.05):
            self._mutate_delete_connection(config)
        if random.random() < getattr(config, "picbreeder_node_del_prob", 0.10):
            self._mutate_delete_node(config)

        # Optional pruning of unused topology
        if getattr(config, "picbreeder_prune_unused", False):
            self._prune_unused_topology(config)

    # -------------------- Channel-masked mutation helpers --------------------

    def _get_allowed_outputs(self, config) -> Optional[Set[int]]:
        """Return set of output keys allowed to be modified, or None for all."""
        try:
            mode = str(getattr(config, "picbreeder_mutation_mode", "all")).lower()
        except Exception:
            mode = "all"
        out_keys = list(config.output_keys)
        if mode == "all" or not out_keys:
            return None
        if mode == "color_only":
            # Expect H,S,B ordering; use first two when available.
            if len(out_keys) >= 3:
                return set(out_keys[:2])
            return None  # no-op if not a 3-channel setup
        if mode == "structure_only":
            # Expect brightness as last when 3-channel; otherwise take the single output.
            if len(out_keys) >= 3:
                return {out_keys[2]}
            return {out_keys[-1]}
        return None

    def _mask_policy(self, config) -> str:
        try:
            policy = str(getattr(config, "picbreeder_mask_policy", "strict")).lower()
        except Exception:
            policy = "strict"
        return policy

    def _compute_outputs_reached(self, config) -> Dict[int, Set[int]]:
        """For every node key (including inputs/outputs), which output keys are reachable downstream?

        Uses reverse adjacency over enabled connections to propagate each output back to its ancestors.
        """
        reverse_adj: Dict[int, List[int]] = {}
        for (src, dst), conn in self.connections.items():
            if not getattr(conn, "enabled", True):
                continue
            reverse_adj.setdefault(dst, []).append(src)

        outputs_reached: Dict[int, Set[int]] = {}
        all_keys = set(self.nodes.keys()) | set(config.input_keys) | set(config.output_keys)
        for k in all_keys:
            outputs_reached[k] = set()

        stack: List[int] = []
        for out in config.output_keys:
            outputs_reached[out].add(out)
            stack.append(out)
        while stack:
            node = stack.pop()
            for src in reverse_adj.get(node, []):
                before = len(outputs_reached[src])
                outputs_reached[src].update(outputs_reached[node])
                if len(outputs_reached[src]) != before:
                    stack.append(src)
        return outputs_reached

    @staticmethod
    def _is_node_allowed(node_key: int, allowed_outputs: Optional[Set[int]], outputs_reached: Optional[Dict[int, Set[int]]]) -> bool:
        if allowed_outputs is None or outputs_reached is None:
            return True
        downstream = outputs_reached.get(node_key, set())
        return bool(downstream) and downstream.issubset(allowed_outputs)

    @staticmethod
    def _is_conn_allowed(conn_key: Sequence[int], allowed_outputs: Optional[Set[int]], outputs_reached: Optional[Dict[int, Set[int]]]) -> bool:
        if allowed_outputs is None or outputs_reached is None:
            return True
        src, dst = conn_key  # type: ignore[misc]
        downstream = outputs_reached.get(dst, set())
        return bool(downstream) and downstream.issubset(allowed_outputs)

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
        allowed = self._get_allowed_outputs(config)
        outputs_reached = self._compute_outputs_reached(config) if allowed is not None else None
        input_keys_ordered = list(config.input_keys)
        for idx in range(len(self._input_activation_names)):
            if allowed is not None and outputs_reached is not None:
                input_key = input_keys_ordered[idx] if idx < len(input_keys_ordered) else None
                if input_key is not None and not self._is_node_allowed(input_key, allowed, outputs_reached):
                    continue
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
        allowed = self._get_allowed_outputs(config)
        output_keys_ordered = list(config.output_keys)
        for idx in range(len(self._output_activation_names)):
            if allowed is not None:
                out_key = output_keys_ordered[idx] if idx < len(output_keys_ordered) else None
                if out_key is not None and out_key not in allowed:
                    continue
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
        allowed = self._get_allowed_outputs(config)
        if allowed is not None:
            outputs_reached = self._compute_outputs_reached(config)
            candidate_keys = [k for k in candidate_keys if self._is_node_allowed(k, allowed, outputs_reached)]
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

        # Restrict to connections that only influence allowed outputs (if any)
        allowed = self._get_allowed_outputs(config)
        if allowed is not None:
            outputs_reached = self._compute_outputs_reached(config)
            enabled_connections = [cg for cg in enabled_connections if self._is_conn_allowed(cg.key, allowed, outputs_reached)]
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
        allowed = self._get_allowed_outputs(config)
        if allowed is None:
            before = set(self.connections)
            super().mutate_add_connection(config)
            added = set(self.connections) - before
            for key in added:
                self.connections[key].weight = self._random_weight()
            return

        outputs_reached = self._compute_outputs_reached(config)
        attempts = 4
        while attempts > 0:
            before = set(self.connections)
            super().mutate_add_connection(config)
            added = set(self.connections) - before
            ok = False
            for key in list(added):
                if self._is_conn_allowed(key, allowed, outputs_reached):
                    self.connections[key].weight = self._random_weight()
                    ok = True
                else:
                    # Remove disallowed addition
                    self.connections.pop(key, None)
            if ok:
                break
            attempts -= 1

    def _mutate_delete_connection(self, config) -> None:
        allowed = self._get_allowed_outputs(config)
        if allowed is None:
            try:
                super().mutate_delete_connection(config)
            except Exception:
                # Some NEAT variants may not implement deletion; ignore if unavailable.
                pass
            return
        outputs_reached = self._compute_outputs_reached(config)
        candidates: List[Sequence[int]] = []
        for key, conn in self.connections.items():
            if not conn.enabled:
                continue
            if self._is_conn_allowed(key, allowed, outputs_reached):
                candidates.append(key)
        if not candidates:
            return
        key = random.choice(candidates)
        self.connections.pop(key, None)

    def _mutate_delete_node(self, config) -> None:
        allowed = self._get_allowed_outputs(config)
        if allowed is None:
            try:
                super().mutate_delete_node(config)
            except Exception:
                # If deletion is unsupported, skip.
                pass
            return
        outputs_reached = self._compute_outputs_reached(config)
        input_keys = set(config.input_keys)
        output_keys = set(config.output_keys)
        candidates = [nid for nid in self.nodes if nid not in input_keys and nid not in output_keys]
        candidates = [nid for nid in candidates if self._is_node_allowed(nid, allowed, outputs_reached)]
        if not candidates:
            return
        nid = random.choice(candidates)
        # Remove node and incident connections
        self.nodes.pop(nid, None)
        to_delete = [key for key in list(self.connections) if key[0] == nid or key[1] == nid]
        for key in to_delete:
            self.connections.pop(key, None)

    def _mutate_weights(self, config) -> None:
        rate = getattr(config, "picbreeder_weight_mutate_rate", 0.0)
        sigma = getattr(config, "picbreeder_weight_sigma", 1.0)
        if rate <= 0.0:
            return

        allowed = self._get_allowed_outputs(config)
        outputs_reached = self._compute_outputs_reached(config) if allowed is not None else None
        for key, connection in self.connections.items():
            if not connection.enabled:
                continue
            if allowed is not None and outputs_reached is not None and not self._is_conn_allowed(key, allowed, outputs_reached):
                continue
            if random.random() < rate:
                connection.weight += random.gauss(0.0, sigma)

    def _random_weight(self) -> float:
        return random.uniform(-1.0, 1.0)

    def _prune_unused_topology(self, config) -> None:
        """Remove nodes and connections that do not contribute to any output.

        Traverses enabled connections backward from outputs to inputs and
        keeps only reachable nodes and their enabled connections.
        """
        if not self.connections:
            return
        input_keys = set(config.input_keys)
        output_keys = set(config.output_keys)

        # Build reverse adjacency of enabled connections: dst -> [src]
        reverse_adj = {}
        for key, conn in self.connections.items():
            if not conn.enabled:
                continue
            src, dst = key
            reverse_adj.setdefault(dst, []).append(src)

        # Backward reachability from outputs
        reachable_nodes = set(output_keys)
        stack = list(output_keys)
        while stack:
            node = stack.pop()
            for src in reverse_adj.get(node, []):
                if src not in reachable_nodes:
                    reachable_nodes.add(src)
                    stack.append(src)

        # Always keep inputs/outputs even if disconnected
        reachable_nodes |= input_keys | output_keys

        # Prune nodes
        nodes_to_remove = [nid for nid in self.nodes.keys() if nid not in reachable_nodes]
        for nid in nodes_to_remove:
            self.nodes.pop(nid, None)

        # Prune connections not between reachable nodes or disabled
        to_delete = []
        for key, conn in self.connections.items():
            src, dst = key
            if not conn.enabled or src not in reachable_nodes or dst not in reachable_nodes:
                to_delete.append(key)
        for key in to_delete:
            self.connections.pop(key, None)


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
