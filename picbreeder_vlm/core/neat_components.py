import gzip
import math
import os
import pickle
import random
import tempfile
from itertools import count
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import neat
from neat.graphs import creates_cycle
from neat.reporting import BaseReporter

CHECKPOINT_SUFFIX = "_checkpoint.pkl.gz"
LATEST_POPULATION_FILENAME = "latest.pkl.gz"


def _identity(z: float) -> float:
    return z


def _sigmoid_unit(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    value = 1.0 / (1.0 + math.exp(-z))
    return (2.0 * value) - 1.0  # Match UnipolarToBipolar wrapping in the Java client.


def _gaussian_unit(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    value = math.exp(-(z * z))
    return (2.0 * value) - 1.0  # Legacy gaussian outputs are converted to bipolar.


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


def apply_picbreeder_config_defaults(
    config: neat.Config,
    enable_output_activations: bool = False,
    enable_input_activations: bool = False,
    enable_crossover: bool = False,
) -> None:
    """Match NEAT settings to the defaults used by CPPNArtEvolution."""

    genome_config = config.genome_config
    output_keys = [int(key) for key in getattr(genome_config, "output_keys", ())]
    if output_keys:
        genome_config.picbreeder_brightness_output_key = output_keys[-1]
        genome_config.picbreeder_color_output_keys = tuple(output_keys[:-1])
    else:
        genome_config.picbreeder_brightness_output_key = None
        genome_config.picbreeder_color_output_keys = tuple()

    config.picbreeder_enable_input_activations = bool(enable_input_activations)
    genome_config.picbreeder_enable_input_activations = bool(enable_input_activations)
    config.picbreeder_enable_output_activations = bool(enable_output_activations)
    genome_config.picbreeder_enable_output_activations = bool(enable_output_activations)

    # Picbreeder exposes a curated function set.
    genome_config.activation_defs.add("identity", _identity)
    genome_config.activation_defs.add("sigmoid", _sigmoid_unit)
    genome_config.activation_defs.add("gaussian", _gaussian_unit)
    genome_config.activation_defs.add("cos", _cosine)
    genome_config.activation_defs.add("sin", _sine)
    genome_config.activation_defs.add("sawtooth_full", _full_sawtooth)
    genome_config.activation_defs.add("square", _square_wave)
    genome_config.activation_options = [
        "sin",
        "sigmoid",
        "gaussian",
        "cos",
        "identity",
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

    # Match RandomNumbers.fullSmallRand() range [-1, 1] used by Java TWEANNs.  # No longer.
    # genome_config.weight_init_stdev = 0.5

    # Match uniform [-3, 3] range used by picbreeder_orangutan.
    genome_config.weight_init_stdev = 1.5
    genome_config.weight_init_type = "uniform"
    genome_config.weight_min_value = -3.0
    genome_config.weight_max_value = 3.0

    genome_config.weight_mutate_rate = 0.0
    genome_config.weight_mutate_power = 1.0
    genome_config.weight_replace_rate = 0.0
    genome_config.enabled_mutate_rate = 0.0

    # Match the CPPN starting scaffold: no hidden nodes and full direct connectivity.
    genome_config.num_hidden = 1
    genome_config.initial_connection = "full_nodirect"
    genome_config.connection_fraction = None

    # Store Picbreeder-specific mutation settings for the custom genome class.
    genome_config.picbreeder_activation_rate = 0.05
    genome_config.picbreeder_weight_mutate_rate = 0.20
    genome_config.picbreeder_mutation_strength = 0.5
    genome_config.picbreeder_weight_power_min = 0.01  # match legacy slider floor
    genome_config.picbreeder_weight_power_max = 2.0
    genome_config.picbreeder_hidden_color_nodes = 2
    genome_config.picbreeder_generator_weights = (
        ("mutate_weights", 10),
        ("add_connection", 6),
        ("add_node", 4),
        ("mutate_activation", 1),
    )
    genome_config.picbreeder_color_link_rate = 0.50
    genome_config.picbreeder_grey_link_rate = 0.50
    genome_config.picbreeder_grey_to_color_link_rate = 0.10

    # Mutation scoping defaults (channel-masked mutation)
    # Modes: "all" (default), "color_only" (limit to H/S), "structure_only" (limit to B)
    genome_config.picbreeder_mutation_mode = "all"

    config.picbreeder_mutation_strength = genome_config.picbreeder_mutation_strength
    config.picbreeder_weight_power_min = genome_config.picbreeder_weight_power_min
    config.picbreeder_weight_power_max = genome_config.picbreeder_weight_power_max

    reproduction_config = config.reproduction_config
    default_rate = None
    if hasattr(reproduction_config, "crossover_rate"):
        default_rate = getattr(reproduction_config, "picbreeder_default_crossover_rate", None)
        if default_rate is None:
            default_rate = getattr(reproduction_config, "crossover_rate", 0.0)
            setattr(reproduction_config, "picbreeder_default_crossover_rate", default_rate)
    if hasattr(reproduction_config, "mating"):
        reproduction_config.mating = bool(enable_crossover)
    if default_rate is not None:
        reproduction_config.crossover_rate = default_rate if enable_crossover else 0.0
    if hasattr(reproduction_config, "mutation_repeats"):
        reproduction_config.mutation_repeats = 0


class PicbreederGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self._node_affinities: Dict[int, str] = {}

    def _require_affinity_map(self) -> Dict[int, str]:
        mapping = getattr(self, "_node_affinities", None)
        if mapping is None:
            raise RuntimeError(
                "PicbreederGenome is missing its node affinity map. "
                "Ensure configure_new has been called and cloning preserves affinities."
            )
        return mapping

    @staticmethod
    def _require_external_affinities(genome, label: str) -> Dict[int, str]:
        mapping = getattr(genome, "_node_affinities", None)
        if mapping is None:
            raise RuntimeError(f"{label} genome is missing _node_affinities; cloning or initialization is broken.")
        return mapping

    _ACTIVATION_CHOICES = (
        "sin",
        "sigmoid",
        "gaussian",
    )

    _INPUT_ACTIVATION_IMPL = {
        "identity": _identity,
        "sin": _sine,
        "cos": _cosine,
        "sigmoid": _sigmoid_unit,
        "gaussian": _gaussian_unit,
        "sawtooth_full": _full_sawtooth,
        "square": _square_wave,
    }

    def _initialize_affinities(self, config) -> None:
        mapping: Dict[int, str] = {}
        for key in config.input_keys:
            mapping[int(key)] = "grey"
        for key in self.nodes.keys():
            mapping[int(key)] = mapping.get(int(key), "grey")
        self._node_affinities = mapping

    def _inherit_affinities(self, parent1, parent2) -> None:
        mapping: Dict[int, str] = {}
        parent_maps = (
            self._require_external_affinities(parent1, "Parent #1"),
            self._require_external_affinities(parent2, "Parent #2"),
        )
        for key in self.nodes.keys():
            affinity = None
            for pm in parent_maps:
                value = pm.get(int(key))
                if value is not None:
                    affinity = value
                    break
            mapping[int(key)] = affinity or "grey"
        self._node_affinities = mapping

    def set_node_affinity(self, node_key: int, affinity: str) -> None:
        mapping = self._require_affinity_map()
        mapping[int(node_key)] = affinity

    def get_node_affinity(self, node_key: int) -> str:
        mapping = self._require_affinity_map()
        key = int(node_key)
        if key not in mapping:
            raise RuntimeError(
                f"Node key {key} has no affinity assigned; this should never happen. "
                "Ensure new nodes set affinities immediately."
            )
        return mapping[key]

    def _target_affinity(self, config) -> Optional[str]:
        mode = str(getattr(config, "picbreeder_mutation_mode", "all")).lower()
        if mode == "color_only":
            return "color"
        if mode == "structure_only":
            return "grey"
        return None

    def _get_mutation_strength(self, config) -> float:
        try:
            value = float(getattr(config, "picbreeder_mutation_strength", 0.5))
        except Exception:
            value = 0.5
        if math.isnan(value) or math.isinf(value):
            value = 0.5
        return max(0.0, min(1.0, value))

    def configure_new(self, config) -> None:
        super().configure_new(config)
        self._initialize_affinities(config)
        self.fitness = None
        inputs_enabled = getattr(config, "picbreeder_enable_input_activations", False)
        self._input_activations_enabled = inputs_enabled
        if inputs_enabled:
            self._initialize_input_activations(config)
        else:
            self._clear_input_activations()
        enabled = getattr(config, "picbreeder_enable_output_activations", False)
        if enabled:
            self._output_activations_enabled = True
            self._initialize_output_activations(config)
        else:
            self._output_activations_enabled = False
            self._clear_output_activations()

    def configure_crossover(self, parent1, parent2, config) -> None:
        super().configure_crossover(parent1, parent2, config)
        self._inherit_affinities(parent1, parent2)
        self.fitness = None
        inputs_enabled = getattr(config, "picbreeder_enable_input_activations", False)
        self._input_activations_enabled = inputs_enabled
        if inputs_enabled:
            self._inherit_input_activations(parent1, parent2, config)
        else:
            self._clear_input_activations()
        enabled = getattr(config, "picbreeder_enable_output_activations", False)
        if enabled:
            self._output_activations_enabled = True
            self._inherit_output_activations(parent1, parent2, config)
        else:
            self._output_activations_enabled = False
            self._clear_output_activations()

    def mutate(self, config) -> None:
        inputs_enabled, outputs_enabled = self._sync_activation_controls(config)

        try:
            mode = str(getattr(config, "picbreeder_mutation_mode", "all")).lower()
        except Exception:
            mode = "all"

        if mode == "color_only":
            self._apply_affinity_generator(config, "color")
        elif mode == "structure_only":
            self._apply_affinity_generator(config, "grey")
        else:
            self._apply_super_sweet_generator(config)

        if inputs_enabled:
            self._mutate_input_activations(config)
        if outputs_enabled:
            self._mutate_output_activations(config)

    def _sync_activation_controls(self, config) -> Tuple[bool, bool]:
        outputs_enabled = bool(getattr(config, "picbreeder_enable_output_activations", False))
        self._output_activations_enabled = outputs_enabled
        if outputs_enabled:
            if not hasattr(self, "_output_activation_names"):
                self._initialize_output_activations(config)
            else:
                self._set_output_activation_names(self._output_activation_names)
        else:
            self._clear_output_activations()

        inputs_enabled = bool(getattr(config, "picbreeder_enable_input_activations", False))
        self._input_activations_enabled = inputs_enabled
        if inputs_enabled:
            if not hasattr(self, "_input_activation_names"):
                self._initialize_input_activations(config)
            else:
                self._set_input_activation_names(self._input_activation_names)
        else:
            self._clear_input_activations()

        return inputs_enabled, outputs_enabled

    def _apply_super_sweet_generator(self, config) -> None:
        generator = self._select_generator(config)
        if generator == "add_node":
            self._mutate_add_node(config)
        elif generator == "add_connection":
            self._mutate_super_add_connection(config)
        elif generator == "mutate_activation":
            self._mutate_activation(config)
        else:
            self._mutate_weights(config)

    def _apply_affinity_generator(self, config, affinity: str) -> None:
        generator = self._select_generator(config)
        if generator == "add_node":
            self._mutate_add_node(config, target_affinity=affinity)
        elif generator == "add_connection":
            self._add_connection_between_affinities(config, affinity, affinity)
        elif generator == "mutate_activation":
            self._mutate_activation(config, target_affinity=affinity)
        else:
            self._mutate_weights(config, target_affinity=affinity)

    def _select_generator(self, config) -> str:
        raw_weights = getattr(config, "picbreeder_generator_weights", None)
        if not raw_weights:
            return "mutate_weights"
        weights: List[Tuple[str, float]] = []
        for name, weight in raw_weights:
            try:
                value = float(weight)
            except Exception:
                continue
            if value <= 0.0:
                continue
            weights.append((str(name), value))
        if not weights:
            return "mutate_weights"
        total = sum(weight for _, weight in weights)
        threshold = random.random() * total
        for name, weight in weights:
            threshold -= weight
            if threshold <= 0.0:
                return name
        return weights[-1][0]

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
        if not getattr(self, "_input_activations_enabled", False):
            return
        if not hasattr(self, "_input_activation_names"):
            self._initialize_input_activations(config)
        rate = getattr(config, "picbreeder_activation_rate", 0.0)
        if rate <= 0.0:
            return
        affinity = self._target_affinity(config)
        if affinity == "color":
            return
        options = tuple(getattr(config, "activation_options", self._ACTIVATION_CHOICES))
        changed = False
        allowed = None
        outputs_reached = None
        if affinity is None:
            allowed = self._get_allowed_outputs(config)
            if allowed is not None:
                outputs_reached = self._compute_outputs_reached(config)
        input_keys_ordered = list(config.input_keys)
        for idx in range(len(self._input_activation_names)):
            if affinity is None and allowed is not None and outputs_reached is not None:
                input_key = input_keys_ordered[idx] if idx < len(input_keys_ordered) else None
                if input_key is not None and not self._is_node_allowed(input_key, allowed, outputs_reached):
                    continue
            if random.random() < rate:
                self._input_activation_names[idx] = random.choice(options)
                changed = True
        if changed:
            self._set_input_activation_names(self._input_activation_names)

    def _clear_input_activations(self) -> None:
        for attribute in ("_input_activation_names", "_input_activation_funcs"):
            if hasattr(self, attribute):
                delattr(self, attribute)

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

    def _mutate_activation(self, config, target_affinity: Optional[str] = None) -> bool:
        rate = getattr(config, "picbreeder_activation_rate", 0.0)
        if rate <= 0.0:
            return False

        options = tuple(getattr(config, "activation_options", self._ACTIVATION_CHOICES))
        if not options:
            return False

        mutated = False
        allowed = None
        outputs_reached = None
        if target_affinity is None:
            allowed = self._get_allowed_outputs(config)
            if allowed is not None:
                outputs_reached = self._compute_outputs_reached(config)
        for key, node in self.nodes.items():
            if key in config.input_keys:
                continue
            if target_affinity is not None:
                if self.get_node_affinity(key) != target_affinity:
                    continue
            elif allowed is not None and outputs_reached is not None:
                if not self._is_node_allowed(key, allowed, outputs_reached):
                    continue
            if random.random() < rate:
                node.activation = random.choice(options)
                mutated = True
        return mutated

    def _mutate_add_node(self, config, target_affinity: Optional[str] = None) -> bool:
        candidates = list(self.connections.values())
        if target_affinity is not None:
            candidates = [cg for cg in candidates if self.get_node_affinity(cg.key[1]) == target_affinity]
        else:
            allowed = self._get_allowed_outputs(config)
            outputs_reached = self._compute_outputs_reached(config) if allowed is not None else None
            if allowed is not None and outputs_reached is not None:
                candidates = [cg for cg in candidates if self._is_conn_allowed(cg.key, allowed, outputs_reached)]
        if not candidates:
            return False

        conn_to_split = random.choice(candidates)
        input_key, output_key = conn_to_split.key

        new_node_id = config.get_new_node_key(self.nodes)
        new_node = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = new_node
        if target_affinity is not None:
            self.set_node_affinity(new_node_id, target_affinity)
        else:
            inferred = self._infer_new_node_affinity(input_key, output_key, config)
            self.set_node_affinity(new_node_id, inferred)

        self.add_connection(config, input_key, new_node_id, self._random_weight(), True)
        self.add_connection(config, new_node_id, output_key, self._random_weight(), True)
        return True

    def _mutate_super_add_connection(self, config) -> bool:
        success = False
        if random.random() < getattr(config, "picbreeder_color_link_rate", 0.5):
            success = self._add_connection_between_affinities(config, "color", "color") or success
        if random.random() < getattr(config, "picbreeder_grey_link_rate", 0.5):
            success = self._add_connection_between_affinities(config, "grey", "grey") or success
        if random.random() < getattr(config, "picbreeder_grey_to_color_link_rate", 0.1):
            success = self._add_connection_between_affinities(config, "grey", "color") or success
        return success

    def _add_connection_between_affinities(self, config, source_affinity: str, dest_affinity: str) -> bool:
        sources = [
            key for key in self._gather_nodes_with_affinity(config, source_affinity, include_inputs=True)
            if key not in config.output_keys
        ]
        destinations = [
            key for key in self._gather_nodes_with_affinity(config, dest_affinity, include_outputs=True)
            if key not in config.input_keys
        ]
        if not sources or not destinations:
            return False

        for _ in range(64):
            in_node = random.choice(sources)
            out_node = random.choice(destinations)
            if in_node == out_node:
                continue
            key = (in_node, out_node)
            existing = self.connections.get(key)
            if existing is not None:
                if not existing.enabled and config.check_structural_mutation_surer():
                    # This should never be the case, since Picbreeder mutation does not disable connections.
                    existing.enabled = True
                    raise RuntimeError(
                        "While adding a new connection, found a disabled connection. "
                        "But Picbreeder mutation should never disable connections."
                    )
                continue
            if config.feed_forward and creates_cycle(list(self.connections), key):
                continue
            connection = self.create_connection(config, in_node, out_node)
            connection.weight = self._random_weight()
            self.connections[key] = connection
            return True
        return False

    def _mutate_weights(self, config: neat.Config, target_affinity: Optional[str] = None) -> bool:
        rate = getattr(config, "picbreeder_weight_mutate_rate", 0.0)
        if rate <= 0.0:
            return False

        min_power = config.picbreeder_weight_power_min
        max_power = config.picbreeder_weight_power_max
        if max_power < min_power:
            raise ValueError("picbreeder_weight_power_max must be >= picbreeder_weight_power_min")
        sigma = min_power + (max_power - min_power) * self._get_mutation_strength(config)
        if sigma <= 0.0:
            return False

        mutated = False
        allowed = None
        outputs_reached = None
        if target_affinity is None:
            allowed = self._get_allowed_outputs(config)
            if allowed is not None:
                outputs_reached = self._compute_outputs_reached(config)
        for key, connection in self.connections.items():
            if not connection.enabled:
                continue
            if target_affinity is not None:
                if self.get_node_affinity(key[0]) != target_affinity or self.get_node_affinity(key[1]) != target_affinity:
                    continue
            elif allowed is not None and outputs_reached is not None:
                if not self._is_conn_allowed(key, allowed, outputs_reached):
                    continue
            if random.random() < rate:
                connection.weight = self._clamp_weight(config, connection.weight + random.gauss(0.0, sigma))
                mutated = True
        return mutated

    def _random_weight(self) -> float:
        return random.uniform(-3.0, 3.0)

    def _clamp_weight(self, config, value: float) -> float:
        minimum = getattr(config, "weight_min_value", -3.0)
        maximum = getattr(config, "weight_max_value", 3.0)
        return max(minimum, min(maximum, value))

    def _infer_new_node_affinity(self, input_key: int, output_key: int, config) -> str:
        if input_key in config.output_keys:
            return self.get_node_affinity(output_key)
        source_aff = self.get_node_affinity(input_key)
        dest_aff = self.get_node_affinity(output_key)
        return random.choice([source_aff, dest_aff])

    def _gather_nodes_with_affinity(
        self,
        config,
        affinity: str,
        include_inputs: bool = False,
        include_outputs: bool = True,
    ) -> List[int]:
        result: Set[int] = set()
        for key in self.nodes.keys():
            if self.get_node_affinity(key) == affinity:
                result.add(int(key))
        if include_inputs:
            for key in config.input_keys:
                if self.get_node_affinity(key) == affinity:
                    result.add(int(key))
        if include_outputs:
            for key in config.output_keys:
                if self.get_node_affinity(key) == affinity:
                    result.add(int(key))
        return list(result)


class GenerationCheckpointer(BaseReporter):
    def __init__(self, output_dir: Path, latest_filename: str = LATEST_POPULATION_FILENAME):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_generation: Optional[int] = None
        self.latest_path = self.output_dir / latest_filename

    def start_generation(self, generation: int) -> None:
        self.current_generation = generation

    def end_generation(self, config: neat.Config, population: Dict[int, neat.DefaultGenome], species_set) -> None:
        if self.current_generation is None:
            return
        next_generation = self.current_generation + 1
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix="latest_population_",
            suffix=".tmp",
            dir=str(self.output_dir),
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        try:
            with gzip.open(tmp_path, "wb", compresslevel=5) as handle:
                data = (next_generation, config, population, species_set, random.getstate())
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(self.latest_path)
        finally:
            tmp_path.unlink(missing_ok=True)


def seed_initial_population(population, genome_config) -> None:
    activation_choices = [opt for opt in getattr(genome_config, "activation_options", [])]
    for genome in population.population.values():
        _seed_picbreeder_genome(genome, genome_config, activation_choices)
    _initialize_color_bootstrap(population, genome_config)


def _seed_picbreeder_genome(
    genome: neat.DefaultGenome,
    genome_config,
    activation_choices: Sequence[str],
) -> None:
    """Rebuild the initial CPPN scaffold to match the legacy Picbreeder Java client.

    The original client randomised a grayscale-only network (inputs -> hidden -> ink)
    and only afterwards grafted on the InkToHSB colour subnet. We recreate that exact
    scaffold so random populations stay functionally aligned with the legacy seeds.
    """

    input_keys = [int(key) for key in getattr(genome_config, "input_keys", ())]
    output_keys = [int(key) for key in getattr(genome_config, "output_keys", ())]
    brightness_key = getattr(genome_config, "picbreeder_brightness_output_key", None)
    if brightness_key is None and output_keys:
        brightness_key = int(output_keys[-1])
    color_keys = tuple(int(key) for key in getattr(genome_config, "picbreeder_color_output_keys", ()))

    keep_keys = set(input_keys + output_keys)
    for key in list(genome.nodes.keys()):
        if int(key) not in keep_keys:
            genome.nodes.pop(key, None)

    for key in keep_keys:
        node = genome.nodes.get(key)
        if node is None:
            node = genome.create_node(genome_config, int(key))
            genome.nodes[int(key)] = node
        if int(key) in input_keys:
            genome.set_node_affinity(key, "grey")
            continue
        if activation_choices:
            node.activation = random.choice(activation_choices)
        affinity = "color" if int(key) in color_keys else "grey"
        genome.set_node_affinity(key, affinity)

    genome.connections.clear()

    hidden_count = max(0, int(getattr(genome_config, "num_hidden", 0)))
    grey_hidden_keys: List[int] = []
    for _ in range(hidden_count):
        node_key = genome_config.get_new_node_key(genome.nodes)
        node_gene = genome.create_node(genome_config, node_key)
        if activation_choices:
            node_gene.activation = random.choice(activation_choices)
        genome.nodes[int(node_key)] = node_gene
        genome.set_node_affinity(node_key, "grey")
        grey_hidden_keys.append(int(node_key))

    if brightness_key is None:
        genome.fitness = None
        return

    if grey_hidden_keys:
        for src in input_keys:
            for dst in grey_hidden_keys:
                _create_connection(genome, genome_config, src, dst)
        for src in grey_hidden_keys:
            _create_connection(genome, genome_config, src, brightness_key)
    else:
        for src in input_keys:
            _create_connection(genome, genome_config, src, brightness_key)

    genome.fitness = None


def _create_connection(genome: neat.DefaultGenome, genome_config, src: int, dst: int) -> None:
    key = (int(src), int(dst))
    if key in genome.connections:
        connection = genome.connections[key]
        connection.enabled = True
        connection.weight = genome._random_weight()
        return
    connection = genome.create_connection(genome_config, int(src), int(dst))
    genome.connections[key] = connection


def _initialize_color_bootstrap(population, genome_config) -> None:
    """Mimic InkToHSB: hue/saturation get deterministic activations and zeroed inputs."""
    brightness_key = getattr(genome_config, "picbreeder_brightness_output_key", None)
    color_keys = tuple(int(key) for key in getattr(genome_config, "picbreeder_color_output_keys", ()))
    if brightness_key is None or len(color_keys) < 2:
        return
    hue_key = int(color_keys[0])
    sat_key = int(color_keys[1])
    brightness_key = int(brightness_key)
    color_targets = (hue_key, sat_key)
    input_keys = [int(key) for key in getattr(genome_config, "input_keys", ())]
    hidden_color_nodes = int(getattr(genome_config, "picbreeder_hidden_color_nodes", 0))

    def _ensure_connection(genome: neat.DefaultGenome, src: int, dst: int):
        key = (src, dst)
        connection = genome.connections.get(key)
        if connection is None:
            connection = genome.create_connection(genome_config, src, dst)
            genome.connections[key] = connection
        return connection

    for genome in population.population.values():
        nodes = genome.nodes
        if hue_key in nodes:
            nodes[hue_key].activation = "sin"
            if hasattr(genome, "set_node_affinity"):
                genome.set_node_affinity(hue_key, "color")
        if sat_key in nodes:
            nodes[sat_key].activation = "sigmoid"
            if hasattr(genome, "set_node_affinity"):
                genome.set_node_affinity(sat_key, "color")
        if hasattr(genome, "set_node_affinity"):
            genome.set_node_affinity(brightness_key, "grey")

        # Remove any pre-existing connections targeting hue/saturation so the scaffold
        # matches the legacy grayscale-first ordering.
        to_remove = [
            key for key in list(genome.connections.keys()) if int(key[1]) in color_targets
        ]
        for key in to_remove:
            genome.connections.pop(key, None)

        color_hidden_keys: List[int] = []
        for _ in range(max(hidden_color_nodes, 0)):
            node_key = genome_config.get_new_node_key(genome.nodes)
            node_gene = genome.create_node(genome_config, node_key)
            genome.nodes[node_key] = node_gene
            if hasattr(genome, "set_node_affinity"):
                genome.set_node_affinity(node_key, "color")
            color_hidden_keys.append(node_key)

        for target in color_targets:
            _ensure_connection(genome, brightness_key, target)

        sources = input_keys + [brightness_key]
        if color_hidden_keys:
            for source in sources:
                for hidden_key in color_hidden_keys:
                    _ensure_connection(genome, int(source), hidden_key)
            for hidden_key in color_hidden_keys:
                for target in color_targets:
                    connection = _ensure_connection(genome, hidden_key, target)
                    connection.weight = 0.0
        else:
            for source in sources:
                for target in color_targets:
                    _ensure_connection(genome, int(source), target)


def sync_population_output_activations(population, genome_config) -> None:
    """Ensure genomes in a population honour the configured output activation toggle."""
    for genome in population.population.values():
        sync_fn = getattr(genome, "sync_output_activations", None)
        if sync_fn is not None:
            sync_fn(genome_config)


def sync_population_node_indexer(population: neat.Population) -> None:
    """Advance the genome node indexer so future mutations use unique node IDs."""
    genome_config = population.config.genome_config
    max_key: Optional[int] = None
    for genome in population.population.values():
        if genome.nodes:
            genome_max = max(genome.nodes)
            if max_key is None or genome_max > max_key:
                max_key = int(genome_max)
    if max_key is None:
        output_keys = list(getattr(genome_config, "output_keys", ()))
        max_key = max(output_keys) if output_keys else 0
    genome_config.node_indexer = count(int(max_key) + 1)


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
