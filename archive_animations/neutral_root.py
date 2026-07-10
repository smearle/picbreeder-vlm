"""A neutral grey "super-root" PicbreederGenome.

Phylogeny in this archive is a forest of 76 disjoint trees -- each rooted at an
agent's gen-0 random CPPN. To stitch the forest into a single connected graph
for walker animations, every real root needs a parent. This module hand-builds
that parent: a structurally-compatible PicbreederGenome whose CPPN evaluates
to a constant ``(hue=0, sat=0, brightness=0.5)`` everywhere, rendering as
solid 128/128/128 grey. Interpolating from this genome to any random gen-0
root smoothly grows the latter's structure (connections / hidden nodes /
output activations) in from zero.

How constant grey is achieved:
- No hidden nodes, no connections. Each output node's value is therefore
  ``activation(bias + response * 0)`` with no inputs contributing.
- This codebase uses BIPOLAR activations: ``sigmoid_unit(z) = 2*sigmoid(z) - 1``
  (range [-1, 1], zero at z=0); ``sine``/``identity`` likewise hit 0 at 0;
  ``gaussian_unit``/``cos`` hit 1 at 0. None of them produce 0.5 at z=0, so
  reaching brightness 0.5 requires a non-zero bias on the brightness output.
- Choices:
    * hue   (key 0): ``sin``,     bias = 0           -> sin(0) = 0      (sat==0 hides hue)
    * sat   (key 1): ``sin``,     bias = 0           -> sin(0) = 0      (clamps to 0 -> pure grey)
    * value (key 2): ``sigmoid``, bias = +ln(3)      -> 2/(1+e^-x)-1 = 0.5 (brightness becomes 0.5)
- ``_output_activations_enabled`` is False so ``transform_outputs`` is a no-op
  (published genomes have it enabled; ``render_lineage_animation`` smoothly
  toggles it back on during interpolation).

Because the existing interpolator in ``render_lineage_animation`` does not
lerp node biases (it assumes biases never change, which holds for this whole
archive -- every genome's biases are exactly 0), the neutral root's non-zero
bias on the brightness output is lost at frame 0 unless we patch it in. The
helper ``patched_assign_interpolated_values`` below adds bias interpolation as
a strict superset of the original behaviour; for genomes with identical
parent/child biases (i.e. everything in this archive) it is a no-op.
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import neat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root

from picbreeder_vlm._paths import NEAT_CONFIG_PATH
from picbreeder_vlm.core.neat_components import PicbreederGenome


# bias on the brightness output that puts sigmoid_unit(bias) = 0.5
_BRIGHTNESS_BIAS = math.log(3.0)   # ≈ 1.0986


def make_neutral_root_genome(
    config: neat.Config,
    key: int = -10000,
) -> PicbreederGenome:
    """Construct the neutral super-root PicbreederGenome."""
    g = PicbreederGenome(key)
    gc = config.genome_config

    input_keys = [int(k) for k in gc.input_keys]
    output_keys = [int(k) for k in gc.output_keys]
    brightness_key = getattr(gc, "picbreeder_brightness_output_key", None)
    if brightness_key is None:
        brightness_key = int(output_keys[-1])
    color_keys = tuple(int(k) for k in getattr(gc, "picbreeder_color_output_keys", (0, 1)) or ())

    # nodes ----------------------------------------------------------------
    g.nodes = {}
    for out_key in output_keys:
        node = g.create_node(gc, int(out_key))
        node.response = 1.0
        node.aggregation = "sum"
        if int(out_key) == int(brightness_key):
            node.activation = "sigmoid"
            node.bias = _BRIGHTNESS_BIAS      # -> sigmoid_unit(bias) = 0.5
        else:
            # hue / saturation: pick any activation whose value at 0 is <= 0;
            # sine is zero, so sat clamps to 0 (pure grey) and hue is irrelevant.
            node.activation = "sin"
            node.bias = 0.0
        g.nodes[int(out_key)] = node
    # No hidden nodes. We add one zero-weight connection from an input to each
    # output node so NEAT's FeedForwardNetwork includes the output nodes in
    # node_evals when the genome is rendered *standalone*. With no connections
    # at all, outputs would be skipped and default to 0 -> pure black instead
    # of grey. ``build_superset`` ignores parent connections that don't appear
    # in the child, so these phantom edges have no effect on interpolation.
    g.connections = {}
    first_input = int(input_keys[0])
    for out_key in output_keys:
        conn = g.create_connection(gc, first_input, int(out_key))
        conn.weight = 0.0
        conn.enabled = True
        g.connections[(first_input, int(out_key))] = conn

    # affinities -----------------------------------------------------------
    affinities = {int(k): "grey" for k in input_keys}
    for k in output_keys:
        affinities[int(k)] = "color" if int(k) in color_keys else "grey"
    g._node_affinities = affinities

    # output / input activation toggles ------------------------------------
    g._output_activations_enabled = False
    g._output_activation_names = []
    g._output_activation_funcs = []
    g._input_activations_enabled = False

    g.fitness = None
    return g


# --------------- bias-aware build_superset + interpolation ---------------

def build_superset_with_biases(parent, child):
    """Wrapper around render_lineage_animation.build_superset that also
    captures per-node start/end biases so the interpolator can lerp them.

    Returns: (superset, node_pairs, conn_pairs, output_stats, bias_pairs)
    where ``bias_pairs[node_key] = (start_bias, end_bias)``. If the parent
    lacks a node, ``start_bias`` falls back to the child's bias (no jump).
    """
    from picbreeder_vlm.viz.render_lineage_animation import build_superset
    superset, node_pairs, conn_pairs, output_stats = build_superset(parent, child)
    bias_pairs: Dict[int, Tuple[float, float]] = {}
    for node_key, child_node in child.nodes.items():
        end_bias = float(child_node.bias)
        p_node = parent.nodes.get(node_key)
        start_bias = float(p_node.bias) if p_node is not None else end_bias
        bias_pairs[int(node_key)] = (start_bias, end_bias)
    return superset, node_pairs, conn_pairs, output_stats, bias_pairs


def patched_assign_interpolated_values(
    genome, node_pairs, conn_pairs, config, step_index, total_steps,
    step_fraction, activation_cache, output_activation_stats, bias_pairs=None,
):
    """Like ``render_lineage_animation.assign_interpolated_values`` but also
    lerps node biases when ``bias_pairs`` is provided. Pure superset: with
    ``bias_pairs=None`` it is identical to the original."""
    from picbreeder_vlm.viz.render_lineage_animation import assign_interpolated_values
    assign_interpolated_values(
        genome, node_pairs, conn_pairs, config, step_index, total_steps,
        step_fraction, activation_cache, output_activation_stats,
    )
    if not bias_pairs:
        return
    for node_key, (sb, eb) in bias_pairs.items():
        if node_key in genome.nodes:
            genome.nodes[node_key].bias = (1.0 - step_fraction) * sb + step_fraction * eb


def render_neutral_to_child_frames(
    parent, child, config, *, steps: int, width: int, height: int,
    variant_mode: str = "color", color_enabled: bool = True,
):
    """End-to-end: build superset from ``parent`` -> ``child`` (with bias
    interpolation), render ``steps`` frames. Returns list of PIL images.
    """
    from picbreeder_vlm.core.rendering import render_genome_image
    from picbreeder_vlm.viz.render_lineage_animation import pick_variant
    superset, node_pairs, conn_pairs, output_stats, bias_pairs = \
        build_superset_with_biases(parent, child)
    activation_cache: Dict[Tuple[str, str, int], str] = {}
    frames = []
    for idx in range(steps):
        fraction = idx / (steps - 1) if steps > 1 else 0.0
        working = copy.deepcopy(superset)
        patched_assign_interpolated_values(
            working, node_pairs, conn_pairs, config, idx, steps, fraction,
            activation_cache, output_stats, bias_pairs=bias_pairs,
        )
        gray, color = render_genome_image(working, config, width, height)
        frames.append(pick_variant(gray, color, variant_mode, color_enabled).convert("RGB"))
    return frames


if __name__ == "__main__":
    import argparse
    import gzip, pickle, zipfile
    from picbreeder_vlm.core.neat_components import apply_picbreeder_config_defaults, InteractiveStagnation
    from picbreeder_vlm.core.picbreeder_reproduction import PicbreederReproduction
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(NEAT_CONFIG_PATH))
    ap.add_argument("--out-dir", type=Path, default=Path("archive_animations/out"))
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=9, help="frames in test strip")
    ap.add_argument("--agent-zip", type=Path,
                    default=Path("sweep_logs/sweep/th0_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s4/agents/agent_000.zip"),
                    help="agent .zip to pull a gen-0 random root from")
    args = ap.parse_args()

    config = neat.Config(
        PicbreederGenome, PicbreederReproduction, neat.DefaultSpeciesSet,
        InteractiveStagnation, args.config,
    )
    apply_picbreeder_config_defaults(
        config, enable_output_activations=True, enable_input_activations=False, enable_crossover=True,
    )
    neutral = make_neutral_root_genome(config)
    print(f"neutral nodes: {sorted(neutral.nodes.keys())}")
    for k, n in sorted(neutral.nodes.items()):
        print(f"  node {k}: activation={n.activation!r} bias={n.bias} response={n.response}")

    # Pull a real gen-0 random root for interpolation
    with zipfile.ZipFile(args.agent_zip) as zf:
        with zf.open(f"{args.agent_zip.stem}/populations/selected_gen_000.pkl.gz") as f, gzip.open(f) as gz:
            child = pickle.load(gz)["selected"][0]["genome"]

    frames = render_neutral_to_child_frames(
        neutral, child, config, steps=args.steps, width=args.size, height=args.size,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    strip = Image.new("RGB", (args.size * args.steps, args.size))
    for i, fr in enumerate(frames):
        strip.paste(fr, (args.size * i, 0))
    out = args.out_dir / "neutral_to_root_strip.png"
    strip.save(out)
    for i, fr in enumerate(frames):
        pix = fr.load()
        print(f"  frame {i}/{args.steps-1}: center RGB = {pix[args.size//2, args.size//2]}, "
              f"corner = {pix[0, 0]}")
    print(f"wrote {out}")
