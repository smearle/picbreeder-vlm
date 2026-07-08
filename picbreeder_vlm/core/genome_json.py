"""Serialize a PicbreederGenome to the compact JSON the browser CPPN engine reads.

This is the bridge between the Python research evolver (which pickles genomes) and
the static site's `cppn.js` (a faithful JS port of the same CPPN/NEAT engine, which
cannot unpickle and instead renders from this JSON). The schema is intentionally
minimal — nodes (key/activation/affinity/input-flag), connections (from/to/weight/
enabled), and the output/input activation transforms the canonical runs use. It is
the single source of truth shared by `tools/build_breed_data.py` (curated bundle)
and `tools/export_genome_json.py` (per-run shards pushed to HF).
"""
from __future__ import annotations

from typing import Any, Dict


def genome_to_json(genome: Any) -> Dict[str, Any]:
    """Return the cppn.js-renderable JSON for one genome (color-mode agnostic)."""
    aff = getattr(genome, "_node_affinities", {}) or {}
    nodes = [
        {"key": int(k), "activation": n.activation, "affinity": aff.get(int(k), "grey"), "input": int(k) < 0}
        for k, n in genome.nodes.items()
    ]
    conns = [
        {"from": int(i), "to": int(o), "weight": round(float(c.weight), 6), "enabled": bool(c.enabled)}
        for (i, o), c in genome.connections.items()
    ]
    j: Dict[str, Any] = {"nodes": nodes, "connections": conns}
    if getattr(genome, "_output_activations_enabled", False) and getattr(genome, "_output_activation_names", None):
        j["outAct"] = list(genome._output_activation_names)
    if getattr(genome, "_input_activations_enabled", False) and getattr(genome, "_input_activation_names", None):
        j["inAct"] = list(genome._input_activation_names)
    return j
