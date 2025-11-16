#!/usr/bin/env python3
"""Render legacy Picbreeder (Java client) genomes as Graphviz diagrams or images.

This helper understands the XML files produced by the original Picbreeder
client (``client.jar`` inside ``webneat/``). It parses the node/link structure
stored in ``genome.xml`` (or zipped variants) and emits an SVG (or other
Graphviz-supported format) that mirrors the topology annotations we generate
for NEAT-Python runs. Optionally, it can convert the legacy genome into a
``neat.DefaultGenome`` and render a high-resolution color (or grayscale) image
using the modern picture renderer.

Example usage::

    python tools/render_legacy_genome.py path/to/genome.xml \
        --output-dir legacy_diagrams --format svg

Requires the ``graphviz`` Python package (already listed in ``requirements.txt``)
and a Graphviz binary (``brew install graphviz`` on macOS). Image rendering
requires ``Pillow`` and ``numpy`` (both already included in the project
dependencies).
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET

import graphviz
import neat
from neat.species import DefaultSpeciesSet

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neat.stagnation import DefaultStagnation

from neat_components import (
    InteractiveStagnation,
    PicbreederGenome,
    apply_picbreeder_config_defaults,
)
from picbreeder_reproduction import PicbreederReproduction
from rendering import render_genome_image


@dataclass(frozen=True)
class LegacyNode:
    key: str
    branch: str
    local_id: str
    type: str
    label: str
    activation: str
    affinity: str
    bias: float


@dataclass(frozen=True)
class LegacyLink:
    key: str
    branch: str
    local_id: str
    source_key: str
    target_key: str
    weight: float


@dataclass(frozen=True)
class LegacyGenome:
    identifier: str
    age: int
    phenotype: str
    parents: Tuple[str, ...]
    nodes: Dict[str, LegacyNode]
    links: Tuple[LegacyLink, ...]


def _read_xml(path: Path) -> ET.Element:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
            if not names:
                raise ValueError(f"No XML content found inside {path}")
            with archive.open(names[0]) as handle:
                return ET.fromstring(handle.read())
    return ET.parse(path).getroot()


def _collect_genome_elements(root: ET.Element) -> List[ET.Element]:
    if root.tag == "genome":
        return [root]
    genomes = list(root.findall(".//genome"))
    if not genomes:
        raise ValueError("Could not locate any <genome> elements in legacy file")
    return genomes


def _ensure_genome_root(root: ET.Element, *, genome_index: int) -> ET.Element:
    genomes = _collect_genome_elements(root)
    try:
        return genomes[genome_index]
    except IndexError as exc:
        raise ValueError(
            f"Legacy file only contains {len(genomes)} genome(s); index {genome_index} is out of range"
        ) from exc


def _parse_marking(element: ET.Element, *, fallback_branch: str = "") -> Tuple[str, str, str]:
    if element.tag == "marking":
        marking = element
    else:
        marking = element.find("marking")
        if marking is None:
            # Some legacy files store the identifier attributes directly on the
            # element (e.g. <identifier branch="foo" id="123"/>). Fall back
            # to those when a nested <marking> tag is absent.
            branch = element.get("branch", fallback_branch)
            local_id = element.get("id")
            if local_id is None:
                raise ValueError(f"Missing <marking> element under <{element.tag}>")
            key = f"{branch}:{local_id}" if branch else local_id
            return key, branch, local_id
    branch = marking.get("branch", fallback_branch)
    local_id = marking.get("id")
    if local_id is None:
        raise ValueError("Legacy marking is missing an 'id' attribute")
    key = f"{branch}:{local_id}" if branch else local_id
    return key, branch, local_id


def _parse_parents(parent_block: Optional[ET.Element]) -> Tuple[str, ...]:
    if parent_block is None:
        return tuple()
    parents: List[str] = []
    for identifier in parent_block.findall("identifier"):
        key, branch, local_id = _parse_marking(identifier)
        parents.append(key)
    return tuple(parents)


def _parse_nodes(block: ET.Element) -> Dict[str, LegacyNode]:
    nodes: Dict[str, LegacyNode] = {}
    for node_el in block.findall("node"):
        key, branch, local_id = _parse_marking(node_el)
        activation = (node_el.findtext("activation") or "identity").strip()
        legacy_node = LegacyNode(
            key=key,
            branch=branch,
            local_id=local_id,
            type=node_el.get("type", "hidden"),
            label=node_el.get("label", ""),
            activation=activation,
            affinity=node_el.get("affinity", ""),
            bias=float(node_el.get("bias", "0.0")),
        )
        nodes[legacy_node.key] = legacy_node
    if not nodes:
        raise ValueError("Legacy genome did not contain any nodes")
    return nodes


def _parse_links(block: ET.Element, nodes: Dict[str, LegacyNode]) -> Tuple[LegacyLink, ...]:
    links: List[LegacyLink] = []
    for link_el in block.findall("link"):
        key, branch, local_id = _parse_marking(link_el)
        src_key, _, _ = _parse_marking(link_el.find("source"))
        dst_key, _, _ = _parse_marking(link_el.find("target"))
        if src_key not in nodes:
            raise ValueError(f"Link {key} references unknown source node {src_key}")
        if dst_key not in nodes:
            raise ValueError(f"Link {key} references unknown target node {dst_key}")
        weight = float(link_el.findtext("weight", "0.0"))
        links.append(
            LegacyLink(
                key=key,
                branch=branch,
                local_id=local_id,
                source_key=src_key,
                target_key=dst_key,
                weight=weight,
            )
        )
    return tuple(links)


def load_legacy_genome(path: Path, *, genome_index: int = 0) -> Tuple[LegacyGenome, int]:
    root = _read_xml(path)
    genomes = _collect_genome_elements(root)
    genome_el = _ensure_genome_root(root, genome_index=genome_index)

    identifier_el = genome_el.find("identifier")
    if identifier_el is None:
        raise ValueError("Legacy genome is missing its <identifier> block")
    identifier_key, _, _ = _parse_marking(identifier_el)

    nodes_block = genome_el.find("nodes")
    links_block = genome_el.find("links")
    if nodes_block is None or links_block is None:
        raise ValueError("Legacy genome must include <nodes> and <links> blocks")

    nodes = _parse_nodes(nodes_block)
    links = _parse_links(links_block, nodes)
    parents = _parse_parents(genome_el.find("parents"))
    age = int(genome_el.get("age", "0"))
    phenotype = genome_el.get("phenotype", "structure")

    legacy = LegacyGenome(
        identifier=identifier_key,
        age=age,
        phenotype=phenotype,
        parents=parents,
        nodes=nodes,
        links=links,
    )
    return legacy, len(genomes)


def _format_bias(bias: float) -> str:
    if math.isclose(bias, 0.0, abs_tol=1e-9):
        return "0.0"
    return f"{bias:+.4f}"


def _node_fill(node: LegacyNode) -> str:
    type_colors = {
        "in": "#d3e4ff",
        "out": "#ffe2cc",
        "hidden": "#f5f5f5",
    }
    affinity_colors = {
        "grey": "#d3e4ff",
        "gray": "#d3e4ff",
        "colour": "#fff3bf",
        "color": "#fff3bf",
        "hsb": "#fff3bf",
    }
    if node.type == "in":
        return type_colors["in"]
    if node.type == "out":
        return type_colors["out"]
    if node.affinity:
        return affinity_colors.get(node.affinity.lower(), type_colors.get(node.type, "#f5f5f5"))
    return type_colors.get(node.type, "#f5f5f5")


def _build_node_label(node: LegacyNode) -> str:
    name = node.label if node.label else node.key
    lines = [name, f"id={node.key}", f"type={node.type}"]
    if node.type != "in":
        lines.append(f"act={node.activation}")
        lines.append(f"bias={_format_bias(node.bias)}")
    if node.affinity and node.type != "in":
        lines.append(f"aff={node.affinity}")
    return "\n".join(lines)


def _edge_attributes(weight: float) -> Tuple[str, str, str]:
    color = "#2f9e44" if weight >= 0 else "#e03131"
    penwidth = f"{0.6 + min(4.0, abs(weight)):.2f}"
    label = f"w={weight:+.4f}"
    return color, penwidth, label


def _graphviz_id(key: str) -> str:
    """Return a Graphviz-safe identifier for a legacy node key."""
    return key.replace(":", "__")


def _normalize_activation(raw: Optional[str], config: neat.Config) -> str:
    text = (raw or "identity").strip()
    if "(" in text:
        text = text.split("(", 1)[0]
    name = text.strip().lower()
    aliases = {
        "gauss": "gaussian",
        "gaussian": "gaussian",
        "fullsawtooth": "sawtooth_full",
        "sawtooth": "sawtooth_full",
    }
    resolved = aliases.get(name, name)
    if not config.genome_config.activation_defs.is_valid(resolved):
        raise ValueError(f"Unsupported activation function '{raw}' (resolved to '{resolved}')")
    return resolved


def _infer_scheme(legacy: LegacyGenome) -> str:
    phenotype = legacy.phenotype.lower()
    if phenotype in {"color", "colour", "toggle"}:
        return "color"
    if phenotype in {"gray", "grey", "grayscale", "structure"}:
        return "gray"
    outputs = [node for node in legacy.nodes.values() if node.type == "out"]
    if any(node.affinity.lower() == "color" for node in outputs):
        return "color"
    return "gray"


def _resolve_neat_config_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    filename = "interactive_config_color"
    config_path = root / "picture2d" / filename
    if not config_path.exists():
        raise FileNotFoundError(f"NEAT config not found at {config_path}")
    return config_path


def _build_neat_config() -> neat.Config:
    config_path = _resolve_neat_config_path()
    config = neat.Config(
        PicbreederGenome,
        PicbreederReproduction,
        DefaultSpeciesSet,
        InteractiveStagnation,
        str(config_path),
    )
    apply_picbreeder_config_defaults(config)
    config.pop_size = max(1, getattr(config, "pop_size", 1) or 1)
    return config


def _extract_genome_key(identifier: str) -> int:
    if not identifier:
        return 0
    candidate = identifier.split(":")[-1]
    try:
        return int(candidate)
    except ValueError:
        try:
            return int(identifier)
        except ValueError:
            return 0


def _coerce_node_id(local_id: str) -> Optional[int]:
    try:
        return int(local_id)
    except (TypeError, ValueError):
        return None


def _legacy_to_picbreeder_genome(legacy: LegacyGenome, config: neat.Config) -> PicbreederGenome:
    genome_key = _extract_genome_key(legacy.identifier)
    genome = config.genome_type(genome_key)
    genome.connections = {}
    genome.nodes = {}
    genome.fitness = None
    genome._node_affinities = {}
    genome._input_activations_enabled = False
    genome._output_activations_enabled = False

    genome_config = config.genome_config

    input_keys = list(genome_config.input_keys)
    output_keys = list(genome_config.output_keys)
    available_inputs = list(input_keys)

    def pop_input(preferred: Optional[int]) -> int:
        if preferred is not None and preferred in available_inputs:
            available_inputs.remove(preferred)
            return preferred
        if available_inputs:
            return available_inputs.pop(0)
        raise ValueError("Legacy genome references more input nodes than available in config")

    label_to_input: Dict[str, int] = {}
    if input_keys:
        label_to_input["x"] = input_keys[0]
    if len(input_keys) >= 2:
        label_to_input["y"] = input_keys[1]
    if len(input_keys) >= 3:
        label_to_input["d"] = input_keys[2]
        label_to_input["r"] = input_keys[2]
        label_to_input["radius"] = input_keys[2]
    if len(input_keys) >= 4:
        label_to_input["bias"] = input_keys[-1]
        label_to_input["b"] = input_keys[-1]

    brightness_key = output_keys[-1] if output_keys else None
    available_outputs = list(output_keys)
    color_queue = [key for key in available_outputs if key != brightness_key]

    def pop_output(preferred: Optional[int], *, prefer_color: bool = False) -> int:
        if preferred is not None and preferred in available_outputs:
            available_outputs.remove(preferred)
            if preferred in color_queue:
                color_queue.remove(preferred)
            return preferred
        if prefer_color and color_queue:
            key = color_queue.pop(0)
            available_outputs.remove(key)
            return key
        if available_outputs:
            key = available_outputs.pop(0)
            if key in color_queue:
                color_queue.remove(key)
            return key
        raise ValueError("Legacy genome has more outputs than NEAT config supports")

    used_keys: Set[int] = set(output_keys + input_keys)
    node_key_map: Dict[str, int] = {}

    # Process input nodes first to reserve their keys.
    inputs = [node for node in legacy.nodes.values() if node.type == "in"]
    for node in inputs:
        label = (node.label or node.local_id).strip().lower()
        key = pop_input(label_to_input.get(label))
        node_key_map[node.key] = key
        affinity = node.affinity.lower() if node.affinity else "grey"
        genome.set_node_affinity(int(key), affinity)

    # Ensure outputs are assigned deterministic keys.
    outputs = [node for node in legacy.nodes.values() if node.type == "out"]
    outputs.sort(key=lambda node: 0 if node.affinity.lower() in {"grey", "gray"} else 1)
    for node in outputs:
        label = (node.label or node.local_id).strip().lower()
        affinity = node.affinity.lower()
        preferred: Optional[int] = None
        if brightness_key is not None:
            if affinity in {"grey", "gray"}:
                preferred = brightness_key
            elif label in {"brightness", "value", "v", "lightness", "luminance"}:
                preferred = brightness_key
        prefer_color = affinity == "color"
        key = pop_output(preferred, prefer_color=prefer_color)
        node_key_map[node.key] = key
        used_keys.add(int(key))
        node_gene = genome.create_node(genome_config, key)
        node_gene.bias = node.bias
        node_gene.activation = _normalize_activation(node.activation, config)
        node_gene.response = 1.0
        genome.nodes[int(key)] = node_gene
        genome.set_node_affinity(int(key), affinity)

    hidden_nodes = [node for node in legacy.nodes.values() if node.type not in {"in", "out"}]
    hidden_nodes.sort(key=lambda node: (_coerce_node_id(node.local_id) or 10**9))
    hidden_start = max(used_keys) + 1 if used_keys else 1000
    hidden_key_iter = itertools.count(start=max(1000, hidden_start))

    def allocate_hidden_key(node: LegacyNode) -> int:
        existing = node_key_map.get(node.key)
        if existing is not None:
            return existing
        candidate = _coerce_node_id(node.local_id)
        if candidate is not None and candidate not in used_keys:
            return candidate
        while True:
            fallback = next(hidden_key_iter)
            if fallback not in used_keys:
                return fallback

    for node in hidden_nodes:
        key = allocate_hidden_key(node)
        used_keys.add(int(key))
        node_key_map[node.key] = key
        node_gene = genome.create_node(genome_config, key)
        node_gene.bias = node.bias
        node_gene.activation = _normalize_activation(node.activation, config)
        node_gene.response = 1.0
        genome.nodes[int(key)] = node_gene
        affinity = node.affinity.lower() if node.affinity else "grey"
        genome.set_node_affinity(int(key), affinity)

    # Recreate connections using the translated keys.
    connection_lookup: Dict[Tuple[int, int], LegacyLink] = {}
    for link in legacy.links:
        src_key = node_key_map.get(link.source_key)
        dst_key = node_key_map.get(link.target_key)
        if src_key is None or dst_key is None:
            raise ValueError(f"Link {link.key} references unknown nodes {link.source_key}->{link.target_key}")
        connection_key = (int(src_key), int(dst_key))
        if connection_key in genome.connections:
            conn_gene = genome.connections[connection_key]
        else:
            conn_gene = genome.create_connection(genome_config, int(src_key), int(dst_key))
            genome.connections[connection_key] = conn_gene
        conn_gene.weight = link.weight
        conn_gene.enabled = True
        connection_lookup[connection_key] = link

    # Attach legacy metadata for debugging/visualization helpers.
    legacy_node_labels: Dict[int, str] = {}
    for legacy_key, neat_key in node_key_map.items():
        legacy_node = legacy.nodes.get(legacy_key)
        if legacy_node is None:
            continue
        label = legacy_node.label or legacy_node.local_id or legacy_key
        legacy_node_labels[int(neat_key)] = label

    setattr(genome, "_legacy_node_key_map", dict(node_key_map))
    setattr(genome, "_legacy_connection_lookup", connection_lookup)
    setattr(genome, "_legacy_node_label_map", legacy_node_labels)

    return genome


def render_legacy_genome(genome: LegacyGenome, *, output_dir: Path, fmt: str = "svg") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{genome.identifier.replace(':', '_')}_age{genome.age}"
    graph = graphviz.Digraph(format=fmt)
    graph.attr(rankdir="LR", fontsize="10")
    graph.attr("node", shape="circle", fontsize="9", width="0.4", height="0.4", style="filled")
    graph.attr("edge", fontsize="8")

    node_ids = {key: _graphviz_id(key) for key in genome.nodes}

    for node in genome.nodes.values():
        attrs = {
            "label": _build_node_label(node),
            "fillcolor": _node_fill(node),
            "shape": "box" if node.type == "in" else ("doubleoctagon" if node.type == "out" else "ellipse"),
        }
        graph.node(node_ids[node.key], _attributes=attrs)

    for link in genome.links:
        color, penwidth, label = _edge_attributes(link.weight)
        graph.edge(
            node_ids[link.source_key],
            node_ids[link.target_key],
            _attributes={
                "color": color,
                "penwidth": penwidth,
                "label": label,
            },
        )

    if genome.parents:
        summary = "Parents: " + ", ".join(genome.parents)
        graph.attr(label=f"Legacy Picbreeder genome {genome.identifier}\nAge={genome.age}, Phenotype={genome.phenotype}\n{summary}")
    else:
        graph.attr(label=f"Legacy Picbreeder genome {genome.identifier}\nAge={genome.age}, Phenotype={genome.phenotype}")
    graph.attr(labelloc="t")

    output_path = output_dir / stem
    rendered = Path(graph.render(str(output_path), cleanup=True))
    return rendered


def render_legacy_image(
    genome: LegacyGenome,
    *,
    output_dir: Path,
    size: int,
    variants: Iterable[str] = ("color",),
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scheme = _infer_scheme(genome)
    config = _build_neat_config(scheme)
    neat_genome = _legacy_to_picbreeder_genome(genome, config)

    gray_image, color_image = render_genome_image(neat_genome, config, size, size)
    stem = f"{genome.identifier.replace(':', '_')}_age{genome.age}"

    normalized: List[str] = []
    for variant in variants:
        if variant not in {"color", "gray"}:
            continue
        if variant not in normalized:
            normalized.append(variant)
    if not normalized:
        normalized = ["color"]

    saved: Dict[str, Path] = {}
    if normalized == ["color"]:
        color_path = output_dir / f"{stem}.png"
        color_image.save(color_path, format="PNG")
        saved["color"] = color_path
    elif normalized == ["gray"]:
        gray_path = output_dir / f"{stem}.png"
        gray_image.save(gray_path, format="PNG")
        saved["gray"] = gray_path
    else:
        color_path = output_dir / f"{stem}_color.png"
        gray_path = output_dir / f"{stem}_gray.png"
        color_image.save(color_path, format="PNG")
        gray_image.save(gray_path, format="PNG")
        saved["color"] = color_path
        saved["gray"] = gray_path

    return saved


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Render a legacy Picbreeder genome (genome.xml) as a Graphviz diagram.")
    parser.add_argument("input", type=Path, help="Path to genome.xml (or a .zip containing it)")
    parser.add_argument("--output-dir", type=Path, default=Path("legacy_diagrams"), help="Directory to write the diagram into")
    parser.add_argument("--format", dest="fmt", default="svg", choices=("svg", "png", "pdf"), help="Graphviz output format")
    parser.add_argument("--genome-index", type=int, default=0, help="Zero-based index of the <genome> element to render within the legacy file")
    parser.add_argument("--skip-diagram", action="store_true", help="Skip Graphviz topology rendering and only process images")
    parser.add_argument(
        "--render-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render the genome using the modern picture renderer (color and grayscale variants by default)",
    )
    parser.add_argument("--image-dir", type=Path, default=Path("legacy_renders"), help="Directory to store rendered images (when --render-image is set)")
    parser.add_argument("--image-size", type=int, default=600, help="Width/height of rendered images in pixels")
    parser.add_argument(
        "--image-variant",
        choices=("color", "gray", "both"),
        default="both",
        help="Which rendered variant(s) to export when --render-image is enabled",
    )
    args = parser.parse_args(argv)

    if args.skip_diagram and not args.render_image:
        parser.error("Nothing to do: enable diagram rendering or pass --render-image.")

    genome, total = load_legacy_genome(args.input, genome_index=args.genome_index)
    index_display = f"{args.genome_index}/{total - 1}" if total > 1 else "0/0"
    print(f"Loaded legacy genome '{genome.identifier}' (index {index_display}) from {args.input}")

    if not args.skip_diagram:
        rendered = render_legacy_genome(genome, output_dir=args.output_dir, fmt=args.fmt)
        print(f"Rendered topology diagram to {rendered}")

    if args.render_image:
        variant_option = args.image_variant
        if variant_option == "both":
            variants = ("color", "gray")
        else:
            variants = (variant_option,)
        image_paths = render_legacy_image(
            genome,
            output_dir=args.image_dir,
            size=args.image_size,
            variants=variants,
        )
        for variant, path in image_paths.items():
            print(f"Rendered {variant} image to {path}")


if __name__ == "__main__":
    main()
