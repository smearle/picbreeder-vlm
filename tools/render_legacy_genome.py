#!/usr/bin/env python3
"""Render legacy Picbreeder (Java client) genomes as Graphviz diagrams.

This helper understands the XML files produced by the original Picbreeder
client (``client.jar`` inside ``webneat/``). It parses the node/link structure
stored in ``genome.xml`` (or zipped variants) and emits an SVG (or other
Graphviz-supported format) that mirrors the topology annotations we generate
for NEAT-Python runs.

Example usage::

    python tools/render_legacy_genome.py path/to/genome.xml \
        --output-dir legacy_diagrams --format svg

Requires the ``graphviz`` Python package (already listed in ``requirements.txt``)
and a Graphviz binary (``brew install graphviz`` on macOS).
"""

from __future__ import annotations

import argparse
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

import graphviz


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


def _ensure_genome_root(root: ET.Element) -> ET.Element:
    if root.tag == "genome":
        return root
    if root.tag == "data":
        genome = root.find("genome")
        if genome is not None:
            return genome
    genome = root.find(".//genome")
    if genome is None:
        raise ValueError("Could not locate <genome> element in legacy file")
    return genome


def _parse_marking(element: ET.Element, *, fallback_branch: str = "") -> Tuple[str, str, str]:
    marking = element.find("marking") if element.tag != "marking" else element
    if marking is None:
        raise ValueError(f"Missing <marking> element under <{element.tag}>")
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


def load_legacy_genome(path: Path) -> LegacyGenome:
    root = _read_xml(path)
    genome_el = _ensure_genome_root(root)

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

    return LegacyGenome(
        identifier=identifier_key,
        age=age,
        phenotype=phenotype,
        parents=parents,
        nodes=nodes,
        links=links,
    )


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


def render_legacy_genome(genome: LegacyGenome, *, output_dir: Path, fmt: str = "svg") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{genome.identifier.replace(':', '_')}_age{genome.age}"
    graph = graphviz.Digraph(format=fmt)
    graph.attr(rankdir="LR", fontsize="10")
    graph.attr("node", shape="circle", fontsize="9", width="0.4", height="0.4", style="filled")
    graph.attr("edge", fontsize="8")

    for node in genome.nodes.values():
        attrs = {
            "label": _build_node_label(node),
            "fillcolor": _node_fill(node),
            "shape": "box" if node.type == "in" else ("doubleoctagon" if node.type == "out" else "ellipse"),
        }
        graph.node(node.key, _attributes=attrs)

    for link in genome.links:
        color, penwidth, label = _edge_attributes(link.weight)
        graph.edge(
            link.source_key,
            link.target_key,
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


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Render a legacy Picbreeder genome (genome.xml) as a Graphviz diagram.")
    parser.add_argument("input", type=Path, help="Path to genome.xml (or a .zip containing it)")
    parser.add_argument("--output-dir", type=Path, default=Path("legacy_diagrams"), help="Directory to write the diagram into")
    parser.add_argument("--format", dest="fmt", default="svg", choices=("svg", "png", "pdf"), help="Graphviz output format")
    args = parser.parse_args(argv)

    genome = load_legacy_genome(args.input)
    rendered = render_legacy_genome(genome, output_dir=args.output_dir, fmt=args.fmt)
    print(f"Rendered legacy genome to {rendered}")


if __name__ == "__main__":
    main()
