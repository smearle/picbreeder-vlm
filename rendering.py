import base64
import math
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

import neat
from pyparsing import Iterable

from picture2d.common import (
    _canvas_coords,
    eval_color_image,
    eval_gray_image,
    eval_mono_image,
)

try:
    import graphviz  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    graphviz = None  # type: ignore[assignment]


def _format_float(value: float) -> str:
    return f"{value:+.5f}"


def decode_image(entry: Dict[str, Any]) -> Image.Image:
    data = base64.b64decode(entry["data"])
    encoding = entry.get("encoding") or "png"
    if encoding == "png":
        try:
            return Image.open(BytesIO(data)).convert("RGB")
        except (UnidentifiedImageError, ValueError):
            pass
    width = int(entry["width"])
    height = int(entry["height"])
    mode = entry.get("mode", "RGB")
    return Image.frombytes(mode, (width, height), data).convert("RGB")


def try_load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, position: Sequence[int], text: str, font: ImageFont.ImageFont) -> None:
    x, y = position
    outline_color = (0, 0, 0)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    draw.text(position, text, font=font, fill=(255, 255, 0))


def create_numbered_grid(
    state: Dict[str, Any],
    margin: int = 12,
    font_size: int = 22,
    selected: Optional[Sequence[int]] = None,
) -> Image.Image:
    rows = int(state["rows"])
    cols = int(state["cols"])
    thumb = int(state["thumbSize"])
    font = try_load_font(font_size)
    selected_set = set(int(s) for s in selected) if selected else set()

    width = (cols * thumb) + ((cols + 1) * margin)
    height = (rows * thumb) + ((rows + 1) * margin)
    canvas = Image.new("RGBA", (width, height), (16, 16, 20, 255))
    draw = ImageDraw.Draw(canvas)

    for entry in state["images"]:
        image = decode_image(entry)
        row = int(entry["row"])
        col = int(entry["col"])
        index = int(entry["index"])

        x = margin + col * (thumb + margin)
        y = margin + row * (thumb + margin)
        canvas.paste(image, (x, y))

        label = str(index)
        draw_label(draw, (x + 6, y + 6), label, font)

        if index in selected_set:
            draw.rectangle(
                (x, y, x + thumb, y + thumb),
                outline=(255, 0, 0),
                width=4,
            )

    return canvas.convert("RGB")


def _resolve_inputs(genome: neat.DefaultGenome, coords: Sequence[float]) -> List[float]:
    inputs = list(coords)
    transformer = getattr(genome, "transform_inputs", None)
    if transformer is not None:
        inputs = transformer(inputs)
    return inputs

def _resolve_outputs(genome: neat.DefaultGenome, outputs: Sequence[float]) -> List[float]:
    transformer = getattr(genome, "transform_outputs", None)
    if transformer is not None:
        return transformer(outputs)
    return list(outputs)


def _legacy_apply_render(value: float) -> float:
    if not math.isfinite(value):
        return 1.0 if value > 0.0 else 0.0
    if value >= 0.0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def _legacy_to_byte(value: float) -> int:
    value = max(0.0, min(1.0, value))
    return int(value * 255.0 + 0.5)


def legacy_eval_mono_image(genome, config, width, height):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    image = []
    for coord_row in _canvas_coords(width, height):
        row = []
        for coords in coord_row:
            inputs = _resolve_inputs(genome, coords)
            raw_output = net.activate(inputs)
            output = _resolve_outputs(genome, raw_output)
            rendered = _legacy_apply_render(output[0])
            row.append(255 if rendered > 0.5 else 0)
        image.append(row)
    return image


def legacy_eval_gray_image(genome, config, width, height):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    image = []
    for coord_row in _canvas_coords(width, height):
        row = []
        for coords in coord_row:
            inputs = _resolve_inputs(genome, coords)
            raw_output = net.activate(inputs)
            output = _resolve_outputs(genome, raw_output)
            rendered = _legacy_apply_render(output[0])
            row.append(_legacy_to_byte(rendered))
        image.append(row)
    return image


def legacy_eval_color_image(genome, config, width, height):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    image = []
    for coord_row in _canvas_coords(width, height):
        row = []
        for coords in coord_row:
            inputs = _resolve_inputs(genome, coords)
            raw_output = net.activate(inputs)
            output = _resolve_outputs(genome, raw_output)
            channels = [_legacy_apply_render(val) for val in output[:3]]
            row.append(tuple(_legacy_to_byte(ch) for ch in channels))
        image.append(row)
    return image


def render_genome_image(
    genome: neat.DefaultGenome,
    config: neat.Config,
    width: int,
    height: int,
    scheme: str,
    palette: str,
) -> Image.Image:
    if scheme == "color" or scheme == "toggle":
        if palette == "sigmoid":
            image_data = legacy_eval_color_image(genome, config, width, height)
        else:
            image_data = eval_color_image(genome, config, width, height)
        mode = "RGB"
    elif scheme == "gray":
        if palette == "sigmoid":
            image_data = legacy_eval_gray_image(genome, config, width, height)
        else:
            image_data = eval_gray_image(genome, config, width, height)
        mode = "L"
    else:
        if palette == "sigmoid":
            image_data = legacy_eval_mono_image(genome, config, width, height)
        else:
            image_data = eval_mono_image(genome, config, width, height)
        mode = "L"

    image = Image.new(mode, (width, height))
    flat_pixels: List[Any] = [pixel for row in image_data for pixel in row]
    image.putdata(flat_pixels)
    if mode != "RGB":
        image = image.convert("RGB")
    return image


def render_genome_diagram(
    genome: neat.DefaultGenome,
    config: neat.Config,
    output_stem: Path,
    show_disabled: bool = True,
    prune_unused: bool = False,
    node_names: Optional[Dict[int, str]] = None,
    node_colors: Optional[Dict[int, str]] = None,
    fmt: str = "svg",
) -> Optional[Path]:
    """Render a network topology diagram, annotating every gene attribute."""

    if graphviz is None:
        return None

    effective_genome = genome
    if prune_unused and hasattr(genome, "get_pruned_copy"):
        effective_genome = genome.get_pruned_copy(config.genome_config)  # type: ignore[attr-defined]

    resolved_node_names = dict(node_names or {})
    resolved_node_colors = dict(node_colors or {})

    node_attr = {
        "shape": "circle",
        "fontsize": "9",
        "height": "0.2",
        "width": "0.2",
    }
    dot = graphviz.Digraph(format=fmt, node_attr=node_attr)

    input_keys = set(config.genome_config.input_keys)
    output_keys = set(config.genome_config.output_keys)

    def build_label(node_key: int, node_type: str, node_gene: Optional[Any]) -> str:
        alias = resolved_node_names.get(node_key, str(node_key))
        label_lines = [alias, f"id={node_key}", f"type={node_type}"]
        if node_gene is not None:
            activation = getattr(node_gene, "activation", "") or "identity"
            aggregation = getattr(node_gene, "aggregation", "") or "sum"
            bias = getattr(node_gene, "bias", 0.0)
            response = getattr(node_gene, "response", 1.0)
            label_lines.extend(
                [
                    f"act={activation}",
                    f"agg={aggregation}",
                    f"bias={_format_float(bias)}",
                    f"resp={_format_float(response)}",
                ]
            )
        return "\n".join(label_lines)

    for input_key in input_keys:
        node_id = str(input_key)
        attrs = {
            "style": "filled",
            "shape": "box",
            "fillcolor": resolved_node_colors.get(input_key, "lightgray"),
            "label": build_label(input_key, "input", None),
        }
        dot.node(node_id, _attributes=attrs)

    for output_key in output_keys:
        node_id = str(output_key)
        node_gene = effective_genome.nodes.get(output_key) if effective_genome else None
        attrs = {
            "style": "filled",
            "fillcolor": resolved_node_colors.get(output_key, "lightblue"),
            "label": build_label(output_key, "output", node_gene),
        }
        dot.node(node_id, _attributes=attrs)

    for node_key, node_gene in effective_genome.nodes.items():
        if node_key in input_keys or node_key in output_keys:
            continue
        node_id = str(node_key)
        attrs = {
            "style": "filled",
            "fillcolor": resolved_node_colors.get(node_key, "white"),
            "label": build_label(node_key, "hidden", node_gene),
        }
        dot.node(node_id, _attributes=attrs)

    for conn in effective_genome.connections.values():
        if not conn.enabled and not show_disabled:
            continue
        weight = getattr(conn, "weight", 0.0)
        src, dst = conn.key
        edge_attrs = {
            "style": "solid" if conn.enabled else "dotted",
            "color": "green" if weight >= 0 else "red",
            "penwidth": f"{0.1 + abs(weight) / 5.0:.3f}",
            "fontsize": "8",
            "label": "\n".join(
                filter(
                    None,
                    [
                        f"w={_format_float(weight)}",
                        None if conn.enabled else "disabled",
                    ],
                )
            ),
        }
        dot.edge(str(src), str(dst), _attributes=edge_attrs)

    if not output_stem.parent.exists():
        output_stem.parent.mkdir(parents=True, exist_ok=True)

    base = output_stem
    if output_stem.suffix:
        base = output_stem.with_suffix("")

    rendered_path = Path(dot.render(str(base), cleanup=True))
    return rendered_path


__all__ = [
    "decode_image",
    "try_load_font",
    "draw_label",
    "create_numbered_grid",
    "legacy_eval_mono_image",
    "legacy_eval_gray_image",
    "legacy_eval_color_image",
    "render_genome_image",
    "render_genome_diagram",
]


def _draw_dotted_rectangle(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[int, int, int, int],
    *,
    color: Tuple[int, int, int],
    width: int,
    dash_length: int = 8,
    gap_length: int = 6,
) -> None:
    """Render a dotted rectangle so publication highlights do not mask red parent outlines."""

    x0, y0, x1, y1 = bbox
    x0, x1 = sorted((int(round(x0)), int(round(x1))))
    y0, y1 = sorted((int(round(y0)), int(round(y1))))

    def while_loop(start: int, end: int, step: int) -> Iterable[int]:
        current = start
        while current <= end:
            yield current
            current += step

    horizontal_step = dash_length + gap_length
    for x in while_loop(x0, x1, horizontal_step):
        draw.line(
            [(x, y0), (min(x + dash_length, x1), y0)],
            fill=color,
            width=width,
        )
        draw.line(
            [(x, y1), (min(x + dash_length, x1), y1)],
            fill=color,
            width=width,
        )

    vertical_step = dash_length + gap_length
    for y in while_loop(y0, y1, vertical_step):
        draw.line(
            [(x0, y), (x0, min(y + dash_length, y1))],
            fill=color,
            width=width,
        )
        draw.line(
            [(x1, y), (x1, min(y + dash_length, y1))],
            fill=color,
            width=width,
        )

