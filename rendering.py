import base64
import math
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import graphviz 
import neat
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import numpy as np
from pyparsing import Iterable

from picture2d.common import (
    _canvas_coords,
    eval_genome_as_grayscale_and_color,
)


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


def _text_size(text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = font.getbbox(text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height


def create_captioned_grid(
    images: Sequence[Image.Image],
    captions: Sequence[str],
    thumb_size: int,
    margin: int,
    font_size: int,
) -> Image.Image:
    font = try_load_font(font_size)
    if not images:
        raise ValueError("No images to render in grid.")

    cols = max(1, int(math.ceil(math.sqrt(len(images)))))
    rows = int(math.ceil(len(images) / cols))
    label_height = max(_text_size(caption, font)[1] for caption in captions) if captions else 0

    cell_height = thumb_size + label_height + 6
    width = (cols * thumb_size) + ((cols + 1) * margin)
    height = (rows * cell_height) + ((rows + 1) * margin)
    canvas = Image.new("RGB", (width, height), (16, 16, 20))
    draw = ImageDraw.Draw(canvas)

    for i, (img, caption) in enumerate(zip(images, captions)):
        row = i // cols
        col = i % cols
        x = margin + col * (thumb_size + margin)
        y = margin + row * (cell_height + margin)

        if img and img.size != (thumb_size, thumb_size):
            img = img.resize((thumb_size, thumb_size), resample=Image.BICUBIC)
        
        if img:
            canvas.paste(img, (x, y))

        text_w, text_h = _text_size(caption, font)
        text_x = x + max(0, (thumb_size - text_w) // 2)
        text_y = y + thumb_size + 2
        draw.text((text_x, text_y), caption, font=font, fill=(255, 255, 0))

    return canvas


def create_numbered_grid(
    population_images: List[Image.Image],
    rows: int,
    cols: int,
    thumb_size: int,
    margin: int = 12,
    font_size: int = 22,
    selected: Optional[Sequence[int]] = None,
) -> Image.Image:
    font = try_load_font(font_size)
    selected_set = set(int(s) for s in selected) if selected else set()

    width = (cols * thumb_size) + ((cols + 1) * margin)
    height = (rows * thumb_size) + ((rows + 1) * margin)
    canvas = Image.new("RGBA", (width, height), (16, 16, 20, 255))
    draw = ImageDraw.Draw(canvas)

    for i, image in enumerate(population_images):
        row = i // cols
        col = i % cols
        x = margin + col * (thumb_size + margin)
        y = margin + row * (thumb_size + margin)
        canvas.paste(image, (x, y))

        label = str(i)
        draw_label(draw, (x + 6, y + 6), label, font)

        if i in selected_set:
            draw.rectangle(
                (x, y, x + thumb_size, y + thumb_size),
                outline=(255, 0, 0),
                width=4,
            )

    return canvas.convert("RGB")


def render_genome_image(
    genome: neat.DefaultGenome,
    config: neat.Config,
    width: int,
    height: int,
) -> Image.Image:
    gray_image_data, color_image_data = eval_genome_as_grayscale_and_color(genome, config, width, height)
    gray_image = Image.fromarray((np.array(gray_image_data) * 255).astype(np.uint8), mode="L")
    color_image = Image.fromarray(np.array(color_image_data, dtype=np.uint8), mode="RGB")
    return gray_image, color_image

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

    input_keys_ordered = list(config.genome_config.input_keys)
    input_keys = set(input_keys_ordered)
    output_keys = set(config.genome_config.output_keys)

    input_activation_map: Dict[int, str] = {}
    activation_names: Optional[Sequence[str]] = None
    for source in (effective_genome, genome):
        names = getattr(source, "_input_activation_names", None)
        if names:
            activation_names = names
            break
    if activation_names:
        for key, name in zip(input_keys_ordered, activation_names):
            if name:
                input_activation_map[key] = str(name)

    def build_label(
        node_key: int,
        node_type: str,
        node_gene: Optional[Any],
        node_activation: Optional[str] = None,
    ) -> str:
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
        elif node_activation:
            label_lines.append(f"act={node_activation}")
        return "\n".join(label_lines)

    for input_key in input_keys:
        node_id = str(input_key)
        attrs = {
            "style": "filled",
            "shape": "box",
            "fillcolor": resolved_node_colors.get(input_key, "lightgray"),
            "label": build_label(
                input_key,
                "input",
                None,
                node_activation=input_activation_map.get(input_key),
            ),
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
    "create_captioned_grid",
    "create_numbered_grid",
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
