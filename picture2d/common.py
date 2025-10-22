import colorsys
import math

import neat

def clamp(value, lower, upper):
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def restrict_hsb_channels(outputs):
    """Mirror CPPNArtEvolution's Full/Half Linear Piecewise range restriction."""
    hue_raw = clamp(outputs[0], -1.0, 1.0)
    saturation = clamp(outputs[1], 0.0, 1.0)
    brightness = abs(clamp(outputs[2], -1.0, 1.0))

    # Java's Color.HSBColor normalizes hue by subtracting Math.floor(hue).
    hue = hue_raw - math.floor(hue_raw)
    return hue, saturation, brightness


def hsb_to_rgb(outputs):
    hue, saturation, brightness = restrict_hsb_channels(outputs)
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, brightness)
    return (
        _to_byte(red),
        _to_byte(green),
        _to_byte(blue),
    )


def _canvas_coords(width, height):
    if width < 1 or height < 1:
        return

    def _scale(value, max_dimension):
        if max_dimension <= 1:
            return 0.0
        return (value / (max_dimension - 1)) * 2.0 - 1.0

    for y in range(height):
        scaled_y = _scale(y, height)
        row = []
        for x in range(width):
            scaled_x = _scale(x, width)
            radius = math.hypot(scaled_x, scaled_y) * math.sqrt(2.0)
            row.append((scaled_x, scaled_y, radius, 1.0))
        yield row


def _to_byte(value):
    value = clamp(value, 0.0, 1.0)
    return int(value * 255.0 + 0.5)


def eval_mono_image(genome, config, width, height):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    image = []
    for coord_row in _canvas_coords(width, height):
        row = []
        for coords in coord_row:
            inputs = list(coords)
            transformer = getattr(genome, "transform_inputs", None)
            if transformer is not None:
                inputs = transformer(inputs)
            output = net.activate(inputs)
            output_transformer = getattr(genome, "transform_outputs", None)
            if output_transformer is not None:
                output = output_transformer(output)
            brightness = abs(clamp(output[0], -1.0, 1.0))
            gray = 255 if brightness > 0.5 else 0
            row.append(gray)
        image.append(row)

    return image


def eval_gray_image(genome, config, width, height):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    image = []
    for coord_row in _canvas_coords(width, height):
        row = []
        for coords in coord_row:
            inputs = list(coords)
            transformer = getattr(genome, "transform_inputs", None)
            if transformer is not None:
                inputs = transformer(inputs)
            output = net.activate(inputs)
            output_transformer = getattr(genome, "transform_outputs", None)
            if output_transformer is not None:
                output = output_transformer(output)
            brightness = abs(clamp(output[0], -1.0, 1.0))
            row.append(_to_byte(brightness))
        image.append(row)

    return image


def eval_color_image(genome, config, width, height):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    image = []
    for coord_row in _canvas_coords(width, height):
        row = []
        for coords in coord_row:
            inputs = list(coords)
            transformer = getattr(genome, "transform_inputs", None)
            if transformer is not None:
                inputs = transformer(inputs)
            output = net.activate(inputs)
            output_transformer = getattr(genome, "transform_outputs", None)
            if output_transformer is not None:
                output = output_transformer(output)
            rgb = hsb_to_rgb(output[:3])
            row.append(rgb)
        image.append(row)

    return image
