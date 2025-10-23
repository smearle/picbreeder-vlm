from colorsys import hsv_to_rgb as colorsys_hsv_to_rgb
import math

import neat


def hsv_to_rgb(h, s, v):
    """
    Convert HSV to RGB.
    All inputs and outputs should be in the range [0, 1].
    """
    h = h * 360.

    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    r1, g1, b1, c1 = c, x, 0, (0 <= h)*(h<60)
    r2, g2, b2, c2 = x, c, 0, (60 <= h)*(h<120)
    r3, g3, b3, c3 = 0, c, x, (120 <= h)*(h<180)
    r4, g4, b4, c4 = 0, x, c, (180 <= h)*(h<240)
    r5, g5, b5, c5 = x, 0, c, (240 <= h)*(h<300)
    r6, g6, b6, c6 = c, 0, x, (300 <= h)*(h<360)

    r = r1 * c1 + r2 * c2 + r3 * c3 + r4 * c4 + r5 * c5 + r6 * c6
    g = g1 * c1 + g2 * c2 + g3 * c3 + g4 * c4 + g5 * c5 + g6 * c6
    b = b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 + b5 * c5 + b6 * c6
    r, g, b = r + m, g + m, b + m
    return clamp(r, 0., 1.), clamp(g, 0., 1.), clamp(b, 0., 1.)


def clamp(value, lower, upper):
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def restrict_hsb_channels(outputs):
    """Wrap hue and clamp saturation/value to keep CPPN outputs inside HSV bounds."""
    hue = (outputs[0] + 1) % 1
    saturation = clamp(outputs[1], 0.0, 1.0)
    brightness = clamp(abs(outputs[2]), 0.0, 1.0)
    return hue, saturation, brightness


def hsb_to_rgb(outputs):
    hue, saturation, brightness = restrict_hsb_channels(outputs)
    red, green, blue = colorsys_hsv_to_rgb(hue, saturation, brightness)
    # red, green, blue = hsv_to_rgb(hue, saturation, brightness)
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
