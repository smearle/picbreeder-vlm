"""Shared reveal renderer for archive animations.

Given continuous node positions, a parent map, and a reveal order, render an MP4
where nodes appear in order, with:

  * lineage edges ALWAYS drawn behind the image thumbnails (two-layer composite,
    so a freshly-spawned line never paints over an existing image);
  * an optional dynamic "zoom-out" viewport that always frames exactly the
    revealed sub-archive (a compact square), so the view zooms out as it grows;
  * an orange highlight on the current growth front.

Positions are in arbitrary units; we scale them into a fixed-resolution "world"
canvas, paste once (cheap), and per frame crop the revealed bounding square and
resize it to the output frame -- which is what produces the zoom.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ORANGE = (255, 108, 0)
VIRTUAL_ROOT = "__root__"


def lineage_colors(order, parent, sat=0.62, val=0.95):
    """Map each node to an RGB colour shared by its whole root-lineage.

    Roots (children of the virtual root) each get a distinct hue (golden-ratio
    spacing for separability); descendants inherit their root's colour.
    """
    import colorsys
    root: Dict[Hashable, Hashable] = {}
    for n in order:
        chain = []
        cur = n
        while True:
            if cur in root:
                r = root[cur]
                break
            p = parent.get(cur, VIRTUAL_ROOT)
            if p == VIRTUAL_ROOT:
                r = cur
                break
            chain.append(cur)
            cur = p
        root[r] = r
        root[n] = r
        for c in chain:
            root[c] = r

    roots_in_order = []
    seen = set()
    for n in order:
        rt = root[n]
        if rt not in seen:
            seen.add(rt)
            roots_in_order.append(rt)
    rcol = {}
    for i, rt in enumerate(roots_in_order):
        h = (i * 0.6180339887) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, sat, val)
        rcol[rt] = (int(r * 255), int(g * 255), int(b * 255))
    return {n: rcol[root[n]] for n in order}


def _scale_to_world(positions, world, thumb_px):
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    span = max(maxx - minx, maxy - miny) or 1.0
    inner = world - 2 * thumb_px
    s = inner / span
    ox = (world - (maxx - minx) * s) / 2
    oy = (world - (maxy - miny) * s) / 2
    return {n: (ox + (x - minx) * s, oy + (y - miny) * s) for n, (x, y) in positions.items()}


def _scale_to_rect(positions, worldW, worldH, margin):
    """Anisotropically scale positions to FILL a (worldW x worldH) rectangle.

    Unlike ``_scale_to_world`` (which fits uniformly and centres, preserving the
    layout's aspect), this stretches x and y independently so a roughly-circular
    radial/force layout is recomputed to occupy a portrait/landscape column --
    twopi's disk becomes an ellipse, an sfdp blob fills the frame.
    """
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    spanx = (maxx - minx) or 1.0
    spany = (maxy - miny) or 1.0
    sx = (worldW - 2 * margin) / spanx
    sy = (worldH - 2 * margin) / spany
    return {n: (margin + (x - minx) * sx, margin + (y - miny) * sy)
            for n, (x, y) in positions.items()}


def render_reveal(
    out: Path,
    positions: Dict[Hashable, Tuple[float, float]],
    parent: Dict[Hashable, Hashable],
    order: Sequence[Hashable],
    thumbs: Dict[Hashable, np.ndarray],
    *,
    frame: int = 1000,
    fps: int = 24,
    per_frame: int = 11,
    frames_per_step: int = 1,
    hold: int = 48,
    zoom: bool = True,
    world: int = 2200,
    thumb_px: int = 24,
    edge_shade: int = 205,
    line_w: int = 2,
    pad_frac: float = 0.06,
    min_span_frac: float = 0.16,
    highlight: bool = True,
    edge_color: Optional[Dict[Hashable, Tuple[int, int, int]]] = None,
    hi_thumbs: Optional[Dict[Hashable, np.ndarray]] = None,
    emphasis_seq: Optional[Sequence[Dict[Hashable, float]]] = None,
    emphasis_border: Optional[Tuple[int, int, int]] = (218, 165, 32),
):
    # ``emphasis_seq`` (optional) drives a per-step "enlarge these nodes" overlay:
    # emphasis_seq[k] maps node-id -> desired thumbnail size IN WORLD PIXELS, for
    # the moment just after the (k+1)-th node is revealed. Emphasised nodes are
    # redrawn enlarged on TOP of the frame every step, so they grow when they join
    # the set and snap back to base size when they leave (e.g. tracking a live
    # leaderboard). ``hi_thumbs`` supplies higher-res sources for crisp enlargement.
    # ``frame`` may be an int (square) or a (width, height) tuple (portrait/
    # landscape). Portrait recomputes the layout to FILL the rectangle and
    # disables the square-viewport zoom.
    portrait = isinstance(frame, (tuple, list))
    if portrait:
        fw, fh = int(frame[0]), int(frame[1])
        zoom = False
        if fh >= fw:
            worldH, worldW = world, max(1, int(round(world * fw / fh)))
        else:
            worldW, worldH = world, max(1, int(round(world * fh / fw)))
    else:
        fw = fh = int(frame)
        worldW = worldH = world

    # thumbnails may be variable-sized (e.g. scaled by #children); use the
    # largest as the edge margin so big nodes don't clip at the world border.
    max_th = max(max(t.shape[0], t.shape[1]) for t in thumbs.values())
    if portrait:
        wp = _scale_to_rect(positions, worldW, worldH, max(thumb_px, max_th))
    else:
        wp = _scale_to_world(positions, world, max(thumb_px, max_th))
    edges = np.full((worldH, worldW, 3), 255, np.uint8)
    thumbsL = np.full((worldH, worldW, 3), 255, np.uint8)
    mask = np.zeros((worldH, worldW), bool)

    def ipos(nid):
        x, y = wp[nid]
        return int(round(x)), int(round(y))

    def node_r(nid):  # display radius (world px) of a node's thumbnail
        h, w = thumbs[nid].shape[:2]
        return max(h, w) / 2

    def paste(nid):
        a = thumbs[nid]
        h, w = a.shape[:2]
        x, y = wp[nid]
        x0, y0 = int(round(x - w / 2)), int(round(y - h / 2))
        x1, y1 = x0 + w, y0 + h
        if x0 < 0 or y0 < 0 or x1 > worldW or y1 > worldH:
            return
        thumbsL[y0:y1, x0:x1] = a
        mask[y0:y1, x0:x1] = True

    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{fw}x{fh}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )

    revealed_x: List[float] = []
    revealed_y: List[float] = []
    revealed: set = set()
    n_frames = 0

    def draw_emphasis(f, emphasis, x0, y0, scx, scy):
        # Redraw emphasised nodes enlarged, on top of the frame. ``emphasis`` maps
        # node-id -> target size in WORLD px; we convert to output px. Drawn back
        # to front (smallest last) so the biggest hub never buries its neighbours.
        for nid, wsize in sorted(emphasis.items(), key=lambda kv: -kv[1]):
            if nid not in wp or nid not in revealed:
                continue
            out_sz = max(thumbs[nid].shape[0], int(round(wsize * min(scx, scy))))
            src = (hi_thumbs or thumbs).get(nid)
            if src is None:
                continue
            tile = cv2.resize(src, (out_sz, out_sz), interpolation=cv2.INTER_AREA)
            cx = int(round((wp[nid][0] - x0) * scx))
            cy = int(round((wp[nid][1] - y0) * scy))
            x0t, y0t = cx - out_sz // 2, cy - out_sz // 2
            x1t, y1t = x0t + out_sz, y0t + out_sz
            # clip to frame
            cx0, cy0 = max(0, x0t), max(0, y0t)
            cx1, cy1 = min(fw, x1t), min(fh, y1t)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            f[cy0:cy1, cx0:cx1] = tile[cy0 - y0t:cy1 - y0t, cx0 - x0t:cx1 - x0t]
            if emphasis_border:
                cv2.rectangle(f, (cx0, cy0), (cx1 - 1, cy1 - 1), emphasis_border, 1, cv2.LINE_AA)

    def emit(new_nodes, repeat=1, emphasis=None):
        nonlocal n_frames
        # viewport
        if zoom and revealed_x:
            minx, maxx = min(revealed_x), max(revealed_x)
            miny, maxy = min(revealed_y), max(revealed_y)
            cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
            side = max(maxx - minx, maxy - miny) + 2 * max_th
            side = max(side * (1 + 2 * pad_frac), world * min_span_frac)
            side = min(side, float(world))
            x0 = int(round(min(max(cx - side / 2, 0), world - side)))
            y0 = int(round(min(max(cy - side / 2, 0), world - side)))
            sw = sh = int(round(side))
        else:
            x0 = y0 = 0
            sw, sh = worldW, worldH
        sub_m = mask[y0:y0 + sh, x0:x0 + sw]
        sub = np.where(sub_m[..., None], thumbsL[y0:y0 + sh, x0:x0 + sw],
                       edges[y0:y0 + sh, x0:x0 + sw])
        f = cv2.resize(sub, (fw, fh), interpolation=cv2.INTER_AREA)
        scx, scy = fw / sw, fh / sh
        if emphasis:
            draw_emphasis(f, emphasis, x0, y0, scx, scy)
        if highlight and new_nodes:
            # during the spawn frames, bring this step's edges to the FRONT (over
            # thumbnails); they only live in the behind-layer afterward, so they
            # recede behind the images on the next step.
            for nid in new_nodes:
                par = parent.get(nid, VIRTUAL_ROOT)
                if par == VIRTUAL_ROOT or par not in wp:
                    continue
                col = edge_color.get(nid, (edge_shade,) * 3) if edge_color else (edge_shade,) * 3
                p1 = (int((wp[par][0] - x0) * scx), int((wp[par][1] - y0) * scy))
                p2 = (int((wp[nid][0] - x0) * scx), int((wp[nid][1] - y0) * scy))
                cv2.line(f, p1, p2, col, line_w + 1, cv2.LINE_AA)
            # outline ONLY the freshly-spawned child (not its parent)
            for nid in new_nodes:
                if nid not in wp:
                    continue
                wx, wy = wp[nid]
                rr = max(2, int(node_r(nid) * 1.35 * min(scx, scy)))
                cv2.circle(f, (int((wx - x0) * scx), int((wy - y0) * scy)), rr, ORANGE, 1, cv2.LINE_AA)
        buf = np.ascontiguousarray(f).tobytes()
        for _ in range(repeat):
            proc.stdin.write(buf)
        n_frames += repeat

    order = list(order)
    i = 0
    while i < len(order):
        batch = order[i:i + per_frame]
        for nid in batch:
            par = parent.get(nid, VIRTUAL_ROOT)
            if par != VIRTUAL_ROOT and par in wp:
                col = edge_color.get(nid, (edge_shade,) * 3) if edge_color else (edge_shade,) * 3
                cv2.line(edges, ipos(par), ipos(nid), col, line_w, cv2.LINE_AA)
        for nid in batch:
            paste(nid)
            revealed.add(nid)
            wx, wy = wp[nid]
            revealed_x.append(wx)
            revealed_y.append(wy)
        last_idx = i + len(batch) - 1
        emph = emphasis_seq[last_idx] if emphasis_seq else None
        emit(batch, repeat=frames_per_step, emphasis=emph)
        i += per_frame

    emit([], repeat=hold, emphasis=(emphasis_seq[-1] if emphasis_seq else None))
    proc.stdin.close()
    proc.wait()
    print(f"wrote {out} ({n_frames} frames, {n_frames / fps:.1f}s, zoom={zoom})")
