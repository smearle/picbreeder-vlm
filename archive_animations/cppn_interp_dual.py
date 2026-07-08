#!/usr/bin/env python3
"""EXPERIMENTAL: morph panel + live CPPN panel, side by side.

Left: the image morphing along the published genome's lineage (same as
``cppn_interp.py``). Right, in an equal square: the underlying CPPN, animated
in lockstep --

  * edges encode magnitude via **width** (blue = negative, red = positive;
    |weight| -> width). **Opacity is reserved for structural state**: alpha=1
    for an enabled edge, and only ramps during enable/disable transitions
    (so an edge that's appearing/disappearing fades in/out at its native
    width, while a weight that simply shrinks toward 0 gets thinner but
    stays fully opaque);
  * each node draws a little curve of its **activation function**, which morphs
    by blending the parent and child function ((1-t)*f_a + t*f_b).

Node positions are a single global layout (graphviz ``dot`` on the union of all
lineage genomes) so the graph never jumps between segments.

Usage:
    python archive_animations/cppn_interp_dual.py \
        --agent-dir archive_animations/_work/agent_000 \
        --out archive_animations/out/cppn_interp_dual_agent000.mp4 \
        --panel 400 --fps 24
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import cppn_interp as ci
from picbreeder_vlm.viz.render_lineage_animation import build_superset, render_frames

POS = np.array([214, 69, 38], float)    # positive weight -> red
NEG = np.array([38, 107, 214], float)   # negative weight -> blue
WHITE = np.array([255, 255, 255], float)


# ---------- global layout via graphviz dot ----------
def _sample_bspline(ctrl, samples_per_segment=14):
    """Graphviz -Tplain emits edges as cubic B-spline control polygons:
    n = 3k + 1 control points, k cubic Bezier segments sharing endpoints.
    Sample each segment into a polyline."""
    if len(ctrl) < 4:
        return list(ctrl)
    out = [ctrl[0]]
    n_segments = (len(ctrl) - 1) // 3
    for seg in range(n_segments):
        p0, p1, p2, p3 = ctrl[3 * seg : 3 * seg + 4]
        for i in range(1, samples_per_segment + 1):
            t = i / samples_per_segment
            mt = 1 - t
            x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
            y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
            out.append((x, y))
    return out


def master_layout(genomes, input_keys, output_keys, panel, margin):
    """Returns (pos, edge_polys). edge_polys maps (a, b) -> polyline (Nx2) in
    panel coordinates, sampled from graphviz's spline routing on the union
    graph. Edges curve around nodes since dot's default edge router avoids
    obstacles."""
    nodes, edges = set(), set()
    for g in genomes:
        nodes.update(g.nodes.keys())
        edges.update(g.connections.keys())
    nodes.update(input_keys)
    nodes.update(output_keys)
    lines = [
        "digraph G {",
        "  rankdir=LR; splines=true; overlap=false;",
        "  node [shape=circle, width=0.2];",
    ]
    lines += [f'  "{k}";' for k in nodes]
    lines += [f'  "{a}" -> "{b}";' for (a, b) in edges]
    # pin inputs to the source rank, outputs to the sink rank
    lines.append("  { rank=source; " + " ".join(f'"{k}";' for k in input_keys) + " }")
    lines.append("  { rank=sink; " + " ".join(f'"{k}";' for k in output_keys) + " }")
    lines.append("}")
    out = subprocess.run(["dot", "-Tplain"], input="\n".join(lines).encode(),
                         capture_output=True).stdout.decode()
    pos = {}
    raw_splines = {}  # (a, b) -> list[(x, y)] in graphviz coords
    for line in out.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "node":
            pos[int(p[1].strip('"'))] = (float(p[2]), float(p[3]))
        elif p[0] == "edge":
            a = int(p[1].strip('"'))
            b = int(p[2].strip('"'))
            n = int(p[3])
            pts = [(float(p[4 + 2 * i]), float(p[5 + 2 * i])) for i in range(n)]
            raw_splines[(a, b)] = pts
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    span = max(maxx - minx, maxy - miny) or 1.0
    s = (panel - 2 * margin) / span
    ox = (panel - (maxx - minx) * s) / 2
    oy = (panel - (maxy - miny) * s) / 2

    def tx(x, y):  # graphviz (y-up) -> panel (y-down)
        return (ox + (x - minx) * s, panel - (oy + (y - miny) * s))

    pos_t = {k: tx(x, y) for k, (x, y) in pos.items()}
    edge_polys = {}
    for (a, b), ctrl in raw_splines.items():
        sampled = _sample_bspline([tx(x, y) for x, y in ctrl])
        edge_polys[(a, b)] = np.array(sampled, dtype=np.float32)
    return pos_t, edge_polys


# ---------- activation curves ----------
def act_samples(fn, xs, yclip):
    ys = np.empty_like(xs)
    for i, x in enumerate(xs):
        try:
            ys[i] = fn(float(x))
        except Exception:
            ys[i] = 0.0
    return np.clip(ys, -yclip, yclip)


def draw_network(panel, pos, edge_polys, node_glyph_g, edges, node_curves,
                 input_keys, output_keys, wref, minw, maxw, xs, yclip):
    """edges: list of (a, b, weight, visible_alpha). node_curves: {key: ys array}.
    edge_polys: {(a, b): Nx2 polyline} from master_layout (graphviz splines)."""
    canvas = np.full((panel, panel, 3), 255, np.uint8)

    def P(k):
        return (int(round(pos[k][0])), int(round(pos[k][1])))

    for a, b, w, alpha in edges:
        if alpha <= 0.02 or a not in pos or b not in pos:
            continue
        col = POS if w >= 0 else NEG
        eff = (WHITE * (1 - alpha) + col * alpha).astype(np.uint8)
        width = int(round(minw + min(abs(w) / wref, 1.0) * (maxw - minw)))
        poly = edge_polys.get((a, b))
        if poly is not None and len(poly) >= 2:
            cv2.polylines(canvas, [poly.round().astype(np.int32)], False,
                          tuple(int(c) for c in eff), max(1, width), cv2.LINE_AA)
        else:
            cv2.line(canvas, P(a), P(b), tuple(int(c) for c in eff),
                     max(1, width), cv2.LINE_AA)

    g = node_glyph_g
    half = g // 2
    for k, ys in node_curves.items():
        if k not in pos:
            continue
        cx, cy = P(k)
        x0, y0 = cx - half, cy - half
        # glyph background + border colour by node type
        cv2.rectangle(canvas, (x0, y0), (x0 + g, y0 + g), (255, 255, 255), -1)
        if k in input_keys:
            # darker than the original (150,150,150) — at 1px LINE_AA against
            # white, light gray was so faint the glyph read as a borderless
            # white blob and edges appeared to vanish into nothing.
            border = (90, 90, 90)
        elif k in output_keys:
            border = (214, 108, 0)
        else:
            border = (40, 40, 40)
        cv2.rectangle(canvas, (x0, y0), (x0 + g, y0 + g), border, 1, cv2.LINE_AA)
        # zero axis
        zy = int(round(y0 + g * (0.5 + 0.0)))
        cv2.line(canvas, (x0 + 2, zy), (x0 + g - 2, zy), (225, 225, 225), 1)
        # activation curve
        pts = []
        for i in range(len(xs)):
            px = x0 + 3 + int((g - 6) * i / (len(xs) - 1))
            py = int(round(y0 + g * (0.5 - 0.46 * ys[i] / yclip)))
            pts.append((px, py))
        cv2.polylines(canvas, [np.array(pts, np.int32)], False, border, 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--panel", type=int, default=400, help="px per square panel (output)")
    ap.add_argument("--img-res", type=int, default=160,
                    help="internal CPPN render resolution (upscaled to --panel); keeps it fast")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--min-steps", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--steps-per-unit", type=float, default=2.0)
    ap.add_argument("--hold-keyframe", type=int, default=6)
    ap.add_argument("--hold-end", type=int, default=36)
    ap.add_argument("--color", choices=["auto", "color", "gray"], default="auto")
    args = ap.parse_args()

    config = ci.build_config()
    gcfg = config.genome_config
    defs = gcfg.activation_defs
    input_keys = set(gcfg.input_keys)
    output_keys = set(gcfg.output_keys)

    chain = ci.load_lineage_chain(args.agent_dir)
    chain = ci.dedupe_keyframes(chain)
    if len(chain) < 2:
        raise SystemExit(f"need >=2 keyframes, got {len(chain)}")
    color_enabled = ci.detect_color(args.agent_dir) if args.color == "auto" else (args.color == "color")
    variant = "auto" if args.color == "auto" else args.color
    genomes = [g for _, g in chain]
    print(f"lineage: {len(chain)} keyframes (keys {[k for k, _ in chain]})")

    P = args.panel
    pos, edge_polys = master_layout(genomes, input_keys, output_keys, P, margin=int(P * 0.09))
    # weight reference for width scaling (|w|/wref clamps the width ramp)
    allw = [abs(c.weight) for g in genomes for c in g.connections.values()]
    wref = max(1.5, float(np.percentile(allw, 90))) if allw else 1.5
    minw, maxw = 1.0, max(2.0, P / 70)
    glyph = max(20, int(P / 9))
    xs = np.linspace(-2.5, 2.5, 24)
    yclip = 1.6

    # canonical (keyframe) edge/curve sets
    def canon_edges(genome):
        # alpha is structural: 1.0 if the edge is enabled, 0.0 if not.
        return [(a, b, c.weight, 1.0 if c.enabled else 0.0)
                for (a, b), c in genome.connections.items()]

    def canon_curves(genome):
        out = {}
        for k, n in genome.nodes.items():
            fn = defs.get(getattr(n, "activation", "identity")) or (lambda v: v)
            out[k] = act_samples(fn, xs, yclip)
        return out

    def net_canon(genome):
        return draw_network(P, pos, edge_polys, glyph, canon_edges(genome), canon_curves(genome),
                            input_keys, output_keys, wref, minw, maxw, xs, yclip)

    # image renders for keyframes (canonical brackets) at low res; upscaled at emit
    R = args.img_res
    canon_img = [ci.canon_frame(g, config, R, variant, color_enabled) for g in genomes]
    canon_net = [net_canon(g) for g in genomes]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    W = 2 * P
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{P}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out)],
        stdin=subprocess.PIPE,
    )

    def emit(img_rgb, net_rgb, repeat=1):
        img = cv2.resize(np.asarray(img_rgb.convert("RGB")), (P, P), interpolation=cv2.INTER_NEAREST)
        frame = np.concatenate([img, net_rgb], axis=1)
        buf = np.ascontiguousarray(frame).tobytes()
        for _ in range(repeat):
            proc.stdin.write(buf)

    emit(canon_img[0], canon_net[0], args.hold_keyframe)
    total = args.hold_keyframe
    for i in range(len(genomes) - 1):
        parent, child = genomes[i], genomes[i + 1]
        dist = ci.visual_distance(canon_img[i], canon_img[i + 1])
        steps = int(min(args.max_steps, max(args.min_steps,
                        round(args.min_steps + args.steps_per_unit * dist))))
        superset, node_pairs, conn_pairs, out_stats = build_superset(parent, child)
        imgs = render_frames(superset, node_pairs, conn_pairs, config, steps=steps,
                             width=R, height=R, variant_mode=variant,
                             color_enabled=color_enabled, output_activation_stats=out_stats)
        # precompute blended activation curves per node across this segment
        for idx in range(1, steps - 1):  # interior frames only
            t = idx / (steps - 1)
            edges = []
            for (a, b), st in conn_pairs.items():
                if st.start_enabled and st.end_enabled:
                    # both keyframes have the edge enabled: lerp weight, full alpha
                    w = (1 - t) * st.start_weight + t * st.end_weight
                    alpha = 1.0
                elif st.start_enabled and not st.end_enabled:
                    # edge is being disabled: hold weight, fade alpha out
                    w = st.start_weight
                    alpha = 1.0 - t
                elif not st.start_enabled and st.end_enabled:
                    # edge is being enabled: hold weight, fade alpha in
                    w = st.end_weight
                    alpha = t
                else:
                    # absent from both: nothing to draw
                    w, alpha = 0.0, 0.0
                edges.append((a, b, w, alpha))
            curves = {}
            for k, (sa, ea) in node_pairs.items():
                if sa == ea:
                    fn = defs.get(sa) or (lambda v: v)
                    curves[k] = act_samples(fn, xs, yclip)
                else:
                    fa, fb = defs.get(sa), defs.get(ea)
                    ya = act_samples(fa or (lambda v: v), xs, yclip)
                    yb = act_samples(fb or (lambda v: v), xs, yclip)
                    curves[k] = np.clip((1 - t) * ya + t * yb, -yclip, yclip)
            net = draw_network(P, pos, edge_polys, glyph, edges, curves, input_keys, output_keys,
                               wref, minw, maxw, xs, yclip)
            emit(imgs[idx], net)
            total += 1
        emit(canon_img[i + 1], canon_net[i + 1], 1 + args.hold_keyframe)
        total += 1 + args.hold_keyframe
        print(f"  {[k for k,_ in chain][i]} -> {[k for k,_ in chain][i+1]}: dist={dist:.1f} steps={steps} (total {total})")

    emit(canon_img[-1], canon_net[-1], args.hold_end)
    total += args.hold_end
    proc.stdin.close()
    proc.wait()
    print(f"wrote {args.out} ({total} frames, {total/args.fps:.1f}s, {W}x{P})")


if __name__ == "__main__":
    main()
