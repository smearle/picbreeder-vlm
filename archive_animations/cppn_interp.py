#!/usr/bin/env python3
"""Animate a Picbreeder-VLM *session* by morphing along the parent->child
lineage of the published image.

We read each genome's recorded ``parents`` and walk the first-parent chain back
from the published genome to the session root, so every consecutive pair is a
real one-mutation parent->child step and the morph is coherent (e.g. ring ->
ring -> bowl -> smile -> fish).

We reuse the topology-matching interpolation from ``render_lineage_animation``
(``build_superset`` + ``render_frames``) and build the NEAT config from
``neat_components`` directly (avoiding the heavy orchestrator import).

We also (1) dedupe identical consecutive genomes into keyframes (the agent often
re-picks an exact copy for several generations), (2) render each keyframe
canonically as the exact bracket between segments to guarantee continuity, and
(3) allocate interpolation frames proportional to the actual visual change between
keyframes (``--min-steps``..``--max-steps``) so a big step gets more frames than
a tiny tweak.

Usage:
    python archive_animations/cppn_interp.py \
        --agent-dir archive_animations/_work/agent_000 \
        --out archive_animations/out/cppn_interp_agent000.mp4 \
        --size 256 --fps 24 --min-steps 8 --max-steps 60 --steps-per-unit 2.0
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import subprocess
from pathlib import Path

import sys

import neat
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for project modules

from picbreeder_vlm._paths import NEAT_CONFIG_PATH
from picbreeder_vlm.core.neat_components import PicbreederGenome, apply_picbreeder_config_defaults, InteractiveStagnation
from picbreeder_vlm.core.picbreeder_reproduction import PicbreederReproduction
from picbreeder_vlm.viz.render_lineage_animation import build_superset, render_frames, pick_variant
from picbreeder_vlm.core.rendering import render_genome_image


def build_config(config_path: str = str(NEAT_CONFIG_PATH)) -> neat.Config:
    config = neat.Config(
        PicbreederGenome, PicbreederReproduction, neat.DefaultSpeciesSet,
        InteractiveStagnation, config_path,
    )
    apply_picbreeder_config_defaults(
        config, enable_output_activations=True, enable_input_activations=False, enable_crossover=True,
    )
    config.pop_size = 15
    return config


def _published_key(agent_dir: Path):
    """genome_key of the published image (last publication), or None."""
    pub = agent_dir / "logs" / "publication_history.jsonl"
    key = None
    if pub.exists():
        for line in pub.read_text().splitlines():
            if line.strip():
                try:
                    key = json.loads(line).get("genome_key", key)
                except Exception:
                    pass
    return key


def load_lineage_chain(agent_dir: Path):
    """Reconstruct the parent->child lineage of the published genome.

    Each selected entry records its genome's ``parents`` (genome keys). We follow
    the first parent from the published genome back to the session root, so every
    consecutive pair is a real parent->child (one mutation).

    Returns [(genome_key, genome), ...] from root to published.
    """
    import re
    pops = agent_dir / "populations"

    def gen_of(p):
        m = re.search(r"selected_gen_(\d+)", p.name)
        return int(m.group(1)) if m else -1

    genome_obj, parent0, first_gen, gen_primary = {}, {}, {}, {}
    for f in sorted(pops.glob("selected_gen_*.pkl.gz"), key=gen_of):
        obj = pickle.load(gzip.open(f, "rb"))
        gen = int(obj["generation"])
        sel = obj.get("selected") or []
        if sel:
            gen_primary[gen] = sel[0]["genome_key"]
        for d in sel:
            k = d["genome_key"]
            genome_obj.setdefault(k, d["genome"])
            first_gen.setdefault(k, gen)
            ps = d.get("parents") or []
            if k not in parent0 or (parent0[k] is None and ps):
                parent0[k] = ps[0] if ps else None

    pub = _published_key(agent_dir)
    if pub is None or pub not in genome_obj:
        # fall back: deepest-generation primary selection
        pub = gen_primary[max(gen_primary)]

    path, seen = [pub], {pub}
    cur = pub
    while True:
        par = parent0.get(cur)
        if par is None:  # data gap (e.g. published genome) -> prev gen's primary pick
            g = first_gen.get(cur)
            par = gen_primary.get(g - 1) if g is not None else None
        if par is None or par in seen or par not in genome_obj:
            break
        path.append(par)
        seen.add(par)
        cur = par
    path.reverse()
    return [(k, genome_obj[k]) for k in path]


def genome_signature(g):
    """Hashable fingerprint to detect consecutive identical selections (holds)."""
    nodes = tuple(sorted((k, round(getattr(n, "bias", 0.0), 6), getattr(n, "activation", ""))
                         for k, n in g.nodes.items()))
    conns = tuple(sorted((k, round(c.weight, 6), bool(c.enabled)) for k, c in g.connections.items()))
    return (nodes, conns)


def dedupe_keyframes(chain):
    """Collapse consecutive identical genomes into keyframes [(gen, genome), ...]."""
    out = []
    last_sig = None
    for gen, g in chain:
        sig = genome_signature(g)
        if sig != last_sig:
            out.append((gen, g))
            last_sig = sig
    return out


def canon_frame(genome, config, size, variant, color_enabled):
    gray, color = render_genome_image(genome, config, size, size)
    return pick_variant(gray, color, variant, color_enabled).convert("RGB")


def visual_distance(a, b):
    return float(np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).mean())


def detect_color(agent_dir: Path) -> bool:
    pub = agent_dir / "logs" / "publication_history.jsonl"
    if pub.exists():
        import json
        for line in pub.read_text().splitlines():
            if line.strip():
                try:
                    return bool(json.loads(line).get("color_enabled", False))
                except Exception:
                    pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-steps", type=int, default=8, help="interp frames for a tiny change")
    ap.add_argument("--max-steps", type=int, default=60, help="interp frames for a big change")
    ap.add_argument("--steps-per-unit", type=float, default=2.0,
                    help="extra frames per unit of mean per-pixel change (0-255)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--hold-keyframe", type=int, default=6, help="frames to pause on each keyframe")
    ap.add_argument("--hold-end", type=int, default=36, help="frames to hold on final image")
    ap.add_argument("--no-dedupe", action="store_true", help="keep identical consecutive selections")
    ap.add_argument("--color", choices=["auto", "color", "gray"], default="auto")
    args = ap.parse_args()

    config = build_config()
    chain = load_lineage_chain(args.agent_dir)
    if not args.no_dedupe:
        chain = dedupe_keyframes(chain)
    if len(chain) < 2:
        raise SystemExit(f"need >=2 distinct genomes, got {len(chain)}")
    color_enabled = detect_color(args.agent_dir) if args.color == "auto" else (args.color == "color")
    variant = "auto" if args.color == "auto" else args.color
    print(f"lineage: {len(chain)} keyframes (keys {[k for k, _ in chain]}), "
          f"color_enabled={color_enabled}, size={args.size}")

    # Render each keyframe canonically; these are the exact brackets that
    # guarantee continuity. Interpolation only fills the *interior*.
    canon = [canon_frame(g, config, args.size, variant, color_enabled) for _, g in chain]

    all_frames = [canon[0]] * args.hold_keyframe
    for i, ((g0, parent), (g1, child)) in enumerate(zip(chain[:-1], chain[1:])):
        dist = visual_distance(canon[i], canon[i + 1])
        steps = int(min(args.max_steps, max(args.min_steps, round(args.min_steps + args.steps_per_unit * dist))))
        superset, node_pairs, conn_pairs, out_stats = build_superset(parent, child)
        frames = render_frames(
            superset, node_pairs, conn_pairs, config,
            steps=steps, width=args.size, height=args.size,
            variant_mode=variant, color_enabled=color_enabled,
            output_activation_stats=out_stats,
        )
        interior = frames[1:-1]                      # exclusive endpoints
        all_frames.extend(interior)
        all_frames.append(canon[i + 1])             # canonical bracket = continuity
        all_frames.extend([canon[i + 1]] * args.hold_keyframe)
        print(f"  gen {g0:>2} -> {g1:<2}: dist={dist:5.1f} steps={steps:>2} (+{len(interior)+1+args.hold_keyframe}, total {len(all_frames)})")

    all_frames.extend([all_frames[-1]] * args.hold_end)

    W = H = args.size
    W += W % 2; H += H % 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", str(args.fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(args.out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in all_frames:
        arr = np.asarray(fr.convert("RGB"))
        if arr.shape[0] != H or arr.shape[1] != W:
            canvas = np.full((H, W, 3), 255, np.uint8)
            canvas[: arr.shape[0], : arr.shape[1]] = arr[:H, :W]
            arr = canvas
        proc.stdin.write(arr.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"wrote {args.out} ({len(all_frames)} frames, {len(all_frames)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
