#!/usr/bin/env python3
"""Batch-render walker grid + synced phylogeny-trails companions for exemplar
runs across hyperparameter settings, and build a browsable HTML index.

For each selected run this produces three clips in the output dir:
  <name>_grid.mp4      -- 6x6 CPPN morph grid (walker_grid.py)
  <name>_trails.mp4    -- synced phylogeny traversal (walker_trails.py)
  <name>_combined.mp4  -- the two hstacked (trails | grid)

The grid and trails use IDENTICAL walk parameters (k, K, alpha, beta, gamma,
max_n, require_genome), and best-first walks are deterministic, so the two
clips are frame-synced and the combined view lines up.

By default it renders a curated set of "exemplar" settings (paper default
first), choosing the richest available seed (most on-disk genomes) for each.
You can instead discover runs by name filters with --include / --exclude, or
pass explicit run dirs with --run.

Examples:
    # curated exemplars, quick preset, paper default first
    .venv/bin/python archive_animations/walker_grid_sweep.py --preset quick

    # just the paper-default setting at full quality
    .venv/bin/python archive_animations/walker_grid_sweep.py \\
        --include th1_ag20_model-gemini-2.5-pro --exclude randp temp ts224 traits goal \\
        --preset full

    # rebuild the index only
    .venv/bin/python archive_animations/walker_grid_sweep.py --index-only
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
VENV_PY = REPO / ".venv" / "bin" / "python"
HERE = Path(__file__).resolve().parent
SWEEP_ROOT = REPO / "sweep_logs"
GRID = HERE / "walker_grid.py"
TRAILS = HERE / "walker_trails.py"

PRESETS = {
    # max_n, K, cell, frames_per_edge, hold, fps, jobs
    "quick": dict(max_n=300, K=40, cell=64, frames_per_edge=8, hold=24, fps=24, jobs=8),
    "full":  dict(max_n=600, K=80, cell=96, frames_per_edge=12, hold=24, fps=24, jobs=8),
}

# Curated exemplar settings (seed-stripped prefixes), paper default FIRST.
# Each is matched against run names; the richest seed (most genomes) is chosen.
EXEMPLARS: List[Tuple[str, str]] = [
    ("paper default (th1, gemini-2.5-pro)",
     "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh"),
    ("random baseline (no VLM, randp2)",
     "ag20_tb-1_scheme-toggle_randp2_rmode-all_nopersonalities_fixed-sesh"),
    ("full context (th-1, gemini-2.5-pro)",
     "th-1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh"),
    ("no context (th0, gemini-2.5-pro)",
     "th0_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh"),
    ("high exploration (th1, randp1)",
     "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_randp1_rmode-all_nopersonalities_fixed-sesh"),
    ("open model (th1, qwen3-vl-30b)",
     "th1_ag20_model-qwen3-vl-30b-fp8_tb-1_scheme-toggle_nopersonalities_fixed-sesh"),
    ("gemini-3-pro-preview (th1)",
     "th1_ag20_model-gemini-3-pro-preview_tb-1_scheme-toggle_nopersonalities_fixed-sesh"),
]


def n_genomes(run: Path) -> int:
    d = run / "archive" / "genomes"
    return len(os.listdir(d)) if d.is_dir() else 0


def richest_seed(prefix: str) -> Optional[Tuple[str, Path, int]]:
    """Among runs whose seed-stripped name == prefix, return the one with the
    most on-disk genomes."""
    best = None
    for meta in SWEEP_ROOT.glob("**/archive/archive_metadata.json"):
        run = meta.parent.parent
        if re.sub(r"_s\d+$", "", run.name) != prefix:
            continue
        n = n_genomes(run)
        if best is None or n > best[2]:
            best = (run.name, run, n)
    return best


def discover(includes, excludes) -> List[Tuple[str, str, Path, int]]:
    """Return [(label, name, run, n_genomes)] for runs matching filters,
    one richest seed per seed-stripped setting."""
    by_setting: Dict[str, Tuple[str, Path, int]] = {}
    for meta in SWEEP_ROOT.glob("**/archive/archive_metadata.json"):
        run = meta.parent.parent
        name = run.name
        if includes and not all(t in name for t in includes):
            continue
        if any(t in name for t in excludes):
            continue
        setting = re.sub(r"_s\d+$", "", name)
        n = n_genomes(run)
        if setting not in by_setting or n > by_setting[setting][2]:
            by_setting[setting] = (name, run, n)
    out = []
    for setting, (name, run, n) in sorted(by_setting.items()):
        out.append((setting, name, run, n))
    return out


def probe_dims(path: Path) -> Optional[Tuple[int, int]]:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
            capture_output=True, text=True, check=True)
        w, h = r.stdout.strip().split("x")
        return int(w), int(h)
    except Exception:
        return None


def render_one(name: str, run: Path, out_dir: Path, p: dict, overwrite: bool,
               theme: str = "dark", stack: str = "vstack",
               refresh_trails: bool = False) -> Dict[str, object]:
    """Render grid + trails + combined for one run.

    refresh_trails: reuse an existing grid (don't re-render the expensive CPPN
    morphs); force re-render of the cheap trails + combined. Useful for changing
    the trail theme / stacking without recomputing morphs.
    """
    grid_out = out_dir / f"{name}_grid.mp4"
    trails_out = out_dir / f"{name}_trails.mp4"
    combined_out = out_dir / f"{name}_combined.mp4"
    info: Dict[str, object] = {"name": name, "ok": False}

    # 1) grid -----------------------------------------------------------------
    if not refresh_trails and (overwrite or not grid_out.exists()):
        cmd = [str(VENV_PY), str(GRID), "--run", str(run), "--out", str(grid_out),
               "--max-n", str(p["max_n"]), "-k", "36", "-K", str(p["K"]),
               "--cell", str(p["cell"]), "--frames-per-edge", str(p["frames_per_edge"]),
               "--hold", str(p["hold"]), "--fps", str(p["fps"]), "--jobs", str(p["jobs"])]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not grid_out.exists():
            info["err"] = (r.stderr.strip().splitlines() or ["(grid failed)"])[-1]
            return info

    if not grid_out.exists():
        info["err"] = "grid missing (need a grid render before --refresh-trails)"
        return info
    dims = probe_dims(grid_out)
    if dims is None:
        info["err"] = "could not probe grid dims"
        return info
    size = dims[1]   # square; height == width

    # 2) trails (matched size + walk params) ---------------------------------
    if overwrite or refresh_trails or not trails_out.exists():
        cmd = [str(VENV_PY), str(TRAILS), "--run", str(run), "--out", str(trails_out),
               "-k", "36", "--method", "best-first", "-K", str(p["K"]),
               "--max-n", str(p["max_n"]), "--require-genome",
               "--size", str(size), "--margin", "16", "--fps", str(p["fps"]),
               "--frames-per-edge", str(p["frames_per_edge"]), "--hold", str(p["hold"]),
               "--theme", theme]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not trails_out.exists():
            info["err"] = (r.stderr.strip().splitlines() or ["(trails failed)"])[-1]
            return info

    # 3) combined: trails on TOP, grid on BOTTOM (vstack) --------------------
    if overwrite or refresh_trails or not combined_out.exists():
        filt = f"[0:v][1:v]{stack}=inputs=2"
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-i", str(trails_out), "-i", str(grid_out),
               "-filter_complex", filt,
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(combined_out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not combined_out.exists():
            info["err"] = (r.stderr.strip().splitlines() or [f"({stack} failed)"])[-1]
            return info

    info["ok"] = True
    return info


def build_index(out_dir: Path, labels: Dict[str, str], ngen: Dict[str, int]):
    combined = sorted(out_dir.glob("*_combined.mp4"),
                      key=lambda p: (re.sub(r"_s\d+$", "", p.stem), p.stem))
    cards = []
    for c in combined:
        name = c.stem[:-len("_combined")]
        setting = re.sub(r"_s\d+$", "", name)
        label = labels.get(setting, setting)
        n = ngen.get(name, "?")
        cards.append(
            f'<figure>'
            f'<video src="{c.name}" controls loop muted preload="metadata" width="320"></video>'
            f'<figcaption><b>{label}</b><br>{name}<br>{n} genomes &middot; trails / grid</figcaption>'
            f'</figure>'
        )
    html = f"""<!doctype html><meta charset=utf-8>
<title>Walker grid + trails sweep</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:1.5em;background:#fafafa;color:#222}}
h1{{font-weight:400}}
.row{{display:flex;flex-wrap:wrap;gap:20px}}
figure{{margin:0}} video{{border-radius:3px;box-shadow:0 1px 4px rgba(0,0,0,.18);background:#fff}}
figcaption{{font-size:12px;color:#555;max-width:640px;word-break:break-all;line-height:1.4;margin-top:4px}}
b{{color:#FF6C00;font-size:14px}}
</style>
<h1>Walker grid + phylogeny trails &nbsp;<small style="font-size:.5em;color:#888">{len(combined)} runs</small></h1>
<p style="color:#666">Top: phylogeny traversal (36 colored walkers). Bottom: the same 36 walks as a CPPN morph grid. Frame-synced.</p>
<div class="row">{''.join(cards)}</div>
"""
    (out_dir / "index.html").write_text(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="quick")
    ap.add_argument("--run", action="append", type=Path, default=[],
                    help="explicit run dir(s); overrides exemplar/discovery")
    ap.add_argument("--include", action="append", default=[],
                    help="discover runs whose name contains ALL of these (one richest seed per setting)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="drop runs whose name contains ANY of these")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--refresh-trails", action="store_true",
                    help="reuse existing grids; only re-render trails + combined "
                         "(cheap; use after changing --theme / --stack)")
    ap.add_argument("--theme", choices=["dark", "light"], default="light",
                    help="trails color theme (light = distinct hues on white; "
                         "dark = luminous on near-black)")
    ap.add_argument("--stack", choices=["vstack", "hstack"], default="vstack",
                    help="vstack = trails on top, grid on bottom (default)")
    ap.add_argument("--index-only", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=HERE / "out" / "grid_sweep")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    p = PRESETS[args.preset]

    # Build work list: (label, name, run) -----------------------------------
    labels: Dict[str, str] = {}
    ngen: Dict[str, int] = {}
    work: List[Tuple[str, Path]] = []

    if args.run:
        for run in args.run:
            name = run.name
            labels[re.sub(r"_s\d+$", "", name)] = name
            ngen[name] = n_genomes(run)
            work.append((name, run))
    elif args.include:
        for setting, name, run, n in discover(args.include, args.exclude):
            labels[setting] = setting
            ngen[name] = n
            work.append((name, run))
    else:
        for label, prefix in EXEMPLARS:
            picked = richest_seed(prefix)
            if picked is None:
                print(f"[skip] no run found for {label}: {prefix}")
                continue
            name, run, n = picked
            labels[prefix] = label
            ngen[name] = n
            work.append((name, run))

    if args.limit:
        work = work[: args.limit]

    if args.index_only:
        build_index(out_dir, labels, ngen)
        print(f"index -> {out_dir / 'index.html'}")
        return

    print(f"{len(work)} run(s) to render [{args.preset}] -> {out_dir}")
    t0 = time.time()
    for i, (name, run) in enumerate(work, 1):
        print(f"\n[{i}/{len(work)}] {name}  ({ngen.get(name, '?')} genomes)")
        t = time.time()
        info = render_one(name, run, out_dir, p, args.overwrite,
                          theme=args.theme, stack=args.stack,
                          refresh_trails=args.refresh_trails)
        tag = "ok  " if info["ok"] else "FAIL"
        print(f"  {tag} {time.time() - t:.0f}s" + ("" if info["ok"] else f"  :: {info.get('err')}"))
        build_index(out_dir, labels, ngen)   # refresh so it's browsable mid-run

    build_index(out_dir, labels, ngen)
    print(f"\ndone in {time.time() - t0:.0f}s -> open {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
