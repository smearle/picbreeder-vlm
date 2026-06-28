#!/usr/bin/env python3
"""Render an archive animation across every run in sweep_logs, and build a
browsable HTML index so you can scan many archives and pick favourites.

Defaults to a fast/small preset of the compact growth animation (so all ~200
runs finish in minutes, parallelised across cores). Re-render favourites at full
quality afterwards with the individual scripts.

Examples:
    # everything (compact, quick), 8 workers
    .venv/bin/python archive_animations/sweep_animations.py --jobs 8

    # only the context-length and noise sweeps for gemini-2.5-pro
    .venv/bin/python archive_animations/sweep_animations.py --include model-gemini-2.5-pro

    # the sfdp tree layout instead
    .venv/bin/python archive_animations/sweep_animations.py --script tree

    # just rebuild the index from already-rendered clips
    .venv/bin/python archive_animations/sweep_animations.py --index-only
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV_PY = REPO / ".venv" / "bin" / "python"
SWEEP_ROOT = REPO / "sweep_logs"

PRESETS = {
    "compact": {
        "script": "archive_grow_compact.py",
        "quick": ["--frame", "560", "--per-frame", "24", "--frames-per-step", "1",
                  "--thumb-px", "16", "--world", "1500", "--node-size", "children"],
        "full": ["--frame", "1000", "--per-frame", "8", "--frames-per-step", "2",
                 "--thumb-px", "26", "--world", "2400", "--node-size", "children"],
    },
    "tree": {
        "script": "archive_tree.py",
        "quick": ["--engine", "sfdp", "--size", "640", "--thumb", "12", "--per-frame", "28"],
        "full": ["--engine", "sfdp", "--size", "1080", "--thumb", "15", "--per-frame", "16"],
    },
}


def discover(min_images, includes, excludes):
    runs = []
    for meta in SWEEP_ROOT.glob("**/archive/archive_metadata.json"):
        run = meta.parent.parent
        name = run.name
        if includes and not all(t in name for t in includes):
            continue
        if any(t in name for t in excludes):
            continue
        imgs = run / "archive" / "images"
        n = len(os.listdir(imgs)) if imgs.is_dir() else 0
        if n < min_images:
            continue
        runs.append((name, run, n))
    runs.sort(key=lambda r: (re.sub(r"_s\d+$", "", r[0]), r[0]))
    return runs


def build_index(out_dir: Path, script: str):
    clips = sorted(out_dir.glob("*.mp4"), key=lambda p: (re.sub(r"_s\d+$", "", p.stem), p.stem))
    # group by setting (seed stripped)
    groups: dict[str, list[Path]] = {}
    for c in clips:
        groups.setdefault(re.sub(r"_s\d+$", "", c.stem), []).append(c)
    rows = []
    for g, cs in groups.items():
        cards = []
        for c in cs:
            n = NIMG.get(c.stem, "?")
            cards.append(
                f'<figure><video src="{c.name}" controls loop muted preload="metadata" '
                f'width="320"></video><figcaption>{c.stem}<br><b>{n}</b> images</figcaption></figure>'
            )
        rows.append(f'<h2>{g}</h2><div class="row">{"".join(cards)}</div>')
    html = f"""<!doctype html><meta charset=utf-8>
<title>Archive animations sweep ({script})</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:1.5em;background:#fafafa;color:#222}}
h1{{font-weight:400}} h2{{font-weight:400;margin:1.4em 0 .4em;color:#444;border-bottom:1px solid #ddd}}
.row{{display:flex;flex-wrap:wrap;gap:14px}}
figure{{margin:0}} video{{border-radius:3px;box-shadow:0 1px 4px rgba(0,0,0,.15);background:#fff}}
figcaption{{font-size:11px;color:#666;max-width:320px;word-break:break-all;line-height:1.35}}
b{{color:#FF6C00}}
</style>
<h1>Archive animations&mdash;{script} &nbsp;<small style="font-size:.5em;color:#888">{len(clips)} clips</small></h1>
<p style="color:#666">Scan and note favourites; re-render those at full quality with the per-run script.</p>
{''.join(rows)}
"""
    (out_dir / "index.html").write_text(html)


NIMG: dict[str, int] = {}
_lock = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", choices=list(PRESETS), default="compact")
    ap.add_argument("--preset", choices=["quick", "full"], default="quick")
    ap.add_argument("--include", action="append", default=[], help="keep runs whose name contains ALL of these")
    ap.add_argument("--exclude", action="append", default=[], help="drop runs whose name contains ANY of these")
    ap.add_argument("--min-images", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REPO / "archive_animations" / "out" / "sweep")
    ap.add_argument("--extra", default="", help="extra args passed through to the animation script")
    ap.add_argument("--index-only", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir / args.script
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = REPO / "archive_animations" / PRESETS[args.script]["script"]
    preset_args = PRESETS[args.script][args.preset]
    extra = shlex.split(args.extra)

    runs = discover(args.min_images, args.include, args.exclude)
    if args.limit:
        runs = runs[: args.limit]
    for name, _run, n in runs:
        NIMG[name] = n

    if args.index_only:
        build_index(out_dir, args.script)
        print(f"index -> {out_dir/'index.html'}")
        return

    todo = []
    for name, run, n in runs:
        out = out_dir / f"{name}.mp4"
        if out.exists() and not args.overwrite:
            continue
        todo.append((name, run, out))
    print(f"{len(runs)} runs match; {len(todo)} to render ({len(runs)-len(todo)} already done) "
          f"-> {out_dir}  [{args.script}/{args.preset}, jobs={args.jobs}]")

    done = [0]
    t0 = time.time()

    def work(job):
        name, run, out = job
        cmd = [str(VENV_PY), str(script_path), "--run", str(run), "--out", str(out),
               *preset_args, *extra]
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = r.returncode == 0 and out.exists()
        with _lock:
            done[0] += 1
            tag = "ok " if ok else "FAIL"
            print(f"[{done[0]}/{len(todo)}] {tag} {time.time()-t:5.1f}s  {name}")
            if not ok:
                print("    " + (r.stderr.strip().splitlines() or ["(no stderr)"])[-1])
            build_index(out_dir, args.script)  # refresh so it's browsable mid-run
        return ok

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(as_completed(ex.submit(work, j) for j in todo))

    build_index(out_dir, args.script)
    print(f"done in {time.time()-t0:.0f}s -> open {out_dir/'index.html'}")


if __name__ == "__main__":
    main()
