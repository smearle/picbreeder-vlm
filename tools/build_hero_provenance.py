#!/usr/bin/env python3
"""Map each hero-banner cell to the exact archive image it came from, so a
double-click on the hero grid can open the inline archive viewer at that run and
park the loupe on that image.

Reads:
  archive_animations/teaser_provenance.json   (hero id -> source PNG path)
  <deploy>/assets/hero_sprites/manifest.json   (the 43 hero cells, in grid order)
Writes:
  <deploy>/assets/hero_sprites/provenance.json
    { "<hero_id>": { run, arc, model, seed, index, axisKey, axisVal } , ... }
  where
    run      : exact sweep run-dir basename (what the viewer mounts)
    arc      : results-table arc key, or null for off-table runs (gemini-3, flash-lite)
    model    : gemini-2.5-pro | gemini-3-pro-preview | gemini-2.5-flash-lite
    seed     : replicate seed
    index    : 0-based publication index of the image in that run (loupe target)
    axisKey  : which intro-viewer knob to highlight (noise|memory|agents|null)
    axisVal  : that knob's button value (string), null when none differs

ASSETS_DIR points at the live (hash-suffixed) deploy; override with argv[1].
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path("/home/jupyter-smearle/picbreeder-vlm")
sys.path.insert(0, str(REPO / "tools"))
from hf_archive_push import parse_config, canonical_arc   # noqa: E402

TEASER = REPO / "archive_animations" / "teaser_provenance.json"


def find_deploy() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    base = Path("/home/jupyter-smearle/smearle.github.io")
    cands = sorted(base.glob("picbreeder-vlm*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in cands:
        if (c / "assets" / "hero_sprites" / "manifest.json").is_file():
            return c
    raise SystemExit("no picbreeder-vlm deploy with hero_sprites/manifest.json found")


# Noise epsilon -> the intro viewer's button label (see intro-thumbs-gallery AI_RUNS).
NOISE_LABEL = {0.05: ".05", 0.25: ".25", 0.5: ".5", 0.75: ".75", 1.0: "1"}


def display_axis(cfg: dict):
    """Which single noise/memory/agents knob differs from the default run, and its
    button value. Model is an orthogonal knob, handled separately."""
    if cfg.get("personalities"):
        return "agents", str(cfg["personalities"])
    eps = cfg.get("noise_eps") or 0.0
    if eps > 0:
        return "noise", NOISE_LABEL.get(eps, str(eps))
    cl = cfg.get("memory_cl")
    if cl is not None and cl != 1:
        return "memory", "20" if cl == -1 else str(cl)
    return None, None


def main() -> int:
    deploy = find_deploy()
    sprites = deploy / "assets" / "hero_sprites"
    manifest = json.loads((sprites / "manifest.json").read_text())
    teaser = json.loads(TEASER.read_text())

    out: dict[str, dict] = {}
    misses = []
    for entry in manifest:
        hid = entry["id"]
        src = teaser.get(hid + ".png")
        if not src:
            misses.append(hid)
            continue
        m = re.search(r"sweep/([^/]+)/archive/images/img_(\d+)\.png", src)
        if not m:
            misses.append(hid)
            continue
        run, num = m.group(1), int(m.group(2))
        cfg = parse_config(run)
        axis_key, axis_val = display_axis(cfg)
        out[hid] = {
            "run": run,
            "arc": canonical_arc(run, cfg),
            "model": cfg.get("model"),
            "seed": cfg.get("seed"),
            "index": num - 1,            # img_000093 -> 92nd publication (0-based)
            "axisKey": axis_key,
            "axisVal": axis_val,
        }

    (sprites / "provenance.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {sprites/'provenance.json'} for {len(out)}/{len(manifest)} hero cells")
    if misses:
        print(f"  WARNING no provenance for: {misses}")
    # quick summary of the runs the hero references
    runs = sorted({(v['run']) for v in out.values()})
    print(f"  {len(runs)} distinct source runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
