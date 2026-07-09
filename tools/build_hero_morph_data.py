#!/usr/bin/env python3
"""Export the hero banner's 43 lineage genome CHAINS as browser-renderable JSON, so
the banner can morph CPPNs live on the GPU (cppn-gl / hero-morph.js). This is the
banner's only render path -- it replaced ~44 MB of pre-baked sprite sheets, now
retired. Resolution-independent + tiny.

Per fig (root->published, deduped consecutive-identical genomes, the exact chain
teaser_lineages.py bakes into each morph clip):
  <deploy>/assets/hero_morph/<fig>.json.gz  = {
     "color": bool,                # single color mode for the whole morph (final image's)
     "pubs":  [canon idx, ...],    # chain positions that are published keyframes (title holds)
     "genomes": [ genome_to_json, ... ]   # cppn.js schema; index 0 = random root
  }
and a manifest <deploy>/assets/hero_morph/manifest.json = {fig: {n, depth, pubs, color, transforms}}.

Run from the repo root (PYTHONPATH=.) so the NEAT genome classes unpickle.

  PYTHONPATH=. python tools/build_hero_morph_data.py            # all 43 hero figs
  PYTHONPATH=. python tools/build_hero_morph_data.py --dry-run  # list, don't write
"""
from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "archive_animations"))

from picbreeder_vlm.core import neat_components  # noqa: F401  (registers genome classes for unpickling)
from picbreeder_vlm.core.genome_json import genome_to_json  # noqa: E402
from teaser_lineages import build_full_lineage, run_of, id_of  # noqa: E402

DEPLOY = Path("/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d")
HERO = DEPLOY / "assets" / "hero_sprites"
OUT = DEPLOY / "assets" / "hero_morph"
TEASER_PROV = REPO / "archive_animations" / "teaser_provenance.json"
SWEEP = REPO / "sweep_logs" / "sweep"


def hero_figs() -> list[str]:
    man = json.loads((HERO / "manifest.json").read_text())
    return [e["fig"] for e in man]        # banner order


def build_fig(fig: str, prov: dict) -> dict | None:
    src = prov.get(fig + ".png") or prov.get(fig)
    if not src:
        print(f"[skip] {fig}: not in teaser_provenance"); return None
    run, tid = run_of(src), id_of(src)
    run_dir = SWEEP / run
    if not run_dir.is_dir():
        print(f"[skip] {fig}: run dir missing ({run})"); return None
    with tempfile.TemporaryDirectory() as wd:
        canon, info = build_full_lineage(run_dir, tid, Path(wd))
    if info["missing"]:
        print(f"[warn] {fig}: {len(info['missing'])} missing agent zips; chain may start mid-branch")
    genomes = [genome_to_json(g) for _gen, g in canon]
    transforms = sum(1 for g in genomes if "inAct" in g or "outAct" in g)
    rec = {
        "color": bool(info["color_enabled"]),
        "pubs": info["pub_canon_idx"],            # canon indices of published genomes (title anchors)
        "titles": info.get("pub_titles", []),     # aligned 1:1 with pubs (archive-metadata titles)
        "genomes": genomes,
    }
    meta = {"n": len(genomes), "depth": info["depth"], "pubs": info["pub_canon_idx"],
            "color": rec["color"], "transforms": transforms}
    return {"rec": rec, "meta": meta}


def main() -> int:
    dry = "--dry-run" in sys.argv
    prov = json.loads(TEASER_PROV.read_text())
    figs = hero_figs()
    print(f"{len(figs)} hero figs -> {OUT}")
    if not dry:
        OUT.mkdir(parents=True, exist_ok=True)
    manifest, tot_bytes, tot_transforms = {}, 0, 0
    for i, fig in enumerate(figs, 1):
        built = build_fig(fig, prov)
        if not built:
            continue
        rec, meta = built["rec"], built["meta"]
        tot_transforms += meta["transforms"]
        if not dry:
            blob = gzip.compress(json.dumps(rec, separators=(",", ":")).encode(), 6)
            (OUT / f"{fig}.json.gz").write_bytes(blob)
            tot_bytes += len(blob)
            meta["bytes"] = len(blob)
        manifest[fig] = meta
        print(f"  [{i:2}/{len(figs)}] {fig}: {meta['n']} genomes (depth {meta['depth']}), "
              f"{len(meta['pubs'])} pubs, color={meta['color']}, transforms={meta['transforms']}"
              + (f", {meta.get('bytes',0)/1024:.0f} KB" if not dry else ""))
    if not dry:
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n{len(manifest)}/{len(figs)} figs exported, {tot_bytes/1e6:.2f} MB total, "
          f"{tot_transforms} genomes use input/output activation transforms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
