#!/usr/bin/env python
"""Adapt the HUMAN Picbreeder archive into a sweep-style run directory so the
archive-growth animation panels can consume it unchanged.

The three triptych panels (archive_tree.py / leaderboard.py /
archive_scroll_lineage.py) all expect a run with:
    <run>/archive/archive_metadata.json   (entries: id, source_entry_ids, added_at, ...)
    <run>/archive/images/<id>.png

This script synthesizes that from the human gallery:
  - gallery order = human pids that have BOTH a 128px thumb and a SigLIP row,
    sorted by integer pid (picbreeder assigns ids incrementally -> publication
    order). Reuses tools/build_human_sprites.{human_ids,human_lineage_entries}.
  - id          = img_{i+1:06d} (same scheme the lineage helper emits)
  - source_entry_ids = in-archive picbreeder branchFrom parent (or [] for roots)
  - added_at    = synthetic monotonically-increasing ISO timestamps (publication
                  order); the panels only sort by this, the absolute value is moot
  - generation  = lineage depth (root = 0)
  - title       = the VLM caption for that image, if available (leaderboard label)
  - color_enabled = True (human CPPNs are colour; only matters for morph mode)
  - images/<id>.png = symlink to human_lineages/lineages/lineage_phylogeny_thumbs_128/<pid>_final.png

No genomes are emitted (human CPPNs aren't pickled), so the scroll panel must run
--reveal pop and the leaderboard --board branched (no VLM ratings exist).
"""
import argparse
import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path("/home/jupyter-smearle/picbreeder-vlm")
THUMBS = REPO / "human_lineages/lineages/lineage_phylogeny_thumbs_128"
CAPTIONS = REPO / "data/human_baseline/res-128/captions_human.json"


def _load_sprites_module():
    spec = importlib.util.spec_from_file_location(
        "build_human_sprites", REPO / "tools" / "build_human_sprites.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _generations(entries):
    """Lineage depth per entry id (root = 0). entries are in publication order and
    a parent always precedes its child, so a single forward pass suffices."""
    depth = {}
    for e in entries:
        src = e["source_entry_ids"]
        depth[e["id"]] = depth.get(src[0], -1) + 1 if src else 0
    return depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO / "archive_animations/out/human_run",
                    help="run directory to create (gets archive/ underneath)")
    ap.add_argument("--n", type=int, default=None,
                    help="cap the archive to the first N publications (default: all)")
    args = ap.parse_args()

    sprites = _load_sprites_module()
    ids = sprites.human_ids()                      # pid strings, publication order
    if args.n:
        ids = ids[: args.n]
    entries = sprites.human_lineage_entries(ids)   # id + source_entry_ids (in-set edges)
    depth = _generations(entries)

    captions = {}
    if CAPTIONS.exists():
        raw = json.loads(CAPTIONS.read_text())
        captions = {k[:-4] if k.endswith(".png") else k: v for k, v in raw.items()}

    base = datetime(2007, 1, 1)
    for i, (pid, e) in enumerate(zip(ids, entries)):
        e["added_at"] = (base + timedelta(seconds=i)).isoformat()
        e["generation"] = depth[e["id"]]
        e["agent_id"] = "human"
        e["color_enabled"] = True
        cap = captions.get(pid)
        if cap:
            e["title"] = cap

    out = args.out
    img_dir = out / "archive" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    # refresh image symlinks
    for old in img_dir.glob("img_*.png"):
        old.unlink()
    n_imgs = 0
    for pid, e in zip(ids, entries):
        src = THUMBS / f"{pid}_final.png"
        if not src.is_file():
            continue
        (img_dir / f"{e['id']}.png").symlink_to(src)
        n_imgs += 1

    meta = {"entries": entries, "n": len(entries), "source": "human-picbreeder"}
    (out / "archive" / "archive_metadata.json").write_text(json.dumps(meta))

    n_edges = sum(1 for e in entries if e["source_entry_ids"])
    print(f"human run -> {out}")
    print(f"  {len(entries)} entries, {n_imgs} image symlinks, "
          f"{n_edges} in-archive parent edges, max generation {max(depth.values())}")


if __name__ == "__main__":
    main()
