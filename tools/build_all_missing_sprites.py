#!/usr/bin/env python3
"""Build sprite sets for every genome-ready run that lacks one locally, so the
preview-first per-run browse (browse.html?run=) can show sprite-sheet thumbnails
for ALL runs instead of live-rendering genomes.

Reuses build_extra_sprites.build_run (build_archive_image_lib -> make_sprite_sheets)
and write_index. The run list is the set of runs missing a local
site/<run>/sprite/layout.json, restricted to those with local sweep data.

  python tools/build_all_missing_sprites.py --runs runs.txt   # build runs in file missing a sprite
  python tools/build_all_missing_sprites.py                    # all local-data runs missing a sprite
  python tools/build_all_missing_sprites.py --dry-run          # just list

Then push with tools/push_sprites.py + tools/build_hf_index.py.
"""
import sys
from pathlib import Path

REPO = Path("/home/jupyter-smearle/picbreeder-vlm")
sys.path.insert(0, str(REPO / "tools"))
from build_extra_sprites import build_run, write_index, SITE, SWEEP   # noqa: E402


def missing_runs():
    have = {p.parent.parent.name for p in SITE.glob("*/sprite/layout.json")}
    if "--runs" in sys.argv:
        runlist = Path(sys.argv[sys.argv.index("--runs") + 1])
        want = [r.strip() for r in runlist.read_text().splitlines() if r.strip()]
    else:
        want = sorted(d.name for d in SWEEP.iterdir() if d.is_dir())
    # keep only runs that have local sweep data and aren't already built
    return [r for r in want if r not in have and (SWEEP / r).is_dir()]


def main():
    dry = "--dry-run" in sys.argv
    todo = missing_runs()
    print(f"{len(todo)} runs missing a local sprite set:", flush=True)
    for r in todo:
        print("   ", r)
    if dry:
        return
    ok = 0
    for i, r in enumerate(todo, 1):
        print(f"\n=== [{i}/{len(todo)}] {r} ===", flush=True)
        try:
            build_run(r)
            ok += 1
        except Exception as e:
            print(f"[ERROR] {r}: {e}", flush=True)
    write_index()
    print(f"\nDONE — built {ok}/{len(todo)} sprite sets", flush=True)


if __name__ == "__main__":
    main()
