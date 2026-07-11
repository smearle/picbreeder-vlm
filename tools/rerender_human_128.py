#!/usr/bin/env python
"""Re-render the human Picbreeder archive at 128px (to match the VLM runs), reusing the repo's
own pipeline: plot_lineage_phylogeny._load_lineage_info reads each genome's OG zips, and
_render_thumbnail_array renders the final genome at the requested resolution.

Output: human_lineages/lineages/lineage_phylogeny_thumbs_128/<pid>_final.png  (same naming as the 64px set)
Idempotent: skips a pid whose 128px thumb already exists. Pass --limit N to test on a few.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.append("/home/jupyter-smearle/picbreeder-vlm")
from picbreeder_vlm._paths import FER_ROOT, ensure_fer_importable
ensure_fer_importable()
from fer.src.plot_lineage_phylogeny import _load_lineage_info, _render_thumbnail_array

ROOT = Path("/home/jupyter-smearle/picbreeder-vlm")
PB = FER_ROOT / "spaghetti/pbRender/genomeAll"
SRC64 = ROOT / "human_lineages/lineages/lineage_phylogeny_thumbs"
OUT = ROOT / "human_lineages/lineages/lineage_phylogeny_thumbs_128"


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    OUT.mkdir(parents=True, exist_ok=True)
    pids = sorted((p.name[: -len("_final.png")] for p in SRC64.glob("*_final.png")), key=int)
    if limit:
        pids = pids[:limit]
    done = ok = miss = 0
    for pid in pids:
        out = OUT / f"{pid}_final.png"
        if out.exists():
            ok += 1; continue
        node = _load_lineage_info(PB, pid)
        if node is None:
            miss += 1; continue
        arr = _render_thumbnail_array(node.final_genome, 128)
        if arr is None:
            miss += 1; continue
        Image.fromarray(arr, "RGB").save(out)
        ok += 1; done += 1
        if done and done % 500 == 0:
            print(f"  rendered {done} (ok={ok} miss={miss})", flush=True)
    print(f"DONE: {ok} thumbs at 128px ({done} new), {miss} missing, of {len(pids)} pids", flush=True)


if __name__ == "__main__":
    main()
