#!/usr/bin/env python3
"""Dump the human + VLM car lineages as individual step images (+ VLM titles)
for dynamic, reflowing placement in the blog post.

VLM lineage  : default gemini run s8, target img_000810 ("Chrome Streamliner"),
               traced back along source_entry_ids[0] to the root.
Human lineage: human Picbreeder PID 464, traced via branchFrom to the root,
               using the pre-rendered 128px thumbnails.

Outputs to <site>/assets/lineages/{vlm,human}/NN.png and manifest.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SITE = Path("/home/jupyter-smearle/smearle.github.io/picbreeder-vlm")
DEST = SITE / "assets" / "lineages"
OUT_PX = 256  # save crisp; CPPN images upscale cleanly

VLM_RUN = REPO / "sweep_logs/sweep/th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s8"
VLM_TARGET = "img_000810"
HUMAN_ROOT = REPO / "fer/spaghetti/pbRender/genomeAll"
HUMAN_PRE = REPO / "fer/src/archive_res-128/images"
HUMAN_TARGET = "464"


def save_img(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    if im.width != OUT_PX:
        im = im.resize((OUT_PX, OUT_PX), Image.LANCZOS)
    im.save(dst)


def extract_vlm():
    meta = json.load(open(VLM_RUN / "archive/archive_metadata.json"))
    by = {str(e["id"]): e for e in meta["entries"]}
    chain, cur, seen = [], VLM_TARGET, set()
    while cur and cur in by and cur not in seen:
        seen.add(cur)
        chain.append(by[cur])
        src = by[cur].get("source_entry_ids") or []
        cur = str(src[0]) if src else None
    chain.reverse()
    items = []
    for i, e in enumerate(chain):
        img = VLM_RUN / "archive/images" / Path(e["image_path"]).name
        out = DEST / "vlm" / f"{i:02d}.png"
        save_img(img, out)
        items.append({"file": f"assets/lineages/vlm/{i:02d}.png",
                      "title": (e.get("title") or "").strip()})
    print(f"VLM: {len(items)} steps -> {[it['title'] for it in items]}")
    return items


def extract_human():
    import visualize_human_ancestry as vha
    nodes = vha.trace_ancestry(HUMAN_TARGET, HUMAN_ROOT)
    chain, cur = [], HUMAN_TARGET
    while cur and cur in nodes:
        chain.append(cur)
        cur = nodes[cur].parent_pid
    chain.reverse()
    items = []
    i = 0
    for pid in chain:
        pre = HUMAN_PRE / f"{pid}.png"
        if not pre.exists():
            print(f"  (skip human PID {pid}: no thumbnail)")
            continue
        out = DEST / "human" / f"{i:02d}.png"
        save_img(pre, out)
        items.append({"file": f"assets/lineages/human/{i:02d}.png", "pid": pid})
        i += 1
    print(f"Human: {len(items)} steps (PIDs {[it['pid'] for it in items]})")
    return items


def main():
    vlm = extract_vlm()
    human = extract_human()
    manifest = {"vlm": vlm, "human": human}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {DEST/'manifest.json'}")


if __name__ == "__main__":
    main()
