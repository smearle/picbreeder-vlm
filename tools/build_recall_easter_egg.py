#!/usr/bin/env python3
"""Build the data for the Semantic Recall panel easter egg.

The blog's Semantic Recall panel (grid_recall.png) shows the 9 best noun->image
matches across every archive. Clicking the panel reveals a draggable strip that
lets the reader scroll through the *whole* THINGS vocabulary -- every noun paired
with the single closest archive image to it, across all VLM sweep runs plus the
human Picbreeder archive -- sorted from best match to worst.

This tool computes that best-image-per-noun mapping (reusing the recall machinery
in build_metric_fig_assets), renders each winning image into a sprite atlas, and
writes a manifest the frontend (assets/page/recall-egg.js) blits from.

Outputs (under <blog>/assets/recall_things/):
    things_atlas.jpg   one atlas image, `cell`x`cell` thumbnails in manifest order
    manifest.json      {cell, cols, count, items:[{noun, sim, cond}, ...]}

Items are in atlas order == sort order (highest similarity first); the frontend
derives each thumbnail's atlas slot from its index (col=i%cols, row=i//cols).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_metric_fig_assets as B  # noqa: E402  (sets up REPO, recall helpers)

REPO = B.REPO
DEFAULT_OUT = Path.home() / "smearle.github.io/picbreeder-vlm-06b0d76d/assets/recall_things"

NOUNS_FILE = REPO / "noun_lists/things_deduped.txt"
NOUN_EMB_FILE = (REPO / "noun_lists/embeddings/"
                 "noun_embeddings_things_deduped_ViT-SO400M-14-SigLIP2_webli.npy")


def best_image_per_noun():
    """For every deduped THINGS noun, the single closest archive image across all
    archives. Returns [(noun, source_key, filename, sim), ...] sorted by sim desc."""
    nouns = [ln.strip().replace("_", " ")
             for ln in NOUNS_FILE.read_text().splitlines() if ln.strip()]
    noun_emb = np.load(NOUN_EMB_FILE).astype(np.float32)
    assert noun_emb.shape[0] == len(nouns), (noun_emb.shape, len(nouns))

    n = len(nouns)
    best_sim = np.full(n, -2.0, np.float32)
    best_src = [None] * n
    best_fn = [None] * n
    n_arch = 0
    for source, emb, fns in B._iter_recall_sources():
        n_arch += 1
        sims = emb @ noun_emb.T            # (n_img, n_noun); embeddings L2-normalized
        mi = sims.argmax(0)               # best image row per noun
        mv = sims[mi, np.arange(n)]
        upd = mv > best_sim
        for p in np.where(upd)[0]:
            best_sim[p] = mv[p]
            best_src[p] = source
            best_fn[p] = fns[mi[p]]
    print(f"  searched {n_arch} archives over {n} nouns")

    order = np.argsort(-best_sim)
    return [(nouns[p], best_src[p], best_fn[p], float(best_sim[p])) for p in order]


def build(out: Path, cell: int, cols: int, quality: int) -> None:
    picks = best_image_per_noun()
    count = len(picks)
    rows = (count + cols - 1) // cols
    print(f"  atlas {cols}x{rows} @ {cell}px -> {cols*cell}x{rows*cell}")

    atlas = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    items = []
    miss = 0
    for i, (noun, src, fn, sim) in enumerate(picks):
        path = B._recall_image_path(src, fn)
        try:
            im = Image.open(path).convert("RGB").resize((cell, cell), Image.LANCZOS)
        except (FileNotFoundError, OSError):
            miss += 1
            im = Image.new("RGB", (cell, cell), (238, 238, 238))
        atlas.paste(im, ((i % cols) * cell, (i // cols) * cell))
        items.append({
            "noun": noun,
            "sim": round(sim, 3),
            "cond": " · ".join(B._recall_condition_parts(src)),
        })

    out.mkdir(parents=True, exist_ok=True)
    atlas.save(out / "things_atlas.jpg", quality=quality, optimize=True)
    manifest = {"cell": cell, "cols": cols, "count": count,
                "atlas": "things_atlas.jpg", "items": items}
    (out / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))

    sz = (out / "things_atlas.jpg").stat().st_size
    print(f"wrote {out/'things_atlas.jpg'}  ({sz/1e6:.2f} MB, {miss} images missing)")
    print(f"wrote {out/'manifest.json'}  ({count} noun-image pairs)")
    print("  best :", ", ".join(f"{p[0]} ({p[3]:.2f})" for p in picks[:8]))
    print("  worst:", ", ".join(f"{p[0]} ({p[3]:.2f})" for p in picks[-8:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cell", type=int, default=96, help="atlas thumbnail size (px)")
    ap.add_argument("--cols", type=int, default=42, help="atlas columns")
    ap.add_argument("--quality", type=int, default=86)
    args = ap.parse_args()
    build(args.out, args.cell, args.cols, args.quality)


if __name__ == "__main__":
    main()
