#!/usr/bin/env python
"""Build coverage-maximizing (FPS) representative grids for the blog's intro
archive thumbnails — both the AI grids and grid_human_representative.png.

Both archives use the SAME pipeline:
  - take the first `limit` published images (default 2000),
  - greedy k-center / farthest-point selection of `k` images over the cached
    SigLIP-2 embeddings,
  - rasterfairy "rect" layout (UMAP, random_state=0),
  - rendered square and resized to 1200x1200.

Alongside each clickable thumbnail (the AI 'default' arc and 'human') a focus
manifest is written to assets/thumb_grids/<key>.json mapping every grid cell to the
0-based publication index of the image it shows. The intro thumbnails read these so a
click on a cell parks the inline viewer's loupe on that exact image (the cell layout
and the manifest are produced in the same render call, so they always agree).

Usage:
    python tools/build_intro_thumb_grids.py            # default arcs + human
    python tools/build_intro_thumb_grids.py default human
    python tools/build_intro_thumb_grids.py noise_0.25 agents_1000 ...

Output: <blog>/assets/grid_ai_<arc>.png (grid_ai_representative.png for 'default'),
        grid_human_representative.png, and assets/thumb_grids/{ai,human}.json.
"""
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from picbreeder_vlm.analysis.embed_and_visualize import (  # noqa: E402
    _greedy_k_center,
    reduce_embeddings,
    render_rectangular_grid,
)
from tools.build_archive_image_lib import ARC_TO_RUN, SWEEP  # noqa: E402

ASSETS = Path("/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets")
# Per-thumbnail "focus" manifests: which archive image sits in each cell, addressed by
# its 0-based publication index (== sprite/chronological index in the inline gallery).
# The intro thumbnails read these so a click on a cell parks the viewer's loupe on that
# exact image. Keyed by the figure's data-src ('ai' = default arc s4, 'human').
FOCUS_DIR = ASSETS / "thumb_grids"
SEED = 4
LIMIT = 2000   # match the human grid's first-2000 sample
K = 36         # representative_k (6x6)
SIZE = 1200    # match grid_human_representative.png

# default set: the featured run + a few visually distinct runs for the stack.
DEFAULT_ARCS = ["default", "agents_1000", "mem_10", "noise_0.5"]


def _write_focus_manifest(key, source, arc, seed, grid_positions, dims, focus):
    """Write thumb_grids/<key>.json mapping each grid cell -> publication index.

    grid_positions[idx] == (col, row) is where rep `idx` landed; focus[idx] is that
    rep's 0-based publication index. Cells with no rep are null.
    """
    if grid_positions is None or dims is None:
        print(f"  [!] no grid layout for {key}; skipping focus manifest")
        return
    cols, rows = int(dims[0]), int(dims[1])
    grid = [[None] * cols for _ in range(rows)]
    for idx, (col, row) in enumerate(grid_positions):
        r, c = int(row), int(col)
        fi = focus[idx]
        if 0 <= r < rows and 0 <= c < cols and fi is not None and fi >= 0:
            grid[r][c] = int(fi)
    FOCUS_DIR.mkdir(parents=True, exist_ok=True)
    out = FOCUS_DIR / f"{key}.json"
    out.write_text(json.dumps(
        {"source": source, "arc": arc, "seed": seed, "cols": cols, "rows": rows, "grid": grid}))
    print(f"  focus manifest -> {out}  ({cols}x{rows})")


def _pub_index(path: Path) -> int:
    """img_000093.png -> 92 (0-based publication index; matches the gallery's focus index)."""
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) - 1


def build(arc: str) -> Path:
    run = SWEEP / f"{ARC_TO_RUN[arc]}_s{SEED}"
    imgs = sorted((run / "archive" / "images").glob("*.png"))
    z = np.load(run / "embeddings_openclip_SigLIP2-B-alignet.npz", allow_pickle=True)
    emb = np.asarray(z["embeddings"])
    n = min(LIMIT, len(imgs), len(emb))
    imgs, emb = imgs[:n], emb[:n]
    coords = reduce_embeddings(emb)               # umap, random_state=0
    centers, _ = _greedy_k_center(emb, K)         # farthest-point / greedy k-center
    rep_coords = coords[centers]
    rep_paths = [imgs[i] for i in centers]

    tmp = Path(f"/tmp/grid_ai_{arc}.png")
    positions, dims = render_rectangular_grid(rep_coords, rep_paths, tmp, return_positions=True)
    name = "grid_ai_representative.png" if arc == "default" else f"grid_ai_{arc}.png"
    out = ASSETS / name
    Image.open(tmp).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS).save(out, optimize=True)
    print(f"[{arc}] {run.name}  ({n} imgs) -> {out}")
    # The clickable AI thumbnail is the 'default' arc; only it needs a focus manifest.
    if arc == "default":
        focus = [_pub_index(p) for p in rep_paths]
        _write_focus_manifest("ai", "ai", "default", SEED, positions, dims, focus)
    return out


def build_human() -> Path:
    """Regenerate grid_human_representative.png + its focus manifest from the cached
    human SigLIP embeddings (data/human_baseline/res-128/limit-2000), mirroring the AI
    pipeline. Focus indices are positions in the gallery's human image order
    (build_human_sprites.human_ids()), so a click parks the loupe on the same pid."""
    from tools.build_human_sprites import human_ids
    npz = REPO / "data" / "human_baseline" / "res-128" / f"limit-{LIMIT}" / "embeddings_openclip_SigLIP2-B-alignet.npz"
    z = np.load(npz, allow_pickle=True)
    emb = np.asarray(z["embeddings"])
    pids = [Path(str(fn)).stem for fn in z["filenames"]]   # <pid>.png -> pid
    thumbs = REPO / "fer/src" / "archive_res-128" / "images"
    coords = reduce_embeddings(emb)
    centers, _ = _greedy_k_center(emb, K)
    rep_coords = coords[centers]
    rep_paths = [thumbs / f"{pids[i]}.png" for i in centers]

    tmp = Path("/tmp/grid_human.png")
    positions, dims = render_rectangular_grid(rep_coords, rep_paths, tmp, return_positions=True)
    out = ASSETS / "grid_human_representative.png"
    Image.open(tmp).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS).save(out, optimize=True)
    print(f"[human] {npz.parent.name}  ({len(pids)} imgs) -> {out}")
    # Bridge each rep pid to its index in the gallery's human image order.
    gallery_idx = {pid: i for i, pid in enumerate(human_ids())}
    missing = [pids[i] for i in centers if pids[i] not in gallery_idx]
    if missing:
        print(f"  [!] {len(missing)} rep pids absent from gallery order (no jump for those cells): {missing[:8]}")
    focus = [gallery_idx.get(pids[i], -1) for i in centers]
    _write_focus_manifest("human", "human", "human", None, positions, dims, focus)
    return out


def main() -> None:
    args = sys.argv[1:] or (DEFAULT_ARCS + ["human"])
    for arc in args:
        if arc == "human":
            build_human()
            continue
        if arc not in ARC_TO_RUN:
            raise SystemExit(f"unknown arc {arc!r}; known: {sorted(ARC_TO_RUN)} (+ 'human')")
        build(arc)


if __name__ == "__main__":
    main()
