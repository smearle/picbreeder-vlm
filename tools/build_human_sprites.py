#!/usr/bin/env python
"""Build the human Picbreeder archive gallery (sprite sets + layout.json) from the pre-rendered
64px thumbs + the already-computed SigLIP embeddings. Human genomes are NOT published (license);
this is view-only render assets. Staged into the local mirror for testing; gh-pages is the prod host.

Build set = the images that have BOTH a thumb (<id>_final.png) AND an embedding (<id>.png row),
so chronological (sort by id) and SigLIP (UMAP+RasterFairy on embeddings) index the same set.

Lineage orderings (Most Branched + Phylogeny) are derived from each image's picbreeder
parent (branchFrom in main.zip), so the human archive offers the same lineage views as the
VLM runs. There are no VLM ratings for human-bred images, so no "Top Rated" ordering is
emitted (the gallery greys that button out for runs that lack it).

  python tools/build_human_sprites.py                 # full rebuild (sheets + all layouts)
  python tools/build_human_sprites.py --layouts-only  # only re-add branched+phylogeny in place
"""
import importlib.util
import json, math, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

REPO = Path("/home/jupyter-smearle/picbreeder-vlm")
THUMBS = REPO / "human_lineages/lineages/lineage_phylogeny_thumbs_128"   # 128px to match the VLM runs
EMB = REPO / "data/human_baseline/res-128/embeddings_openclip_SigLIP2-B-alignet.npz"
SITE = REPO / "archive_animations/_archive_mirror/site"
PB = REPO / "fer/spaghetti/pbRender/genomeAll"                    # per-pid main.zip (carries branchFrom)
RUN, CELL = "human", 128


def _load_balib():
    """Import the lineage-layout generators from build_archive_image_lib without running it.

    balib imports umap at module load only for its SigLIP/UMAP path, which we don't use here;
    stub it out if it can't import (numpy/scipy ABI drift in some envs) so the pure-python
    lineage generators we DO need still load."""
    import types
    try:
        import umap  # noqa: F401
    except Exception:
        sys.modules.setdefault("umap", types.ModuleType("umap"))
    spec = importlib.util.spec_from_file_location(
        "balib", REPO / "tools" / "build_archive_image_lib.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def human_ids():
    """Gallery image order: human pids that have BOTH a 128px thumb and a SigLIP row,
    sorted by integer pid (≈ publication order — picbreeder assigns ids incrementally).
    Image i of the gallery (sprite img_{i+1:06d}) is human pid ids[i]."""
    d = np.load(EMB, allow_pickle=True)
    row_of = {str(fn)[:-4] for fn in d["filenames"]}
    return sorted((i for i in row_of if (THUMBS / f"{i}_final.png").is_file()), key=int)


def _parent_pid(pid):
    """Picbreeder parent of an image: the branchFrom @branch recorded in its main.zip.
    Lightweight read (no genome render); returns the parent pid string or None."""
    sys.path.append(str(REPO))
    from fer.src.picbreeder_util import load_zip_xml_as_dict
    mz = PB / pid / "main.zip"
    if not mz.exists():
        return None
    try:
        d = load_zip_xml_as_dict(str(mz))
    except Exception:
        return None
    g = d.get("genome", {}) or {}
    s = g.get("series", {}) or {}
    if not s and "generation" in g:
        s = g
    bf = s.get("branchFrom")
    return bf.get("@branch") if isinstance(bf, dict) else None


def human_lineage_entries(ids):
    """Build archive-style entries (id + source_entry_ids) aligned to the gallery order,
    so the shared branched/phylogeny layout generators can consume them. A parent edge is
    kept only when the parent is itself in the gallery (an in-archive published image)."""
    idx_of = {pid: i for i, pid in enumerate(ids)}
    entries = []
    for i, pid in enumerate(ids):
        p = _parent_pid(pid)
        src = [f"img_{idx_of[p] + 1:06d}"] if (p is not None and p in idx_of) else []
        entries.append({"id": f"img_{i + 1:06d}", "source_entry_ids": src})
    return entries


def add_human_lineage_layouts(out):
    """Merge Most-Branched + Phylogeny orderings into an already-built human layout.json
    (in place). No ratings ordering — human-bred images carry no VLM ratings."""
    balib = _load_balib()
    layout_path = out / "layout.json"
    layout = json.loads(layout_path.read_text())
    n = int(layout["n"])
    cols = int(layout["layouts"]["chronological"]["dims"][1])
    ids = human_ids()
    if len(ids) != n:
        raise SystemExit(f"id list ({len(ids)}) != layout n ({n}); rebuild sprites first")
    entries = human_lineage_entries(ids)
    n_edges = sum(1 for e in entries if e["source_entry_ids"])
    layout["layouts"]["branched"] = balib.branched_layout(entries, n, cols)
    layout["layouts"]["phylogeny"] = balib.phylogeny_layout(entries, n)
    layout_path.write_text(json.dumps(layout))
    variants = list(layout["layouts"]["phylogeny"].get("variants", {}))
    print(f"human lineage layouts: +branched,phylogeny ({n_edges}/{n} in-archive parent edges; "
          f"phylogeny variants {variants})")


def main():
    if "--layouts-only" in sys.argv:
        add_human_lineage_layouts(SITE / RUN / "sprite")
        return
    import umap, rasterfairy
    # rasterfairy (old) uses the NumPy-2.0-removed aliases np.float/np.int/np.bool; restore them
    for _a, _t in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, _a):
            setattr(np, _a, _t)
    d = np.load(EMB, allow_pickle=True)
    fns = [str(x) for x in d["filenames"]]
    emb = np.asarray(d["embeddings"], dtype=np.float32)
    row_of = {fn[:-4]: i for i, fn in enumerate(fns)}          # '340.png' -> row; key '340'
    ids = sorted((i for i in row_of if (THUMBS / f"{i}_final.png").is_file()), key=int)
    n = len(ids)
    print(f"{n} human images with both thumb + embedding (of {len(fns)} emb, {len(list(THUMBS.glob('*_final.png')))} thumbs)")
    embo = emb[[row_of[i] for i in ids]]

    cols = math.ceil(math.sqrt(n)); rows = math.ceil(n / cols)
    chrono_rc = [[i // cols, i % cols] for i in range(n)]

    print(f"UMAP on {embo.shape} ...")
    coords = umap.UMAP(n_components=2, random_state=0).fit_transform(embo)
    # pass an explicit (width,height) target: rasterfairy's auto-arrangement path is buggy
    # (NameError: prime) for some counts like 9375. cols*rows >= n, trailing cells stay empty.
    gp, dims = rasterfairy.transformPointCloud2D(np.asarray(coords, dtype=float), target=(cols, rows))
    gp = np.rint(np.asarray(gp, dtype=float)).astype(int)
    sc, sr = int(dims[0]), int(dims[1])
    siglip_rc = [[int(r), int(c)] for c, r in gp.tolist()]

    tmp = tempfile.mkdtemp()
    for i, idi in enumerate(ids):
        shutil.copy(THUMBS / f"{idi}_final.png", Path(tmp) / f"img_{i+1:06d}.png")
    out = SITE / RUN / "sprite"
    subprocess.run([sys.executable, str(REPO / "archive_animations/make_sprite_sheets.py"),
                    "--thumbs", tmp, "--out", str(out), "--cell", str(CELL), "--ext", "webp"], check=True)
    shutil.rmtree(tmp)

    layout = {"arc": "human", "seed": None, "n": n, "thumb_px": CELL,
              "filename_template": "img_{i06}.png",
              "layouts": {"chronological": {"dims": [rows, cols], "rc": chrono_rc},
                          "siglip": {"dims": [sr, sc], "rc": siglip_rc}}}
    (out / "layout.json").write_text(json.dumps(layout))
    add_human_lineage_layouts(out)   # +branched,phylogeny (chronological/siglip already written)

    idxp = SITE.parent / "index.json"
    idx = json.load(open(idxp))
    idx["runs"] = [r for r in idx["runs"] if r["run"] != RUN]
    idx["runs"].append({"run": RUN, "arc": "human", "seed": None,
                        "label": "Human Picbreeder archive", "n_images": n,
                        "has": {"sprite": True}})   # no `base` => local; prod HF index sets base=gh-pages
    idx["n_runs"] = len(idx["runs"])
    json.dump(idx, open(idxp, "w"), indent=1)
    print(f"human gallery: {n} imgs -> {out}")


if __name__ == "__main__":
    main()
