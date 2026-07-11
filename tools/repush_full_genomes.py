#!/usr/bin/env python3
"""Re-push full genomes + metadata + alignet embeddings for truncated HF runs.

Background: a handful of flagship runs were rsync'd off the cluster with only the
first ~1003 genomes (a known local-sync cap), and those truncated copies are what
reached the public HF dataset. Torch (`ssh torch`) holds the full 2000-3400-genome
archives. This tool takes the full `archive/genomes/` + `archive_metadata.json`
(already staged locally from torch) and, per run:

  1. re-renders every genome to archive/images/ (canonical variant=auto render,
     the same path pull_run_from_hf uses -- the dataset ships no images);
  2. embeds the images with the SigLIP2-B-alignet TF SavedModel, writing
     `embeddings_openclip_SigLIP2-B-alignet.npz` byte-for-byte the way
     embed_and_visualize does (filenames + L2-normalized embeddings);
  3. tars the genomes to `genomes.tar.gz`;
  4. uploads genomes.tar.gz + archive_metadata.json + the npz to the HF dataset,
     replacing the truncated versions.

The alignet SavedModel and NEAT config are loaded once and reused across runs.

Usage:
  # stage <run>/archive/{genomes,archive_metadata.json} from torch first, then:
  .venv/bin/python tools/repush_full_genomes.py --staging <dir> --run <name> [...] [--push]
  .venv/bin/python tools/repush_full_genomes.py --staging <dir> --runs-file <file> [--push]
Without --push it renders + embeds + tars (a dry run you can inspect) but uploads nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATASET = "picbreeder-vlm/picbreeder-vlm-archive"
EMB_MODEL = "SigLIP2-B-alignet"
EMB_NPZ = f"embeddings_openclip_{EMB_MODEL}.npz"


# --- parallel render: each worker loads the NEAT config once (initializer) and
#     renders a shard of entries. render_entry_image is CPU-bound (~2.6/s serial),
#     so 36 cores take ~9h down to ~15min.
_W = {}


def _render_init(archive_dir_str, images_dir_str):
    from picbreeder_vlm.viz.render_archive import load_config
    from picbreeder_vlm._paths import NEAT_CONFIG_PATH
    _W["config"] = load_config(NEAT_CONFIG_PATH)
    _W["archive_dir"] = Path(archive_dir_str)
    _W["images_dir"] = Path(images_dir_str)


def _render_one(entry, overwrite=False):
    from picbreeder_vlm.viz.render_archive import render_entry_image
    out = render_entry_image(
        entry, _W["config"], image_size=128, archive_dir=_W["archive_dir"],
        output_dir=_W["images_dir"], variant_mode="auto", overwrite=overwrite,
    )
    return out is not None


def render_run_images(run_dir: Path, config=None, overwrite: bool = False, workers: int = 32) -> int:
    import multiprocessing as mp
    from functools import partial
    archive_dir = run_dir / "archive"
    data = json.loads((archive_dir / "archive_metadata.json").read_text(encoding="utf-8"))
    images_dir = archive_dir / "images"
    images_dir.mkdir(exist_ok=True)
    entries = data.get("entries", [])
    rendered = 0
    # spawn (not fork): the parent has the alignet TF/CUDA context loaded, which does
    # not survive fork cleanly; spawned workers start fresh and only touch CPU rendering.
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=workers,
        initializer=_render_init,
        initargs=(str(archive_dir), str(images_dir)),
    ) as pool:
        for ok in pool.imap_unordered(partial(_render_one, overwrite=overwrite), entries, chunksize=16):
            rendered += int(ok)
    return rendered


def embed_run(run_dir: Path, model, preprocess, device, batch_size=256) -> int:
    """Write embeddings_openclip_SigLIP2-B-alignet.npz for all rendered images."""
    from picbreeder_vlm.vlm.model_loader import embed_images
    images_dir = run_dir / "archive" / "images"
    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise SystemExit(f"no images in {images_dir}")
    filenames, embeddings = embed_images(model, preprocess, image_paths, device, batch_size=batch_size)
    emb_out = run_dir / EMB_NPZ
    np.savez_compressed(emb_out, filenames=np.array(filenames), embeddings=embeddings)
    return len(filenames)


def tar_genomes(run_dir: Path) -> Path:
    genomes = run_dir / "archive" / "genomes"
    out = run_dir / "genomes.tar.gz"
    with tarfile.open(out, "w:gz") as t:
        # arcname "genomes/<id>.pkl" -- matches the dataset's existing layout
        t.add(genomes, arcname="genomes")
    return out


def push_run(run: str, run_dir: Path, api) -> None:
    from huggingface_hub import CommitOperationAdd
    ops = [
        CommitOperationAdd(f"results/{run}/genomes.tar.gz", str(run_dir / "genomes.tar.gz")),
        CommitOperationAdd(f"results/{run}/archive_metadata.json", str(run_dir / "archive" / "archive_metadata.json")),
        CommitOperationAdd(f"results/{run}/{EMB_NPZ}", str(run_dir / EMB_NPZ)),
    ]
    api.create_commit(
        repo_id=DATASET, repo_type="dataset", operations=ops,
        commit_message=f"Re-push full genomes+metadata+embeddings for {run} (was truncated at local-sync cap)",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", required=True, type=Path, help="root holding <run>/archive/{genomes,archive_metadata.json}")
    ap.add_argument("--run", action="append", default=[], dest="runs")
    ap.add_argument("--runs-file", type=Path)
    ap.add_argument("--push", action="store_true", help="upload to HF (otherwise dry: render+embed+tar only)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=32, help="parallel render workers")
    ap.add_argument("--overwrite-images", action="store_true")
    args = ap.parse_args()

    runs = list(args.runs)
    if args.runs_file:
        runs += [l.strip() for l in args.runs_file.read_text().splitlines() if l.strip()]
    if not runs:
        ap.error("no runs given")

    import torch
    from picbreeder_vlm.viz.render_archive import load_config
    from picbreeder_vlm._paths import NEAT_CONFIG_PATH
    from picbreeder_vlm.vlm.model_loader import load_model_by_name

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"device={device}  runs={len(runs)}  push={args.push}")

    config = load_config(NEAT_CONFIG_PATH)
    print(f"loading embedding model {EMB_MODEL} ...")
    model, preprocess, _ = load_model_by_name(EMB_MODEL, pretrained="webli", device=device)

    api = None
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()

    ok, failed = [], []
    for i, run in enumerate(runs, 1):
        run_dir = args.staging / run
        meta = run_dir / "archive" / "archive_metadata.json"
        try:
            n_entries = len(json.loads(meta.read_text())["entries"])
            n_gen = len(list((run_dir / "archive" / "genomes").glob("*.pkl")))
            print(f"\n[{i}/{len(runs)}] {run}  entries={n_entries} genomes={n_gen}")
            r = render_run_images(run_dir, overwrite=args.overwrite_images, workers=args.workers)
            print(f"  rendered {r} images")
            n = embed_run(run_dir, model, preprocess, device)
            print(f"  embedded {n} -> {EMB_NPZ}")
            tp = tar_genomes(run_dir)
            print(f"  tarred genomes -> {tp.name} ({tp.stat().st_size/1e6:.1f} MB)")
            if args.push:
                push_run(run, run_dir, api)
                print("  PUSHED to HF")
            ok.append(run)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAILED: {e}")
            failed.append((run, str(e)))

    print(f"\n=== done: {len(ok)} ok, {len(failed)} failed ===")
    for run, err in failed:
        print(f"  FAIL {run}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
