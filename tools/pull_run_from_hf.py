#!/usr/bin/env python3
"""Reconstruct runnable local run directories from the HF archive dataset.

The published dataset `picbreeder-vlm/picbreeder-vlm-archive` stores each run's
archive in a *flattened, packed* form under ``results/<run>/`` -- genomes packed
into ``genomes.tar.gz`` (131k loose ``.pkl`` would wreck the repo tree), agent
logs in ``agents.tar``, cached embeddings in an ``.npz``, and the curated result
JSONs promoted to the run root. Crucially it ships **no rendered images** (they
regenerate deterministically from the genomes) and the stored ``image_path`` /
``genome_path`` values are absolute paths from the cluster the run was produced
on, so they do not resolve elsewhere.

This tool downloads one or more runs and rebuilds the on-disk layout that the
evaluation / figure / resume code expects::

    <dest>/<run>/
      data_manifest.json
      embeddings_openclip_*.npz              # coverage-eval embedding cache (run root)
      archive/
        archive_metadata.json                # image_path/genome_path rewritten to here
        genomes/img_*.pkl                    # extracted from genomes.tar.gz
        images/img_*.png                     # rendered from the genomes (variant=auto)
        archive_grid.json, captions_*.json, phylogeny_metrics.json, ...
      agents/agent_*.zip                     # extracted from agents.tar (--with-agents)

By default ``<dest>`` is ``sweep_logs/sweep`` -- the sweep entry point's default
``log_dir`` -- so the runs land exactly where ``picbreeder_vlm.experiments.sweep``
(eval/cross_eval) looks for them, and where you point ``evolve_collaborative.py
experiment_dir=<...> resume=true`` to continue a run.

Usage
-----
    # one or more explicit runs
    PYTHONPATH=. python tools/pull_run_from_hf.py <run_name> [<run_name> ...]

    # every run in the dataset
    PYTHONPATH=. python tools/pull_run_from_hf.py --all

    # skip image rendering (metadata/eval-cache only) or skip agent logs
    PYTHONPATH=. python tools/pull_run_from_hf.py <run> --no-images --with-agents

Run from the repo root so the genome classes (neat_components) import for
unpickling and rendering.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Iterable, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from huggingface_hub import HfApi, snapshot_download

# Register genome classes for unpickling, and pull in the faithful renderer.
from picbreeder_vlm.core import neat_components  # noqa: F401
from picbreeder_vlm._paths import NEAT_CONFIG_PATH
from picbreeder_vlm.viz.render_archive import load_config, render_entry_image

DATASET = "picbreeder-vlm/picbreeder-vlm-archive"

# HF-flat filename -> where it belongs in a reconstructed run dir.
#   "archive"  -> <run>/archive/<name>
#   "root"     -> <run>/<name>
_ROOT_FILES = {"data_manifest.json"}
_ROOT_PREFIXES = ("embeddings_",)  # npz embedding caches live at the run root


def list_dataset_runs(token: Optional[str]) -> List[str]:
    api = HfApi(token=token)
    info = api.dataset_info(DATASET, files_metadata=False)
    runs = sorted({
        s.rfilename.split("/")[1]
        for s in (info.siblings or [])
        if s.rfilename.startswith("results/") and len(s.rfilename.split("/")) > 2
    })
    return runs


def _place(name: str) -> str:
    """Return 'root' or 'archive' for a flat result file."""
    if name in _ROOT_FILES or any(name.startswith(p) for p in _ROOT_PREFIXES):
        return "root"
    return "archive"


def _prune_dangling(archive_dir: Path, require_images: bool = False) -> int:
    """Drop metadata entries whose genome .pkl wasn't shipped, keeping the archive
    self-consistent. Many runs ship only the first ~1003 genomes (the genome sync
    gap), yet archive_metadata.json lists every published entry. Downstream code
    (create_archive_grid on publish, image-order inference, resume/continue) assumes
    every listed entry has its files, so a partial run otherwise crashes. next_id is
    left untouched so genuinely new entries still get fresh, non-colliding ids.
    Returns the number of entries dropped.
    """
    meta_path = archive_dir / "archive_metadata.json"
    if not meta_path.exists():
        return 0
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    genomes_dir = archive_dir / "genomes"
    images_dir = archive_dir / "images"
    entries = data.get("entries", [])

    def has_files(entry) -> bool:
        eid = str(entry.get("id") or "").strip()
        if not (genomes_dir / f"{eid}.pkl").exists():
            return False
        # If images were rendered, require the image too: create_archive_grid (run on
        # every publish) and image-order inference both read archive/images/<id>.png,
        # so an entry with a genome but a missing/failed render still breaks a continue.
        if require_images and not (images_dir / f"{eid}.png").exists():
            return False
        return True

    kept = [e for e in entries if has_files(e)]
    dropped = len(entries) - len(kept)
    if dropped:
        data["entries"] = kept
        meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return dropped


def _rewrite_metadata_paths(archive_dir: Path) -> int:
    """Point every entry's image_path/genome_path at this reconstructed archive.

    The dataset stores absolute cluster paths; ``load_genome`` (used by resume /
    branching) reads ``genome_path`` directly with no fallback, so without this
    a reconstructed run cannot be resumed. Returns the number of entries fixed.
    """
    meta_path = archive_dir / "archive_metadata.json"
    if not meta_path.exists():
        return 0
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    images_dir = (archive_dir / "images").resolve()
    genomes_dir = (archive_dir / "genomes").resolve()
    n = 0
    for entry in data.get("entries", []):
        entry_id = str(entry.get("id") or "").strip()
        if not entry_id:
            continue
        entry["image_path"] = str(images_dir / f"{entry_id}.png")
        entry["genome_path"] = str(genomes_dir / f"{entry_id}.pkl")
        n += 1
    meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return n


def _extract_tar(tar_path: Path, dest: Path, gzip: bool) -> int:
    if not tar_path.exists():
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    mode = "r:gz" if gzip else "r"
    with tarfile.open(tar_path, mode) as tf:
        members = tf.getmembers()
        tf.extractall(dest)
    return len(members)


def reconstruct_run(
    run: str,
    src_dir: Path,
    dest_root: Path,
    render_images: bool = True,
    with_agents: bool = False,
    overwrite: bool = False,
    prune_dangling: bool = True,
) -> Path:
    """Build <dest_root>/<run>/ from a downloaded results/<run>/ directory."""
    run_dir = dest_root / run
    archive_dir = run_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "genomes").mkdir(exist_ok=True)
    (archive_dir / "images").mkdir(exist_ok=True)

    # 1. Un-flatten the loose result files.
    for f in sorted(src_dir.iterdir()):
        if f.is_dir():
            continue
        name = f.name
        if name in ("genomes.tar.gz", "agents.tar"):
            continue
        target = (run_dir if _place(name) == "root" else archive_dir) / name
        shutil.copy2(f, target)

    # 2. Extract genomes into archive/genomes/.
    n_gen = _extract_tar(src_dir / "genomes.tar.gz", archive_dir / "genomes", gzip=True)
    print(f"  [{run}] genomes extracted: {n_gen}")

    # 3. Optionally extract agent logs into agents/.
    if with_agents:
        n_ag = _extract_tar(src_dir / "agents.tar", run_dir / "agents", gzip=False)
        print(f"  [{run}] agent zips extracted: {n_ag}")

    # 4. Rewrite the metadata paths to point here (needed for resume/branching).
    n_fixed = _rewrite_metadata_paths(archive_dir)
    print(f"  [{run}] metadata entries repathed: {n_fixed}")

    # 5. Render images from the genomes (faithful variant=auto: per-entry color).
    if render_images:
        meta = archive_dir / "archive_metadata.json"
        if not meta.exists():
            print(f"  [{run}] no archive_metadata.json; skipping image render")
        else:
            data = json.loads(meta.read_text(encoding="utf-8"))
            config = load_config(NEAT_CONFIG_PATH)
            images_dir = archive_dir / "images"
            rendered = skipped = 0
            for entry in data.get("entries", []):
                out = render_entry_image(
                    entry, config, image_size=128, archive_dir=archive_dir,
                    output_dir=images_dir, variant_mode="auto", overwrite=overwrite,
                )
                if out is None:
                    skipped += 1
                else:
                    rendered += 1
            print(f"  [{run}] images rendered: {rendered} (skipped/missing genome: {skipped})")

    # 6. Prune metadata entries that lack their files, so the archive is self-consistent
    #    (avoids missing-file crashes in eval / archive-grid / continue). Requires the
    #    rendered image too when images were rendered, so a partial render is also caught.
    if prune_dangling:
        n_dropped = _prune_dangling(archive_dir, require_images=render_images)
        if n_dropped:
            print(f"  [{run}] pruned dangling entries (missing genome/image): {n_dropped}")

    return run_dir


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="run name(s) to reconstruct (dirs under results/ in the dataset)")
    p.add_argument("--all", action="store_true", help="reconstruct every run in the dataset")
    p.add_argument("--dest", default=str(REPO / "sweep_logs" / "sweep"),
                   help="destination sweep dir (default: sweep_logs/sweep, the sweep entry point's default "
                        "log_dir, so `-m picbreeder_vlm.experiments.sweep ... eval_*=true` finds the runs)")
    p.add_argument("--no-images", dest="render_images", action="store_false",
                   help="do not render images from genomes (metadata + eval cache only)")
    p.add_argument("--with-agents", action="store_true", help="also extract agents.tar into <run>/agents/")
    p.add_argument("--keep-dangling", dest="prune_dangling", action="store_false",
                   help="keep archive_metadata entries whose genome wasn't shipped (default: prune them "
                        "so the archive is self-consistent for eval/continue)")
    p.add_argument("--overwrite", action="store_true", help="re-render images even if they already exist")
    p.add_argument("--cache-dir", default=str(REPO / ".hf_pull_cache"),
                   help="where to download the raw results/<run>/ files before reconstruction")
    args = p.parse_args(argv)

    import os
    token = os.environ.get("HF_TOKEN") or None

    if args.all:
        runs = list_dataset_runs(token)
    else:
        runs = list(args.runs)
    if not runs:
        p.error("give one or more run names, or --all")

    print(f"Reconstructing {len(runs)} run(s) into {args.dest}")
    patterns = [f"results/{r}/*" for r in runs]
    cache = snapshot_download(
        DATASET, repo_type="dataset", allow_patterns=patterns,
        local_dir=args.cache_dir, token=token,
    )
    src_results = Path(cache) / "results"
    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)

    done = []
    for run in runs:
        src = src_results / run
        if not src.is_dir():
            print(f"  [{run}] WARNING: not found in dataset; skipping")
            continue
        run_dir = reconstruct_run(
            run, src, dest_root,
            render_images=args.render_images, with_agents=args.with_agents,
            overwrite=args.overwrite, prune_dangling=args.prune_dangling,
        )
        done.append(run_dir)

    print(f"\nDone. Reconstructed {len(done)} run(s):")
    for d in done:
        print(f"  {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
