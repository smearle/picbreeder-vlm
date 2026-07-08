#!/usr/bin/env python3
"""Export per-run `genomes.json.gz` — the browser-renderable form of a run's genomes.

The HF push (`tools/hf_archive_push.py`) ships genomes as pickled `.pkl`
(`genomes.tar.gz`), which the static breed site can't read. This converts a run's
`archive/genomes/*.pkl` into a single gzipped JSON map `{img_id: genome_json}`
(schema = genome_json.genome_to_json, exactly what cppn.js renders), written to
`<run>/archive/genomes.json.gz`. The site then lazily fetches that file per run
(joined with the already-pushed archive_metadata.json for titles/ratings/agent),
so the whole published archive can be browsed/branched without bundling anything.

Usage:
  PYTHONPATH=. python tools/export_genome_json.py RUN_DIR [RUN_DIR ...]
  PYTHONPATH=. python tools/export_genome_json.py --all        # every complete run
  ... --force   # re-export even if genomes.json.gz exists and is newer
Run from the repo root so the genome classes (neat_components) import for unpickling.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
SWEEP = REPO / "sweep_logs" / "sweep"

from picbreeder_vlm.core import neat_components  # noqa: F401  (registers genome classes for unpickling)
from picbreeder_vlm.core.genome_json import genome_to_json


def export_run(run_dir: Path, force: bool = False) -> dict | None:
    gen_dir = run_dir / "archive" / "genomes"
    if not gen_dir.is_dir():
        return None
    out = run_dir / "archive" / "genomes.json.gz"
    pkls = sorted(p for p in gen_dir.iterdir() if p.suffix == ".pkl")
    if not pkls:
        return None
    if out.exists() and not force and out.stat().st_mtime >= max(p.stat().st_mtime for p in pkls):
        print(f"[skip] {run_dir.name} (up to date, {len(pkls)} genomes)")
        return {"run": run_dir.name, "n": len(pkls), "skipped": True}

    mapping: dict[str, dict] = {}
    bad = 0
    for p in pkls:
        try:
            with open(p, "rb") as fh:
                mapping[p.stem] = genome_to_json(pickle.load(fh))
        except Exception:
            bad += 1
    tmp = out.with_suffix(".gz.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as fh:
        json.dump(mapping, fh, separators=(",", ":"))
    tmp.replace(out)
    mb = out.stat().st_size / 1e6
    print(f"[done] {run_dir.name}: {len(mapping)} genomes -> {out.name} ({mb:.2f} MB)"
          + (f"  [{bad} unreadable]" if bad else ""))
    return {"run": run_dir.name, "n": len(mapping), "bad": bad, "mb": round(mb, 2)}


def complete_runs() -> list[Path]:
    out = []
    for d in sorted(SWEEP.iterdir()):
        g = d / "archive" / "genomes"
        if d.is_dir() and g.is_dir() and any(g.iterdir()):
            out.append(d)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directories (default: use --all)")
    ap.add_argument("--all", action="store_true", help="every complete run under sweep_logs/sweep")
    ap.add_argument("--force", action="store_true", help="re-export even if up to date")
    args = ap.parse_args(argv)

    runs = [Path(r) for r in args.runs]
    if args.all:
        runs = complete_runs()
    if not runs:
        ap.error("pass run dirs or --all")
    print(f"exporting genome JSON for {len(runs)} run(s)", file=sys.stderr)
    tot = 0
    for r in runs:
        info = export_run(r, force=args.force)
        if info:
            tot += info["n"]
    print(f"total genomes exported/checked: {tot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
