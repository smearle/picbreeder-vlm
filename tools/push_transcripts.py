#!/usr/bin/env python3
"""Push per-run transcript bundles (built by tools/build_transcript_data.py) to the
HF dataset under ``transcripts/<run>/`` and regenerate the blog's run manifest.

The blog keeps only ONE demo run git-committed under
``<blog>/assets/transcripts/<demo>/``; every other run is built into a staging dir
and lives on HF only (fetched on demand by the viewer, honouring ``?archiveBase=``
for the private repo — same pattern as the archive gallery / morph viewer).

``manifest.json`` (committed, small) is the viewer's run list: the demo run
(``hf`` unset → loads locally) plus every staged/HF run (``hf:true`` → loads from
HF). Run it after building/extending bundles:

    .venv/bin/python tools/push_transcripts.py --stage _transcripts_stage --all
    .venv/bin/python tools/push_transcripts.py --stage _transcripts_stage --only-run <run>
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "picbreeder-vlm/picbreeder-vlm-archive"
BLOG = Path.home() / "smearle.github.io" / "picbreeder-vlm-06b0d76d"
BLOG_TX = BLOG / "assets" / "transcripts"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.hf_archive_push import parse_config, canonical_arc  # noqa: E402

ARC_LABEL = {
    "default": "Default", "noise_0.05": "Noise 0.05", "noise_0.25": "Noise 0.25",
    "noise_0.5": "Noise 0.5", "noise_0.75": "Noise 0.75", "noise_1.0": "Noise 1.0",
    "mem_0": "Memory 0", "mem_2": "Memory 2", "mem_10": "Memory 10", "mem_20": "Memory 20",
    "agents_10": "Agents 10", "agents_100": "Agents 100", "agents_1000": "Agents 1000",
    "random": "Random",
}


def label_for(run: str, idx: dict | None) -> str:
    cfg = parse_config(run)
    arc = canonical_arc(run, cfg)
    parts = [ARC_LABEL.get(arc, arc or run)]
    if cfg.get("model"):
        parts.append(cfg["model"])
    if cfg.get("seed") is not None:
        parts.append("seed " + str(cfg["seed"]))
    base = " · ".join(parts)
    if idx:
        n, tot = idx.get("n_agents"), idx.get("total_agents")
        if n and tot and tot > n:
            base += " (" + str(n) + " of " + str(tot) + ")"
    return base


def run_index(run_dir: Path) -> dict | None:
    p = run_dir / "index.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def regenerate_manifest(stage: Path, demo_run: str) -> None:
    """manifest.json = demo run (local) + every staged run (hf:true), arc-sorted."""
    runs = []
    # local demo run (must exist under the committed blog dir)
    demo_dir = BLOG_TX / demo_run
    didx = run_index(demo_dir)
    if didx is not None:
        runs.append({"run": demo_run, "label": label_for(demo_run, didx) + " — demo",
                     "arc": didx.get("arc"), "model": didx.get("model"),
                     "seed": didx.get("seed")})
    # staged / HF runs
    seen = {demo_run}
    for rd in sorted(p for p in stage.iterdir() if p.is_dir()):
        if rd.name in seen:
            continue
        idx = run_index(rd)
        if idx is None:
            continue
        seen.add(rd.name)
        runs.append({"run": rd.name, "label": label_for(rd.name, idx), "hf": True,
                     "arc": idx.get("arc"), "model": idx.get("model"), "seed": idx.get("seed")})
    BLOG_TX.mkdir(parents=True, exist_ok=True)
    (BLOG_TX / "manifest.json").write_text(json.dumps({"runs": runs}, indent=1))
    print(f"[manifest] {len(runs)} runs -> {BLOG_TX/'manifest.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, default=Path("_transcripts_stage"),
                    help="dir holding built per-run bundles to push (default _transcripts_stage)")
    ap.add_argument("--only-run", default="", help="push just this run dir name")
    ap.add_argument("--all", action="store_true", help="push every run dir in --stage")
    ap.add_argument("--demo-run",
                    default="th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s3",
                    help="run kept git-committed under the blog (local), not pushed")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a run whose transcripts/<run>/index.json already exists on HF")
    ap.add_argument("--no-manifest", action="store_true", help="don't regenerate manifest.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stage = args.stage
    if not stage.exists():
        raise SystemExit(f"staging dir not found: {stage}")

    in_progress = ""
    cur = stage / ".current"
    if cur.exists():
        in_progress = cur.read_text().strip()

    if args.only_run:
        run_dirs = [stage / args.only_run]
    elif args.all:
        run_dirs = sorted(p for p in stage.iterdir()
                          if p.is_dir() and (p / "index.json").exists() and p.name != in_progress)
        if in_progress:
            print(f"[note] skipping in-progress run {in_progress}")
    else:
        raise SystemExit("pass --all or --only-run <run>")

    api = HfApi()
    pushed = 0
    for rd in run_dirs:
        run = rd.name
        if not (rd / "index.json").exists():
            print(f"[skip] {run} (no index.json)")
            continue
        sentinel = f"transcripts/{run}/index.json"
        if args.skip_existing and api.file_exists(repo_id=REPO, filename=sentinel, repo_type="dataset"):
            print(f"[skip] {run} (already on HF)")
            continue
        idx = run_index(rd)
        size_mb = sum(p.stat().st_size for p in rd.rglob("*") if p.is_file()) / 1e6
        if args.dry_run:
            print(f"[dry] would push {run}  ({idx.get('n_agents')} agents, {size_mb:.1f} MB)")
            continue
        api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(rd),
                          path_in_repo=f"transcripts/{run}",
                          commit_message=f"Add transcripts for {run} ({idx.get('n_agents')} agents)")
        print(f"[done] {run}  ({idx.get('n_agents')} agents, {size_mb:.1f} MB)")
        pushed += 1

    if not args.no_manifest and not args.dry_run:
        regenerate_manifest(stage, args.demo_run)
    print(f"\npushed {pushed} run(s)")


if __name__ == "__main__":
    main()
