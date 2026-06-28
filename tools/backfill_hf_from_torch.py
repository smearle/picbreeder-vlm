#!/usr/bin/env python
"""Self-contained HF backfill, run ON torch (data is local there; only uploads).

For every COMPLETE run under sweep (non-empty archive/genomes + archive/images),
ensure the HF dataset has, under results/<run>/:
  - agents.tar       complete pack of agents/*.zip   (re-pushed if remote missing/smaller)
  - genomes.tar.gz   pack of archive/genomes/*.pkl    (pushed if remote missing)
  - <result>.json    curated metrics/metadata/configs (pushed if remote missing)
  - data_manifest.json

Idempotent + resumable: a per-run file is uploaded only if the remote copy is
missing or (for agents.tar) smaller than the local complete pack. No repo/neat
deps — just stdlib + huggingface_hub. Run with --dry to see the plan + sizes.

  .venv/bin/python tools/backfill_hf_from_torch.py --dry
  nohup .venv/bin/python tools/backfill_hf_from_torch.py > /scratch/se2161/hf_backfill.log 2>&1 &
"""
import os, sys, glob, json, io, tarfile, tempfile, fnmatch, shutil, time
from huggingface_hub import HfApi

REPO = "picbreeder-vlm/picbreeder-vlm-archive"
SWEEP = "/scratch/se2161/picbreeder-vlm/sweep_logs/sweep"
TMPROOT = "/scratch/se2161/tmp_hf_backfill"
DRY = "--dry" in sys.argv
RESULT_GLOBS = ["archive_metadata.json", "captions_*.json", "metrics_*.json",
                "phylogeny_metrics.json", "leaf_agent_lineages.json", "archive_grid.json"]
api = HfApi()


def complete_runs():
    out = []
    for d in sorted(os.listdir(SWEEP)):
        g = os.path.join(SWEEP, d, "archive", "genomes")
        im = os.path.join(SWEEP, d, "archive", "images")
        if os.path.isdir(g) and os.path.isdir(im) and os.listdir(g) and os.listdir(im):
            out.append(d)
    return out


def remote_sizes():
    """One recursive listing -> {path: size}."""
    m = {}
    for it in api.list_repo_tree(repo_id=REPO, repo_type="dataset", recursive=True):
        size = getattr(it, "size", None)
        if size is not None:
            m[it.path] = size
    return m


def results_for(run):
    ad = os.path.join(SWEEP, run, "archive")
    out = []
    for f in sorted(os.listdir(ad)):
        fp = os.path.join(ad, f)
        if (os.path.isfile(fp) and f.endswith(".json")
                and not any(x in f.lower() for x in ("embedding", "cache"))
                and any(fnmatch.fnmatch(f, g) for g in RESULT_GLOBS)):
            out.append(fp)
    return out


def main():
    runs = complete_runs()
    print(f"complete runs on torch: {len(runs)}")
    print("fetching remote inventory…")
    rem = remote_sizes()
    print(f"remote files: {len(rem)}")

    plan_bytes = 0
    todo = []
    for run in runs:
        need = []
        zips = sorted(glob.glob(os.path.join(SWEEP, run, "agents", "*.zip")))
        zbytes = sum(os.path.getsize(z) for z in zips)
        ra = rem.get(f"results/{run}/agents.tar")
        if zips and (ra is None or ra < zbytes * 0.99):
            need.append(("agents.tar", zbytes, "missing" if ra is None else f"partial {ra/1e6:.0f}/{zbytes/1e6:.0f}MB"))
        if f"results/{run}/genomes.tar.gz" not in rem:
            gbytes = sum(os.path.getsize(p) for p in glob.glob(os.path.join(SWEEP, run, "archive", "genomes", "*")))
            need.append(("genomes.tar.gz", gbytes, "missing"))
        for r in results_for(run):
            if f"results/{run}/{os.path.basename(r)}" not in rem:
                need.append((os.path.basename(r), os.path.getsize(r), "missing"))
        if f"results/{run}/data_manifest.json" not in rem:
            need.append(("data_manifest.json", 0, "missing"))
        if need:
            todo.append((run, need))
            plan_bytes += sum(n[1] for n in need)

    print(f"\nruns needing work: {len(todo)}   approx upload: {plan_bytes/1e9:.1f} GB")
    for run, need in todo:
        tags = ", ".join(f"{n[0]}({n[2]})" for n in need)
        print(f"  {run[:62]:62s} {tags}")
    if DRY:
        print("\n[dry run] nothing uploaded.")
        return

    os.makedirs(TMPROOT, exist_ok=True)
    done = 0
    for run, need in todo:
        names = {n[0] for n in need}
        stage = tempfile.mkdtemp(dir=TMPROOT)
        try:
            if "agents.tar" in names:
                zips = sorted(glob.glob(os.path.join(SWEEP, run, "agents", "*.zip")))
                with tarfile.open(os.path.join(stage, "agents.tar"), "w") as tf:
                    for p in zips:
                        tf.add(p, arcname=os.path.basename(p))
            if "genomes.tar.gz" in names:
                with tarfile.open(os.path.join(stage, "genomes.tar.gz"), "w:gz") as tf:
                    for p in sorted(glob.glob(os.path.join(SWEEP, run, "archive", "genomes", "*"))):
                        if os.path.isfile(p):
                            tf.add(p, arcname=os.path.basename(p))
            for r in results_for(run):
                if os.path.basename(r) in names:
                    shutil.copy(r, os.path.join(stage, os.path.basename(r)))
            # always (re)write manifest when touching a run
            man = {"run": run, "n_genomes": len(os.listdir(os.path.join(SWEEP, run, "archive", "genomes"))),
                   "n_images": len(os.listdir(os.path.join(SWEEP, run, "archive", "images"))),
                   "n_agents": len(glob.glob(os.path.join(SWEEP, run, "agents", "*.zip"))),
                   "backfilled": True}
            with open(os.path.join(stage, "data_manifest.json"), "w") as f:
                json.dump(man, f)
            api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=stage,
                              path_in_repo=f"results/{run}",
                              commit_message=f"Backfill {','.join(sorted(names))} for {run}")
            done += 1
            print(f"[{done}/{len(todo)}] {run}  ({', '.join(sorted(names))})", flush=True)
        except Exception as e:
            print(f"[ERR] {run}: {e}", flush=True)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    print(f"\nDONE: uploaded for {done}/{len(todo)} runs")


if __name__ == "__main__":
    main()
