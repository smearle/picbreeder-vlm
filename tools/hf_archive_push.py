#!/usr/bin/env python
"""Push curated paper artifacts (genomes, agent logs, clean results) for complete
Picbreeder-VLM runs to the private HF dataset `picbreeder-vlm/picbreeder-vlm-archive`.

Per run, under runs/<run_dir>/:
  - genomes.tar.gz        packed CPPN genomes (131k loose .pkl would wreck the repo tree)
  - agents/*.zip          agent logs: reasoning, per-gen selections, lineage jsonl (already packed)
  - results/*.json        curated metrics/metadata (NOT embeddings/npz/npy/pdf)
  - data_manifest.json    what artifacts this run has + parsed config

Then rebuilds runs/index.json across all runs present locally.

Idempotent/resumable: a run's stage is skipped if its sentinel already exists in the repo.
"""
import argparse, io, json, os, re, sys, tarfile, tempfile
from pathlib import Path
from huggingface_hub import HfApi

REPO = "picbreeder-vlm/picbreeder-vlm-archive"
SWEEP = Path("/home/jupyter-smearle/picbreeder-vlm/sweep_logs/sweep")

# Curated result JSONs to include from each run's archive/ dir (exclude embeddings/npz/npy/pdf).
RESULT_GLOBS = [
    "archive_metadata.json", "captions_*.json", "metrics_*.json",
    "phylogeny_metrics.json", "leaf_agent_lineages.json", "archive_grid.json",
]
RESULT_EXCLUDE = ("embedding", "embeddings", "cache")  # substrings that mark bloat


def complete_runs():
    out = []
    for d in sorted(SWEEP.iterdir()):
        if not d.is_dir():
            continue
        imgs = d / "archive" / "images"
        gen = d / "archive" / "genomes"
        if imgs.is_dir() and any(imgs.iterdir()) and gen.is_dir() and any(gen.iterdir()):
            out.append(d)
    return out


def parse_config(run):
    def grab(pat, cast=str, default=None):
        m = re.search(pat, run)
        return cast(m.group(1)) if m else default
    model = grab(r'model-([a-z0-9.\-]+?)_tb', default="gemini-2.5-pro")
    cfg = {
        "model": model,
        "memory_cl": grab(r'(?:^|_)th(-?\d+)_', int),
        "noise_eps": grab(r'randp([0-9.]+)', float, 0.0),
        "personalities": grab(r'traits(\d+)', int, 0),
        "seed": grab(r'_s(\d+)$', int),
    }
    return cfg


def parse_arc(run, cfg):
    """Best-effort map to the blog results-table data-arc key (None if not a table cell)."""
    if "gemini-random" in run:
        return "random"
    if cfg["model"] != "gemini-2.5-pro":
        return None
    if "traits" in run:
        return f"agents_{cfg['personalities']}" if cfg["personalities"] in (10, 100, 1000) else None
    if "randp" in run:
        return {0.05: "noise_0.05", 0.25: "noise_0.25", 0.5: "noise_0.5",
                0.75: "noise_0.75", 1.0: "noise_1.0"}.get(cfg["noise_eps"])
    cl = cfg["memory_cl"]
    if cl == 1:
        return "default"
    return {0: "mem_0", 2: "mem_2", 10: "mem_10", 20: "mem_20"}.get(cl)


def pick_results(archive_dir):
    import fnmatch
    picked = []
    for f in sorted(archive_dir.iterdir()):
        if not f.is_file() or f.suffix != ".json":
            continue
        if any(x in f.name.lower() for x in RESULT_EXCLUDE):
            continue
        if any(fnmatch.fnmatch(f.name, g) for g in RESULT_GLOBS):
            picked.append(f)
    return picked


def make_genomes_tar(gen_dir, dest):
    with tarfile.open(dest, "w:gz") as tf:
        for p in sorted(gen_dir.iterdir()):
            if p.is_file():
                tf.add(p, arcname=p.name)


def make_agents_tar(zips, dest):
    # store (no recompress) — agent .zip files are already compressed; this packs
    # up to ~3.5k tiny zips/run into ONE file so the repo tree stays sane.
    with tarfile.open(dest, "w") as tf:
        for p in zips:
            tf.add(p, arcname=p.name)


def stage_and_upload(api, run_dir, dry=False):
    run = run_dir.name
    archive = run_dir / "archive"
    cfg = parse_config(run)
    arc = parse_arc(run, cfg)
    n_images = sum(1 for _ in (archive / "images").iterdir())
    n_genomes = sum(1 for _ in (archive / "genomes").iterdir())
    agent_zips = sorted((run_dir / "agents").glob("*.zip")) if (run_dir / "agents").is_dir() else []
    results = pick_results(archive)

    info = {"run": run, "arc": arc, "config": cfg, "n_images": n_images,
            "n_genomes": n_genomes, "n_agents": len(agent_zips),
            "results": [r.name for r in results]}
    if dry:
        gsz = sum(p.stat().st_size for p in (archive / "genomes").iterdir() if p.is_file())
        zsz = sum(p.stat().st_size for p in agent_zips)
        rsz = sum(p.stat().st_size for p in results)
        info["_MB"] = {"genomes": round(gsz/1e6, 1), "agents": round(zsz/1e6, 1),
                       "results": round(rsz/1e6, 1)}
        print(json.dumps(info))
        return info

    sentinel = f"results/{run}/genomes.tar.gz"
    if api.file_exists(repo_id=REPO, filename=sentinel, repo_type="dataset"):
        print(f"[skip] {run} (already pushed)")
        return info

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / run
        stage.mkdir(parents=True)
        make_genomes_tar(archive / "genomes", stage / "genomes.tar.gz")
        if agent_zips:
            make_agents_tar(agent_zips, stage / "agents.tar")
        for r in results:                       # curated eval/metric JSONs, flat under results/<run>/
            (stage / r.name).write_bytes(r.read_bytes())
        (stage / "data_manifest.json").write_text(json.dumps(info))
        api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(stage),
                          path_in_repo=f"results/{run}",
                          commit_message=f"Add genomes+logs+results for {run}")
    print(f"[done] {run}  genomes+{len(results)} results+{len(agent_zips)} logs")
    return info


def rebuild_index(api, infos, dry=False):
    runs = []
    for i in sorted(infos, key=lambda x: x["run"]):
        runs.append({"run": i["run"], "arc": i.get("arc"), "seed": i["config"].get("seed"),
                     "label": i.get("label") or i["run"], "n_images": i["n_images"],
                     "config": i["config"], "has": {
                         "images": i["n_images"] > 0, "genomes": i["n_genomes"] > 0,
                         "genome_json": i.get("has_genome_json", False),
                         "logs": i["n_agents"] > 0, "results": bool(i["results"])}})
    idx = {"dataset": REPO, "n_runs": len(runs), "runs": runs}
    if dry:
        print(f"[index] would write {len(runs)} runs")
        return
    api.upload_file(repo_id=REPO, repo_type="dataset", path_in_repo="runs/index.json",
                    path_or_fileobj=io.BytesIO(json.dumps(idx, indent=1).encode()),
                    commit_message=f"Rebuild index ({len(runs)} runs)")
    print(f"[index] wrote {len(runs)} runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only first N complete runs")
    ap.add_argument("--only-run", default="", help="exact run dir name to push")
    ap.add_argument("--no-index", action="store_true", help="skip rebuilding runs/index.json")
    args = ap.parse_args()

    runs = complete_runs()
    if args.only_run:
        runs = [r for r in runs if r.name == args.only_run]
    if args.limit:
        runs = runs[:args.limit]
    print(f"{len(runs)} run(s) selected", file=sys.stderr)

    api = HfApi()
    infos = []
    for r in runs:
        try:
            infos.append(stage_and_upload(api, r, dry=args.dry_run))
        except Exception as e:
            print(f"[ERROR] {r.name}: {e}", file=sys.stderr)
    # NOTE: the top-level runs index is now built by tools/build_hf_index.py (scans the repo
    # for site/ + results/ artifacts across ALL runs); this tool only pushes results/<run>/.
    if not args.no_index:
        print("[index] skipped — run `python tools/build_hf_index.py` to rebuild runs index", file=sys.stderr)


if __name__ == "__main__":
    main()
