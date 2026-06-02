#!/usr/bin/env python
"""Build the dataset's top-level index.json by scanning what's actually present on HF.
One manifest the site reads: per run -> arc, seed, has{sprite, genome_json, genomes, logs,
embeddings, results}, plus n_images (from the local sprite layout when available) and an
optional per-run `base` (left unset here; the human row gets one when its assets land on gh-pages).

  python tools/build_hf_index.py [--dry-run]
"""
import io, json, sys
from pathlib import Path
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).parent))
from hf_archive_push import parse_config, parse_arc   # noqa: E402

REPO = "picbreeder-vlm/picbreeder-vlm-archive"
MIRROR_INDEX = Path("/home/jupyter-smearle/picbreeder-vlm/archive_animations/_archive_mirror/index.json")


def main():
    api = HfApi()
    files = set(api.list_repo_files(REPO, repo_type="dataset"))

    runs = set()
    for p in files:
        parts = p.split("/")
        if len(parts) >= 2 and parts[0] in ("site", "results"):
            runs.add(parts[1])

    # trust the local mirror's arc/seed/n_images for sprite runs (built from the canonical
    # ARC_TO_RUN map); parse_arc's name heuristic mismaps mem_20 (th-1) and random (randp2).
    mirror = {}
    if MIRROR_INDEX.is_file():
        for r in json.load(open(MIRROR_INDEX)).get("runs", []):
            mirror[r["run"]] = r

    out = []
    for run in sorted(runs):
        cfg = parse_config(run)
        m = mirror.get(run, {})
        has = {
            "sprite": f"site/{run}/sprite/layout.json" in files,
            "genome_json": f"site/{run}/genomes.json.gz" in files,
            "genomes": f"results/{run}/genomes.tar.gz" in files,
            "logs": f"results/{run}/agents.tar" in files,
            "embeddings": any(p.startswith(f"results/{run}/embeddings_") and p.endswith(".npz") for p in files),
            "results": any(p.startswith(f"results/{run}/") and p.endswith(".json") for p in files),
        }
        out.append({"run": run, "arc": m.get("arc") or parse_arc(run, cfg),
                    "seed": m.get("seed", cfg.get("seed")),
                    "config": cfg, "n_images": m.get("n_images"), "has": has})

    idx = {"dataset": REPO, "n_runs": len(out),
           "n_sprite": sum(r["has"]["sprite"] for r in out),
           "runs": out}
    print(f"{len(out)} runs | sprite={idx['n_sprite']} genomes={sum(r['has']['genomes'] for r in out)} "
          f"logs={sum(r['has']['logs'] for r in out)} embeddings={sum(r['has']['embeddings'] for r in out)}")
    if "--dry-run" in sys.argv:
        return
    api.upload_file(repo_id=REPO, repo_type="dataset", path_in_repo="index.json",
                    path_or_fileobj=io.BytesIO(json.dumps(idx, indent=1).encode()),
                    commit_message=f"Rebuild index ({len(out)} runs, {idx['n_sprite']} with sprites)")
    print("wrote index.json")


if __name__ == "__main__":
    main()
