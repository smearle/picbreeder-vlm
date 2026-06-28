#!/usr/bin/env python
"""Publish the converted human Picbreeder genomes to the breed archive dataset so
breed/browse.html?run=human renders + branches them client-side.

Uploads (to picbreeder-vlm/picbreeder-vlm-archive, dataset repo):
  site/human/genomes.json.gz          {img_id: cppn genome json}
  results/human/archive_metadata.json {entries:[...], n, source}
  results/human/tree.json.gz          family-tree lineage sidecar (PB.treeShardURL)

and flips the human entry's has.genome_json -> true in index.json (HF + the local
gh-pages mirror) so runs.html lists it as browsable. Touches ONLY human paths, so
it won't collide with the concurrent AI-genome uploads.

Unlike the VLM runs (whose tree shards ship bundled in the blog repo under
breed/data/tree/), the human archive keeps its shard here on HF next to its
genomes/metadata; the breed site fetches it via archiveBase() (PB.treeShardURL).
"""
import argparse
import gzip
import importlib.util
import io
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "picbreeder-vlm/picbreeder-vlm-archive"
TOOLS = Path(__file__).resolve().parent
SRC = Path("/home/jupyter-smearle/picbreeder-vlm/archive_animations/out/human_genomes")
GHPAGES_INDEX = Path.home() / "smearle.github.io/picbreeder-vlm-06b0d76d/index.json"


def build_human_tree_shard() -> Path:
    """Build the human family-tree sidecar from the local archive metadata and
    write it next to the genomes (SRC/results/human/tree.json.gz). Reuses
    build_tree_data.build_human so the format matches the VLM run shards."""
    spec = importlib.util.spec_from_file_location("btd", TOOLS / "build_tree_data.py")
    btd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(btd)
    data = btd.build_human()
    if data is None:
        raise SystemExit("build_human() returned None — run tools/build_human_genomes.py first")
    out = SRC / "results/human/tree.json.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print(f"built human tree shard: {data['n']} nodes, depth {data['depth']} -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    api = HfApi()

    build_human_tree_shard()
    uploads = [
        (SRC / "site/human/genomes.json.gz", "site/human/genomes.json.gz"),
        (SRC / "results/human/archive_metadata.json", "results/human/archive_metadata.json"),
        (SRC / "results/human/tree.json.gz", "results/human/tree.json.gz"),
    ]
    for local, remote in uploads:
        assert local.is_file(), f"missing {local}"
        print(f"{'[dry] ' if args.dry_run else ''}upload {local} ({local.stat().st_size/1e6:.1f} MB) -> {remote}")
        if not args.dry_run:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                            repo_id=REPO, repo_type="dataset",
                            commit_message="Add human Picbreeder genomes + metadata")

    # flip has.genome_json on the human entry in index.json (HF copy). Fetch fresh
    # (force_download) right before editing to minimise clobbering a concurrent
    # AI-genome uploader that may also be rewriting index.json.
    idx_path = hf_hub_download(REPO, "index.json", repo_type="dataset", force_download=True)
    idx = json.loads(Path(idx_path).read_text())
    hits = 0
    for r in idx.get("runs", []):
        if r.get("run") == "human":
            r.setdefault("has", {})["genome_json"] = True
            hits += 1
    print(f"index.json: marked {hits} human entry has.genome_json=true")
    if not args.dry_run and hits:
        api.upload_file(path_or_fileobj=io.BytesIO(json.dumps(idx).encode()),
                        path_in_repo="index.json", repo_id=REPO, repo_type="dataset",
                        commit_message="human: has.genome_json=true")
        # mirror the flag into the local gh-pages index.json too (kept in git)
        if GHPAGES_INDEX.is_file():
            gidx = json.loads(GHPAGES_INDEX.read_text())
            for r in gidx.get("runs", []):
                if r.get("run") == "human":
                    r.setdefault("has", {})["genome_json"] = True
            GHPAGES_INDEX.write_text(json.dumps(gidx))
            print(f"updated local {GHPAGES_INDEX}")
    print("done.")


if __name__ == "__main__":
    main()
