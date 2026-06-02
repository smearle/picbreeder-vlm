#!/usr/bin/env python
"""One-time, server-side reorg of the HF dataset from the old flat `runs/<run>/...`
into `site/<run>/...` (web assets) + `results/<run>/...` (repro). Uses CommitOperationCopy
(no re-upload — pure server-side metadata) + CommitOperationDelete, batched into a few commits.

Mapping for each old path  runs/<run>/<rest>:
  rest = sprite/...        -> site/<run>/sprite/...
  rest = genomes.json.gz   -> site/<run>/genomes.json.gz      (renderable genomes = site asset)
  rest = results/<f>       -> results/<run>/<f>                (flatten the inner results/)
  else (genomes.tar.gz, agents.tar, data_manifest.json) -> results/<run>/<rest>
  runs/index.json          -> deleted (rebuilt at top level by build_hf_index.py)

Idempotent: only acts on paths still under runs/.
"""
import sys
from huggingface_hub import HfApi
from huggingface_hub import CommitOperationCopy, CommitOperationDelete

REPO = "picbreeder-vlm/picbreeder-vlm-archive"
BATCH = 200


def dst_for(path):
    if not path.startswith("runs/"):
        return None
    if path == "runs/index.json":
        return "__DELETE__"
    rest = path[len("runs/"):]
    run, _, tail = rest.partition("/")
    if not tail:
        return None
    if tail.startswith("sprite/"):
        return f"site/{run}/{tail}"
    if tail == "genomes.json.gz":
        return f"site/{run}/genomes.json.gz"
    if tail.startswith("results/"):
        return f"results/{run}/{tail[len('results/'):]}"
    return f"results/{run}/{tail}"


def main():
    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")
    ops = []
    for p in files:
        d = dst_for(p)
        if d is None:
            continue
        if d == "__DELETE__":
            ops.append(CommitOperationDelete(path_in_repo=p))
        else:
            ops.append(CommitOperationCopy(src_path_in_repo=p, path_in_repo=d))
            ops.append(CommitOperationDelete(path_in_repo=p))
    copies = sum(isinstance(o, CommitOperationCopy) for o in ops)
    dels = sum(isinstance(o, CommitOperationDelete) for o in ops)
    print(f"{copies} copies + {dels} deletes across {len({o.path_in_repo.split('/')[1] for o in ops if '/' in o.path_in_repo})} runs")
    if "--dry-run" in sys.argv:
        for o in ops[:12]:
            kind = "COPY" if isinstance(o, CommitOperationCopy) else "DEL "
            print(" ", kind, getattr(o, "src_path_in_repo", o.path_in_repo), "->", o.path_in_repo)
        return
    # commit in batches (copies+deletes interleaved per run keep a run atomic-ish; HF copies read pre-commit state)
    for i in range(0, len(ops), BATCH):
        chunk = ops[i:i + BATCH]
        api.create_commit(repo_id=REPO, repo_type="dataset", operations=chunk,
                          commit_message=f"Reorg runs/->site|results [{i//BATCH+1}]")
        print(f"  committed batch {i//BATCH+1} ({len(chunk)} ops)")
    print("migration done")


if __name__ == "__main__":
    main()
