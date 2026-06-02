#!/usr/bin/env python
"""Push the per-run SigLIP embeddings (the GPU-expensive eval input) to results/<run>/, in ONE
commit. With these + captions + code, the whole cross-eval (coverage, novelty, the results-table
numbers) is reproducible without re-embedding. ~547 MB across the runs that have them.
"""
import sys
from pathlib import Path
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).parent))
from hf_archive_push import complete_runs, SWEEP   # noqa: E402

REPO = "picbreeder-vlm/picbreeder-vlm-archive"
EMB = "embeddings_openclip_SigLIP2-B-alignet.npz"


def main():
    runs = [d for d in complete_runs() if (d / EMB).is_file()]
    allow = [f"{d.name}/{EMB}" for d in runs]
    print(f"{len(runs)} complete runs have {EMB}")
    if "--dry-run" in sys.argv:
        total = sum((d / EMB).stat().st_size for d in runs)
        print(f"would upload {total/1e6:.0f} MB -> results/<run>/{EMB}")
        return
    HfApi().upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(SWEEP),
                          path_in_repo="results", allow_patterns=allow,
                          commit_message=f"Add SigLIP embeddings for {len(runs)} runs")
    print(f"[embeddings] uploaded {len(runs)} npz -> results/<run>/")


if __name__ == "__main__":
    main()
