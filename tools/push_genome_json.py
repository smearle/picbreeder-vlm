#!/usr/bin/env python
"""Push every run's renderable genomes.json.gz (the breed-site fuel) to site/<run>/genomes.json.gz
in ONE commit. Stages a temp tree so the in-repo path is site/<run>/genomes.json.gz (not .../archive/).
"""
import shutil, sys, tempfile
from pathlib import Path
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).parent))
from hf_archive_push import complete_runs   # noqa: E402

REPO = "picbreeder-vlm/picbreeder-vlm-archive"


def main():
    runs = [d for d in complete_runs() if (d / "archive" / "genomes.json.gz").is_file()]
    print(f"{len(runs)} runs have genomes.json.gz")
    if "--dry-run" in sys.argv:
        total = sum((d / "archive" / "genomes.json.gz").stat().st_size for d in runs)
        print(f"would upload {total/1e6:.0f} MB -> site/<run>/genomes.json.gz")
        return
    tmp = Path(tempfile.mkdtemp())
    for d in runs:
        (tmp / d.name).mkdir(parents=True)
        shutil.copy(d / "archive" / "genomes.json.gz", tmp / d.name / "genomes.json.gz")
    HfApi().upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(tmp),
                          path_in_repo="site",
                          commit_message=f"Add renderable genomes.json.gz for {len(runs)} runs")
    shutil.rmtree(tmp)
    print(f"[site] uploaded {len(runs)} genomes.json.gz -> site/<run>/")
    print("[index] run `python tools/build_hf_index.py` to set has.genome_json")


if __name__ == "__main__":
    main()
