#!/usr/bin/env python3
"""Push several staged transcript bundles to HF in ONE commit per batch.

``push_transcripts.py`` does one ``upload_folder`` (= one commit) per run, which
trips HF's 128-commits-per-hour repo limit once you push more than ~128 runs at a
go. This batches N runs into a single ``create_commit``, so 139 runs cost 2 commits
instead of 139.

    python -m tools.push_transcripts_batched --stage _transcripts_stage_12 --runs remaining.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "picbreeder-vlm/picbreeder-vlm-archive"


def ops_for(stage: Path, run: str) -> list[CommitOperationAdd]:
    rd = stage / run
    return [CommitOperationAdd(path_in_repo=f"transcripts/{run}/{p.relative_to(rd).as_posix()}",
                               path_or_fileobj=str(p))
            for p in sorted(rd.rglob("*")) if p.is_file()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True, help="file with one run name per line")
    ap.add_argument("--batch", type=int, default=13, help="runs per commit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = [r for r in args.runs.read_text().split() if r]
    api = HfApi()
    for i in range(0, len(runs), args.batch):
        chunk = runs[i:i + args.batch]
        ops = [op for r in chunk for op in ops_for(args.stage, r)]
        mb = sum(Path(op.path_or_fileobj).stat().st_size for op in ops) / 1e6
        msg = f"Add transcripts for {len(chunk)} runs ({chunk[0]}…)"
        print(f"[batch {i//args.batch + 1}] {len(chunk)} runs, {len(ops)} files, {mb:.0f} MB")
        if args.dry_run:
            continue
        api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, commit_message=msg)
        print("  committed")


if __name__ == "__main__":
    main()
