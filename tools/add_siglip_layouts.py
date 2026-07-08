#!/usr/bin/env python3
"""Add the SigLIP-sorted ordering to already-built sprite layout.json files in place.

Many runs' sprites were built without the SigLIP layout (embeddings weren't present
at build time, or umap/rasterfairy were unavailable), so the gallery greys out the
"SigLIP" grid sort for them. This recomputes JUST that ordering from the cached
SigLIP-2 embeddings and merges layouts.siglip into the existing
site/<run>/sprite/layout.json — no sheet re-packing, no UMAP for layouts that exist.
Mirrors tools/add_lineage_layouts.py.

Embeddings source per run, in order:
  1. sweep_logs/sweep/<run>/embeddings_openclip_SigLIP2-B-alignet.npz   (local)
  2. HF results/<run>/embeddings_openclip_SigLIP2-B-alignet.npz         (with --download)

  .venv/bin/python tools/add_siglip_layouts.py --dry-run          # list what's missing + emb source
  .venv/bin/python tools/add_siglip_layouts.py <run>              # one run
  .venv/bin/python tools/add_siglip_layouts.py --download         # all missing, pulling npz from HF as needed
Re-push the changed layouts with tools/push_sprites.py (or --push here).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO = Path("/home/jupyter-smearle/picbreeder-vlm")
SWEEP = REPO / "sweep_logs" / "sweep"
SITE = REPO / "archive_animations" / "_archive_mirror" / "site"
HF_REPO = "picbreeder-vlm/picbreeder-vlm-archive"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
NPZ = "embeddings_openclip_SigLIP2-B-alignet.npz"

# import siglip_layout from build_archive_image_lib without running its main()
_spec = importlib.util.spec_from_file_location(
    "balib", REPO / "tools" / "build_archive_image_lib.py")
_balib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_balib)  # type: ignore


def hf_token() -> str | None:
    for p in (Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"):
        if p.is_file():
            return p.read_text().strip()
    return os.environ.get("HF_TOKEN")


def fetch_npz(run: str, dst: Path, token: str | None) -> bool:
    url = f"{HF_BASE}/results/{run}/{NPZ}"
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
            f.write(r.read())
        return True
    except Exception as e:  # noqa: BLE001
        print(f"    HF npz fetch failed: {type(e).__name__} {getattr(e, 'code', e)}")
        return False


def missing_runs() -> list[str]:
    out = []
    for lp in sorted(SITE.glob("*/sprite/layout.json")):
        try:
            lay = json.load(open(lp))
        except Exception:  # noqa: BLE001
            continue
        if "siglip" not in lay.get("layouts", {}):
            out.append(lp.parent.parent.name)
    return out


def npz_for(run: str, download: bool, token: str | None, tmp: Path) -> Path | None:
    local = SWEEP / run / NPZ
    if local.is_file():
        return local
    if download:
        dst = tmp / f"{run}.npz"
        if fetch_npz(run, dst, token):
            return dst
    return None


def add_siglip(run: str, download: bool, token: str | None, tmp: Path) -> str:
    lp = SITE / run / "sprite" / "layout.json"
    if not lp.is_file():
        return "no-layout"
    lay = json.load(open(lp))
    if "siglip" in lay.get("layouts", {}):
        return "already"
    npz = npz_for(run, download, token, tmp)
    if npz is None:
        return "no-embeddings"
    n_keep = lay.get("n")
    if not n_keep:
        # derive from the chronological rc length
        rc = lay["layouts"]["chronological"]["rc"]
        n_keep = len(rc)
    rc, dims = _balib.siglip_layout(npz, n_keep=n_keep)
    # siglip may cover fewer images than chronological (embeddings computed before the
    # last few were published) — that's normal; 16/43 existing siglip runs are shorter.
    # The gallery tolerates rc shorter than n. Only reject an empty/garbage result.
    if not rc or len(rc) > n_keep:
        return f"bad-rc({len(rc)} vs n={n_keep})"
    lay["layouts"]["siglip"] = {"dims": dims, "rc": rc}
    tmp_out = lp.with_suffix(".json.tmp")
    json.dump(lay, open(tmp_out, "w"))
    tmp_out.replace(lp)
    return "OK"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="specific run dirs (default: all missing)")
    ap.add_argument("--download", action="store_true", help="pull npz from HF results/ when not local")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--push", action="store_true", help="upload changed layout.json to HF after building")
    args = ap.parse_args()

    runs = args.runs or missing_runs()
    token = hf_token()
    print(f"{len(runs)} run(s) missing siglip\n")

    if args.dry_run:
        for r in runs:
            local = (SWEEP / r / NPZ).is_file()
            print(f"  {'local-emb ' if local else 'HF-emb    '} {r}")
        return

    changed, tally = [], {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for i, r in enumerate(runs, 1):
            res = add_siglip(r, args.download, token, tmp)
            tally[res] = tally.get(res, 0) + 1
            if res == "OK":
                changed.append(r)
            print(f"[{i}/{len(runs)}] {res:20} {r}", flush=True)

    print("\nsummary:", tally)
    if changed:
        Path(REPO / "_siglip_changed.txt").write_text("\n".join(changed) + "\n")
        print(f"wrote {len(changed)} changed run names -> _siglip_changed.txt")

    if args.push and changed:
        from huggingface_hub import HfApi, CommitOperationAdd
        api = HfApi()
        ops = [CommitOperationAdd(path_in_repo=f"site/{r}/sprite/layout.json",
                                  path_or_fileobj=str(SITE / r / "sprite" / "layout.json"))
               for r in changed]
        api.create_commit(repo_id=HF_REPO, repo_type="dataset", operations=ops,
                          commit_message=f"Add SigLIP layout to {len(changed)} runs")
        print(f"[push] committed {len(ops)} layout.json -> HF")


if __name__ == "__main__":
    main()
