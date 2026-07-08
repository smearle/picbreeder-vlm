#!/usr/bin/env python3
"""Embed the full human Picbreeder archive in the joint SigLIP2 webli space and
save embeddings WITH their image filenames.

The plain image_embeddings_cache_ViT-SO400M-14-SigLIP2_webli.npy in
fer/src/archive_res-128/ has no filename sidecar and can lag the on-disk image
set, so the row->image mapping is ambiguous. build_metric_fig_assets.py's
global Semantic Recall grid needs an exact mapping, so we (re)embed here and
write fer/src/archive_res-128/webli_emb_named.npz {filenames, embeddings}.

Run:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/embed_human_webli.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from picbreeder_vlm.vlm.model_loader import load_model_by_name, embed_images  # noqa: E402
from picbreeder_vlm.core.utils import load_human_archive_images                 # noqa: E402


def main() -> None:
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    arc = REPO / "fer/src/archive_res-128"
    paths = load_human_archive_images(arc / "images")
    print(f"images: {len(paths)}")
    model, preprocess, _tok = load_model_by_name("ViT-SO400M-14-SigLIP2", "webli", dev)
    names, emb = embed_images(model, preprocess, paths, torch.device(dev), batch_size=128)
    out = arc / "webli_emb_named.npz"
    np.savez(out, filenames=np.array(names), embeddings=emb.astype(np.float32))
    print(f"saved {emb.shape} -> {out}")


if __name__ == "__main__":
    main()
