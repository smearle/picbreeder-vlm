#!/usr/bin/env python3
"""Diagnostic: WHY do the noun-stars collapse in a joint text-image UMAP?
Shows (1) the joint UMAP colored by modality (images vs THINGS nouns) and
(2) the raw cosine-similarity structure (image-image / text-text / image-text)
that drives the modality gap."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_metric_fig_assets as B

run = B.DEFAULT_RUN.resolve()
out = (B.REPO / "figures/metric_figs").resolve()

fnames, img_emb, noun_emb, n_nouns, chosen = B._recall_chosen(run)
img_emb = img_emb / (np.linalg.norm(img_emb, axis=1, keepdims=True) + 1e-9)
noun_emb = noun_emb / (np.linalg.norm(noun_emb, axis=1, keepdims=True) + 1e-9)
chosen_ni = [c[1] for c in chosen]
print(f"images: {img_emb.shape[0]}   nouns: {noun_emb.shape[0]}   chosen: {len(chosen)}")

# ---- (1) joint UMAP over ALL images + ALL nouns, colored by modality ----
comb = np.vstack([img_emb, noun_emb])
xy = B._umap2d(comb, B.SCATTER_CFG["recall"]["nn"],
               B.SCATTER_CFG["recall"]["min_dist"], B.SCATTER_CFG["recall"]["seed"])
ni = img_emb.shape[0]
img_xy, noun_xy = xy[:ni], xy[ni:]

# ---- (2) cosine-sim structure (subsample for the histogram) ----
rng = np.random.RandomState(0)
si = rng.choice(img_emb.shape[0], min(600, img_emb.shape[0]), replace=False)
sn = rng.choice(noun_emb.shape[0], min(600, noun_emb.shape[0]), replace=False)
I, N = img_emb[si], noun_emb[sn]


def _offdiag(M):
    return M[~np.eye(M.shape[0], dtype=bool)]


ii = _offdiag(I @ I.T)
nn = _offdiag(N @ N.T)
inb = (I @ N.T).ravel()
print(f"mean cos  image-image={ii.mean():.3f}  text-text={nn.mean():.3f}  "
      f"image-text={inb.mean():.3f}")
print(f"text-text spread is {ii.std()/ (nn.std()+1e-9):.1f}x TIGHTER than image-image "
      f"(std: img={ii.std():.3f} txt={nn.std():.3f})")

fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.6))

a0.scatter(img_xy[:, 0], img_xy[:, 1], s=5, c="#7a52c0", alpha=0.25,
           linewidths=0, label=f"archive images ({ni})")
a0.scatter(noun_xy[:, 0], noun_xy[:, 1], s=8, c="#d08b00", alpha=0.45,
           linewidths=0, label=f"THINGS nouns ({noun_emb.shape[0]})")
a0.scatter(noun_xy[chosen_ni, 0], noun_xy[chosen_ni, 1], s=130, marker="*",
           c="#3a1a70", edgecolors="white", linewidths=0.8, zorder=5,
           label="curated nouns")
for k, ci in enumerate(chosen_ni):
    a0.annotate(chosen[k][0], noun_xy[ci], xytext=(5, 4),
                textcoords="offset points", fontsize=8, color="#3a1a70")
a0.set_title("Joint text-image UMAP (the intermediary)\n"
             "all text lands in ONE tight clump = modality gap", fontsize=10)
a0.set_xticks([]); a0.set_yticks([])
a0.legend(loc="lower left", fontsize=7, framealpha=0.9)
for s in a0.spines.values():
    s.set_visible(False)

bins = np.linspace(-0.2, 1.0, 60)
a1.hist(ii, bins=bins, alpha=0.55, color="#7a52c0", density=True, label="image-image")
a1.hist(nn, bins=bins, alpha=0.55, color="#d08b00", density=True, label="text-text")
a1.hist(inb, bins=bins, alpha=0.45, color="#888", density=True, label="image-text (cross)")
a1.axvline(ii.mean(), color="#7a52c0", lw=1.2, ls="--")
a1.axvline(nn.mean(), color="#d08b00", lw=1.2, ls="--")
a1.axvline(inb.mean(), color="#444", lw=1.2, ls="--")
a1.set_title("Why: cosine-similarity structure\n"
             "text embeddings are mutually FAR more similar than images are",
             fontsize=10)
a1.set_xlabel("cosine similarity"); a1.set_yticks([])
a1.legend(loc="upper left", fontsize=8)
for s in ["top", "right", "left"]:
    a1.spines[s].set_visible(False)

fig.tight_layout()
op = out / "recall_modality_diag.png"
fig.savefig(op, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
print("wrote", op)
