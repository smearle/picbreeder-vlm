#!/usr/bin/env python3
"""
Build per-seed blog assets for the results table in
smearle.github.io/picbreeder-vlm-06b0d76d/index.html

For each (config, seed in {3,4,5}) we emit, into ASSETS_DIR:
  <arc>_s<N>.jpg         560-edge thumb    (used in popup + isolated tab)
  <arc>_s<N>_full.jpg    1600-edge full    (loaded on demand for fullscreen)
  <arc>_s<N>_siglip.jpg  1400-edge JPEG    (rasterized from cross_eval UMAP grid)
  <arc>_s<N>_tree.png    ~1500-edge PNG    (rasterized from archive_phylogeny PDF)

Source PNGs/PDFs are huge (~20-50 MB each); we re-encode at moderate JPEG quality
to keep total page weight low. The browser still lazy-loads each asset only when
the user clicks the matching tab, so these don't affect first-paint cost.

Growth videos (.mp4) already live in ASSETS_DIR (one per arc, not per seed);
they're reused as-is.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PIL import Image

# Pillow caps PNG decode at ~89 Mpx by default; some grids run ~75-80 Mpx
# (e.g. 6000×9300, 7992×7992 -> ~63 Mpx) so bump the limit.
Image.MAX_IMAGE_PIXELS = 400_000_000

REPO = Path("/home/jupyter-smearle/picbreeder-vlm")
SWEEP = REPO / "sweep_logs" / "sweep"
XEVAL = REPO / "cross_eval"
ASSETS_DIR = Path("/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets/archives")

SEEDS = [3, 4, 5]

# (arc, sweep_run_basename, cross_eval_dir_or_None)
# - sweep_run_basename excludes the trailing "_s<N>"
# - cross_eval_dir is relative to XEVAL and holds embed_grid_rect_..._seed<N>.png
#   (the SigLIP UMAP grids).  None -> no per-seed SigLIP available; we fall
#   back to the per-run UMAP grid sitting next to the archive dir.
CONFIGS: list[tuple[str, str, Optional[str]]] = [
    ("default",
     "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh",
     "chat_history_turns/Context_length___1"),
    ("noise_0.05",
     "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_randp0.05_rmode-all_nopersonalities_fixed-sesh",
     r"full_rand_select_prob/$\epsilon$___0.05"),
    ("noise_0.25",
     "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_randp0.25_rmode-all_nopersonalities_fixed-sesh",
     r"full_rand_select_prob/$\epsilon$___0.25"),
    ("noise_0.5",
     "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_randp0.5_rmode-all_nopersonalities_fixed-sesh",
     r"full_rand_select_prob/$\epsilon$___0.5"),
    ("noise_0.75",
     "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_randp0.75_rmode-all_nopersonalities_fixed-sesh",
     r"full_rand_select_prob/$\epsilon$___0.75"),
    ("noise_1.0",
     "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_randp1_rmode-all_nopersonalities_fixed-sesh",
     r"full_rand_select_prob/$\epsilon$___1"),
    ("mem_0",
     "th0_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh",
     "chat_history_turns/Context_length___0"),
    ("mem_2",
     "th2_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh",
     "chat_history_turns/Context_length___2"),
    ("mem_10",
     "th10_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh",
     "chat_history_turns/Context_length___10"),
    ("mem_20",
     "th-1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh",
     "chat_history_turns/Context_length___20"),
    ("agents_10",
     "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_traits10_fixed-sesh",
     "traits/Num._agents___10"),
    ("agents_100",
     "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_traits100_fixed-sesh",
     "traits/Num._agents___100"),
    ("agents_1000",
     "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_traits1000_fixed-sesh",
     "traits/Num._agents___1000"),
    ("random",
     "ag20_tb-1_scheme-toggle_randp2_rmode-all_nopersonalities_fixed-sesh",
     None),  # random has only a per-run UMAP grid (no seedN suffix)
]


# -------- helpers ------------------------------------------------------------

def downscale_jpeg(src: Path, dst: Path, max_edge: int, quality: int = 85,
                   force: bool = False) -> bool:
    """Downscale src image to max_edge on long side, save JPEG to dst.
    Returns True if a write occurred."""
    if not src.is_file():
        print(f"  MISS {src}")
        return False
    if dst.exists() and not force and dst.stat().st_mtime >= src.stat().st_mtime:
        return False
    img = Image.open(src)
    img.load()
    # Flatten any alpha onto white so JPEGs are sensible.
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = max_edge / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=quality, optimize=True, progressive=True)
    print(f"  wrote {dst.name} ({dst.stat().st_size//1024} KB)")
    return True


def native_webp(src: Path, dst: Path, quality: int = 82,
                force: bool = False) -> bool:
    """Encode src image to dst as WebP at intrinsic resolution (no resize).

    Used for the 'click again to native size' tier: each cell renders at the
    same pixel size the agent saw during evolution.  WebP is ~20% smaller than
    JPEG at matched perceptual quality on busy/colorful images.
    """
    if not src.is_file():
        print(f"  MISS {src}")
        return False
    if dst.exists() and not force and dst.stat().st_mtime >= src.stat().st_mtime:
        return False
    img = Image.open(src); img.load()
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # method=6 = best compression; ~2x slower than method=4 but worth it once.
    img.save(dst, "WEBP", quality=quality, method=6)
    print(f"  wrote {dst.name} ({dst.stat().st_size//1024} KB, {img.size[0]}x{img.size[1]})")
    return True


def pdf_to_png(pdf: Path, dst: Path, target_w: int = 4000,
               force: bool = False) -> bool:
    """Rasterize first page of pdf to dst PNG at target_w pixels wide.

    The phylogeny pages are wildly wide horizontal strips (~28:1), so we fix
    width rather than the long side; height comes out proportional.
    """
    if not pdf.is_file():
        print(f"  MISS {pdf}")
        return False
    if dst.exists() and not force and dst.stat().st_mtime >= pdf.stat().st_mtime:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    stem = dst.with_suffix("")
    # Clean stale outputs from a prior failed run.
    for old in dst.parent.glob(stem.name + "-*.png"):
        old.unlink(missing_ok=True)
    out = subprocess.run(
        ["pdftoppm", "-scale-to-x", str(target_w), "-scale-to-y", "-1",
         "-png", "-f", "1", "-l", "1", str(pdf), str(stem)],
        capture_output=True,
    )
    if out.returncode != 0:
        print(f"  pdftoppm FAILED on {pdf}: {out.stderr!r}")
        return False
    # pdftoppm writes "<stem>-1.png" (or "-01" etc.); pick the first.
    candidates = sorted(dst.parent.glob(stem.name + "-*.png"))
    if not candidates or candidates[0].stat().st_size == 0:
        print(f"  pdftoppm produced no usable output for {pdf}")
        return False
    raw = candidates[0]
    # Re-encode as optimized PNG (smaller than pdftoppm's default), then drop raw.
    img = Image.open(raw); img.load()
    img.save(dst, "PNG", optimize=True)
    raw.unlink(missing_ok=True)
    print(f"  wrote {dst.name} ({dst.stat().st_size//1024} KB)")
    return True


def pdf_to_jpeg(pdf: Path, dst: Path, max_edge: int = 1400,
                quality: int = 85, force: bool = False) -> bool:
    """Rasterize first page of pdf to dst JPEG at max_edge on the long side."""
    if not pdf.is_file():
        print(f"  MISS {pdf}")
        return False
    if dst.exists() and not force and dst.stat().st_mtime >= pdf.stat().st_mtime:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    stem = dst.with_suffix("")
    for old in dst.parent.glob(stem.name + "-*.png"):
        old.unlink(missing_ok=True)
    # 150 DPI tends to produce a reasonable working canvas for square-ish PDFs.
    out = subprocess.run(
        ["pdftoppm", "-r", "150", "-png", "-f", "1", "-l", "1",
         str(pdf), str(stem)],
        capture_output=True,
    )
    if out.returncode != 0:
        print(f"  pdftoppm FAILED on {pdf}: {out.stderr!r}")
        return False
    candidates = sorted(dst.parent.glob(stem.name + "-*.png"))
    if not candidates or candidates[0].stat().st_size == 0:
        print(f"  pdftoppm produced no usable output for {pdf}")
        return False
    raw = candidates[0]
    img = Image.open(raw); img.load()
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = max_edge / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.LANCZOS)
    img.save(dst, "JPEG", quality=quality, optimize=True, progressive=True)
    raw.unlink(missing_ok=True)
    print(f"  wrote {dst.name} ({dst.stat().st_size//1024} KB)")
    return True


# -------- main ---------------------------------------------------------------

def render_radial_phylogeny(run: Path, force: bool = False) -> Optional[Path]:
    """Ensure run/archive/archive_phylogeny_radial.pdf exists (twopi images-mode).

    Cached as a vector PDF next to the run's metadata; re-rendered only when the
    metadata is newer (or --force). twopi on a few-thousand-node archive is the
    expensive step, so we keep the PDF and just re-rasterize cheaply downstream.
    """
    archive = run / "archive"
    meta = archive / "archive_metadata.json"
    if not meta.is_file():
        print(f"  MISS {meta}")
        return None
    pdf = archive / "archive_phylogeny_radial.pdf"
    if pdf.is_file() and not force and pdf.stat().st_mtime >= meta.stat().st_mtime:
        return pdf
    cmd = [
        sys.executable, str(REPO / "visualize_archive_phylogeny.py"),
        f"experiment_dir={run.resolve()}",
        "layout=radial", "format=pdf", "mode=images",
        f"output={pdf.resolve()}",
    ]
    out = subprocess.run(cmd, capture_output=True, cwd=str(REPO))
    if out.returncode != 0 or not pdf.is_file():
        tail = out.stderr.decode("utf-8", "replace")[-500:]
        print(f"  radial render FAILED {run.name}: {tail!r}")
        return None
    return pdf


def pdf_to_png_square(pdf: Path, dst: Path, target: int = 3200,
                      force: bool = False) -> bool:
    """Rasterize first page to dst PNG with the long edge scaled to `target`.

    Radial trees are roughly square, so we fix the long side (unlike the wide
    horizontal strip, which fixed width). Output is re-encoded as optimized PNG.
    """
    if not pdf.is_file():
        print(f"  MISS {pdf}")
        return False
    if dst.exists() and not force and dst.stat().st_mtime >= pdf.stat().st_mtime:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    stem = dst.with_suffix("")
    for old in dst.parent.glob(stem.name + "-*.png"):
        old.unlink(missing_ok=True)
    out = subprocess.run(
        ["pdftoppm", "-scale-to", str(target), "-png", "-f", "1", "-l", "1",
         str(pdf), str(stem)],
        capture_output=True,
    )
    if out.returncode != 0:
        print(f"  pdftoppm FAILED on {pdf}: {out.stderr!r}")
        return False
    candidates = sorted(dst.parent.glob(stem.name + "-*.png"))
    if not candidates or candidates[0].stat().st_size == 0:
        print(f"  pdftoppm produced no usable output for {pdf}")
        return False
    raw = candidates[0]
    img = Image.open(raw); img.load()
    img.save(dst, "PNG", optimize=True)
    raw.unlink(missing_ok=True)
    print(f"  wrote {dst.name} ({dst.stat().st_size // 1024} KB)")
    return True


def main(argv: list[str]) -> int:
    force = "--force" in argv
    only = [a for a in argv[1:] if not a.startswith("--")]

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_seen = 0
    for arc, run_basename, xeval_dir in CONFIGS:
        if only and arc not in only:
            continue
        print(f"[{arc}]  run={run_basename}")
        for s in SEEDS:
            n_seen += 1
            run = SWEEP / f"{run_basename}_s{s}"

            # archive grid (publication-order)
            grid = run / "archive" / "archive_grid.png"
            if downscale_jpeg(grid, ASSETS_DIR / f"{arc}_s{s}.jpg",
                              max_edge=560, force=force):
                n_written += 1
            if downscale_jpeg(grid, ASSETS_DIR / f"{arc}_s{s}_full.jpg",
                              max_edge=1600, quality=88, force=force):
                n_written += 1

            # archive_grid native (intrinsic resolution, click-again-to-zoom tier).
            if grid.is_file():
                if native_webp(grid, ASSETS_DIR / f"{arc}_s{s}_native.webp",
                               force=force):
                    n_written += 1

            # SigLIP-sorted UMAP grid: prefer cross_eval/<xeval_dir>/embed_grid_rect_..._seed<N>.png,
            # else fall back to rasterizing the per-run UMAP PDF (random only).
            siglip_dst = ASSETS_DIR / f"{arc}_s{s}_siglip.jpg"
            siglip_src_png = None  # set when we have a real PNG source for native
            wrote_siglip = False
            if xeval_dir is not None:
                siglip_png = (XEVAL / xeval_dir /
                              f"embed_grid_rect_SigLIP2-B-alignet_umap_seed{s}.png")
                if siglip_png.is_file():
                    siglip_src_png = siglip_png
                wrote_siglip = downscale_jpeg(
                    siglip_png, siglip_dst, max_edge=1400, force=force)
            if not wrote_siglip:
                # Fallback: per-run PDF (random has only this).
                siglip_pdf = run / "embed_grid_rect_SigLIP2-B-alignet_umap.pdf"
                wrote_siglip = pdf_to_jpeg(
                    siglip_pdf, siglip_dst, max_edge=1400, force=force)
            if wrote_siglip:
                n_written += 1

            # SigLIP native: only emit when the source is a PNG (raster) — for
            # `random` it'd require re-rastering the PDF at >2x size, which is
            # disk-heavy and rarely the row a reader will zoom into. Skip it.
            if siglip_src_png is not None:
                if native_webp(siglip_src_png,
                               ASSETS_DIR / f"{arc}_s{s}_siglip_native.webp",
                               force=force):
                    n_written += 1

            # phylogeny: radial (twopi) "tree of life" view. We render a cached
            # vector PDF per run (images-mode), then rasterize to a square PNG.
            tree_pdf = render_radial_phylogeny(run, force=force)
            if tree_pdf and pdf_to_png_square(
                    tree_pdf, ASSETS_DIR / f"{arc}_s{s}_tree.png",
                    target=3200, force=force):
                n_written += 1

    print(f"\nDone. Inspected {n_seen} (config,seed) pairs; wrote/refreshed "
          f"{n_written} files into {ASSETS_DIR}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
