#!/usr/bin/env python
"""Build layout.json + sprite sheets for ONE representative seed of every results-table arc,
staging them into the blog's local archive mirror so the gallery can be tested for all rows
(and later pushed to HF). Reuses tools/build_archive_image_lib.py (layout + thumbs) and
archive_animations/make_sprite_sheets.py (packing), so the on-disk format matches the demo.

Per arc -> runs/<full_sweep_run>_s<seed>/sprite/{layout.json, sprites.json, sheets/*.webp}
Then writes runs/index.json over all sprite runs present in the mirror.

Idempotent: skips an arc whose sprite/layout.json already exists.
"""
import json, os, shutil, subprocess, sys
from pathlib import Path

REPO = Path("/home/jupyter-smearle/picbreeder-vlm")
ASSETS = Path("/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets/archives")
MIRROR_ROOT = REPO / "archive_animations" / "_archive_mirror"   # mirrors the HF tree: {index.json, site/<run>/sprite/}
SITE = MIRROR_ROOT / "site"
PY = sys.executable

# arc -> full sweep run-dir base (seed appended below); mirrors build_archive_image_lib.ARC_TO_RUN
ARC_TO_RUN = {
    "default":     "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh",
    "noise_0.05":  "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_randp0.05_rmode-all_nopersonalities_fixed-sesh",
    "noise_0.25":  "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_randp0.25_rmode-all_nopersonalities_fixed-sesh",
    "noise_0.5":   "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_randp0.5_rmode-all_nopersonalities_fixed-sesh",
    "noise_0.75":  "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_randp0.75_rmode-all_nopersonalities_fixed-sesh",
    "noise_1.0":   "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_randp1_rmode-all_nopersonalities_fixed-sesh",
    "mem_0":       "th0_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh",
    "mem_2":       "th2_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh",
    "mem_10":      "th10_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh",
    "mem_20":      "th-1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh",
    "agents_10":   "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_traits10_fixed-sesh",
    "agents_100":  "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_traits100_fixed-sesh",
    "agents_1000": "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_traits1000_fixed-sesh",
    "random":      "ag20_tb-1_scheme-toggle_randp2_rmode-all_nopersonalities_fixed-sesh",
}
# Seeds to build a sprite set for, per arc. The metrics table averages over the
# replicate seeds s3/s4/s5; the gallery now exposes all three via a seed sub-toggle.
# Override with e.g. SEEDS=4 (env) to rebuild a single seed.
SEEDS = [int(s) for s in os.environ.get("SEEDS", "3,4,5").split(",") if s.strip()]


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def build_arc(arc, seed):
    full_run = f"{ARC_TO_RUN[arc]}_s{seed}"
    sprite_dir = SITE / full_run / "sprite"
    if (sprite_dir / "layout.json").is_file():
        print(f"[skip] {arc} s{seed} ({full_run}) — sprite already built", flush=True)
        return full_run
    print(f"[build] {arc} s{seed} -> {full_run}", flush=True)
    lib = ASSETS / f"{arc}_s{seed}_lib"
    run([PY, REPO / "tools" / "build_archive_image_lib.py", f"{arc}_s{seed}"])
    run([PY, REPO / "archive_animations" / "make_sprite_sheets.py",
         "--thumbs", lib / "thumbs", "--layout", lib / "layout.json",
         "--out", sprite_dir, "--ext", "webp"])
    shutil.rmtree(lib, ignore_errors=True)   # don't leave ~3k thumbs bloating the site repo
    return full_run


def write_index():
    """Scan every sprite dir in the mirror and emit one index entry per (arc, seed).

    Existing per-run fields (e.g. the human row's `base` pointing at gh-pages, and
    its seed=None) are preserved by merging over the prior index.json.
    """
    idx_path = MIRROR_ROOT / "index.json"
    prior = {}
    if idx_path.is_file():
        try:
            prior = {e["run"]: e for e in json.load(open(idx_path)).get("runs", [])}
        except Exception:
            prior = {}
    entries = []
    for lay_path in sorted(SITE.glob("*/sprite/layout.json")):
        full_run = lay_path.parent.parent.name
        lay = json.load(open(lay_path))
        e = dict(prior.get(full_run, {}))      # carry over base / extra fields
        e.update({"run": full_run, "arc": lay.get("arc"), "seed": lay.get("seed"),
                  "label": e.get("label", lay.get("arc")), "n_images": lay.get("n")})
        has = dict(e.get("has", {})); has["sprite"] = True; e["has"] = has
        entries.append(e)
    idx = {"dataset": "picbreeder-vlm/picbreeder-vlm-archive", "n_runs": len(entries),
           "runs": sorted(entries, key=lambda e: e["run"])}
    idx_path.write_text(json.dumps(idx, indent=1))
    print(f"[index] wrote {len(entries)} sprite runs -> {idx_path}", flush=True)


def main():
    only = sys.argv[1:] or list(ARC_TO_RUN)
    for arc in only:
        for seed in SEEDS:
            try:
                build_arc(arc, seed)
            except Exception as e:
                print(f"[ERROR] {arc} s{seed}: {e}", flush=True)
    write_index()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
