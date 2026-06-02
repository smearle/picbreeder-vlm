#!/usr/bin/env python
"""Build layout.json + sprite sheets for ONE representative seed of every results-table arc,
staging them into the blog's local archive mirror so the gallery can be tested for all rows
(and later pushed to HF). Reuses tools/build_archive_image_lib.py (layout + thumbs) and
archive_animations/make_sprite_sheets.py (packing), so the on-disk format matches the demo.

Per arc -> runs/<full_sweep_run>_s<seed>/sprite/{layout.json, sprites.json, sheets/*.webp}
Then writes runs/index.json over all sprite runs present in the mirror.

Idempotent: skips an arc whose sprite/layout.json already exists.
"""
import json, shutil, subprocess, sys
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
SEED = 4


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def build_arc(arc):
    full_run = f"{ARC_TO_RUN[arc]}_s{SEED}"
    sprite_dir = SITE / full_run / "sprite"
    if (sprite_dir / "layout.json").is_file():
        print(f"[skip] {arc} ({full_run}) — sprite already built", flush=True)
        return full_run
    print(f"[build] {arc} -> {full_run}", flush=True)
    lib = ASSETS / f"{arc}_s{SEED}_lib"
    run([PY, REPO / "tools" / "build_archive_image_lib.py", f"{arc}_s{SEED}"])
    run([PY, REPO / "archive_animations" / "make_sprite_sheets.py",
         "--thumbs", lib / "thumbs", "--layout", lib / "layout.json",
         "--out", sprite_dir, "--ext", "webp"])
    shutil.rmtree(lib, ignore_errors=True)   # don't leave ~3k thumbs bloating the site repo
    return full_run


def write_index(runs_by_arc):
    entries = []
    for arc, full_run in runs_by_arc.items():
        lay = json.load(open(SITE / full_run / "sprite" / "layout.json"))
        entries.append({"run": full_run, "arc": arc, "seed": SEED,
                        "label": arc, "n_images": lay.get("n"),
                        "has": {"sprite": True}})
    idx = {"dataset": "picbreeder-vlm/picbreeder-vlm-archive", "n_runs": len(entries),
           "runs": sorted(entries, key=lambda e: e["run"])}
    (MIRROR_ROOT / "index.json").write_text(json.dumps(idx, indent=1))
    print(f"[index] wrote {len(entries)} sprite runs -> {MIRROR_ROOT/'index.json'}", flush=True)


def main():
    only = sys.argv[1:] or list(ARC_TO_RUN)
    built = {}
    for arc in only:
        try:
            built[arc] = build_arc(arc)
        except Exception as e:
            print(f"[ERROR] {arc}: {e}", flush=True)
    write_index(built)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
