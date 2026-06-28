#!/usr/bin/env python3
"""Build sprite archives for runs the hero banner references but that aren't yet in
the local mirror (extra seeds, gemini-3 / flash-lite models). Stages them next to
the published table runs so the inline archive viewer can open ANY hero cell's run.

  python tools/build_extra_sprites.py            # build every missing hero run
  python tools/build_extra_sprites.py --dry-run  # just list what would build

Per run -> site/<run>/sprite/{layout.json, sprites.json, sheets/*.webp}
Then rewrites the mirror index.json over ALL sprite runs, now carrying `model` and
a full `config` per entry (so the viewer can resolve arc+model+seed locally).
No HF push — that's tools/push_sprites.py + tools/build_hf_index.py once reviewed.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/jupyter-smearle/picbreeder-vlm")
sys.path.insert(0, str(REPO / "tools"))
from hf_archive_push import parse_config, canonical_arc, is_canonical_run   # noqa: E402

SWEEP = REPO / "sweep_logs" / "sweep"
MIRROR = REPO / "archive_animations" / "_archive_mirror"
SITE = MIRROR / "site"
ASSETS = Path("/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets")
PROV = ASSETS / "hero_sprites" / "provenance.json"
PY = sys.executable


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def build_run(run_name: str):
    sprite_dir = SITE / run_name / "sprite"
    if (sprite_dir / "layout.json").is_file():
        print(f"[skip] {run_name} — sprite already built", flush=True)
        return
    if not (SWEEP / run_name).is_dir():
        print(f"[MISS] {run_name} — no run dir on disk; skipping", flush=True)
        return
    print(f"[build] {run_name}", flush=True)
    lib = ASSETS / "archives" / f"{run_name}_lib"
    run([PY, REPO / "tools" / "build_archive_image_lib.py", "--run", run_name, "--out", lib])
    run([PY, REPO / "archive_animations" / "make_sprite_sheets.py",
         "--thumbs", lib / "thumbs", "--layout", lib / "layout.json",
         "--out", sprite_dir, "--ext", "webp"])
    shutil.rmtree(lib, ignore_errors=True)


def write_index():
    idx_path = MIRROR / "index.json"
    prior = {}
    if idx_path.is_file():
        prior = {e["run"]: e for e in json.load(open(idx_path)).get("runs", [])}
    entries = []
    for lay_path in sorted(SITE.glob("*/sprite/layout.json")):
        full_run = lay_path.parent.parent.name
        lay = json.load(open(lay_path))
        cfg = parse_config(full_run)
        e = dict(prior.get(full_run, {}))            # carry over base / label / has extras
        arc = lay.get("arc") or canonical_arc(full_run, cfg)
        e.update({"run": full_run, "arc": arc, "seed": lay.get("seed", cfg.get("seed")),
                  "model": lay.get("model", cfg.get("model")), "config": cfg,
                  "canonical": is_canonical_run(full_run, cfg),
                  "label": e.get("label", arc), "n_images": lay.get("n")})
        has = dict(e.get("has", {})); has["sprite"] = True; e["has"] = has
        entries.append(e)
    idx = {"dataset": "picbreeder-vlm/picbreeder-vlm-archive", "n_runs": len(entries),
           "runs": sorted(entries, key=lambda e: e["run"])}
    idx_path.write_text(json.dumps(idx, indent=1))
    print(f"[index] wrote {len(entries)} sprite runs -> {idx_path}", flush=True)


def main():
    dry = "--dry-run" in sys.argv
    prov = json.loads(PROV.read_text())
    have = {p.parent.parent.name for p in SITE.glob("*/sprite/layout.json")}
    want = sorted({v["run"] for v in prov.values()})
    missing = [r for r in want if r not in have]
    print(f"{len(want)} hero runs · {len(missing)} missing locally:")
    for r in missing:
        print("   ", r)
    if dry:
        return
    for r in missing:
        try:
            build_run(r)
        except Exception as e:
            print(f"[ERROR] {r}: {e}", flush=True)
    write_index()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
