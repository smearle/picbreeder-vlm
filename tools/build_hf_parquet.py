#!/usr/bin/env python
"""Build a HuggingFace-native, browsable *face* of the generated-image corpus on top of
the existing web-asset layout in `picbreeder-vlm/picbreeder-vlm-archive`.

The blog needs the sprite atlases (`site/<run>/sprite/sheets/*.webp`), so those stay. But an
atlas is unreadable to the HF Dataset Viewer, so this script emits the missing viewer face:

  data/images/<split>/<run>.parquet

one row per generated image, with the 128px thumbnail embedded under a `datasets.Image()` feature
(so the Viewer renders thumbnails and `load_dataset` decodes them), plus per-image provenance and
the run's *config axes as first-class columns* — `memory_cl`, `noise_eps`, `n_personalities`, `model`,
`seed` — rather than leaving them buried in the run-name string.

Captions get one column per captioner (`caption` = the paper's gemini-2.5-pro pass,
`caption_qwen3_vl_8b` = a paired second opinion on 13 runs), since a caption describes an
(image, captioner) pair rather than the image alone.

Pixels come from the atlas (the exact thumbnails already published); full-resolution renders are
reproducible from the CPPN genomes bundled at `site/<run>/genomes.json.gz` / `results/<run>/genomes.tar.gz`.

Splits are the experimental *condition* (arc) — NOT an ML `train` split, which this generated,
un-partitioned corpus does not have. VLM-model ablations (no arc) land in `model_ablation`.

  python tools/build_hf_parquet.py --out /path/to/staging [--limit N] [--runs substr] [--dry-run]
  python tools/build_hf_parquet.py --out /path/to/staging --push        # upload data/ + emit configs
"""
import argparse, io, json, re, sys
from pathlib import Path

from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from hf_archive_push import parse_config, canonical_arc, is_canonical_run  # noqa: E402

REPO = "picbreeder-vlm/picbreeder-vlm-archive"
ROOT = Path("/home/jupyter-smearle/picbreeder-vlm")
MIRROR_SITE = ROOT / "archive_animations/_archive_mirror/site"
# where a run's archive/ (metadata + captions) may live locally, first hit wins
META_BASES = ["sweep_logs/sweep", "logs_collaborative", "logs"]

# A run may carry captions from SEVERAL captioner models: `captions_<model>.json` records the
# *captioner*, not the run's breeder model. A caption is a fact about an (image, captioner) pair, so
# each captioner gets its OWN column -- never merge them into one and never let one clobber another.
PRIMARY_CAPTIONER = "gemini-2.5-pro"          # -> `caption`
SECONDARY_CAPTIONERS = {"qwen3-vl-8b": "caption_qwen3_vl_8b"}

FEATURES = Features({
    "image": HFImage(),
    "image_id": Value("string"),
    # --- run identity + config axes (parsed out of the run name, not left embedded in it) ---
    "run": Value("string"),
    "arc": Value("string"),
    "model": Value("string"),
    "seed": Value("int64"),
    "memory_cl": Value("int64"),
    "noise_eps": Value("float64"),
    "n_personalities": Value("int64"),
    "canonical": Value("bool"),
    # --- per-image provenance ---
    "generation": Value("int64"),
    "agent_id": Value("string"),
    "genome_key": Value("int64"),
    "parent_genome_key": Value("int64"),
    "n_published_children": Value("int64"),
    "color_enabled": Value("bool"),
    # --- VLM annotations. One column per captioner; `caption` is the paper's gemini-2.5-pro pass. ---
    "caption": Value("string"),
    "caption_qwen3_vl_8b": Value("string"),
    "vlm_rating_mean": Value("float64"),
    "vlm_rating_count": Value("int64"),
})


def sanitize_split(arc):
    """Arc -> valid HF split name (lowercase [a-z0-9_], no leading digit)."""
    if arc is None:
        return "model_ablation"
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(arc)).strip("_").lower()
    if s and s[0].isdigit():
        s = "x_" + s
    return s or "other"


def find_archive_dir(run):
    for base in META_BASES:
        p = ROOT / base / run / "archive"
        if (p / "archive_metadata.json").is_file():
            return p
    return None


def load_caption_sets(archive_dir):
    """-> {captioner_model: {img_XXXXXX.png: caption}}. Every captioner kept separate."""
    return {f.name[len("captions_"):-len(".json")]: json.load(open(f))
            for f in archive_dir.glob("captions_*.json")}


def n_thumbs(sprite_dir):
    """How many images the atlas actually contains."""
    return json.load(open(sprite_dir / "sprites.json"))["n"]


def atlas_thumbs(sprite_dir, n):
    """Yield (index, WebP bytes) for images 0..n-1 decoded from the sheet atlas.
    Packing matches make_sprite_sheets.py: i -> sheet i//per_sheet, cell i%per_sheet."""
    sp = json.load(open(sprite_dir / "sprites.json"))
    cell, cols, per = sp["cell"], sp["sheet_cols"], sp["per_sheet"]
    tmpl = sp["sheet_tmpl"]
    sheet, sheet_idx = None, -1
    for i in range(n):
        s = i // per
        if s != sheet_idx:
            sheet = Image.open(sprite_dir / tmpl.format(s=s)).convert("RGB")
            sheet_idx = s
        within = i % per
        x, y = (within % cols) * cell, (within // cols) * cell
        buf = io.BytesIO()
        # re-encode as WebP to match the source atlas (~5x smaller than PNG, viewer-decodable)
        sheet.crop((x, y, x + cell, y + cell)).save(buf, format="WEBP", quality=88, method=4)
        yield i, buf.getvalue()


def build_run(run, out_dir):
    """Write out_dir/<split>/<run>.parquet. Returns (split, n_rows) or None if skipped."""
    cfg = parse_config(run)
    arc = canonical_arc(run, cfg)
    split = sanitize_split(arc)
    dest = out_dir / split / f"{run}.parquet"
    if dest.exists():
        import pyarrow.parquet as pq
        return (split, pq.read_metadata(dest).num_rows)

    sprite_dir = MIRROR_SITE / run / "sprite"
    archive_dir = find_archive_dir(run)
    if not (sprite_dir / "sprites.json").is_file() or archive_dir is None:
        return None

    meta = json.load(open(archive_dir / "archive_metadata.json"))
    entries = meta.get("entries", [])
    cap_sets = load_caption_sets(archive_dir)
    primary = cap_sets.get(PRIMARY_CAPTIONER, {})

    # The atlas can lag the metadata (a run publishes an entry after its sprite sheets were packed).
    # Entries beyond the atlas have NO pixels -- cropping them yields the sheet's blank background,
    # so drop the tail rather than emit a phantom grey image with real metadata attached.
    n_thumb = n_thumbs(sprite_dir)
    if len(entries) != n_thumb:
        print(f"    [warn] {run}: {len(entries)} entries vs {n_thumb} thumbnails "
              f"-> dropping {len(entries) - n_thumb} un-rendered entr(y|ies)", flush=True)
    n = min(len(entries), n_thumb)
    entries = entries[:n]

    cols = {k: [] for k in FEATURES}
    thumbs = {i: b for i, b in atlas_thumbs(sprite_dir, n)}
    canon = is_canonical_run(run, cfg)
    for i, e in enumerate(entries):
        iid = e.get("id") or f"img_{i + 1:06d}"
        ratings = [r for r in (e.get("vlm_ratings") or []) if isinstance(r, (int, float))]
        parents = e.get("parent_genome_keys") or []
        key = f"{iid}.png"
        cols["image"].append({"bytes": thumbs.get(i), "path": f"{iid}.webp"})
        cols["image_id"].append(iid)
        cols["run"].append(run)
        cols["arc"].append(arc)
        cols["model"].append(cfg.get("model"))
        cols["seed"].append(cfg.get("seed"))
        cols["memory_cl"].append(cfg.get("memory_cl"))
        cols["noise_eps"].append(cfg.get("noise_eps"))
        cols["n_personalities"].append(cfg.get("personalities"))
        cols["canonical"].append(bool(canon))
        cols["generation"].append(e.get("generation"))
        cols["agent_id"].append(e.get("agent_id"))
        cols["genome_key"].append(e.get("genome_key"))
        cols["parent_genome_key"].append(parents[0] if parents else None)
        cols["n_published_children"].append(e.get("n_published_children"))
        cols["color_enabled"].append(bool(e.get("color_enabled")))
        cols["caption"].append(primary.get(key))
        for m, col in SECONDARY_CAPTIONERS.items():
            cols[col].append(cap_sets.get(m, {}).get(key))
        cols["vlm_rating_mean"].append(sum(ratings) / len(ratings) if ratings else None)
        cols["vlm_rating_count"].append(len(ratings))

    dest.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict(cols, features=FEATURES).to_parquet(dest)
    return (split, n)


def emit_configs(out_dir):
    """Print the README `configs:` block from what actually got written."""
    splits = sorted(d.name for d in (out_dir / "data" / "images").iterdir() if d.is_dir())
    lines = ["configs:", "- config_name: images", "  data_files:"]
    for s in splits:
        lines.append(f"  - split: {s}")
        lines.append(f"    path: data/images/{s}/*.parquet")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="staging dir (parquet written under <out>/data/images/)")
    ap.add_argument("--limit", type=int, default=0, help="only build the first N runs (0 = all)")
    ap.add_argument("--runs", default="", help="only runs whose name contains this substring")
    ap.add_argument("--push", action="store_true", help="upload <out>/data to HF after building")
    ap.add_argument("--dry-run", action="store_true", help="list runs + splits, build nothing")
    args = ap.parse_args()

    out = Path(args.out)
    img_out = out / "data" / "images"
    runs = sorted(d.name for d in MIRROR_SITE.iterdir()
                  if (d / "sprite" / "sprites.json").is_file() and args.runs in d.name)
    if args.limit:
        runs = runs[:args.limit]

    from collections import Counter
    if args.dry_run:
        c = Counter(sanitize_split(canonical_arc(r, parse_config(r))) for r in runs)
        print(f"{len(runs)} runs -> {len(c)} splits")
        for s, n in sorted(c.items()):
            print(f"  {s}: {n} runs")
        return

    total_rows = 0
    per_split = Counter()
    for k, run in enumerate(runs, 1):
        res = build_run(run, img_out)
        if res is None:
            print(f"[{k}/{len(runs)}] SKIP {run} (no atlas or metadata)", flush=True)
            continue
        split, nrows = res
        per_split[split] += nrows
        total_rows += nrows
        print(f"[{k}/{len(runs)}] {run} -> {split} ({nrows} imgs)", flush=True)

    size_mb = sum(p.stat().st_size for p in img_out.rglob("*.parquet")) / 1e6
    print(f"\n{total_rows} rows across {len(per_split)} splits, {size_mb:.1f} MB parquet")
    for s, n in sorted(per_split.items()):
        print(f"  {s}: {n} rows")
    print("\n--- README configs block ---\n" + emit_configs(out))

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(out / "data"),
                          path_in_repo="data", allow_patterns=["**/*.parquet"],
                          commit_message=f"Rebuild image parquet ({total_rows} imgs, {len(per_split)} condition splits)")
        print(f"[push] uploaded {total_rows} rows -> {REPO}/data/images/")


if __name__ == "__main__":
    main()
