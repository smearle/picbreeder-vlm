#!/usr/bin/env python3
"""Export the data that powers the in-browser Picbreeder replica (blog `breed/`).

Produces `breed/data/archive.json`, a single file that the static page consumes:
  - `items`:  per-image record { title, model, agent, gen, added_at, rating, nrat,
              children, color, g (the CPPN genome as JSON for client-side render+branch) }
  - `lists`:  Picbreeder browse categories, each an ordered list of item keys:
                editors        -> the banner / hero-grid images (cross-run)
                highest_rated  -> VLM mean rating, descending
                best_new       -> recently-added, high-rated
                newest         -> most recently added (run-time chronology)
                most_branched  -> most published children
                random         -> a shuffled sample
  - `teasers`: a couple of keys for the homepage Branch / Family-Tree boxes.

Browse categories are drawn from one canonical, coherent run (so chronology /
branch counts / ratings are mutually consistent). Editor's Picks are the actual
banner images, whose genomes are recovered either from teaser provenance or by
render-matching the published thumbnail against locally-available genomes.

The genomes render pixel-identically in the browser (verified), so we ship only
genome JSON — no PNG thumbnails — keeping the payload small.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import neat
from neat_components import (
    InteractiveStagnation,
    PicbreederGenome,
    apply_picbreeder_config_defaults,
)
from picbreeder_reproduction import PicbreederReproduction
from picture2d.common import eval_genome_as_grayscale_and_color
from genome_json import genome_to_json as _genome_to_json

REPO = Path(__file__).resolve().parent.parent
BLOG = Path.home() / "smearle.github.io" / "picbreeder-vlm-06b0d76d"
HERO = BLOG / "assets" / "hero_sprites"
OUT = BLOG / "breed" / "data"
CANON = REPO / "sweep_logs/sweep/th-1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s3"

# How many items per browse category (kept modest to bound payload).
N_RATED, N_NEW, N_NEWEST, N_BRANCHED, N_RANDOM = 36, 36, 36, 36, 48
MATCH_RES = 48          # resolution for render-matching hero thumbnails
MATCH_MAX_MSE = 1300.0  # accept a hero genome match below this (jpeg-noisy)
MATCH_MAX_CAND = 20     # cap candidate genomes per hero (provenance-exact first)

random.seed(7)

# ---------------------------------------------------------------- NEAT config
_config = neat.Config(
    PicbreederGenome,
    PicbreederReproduction,
    neat.DefaultSpeciesSet,
    InteractiveStagnation,
    str(REPO / "picture2d" / "interactive_config_color"),
)
apply_picbreeder_config_defaults(_config, enable_output_activations=True)


def render_np(genome, size, color):
    gray, col = eval_genome_as_grayscale_and_color(genome, _config, size, size)
    if color:
        return np.asarray(col, dtype=np.float64)
    arr = (np.asarray(gray) * 255.0)
    return np.stack([arr, arr, arr], axis=-1)


def genome_to_json(genome, color=None):
    # `color` is accepted for call-site compatibility but the schema is color-agnostic
    # (the item's own `color` flag drives render mode); delegate to the shared serializer.
    return _genome_to_json(genome)


def mean_rating(entry):
    rs = [r for r in (entry.get("vlm_ratings") or []) if isinstance(r, (int, float))]
    return (sum(rs) / len(rs), len(rs)) if rs else (None, 0)


def parse_ts(s):
    try:
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return 0.0


def model_from_path(p: str) -> str:
    m = re.search(r"model-([a-z0-9.\-]+?)_tb", p)
    return m.group(1) if m else "gemini-2.5-pro"


def agent_num(agent_id: str) -> str:
    m = re.search(r"(\d+)", agent_id or "")
    return str(int(m.group(1))) if m else (agent_id or "?")


# ---------------------------------------------------------------- caption tags
# The VLM captioned every archive image (one-sentence descriptions, e.g. "A
# stylized black-and-white owl."). We mine those into the "Top Categories" tag
# cloud and a tag -> [item] index that browse.html?tag= filters on. The goal is
# concrete OBJECT nouns (owl, mask, teapot) — so we drop function words and the
# ubiquitous style/colour descriptors that otherwise dominate (abstract,
# symmetrical, grayscale, swirling, ...). No NLP deps available, so this is a
# curated stop/descriptor list + naive singularisation rather than POS tagging.
_TAG_STOP = set("""a an the of with and or in on at to from for as is are was were be this that these
those it its into out over under near by between within without around about above below up down left
right their your our his her some many few several pair series set group collection number lots more
most against onto upon off through behind front back side which who what when where while also than
then like likely possibly perhaps maybe appears appearing seems seeming looks looking resembling
resembles digitally""".split())
_TAG_DESCRIPTOR = set("""abstract symmetrical symmetric asymmetric pattern design image picture shape
form figure background object outline line drawing drawn sketch sketched doodle render rendering
rendered illustration swirling swirl swirled curved curving curve wavy flowing flow concentric
radiating spiral spiraling intricate complex simple stylized stylised ornate detailed smooth rough
soft sharp blurry blurred fuzzy distorted glowing iridescent shiny metallic glossy matte textured
gradient geometric organic surreal psychedelic vibrant minimalist minimal plain plainly clean busy
decorative abstractly cartoon cartoonish comic finish winged dark light bright pale deep vivid muted
dull colorful multicolored monochrome grayscale greyscale neon black white gray grey red blue green
yellow purple orange pink brown gold golden silver pastel teal cyan magenta violet beige tan turquoise
lavender crimson maroon navy olive indigo shade tone hue colored coloured colour colours color colors
warm cool earthy vertical horizontal diagonal central centered centred round rounded circular oval
elliptical spherical small large big tiny huge wide narrow tall short thin thick long flat single
multiple various different main outer inner top bottom middle edge close closeup pop warped twisted
melted floating hovering layered nested overlapping repeating tiled mirrored highlight highlighted
shaped""".split())


def _singular(w: str) -> str:
    if len(w) <= 3:
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "sses", "xes")):
        return w[:-2]
    if w.endswith("ses"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def caption_tags(caption: str) -> list[str]:
    """Object-noun tags (lowercase, singularised, de-duped) mined from a caption."""
    out, seen = [], set()
    for w in re.findall(r"[a-zA-Z]+", (caption or "").lower()):
        if len(w) < 3 or w in _TAG_STOP or w in _TAG_DESCRIPTOR:
            continue
        s = _singular(w)
        if len(s) < 3 or s in _TAG_STOP or s in _TAG_DESCRIPTOR or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


# =============================================================== canonical run
print(f"[canon] loading {CANON.name}")
meta = json.load(open(CANON / "archive" / "archive_metadata.json"))
entries = meta["entries"]
gdir = CANON / "archive" / "genomes"
canon_model = model_from_path(str(CANON))

# Per-image VLM captions ({ "img_000001.png": "A stylized owl." }); used to tag
# the bundled canonical items. Captions exist for the first ~2000 ids only, so
# some bundled items are untagged — that's fine, they just don't appear in any
# category list.
_cap_path = CANON / "archive" / f"captions_{canon_model}.json"
captions = json.load(open(_cap_path)) if _cap_path.exists() else {}
print(f"[canon] {len(captions)} captions from {_cap_path.name}")

recs = []
for e in entries:
    gp = gdir / (e["id"] + ".pkl")
    if not gp.exists():
        continue
    r, nr = mean_rating(e)
    recs.append(
        {
            "id": e["id"],
            "title": e.get("title") or "Untitled",
            "agent": agent_num(e.get("agent_id", "")),
            "gen": e.get("generation"),
            "ts": parse_ts(e.get("added_at", "")),
            "added_at": e.get("added_at", ""),
            "rating": r,
            "nrat": nr,
            "children": int(e.get("n_published_children") or 0),
            "color": bool(e.get("color_enabled", True)),
            "gp": gp,
        }
    )
print(f"[canon] {len(recs)} entries with local genomes")

by_id = {r["id"]: r for r in recs}
rated = [r for r in recs if r["rating"] is not None and r["nrat"] >= 5]

highest_rated = sorted(rated, key=lambda r: (-r["rating"], -r["nrat"]))[:N_RATED]
newest = sorted(recs, key=lambda r: -r["ts"])[:N_NEWEST]
# "best new": among the most-recently-added third, the highest rated
recent_cut = sorted(recs, key=lambda r: -r["ts"])[: max(1, len(recs) // 3)]
recent_rated = [r for r in recent_cut if r["rating"] is not None and r["nrat"] >= 3]
best_new = sorted(recent_rated, key=lambda r: (-r["rating"], -r["ts"]))[:N_NEW]
most_branched = sorted([r for r in recs if r["children"] > 0], key=lambda r: -r["children"])[:N_BRANCHED]
random_pick = random.sample(recs, min(N_RANDOM, len(recs)))

# =============================================================== editor's picks (banner)
print("[hero] resolving banner genomes")
hero_manifest = json.load(open(HERO / "manifest.json"))
titles = json.load(open(HERO / "titles.json"))
prov = json.load(open(REPO / "archive_animations" / "teaser_provenance.json"))


def hero_title(hid):
    frames = titles.get(hid, {}).get("frames", [])
    for f in reversed(frames):
        if f.get("title"):
            return f["title"]
    return hid


# precompute candidate genome paths per image-id across all local runs
print("[hero] indexing local genome files by id")
id_to_paths: dict[str, list[Path]] = {}
for base in (REPO / "sweep_logs", REPO / "sweep_logs_bkp"):
    if not base.exists():
        continue
    for gp in base.glob("**/archive/genomes/*.pkl"):
        id_to_paths.setdefault(gp.stem, []).append(gp)

editors = []
hero_items = {}
match_report = []
for hi, h in enumerate(hero_manifest):
    hid = h["id"]
    print(f"[hero] {hi + 1}/{len(hero_manifest)} {hid}", flush=True)
    thumb = HERO / (hid + "_thumb.jpg")
    if not thumb.exists():
        continue
    ref = np.asarray(Image.open(thumb).convert("RGB").resize((MATCH_RES, MATCH_RES)), dtype=np.float64)

    cand_paths: list[Path] = []
    prov_path = prov.get(hid + ".png")
    if prov_path:
        pp = REPO / prov_path.replace("/images/", "/genomes/").replace(".png", ".pkl")
        if pp.exists():
            cand_paths.append(pp)
    cand_paths += [p for p in id_to_paths.get(hid, []) if p not in cand_paths]
    cand_paths = cand_paths[:MATCH_MAX_CAND]

    best = None  # (mse, color, genome, path)
    for cp in cand_paths:
        try:
            g = pickle.load(open(cp, "rb"))
        except Exception:
            continue
        for color in (True, False):
            try:
                im = render_np(g, MATCH_RES, color)
            except Exception:
                continue
            mse = float(np.mean((im - ref) ** 2))
            if best is None or mse < best[0]:
                best = (mse, color, g, cp)
    if best is None:
        match_report.append((hid, "NO-CANDIDATE", None))
        continue
    mse, color, g, cp = best
    match_report.append((hid, round(mse, 1), cp.parent.parent.parent.name))
    key = "h_" + hid
    item = {
        "title": hero_title(hid),
        "model": model_from_path(str(cp)) if mse <= MATCH_MAX_MSE else "gemini-2.5-pro",
        "agent": "",
        "gen": None,
        "rating": None,
        "nrat": 0,
        "children": None,
        "color": color,
    }
    if mse <= MATCH_MAX_MSE:
        # exact genome recovered -> render crisply + fully branchable
        item["g"] = genome_to_json(g, color)
    else:
        # genome lives only on the cluster -> show the published thumbnail (view-only)
        item["thumb"] = f"../assets/hero_sprites/{hid}_thumb.jpg"
    hero_items[key] = item
    editors.append((key, mse <= MATCH_MAX_MSE))

# matched (branchable) banner images first, then view-only ones
editors_matched = [k for (k, m) in editors if m]
editors_view = [k for (k, m) in editors if not m]
editors = editors_matched + editors_view
ok = len(editors_matched)
print(f"[hero] matched {ok}/{len(hero_manifest)} banner images (<= MSE {MATCH_MAX_MSE}); {len(editors_view)} view-only")
for hid, m, run in sorted(match_report, key=lambda x: (x[1] if isinstance(x[1], float) else 1e9)):
    flag = "" if (isinstance(m, float) and m <= MATCH_MAX_MSE) else "  [skip]"
    print(f"        {hid}  mse={m}  {run}{flag}")

# =============================================================== top artists
# In the personality-traits sweep, every agent is assigned ONE second-person
# personality trait (from personality_traits.json) and all agents share the same
# model (gemini-2.5-pro) and image-rating prompt. So each distinct trait is a
# fair "artist": we attribute every published image's mean community rating to
# the trait of the agent that made it, pooling across the traits10/100/1000 runs
# and their seeds. (This is the comparison the earlier per-config leaderboard
# could not make fairly, since different configs rate with different settings.)
SWEEP_DIR = REPO / "sweep_logs" / "sweep"
TRAITS_META = REPO / "traits_meta_cluster"   # compact {agent_id: trait} maps per run
TRAITS_GENOMES = REPO / "traits_genomes_cluster"  # genomes pulled for user-publication pages
USERNAMES = json.load(open(REPO / "traits_usernames.json"))
ARTIST_MIN_IMGS = 50    # images attributed to a trait before it can rank
N_ARTISTS = 8           # artists shown in the homepage sidebar leaderboard


def slugless(trait: str) -> str:
    """Username for a trait; fall back to a trimmed trait if unseen."""
    return USERNAMES.get(trait) or re.sub(r"\W+", "", trait.title())[:18] or "Artist"


def run_tag(run: str) -> str:
    """Short, unique-per-run prefix for publication item keys (e.g. t10s3)."""
    m = re.search(r"traits(\d+).*_(s\d+)$", run)
    return f"t{m.group(1)}{m.group(2)}" if m else re.sub(r"\W+", "", run)[-8:]


def genome_path_for(run: str, img_id: str) -> Path | None:
    """Genome pkl for a traits-run image: local sweep_logs first, then the pulled
    cache (which mirrors the sweep layout: <run>/archive/genomes/<id>.pkl)."""
    rel = Path(run) / "archive" / "genomes" / (img_id + ".pkl")
    for base in (SWEEP_DIR, TRAITS_GENOMES):
        p = base / rel
        if p.exists():
            return p
    return None


print("[artists] attributing image ratings to agent personality traits")
trait_ratings: dict[str, list] = {}
trait_seeds: dict[str, set] = {}
trait_entries: dict[str, list] = {}   # trait -> [{run, id, rating, title, color, agent}]
n_traits_runs = 0
for amap in sorted(TRAITS_META.glob("*/agent_traits.json")):
    run = amap.parent.name
    arch = SWEEP_DIR / run / "archive" / "archive_metadata.json"
    if not arch.exists():
        print(f"        [skip] no local archive for {run}")
        continue
    n_traits_runs += 1
    seed = run.rsplit("_", 1)[-1]
    agent_to_trait = json.load(open(amap))
    for e in json.load(open(arch))["entries"]:
        trait = agent_to_trait.get(e.get("agent_id"))
        if not trait:
            continue
        r, nr = mean_rating(e)
        if r is None:
            continue
        trait_ratings.setdefault(trait, []).append(r)
        trait_seeds.setdefault(trait, set()).add(seed)
        trait_entries.setdefault(trait, []).append({
            "run": run, "id": e["id"], "rating": r, "nrat": nr,
            "children": int(e.get("n_published_children") or 0),
            "title": e.get("title") or "Untitled",
            "color": bool(e.get("color_enabled", True)),
            "agent": agent_num(e.get("agent_id", "")),
        })

artist_rows = []
for trait, rl in trait_ratings.items():
    if len(rl) < ARTIST_MIN_IMGS:
        continue
    children = sum(c["children"] for c in trait_entries.get(trait, []))
    artist_rows.append({
        "user": slugless(trait), "trait": trait,
        "rating": round(sum(rl) / len(rl), 2), "nimg": len(rl),
        "seeds": len(trait_seeds[trait]),
        "branched": children,   # total published children across this trait's images
    })
artist_rows.sort(key=lambda r: -r["rating"])
artists = artist_rows[:N_ARTISTS]          # sidebar leaderboard (top-rated)
all_artists = artist_rows                  # full set for the All-Personalities page
n_artist_cfgs = len(artist_rows)
print(f"[artists] {n_traits_runs} traits runs; {len(trait_ratings)} traits seen, "
      f"{n_artist_cfgs} with >= {ARTIST_MIN_IMGS} imgs; top {len(artists)}:")
for r in artists:
    print(f"        {r['rating']:.2f}  {r['user']:<22}  [{r['seeds']} seeds, {r['nimg']} imgs, "
          f"{r['branched']} branched]  {r['trait'][:50]}")

# Per-artist galleries are NOT embedded: every published image is referenced by a
# `run::id` key, and browse.html lazily fetches each run's genomes.json.gz from the
# HF archive on demand (the same path whole-run galleries use). So every ranked
# personality gets its complete, branchable gallery with zero bundled genomes.
print("[artists] building per-user publication reference lists (run::id, HF-lazy)")
user_pubs: dict[str, list] = {}   # username -> entries (all, rating-sorted)
for a in all_artists:
    cands = sorted(trait_entries.get(a["trait"], []), key=lambda x: -x["rating"])
    user_pubs[a["user"]] = cands
    a["npub"] = len(cands)

# =============================================================== assemble
items = {}


def add_canon(r):
    key = "c_" + r["id"]
    if key in items:
        return key
    g = pickle.load(open(r["gp"], "rb"))
    cap = captions.get(r["id"] + ".png")
    items[key] = {
        "title": r["title"],
        "model": canon_model,
        "agent": r["agent"],
        "gen": r["gen"],
        "added_at": r["added_at"],
        "rating": (round(r["rating"], 2) if r["rating"] is not None else None),
        "nrat": r["nrat"],
        "children": r["children"],
        "color": r["color"],
        "g": genome_to_json(g, r["color"]),
    }
    if cap:
        items[key]["caption"] = cap
        tags = caption_tags(cap)
        if tags:
            items[key]["tags"] = tags
    return key


def keys_for(lst):
    return [add_canon(r) for r in lst]

lists = {
    "editors": editors,
    "highest_rated": keys_for(highest_rated),
    "best_new": keys_for(best_new),
    "newest": keys_for(newest),
    "most_branched": keys_for(most_branched),
    "random": keys_for(random_pick),
}
items.update(hero_items)

# Per-artist publication galleries (clicking a username): slim cards written to a
# SEPARATE browse-only file (data/user_cards.json), not bundled into the homepage
# archive.json. browse.html renders each cell from its card (title/rating/agent)
# and the run's HF sprite-sheet thumbnail; the genome is fetched lazily only on
# Evolve / DNA. Runs are de-duped into a table since each 89-char name repeats
# across ~140 of a user's images. Mirrors tools/build_user_cards.py (kept as a
# dependency-free standalone for quick refreshes).
_run_table: list[str] = []
_run_idx: dict[str, int] = {}


def _run_i(run: str) -> int:
    if run not in _run_idx:
        _run_idx[run] = len(_run_table)
        _run_table.append(run)
    return _run_idx[run]


user_cards: dict[str, list] = {}
for username, pubs in user_pubs.items():
    user_cards[username] = [[
        _run_i(c["run"]), int(c["id"].split("_")[-1]), c["title"],
        (round(c["rating"], 2) if c["rating"] is not None else None),
        c.get("nrat", 0), c["agent"], 1 if c["color"] else 0, c["children"],
    ] for c in pubs]
json.dump({"runs": _run_table, "users": user_cards},
          open(OUT / "user_cards.json", "w"), separators=(",", ":"))
print(f"[artists] {sum(len(v) for v in user_cards.values())} publication cards "
      f"across {len(user_cards)} users -> user_cards.json (sprite thumbs, lazy genomes)")

# Caption-derived categories over the bundled canonical items. tag_index maps
# each object tag -> the item keys carrying it (browse.html?tag=); top_tags is
# the homepage cloud (tags on >=2 images, frequency-ranked, capped). Singletons
# stay searchable via tag_index but are kept out of the cloud to avoid noise.
from collections import Counter as _Counter  # noqa: E402  (local, near use)

tag_index: dict[str, list] = {}
for _k, _it in items.items():
    for _t in _it.get("tags", []):
        tag_index.setdefault(_t, []).append(_k)
_tag_freq = _Counter({t: len(ks) for t, ks in tag_index.items()})
top_tags = [[t, c] for t, c in _tag_freq.most_common() if c >= 2][:30]
print(f"[tags] {len(tag_index)} distinct tags over bundled items; "
      f"{len(top_tags)} in cloud (>=2 imgs). top: "
      + ", ".join(f"{t}({c})" for t, c in top_tags[:12]))

teasers = {
    "branch": editors_matched[:2] if len(editors_matched) >= 2 else lists["highest_rated"][:2],
    "tree": editors_matched[2:4] if len(editors_matched) >= 4 else lists["most_branched"][:2],
    "merch": editors_matched[4:6] if len(editors_matched) >= 6 else lists["highest_rated"][2:4],
}

data = {
    "canon_run": CANON.name,
    "canon_model": canon_model,
    "n_archive": len(recs),
    "items": items,
    "lists": lists,
    "tag_index": tag_index,
    "top_tags": top_tags,
    "teasers": teasers,
    "artists": artists,
    "all_artists": all_artists,
    "n_artist_users": n_artist_cfgs,
}

OUT.mkdir(parents=True, exist_ok=True)
out_path = OUT / "archive.json"
json.dump(data, open(out_path, "w"), separators=(",", ":"))
size_kb = out_path.stat().st_size / 1024
print(f"[done] {out_path}  ({len(items)} items, {size_kb:.0f} KB)")
print("       lists:", {k: len(v) for k, v in lists.items()})
