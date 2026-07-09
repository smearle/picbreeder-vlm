#!/usr/bin/env python3
"""Export the lineage data that powers the in-browser Picbreeder "Family Tree".

The original picbreeder.org served a `familytree.php` page hosting a Java
`TreeOfLifeApplet`: a pannable/zoomable node-link tree, rooted at a published
image, where clicking a node revealed the images other users branched from it.
We reconstruct that here. The branch relation among *published* images is
recorded directly in each archive entry's `source_entry_ids` (the published
elite(s) an agent started its branch from), which forms a clean single-parent
forest — exactly a family tree.

We export one **per-run lineage sidecar** for every run published to the HF
archive (the dataset already hosts each run's `site/<run>/genomes.json.gz`, which
the viewer streams on demand). Each sidecar is tiny (titles/ratings/agent/parent,
no genomes); the genomes themselves stay on HF except for the canonical run,
whose genome shard is bundled so the default view loads instantly and offline.

Outputs, under `breed/data/`:
  - `tree/manifest.json`        { canon, runs:[{run,label,group,model,seed,
                                  traits,n,depth}] } — drives the run dropdown
  - `tree/<run>.json.gz`        per run: { run, model, traits, n, roots,
                                  featured, default, nodes:{id:{t,a,g,r,n,c,col,
                                  p, u?}} }   (u = personality username, traits runs)
  - `tree.json.gz`              copy of the canonical run's sidecar (back-comp:
                                  play.html?tree= reads it for titles/colour)
  - `tree_genomes.json.gz`      the canonical run's genome shard, bundled

For traits runs (config.personalities > 0) each agent has one second-person
personality trait; we map agent→trait→username (the same attribution
build_breed_data.py uses for the "Top Artists" board) and store it as node.u, so
the viewer can caption a tile by its artist instead of "agent NN".
"""
from __future__ import annotations

import gzip
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parent.parent
BLOG = Path.home() / "smearle.github.io" / "picbreeder-vlm-06b0d76d"
OUT = BLOG / "breed" / "data"
TREE_OUT = OUT / "tree"
SWEEP = REPO / "sweep_logs" / "sweep"
SWEEP_ALT = REPO / "sweep_logs"
TRAITS_META = REPO / "traits_meta_cluster"   # <run>/agent_traits.json  (agent_id -> trait)
USERNAMES = json.load(open(REPO / "data" / "traits_usernames.json"))  # trait -> username

CANON = "th-1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s3"
N_FEATURED = 12
HF_REPO = "picbreeder-vlm/picbreeder-vlm-archive"

# Human Picbreeder archive: its genomes/metadata aren't on HF as a sweep run;
# build_human_genomes.py writes them here in the same archive_metadata format.
HUMAN_META = REPO / "archive_animations/out/human_genomes/results/human/archive_metadata.json"


def mean_rating(entry):
    rs = [r for r in (entry.get("vlm_ratings") or []) if isinstance(r, (int, float))]
    return (round(sum(rs) / len(rs), 2), len(rs)) if rs else (None, 0)


def agent_num(agent_id: str) -> str:
    m = re.search(r"(\d+)", agent_id or "")
    return str(int(m.group(1))) if m else (agent_id or "?")


def img_index(img_id: str) -> int:
    """`img_000042` -> 42 (the published, 1-based image index)."""
    return int(img_id.split("_")[1])


def meta_path_for(run: str) -> Path | None:
    for base in (SWEEP, SWEEP_ALT):
        p = base / run / "archive" / "archive_metadata.json"
        if p.exists():
            return p
    return None


def trait_usernames_for(run: str) -> dict[str, str]:
    """agent_id -> username for a traits run (empty if unavailable)."""
    amap = TRAITS_META / run / "agent_traits.json"
    if not amap.exists():
        return {}
    agent_to_trait = json.load(open(amap))
    out = {}
    for aid, trait in agent_to_trait.items():
        u = USERNAMES.get(trait)
        if u:
            out[aid] = u
    return out


def run_label(run: str, cfg: dict) -> tuple[str, str]:
    """(group, label) for the dropdown. Group = model; label = the run's
    distinguishing knobs (thinking budget, variant, seed)."""
    model = cfg.get("model") or "gemini-2.5-pro"
    tb = re.search(r"(?:^|_)th(-?\d+)_", run)
    bits = [f"think&nbsp;{tb.group(1)}" if tb else "think&nbsp;1"]
    if cfg.get("personalities"):
        bits.append(f"{cfg['personalities']}&nbsp;personas")
    mp = re.search(r"randp([0-9.]+)", run)
    if mp:
        bits.append(f"rand&nbsp;{mp.group(1)}")
    if "goal-none" in run:
        bits.append("no&nbsp;goal")
    if "ts224" in run:
        bits.append("tourney&nbsp;224")
    tm = re.search(r"temp([0-9a-z-]+)", run)
    if tm:
        bits.append(f"temp&nbsp;{tm.group(1)}")
    if "include-branch-img" in run:
        bits.append("branch&nbsp;img")
    seed = cfg.get("seed")
    if seed is None:
        s = re.search(r"_s(\d+)$", run)
        seed = int(s.group(1)) if s else "?"
    bits.append(f"s{seed}")
    return model, " · ".join(bits)


def genome_ids_for(run: str) -> set[str]:
    """Ids that actually have a renderable genome shard (the archive persists a
    subset — capacity-limited runs keep fewer genomes than published entries)."""
    gp = meta_path_for(run).parent / "genomes.json.gz" if meta_path_for(run) else None
    if not gp or not gp.exists():
        return set()
    try:
        with gzip.open(gp, "rt") as fh:
            return set(json.load(fh).keys())
    except Exception:
        return set()


def build_run(run: str, cfg: dict) -> dict | None:
    mp = meta_path_for(run)
    if mp is None:
        print(f"  [skip] no local metadata: {run}")
        return None
    entries = json.load(open(mp)).get("entries", [])
    if not entries:
        return None
    usernames = trait_usernames_for(run) if cfg.get("personalities") else {}
    return _assemble(run, cfg.get("model") or "gemini-2.5-pro",
                     int(cfg.get("personalities") or 0), entries,
                     genome_ids_for(run), usernames)


def _assemble(run: str, model: str, traits: int, entries: list,
              gids: set[str], usernames: dict, human: bool = False) -> dict | None:
    by_id = {e["id"]: e for e in entries}
    all_ids = set(by_id)
    if not all_ids:
        return None

    # Raw branch parent (the published elite this entry branched from).
    raw_parent = {}
    for e in entries:
        src = [s for s in (e.get("source_entry_ids") or []) if s in all_ids]
        raw_parent[e["id"]] = src[0] if src else None

    # Keep only nodes with a genome; re-link each survivor to its nearest
    # genome-having ancestor (contract edges through dropped nodes) so every
    # tile in the tree renders. Canon keeps all genomes → no change.
    ids = {i for i in all_ids if i in gids}
    if not ids:
        print(f"  [skip] no genomes for {run}")
        return None

    def nearest_kept_ancestor(i):
        p = raw_parent.get(i)
        while p is not None and p not in ids:
            p = raw_parent.get(p)
        return p

    parent = {i: nearest_kept_ancestor(i) for i in ids}

    children = defaultdict(list)
    for cid, pid in parent.items():
        if pid:
            children[pid].append(cid)
    for pid in children:
        children[pid].sort(key=lambda c: (-len(children.get(c, [])), c))

    nodes = {}
    for e in entries:
        cid = e["id"]
        if cid not in ids:
            continue
        r, nr = mean_rating(e)
        nd = {
            "t": e.get("title") or "Untitled",
            # human archive has no per-agent attribution (agent_id == "human")
            "a": "" if human else agent_num(e.get("agent_id", "")),
            "g": e.get("generation"),
            "r": r,
            "n": nr,
            "c": len(children.get(cid, [])),
            "col": bool(e.get("color_enabled", True)),
            "p": parent[cid],
        }
        u = usernames.get(e.get("agent_id"))
        if u:
            nd["u"] = u
        nodes[cid] = nd

    size, depth = {}, {}

    def subtree_size(x):
        if x in size:
            return size[x]
        s = 1
        for c in children.get(x, []):
            s += subtree_size(c)
        size[x] = s
        return s

    def dep(x):
        if x in depth:
            return depth[x]
        p = parent[x]
        depth[x] = 0 if p is None else dep(p) + 1
        return depth[x]

    for cid in ids:
        subtree_size(cid); dep(cid)

    roots = sorted((c for c in ids if parent[c] is None), key=lambda x: -size[x])
    featured = sorted(ids, key=lambda x: (-nodes[x]["c"], -size[x]))[:N_FEATURED]
    leaves = [c for c in ids if nodes[c]["c"] == 0]
    default = max(leaves or list(ids), key=lambda x: (depth[x], nodes[x]["r"] or 0))

    # Captions (title + mean rating) for *every* published image, dense on the
    # image index (slot i == img_{i+1:06d}), not just the genome-having nodes
    # above. The gallery captions images it can't breed from, so gating these on
    # a genome silently blanked the titles of runs whose genome dump is partial.
    n_img = max(img_index(i) for i in all_ids)
    titles: list = [None] * n_img
    ratings: list = [None] * n_img
    for e in entries:
        k = img_index(e["id"]) - 1
        titles[k] = e.get("title") or None
        ratings[k] = mean_rating(e)[0]

    return {
        "run": run,
        "model": model,
        "traits": traits,
        "n": len(nodes),
        "depth": max(depth.values()) if depth else 0,
        "roots": roots,
        "featured": featured,
        "default": default,
        "nodes": nodes,
        "ti": titles,
        "ra": ratings,
    }


def build_human() -> dict | None:
    """Lineage sidecar for the human Picbreeder archive (run id ``human``).

    Unlike the VLM runs this isn't on HF as a sweep; its entries come from
    build_human_genomes.py, which already records each image's in-archive
    picbreeder parent in ``source_entry_ids`` and converts a genome for every
    kept entry — so all entries have a renderable genome (gids = every id)."""
    if not HUMAN_META.exists():
        print(f"  [skip] no human metadata at {HUMAN_META} "
              "(run tools/build_human_genomes.py first)")
        return None
    entries = json.load(open(HUMAN_META)).get("entries", [])
    gids = {e["id"] for e in entries}
    return _assemble("human", "Picbreeder (human)", 0, entries, gids, {}, human=True)


def main():
    idx = json.load(open(hf_hub_download(HF_REPO, "index.json", repo_type="dataset")))
    runs = [r for r in idx["runs"] if (r.get("has") or {}).get("genome_json")]
    print(f"[tree] {len(runs)} runs with genomes on HF")

    TREE_OUT.mkdir(parents=True, exist_ok=True)
    manifest_runs = []
    canon_data = None
    for r in runs:
        run, cfg = r["run"], (r.get("config") or {})
        data = build_run(run, cfg)
        if data is None:
            continue
        with gzip.open(TREE_OUT / f"{run}.json.gz", "wt") as fh:
            json.dump(data, fh, separators=(",", ":"))
        group, label = run_label(run, cfg)
        manifest_runs.append({
            "run": run, "label": label, "group": group, "model": data["model"],
            "seed": cfg.get("seed"), "traits": data["traits"],
            "n": data["n"], "depth": data["depth"],
        })
        if run == CANON:
            canon_data = data
        print(f"  {data['n']:>5} imgs  depth {data['depth']:>2}  {run}")

    # human Picbreeder archive — add its manifest entry so the detail-page lineage
    # panel and the Family Tree (run=human) resolve. Its shard is NOT bundled in
    # the blog like the VLM runs; it lives on HF next to the human genomes
    # (results/human/tree.json.gz) and is uploaded by tools/push_human_genomes.py.
    human_data = build_human()
    if human_data is not None:
        manifest_runs.append({
            "run": "human", "label": "Human Picbreeder archive",
            "group": "Picbreeder (human)", "model": human_data["model"],
            "seed": None, "traits": 0,
            "n": human_data["n"], "depth": human_data["depth"],
        })
        print(f"  {human_data['n']:>5} imgs  depth {human_data['depth']:>2}  human (shard -> HF)")

    # sort: canon-model first, then by model, then deepest/biggest trees first
    manifest_runs.sort(key=lambda m: (m["group"] != "gemini-2.5-pro", m["group"], -m["n"]))
    json.dump({"canon": CANON, "n_runs": len(manifest_runs), "runs": manifest_runs},
              open(TREE_OUT / "manifest.json", "w"), separators=(",", ":"))
    print(f"[tree] manifest: {len(manifest_runs)} runs -> {TREE_OUT/'manifest.json'}")

    # canonical run: bundle its sidecar (back-comp for play.html?tree=) + genome shard
    if canon_data is None:
        raise SystemExit(f"canonical run {CANON} not built — check it exists locally")
    with gzip.open(OUT / "tree.json.gz", "wt") as fh:
        json.dump(canon_data, fh, separators=(",", ":"))
    src_gz = meta_path_for(CANON).parent / "genomes.json.gz"
    shutil.copyfile(src_gz, OUT / "tree_genomes.json.gz")
    print(f"[tree] bundled canon sidecar + genomes ({(OUT/'tree_genomes.json.gz').stat().st_size/1e6:.1f} MB)")

    # `human` is in the manifest but its shard ships on HF, not here — skip absentees.
    shards = [p for p in ((TREE_OUT / f"{m['run']}.json.gz") for m in manifest_runs) if p.exists()]
    total = sum(p.stat().st_size for p in shards)
    print(f"[tree] {len(shards)} sidecars, {total/1e6:.1f} MB total gz")


if __name__ == "__main__":
    main()
