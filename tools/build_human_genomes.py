#!/usr/bin/env python
"""Convert the original Picbreeder (human) CPPN genomes into the breed site's
JSON genome format so breed/browse.html?run=human renders + branches them
client-side, exactly like the AI runs.

Each human series' published genome lives in fer/spaghetti/pbRender/genomeAll/<pid>/rep.zip
as an rtNEAT XML chromosome. fer.src.process_pb.load_pbcppn already parses that
into a NEAT graph (nodes with activation, weighted links, and the special
input/output node ids). We re-key that graph onto the breed CPPN substrate:

  inputs   x,y,d,bias            -> keys -1,-2,-3,-4   (cppn.js sets these directly)
  outputs  hue,saturation,bright -> keys  0, 1, 2      (HSV, same as restrict_hsb_channels)
  hidden                          -> fresh positive ints

The activation set used by the human genomes is exactly {sin,cos,sigmoid,gaussian,
identity} — the five cppn.js implements, with matching bipolar sigmoid/gaussian —
so we bake each node's activation straight in and emit NO outAct (cppn.js applies
the node activation once, then h=(o0+1)%1, s=clip(o1), v=clip|o2|, hsv2rgb — the
same pipeline as fer.do_forward_pass). Grayscale ("ink") genomes get color_enabled
False so they render via |brightness|.

Output (mirrors what loadRun() in pbsite.js expects for an HF run):
  <out>/site/human/genomes.json.gz                {img_id: genome_json}
  <out>/results/human/archive_metadata.json       {entries:[{id, source_entry_ids,
                                                    title, agent_id, color_enabled,
                                                    added_at, generation}], n, source}

Order + ids + lineage come from tools/build_human_sprites (so genomes line up 1:1
with the already-published sprite sheets / layout.json).

  python tools/build_human_genomes.py                 # full build (all ~9375)
  python tools/build_human_genomes.py --n 200         # first 200 (quick test)
"""
import argparse
import gzip
import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path("/home/jupyter-smearle/picbreeder-vlm")
from picbreeder_vlm._paths import FER_ROOT, ensure_fer_importable  # noqa: E402
PB = FER_ROOT / "spaghetti/pbRender/genomeAll"           # <pid>/rep.zip published genome
CAPTIONS = REPO / "data/human_baseline/res-128/captions_human.json"

# special-node label -> breed key
SPECIAL_KEY = {"x": -1, "y": -2, "d": -3, "bias": -4, "h": 0, "s": 1, "v": 2}
INPUT_KEYS = {-1, -2, -3, -4}
COLOR_KEYS = {0, 1}                                       # hue, saturation


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reconcile the two renderers' coordinate fields by absorbing the difference into
# the (linear) input-link weights, so cppn.js reproduces the published fer thumbs:
#   fer  (thumbnails):  x,y = linspace(-1,1,R);          d = hypot(x,y)*1.4
#   cppn.js:            x,y = (2*xx-(R-1))/R  (->*R/(R-1) short of fer's grid);  d = hypot*sqrt2
# At the R=128 thumbnail grid, fer's x is cppn's x * 128/127. So scale x,y links by
# 128/127, and d links by that *and* the 1.4/sqrt2 radius-multiplier ratio. (Unlike
# the AI genomes — rendered on cppn.js's own grid — the human thumbs use fer's grid,
# so this alignment is what makes the in-browser render match the sprite sheet.)
XY_FIX = 128.0 / 127.0
D_FIX = XY_FIX * 1.4 / 2.0 ** 0.5


def _ensure_list(v):
    return v if isinstance(v, list) else [v]


def _safe_int(v, default=-1):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def load_final_genome(pid):
    """Return the final-generation genome dict for a series, replicating
    fer.src.plot_lineage_phylogeny._load_lineage_info: main.zip names the last
    generation + its storage shard, then we pull that generation's chromosome out
    of <storage>.zip. This is the genome the published <pid>_final.png renders —
    rep.zip is only a (sometimes earlier) representative, so we must NOT use it."""
    ensure_fer_importable()
    from fer.src.picbreeder_util import load_zip_xml_as_dict
    main_zip = PB / pid / "main.zip"
    if not main_zip.exists():
        return None
    series = load_zip_xml_as_dict(str(main_zip)).get("genome", {})
    if "series" in series:
        series = series["series"]
    gens = series.get("generation")
    if not gens:
        return None
    gens = _ensure_list(gens)
    last = max(gens, key=lambda g: _safe_int(g.get("@number")))
    storage_id, last_num = last.get("@storage"), _safe_int(last.get("@number"))
    if storage_id is None:
        return None
    storage_zip = PB / pid / f"{storage_id}.zip"
    if not storage_zip.exists():
        return None
    gdata = load_zip_xml_as_dict(str(storage_zip)).get("genome", {})
    sgens = gdata["generation"] if "storage" not in gdata else gdata["storage"]["generation"]
    sgens = _ensure_list(sgens)
    target = next((g for g in sgens if _safe_int(g.get("@number")) == last_num), None) \
        or (sgens[-1] if sgens else None)
    if target is None:
        return None
    if "population" in target:
        pop = target["population"]
        return pop[-1] if isinstance(pop, list) else (pop if "nodes" in pop else None)
    if "genome" in target:
        g = target["genome"]
        return g[-1] if isinstance(g, list) else g
    return target if "nodes" in target else None


def parse_pbcppn(genome):
    """Picbreeder genome dict (final_genome) -> NEAT graph, mirroring
    fer.src.save_lineage_figures.load_pbcppn. Grayscale 'ink' genomes get a
    synthesised zero-weight HSV head so every graph exposes h/s/v outputs."""
    nodes, links = [], []
    for node in _ensure_list(genome["nodes"]["node"]):
        nid = f"{node['marking']['@branch']}_{node['marking']['@id']}"
        nodes.append(dict(label=node.get("@label", ""), id=nid,
                          activation=node["activation"]["#text"][:-3]))   # strip "(x)"
    for link in _ensure_list(genome["links"]["link"]):
        src = f"{link['source']['@branch']}_{link['source']['@id']}"
        tgt = f"{link['target']['@branch']}_{link['target']['@id']}"
        links.append(dict(source=src, target=tgt, weight=float(link["weight"]["#text"])))

    labels = [n["label"] for n in nodes]
    color = "ink" not in labels
    if not color:                                      # grayscale -> synth HSV head
        v = nodes[labels.index("ink")]
        v["label"] = "brightness"
        nodes.append(dict(label="hue", id="syn_hue", activation="identity"))
        nodes.append(dict(label="saturation", id="syn_sat", activation="identity"))
        links.append(dict(source=v["id"], target="syn_hue", weight=0.0))
        links.append(dict(source=v["id"], target="syn_sat", weight=0.0))

    label2id = {}
    for n in nodes:
        if n["label"]:
            label2id.setdefault(n["label"], n["id"])
    special = {k: label2id[lbl] for k, lbl in
               (("x", "x"), ("y", "y"), ("d", "d"), ("bias", "bias"),
                ("h", "hue"), ("s", "saturation"), ("v", "brightness"))}
    return dict(nodes=nodes, links=links, special_nodes=special), color


def convert_one(pid):
    """Picbreeder series pid -> (genome_json, color_enabled), or (None, None) if
    the final genome can't be located."""
    genome = load_final_genome(pid)
    if genome is None or "nodes" not in genome:
        return None, None
    nn, color = parse_pbcppn(genome)
    special = nn["special_nodes"]
    scale_by_src = {special["x"]: XY_FIX, special["y"]: XY_FIX, special["d"]: D_FIX}
    # map every node id -> breed key (specials fixed, hidden -> fresh positive ints)
    id2key = {}
    for label, key in SPECIAL_KEY.items():
        id2key[special[label]] = key
    nxt = 3
    for n in nn["nodes"]:
        if n["id"] not in id2key:
            id2key[n["id"]] = nxt
            nxt += 1

    nodes_json = []
    for n in nn["nodes"]:
        key = id2key[n["id"]]
        is_in = key in INPUT_KEYS
        nodes_json.append({
            "key": key,
            "activation": "identity" if is_in else n["activation"],
            "affinity": "color" if key in COLOR_KEYS else "grey",
            "input": is_in,
        })

    conns_json = []
    seen = set()
    for l in nn["links"]:
        i, o = id2key[l["source"]], id2key[l["target"]]
        if (i, o) in seen:          # collapse any duplicate edge (cppn.js keys by "i,o")
            continue
        seen.add((i, o))
        w = float(l["weight"]) * scale_by_src.get(l["source"], 1.0)   # input-coord alignment
        conns_json.append({"from": i, "to": o, "weight": w, "enabled": True})

    return {"nodes": nodes_json, "connections": conns_json}, color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "archive_animations/out/human_genomes",
                    help="dir to write site/human/ and results/human/ under")
    ap.add_argument("--n", type=int, default=None, help="cap to first N publications")
    args = ap.parse_args()

    bhs = _load("build_human_sprites", REPO / "tools" / "build_human_sprites.py")

    ids = bhs.human_ids()
    if args.n:
        ids = ids[: args.n]
    entries = bhs.human_lineage_entries(ids)            # id (img_xxxxxx) + source_entry_ids

    # lineage depth (entries are publication order; parent precedes child)
    depth = {}
    for e in entries:
        src = e["source_entry_ids"]
        depth[e["id"]] = depth.get(src[0], -1) + 1 if src else 0

    captions = {}
    if CAPTIONS.exists():
        raw = json.loads(CAPTIONS.read_text())
        captions = {k[:-4] if k.endswith(".png") else k: v for k, v in raw.items()}

    genomes = {}
    base = datetime(2007, 1, 1)
    n_color = n_grey = n_skip = 0
    for i, (pid, e) in enumerate(zip(ids, entries)):
        try:
            g, color = convert_one(pid)
        except Exception as ex:
            print(f"  [skip] {pid}: {ex}")
            n_skip += 1
            continue
        if g is None:
            print(f"  [skip] {pid}: no final genome")
            n_skip += 1
            continue
        genomes[e["id"]] = g
        e["added_at"] = (base + timedelta(seconds=i)).isoformat()
        e["generation"] = depth[e["id"]]
        e["agent_id"] = "human"
        e["color_enabled"] = bool(color)
        cap = captions.get(pid)
        if cap:
            e["title"] = cap
        n_color += int(color)
        n_grey += int(not color)
        if (i + 1) % 1000 == 0:
            print(f"  ...{i + 1}/{len(ids)}")

    # keep only entries whose genome converted (so metadata <-> genomes stay aligned)
    entries = [e for e in entries if e["id"] in genomes]

    site_dir = args.out / "site" / "human"
    res_dir = args.out / "results" / "human"
    site_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    with gzip.open(site_dir / "genomes.json.gz", "wt") as f:
        json.dump(genomes, f, separators=(",", ":"))
    meta = {"entries": entries, "n": len(entries), "source": "human-picbreeder"}
    (res_dir / "archive_metadata.json").write_text(json.dumps(meta, separators=(",", ":")))

    gz_mb = (site_dir / "genomes.json.gz").stat().st_size / 1e6
    print(f"human genomes -> {args.out}")
    print(f"  {len(genomes)} genomes ({n_color} colour, {n_grey} grey), {n_skip} skipped")
    print(f"  genomes.json.gz {gz_mb:.1f} MB; {len(entries)} metadata entries; "
          f"max generation {max(depth.values())}")


if __name__ == "__main__":
    main()
