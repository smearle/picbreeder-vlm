#!/usr/bin/env python3
"""Build the data bundle for the blog's CPPN explainer figures.

Two sources of genomes, both emitted as the `cppn.js`-renderable JSON the blog
already uses (nodes: key/activation/affinity/input; connections: from/to/weight/
enabled):

  * HUMAN  — the famous human-evolved Picbreeder genomes (skull, butterfly,
             apple) that Kumar et al. (2025) shipped as fer/data/picbreeder_*/
             pbcppn.pkl. We remap their string node-ids onto the canonical
             input keys (-1=x,-2=y,-3=d,-4=bias) and output keys (0=hue,
             1=saturation,2=brightness), exactly the convention cppn.js renders.
  * AI     — a handful of our VLM-evolved genomes, lifted verbatim from the
             already-published breed/data/archive.json.

To sanity-check the human conversion (whose original engine differs slightly
from ours) we re-implement cppn.js's renderGenome faithfully in numpy and write
a contact sheet next to each human genome's reference img.png, plus the AI
candidates, to /tmp/cppn_contact.png. The curated, color-correct bundle lands at
breed/data/cppn_explainer.json.
"""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
from picbreeder_vlm._paths import FER_ROOT, ensure_fer_importable  # noqa: E402
BLOG = Path.home() / "smearle.github.io" / "picbreeder-vlm-06b0d76d"
ARCHIVE = BLOG / "breed" / "data" / "archive.json"
OUT = BLOG / "breed" / "data" / "cppn_explainer.json"
CONTACT = Path("/tmp/cppn_contact.png")

# ----------------------------------------------------------------- cppn.js port
SQRT2 = math.sqrt(2.0)


def _clamp(z, lo, hi):
    return np.clip(z, lo, hi)


def act(name, z):
    if name == "identity" or name is None:
        return z
    if name == "sigmoid":
        z = _clamp(z, -60, 60)
        return 2.0 / (1.0 + np.exp(-z)) - 1.0
    if name == "gaussian":
        z = _clamp(z, -60, 60)
        return 2.0 * np.exp(-z * z) - 1.0
    if name == "cos":
        return np.cos(_clamp(z, -60, 60))
    if name == "sin":
        return np.sin(_clamp(z, -60, 60))
    return z


def _layers(node_keys, conns):
    """Feed-forward evaluation order over non-input nodes (matches cppn.js)."""
    inputs = {-1, -2, -3, -4}
    known = set(inputs)
    incoming = {}
    for c in conns:
        if not c.get("enabled", True):
            continue
        incoming.setdefault(c["to"], []).append((c["from"], c["weight"]))
    targets = [k for k in node_keys if k not in inputs]
    order = []
    remaining = set(targets)
    # ensure outputs present even with no incoming
    for k in (0, 1, 2):
        incoming.setdefault(k, [])
        remaining.add(k)
    guard = 0
    while remaining and guard < 10000:
        guard += 1
        progressed = False
        for k in list(remaining):
            if all(src in known for (src, _w) in incoming.get(k, [])):
                order.append((k, incoming.get(k, [])))
                known.add(k)
                remaining.discard(k)
                progressed = True
        if not progressed:
            # cycle / unreachable: emit the rest in arbitrary order
            for k in list(remaining):
                order.append((k, incoming.get(k, [])))
                known.add(k)
            break
    return order


def render(gj, res=110, color=True):
    node_keys = [n["key"] for n in gj["nodes"]]
    actmap = {n["key"]: n.get("activation") for n in gj["nodes"]}
    order = _layers(node_keys, gj["connections"])

    xs = ((np.arange(res) * 2) - (res - 1)) / res
    sx, sy = np.meshgrid(xs, xs)            # sy varies by row, sx by col
    d = np.hypot(sx, sy) * SQRT2
    v = {-1: sx, -2: sy, -3: d, -4: np.ones_like(sx), 0: np.zeros_like(sx),
         1: np.zeros_like(sx), 2: np.zeros_like(sx)}
    for k in node_keys:                      # pre-seed so dangling refs read as 0
        v.setdefault(k, np.zeros_like(sx))
    for k, links in order:
        s = np.zeros_like(sx)
        for (src, w) in links:
            s = s + v.get(src, 0.0) * w
        v[k] = act(actmap.get(k), s)
    o0, o1, o2 = v[0], v[1], v[2]
    if color:
        h = np.mod(o0 + 1.0, 1.0)
        s = np.clip(o1, 0, 1)
        val = np.clip(np.abs(o2), 0, 1)
        img = _hsv2rgb(h, s, val)
    else:
        g = np.clip(np.abs(o2), 0, 1)
        img = np.stack([g, g, g], axis=-1)
    return (img * 255 + 0.5).astype(np.uint8)


def _hsv2rgb(h, s, v):
    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    out = np.stack([r, g, b], axis=-1)
    # s==0 -> gray
    gray = np.stack([v, v, v], axis=-1)
    return np.where((s == 0)[..., None], gray, out)


# ------------------------------------------------- human pbcppn -> cppn.js JSON
def convert_human(pkl_path: Path):
    d = pickle.load(open(pkl_path, "rb"))
    nodes = d["nodes"]
    links = d["links"]
    special = d["special_nodes"]   # {role: node_id}; roles vary in spelling

    # role -> canonical cppn.js key
    def role_key(role):
        r = str(role).lower()
        if r in ("x",):
            return -1
        if r in ("y",):
            return -2
        if r in ("d", "dist", "distance"):
            return -3
        if r in ("bias", "b"):
            return -4
        if r in ("hue", "h"):
            return 0
        if r in ("saturation", "sat", "s"):
            return 1
        if r in ("brightness", "bright", "v", "value", "ink"):
            return 2
        return None

    id_to_key = {}
    for role, nid in special.items():
        k = role_key(role)
        if k is not None:
            id_to_key[nid] = k

    nxt = 3
    actmap = {}
    for n in nodes:
        actmap[n["id"]] = n.get("activation")
    for n in nodes:
        if n["id"] not in id_to_key:
            id_to_key[n["id"]] = nxt
            nxt += 1

    out_nodes = []
    for n in nodes:
        k = id_to_key[n["id"]]
        is_in = k < 0
        out_nodes.append({
            "key": k,
            "activation": None if is_in else (n.get("activation") or "identity"),
            "affinity": "grey",          # affinity only matters for breeding, not render
            "input": is_in,
        })
    # make sure all four inputs + three outputs exist
    present = {n["key"] for n in out_nodes}
    for k in (-1, -2, -3, -4):
        if k not in present:
            out_nodes.append({"key": k, "activation": None, "affinity": "grey", "input": True})
    for k in (0, 1, 2):
        if k not in present:
            out_nodes.append({"key": k, "activation": "identity", "affinity": "grey", "input": False})

    out_conns = []
    for l in links:
        src = id_to_key.get(l["source"])
        dst = id_to_key.get(l["target"])
        if src is None or dst is None:
            continue
        out_conns.append({"from": src, "to": dst, "weight": float(l["weight"]), "enabled": True})

    return {"nodes": out_nodes, "connections": out_conns}


# ------------------------------------------ original Picbreeder XML dump -> JSON
# The famous skull/butterfly/apple shipped as pkls, but the rest of the original
# Picbreeder archive lives as XML chromosomes in fer/spaghetti/pbRender/genomeAll
# (anonymous — no titles). We convert published colour genomes the same way.
PB_DIR = FER_ROOT / "spaghetti/pbRender/genomeAll"
_PB_IN = {"x": -1, "y": -2, "d": -3, "bias": -4}
_PB_OUT = {"hue": 0, "saturation": 1, "brightness": 2}


def _pb_listify(x):
    return x if isinstance(x, list) else ([] if x is None else [x])


def _pb_final_genome(pid: str):
    """The published genome for a Picbreeder series id (its last generation)."""
    ensure_fer_importable()
    from fer.src.picbreeder_util import load_zip_xml_as_dict
    from fer.src.lineage_utils import _ensure_list

    def si(v):
        try:
            return int(v)
        except Exception:
            return -1

    md = load_zip_xml_as_dict(str(PB_DIR / pid / "main.zip")).get("genome", {})
    series = md.get("series", {}) or ({} if "generation" not in md else md)
    gens = _ensure_list(series.get("generation"))
    if not gens:
        return None
    last = max(gens, key=lambda g: si(g.get("@number", -1)))
    sid, num = last.get("@storage"), si(last.get("@number"))
    sz = PB_DIR / pid / f"{sid}.zip"
    if not sid or not sz.exists():
        return None
    sd = load_zip_xml_as_dict(str(sz)).get("genome", {}).get("storage", {})
    sg = _ensure_list(sd.get("generation"))
    tg = next((g for g in sg if si(g.get("@number")) == num), sg[-1] if sg else None)
    if not tg:
        return None
    if "population" in tg:
        pop = tg["population"]
        return pop[-1] if isinstance(pop, list) else (pop if "nodes" in pop else None)
    if "genome" in tg:
        gc = tg["genome"]
        return gc[-1] if isinstance(gc, list) else gc
    return tg if "nodes" in tg else None


def convert_dump(pid: str):
    """XML Picbreeder chromosome -> cppn.js JSON (same schema as convert_human)."""
    g = _pb_final_genome(pid)
    if not g:
        raise RuntimeError(f"no published genome for pid {pid}")
    nodes = _pb_listify(g.get("nodes", {}).get("node"))
    links = _pb_listify(g.get("links", {}).get("link"))

    def mk(m):
        return f"{m.get('@branch')}:{m.get('@id')}"

    id_to_key, nxt, out_nodes = {}, [3], []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        lab = n.get("@label", "")
        k = _PB_IN.get(lab, _PB_OUT.get(lab))
        if k is None:
            k = nxt[0]; nxt[0] += 1
        id_to_key[mk(n["marking"])] = k
        is_in = k < 0
        a = n.get("activation", {})
        act = (a.get("#text", "identity") if isinstance(a, dict) else "identity").replace("(x)", "")
        out_nodes.append({"key": k, "activation": None if is_in else act,
                          "affinity": "grey", "input": is_in})
    present = {n["key"] for n in out_nodes}
    for k in (-1, -2, -3, -4):
        if k not in present:
            out_nodes.append({"key": k, "activation": None, "affinity": "grey", "input": True})
    for k in (0, 1, 2):
        if k not in present:
            out_nodes.append({"key": k, "activation": "identity", "affinity": "grey", "input": False})
    out_conns = []
    for l in links:
        if not isinstance(l, dict):
            continue
        s, t = id_to_key.get(mk(l["source"])), id_to_key.get(mk(l["target"]))
        if s is None or t is None:
            continue
        w = l.get("weight", {})
        out_conns.append({"from": s, "to": t,
                          "weight": float(w.get("#text", 0) if isinstance(w, dict) else w),
                          "enabled": True})
    return {"nodes": out_nodes, "connections": out_conns}


# ----------------------------------------------------------------- assemble
genomes = {}

HUMAN = [
    ("skull", "Skull", FER_ROOT / "data/picbreeder_skull"),
    ("butterfly", "Butterfly", FER_ROOT / "data/picbreeder_butterfly"),
    ("apple", "Apple", FER_ROOT / "data/picbreeder_apple"),
]
human_refs = {}
for hid, label, ddir in HUMAN:
    pkl = ddir / "pbcppn.pkl"
    if not pkl.exists():
        print("[human] MISSING", pkl)
        continue
    gj = convert_human(pkl)
    genomes["h_" + hid] = {"label": label, "source": "human", "title": label,
                            "color": True, "g": gj}
    ref = ddir / "img.png"
    if ref.exists():
        human_refs["h_" + hid] = ref
    print(f"[human] {hid}: {len(gj['nodes'])} nodes, {len(gj['connections'])} conns")

# AI candidates from the published archive (already cppn.js JSON)
arch = json.load(open(ARCHIVE))
items = arch["items"]
# a spread of recognizable titles; fall back to highest_rated order
WANT = ["c_img_000228", "c_img_000316", "c_img_000063", "c_img_000143",
        "c_img_000176", "c_img_000604", "c_img_000591", "c_img_000603"]
ai_keys = [k for k in WANT if k in items]
for k in (arch.get("lists", {}).get("highest_rated") or []):
    if k not in ai_keys and items.get(k, {}).get("g"):
        ai_keys.append(k)
    if len(ai_keys) >= 16:
        break
for k in ai_keys:
    it = items[k]
    if not it.get("g"):
        continue
    genomes["a_" + k] = {"label": it.get("title") or k, "source": "ai",
                          "title": it.get("title") or k,
                          "color": bool(it.get("color", True)), "g": it["g"]}

print(f"[bundle] {len(genomes)} genomes ({sum(1 for v in genomes.values() if v['source']=='human')} human, "
      f"{sum(1 for v in genomes.values() if v['source']=='ai')} ai)")

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump({"genomes": genomes}, open(OUT, "w"), separators=(",", ":"))
print("[out]", OUT, f"{OUT.stat().st_size/1024:.0f} KB")

# ----------------------------------------------------------------- contact sheet
CELL = 110
PAD = 6
LBLH = 16
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
except Exception:
    font = ImageFont.load_default()

# rows: human (ours render | reference) ; then AI grid
rows = []
for gid, ref in human_refs.items():
    g = genomes[gid]
    mine = Image.fromarray(render(g["g"], CELL, g["color"]))
    refimg = Image.open(ref).convert("RGB").resize((CELL, CELL))
    rows.append([(g["label"] + " (ours)", mine), (g["label"] + " (ref)", refimg)])

ai = [(gid, genomes[gid]) for gid in genomes if genomes[gid]["source"] == "ai"]
for i in range(0, len(ai), 4):
    row = []
    for gid, g in ai[i:i + 4]:
        im = Image.fromarray(render(g["g"], CELL, g["color"]))
        row.append((g["label"][:16], im))
    rows.append(row)

cols = max(len(r) for r in rows)
W = cols * (CELL + PAD) + PAD
H = len(rows) * (CELL + LBLH + PAD) + PAD
sheet = Image.new("RGB", (W, H), (255, 255, 255))
draw = ImageDraw.Draw(sheet)
for ri, row in enumerate(rows):
    y = PAD + ri * (CELL + LBLH + PAD)
    for ci, (lbl, im) in enumerate(row):
        x = PAD + ci * (CELL + PAD)
        sheet.paste(im, (x, y))
        draw.text((x, y + CELL + 2), lbl, fill=(0, 0, 0), font=font)
sheet.save(CONTACT)
print("[contact]", CONTACT)
