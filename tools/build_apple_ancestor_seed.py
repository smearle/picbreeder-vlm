#!/usr/bin/env python3
"""
Emit the genome the inline blog breeder seeds generation 0 from: an ANCESTOR of the
famous Picbreeder apple, taken from partway up the apple's own evolutionary lineage.

The apple shown in the two CPPN explainer figures (h_apple) is the final, published
genome of Picbreeder series 5736 (generation 320 — 83 nodes / 264 conns, matching
fer/data/picbreeder_apple/pbcppn.pkl). Picbreeder records that series as a lineage
root (its main.zip branchFrom is empty), so there is no separate "publication before
it" to seed from. We instead reach back into the apple's own history: around
generation 140 the lineage is a clean GREEN apple with a stem and leaf — recognisably
the apple's ancestor, yet clearly distinct from the final red fruit. The breeder then
re-discovers the red apple (and everything else) by selective breeding from there.

Output schema is exactly what breed/cppn.js's genomeFromJSON() consumes
({nodes, connections}), plus a "_meta" block (ignored by the loader, kept for
provenance). Writes into the live blog at ASSETS_DIR (see [[archive-viewer-asset-paths]]).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from picbreeder_vlm._paths import FER_ROOT, ensure_fer_importable
ensure_fer_importable()
from fer.src.picbreeder_util import load_zip_xml_as_dict
from fer.src.lineage_utils import _ensure_list

PB = FER_ROOT / "spaghetti/pbRender/genomeAll"
ASSETS_DIR = Path.home() / "smearle.github.io/picbreeder-vlm-06b0d76d"
OUT = ASSETS_DIR / "breed/data/apple_ancestor.json"

PID = "5736"           # the apple series (apple = its final generation, 320)
GEN = 140              # green-apple ancestor: distinct from the final red apple
FINAL_GEN = 320

_PB_IN = {"x": -1, "y": -2, "d": -3, "bias": -4}
_PB_OUT = {"hue": 0, "saturation": 1, "brightness": 2}


def _si(v):
    try:
        return int(v)
    except Exception:
        return -1


def _listify(x):
    return x if isinstance(x, list) else ([] if x is None else [x])


def generation_genome(pid: str, num: int):
    """The raw XML genome dict for one generation of a Picbreeder series."""
    md = load_zip_xml_as_dict(str(PB / pid / "main.zip")).get("genome", {})
    gens = _ensure_list(md.get("series", {}).get("generation"))
    ent = next((g for g in gens if _si(g.get("@number")) == num), None)
    if not ent:
        return None
    sd = load_zip_xml_as_dict(str(PB / pid / f"{ent.get('@storage')}.zip"))
    sd = sd.get("genome", {}).get("storage", {})
    tg = next((g for g in _ensure_list(sd.get("generation")) if _si(g.get("@number")) == num), None)
    if not tg:
        return None
    if "population" in tg:
        pop = tg["population"]
        return pop[-1] if isinstance(pop, list) else (pop if "nodes" in pop else None)
    if "genome" in tg:
        gc = tg["genome"]
        return gc[-1] if isinstance(gc, list) else gc
    return tg if "nodes" in tg else None


def convert(g) -> dict:
    """XML Picbreeder chromosome -> cppn.js JSON (mirrors tools/build_cppn_explainer.convert_dump)."""
    nodes = _listify(g.get("nodes", {}).get("node"))
    links = _listify(g.get("links", {}).get("link"))

    def mk(m):
        return f"{m.get('@branch')}:{m.get('@id')}"

    id_to_key, nxt, out_nodes = {}, [3], []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        k = _PB_IN.get(n.get("@label", ""), _PB_OUT.get(n.get("@label", "")))
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


def main():
    g = generation_genome(PID, GEN)
    if g is None:
        raise SystemExit(f"no generation {GEN} for series {PID}")
    gj = convert(g)
    gj["_meta"] = {"title": "Apple (ancestor)", "source": "human",
                   "pid": PID, "generation": GEN, "final_generation": FINAL_GEN}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gj))
    print(f"[apple-ancestor] series {PID} gen {GEN}: "
          f"{len(gj['nodes'])} nodes, {len(gj['connections'])} conns -> {OUT}")


if __name__ == "__main__":
    main()
