"""Build the sfdp-morph viewer's data for one run:

  * tree_ancestors.json.gz  — each founder (root publication)'s REAL gen-0 random-init
                              CPPN genome (selected_gen_000 from the founding agent's
                              zip), serialized to cppn.js JSON. One ancestor node per
                              founding agent; an agent that published several roots
                              shares one ancestor (same session, same gen-0).
  * tree_sfdp.json.gz       — Graphviz sfdp coordinates for the AUGMENTED graph
                              (publications + ancestor nodes + ancestor->root edges),
                              so founders are no longer isolated dots and every founder
                              has a morph back to its random initial CPPN.

Reads the canonical-run source data under sweep_logs/sweep/<run>/, writes into the
live blog's breed/data/ dir. Re-runnable. See [[scrubbable-morph-viewer]].
"""
from __future__ import annotations
import gzip, json, os, pickle, subprocess, zipfile, math, statistics, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from genome_json import genome_to_json
BLOG_DATA = os.path.expanduser(
    "~/smearle.github.io/picbreeder-vlm-06b0d76d/breed/data")
RUN = "th-1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s3"
AGENTS = os.path.join(REPO, "sweep_logs", "sweep", RUN, "agents")


def load_gz_json(path):
    with gzip.open(path) as f:
        return json.load(f)


def gen0_genome_json(agent_id):
    """Real gen-0 (random init) genome for a founding agent, as cppn.js JSON."""
    zpath = os.path.join(AGENTS, f"agent_{int(agent_id):03d}.zip")
    if not os.path.exists(zpath):
        return None
    with zipfile.ZipFile(zpath) as zf:
        hits = [n for n in zf.namelist() if "selected_gen_000" in n]
        if not hits:
            return None
        rec = pickle.loads(gzip.decompress(zf.read(hits[0])))
    sel = rec.get("selected") or []
    if not sel:
        return None
    return genome_to_json(sel[0]["genome"])


def main():
    tree = load_gz_json(os.path.join(BLOG_DATA, "tree.json.gz"))
    nodes = tree["nodes"]
    roots = [i for i, n in nodes.items() if n.get("p") is None]

    # one ancestor per founding AGENT (dedupe roots bred in the same session)
    ancestors = {}                      # ancId -> {genome, agent, children:[rootId]}
    root_to_anc = {}
    missing = []
    for rid in roots:
        agent = str(nodes[rid].get("a"))
        anc_id = f"anc_a{agent}"
        if anc_id not in ancestors:
            gj = gen0_genome_json(agent)
            if gj is None:
                missing.append((rid, agent))
                continue
            ancestors[anc_id] = {"genome": gj, "agent": agent, "children": []}
        ancestors[anc_id]["children"].append(rid)
        root_to_anc[rid] = anc_id

    print(f"roots: {len(roots)}  ancestors (unique agents): {len(ancestors)}  "
          f"missing gen-0: {len(missing)}")
    if missing:
        print("  missing:", missing[:10])

    with gzip.open(os.path.join(BLOG_DATA, "tree_ancestors.json.gz"), "wt") as f:
        json.dump({"run": RUN, "ancestors": ancestors, "root_to_anc": root_to_anc}, f)

    # ---- augmented sfdp layout: publications + ancestors + ancestor->root edges ----
    ids = list(nodes) + list(ancestors)
    idx = {i: n for n, i in enumerate(ids)}
    edges = []
    for i, n in nodes.items():
        p = n.get("p")
        if p is not None and p in idx:
            edges.append((idx[p], idx[i]))
    for rid, anc in root_to_anc.items():
        edges.append((idx[anc], idx[rid]))

    L = ["graph G {", "  node [shape=point];"]
    for i in ids:
        L.append(f"n{idx[i]};")
    for a, b in edges:
        L.append(f"n{a} -- n{b};")
    L.append("}")
    dot = "\n".join(L)
    # neato stress (SGD) gives far more uniform on-screen edge lengths than sfdp
    # (edge-length CV ~0.31 vs ~0.91; ~7s offline for ~2.7k nodes). Pass --engine sfdp
    # for the looser, faster organic layout instead.
    engine = "sfdp" if "--engine" in sys.argv and sys.argv[sys.argv.index("--engine") + 1] == "sfdp" else "neato"
    cmd = ["neato", "-Gmode=sgd", "-Tplain"] if engine == "neato" else ["sfdp", "-Tplain"]
    out = subprocess.run(cmd, input=dot, capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        print(f"{engine} failed:", out.stderr[:300]); sys.exit(1)
    print(f"layout engine: {engine}")

    pos = {}
    for ln in out.stdout.splitlines():
        f = ln.split()
        if f and f[0] == "node":
            pos[ids[int(f[1][1:])]] = [round(float(f[2]), 3), round(float(f[3]), 3)]
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    bbox = {"minx": min(xs), "maxx": max(xs), "miny": min(ys), "maxy": max(ys)}

    # report edge-length uniformity
    Ls = [math.hypot(pos[a][0]-pos[b][0], pos[a][1]-pos[b][1])
          for a, b in [(ids[x], ids[y]) for x, y in edges] if a in pos and b in pos]
    Ls = [x for x in Ls if x > 0]
    cv = statistics.pstdev(Ls) / statistics.mean(Ls)
    print(f"{engine}: {len(pos)} nodes, {len(edges)} edges, edge-length CV {cv:.2f}")

    with gzip.open(os.path.join(BLOG_DATA, "tree_sfdp.json.gz"), "wt") as f:
        json.dump({"run": RUN, "pos": pos, "bbox": bbox,
                   "ancestors": list(ancestors)}, f)
    print("wrote tree_ancestors.json.gz + tree_sfdp.json.gz")


if __name__ == "__main__":
    main()
