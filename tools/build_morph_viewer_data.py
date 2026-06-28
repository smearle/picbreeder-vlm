"""Per-run data for the phylogeny morph viewer (breed/sfdp-morph-demo.html).

Enumerates the canonical run set from the blog's index.json (same set the archive
viewer offers) and, for each run with a local tree sidecar, writes under
breed/data/morph/<run>/:
  - layout.json.gz   {run, label, bbox, engine, nodes:{id:{x,y,p,t,col,is_anc,is_init,a}}}
                     neato-stress layout of the augmented graph: publications +
                     gen-0 ancestors + a neutral grey "init" super-root joined to
                     every ancestor (laid out WITH the graph so it sits centrally).
  - anc.json.gz      {anc_id: cppn.js-JSON}  ONLY the gen-0 ancestor genomes (tiny).
  - genomes.json.gz  {id: cppn.js-JSON}  published + ancestor genomes — written ONLY
                     for CURATED runs, so the bundled blog figure works offline.

The heavy published genomes for every OTHER run are fetched on demand by the viewer
from HF (/site/<run>/genomes.json.gz, with the ?archiveBase= proxy fallback) — the
genome keys are the tree node ids (img_NNNNNN), so they key-match the layout.

Plus breed/data/morph/manifest.json: {default, runs:[{run,label,group,n,n_anc,bundled}]}.
See [[scrubbable-morph-viewer]]. Gen-0 extraction unpickles genomes that import
`neat` (pip install neat-python); missing zips → ancestors omitted (graceful).
"""
from __future__ import annotations
import gzip, json, os, pickle, re, subprocess, zipfile, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from genome_json import genome_to_json

BLOG = os.path.expanduser("~/smearle.github.io/picbreeder-vlm-06b0d76d/breed")
SITE = os.path.dirname(BLOG)                       # …/picbreeder-vlm-06b0d76d
TREE_DIR = os.path.join(BLOG, "data", "tree")
OUT_DIR = os.path.join(BLOG, "data", "morph")
SWEEP = os.path.join(REPO, "sweep_logs", "sweep")
INDEX = os.path.join(SITE, "index.json")           # canonical run manifest (archive viewer)

DEFAULT_RUN = "th-1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s3"

# Runs whose full genomes.json.gz (publications + ancestors) we BUNDLE on gh-pages so
# the embedded blog figure renders morphs offline (no HF/proxy needed). Everything
# else fetches published genomes from HF on demand.
CURATED = {
    DEFAULT_RUN,
    "th0_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s3",
    "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s3",
    "th10_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s3",
    "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_randp0.25_rmode-all_nopersonalities_fixed-sesh_s3",
    "ag20_tb-1_scheme-toggle_randp2_rmode-all_nopersonalities_fixed-sesh_s3",
    "th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_traits1000_fixed-sesh_s3",
    "th0_ag20_model-qwen3-vl-30b-fp8_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s3",
    "th1_ag20_model-gemini-3-pro-preview_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s3",
}


def load_gz(path):
    with gzip.open(path) as f:
        return json.load(f)


def pretty_arc(arc):
    if arc.startswith("mem_"):    return "memory " + arc[4:]
    if arc.startswith("noise_"):  return "noise " + arc[6:]
    if arc.startswith("agents_"): return arc[7:] + " personalities"
    if arc == "random":           return "random selection"
    if arc == "default":          return "baseline"
    return arc


def label_of(e):
    if e.get("arc") == "human":
        return e.get("label") or "Human Picbreeder archive"
    c = e.get("config") or {}
    parts = []
    m = c.get("model")
    if m and m != "gemini-2.5-pro":
        parts.append(m)
    parts.append(pretty_arc(e.get("arc") or e.get("run")))
    if c.get("seed") is not None:
        parts.append(f"seed {c['seed']}")
    return " · ".join(parts)


def entry_from_runid(run):
    """Minimal index-style entry for a CURATED run absent from index.json."""
    m = re.search(r"model-([a-z0-9.\-]+?)_(?:tb|ag)", run)
    s = re.search(r"_s(\d+)$", run)
    return {"run": run, "arc": "default",
            "config": {"model": m.group(1) if m else "gemini-2.5-pro",
                       "seed": int(s.group(1)) if s else None}}


def group_of(e):
    arc = e.get("arc") or ""
    m = (e.get("config") or {}).get("model")
    if arc == "human":                 return "Human"
    if m and m != "gemini-2.5-pro":    return "Selector model"
    if arc.startswith("mem_"):         return "Memory length"
    if arc.startswith("noise_"):       return "Selection noise"
    if arc.startswith("agents_"):      return "Personalities"
    if arc == "random":                return "Random selection"
    if arc == "default":               return "Baseline"
    return "Other"


def gen0_json(run, agent_id):
    try:
        aid = int(agent_id)
    except (TypeError, ValueError):
        return None                       # human archive (no per-agent gen-0 zips)
    z = os.path.join(SWEEP, run, "agents", f"agent_{aid:03d}.zip")
    if not os.path.exists(z):
        return None
    try:
        with zipfile.ZipFile(z) as zf:
            hit = [n for n in zf.namelist() if "selected_gen_000" in n]
            if not hit:
                return None
            rec = pickle.loads(gzip.decompress(zf.read(hit[0])))
    except Exception:
        return None
    sel = rec.get("selected") or []
    return genome_to_json(sel[0]["genome"]) if sel else None


# neato's stress-majorization (-Gmode=sgd) gives the most uniform edge lengths but is
# O(n^2) and unusable past a few thousand nodes; sfdp's multilevel solver stays fast at
# the human archive's ~9k nodes. Pick by size.
SGD_MAX = 4000


def neato(ids, edges):
    idx = {i: n for n, i in enumerate(ids)}
    L = ["graph G {", "  node [shape=point];"]
    L += [f"n{idx[i]};" for i in ids]
    L += [f"n{idx[a]} -- n{idx[b]};" for a, b in edges]
    L.append("}")
    cmd = (["neato", "-Gmode=sgd", "-Tplain"] if len(ids) <= SGD_MAX
           else ["sfdp", "-Goverlap=prism", "-Tplain"])
    out = subprocess.run(cmd, input="\n".join(L),
                         capture_output=True, text=True, timeout=900)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[:300])
    pos = {}
    for ln in out.stdout.splitlines():
        f = ln.split()
        if f and f[0] == "node":
            pos[ids[int(f[1][1:])]] = [round(float(f[2]), 3), round(float(f[3]), 3)]
    return pos


def human_tree():
    """Build the human archive's tree sidecar in memory from the original Picbreeder
    branchFrom lineage (same id<->pid mapping as the published sprites/genomes, so the
    morph viewer's HF genomes key-match). No per-agent gen-0 zips exist for the human
    archive, so founders attach straight to the grey neutral-init super-root."""
    import importlib.util, types
    sys.modules.setdefault("umap", types.ModuleType("umap"))
    spec = importlib.util.spec_from_file_location(
        "bhs", os.path.join(REPO, "tools", "build_human_sprites.py"))
    bhs = importlib.util.module_from_spec(spec); spec.loader.exec_module(bhs)
    ids = bhs.human_ids()
    nodes = {}
    for e in bhs.human_lineage_entries(ids):
        src = e["source_entry_ids"]
        nodes[e["id"]] = {"p": src[0] if src else None, "a": None, "t": None, "col": True}
    return {"nodes": nodes}


def build_run(entry, tree=None):
    run, label, group = entry["run"], label_of(entry), group_of(entry)
    if tree is None:
        tree_path = os.path.join(TREE_DIR, run + ".json.gz")
        if not os.path.exists(tree_path):
            print(f"  SKIP {run[:48]}: no tree sidecar"); return None
        tree = load_gz(tree_path)
    nodes = tree["nodes"]
    roots = [i for i, n in nodes.items() if n.get("p") is None]

    # gen-0 ancestors (best-effort from local agent zips)
    ancestors, root_to_anc, anc_gen = {}, {}, {}
    for rid in roots:
        agent = str(nodes[rid].get("a"))
        anc_id = f"anc_a{agent}"
        if anc_id not in ancestors:
            gj = gen0_json(run, agent)
            if gj is None:
                continue
            ancestors[anc_id] = agent
            anc_gen[anc_id] = gj
        root_to_anc[rid] = anc_id

    # augmented graph: publications + ancestors + a grey "init" super-root, laid out
    # together so the solver centres it. EVERY root connects to init — via its
    # random-init ancestor when we have the gen-0 genome (grey→random→founder), else
    # directly (grey→founder). The grey genome is synthetic (no zip needed), so this
    # guarantees one connected, grey-rooted tree for every run, zips or not.
    INIT = "__init__"
    roots_set = set(roots)
    ids = list(nodes) + list(ancestors) + [INIT]
    edges = []
    for i, n in nodes.items():
        p = n.get("p")
        if p is not None and p in nodes:
            edges.append((p, i))
    for anc in ancestors:
        edges.append((INIT, anc))
    for rid in roots:
        edges.append((root_to_anc[rid], rid) if rid in root_to_anc else (INIT, rid))
    pos = neato(ids, edges)
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    bbox = {"minx": min(xs), "maxx": max(xs), "miny": min(ys), "maxy": max(ys)}

    out_nodes = {}
    for i, n in nodes.items():
        if i in pos:
            par = root_to_anc.get(i) or (INIT if i in roots_set else n.get("p"))
            out_nodes[i] = {"x": pos[i][0], "y": pos[i][1], "p": par,
                            "t": n.get("t"), "col": n.get("col", True) is not False, "a": n.get("a")}
    for anc, agent in ancestors.items():
        if anc in pos:
            out_nodes[anc] = {"x": pos[anc][0], "y": pos[anc][1], "p": INIT,
                              "t": "random init", "col": True, "a": agent, "is_anc": True}
    if INIT in pos:
        out_nodes[INIT] = {"x": pos[INIT][0], "y": pos[INIT][1], "p": None,
                           "t": "neutral init (grey)", "col": True, "is_init": True}

    run_dir = os.path.join(OUT_DIR, run)
    os.makedirs(run_dir, exist_ok=True)
    engine = "neato" if len(ids) <= SGD_MAX else "sfdp"
    with gzip.open(os.path.join(run_dir, "layout.json.gz"), "wt") as f:
        json.dump({"run": run, "label": label, "engine": engine, "bbox": bbox, "nodes": out_nodes}, f)
    # tiny ancestor-genome sidecar (every run) — published genomes come from HF
    with gzip.open(os.path.join(run_dir, "anc.json.gz"), "wt") as f:
        json.dump(anc_gen, f)
    # bundle full genomes (pub + anc) only for curated runs → offline blog figure
    bundled = False
    gen_path = os.path.join(SWEEP, run, "archive", "genomes.json.gz")
    if run in CURATED and os.path.exists(gen_path):
        merged = dict(load_gz(gen_path)); merged.update(anc_gen)
        with gzip.open(os.path.join(run_dir, "genomes.json.gz"), "wt") as f:
            json.dump(merged, f)
        bundled = True

    info = {"run": run, "label": label, "group": group, "n": len(nodes),
            "n_anc": len(ancestors), "bundled": bundled}
    flag = "bundled" if bundled else "HF    "
    print(f"  [{flag}] {label[:34]:34s} nodes {len(nodes):5d} anc {len(ancestors):3d}/{len(roots)}")
    return info


# dropdown group order; "Human" sits after the ablations, before catch-all "Other"
GROUP_ORDER = ["Baseline", "Memory length", "Selection noise", "Personalities",
               "Random selection", "Selector model", "Human", "Other"]


def build_human():
    """The human Picbreeder archive: tree from its branchFrom lineage, genomes from HF."""
    try:
        return build_run({"run": "human", "arc": "human"}, tree=human_tree())
    except Exception as ex:
        print(f"  ERR human: {ex}"); return None


def write_manifest(manifest):
    manifest.sort(key=lambda m: (GROUP_ORDER.index(m["group"]) if m["group"] in GROUP_ORDER else 99, m["label"]))
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump({"default": DEFAULT_RUN, "runs": manifest}, f, indent=1)
    print(f"wrote manifest with {len(manifest)} runs "
          f"({sum(1 for m in manifest if m['bundled'])} bundled, "
          f"{sum(1 for m in manifest if m['n_anc']) } with ancestors)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None

    # `human` (or any single-run arg): rebuild just the match(es) and MERGE into the
    # existing manifest, leaving every other run's data and ordering in place.
    if only:
        rebuilt = []
        if "human" == only or "human" in only:
            info = build_human()
            if info:
                rebuilt.append(info)
        idx = json.load(open(INDEX))
        for e in idx.get("runs", []):
            r = e.get("run")
            if not r or e.get("arc") == "human" or only not in r:
                continue
            try:
                info = build_run(e)
            except Exception as ex:
                print(f"  ERR {r[:48]}: {ex}"); info = None
            if info:
                rebuilt.append(info)
        mpath = os.path.join(OUT_DIR, "manifest.json")
        mf = json.load(open(mpath)) if os.path.exists(mpath) else {"default": DEFAULT_RUN, "runs": []}
        keep = {m["run"] for m in rebuilt}
        write_manifest([m for m in mf["runs"] if m["run"] not in keep] + rebuilt)
        return

    idx = json.load(open(INDEX))
    entries, seen = [], set()
    for e in idx.get("runs", []):
        r = e.get("run")
        if not r or r in seen or e.get("arc") == "human":   # human built separately below
            continue
        seen.add(r); entries.append(e)
    # include curated runs that aren't in index.json (e.g. Qwen, the Gemini-3 example)
    for run in CURATED:
        if run not in seen and os.path.exists(os.path.join(TREE_DIR, run + ".json.gz")):
            seen.add(run); entries.append(entry_from_runid(run))

    manifest = []
    for e in entries:
        try:
            info = build_run(e)
        except Exception as ex:
            print(f"  ERR {e['run'][:48]}: {ex}"); info = None
        if info:
            manifest.append(info)

    human_info = build_human()                 # the human archive (genomes fetched from HF)
    if human_info:
        manifest.append(human_info)

    write_manifest(manifest)


if __name__ == "__main__":
    main()
