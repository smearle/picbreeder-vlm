# Third-party code

Vendored external code, kept for reference, provenance, and — in the case of
`fer/` — as a live asset dependency. The evolutionary loop itself is the
neat-python-based reimplementation under `picbreeder_vlm/`; nothing here is part
of that loop.

## `webneat/`

The original Picbreeder Java client, written by **Nick Beato** (the `README.txt`
and `GETTING_STARTED.txt` are his; `progress.txt` is dated 12/15/06). We obtained
it via Sebastian Risi. Nothing here runs — it is consulted as the authoritative
record of how the original system behaved.

Two mutation parameters in `picbreeder_vlm/core/neat_components.py` were matched
against it (commit `f328bf8`): the weight range of newly-added connections
(`_random_weight`, now ±3.0) and the floor of mutation strength
(`picbreeder_weight_power_min`, now 0.01). It is also the source of truth for the
original evolve-screen appearance, which the interactive `breed/` site reproduces.

`tools/render_legacy_genome.py` parses the genome XML format this client writes.

The tree is preserved as received, including compiled `.class` files and the
bundled jars. Note that `webneat/jar/LICENSE` is a **Sun Microsystems** license
covering only the bundled toolbar icons (the `jlfgr` graphics set) — it is not a
license for the WebNEAT source, which carries none.

## `fer/`

Vendored from [akarshkumar0101/fer](https://github.com/akarshkumar0101/fer)
(Apache-2.0), the code for *The Fractured Entangled Representation Hypothesis*
(Kumar et al., 2025). Unlike `webneat/`, this is **not** inert: we read its
Picbreeder genome archive and pre-rendered human images from many build and
analysis scripts (`tools/build_human_*.py`, `tools/build_cppn_explainer.py`,
`picbreeder_vlm/analysis/*_human_archive.py`, and others). Resolve its location
through `picbreeder_vlm._paths.FER_ROOT` rather than hard-coding the path.

What the pipeline actually reads:

- `fer/spaghetti/pbRender/genomeAll/<pid>/` — published Picbreeder genomes
  (git-ignored, ~761M; regenerable / re-fetchable).
- `fer/src/archive_res-128/` — pre-rendered human archive images and embedding
  caches (git-ignored, ~736M; regenerable).
- `fer/data/picbreeder_*` and `sgd_*` — the skull/butterfly/apple CPPNs used by
  `build_cppn_explainer.py` and the inline-breeder seed. Tracked.

The upstream paper's own posterity (its `assets/` figure PDFs, the `cppn-x/`
Windows visualizer, and the two large analysis notebooks) was **removed** when we
vendored the trimmed snapshot — none of it is read by this repo. Recover it from
git history or upstream if ever needed. The `fer/src/*.py` lineage/phylogeny
scripts are our own additions alongside the upstream code.
