# Reference material

Third-party code kept for reference and for posterity. **Nothing here runs as part
of the experiments** — the evolutionary loop is the neat-python-based
reimplementation under `picbreeder_vlm/`.

## `webneat/`

The original Picbreeder Java client, written by **Nick Beato** (the `README.txt` and
`GETTING_STARTED.txt` are his; `progress.txt` is dated 12/15/06). We obtained it via
Sebastian Risi.

We consulted it as the authoritative record of how the original system behaved.
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
