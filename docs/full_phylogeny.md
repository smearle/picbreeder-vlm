# Picbreeder Global Phylogeny

`fer/src/plot_full_phylogeny.py` rebuilds the entire Picbreeder tree by stitching together every lineage and identifying the parent/child overlap across runs. Thicker, labeled edges in the rendered figure correspond to branches that are reused by multiple descendant lineages.

## Quick Start

```bash
/Users/samearle/picbreeder-vlm/.venv/bin/python fer/src/plot_full_phylogeny.py \
  --pb-dir fer/spaghetti/pbRender/genomeAll \
  --output human_lineages/lineages/full_phylogeny \
  --format pdf
```

The script automatically scans every pid folder under `--pb-dir`, reconstructs parent pointers from each `main.zip`, and writes a Graphviz figure to `<output>.<format>`. If the provided `--output` already has a suffix it will be stripped before rendering.

## Useful Flags

- `--limit N`&mdash;restricts the number of pid folders that are inspected, which is handy for smoke tests.
- `--min-edge-weight K`&mdash;hides edges that only appear in fewer than `K` distinct lineages so that heavily reused branches are easier to see.
- `--rankdir {LR,TB}`&mdash;set to `TB` if you prefer the tree to grow top-to-bottom instead of left-to-right.

Graphviz must be installed and discoverable in your `PATH` because the script calls `graphviz.Digraph.render(...)` under the hood. tqdm is used for progress reporting, and both dependencies are already listed in `requirements.txt`.
