# archive_animations

Methods for animating Picbreeder-VLM runs for the blog post. Two families:

1. **Archive structure** — how the shared archive grows and *branches* over time.
2. **CPPN interpolation** — morphing through the CPPNs an agent selected across
   the 20 generations of a single session.

All scripts use the project venv (NEAT, OpenCV, PIL, numpy) and `ffmpeg`:

```bash
.venv/bin/python archive_animations/<script>.py ...   # run from repo root
```

Outputs go to `archive_animations/out/` (gitignored).

---

## Data sources

A completed run lives under e.g.
`sweep_logs/sweep/th0_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s4/`
(3,427 published images — one of the richest archives; `s3`/`s5` are similar).

- `archive/archive_metadata.json` → `entries`, each with:
  `id` (e.g. `img_000001`), `title`, `added_at` (publication order),
  `source_entry_ids` (parent **archive** image it was branched from; empty = root),
  `n_published_children`, `agent_id`, `generation`, `color_enabled`, …
  ⚠️ `image_path`/`genome_path` are **stale absolute paths** from the original
  cluster (`/scratch/...`); use basenames against the local `archive/images/{id}.png`.
- `archive/images/{id}.png` → pre-rendered 128² images (no NEAT needed to reuse).
- `agents/agent_NNN.zip` → per-agent data. Inside, `populations/selected_gen_NNN.pkl.gz`
  is `{"generation": N, "selected": [{"genome": PicbreederGenome, "genome_key", …}]}`
  — the genome the agent actually chose at gen N. Consecutive selections form the
  session lineage. (Unzip an agent into `_work/` before using `cppn_interp.py`.)

Phylogeny of the s4 run: 76 roots, 3,351 branched, max tree depth 28, and a long
tail of leaves with a few heavy hubs (one image has 42 published children) — the
imbalanced "re-branch from favorites" structure the paper describes.

---

## Scripts

### `anim_render.py` — shared reveal renderer
`render_reveal(out, positions, parent, order, thumbs, ...)` drives every
structure animation. Given continuous node positions + a reveal order it:
- draws lineage edges **behind** the thumbnails (two-layer composite, so a newly
  spawned line never paints over an existing image);
- colours each edge by its **root lineage** (`lineage_colors()`; pass
  `edge_color=`), so families read as distinct colours -- or grey via `--plain-edges`;
- optionally **zooms out**: each frame crops the revealed bounding square from a
  fixed hi-res "world" canvas and resizes to the output frame, so the view always
  frames exactly the current archive and zooms out as it grows;
- outlines the growth front in Sakana orange -- **both the new child and its parent**.
Pacing: `per_frame` = images revealed per step (keep this at the natural rate);
`frames_per_step` = how many frames to **dwell** on each step (raise this to slow
down without trickling fewer images in -- compact growth defaults to 8 imgs/step
× 2 frames). To re-time a finished clip: `ffmpeg -i in.mp4 -filter:v
"setpts=2.0*PTS" out.mp4`.

### `archive_grow_compact.py` — compact, zooming growth (closest to the brief)
Places images on an integer lattice in publication order: each new image takes
the free cell nearest its parent, tie-broken toward the centroid
(`--placement compact`, dense square-ish blob; new nodes land on the growing
perimeter) or away from it (`--placement outward`). Fed to `render_reveal` with
zoom on.

```bash
.venv/bin/python archive_animations/archive_grow_compact.py --run sweep_logs/sweep/th0_..._s4 \
  --out archive_animations/out/archive_compact_s4.mp4 --frame 1000 --per-frame 8 --fps 24
```
`--node-size children` scales each thumbnail by how many published children it
has (`--size-min`/`--size-max`, in × `--thumb-px`), so hub "ancestor" images grow
into the centre of their colored lineage burst; default `uniform` keeps them equal.

Caveat: placement is incremental (no global re-pack), so a hub that keeps
spawning yields some long edges reaching to the rim -- inherent to embedding a
hubby tree in a compact grid.

### `archive_tree.py` — branching structure (force-directed / radial)
Builds the phylogenetic forest (`source_entry_ids[0]` = parent), lays it out, and
reveals nodes in publication order with edges drawn to parents (orange = growth
front). Image thumbnails are the nodes.

```bash
.venv/bin/python archive_animations/archive_tree.py \
  --run sweep_logs/sweep/th0_..._s4 --out archive_animations/out/tree_sfdp.mp4 \
  --engine sfdp --size 1200 --thumb 15 --per-frame 22 --fps 24 --edge-shade 228
```
`--engine`: `sfdp`/`fdp`/`neato` (force-directed; **sfdp looks best** — lineages
form distinct "galaxies", hubs become dense clusters), `twopi` (graphviz radial),
`radial` (built-in radial; crude). Force engines need graphviz (`sfdp` etc., already installed).
Rendering goes through `anim_render.render_reveal`, so edges sit behind thumbnails
automatically; add `--zoom` for the zoom-out viewport (best paired with a layout
whose early publications are spatially close, e.g. the compact lattice).

**Speed:** every script is slowed by lowering `--per-frame`. To half-speed an
existing MP4 without re-rendering: `ffmpeg -i in.mp4 -filter:v "setpts=2.0*PTS" out.mp4`.

### `cppn_interp.py` — session morph
Morphs (weights + activations, via `render_lineage_animation.build_superset`/
`render_frames`) along the **true parent→child lineage of the published image**.

Reads each genome's recorded `parents` and walks the first-parent chain back
from the published genome to the root, so every step is one real mutation
(ring→ring→bowl→smile→fish).

We also **dedupe** identical consecutive genomes into keyframes (the agent re-picks
exact copies for many generations); render each keyframe **canonically** as the
exact bracket between segments (guaranteed continuity); and allocate frames
**proportional to actual visual change** (`--min-steps`..`--max-steps`, scaled by
`--steps-per-unit`).

```bash
unzip -oq sweep_logs/sweep/th0_..._s4/agents/agent_000.zip -d archive_animations/_work
.venv/bin/python archive_animations/cppn_interp.py \
  --agent-dir archive_animations/_work/agent_000 \
  --out archive_animations/out/cppn_interp_agent000.mp4 \
  --size 256 --fps 24 --min-steps 8 --max-steps 60 --steps-per-unit 2.0
```
Note: 256² CPPN eval is pure-Python and slow (~minutes); keep size modest while
iterating. A genuinely large CPPN step (e.g. a ring "opening up" when a new gene
activates) is nonlinear and will still morph fast even over many frames; this
is inherent to the genome change. `--max-steps` spreads it across more frames
so it reads as a morph rather than a cut.

### `sweep_animations.py` — render across the whole sweep + browse
Renders an animation for every run under `sweep_logs/` (skipping empties) and
writes a browsable `index.html` (videos grouped by setting with seed stripped,
labelled by image count) so you can scan many archives and pick favourites.
Defaults to a fast/small **quick** preset, parallelised across cores; re-render
favourites at full quality with the per-run scripts afterwards.

```bash
# all runs, compact quick, 10 workers -> out/sweep/compact/index.html
.venv/bin/python archive_animations/sweep_animations.py --jobs 10
# filter, choose layout, full quality, or just rebuild the index:
.venv/bin/python archive_animations/sweep_animations.py --include model-gemini-2.5-pro
.venv/bin/python archive_animations/sweep_animations.py --script tree
.venv/bin/python archive_animations/sweep_animations.py --preset full --include randp0.25
.venv/bin/python archive_animations/sweep_animations.py --index-only
```
Outputs: `out/sweep/<script>/<run-name>.mp4` + `index.html`. Skips already-rendered
clips (resumable); `--overwrite` to force. The index refreshes after each clip,
so you can open it mid-run. Run names encode the paper axes: `th{N}`=context
length (`th-1`=full), `randp{X}`=exploration noise &epsilon;, `traits{N}`=number of
agents, `model-*`=VLM, `randp2 rmode-all` (no model)=the random baseline.

### `archive_grow_square.py` — always-square, border-fill, smooth zoom
Lays images in publication order along concentric square rings from the centre
out, so each step's newest ~20 land **around the border** of an always-square
block; the viewport zooms out **continuously** (`half-side ∝ sqrt(count)`) to keep
the whole square framed. Centre = earliest publications, outer rings = latest
(nicely shows drift over time). Edges are omitted (hidden in a packed grid); the
newest border images get the orange outline.

```bash
.venv/bin/python archive_animations/archive_grow_square.py \
  --run sweep_logs/sweep/<run> --out archive_animations/out/archive_square.mp4 \
  --frame 1000 --per-step 20 --cell-px 26 --fps 24
```
Slow it via lower `--fps` or `--per-step`. Longest LOCAL runs: `..._randp2_..._s3`
(3,592, random baseline) and `th0_..._s4` (3,427, gemini). The 9,377-agent
`LongSweep`/`LongSweep2` seeds (see `sweep_configs.py`) are not on local disk.

### `archive_scroll.py` — fixed-zoom scrolling feed with parent→child morphs
Fixed-zoom grid of `--cols` columns. **Sequential, phase-separated** per row so
motion and morph never overlap:

1. **Fill phase** — the first `--rows-visible` generations land **top-to-bottom**
   (no scrolling) until the frame is full.
2. **Steady phase** — each later row runs: **scroll** up one row (top exits, bottom
   clears) → **descend** into the bottom row → **morph in place** (stationary).

Branched children descend from their **parent's on-screen tile**; **roots appear
in place** on their slot (random-init, no parent) — both then CPPN-**morph from that
source into the published form** (parent archive genome for branched; the session's
gen-0 random-init genome from `agents/<agent>.zip → selected_gen_000.pkl.gz` for
roots, choosing the gen-0 pick sharing the most connection genes with the published
genome when the agent crossed over).

Pacing: `--descent-frames` (the fall is rendered 1.5× faster than this), `--morph-frames`
(in-place morph), `--scroll-frames` (between rows), `--hold` (final). `--outline`
rings falling/morphing tiles.

```bash
.venv/bin/python archive_animations/archive_scroll.py --run sweep_logs/sweep/th0_..._s4 \
  --out archive_animations/out/archive_scroll.mp4 --n 200 --cols 9 --cell 64 \
  --morph-steps 10 --morph-frames 18 --row-frames 18 --jobs 12
```
Needs genomes for each image **and its parent**, so `--n` must stay within the
range the run saved (≈first 1000 for `th0_..._s4`). CPPN morph precompute is
parallelised (`--jobs`); `--row-frames`/`--morph-frames` control scroll/morph
speed; `--outline` rings tiles still mid-morph.

### `archive_scroll_lineage.py` — continuous scrolling feed (this is the on-site "being born" video)
The continuous-scroll variant of `archive_scroll.py`: the grid scrolls up
**smoothly** (never pausing) in strict feed order (`col = i % C`, `row = i // C`),
parents are never reorganized. This is the script behind the blog's
`assets/scrolls/scroll_randp*.mp4` feeds.

Each steady row is one event of `--descent-frames`: the whole grid scrolls up
one row while the new clones **descend into the bottom row and CPPN-morph from
their source into their published form in the same motion** (morph rides the
descent — a row is "born" in one move rather than falling first and morphing
after). A branched child launches from its parent's on-screen position; once the
parent has **scrolled out of view** it launches from the **invisible row just
above the top** (screen-row -1) at the parent's column — a short fall from above
the top, not a long diagonal back to the distant parent. Roots (session
founders) rise from below the bottom row, morphing out of a random-initial CPPN.
On a 1,500-pub feed most parents are out of view, so the above-top launch is the
common case.

```bash
.venv/bin/python archive_animations/archive_scroll_lineage.py \
  --run sweep_logs/sweep/th0_..._s4 --out archive_animations/out/archive_scroll_lineage.mp4 \
  --n 200 --cols 9 --rows-visible 7 --cell 88 --morph-steps 10 --jobs 12
```
Pacing: `--descent-frames` (combined descent+morph+scroll window per row — bump
it if the morph reads too fast), `--hold` (final). `--morph-frames` is now
ignored (kept so old commands still parse).

### `archive_family.py` — reorganizing grid that keeps parents on-screen
A fixed `--cols`×`--rows` grid that stays full and continually refreshes so a
**parent is always visible the moment its child is branched**. Children are
processed in waves of `--cols`; each new child flies in from its parent's tile to
the nearest **free or "done"** slot ("done" = a tile with no remaining children)
and morphs from a clone of its source. A tile is **never evicted while it still
has children to spawn** (and is re-introduced if it ever was), so branches always
originate from a visible parent; roots appear in place and morph from random-init.
Settled tiles don't move — the grid "reorganizes" by what fills each slot.

Feasible because at most ~19 parents are active at once (≪ 63 slots), so only
"done" tiles are ever evicted.

```bash
.venv/bin/python archive_animations/archive_family.py --run sweep_logs/sweep/th0_..._s4 \
  --out archive_animations/out/archive_family.mp4 --n 200 --cols 9 --rows 7 --cell 88 \
  --morph-frames 30 --descent-frames 20 --jobs 12
```
`--outline` rings the flying/morphing tiles (helps see short fly-ins on a full grid).

### `archive_growth.py` — row-major fill (superseded)
Fills a grid in publication order. Kept for reference, but it doesn't reveal
branching structure (the final frame ≈ the whole story), so prefer `archive_tree.py`.

---

### `agent_life/` — life of a single agent (interactive web player)
Walks one agent's 20-generation session: candidate grids (picks outlined), the
image it grows (CPPN morph along the true published lineage), reasoning (current +
scrolling history), and **TTS narration** of that reasoning. A self-contained
static folder for the blog. See `agent_life/README.md`.
```bash
.venv/bin/python archive_animations/agent_life/build_assets.py \
  --agent-dir archive_animations/_work/agent_000 \
  --out archive_animations/agent_life/out/agent_000 --title "Minimalist Fish"
.venv/bin/python archive_animations/agent_life/tts.py \
  --out archive_animations/agent_life/out/agent_000 --backend edge
( cd archive_animations/agent_life/out/agent_000 && python3 -m http.server 8772 )
```

## TODO / ideas
- True **center-out temporal bloom**: keep angle from a force/radial layout but set
  radius ∝ `added_at` so the archive literally grows outward from the center.
- Human-archive tree for side-by-side contrast (balanced vs imbalanced).
- `agent_life`: stitch the per-gen assets into a single narrated MP4 (the
  interactive player already overlays reasoning on the lineage morph).
