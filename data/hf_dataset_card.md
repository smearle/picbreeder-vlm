---
license: cc-by-nc-4.0
pretty_name: Picbreeder-VLM Archive
language:
- en
tags:
- open-endedness
- evolutionary-computation
- cppn
- neat
- vision-language-models
- generated-images
- picbreeder
task_categories:
- image-to-text
size_categories:
- 100K<n<1M
annotations_creators:
- machine-generated
source_datasets:
- original
configs:
- config_name: images
  data_files:
  - split: default
    path: data/images/default/*.parquet
  - split: mem_0
    path: data/images/mem_0/*.parquet
  - split: mem_2
    path: data/images/mem_2/*.parquet
  - split: mem_10
    path: data/images/mem_10/*.parquet
  - split: mem_20
    path: data/images/mem_20/*.parquet
  - split: noise_0_05
    path: data/images/noise_0_05/*.parquet
  - split: noise_0_25
    path: data/images/noise_0_25/*.parquet
  - split: noise_0_5
    path: data/images/noise_0_5/*.parquet
  - split: noise_0_75
    path: data/images/noise_0_75/*.parquet
  - split: noise_1_0
    path: data/images/noise_1_0/*.parquet
  - split: random
    path: data/images/random/*.parquet
  - split: agents_10
    path: data/images/agents_10/*.parquet
  - split: agents_100
    path: data/images/agents_100/*.parquet
  - split: agents_1000
    path: data/images/agents_1000/*.parquet
  - split: model_ablation
    path: data/images/model_ablation/*.parquet
---

# Picbreeder-VLM Archive

Every image evolved by the swarm of vision-language-model "breeders" in
**[In Search of the Ingredients of Open-Endedness: Replicating Picbreeder with Large Vision-Language Models](https://arxiv.org/abs/2605.23908)**
(GECCO 2026), together with the CPPN genomes that produced them, the agents' reasoning transcripts, the
lineage graphs, and the analysis artifacts behind the paper and blog.

The original [Picbreeder](https://en.wikipedia.org/wiki/Picbreeder) (Secretan et al., 2008) let crowds of
*humans* collaboratively evolve images from
[CPPN](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network) genomes, discovering an
open-ended tree of recognizable pictures. This dataset is what happened when we replaced the humans with
VLM agents: agents join a shared archive, look at candidate images, pick and mutate them toward whatever
they "see," and publish their discoveries for others to branch from.

- 📝 **Blog / interactive report:** <https://pub.sakana.ai/picbreeder-vlm>
- 📄 **Paper (arXiv):** <https://arxiv.org/abs/2605.23908>

## How the images were made

Every image here was produced by the same loop, run thousands of times. Its unit is a **session**.

A session begins when a VLM agent either **branches** an image already published to the shared archive,
or starts from a fresh population of randomly initialized CPPNs. The agent is shown a population of 15
CPPN-images and selects one or several as parents. Random mutation — and crossover, when it selects more
than one — produces the next population, and the agent selects again; exact copies of the parents are
always carried among the offspring. At the **20th generation** the agent must choose one image to publish
to the archive and give it a title. That published image, with its genome and its lineage, is one row of
this dataset.

At any step the agent may toggle color mode, set mutation strength anywhere in `[0.01, 1]` (default
`0.5`), and, in color mode, restrict mutation to color-only or structure-only. Random initial populations
start grayscale; branched images inherit the color mode they were published under. These controls mirror
the original Picbreeder interface.

Agents act **concurrently against one shared archive**, so an image published by one agent becomes
available for another to branch. That is what makes a run a single collaborative phylogeny rather than a
bundle of independent searches. Each run is 2,000 sessions.

The experimental conditions perturb exactly three things about this loop: how often an agent's selection
is overridden by a **random** one (ε), how much of its own history the agent **remembers**
(`memory_cl`), and how many distinct **personalities** the agent pool draws from (`n_personalities`).

## The paper's sweep

The results in the paper come from **185,013 images across 81 runs and 14 conditions**, all bred by
`gemini-2.5-pro`, six seeds per condition. These are exactly the rows where

```python
canonical == True and model == "gemini-2.5-pro"
```

Everything else here is supplementary — see [Beyond the paper](#beyond-the-paper) below.

```python
from datasets import load_dataset

ds = load_dataset("picbreeder-vlm/picbreeder-vlm-archive", "images", split="default")
paper = ds.filter(lambda r: r["canonical"] and r["model"] == "gemini-2.5-pro")

paper[0]["image"]      # PIL.Image (128x128)
paper[0]["caption"]    # "An abstract black and white oval shape with a motion blur effect."
paper[0]["memory_cl"]  # 1
```

Browse the images directly in the **Dataset Viewer** above — sort by `vlm_rating_mean`, filter on
`canonical`, or full-text search `caption`.

### Splits

Each split is an **experimental condition** from the sweep.

| Split | Seeds | Paper images | Condition |
| --- | ---: | ---: | --- |
| `default` | 6 | 15,755 | The standard configuration: 20 agents, `memory_cl = 1`, no noise. |
| `mem_0` | 6 | 15,922 | **Memory ablation** — agent retains no prior selections. |
| `mem_2` | 6 | 14,883 | Memory ablation, `memory_cl = 2`. |
| `mem_10` | 6 | 15,569 | Memory ablation, `memory_cl = 10`. |
| `mem_20` | 6 | 15,140 | Memory ablation, unbounded memory (`memory_cl = -1`). |
| `noise_0_05` | 6 | 12,000 | **Selection-noise ablation**, ε = 0.05. |
| `noise_0_25` | 6 | 13,002 | Selection-noise ablation, ε = 0.25. |
| `noise_0_5` | 6 | 12,001 | Selection-noise ablation, ε = 0.5. |
| `noise_0_75` | 6 | 12,000 | Selection-noise ablation, ε = 0.75. |
| `noise_1_0` | 6 | 12,001 | Selection-noise ablation, ε = 1.0. |
| `random` | 3 | 10,731 | **Control** — every selection random (ε = 2.0); the VLM is never consulted. |
| `agents_10` | 6 | 12,001 | **Agent-diversity ablation** — 10 distinct agent personalities. |
| `agents_100` | 6 | 12,007 | Agent-diversity ablation, 100 personalities. |
| `agents_1000` | 6 | 12,001 | Agent-diversity ablation, 1000 personalities. |

ε is the probability that an agent's selection is short-circuited and replaced by a random pick.
`memory_cl = -1` is a sentinel meaning *unbounded*, which the paper labels `mem_20`.

### Where the annotations come from

Neither the ratings nor the captions were visible to the agents while they evolved. **Both are post-hoc
annotations, added after the runs finished, to make the archives measurable.** Treat them as model
outputs under study, not as ground truth and not as the selection signal.

**Ratings** (`vlm_rating_mean`, `vlm_rating_count`) come from a separate pass in which a VLM is shown
batches of published images and asked to score each **1–5** with a one-sentence justification. An image
may be scored in more than one pass, so `vlm_rating_count` is the number of scores its mean is taken
over — it varies per image and is *not* a fixed panel size. Images published earlier tend to accumulate
more scores, so `vlm_rating_count` correlates with publication time; keep that in mind before reading
`vlm_rating_mean` as a like-for-like quality ranking.

**Captions** come from a `gemini-2.5-pro` pass over each archive. Their purpose is evaluation: the
captions are embedded in a text-embedding space, and the spread of that cloud is the paper's **semantic
coverage** metric. In practice a caption is a single descriptive sentence — *"A stylized, black-and-white
drawing of a bird's head against a heart-shaped background."* — with a median of 11 words.

Some runs were captioned only up to a per-run cap, applied in publication order. Where a run is partly
captioned, **the captioned rows are a chronological prefix of it, not a random sample** — so a
caption-based analysis of such a run is an analysis of its earlier sessions. Filter `caption is not null`
and check coverage per `run` before pooling.

The paper's *Semantic Recall* metric uses no captions at all: it embeds the images themselves with
SigLIP2 and measures cosine distance to the 1,824 deduplicated THINGS class names.

### Columns

Columns contain image metadata, including the image's provenance from within the experimental run that
generated it and the experiment's hyperparameters.

| Column | Type | Description |
| --- | --- | --- |
| `image` | `Image` | The evolved image, 128×128 (see *Image provenance*). |
| `image_id` | `string` | Per-run identifier, e.g. `img_000001`. |
| `run` | `string` | Full sweep-run name — the opaque provenance key. |
| `arc` | `string` | Condition key (matches the split, un-sanitized: `noise_0.05`). |
| `model` | `string` | VLM backend that acted as the breeder. |
| `seed` | `int64` | Run seed. |
| `memory_cl` | `int64` | Agent memory context length; `-1` = unbounded. |
| `noise_eps` | `float64` | Random-selection probability ε; `0.0` = none, `2.0` = fully random. |
| `n_personalities` | `int64` | Number of distinct agent personalities; `0` = homogeneous. |
| `canonical` | `bool` | Whether this run is the standard config for its condition. |
| `generation` | `int64` | Evolutionary generation at which the image was published. |
| `agent_id` | `string` | Which agent published it. |
| `genome_key` | `int64` | CPPN genome id (joins to the genome archives). |
| `parent_genome_key` | `int64` | Primary parent's genome id; null for roots. |
| `n_published_children` | `int64` | How many published images branched from this one. |
| `color_enabled` | `bool` | Whether the CPPN rendered in color. |
| `caption` | `string` | `gemini-2.5-pro` description; null where that pass didn't cover the image. |
| `caption_qwen3_vl_8b` | `string` | Second caption set, from a `qwen3-vl-8b` pass over 13 paper runs. |
| `vlm_rating_mean` | `float64` | Mean of the 1–5 ratings this image received. |
| `vlm_rating_count` | `int64` | How many ratings that mean is over (`0` where unrated). |

A caption is a fact about an *(image, captioner)* pair, so each captioner gets its own column. The column
name records the captioner, never the breeder — a `gemini-2.5-pro` row can carry a `qwen3-vl-8b` caption.

### Image provenance

The `image` column carries the **128×128 thumbnail** — the exact pixels the blog gallery renders,
re-encoded to WebP from each run's sprite atlas. These are thumbnails, not the original renders.

Because every image is produced by a **CPPN**, it is resolution-independent: you can re-render any row at
arbitrary resolution from the genome archives (`site/<run>/genomes.json.gz`, or
`results/<run>/genomes.tar.gz`), joining on `genome_key`.

---

## Beyond the paper

Alongside the sweep, this repo ships **78,589 images from 98 supplementary runs**: alternative VLM
backends, and non-canonical variants of the sweep conditions. They share the same splits, so filter on
`canonical and model == "gemini-2.5-pro"` when you want the paper's rows and nothing else.

| Breeder model | Runs | Images |
| --- | ---: | ---: |
| `gemini-2.5-pro` (non-canonical variants) | 41 | 26,133 |
| `qwen3-vl-30b-fp8` | 27 | 24,060 |
| `qwen3-vl-8b` | 12 | 10,359 |
| `gemini-3-pro-preview` | 9 | 9,002 |
| `gemini-2.5-flash-lite` | 3 | 3,017 |
| `gemini-2.5-flash` | 3 | 3,010 |
| `gemini-random` | 3 | 3,008 |

The `model_ablation` split holds the runs that vary the breeder model, and is therefore entirely
supplementary. **No supplementary run was captioned** — the captioning pass only ever ran over the
paper's sweep — so both caption columns are null throughout them, and about half their images (48%)
carry a rating. Filter `caption is not null` before any caption-based analysis.

### What's not in the parquet

Per-rater rating vectors and per-rater comments (tens of values per image) are omitted to keep the
parquet lean and browsable; `vlm_rating_mean` / `vlm_rating_count` summarize them. The full vectors live
in `results/<run>/archive_metadata.json`.

## Repository layout

The parquet under `data/` is a **browsable face** of the corpus. The rest of the repo is the raw
material, kept in the layout the interactive blog fetches:

| Path | Contents |
| --- | --- |
| `data/images/<condition>/<run>.parquet` | Per-image rows (this card's `images` config). |
| `site/<run>/sprite/` | Sprite atlases (`sheets/*.webp` + `layout.json`) the gallery renders. |
| `site/<run>/genomes.json.gz` | Renderable CPPN genomes for the run. |
| `results/<run>/` | Repro artifacts: `genomes.tar.gz`, `agents.tar` (agent logs), `archive_metadata.json`, captions, phylogeny + embedding metrics, `embeddings_*.npz`. |
| `transcripts/<run>/agent_*/` | Per-agent reasoning transcripts (`transcript.json`) + the image atlas each agent saw. |
| `hero_sprites/` | Hand-picked images used in the blog's hero animation. |
| `index.json` | Manifest of every run: condition, seed, model, and which artifacts it has. |

## Limitations and intended use

- The images are **machine-generated abstract art**, not natural images, and the captions and ratings are
  **machine-generated** too. Treat `caption` and `vlm_rating_mean` as *model outputs under study*, not
  ground truth — a large part of the paper is about how unreliable they are.
- The `random` split is a control in which selection carries no signal by construction. Don't pool it
  with the others without meaning to.
- Selection in the noise ablations is randomized by design — that *is* the ablation.
- This corpus is intended for research on open-endedness, novelty search, and VLM evaluation behavior. It
  is not a benchmark, and it has no held-out test set.

## Citation

```bibtex
@inproceedings{earle2026picbreedervlm,
  title     = {In Search of the Ingredients of Open-Endedness: Replicating Picbreeder with Large Vision-Language Models},
  author    = {Earle, Sam and Arulkumaran, Kai and Dai, Andrew and Kumar, Akarsh and Togelius, Julian and Risi, Sebastian},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference (GECCO)},
  year      = {2026}
}
```

Please also credit the original Picbreeder: Secretan et al., *Picbreeder: Evolving Pictures Collaboratively
Online*, CHI 2008.

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — attribution required, non-commercial use
only.
