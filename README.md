# Picbreeder‑VLM

### In Search of the Ingredients of Open‑Endedness: Replicating Picbreeder with Large Vision‑Language Models

Sam Earle, Kai Arulkumaran, Andrew Dai, Akarsh Kumar, Julian Togelius, Sebastian Risi — **GECCO 2026** (Best Paper nominee)

[**📝 Blog / interactive report**](https://smearle.github.io/picbreeder-vlm-06b0d76d/) ·
[**📄 Paper (arXiv)**](https://arxiv.org/abs/2605.23908) ·
[**🤗 Archive dataset**](https://huggingface.co/datasets/picbreeder-vlm/picbreeder-vlm-archive) ·
[**🧬 Breed your own (demo)**](https://smearle.github.io/picbreeder-vlm-06b0d76d/breed/)

---

The original **Picbreeder** (Secretan et al., 2008) let *crowds of humans* evolve
images collaboratively, discovering a famous open‑ended tree of recognizable
pictures (the Skull, the Butterfly, the Car…) from simple
[CPPN](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network)
genomes. This project asks whether a **swarm of Vision‑Language Models**, standing
in for the human breeders, can reproduce that open‑ended dynamic — agents join a
shared archive, look at candidate images, pick and mutate them toward whatever
they "see," and publish their discoveries for others to branch from.

This repo contains the full research codebase: the multi‑agent evolutionary
simulation, the VLM backends, the NEAT/CPPN engine, the analysis & figure
pipeline behind the paper and blog, and the tooling for the interactive
[Picbreeder homage site](#-the-picbreeder-homage-breed-your-own).

## Links

| | |
| --- | --- |
| 📝 **Blog / interactive report** | <https://smearle.github.io/picbreeder-vlm-06b0d76d/> |
| 📄 **Paper (arXiv)** | <https://arxiv.org/abs/2605.23908> |
| 🤗 **Archive dataset** (evolved genomes, images, lineages, VLM captions) | <https://huggingface.co/datasets/picbreeder-vlm/picbreeder-vlm-archive> |
| 🧬 **Breed‑your‑own demo** (CPPN homage in the browser) | <https://smearle.github.io/picbreeder-vlm-06b0d76d/breed/> |
| 🛰️ **Community API** (Space backing the demo) | <https://huggingface.co/spaces/picbreeder-vlm/picbreeder-community-api> |
| 🤗 **Community dataset** (user‑bred genomes) | <https://huggingface.co/datasets/picbreeder-vlm/picbreeder-vlm-community> |

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # exposes the `picbreeder_vlm` package + pickle-compat shims
```

## Quickstart

Run a small collaborative sweep locally (no Slurm). With no cloud API key you can
drive the pipeline entirely from a **local Qwen VLM** backend:

```bash
# start a local Qwen vLLM server (see the script for model/GPU options)
./serve_local_vlm.sh

# a short local run against it (test_mode caps to 2 agents × 3 generations)
.venv/bin/python -m picbreeder_vlm.experiments.sweep \
    sweep_name=chat_history_turns slurm=false \
    model=qwen3-vl-8b test_mode=true
```

Outputs land in `logs_collaborative/sweep/<sweep_name>/<run>/` (evolved images,
genome `.pkl`s, and JSON archive snapshots). See **[AGENTS.md](AGENTS.md)** for the
full sweep / evaluation / cross‑eval workflow and the local‑VLM server setup.

## Repository map

Source lives in the **`picbreeder_vlm/`** package:

| subpackage | contents |
| --- | --- |
| `picbreeder_vlm.core` | config, CPPN/NEAT genome (`neat_components`), reproduction, rendering, archive management |
| `picbreeder_vlm.vlm` | VLM backends (Gemini / Qwen / …), prompts, chat/conversation history |
| `picbreeder_vlm.agents` | the collaborative multi‑agent loop, single‑agent runner, human interactive mode |
| `picbreeder_vlm.experiments` | sweep orchestration (Hydra + Submitit), run configs, experiment CLIs |
| `picbreeder_vlm.analysis` | coverage / embedding / captioning / phylogeny / rating metrics |
| `picbreeder_vlm.viz` | figure and lineage/archive visualization |
| `picbreeder_vlm.niches` | CLIP noun‑niche evolution‑strategy experiments |
| `picbreeder_vlm.nouns` · `picbreeder_vlm.bench` | vocabulary wrangling · VLM benchmarking/probing |

Other top‑level directories: **`tools/`** (blog & paper asset builders),
**`archive_animations/`** (lineage/teaser animations), **`paper/`** (LaTeX + figure
sources), **`community/`** (the Hugging Face Space that backs the breed demo),
**`webneat/`** (the original Java WebNEAT client, kept for reference / legacy genome
parsing), **`picture2d/`** (legacy Picbreeder NEAT config & rendering).

> **Note — pickle‑compat shims.** The thin modules `neat_components.py`,
> `config.py`, `picbreeder_reproduction.py`, `archive_manager.py` and
> `rendering.py` at the repo root simply re‑export from `picbreeder_vlm/`. They
> exist so that genome `.pkl` files in the archive dataset (which store their
> original module paths) still load. New code should import from the package.

## 🧬 The Picbreeder homage — breed your own

Alongside the VLM experiments we host a small, faithful browser reimplementation
of Picbreeder where **you** can pick and mutate CPPN images, publish your
discoveries, and branch off other people's:

- **Try it:** <https://smearle.github.io/picbreeder-vlm-06b0d76d/breed/>
- Published genomes are stored in the **community dataset**
  (`picbreeder-vlm/picbreeder-vlm-community`) via a small FastAPI gateway that
  runs as a **Hugging Face Space** — its source is in [`community/`](community/).

## Data

The [**archive dataset**](https://huggingface.co/datasets/picbreeder-vlm/picbreeder-vlm-archive)
holds the evolved archives across runs: rendered images, CPPN genomes, full
lineages, per‑image VLM captions/ratings, and the sprite sheets + orderings that
power the interactive gallery in the blog.

## Citation

```bibtex
@inproceedings{earle2026picbreedervlm,
  title     = {In Search of the Ingredients of Open-Endedness: Replicating Picbreeder with Large Vision-Language Models},
  author    = {Earle, Sam and Arulkumaran, Kai and Dai, Andrew and Kumar, Akarsh and Togelius, Julian and Risi, Sebastian},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '26)},
  year      = {2026}
}
```

## Acknowledgements

Builds on the original **Picbreeder** (Secretan, Beato, D'Ambrosio, Rodriguez,
Campbell & Stanley, 2008) and the NEAT/CPPN lineage of work by Kenneth O. Stanley
and collaborators. The `webneat/` client is the original WebNEAT code by Nick Beato.
