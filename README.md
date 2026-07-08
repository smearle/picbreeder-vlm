# Picbreeder‑VLM

### In Search of the Ingredients of Open‑Endedness: Replicating Picbreeder with Large Vision‑Language Models

Sam Earle, Kai Arulkumaran, Andrew Dai, Akarsh Kumar, Julian Togelius, Sebastian Risi—**GECCO 2026** (Best Paper nominee)

[**📝 Blog / interactive report**](https://pub.sakana.ai/picbreeder-vlm) ·
[**📄 Paper (arXiv)**](https://arxiv.org/abs/2605.23908) ·
[**🤗 Archive dataset**](https://huggingface.co/datasets/picbreeder-vlm/picbreeder-vlm-archive) ·
[**🧬 Breed your own (demo)**](https://pub.sakana.ai/picbreeder-vlm/breed/)

---

The original **Picbreeder** (Secretan et al., 2008) let *crowds of humans* evolve
images collaboratively, discovering a famous open‑ended tree of recognizable
pictures (the Skull, the Butterfly, the Car…) from simple
[CPPN](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network)
genomes. This project asks whether a **swarm of Vision‑Language Models**, standing
in for the human breeders, can reproduce that open‑ended dynamic—agents join a
shared archive, look at candidate images, pick and mutate them toward whatever
they "see," and publish their discoveries for others to branch from.

This repo contains the full research codebase: the multi‑agent evolutionary
simulation, the VLM backends, the NEAT/CPPN engine, the analysis & figure
pipeline behind the paper and blog, and the tooling for the interactive
[Picbreeder homage site](#-the-picbreeder-homage-breed-your-own).

## Links

| | |
| --- | --- |
| 📝 **Blog / interactive report** | <https://pub.sakana.ai/picbreeder-vlm> |
| 📄 **Paper (arXiv)** | <https://arxiv.org/abs/2605.23908> |
| 🤗 **Archive dataset** (evolved genomes, images, lineages, VLM captions) | <https://huggingface.co/datasets/picbreeder-vlm/picbreeder-vlm-archive> |
| 🧬 **Breed‑your‑own demo** (CPPN homage in the browser) | <https://pub.sakana.ai/picbreeder-vlm/breed/> |
| 🤗 **Community dataset** (user‑bred genomes) | <https://huggingface.co/datasets/picbreeder-vlm/picbreeder-vlm-community> |

## Install

We use [**uv**](https://docs.astral.sh/uv/):

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .          # exposes the `picbreeder_vlm` package + pickle-compat shims
```

## Quickstart

The collaborative loop needs a vision‑language model to act as the "breeder." You
can drive it **two ways**—a hosted API or a fully local model:

- **Hosted API (Gemini).** Set your key and pick a Gemini model:
  ```bash
  export GEMINI_API_KEY=...        # get one from Google AI Studio
  # then pass e.g. model=gemini-2.5-pro (also: gemini-2.5-flash, gemini-3-pro-preview)
  ```
- **Local model (Qwen, no API key).** Pass a local Qwen model—weights run on your
  own GPU, either in‑process (HuggingFace/vLLM) or against a local server:
  ```bash
  # in-process weights (simplest): model=qwen3-vl-8b   (also 2b / 4b / 32b)
  # or start a shared local vLLM server and point runs at it:
  ./scripts/serve_local_vlm.sh     # serves an OpenAI-compatible endpoint
  # then pass model=remote:Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
  ```

### Run one collaborative session—`train_collaborative.py`

`train_collaborative.py` is the **main entry point** for a single run: a session of
agents that join a shared archive, evolve CPPN images with the VLM in the loop, and
publish their discoveries. Override any config field Hydra‑style (`key=value`):

```bash
# with a hosted API model
GEMINI_API_KEY=... python train_collaborative.py \
    model=gemini-2.5-pro num_agents=5 agent_generations=20 seed=0

# with a local model (no key)
python train_collaborative.py \
    model=qwen3-vl-8b num_agents=5 agent_generations=20 seed=0
```

Outputs land in `logs_collaborative/<experiment>/` (evolved images, genome
`.pkl`s, and JSON archive snapshots). Useful CLI args (full list & defaults in
[`picbreeder_vlm/core/config.py`](picbreeder_vlm/core/config.py)):

| arg | what it does |
| --- | --- |
| `model=` | VLM to use: `gemini-2.5-pro` (API) · `qwen3-vl-8b` (local) · `remote:Qwen/…` (local server) |
| `num_agents=` · `agent_generations=` | how many agents run, and generations per agent |
| `seed=` | RNG seed for a reproducible run |
| `goal=` | breeding objective prompt—one of the keys in [`GOAL_PROMPTS`](picbreeder_vlm/vlm/prompts.py) (`familiar_objects`, `objective_free`, …) |
| `rows=` `cols=` `select_k=` | CPPN grid size shown to the VLM, and max parents picked per step |
| `chat_history_turns=` | prior turns each agent sees (`-1` = keep all) |
| `enable_crossover=` `rand_select_prob=` | Picbreeder‑style crossover; prob. of a random (non‑VLM) pick |
| `scheme=` | render scheme: `color` / `gray` / `toggle` |
| `n_personality_traits=` `generate_personalities=` | give agents random personas |
| `test_mode=true` | quick validation run (caps to 2 agents × 3 generations) |
| `resume=true` | resume an interrupted run in the same `experiment_dir` |

### Sweep hyperparameters—locally or on SLURM

Named sweeps live in
[`picbreeder_vlm/experiments/sweep_configs.py`](picbreeder_vlm/experiments/sweep_configs.py)
(registered in `_NAMED_SWEEPS`); each expands its list‑valued fields into a
Cartesian product of runs. Launch them with the sweep entry point:

```bash
# run every combination locally, one after another
python -m picbreeder_vlm.experiments.sweep sweep_name=chat_history_turns_qwen slurm=false

# or fan the same sweep out as a SLURM array job (via Submitit)
python -m picbreeder_vlm.experiments.sweep sweep_name=chat_history_turns_qwen slurm=true
```

Useful sweep args:

| arg | what it does |
| --- | --- |
| `sweep_name=` | which named sweep to run (see `_NAMED_SWEEPS`) |
| `slurm=false\|true` | run locally in sequence, or submit a SLURM array via Submitit |
| `model=[a,b]` `seed=[1,2,3]` `chat_history_turns=[0,1]` | **list** overrides—swept as a product (note the brackets) |
| `num_agents=` | scalar overrides apply to every run in the sweep |
| `eval_visual_coverage=true` `eval_noun_coverage=true` `eval_tree=true` | run the novelty / alignment / phylogeny evaluations over completed runs |
| `cross_eval=true` | aggregate all seeds into summary plots & tables under `cross_eval/<sweep_name>/` |

See **[AGENTS.md](AGENTS.md)** for the full sweep / evaluation / cross‑eval
workflow and the local vLLM server setup.

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

> **Note—pickle‑compat shims.** The thin modules `neat_components.py`,
> `config.py`, `picbreeder_reproduction.py`, `archive_manager.py` and
> `rendering.py` at the repo root simply re‑export from `picbreeder_vlm/`. They
> exist so that genome `.pkl` files in the archive dataset (which store their
> original module paths) still load. New code should import from the package.

## 🧬 The Picbreeder homage—breed your own

Alongside the VLM experiments we host a small, faithful browser reimplementation
of Picbreeder where **you** can pick and mutate CPPN images, publish your
discoveries, and branch off other people's:

- **Try it:** <https://pub.sakana.ai/picbreeder-vlm/breed/>
- Published genomes are stored in the **community dataset**
  (`picbreeder-vlm/picbreeder-vlm-community`) via a small FastAPI gateway that
  runs as a **Hugging Face Space**—its source is in [`community/`](community/).

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
