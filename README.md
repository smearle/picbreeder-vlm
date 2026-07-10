# AI Picbreeder

## Collaborative evolution of neural image archives by large vision language models

<!-- ### In Search of the Ingredients of Open‑Endedness: Replicating Picbreeder with Large Vision‑Language Models -->

<!-- Sam Earle, Kai Arulkumaran, Andrew Dai, Akarsh Kumar, Julian Togelius, Sebastian Risi—**GECCO 2026** (Best Paper nominee) -->

[**📝 Blog**](https://pub.sakana.ai/picbreeder-vlm) ·
[**🔍 Viewer**](https://pub.sakana.ai/picbreeder-vlm/archive.html) ·
[**📄 Paper**](https://arxiv.org/abs/2605.23908) ·
[**🤗 Dataset**](https://huggingface.co/datasets/picbreeder-vlm/picbreeder-vlm-archive)

[![tests](https://github.com/smearle/picbreeder-vlm/actions/workflows/tests.yml/badge.svg)](https://github.com/smearle/picbreeder-vlm/actions/workflows/tests.yml)

![One cycle of the PicbreederVLM loop: agents sample from the shared archive, a VLM breeder mutates its picks across generations of evolution, and a VLM critic rates the results back into the archive.](figures/system_fig/system_overview.gif)

---

**Picbreeder** (Secretan et al., 2008) was a casual online 2D art-making tool that had crowds of humans evolve
images in concert. Over time the users grew an open‑ended tree of diverse and often recognizable artifacts (butterflies, skulls, automobiles), through indirect encodings of these images that cast them as evolvable
[CPPNs](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network).
This project asks whether a swarm of Vision‑Language Models can replace the human breeders and convincingly reproduce this open-ended effect. 

In this codebase, we have VLM agents contribute in parallel to an ever-growing archive of shared CPPN-images.
In each breeding session, the VLM considers a sample of candidate images for branching, then interactively evolves the lineage of the chosen parent generation-by-generation, making selections and adjusting breeding reproduction hyperparameters along the way, and finally selecting an image for publication.
Multiple such sessions occur in parallel, along with intermittent critic agent sessions, in which VLMs rate images in the archive.
Candidate CPPNs from the archive are drawn at the beginning of these sessions according to metadata like mean ratings, recency, or number of children in the phylogeny of published images.

This repo contains the full research codebase: the multi‑agent evolutionary
simulation, the VLM backends, the NEAT/CPPN engine, and the analysis & figure
pipeline behind the paper and blog.

## Links

| | |
| --- | --- |
| 📝 **Blog / interactive report** | <https://pub.sakana.ai/picbreeder-vlm> |
| 🔍 **Archive viewer** (browse every published image, run by run) | <https://pub.sakana.ai/picbreeder-vlm/archive.html> |
| 📄 **Paper (arXiv)** | <https://arxiv.org/abs/2605.23908> |
| 🤗 **Archive dataset** (evolved genomes, images, lineages, VLM captions) | <https://huggingface.co/datasets/picbreeder-vlm/picbreeder-vlm-archive> |

## Install

We use [**uv**](https://docs.astral.sh/uv/):

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .          # exposes the `picbreeder_vlm` package + pickle-compat shims
```

## Quickstart

The collaborative loop needs a vision‑language model to act as the "breeder." You
can drive it **two ways**: a hosted API or a local model:

- **Hosted API (Gemini).** Set your key and pick a Gemini model:
  ```bash
  export GEMINI_API_KEY=...        # get one from Google AI Studio
  # then pass e.g. model=gemini-2.5-pro (also: gemini-2.5-flash, gemini-3-pro-preview)
  ```
- **Local model (Qwen, no API key).** Weights run on your own GPU via vLLM. Name a
  Qwen model and the run manages the model for you:
  ```bash
  python evolve_collaborative.py model=qwen3-vl-8b   # also 2b / 4b / 30b-fp8
  ```
  With several agent workers, the run starts one vLLM server and points all workers
  at it, then shuts it down on exit. With a single worker it just loads the weights
  in‑process.

  For repeated runs against a big model, start the server **once** yourself and let
  runs attach to it, so you pay the model load only once and the server outlives any
  single run:
  ```bash
  ./scripts/serve_local_vlm.sh     # serves an OpenAI-compatible endpoint
  python evolve_collaborative.py model=remote:Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
  ```
  A `remote:` model never starts a server—it connects to one already listening at
  `VLLM_BASE_URL` (default `http://localhost:8000/v1`), waiting for it to come up.
  This approach allows us to efficiently run parallelized multi-experiment sweeps on a SLURM cluster, with a large local model served on a multi-GPU node, with various CPU-only nodes running the experiments.

  On a machine without vLLM installed, single-worker runs fall back to loading the
  weights in-process through HuggingFace `transformers`.

### Run one collaborative session with `evolve_collaborative.py`

`evolve_collaborative.py` is the main entry point for a single run: a swarm of parallel VLM agents that collaborate on a shared archive, branching and interactively evolving CPPN images through selection, and
publishing their discoveries. Override any config field Hydra‑style (`key=value`):

```bash
# with a hosted API model
GEMINI_API_KEY=... python evolve_collaborative.py \
    model=gemini-2.5-pro num_agents=5 agent_generations=20 seed=0

# with a local model (no key)
python evolve_collaborative.py \
    model=qwen3-vl-8b num_agents=5 agent_generations=20 seed=0
```

Outputs are written to `logs_collaborative/<experiment>/` (evolved images, genome
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
| `eval_visual_coverage=true` `eval_semantic_recall=true` `eval_tree=true` | run the novelty / alignment / phylogeny evaluations over completed runs |
| `cross_eval=true` | aggregate all seeds into summary plots & tables under `cross_eval/<sweep_name>/` |

#### Sweeping a big local model: one GPU daemon, many CPU jobs

A sweep of `model=qwen3-vl-8b` gives every array task its own GPU and makes each
one load the weights from scratch. For a model too big or too slow to load per
task, split the work instead: **one** GPU job serves the model for the whole
sweep, and the sweep's array tasks run CPU‑only and talk to it over HTTP. This is
how the 235B runs were done.

The server is a standalone SLURM job, not a member of the sweep. It writes its
own `host:port` to `vllm_daemon.endpoint` as soon as it starts—*before* the
weights finish loading—so the agent jobs can be queued immediately and will wait
politely for it to come up:

```bash
# 1. start the daemon (owns the GPUs; serves until it hits its walltime)
sbatch scripts/daemon_vllm_235b.sbatch

# 2. point a CPU-only sweep at it
export VLLM_BASE_URL=$(cat vllm_daemon.endpoint)
python -m picbreeder_vlm.experiments.sweep \
    sweep_name=qwen_235b_bf16_daemon slurm=true partition=cpu_short
```

The sweep config ([`Qwen235BBf16DaemonSweep`](picbreeder_vlm/experiments/sweep_configs.py))
carries `gpu=False` and a `remote:` model, so `is_local_model` is False and no array
task tries to start a server of its own. A task still waiting on the daemon keeps
re‑reading `vllm_daemon.endpoint`, so it follows a daemon that restarts onto a new
node; a task whose agents have already started only picks up a moved endpoint when
SLURM requeues it. Besides skipping the repeated model load, this dodges the
GPU‑utilisation killer: the agent jobs hold no GPU, and the daemon's utilisation
stays high because every config feeds it at once.

`daemon_vllm_235b.sbatch` hard‑codes an account, partition, and `/scratch` paths—copy
and edit it for your own cluster.

See **[AGENTS.md](AGENTS.md)** for the full sweep / evaluation / cross‑eval
workflow and the local vLLM server setup.

## Repository map

Source lives in the **`picbreeder_vlm/`** package:

| subpackage | contents |
| --- | --- |
| `picbreeder_vlm.core` | config, CPPN/NEAT genome (`neat_components`), reproduction, rendering, archive management |
| `picbreeder_vlm.vlm` | VLM backends (Gemini / Qwen / …), prompts, chat/conversation history |
| `picbreeder_vlm.agents` | the collaborative multi‑agent loop, single‑agent runner |
| `picbreeder_vlm.experiments` | sweep orchestration (Hydra + Submitit), run configs, experiment CLIs |
| `picbreeder_vlm.analysis` | coverage / embedding / captioning / phylogeny / rating metrics |
| `picbreeder_vlm.viz` | figure and lineage/archive visualization |
| `picbreeder_vlm.niches` | CLIP noun‑niche evolution‑strategy experiments |
| `picbreeder_vlm.nouns` · `picbreeder_vlm.bench` | vocabulary wrangling · VLM benchmarking/probing |

Other top‑level directories: **`tools/`** (blog & figure asset builders),
**`archive_animations/`** (lineage/teaser animations), **`figures/`** (TikZ sources for
the blog figures), **`data/`** (committed data),
**`reference/`** (third‑party material, nothing imported by the experiments).

Two pieces of that lineage are worth naming. Our CPPN rasterizer and NEAT preset
(`picbreeder_vlm/core/picture2d.py`, `picbreeder_vlm/core/interactive_config_color`)
began as a fork of the `examples/picture2d` demo in
[neat‑python](https://github.com/CodeReclaimers/neat-python) (BSD‑3‑Clause) and have
since diverged substantially — four CPPN inputs, a fully connected initial topology,
and the `PicbreederGenome` / `PicbreederReproduction` operators. Those operators were
in turn calibrated against **`reference/webneat/`**, Nick Beato's original Picbreeder
Java client (obtained via Sebastian Risi), which is the authority for how the 2008
system actually behaved; the mutation weight range and mutation‑strength floor in
`core/neat_components.py` were matched to it directly.

## Data

The [**archive dataset**](https://huggingface.co/datasets/picbreeder-vlm/picbreeder-vlm-archive)
holds the evolved archives across runs: rendered images, CPPN genomes, full
lineages, per‑image VLM captions/ratings, and the sprite sheets + orderings that
power the interactive gallery in the blog.

## Tests

```bash
uv pip install -r requirements-test.txt   # no torch / vLLM needed
pytest
```

Mostly, these tests exist to ensure that through refactoring, reorganization, and potential extension, the experiments from our paper (and other supplementary, contemporaneous experiments) remain reproducible. There are probably more tests worth adding here, but this is a start.

So far, the tests cover the core CPPN engine and some of the experiment directory and multi-agent plumbing: the NEAT
preset loads and renders deterministically, run‑directory names still match the
published runs, worker subprocesses round‑trip their config, and the top‑level
pickle‑compat shims keep archived genomes loadable. CI runs them on every push and
pull request against Python 3.11 and 3.13.

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
and collaborators. `reference/webneat/` is the original WebNEAT code by Nick Beato.
`fer/` is vendored from [akarshkumar0101/fer](https://github.com/akarshkumar0101/fer)
(Apache‑2.0), the code for *The Fractured Entangled Representation Hypothesis*; we use
its Picbreeder genome parsing and have added our own lineage/phylogeny scripts alongside
it. Our CPPN rendering derives from
[neat‑python](https://github.com/CodeReclaimers/neat-python) (BSD‑3‑Clause).
