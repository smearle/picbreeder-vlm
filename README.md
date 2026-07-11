# AI Picbreeder

## Collaborative evolution of neural image archives by large vision language models

This is the codebase for the paper [_In Search of the Ingredients of Open‑Endedness: Replicating Picbreeder with Large Vision‑Language Models_](https://arxiv.org/abs/2605.23908), by Sam Earle, Kai Arulkumaran, Andrew Dai, Akarsh Kumar, Julian Togelius, Sebastian Risi, at GECCO 2026.

[**📝 Blog**](https://pub.sakana.ai/picbreeder-vlm) ·
[**🔍 Viewer**](https://pub.sakana.ai/picbreeder-vlm/archive.html) ·
[**📄 Paper**](https://arxiv.org/abs/2605.23908) ·
[**🤗 Dataset**](https://huggingface.co/datasets/picbreeder-vlm/picbreeder-vlm-archive)

[![tests](https://github.com/smearle/picbreeder-vlm/actions/workflows/tests.yml/badge.svg)](https://github.com/smearle/picbreeder-vlm/actions/workflows/tests.yml)

![One cycle of the PicbreederVLM loop: agents sample from the shared archive, a VLM breeder mutates its picks across generations of evolution, and a VLM critic rates the results back into the archive.](figures/system_fig/system_overview.gif)

---

**Picbreeder** [[1](https://www.campbellssite.com/papers/secretan_chi08.pdf),[2](https://stars.library.ucf.edu/cgi/viewcontent.cgi?article=2880&context=facultybib2010)] was a casual online 2D art-making tool that had crowds of humans evolve
images in concert. Over time the users grew an open‑ended tree of diverse and often recognizable artifacts (butterflies, skulls, automobiles), through indirect encodings of these images that cast them as evolvable
[CPPNs](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network).
This project asks whether a swarm of Vision‑Language Models can replace the human breeders and convincingly reproduce this open-ended effect. 

In this codebase, we have VLM agents contribute in parallel to an ever-growing archive of shared CPPN-images.
In each breeding session, the VLM considers a sample of candidate images for branching, then interactively evolves the lineage of the chosen parent generation-by-generation, making selections and adjusting breeding reproduction hyperparameters along the way, and finally selecting an image for publication.
Multiple such sessions occur in parallel, along with intermittent critic agent sessions, in which VLMs rate images in the archive.
Candidate CPPNs from the archive are drawn at the beginning of these sessions according to metadata like mean ratings, recency, or number of children in the phylogeny of published images.

This repo contains the multi‑agent Picbreeder evolutionary
simulation loop, including VLM backends, a Python implementation of the original NEAT/CPPN engine, and the archive evaluation routines used to generate the paper's quantitative results.
It also contains code for generating figures for the paper, and the code for the blog, including an interactive interface for viewing archives, and a partial reconstruction of the original Picbreeder interface.

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

`requirements.txt` covers the evolution loop, the hosted‑API (Gemini) path, and all
evaluation/figure tooling. Two extras are opt‑in:

- **Local vLLM models** (the `qwen3-vl-*` / `remote:` paths): also
  `uv pip install -r requirements-local.txt`. It is kept separate on purpose—an
  unconstrained `vllm` can resolve to an ancient source‑only build that fails to
  compile on modern GPUs.
- **GPU torch.** The default PyPI `torch`/`torchvision` wheels target one CUDA build.
  On a recent NVIDIA GPU (e.g. Blackwell / RTX 50‑series) install a matching build
  first, e.g. `uv pip install "torch>=2.4" "torchvision>=0.19" --index-url
  https://download.pytorch.org/whl/cu128`. CPU‑only is fine for evaluations and
  figures that reuse the dataset's cached embeddings
  (`--index-url https://download.pytorch.org/whl/cpu`).

## Quickstart

The AI Picbreeder loop needs a vision‑language model to act as the "breeder," you can provide an API key or spin up a local model (if your compute is sufficient):

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
- **No model (mock).** `model=mock` is a zero‑cost, dependency‑free stand‑in that
  returns well‑formed random selections. It needs no API key, GPU, or download and
  exercises the full evolution / archive / continue‑a‑run plumbing—handy for smoke
  tests, CI, and trying the pipeline before committing real compute:
  ```bash
  python evolve_collaborative.py model=mock num_agents=2 agent_generations=3
  ```

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
| `model=` | VLM to use: `gemini-2.5-pro` (API) · `qwen3-vl-8b` (local) · `remote:Qwen/…` (local server) · `mock` (no‑cost stand‑in) |
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

`daemon_vllm_235b.sbatch` defaults to the torch cluster, but you can retarget it
without editing the file: override the SLURM resources through the standard
`SBATCH_ACCOUNT` / `SBATCH_PARTITION` / `SBATCH_QOS` env vars (or `sbatch` flags),
and the repo path, model, port, and vLLM knobs through the `PB_DAEMON_*` env vars
documented in the script's header. Submit from the repo root so its relative log
paths resolve.

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

Top‑level layout:

```
picbreeder-vlm/
├─ picbreeder_vlm/          # the Python package (subpackages in the table above)
├─ tools/                   # blog & figure asset builders, HF sync
├─ archive_animations/      # lineage / teaser animations
├─ third-party/             # vendored external code — nothing here runs in the experiments
│  ├─ webneat/              #   original Picbreeder Java client (Beato); NEAT calibration source
│  └─ fer/                  #   akarshkumar0101/fer human archive; resolve via _paths.FER_ROOT
├─ data/                    # committed data (noun lists, human‑baseline metric JSONs)
├─ figures/                 # TikZ sources for the paper/blog figures
└─ blog/                    # interactive blog, archive viewer, and breeding site
```

Run outputs (`logs_collaborative/`, `sweep_logs/sweep/<exp_name>/`, `cross_eval/`) are
generated and gitignored — see [Data](#data) below.

Our CPPN rasterizer and NEAT preset
(`picbreeder_vlm/core/picture2d.py`, `picbreeder_vlm/core/interactive_config_color`)
began as a fork of the `examples/picture2d` demo in
[neat‑python](https://github.com/CodeReclaimers/neat-python) and have
since diverged substantially. We now use four CPPN inputs, a fully connected initial topology,
and the `PicbreederGenome` / `PicbreederReproduction` operators. Those operators were
in turn calibrated against **`third-party/webneat/`**, a copy of the original Picbreeder
Java client and the source of truth in terms of how the 2008
system actually behaved. The mutation weight range and mutation‑strength floor in
`core/neat_components.py` were also matched to it.

We validated the historical accuracy of our Picbreeder CPPN implementation by ensuring genomes from the original human experiment rendered identically across our implementation in `picbreeder_vlm`, the re-implemented renderer in `third-party/fer` and the original at `third-party/web-neat`.
To ensure equivalence of the interactive NEAT algorithm, we verified qualitatively the similarity between breeding from random initial networks in our version and the Java client.
We additionally computed the likelihood that selected children from the original human lineages *would have* been generated as offspring given their parents' genomes and our inferences about the mutation/crossovers and hyperparameters, and visually compared the historical child to offspring of the parent generated by our pipeline.

## Data

The [**archive dataset**](https://huggingface.co/datasets/picbreeder-vlm/picbreeder-vlm-archive)
holds the evolved archives across runs, under `results/<run>/`: packed CPPN genomes
(`genomes.tar.gz`), agent logs (`agents.tar`), full lineages, per‑image VLM
captions/ratings, cached image embeddings, and curated metric JSONs.
The dataset also includes the sprite
sheets & orderings that power the blog's interactive galleries. We include images for convenience, but generally they do not need to be
stored as they re‑render deterministically from the genomes.

### Reconstruct a run for evaluation, figures, or continued evolution

The dataset stores each run in a flattened, packed form (and with the absolute
paths of the cluster it was produced on). To turn one back into a runnable
experiment directory—unpack the genomes, re‑render the images, fix up the metadata
paths—use the helper:

```bash
# reconstruct one or more runs into sweep_logs/sweep/<run>/ (where the tools look)
PYTHONPATH=. python tools/pull_run_from_hf.py <run_name> [<run_name> ...]
PYTHONPATH=. python tools/pull_run_from_hf.py --all          # every run
```

You can then run the evaluations/figures over the reconstructed runs exactly as for
a local sweep (see [AGENTS.md](AGENTS.md)), e.g.:

```bash
python -m picbreeder_vlm.experiments.sweep \
    sweep_name=chat_history_turns eval_visual_coverage=true slurm=false
```

**Continue evolving a published archive.** Point a run at a reconstructed
directory; new agents load the existing archive, branch from it, and append new
publications. `mock` lets you try it with zero compute:

```bash
PYTHONPATH=. python tools/pull_run_from_hf.py <run_name>
python evolve_collaborative.py model=mock resume=true \
    experiment_dir=sweep_logs/sweep/<run_name> num_agents=<original+N>
```

Two caveats the tool handles for you: many runs shipped only their first ~1000
genomes (a historical sync gap), and per‑image `image_path`/`genome_path` in the
metadata are stale absolute paths. `pull_run_from_hf.py` rewrites the paths and, by
default, prunes archive entries whose genome wasn't shipped so the reconstructed
archive is self‑consistent. Evaluations reuse the shipped embedding caches, so they
run on CPU; the phylogeny (`eval_tree`) figure additionally needs the system
Graphviz `dot` binary (`apt install graphviz`).

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
and collaborators. `third-party/webneat/` is the original WebNEAT code by Nick Beato.
`third-party/fer/` is vendored from [akarshkumar0101/fer](https://github.com/akarshkumar0101/fer)
(Apache‑2.0), the code for *The Fractured Entangled Representation Hypothesis*; we use
its Picbreeder genome parsing and have added our own lineage/phylogeny scripts alongside
it. Our CPPN rendering derives from
[neat‑python](https://github.com/CodeReclaimers/neat-python) (BSD‑3‑Clause).
