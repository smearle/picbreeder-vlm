The venv you need is here: `.venv/bin/python`

# Picbreeder-VLM Codebase Documentation

This document serves as a guide for agents and developers working on the Picbreeder-VLM project. It outlines the project architecture, key files, and operational workflows.

## 🗺️ Project Roadmap

The codebase is organized into several distinct layers, from high-level orchestration to low-level evolutionary mechanics.

### 1. Orchestration & Configuration
*   **`sweep.py`**: The main entry point. Orchestrates experiments (sweeps) locally or on Slurm. It generates individual run configurations and launches them.
*   **`config.py`**: Defines the `PicbreederConfig` dataclass. This is the single source of truth for experiment settings (grid size, model choice, evolution parameters).
*   **`sweep_configs.py`**: Defines the search spaces for hyperparameters. Contains `SweepConfig` and named sweep classes (e.g., `ChatHistoryTurnsSweep`).

### 2. Core Agent Logic
*   **`collaborative_multi_agent.py`**: The heart of the simulation. It runs the main loop where agents join, evolve images, and publish to the shared archive.
*   **`agent_runner.py`**: Manages the lifecycle of a single agent process, handling its interaction with the NEAT population and the VLM.
*   **`interactive_evolution.py`**: A standalone script for *human* interactive evolution (legacy Picbreeder style). Useful for debugging rendering or NEAT mechanics without VLM overhead.

### 3. VLM & AI Layer
*   **`vlm_backends.py`**: The unified interface for AI models. Abstracts away differences between Gemini, Qwen, and other models. Handles image/text queries and chat history.
*   **`chat.py`**: Manages conversation history and context for agents.
*   **`prompts.py`**: Stores system prompts and goal definitions used to guide the VLMs.

### 4. Evolutionary Engine (NEAT)
*   **`picbreeder_reproduction.py`**: Custom NEAT reproduction logic. Implements the specific crossover and mutation rules that emulate the original Picbreeder web app.
*   **`neat_components.py`**: Core NEAT utilities, including the `PicbreederGenome` class and stagnation logic.

### 5. Archive & State Management
*   **`archive_manager.py`**: Manages the persistent shared archive. Handles file locking, saving images/genomes, and tracking lineage/metadata (`ArchiveEntry`).

### 6. Analysis & Evaluation
*   **`embed_and_visualize.py`**: Generates embeddings (CLIP/SigLIP) for archives and creates visualizations (t-SNE/UMAP).
*   **`compute_noun_similarity.py`**: Measures how well evolved images match target nouns (alignment metrics).
*   **`plot_novelty_over_time.py`**: Visualizes the novelty of the archive over time.
*   **`tree_metrics.py`**: Calculates phylogenetic statistics (tree depth, branching factors).

---

# Sweep System Documentation

The `sweep.py` script is the primary entry point for launching collaborative multi-agent experiments. It is built on [Hydra](https://hydra.cc/) and [Submitit](https://github.com/facebookincubator/submitit).

## Core Concepts

*   **Sweep Configs**: Defined in `sweep_configs.py`. These classes (e.g., `ChatHistoryTurnsSweep`) define the search space. Fields with `List` types (e.g., `seed: List[int]`) are expanded into a Cartesian product of jobs.
*   **Run Config**: Defined in `config.py` (`PicbreederConfig`). This is the configuration for a single experiment run.
*   **Execution**: `sweep.py` generates individual `PicbreederConfig` objects from the selected `SweepConfig` and launches them either sequentially (local) or as array jobs (Slurm).

## Common Commands

### 1. Launch a Sweep Locally
Run the `chat_history_turns` sweep locally (no Slurm):
```bash
.venv/bin/python sweep.py sweep_name=chat_history_turns slurm=false
```

### 2. Launch on Slurm
Submit the same sweep to the cluster:
```bash
.venv/bin/python sweep.py sweep_name=chat_history_turns slurm=true
```

### 3. Run Evaluations
Evaluations are often run as a separate step after training. You can enable specific evaluation pipelines using flags. Note that you generally want to use the same `sweep_name` so it finds the correct experiment directories.

**Visual Coverage (Novelty):**
```bash
.venv/bin/python sweep.py sweep_name=chat_history_turns eval_visual_coverage=true slurm=false
```

**Noun Coverage (Alignment):**
```bash
.venv/bin/python sweep.py sweep_name=chat_history_turns eval_noun_coverage=true slurm=false
```

**Phylogeny (Tree) Metrics:**
```bash
.venv/bin/python sweep.py sweep_name=chat_history_turns eval_tree=true slurm=false
```

### 4. Cross-Evaluation (Aggregation)
To aggregate results across all seeds and generate summary plots/tables:
```bash
.venv/bin/python sweep.py sweep_name=chat_history_turns cross_eval=true
```
This will look for completed runs in `logs_collaborative/sweep/<sweep_name>/` and output results to `cross_eval/<sweep_name>`.

### CLI Overrides
You can override any parameter from the command line. CLI overrides take precedence over the named sweep defaults.

```bash
.venv/bin/python sweep.py sweep_name=chat_history_turns num_agents=50
```
This will run the `chat_history_turns` sweep but force `num_agents` to 50 for all runs.

## Adding a New Sweep

1.  Open `sweep_configs.py`.
2.  Define a new dataclass inheriting from `SweepConfig`.
3.  Override fields with lists of values to sweep over.
4.  Register it in `_NAMED_SWEEPS`.

Example:
```python
@dataclass
class MyNewSweep(SweepConfig):
    # Sweep over 3 seeds
    seed: List[int] = field(default_factory=lambda: [1, 2, 3])
    # Sweep over 2 models
    model: List[str] = field(default_factory=lambda: ["gemini-2.5-pro", "gemini-2.5-flash"])
```

## Directory Structure

*   `logs_collaborative/`: Base directory for all experiment outputs.
    *   `sweep/<sweep_name>/<run_name>/`: Individual run outputs.
        *   `archive/`: Saved images (PNG) and genomes (pickle).
        *   `archive_history/`: JSON snapshots of the archive state over time.
        *   `experiment_stats.json`: Run statistics.
        *   `generations/`: (Optional) Per-generation visualizations if enabled.
*   `cross_eval/<sweep_name>/`: Aggregated results (plots, tables, LaTeX).
*   `noun_lists/`: Text files containing lists of target nouns for alignment tasks.
*   `initial_populations/`: Seed populations for runs.
*   `picture2d/`: Legacy Picbreeder rendering and NEAT configuration files.