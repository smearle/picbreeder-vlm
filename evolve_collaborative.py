#!/usr/bin/env python3
"""Run one collaborative multi-agent Picbreeder-VLM session.

This is the main entry point for one run (a session of agents that join a shared
archive, evolve CPPN images with a VLM in the loop, and publish discoveries).
It is a thin wrapper around the Hydra app in
``picbreeder_vlm.agents.collaborative_multi_agent`` so it can be launched as a
plain script:

    python evolve_collaborative.py model=qwen3-vl-8b num_agents=5 agent_generations=20

Any PicbreederConfig field (see picbreeder_vlm/core/config.py) can be overridden
on the command line, Hydra-style ``key=value``. To sweep over hyperparameters
locally or on SLURM, use ``python -m picbreeder_vlm.experiments.sweep`` instead.
"""
from picbreeder_vlm.agents.collaborative_multi_agent import main

if __name__ == "__main__":
    main()
