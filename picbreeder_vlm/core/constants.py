from pathlib import Path
from typing import Sequence

ARCHIVE_DIR_NAME = "archive"
DEFAULT_BASELINE_SELECTION_LIMIT = 1
AGENT_DIR_PREFIX = "agent_"
DEFAULT_BRANCHING_ARCHIVE_SAMPLE = 100
PERSONALITY_TOTAL = 100
PERSONALITY_BATCH_SIZE = 10
RATE_EVERY = 5
RATING_BATCH_SIZE = 100
REPO_ROOT = Path(__file__).resolve().parents[2]
# Get the repo name from the REPO_ROOT path
REPO_NAME = REPO_ROOT.name
SELECTION_BASELINES: Sequence[str] = ("none", "random", "max-depth", "max-nodes", "clip-nouns")
# Committed data lives under data/. These are repo-relative and get anchored to a
# base (the launch cwd or REPO_ROOT) at each call site.
DATA_DIR = Path("data")
NOUN_LISTS_DIR = DATA_DIR / "noun_lists"
HUMAN_BASELINE_DIR = DATA_DIR / "human_baseline"
MAX_MODEL_LEN = 50_000
