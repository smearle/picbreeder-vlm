from pathlib import Path
from typing import Sequence

ARCHIVE_DIR_NAME = "archive"
DEFAULT_AGENT_GENERATIONS = 20
DEFAULT_BASELINE_SELECTION_LIMIT = 1
DEFAULT_CHAT_HISTORY_TURNS = -1  # Unlimited conversational history.
AGENT_DIR_PREFIX = "agent_"
DEFAULT_BRANCHING_ARCHIVE_SAMPLE = 100
PERSONALITY_TOTAL = 100
PERSONALITY_BATCH_SIZE = 10
RATE_EVERY = 5
REPO_ROOT = Path(__file__).resolve().parent
SELECTION_BASELINES: Sequence[str] = ("none", "random", "max-depth", "max-nodes")