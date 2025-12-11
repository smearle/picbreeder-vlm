#!/usr/bin/env python3
"""Render archive image scores stored in archive_metadata.json using the existing VLM rating visualizer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import hydra
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd

from config import PicbreederConfig, ensure_valid_config
from utils import _ensure_absolute
from rate_archive_with_vlm import load_archive_entries, render_ranked_figure, summarize_scores


@dataclass
class ArchiveRatingsConfig(PicbreederConfig):
    output: Optional[Path] = None
    summary_json: Optional[Path] = None
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            help=HelpConf(
                app_name="visualize_archive_ratings",
                header=(
                    "Hydra entry point for visualizing archive VLM ratings.\n"
                    "\n"
                    "Common overrides:\n"
                    "  experiment_dir      Point to the archive root explicitly.\n"
                    "  goal/scheme/seed    Combine with ensure_valid_config to infer the archive directory.\n"
                    "  output/summary_json Customize export paths.\n"
                ),
                footer="Override with +option=value (e.g. output=outputs/vlm_plot.pdf).",
            )
        )
    )


ConfigStore.instance().store(name="archive_ratings_base", node=ArchiveRatingsConfig)


def _resolve_optional_path(value: Optional[Path], base: Path) -> Optional[Path]:
    if value is None:
        return None
    return _ensure_absolute(Path(value), base)


def extract_scores(metadata: Dict) -> Dict[str, List[float]]:
    scores: Dict[str, List[float]] = {}
    for entry in metadata.get("entries", []):
        image_id = entry.get("id")
        values = entry.get("vlm_ratings") or []
        numeric_values = [float(value) for value in values if isinstance(value, (int, float))]
        if image_id and numeric_values:
            scores[image_id] = numeric_values
    return scores


@hydra.main(version_base="1.3", config_path=None, config_name="archive_ratings_base")
def main(cfg: ArchiveRatingsConfig) -> None:
    original_cwd = Path(get_original_cwd())
    validated_cfg = ensure_valid_config(cfg, original_cwd=original_cwd)

    exp_dir = Path(validated_cfg.experiment_dir).resolve()
    archive_dir = exp_dir / "archive"
    metadata_path = archive_dir / "archive_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find {metadata_path}")

    entries, _ = load_archive_entries(archive_dir)

    metadata = json.loads(metadata_path.read_text())
    scores = extract_scores(metadata)
    if not scores:
        print("No VLM ratings found in metadata; nothing to visualize.")
        return

    summary = summarize_scores(entries, scores)
    if not summary:
        print("No ratings correspond to images on disk; nothing to visualize.")
        return

    output_path = _resolve_optional_path(validated_cfg.output, original_cwd)
    if output_path is None:
        output_path = archive_dir / "vlm_ratings_metadata.pdf"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    render_ranked_figure(summary, output_path)
    print(f"Saved ranked figure to {output_path}")

    summary_path = _resolve_optional_path(validated_cfg.summary_json, original_cwd)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Wrote summary statistics to {summary_path}")


if __name__ == "__main__":
    main()
