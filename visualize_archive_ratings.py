#!/usr/bin/env python3
"""Render archive image scores stored in archive_metadata.json using the existing VLM rating visualizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from rate_archive_with_vlm import load_archive_entries, render_ranked_figure, summarize_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metadata_path",
        type=Path,
        help="Path to archive_metadata.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the ranked figure PNG (defaults to metadata directory)",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write the numeric summary as JSON",
    )
    return parser.parse_args()


def extract_scores(metadata: Dict) -> Dict[str, List[float]]:
    scores: Dict[str, List[float]] = {}
    for entry in metadata.get("entries", []):
        image_id = entry.get("id")
        values = entry.get("vlm_ratings") or []
        numeric_values = [float(value) for value in values if isinstance(value, (int, float))]
        if image_id and numeric_values:
            scores[image_id] = numeric_values
    return scores


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata_path.expanduser().resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find {metadata_path}")
    if metadata_path.name != "archive_metadata.json":
        raise ValueError("metadata_path must point to archive_metadata.json")

    archive_dir = metadata_path.parent
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

    output_path = args.output.expanduser().resolve() if args.output else archive_dir / "vlm_ratings_metadata.png"
    render_ranked_figure(summary, output_path)
    print(f"Saved ranked figure to {output_path}")

    if args.summary_json:
        summary_path = args.summary_json.expanduser().resolve()
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Wrote summary statistics to {summary_path}")


if __name__ == "__main__":
    main()
