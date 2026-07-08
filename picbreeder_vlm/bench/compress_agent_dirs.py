#!/usr/bin/env python3
"""Compress existing agent directories to reduce inode count on cluster filesystems.

This script finds uncompressed agent directories in experiment folders and
compresses them to .zip archives, significantly reducing the file count.

Usage:
    # Compress all agent dirs in a specific experiment:
    python compress_agent_dirs.py /path/to/experiment_dir

    # Compress all experiments in a sweep directory:
    python compress_agent_dirs.py /path/to/sweep_logs/sweep_name

    # Dry run to see what would be compressed:
    python compress_agent_dirs.py /path/to/experiment_dir --dry-run

    # Compress with verbose output:
    python compress_agent_dirs.py /path/to/experiment_dir -v
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

AGENT_DIR_PREFIX = "agent_"
AGENT_ARCHIVE_SUFFIX = ".zip"


def find_experiment_dirs(root: Path) -> List[Path]:
    """Find experiment directories containing an 'agents' subdirectory."""
    experiments = []
    
    # Check if root itself is an experiment directory
    if (root / "agents").is_dir():
        experiments.append(root)
    
    # Check immediate subdirectories (for sweep directories)
    for child in root.iterdir():
        if child.is_dir() and (child / "agents").is_dir():
            experiments.append(child)
    
    return sorted(experiments)


def find_uncompressed_agent_dirs(agents_dir: Path) -> List[Path]:
    """Find agent directories that haven't been compressed yet."""
    uncompressed = []
    
    for item in agents_dir.iterdir():
        if not item.is_dir():
            continue
        if not item.name.startswith(AGENT_DIR_PREFIX):
            continue
        # Check if already has a corresponding .zip
        archive_path = item.with_suffix(AGENT_ARCHIVE_SUFFIX)
        if archive_path.exists():
            # Directory exists alongside archive - unusual, skip
            continue
        uncompressed.append(item)
    
    return sorted(uncompressed)


def count_files_in_dir(path: Path) -> int:
    """Count total files in a directory recursively."""
    count = 0
    try:
        for _ in path.rglob("*"):
            if _.is_file():
                count += 1
    except (OSError, PermissionError):
        pass
    return count


def compress_agent_directory(agent_dir: Path, verbose: bool = False) -> Tuple[bool, int]:
    """Compress an agent directory to a .zip archive.
    
    Returns (success, files_removed).
    """
    if not agent_dir.is_dir():
        return False, 0
    
    archive_path = agent_dir.with_suffix(AGENT_ARCHIVE_SUFFIX)
    files_before = count_files_in_dir(agent_dir)
    
    try:
        base_name = str(archive_path.with_suffix(""))
        created_path = shutil.make_archive(base_name, "zip", agent_dir.parent, agent_dir.name)
        shutil.rmtree(agent_dir, ignore_errors=True)
        if verbose:
            print(f"  Compressed: {agent_dir.name} ({files_before} files -> 1 zip)")
        return True, files_before - 1  # -1 because we created 1 zip file
    except Exception as exc:
        print(f"  ERROR compressing {agent_dir}: {exc}", file=sys.stderr)
        # Clean up partial archive
        if archive_path.exists():
            try:
                archive_path.unlink()
            except OSError:
                pass
        return False, 0


def process_experiment(
    exp_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> Tuple[int, int, int]:
    """Process a single experiment directory.
    
    Returns (agents_compressed, agents_failed, files_saved).
    """
    agents_dir = exp_dir / "agents"
    if not agents_dir.is_dir():
        return 0, 0, 0
    
    uncompressed = find_uncompressed_agent_dirs(agents_dir)
    if not uncompressed:
        if verbose:
            print(f"  No uncompressed agents found in {exp_dir.name}")
        return 0, 0, 0
    
    compressed = 0
    failed = 0
    files_saved = 0
    
    for agent_dir in uncompressed:
        if dry_run:
            file_count = count_files_in_dir(agent_dir)
            print(f"  [DRY RUN] Would compress: {agent_dir.name} ({file_count} files)")
            compressed += 1
            files_saved += file_count - 1
        else:
            success, saved = compress_agent_directory(agent_dir, verbose=verbose)
            if success:
                compressed += 1
                files_saved += saved
            else:
                failed += 1
    
    return compressed, failed, files_saved


def main():
    parser = argparse.ArgumentParser(
        description="Compress agent directories to reduce inode count.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to experiment directory or sweep directory containing experiments",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be compressed without actually doing it",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed progress for each agent directory",
    )
    args = parser.parse_args()
    
    root = args.path.resolve()
    if not root.exists():
        print(f"Error: Path does not exist: {root}", file=sys.stderr)
        sys.exit(1)
    
    experiments = find_experiment_dirs(root)
    if not experiments:
        print(f"No experiment directories found in {root}")
        sys.exit(0)
    
    print(f"Found {len(experiments)} experiment(s) to process")
    if args.dry_run:
        print("DRY RUN - no changes will be made\n")
    
    total_compressed = 0
    total_failed = 0
    total_files_saved = 0
    
    for exp_dir in experiments:
        print(f"\nProcessing: {exp_dir.name}")
        compressed, failed, files_saved = process_experiment(
            exp_dir,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        total_compressed += compressed
        total_failed += failed
        total_files_saved += files_saved
        
        if compressed > 0 or failed > 0:
            status = f"  {compressed} compressed"
            if failed > 0:
                status += f", {failed} failed"
            if not args.dry_run:
                status += f" (~{files_saved} files saved)"
            print(status)
    
    print(f"\n{'='*50}")
    print(f"Total: {total_compressed} agent dirs compressed")
    if total_failed > 0:
        print(f"       {total_failed} failed")
    if total_files_saved > 0:
        action = "would save" if args.dry_run else "saved"
        print(f"       ~{total_files_saved:,} files {action}")


if __name__ == "__main__":
    main()
